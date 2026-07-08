"""
Two-stage HOI detector with enhanced visual context

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch import nn, Tensor
from collections import OrderedDict
from typing import Optional, Tuple, List
from torchvision.ops import FeaturePyramidNetwork

from transformers import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    SwinTransformer,
)

from ops import (
    binary_focal_loss_with_logits,
    compute_spatial_encodings,
    prepare_region_proposals,
    associate_with_ground_truth,
    compute_prior_scores,
    compute_sinusoidal_pe
)

from detr.models import build_model as build_base_detr
from h_detr.models import build_model as build_advanced_detr
from detr.models.position_encoding import PositionEmbeddingSine
from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list

class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z

class HumanObjectMatcher(nn.Module):
    def __init__(self, repr_size, num_verbs, obj_to_verb, dropout=.1, human_idx=0):
        super().__init__()
        self.repr_size = repr_size
        self.num_verbs = num_verbs
        self.human_idx = human_idx
        self.obj_to_verb = obj_to_verb

        self.ref_anchor_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, repr_size), nn.ReLU(),
        )
        self.encoder = TransformerEncoder(num_layers=2, dropout=dropout)
        self.mmf = MultiModalFusion(512, repr_size, repr_size)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h]==self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)

        # Modulate the positional embeddings with box widths and heights by
        # applying different temperatures to x and y
        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()    # n_query, 2
        # Note that the positional embeddings are stacked as [pe(y), pe(x)]
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        return box_pe, c_pe

    def forward(self, region_props, image_sizes, device=None):
        if device is None:
            device = region_props[0]["hidden_states"].device

        ho_queries = []
        paired_indices = []
        prior_scores = []
        object_types = []
        positional_embeds = []
        for i, rp in enumerate(region_props):
            boxes, scores, labels, embeds = rp.values()
            nh = self.check_human_instances(labels)
            n = len(boxes)
            # Enumerate instance pairs
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < nh)).unbind(1)
            # Skip image when there are no valid human-object pairs
            if len(x_keep) == 0:
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                positional_embeds.append({})
                continue
            x = x.flatten(); y = y.flatten()
            # Compute spatial features
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x],], [boxes[y],], [image_sizes[i],]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)

            box_pe, c_pe = self.compute_box_pe(boxes, embeds, image_sizes[i])
            embeds, _ = self.encoder(embeds.unsqueeze(1), box_pe.unsqueeze(1))
            embeds = embeds.squeeze(1)
            # Compute human-object queries
            ho_q = self.mmf(
                torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),
                pairwise_spatial_reshaped[x_keep, y_keep]
            )
            # Append matched human-object pairs
            ho_queries.append(ho_q)
            paired_indices.append(torch.stack([x_keep, y_keep], dim=1))
            prior_scores.append(compute_prior_scores(
                x_keep, y_keep, scores, labels, self.num_verbs, self.training,
                self.obj_to_verb
            ))
            object_types.append(labels[y_keep])
            positional_embeds.append({
                "centre": torch.cat([c_pe[x_keep], c_pe[y_keep]], dim=-1).unsqueeze(1),
                "box": torch.cat([box_pe[x_keep], box_pe[y_keep]], dim=-1).unsqueeze(1)
            })

        return ho_queries, paired_indices, prior_scores, object_types, positional_embeds

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)

class FeatureHead(nn.Module):
    def __init__(self, dim, dim_backbone, return_layer, num_layers):
        super().__init__()
        self.dim = dim
        self.dim_backbone = dim_backbone
        self.return_layer = return_layer

        in_channel_list = [
            int(dim_backbone * 2 ** i)
            for i in range(return_layer + 1, 1)
        ]
        self.fpn = FeaturePyramidNetwork(in_channel_list, dim)
        self.layers = nn.Sequential(
            Permute([0, 2, 3, 1]),
            SwinTransformer(dim, num_layers)
        )
    def forward(self, x):
        pyramid = OrderedDict(
            (f"{i}", x[i].tensors)
            for i in range(self.return_layer, 0)
        )
        mask = x[self.return_layer].mask
        x = self.fpn(pyramid)[f"{self.return_layer}"]
        x = self.layers(x)
        return x, mask

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class PViC(nn.Module):
    """Two-stage HOI detector with enhanced visual context"""

    def __init__(self,
        detector: Tuple[nn.Module, str], postprocessor: nn.Module,
        feature_head: nn.Module, ho_matcher: nn.Module,
        triplet_decoder: nn.Module, num_verbs: int,
        repr_size: int = 384, human_idx: int = 0,
        # Focal loss hyper-parameters
        alpha: float = 0.5, gamma: float = .1,
        # Sampling hyper-parameters
        box_score_thresh: float = .05,
        min_instances: int = 3,
        max_instances: int = 15,
        raw_lambda: float = 2.8,
    ) -> None:
        super().__init__()

        self.detector = detector[0]
        self.od_forward = {
            "base": self.base_forward,
            "advanced": self.advanced_forward,
        }[detector[1]]
        self.postprocessor = postprocessor

        self.ho_matcher = ho_matcher
        self.feature_head = feature_head
        self.kv_pe = PositionEmbeddingSine(128, 20, normalize=True)
        self.decoder = triplet_decoder
        self.binary_classifier = nn.Linear(repr_size, num_verbs)

        self.repr_size = repr_size
        self.human_idx = human_idx
        self.num_verbs = num_verbs
        self.alpha = alpha
        self.gamma = gamma
        self.box_score_thresh = box_score_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.raw_lambda = raw_lambda

    def freeze_detector(self):
        for p in self.detector.parameters():
            p.requires_grad = False
        self.detector.eval()

    @staticmethod
    def _prepare_rgip_attn_context(ctx, n_queries):
        """Return a decoder-safe RGIP attention context.

        The teacher builds replay contexts in its own pair order, while the
        student decoder can occasionally see fewer/differently ordered HO
        queries.  If all modulated pairs fall outside the student's available
        query range, attention modulation silently becomes a no-op.  For the
        attention branch only, pack the highest-SIS valid confounder entries
        into the available query slots; the original context is still used for
        loss weighting/distillation.
        """
        if ctx is None or n_queries is None or n_queries <= 0:
            return ctx
        modulate_mask = ctx.get('modulate_mask', None)
        if modulate_mask is None or modulate_mask.numel() == 0:
            return ctx

        max_attn_pairs = int(ctx.get('max_attn_pairs', 16) or 16)
        max_attn_pairs = max(1, max_attn_pairs)

        n_ctx = min(int(n_queries), int(modulate_mask.shape[0]))
        if n_ctx > 0 and bool(modulate_mask[:n_ctx].any().item()):
            valid_in_range = torch.nonzero(modulate_mask[:n_ctx], as_tuple=False).flatten()
            if valid_in_range.numel() > max_attn_pairs:
                sis = ctx.get('sis', None)
                if sis is not None and sis.numel() > 0:
                    valid_sis = sis[valid_in_range].to(dtype=torch.float32)
                    order = torch.argsort(valid_sis, descending=True)
                    valid_in_range = valid_in_range[order]
                selected = valid_in_range[:max_attn_pairs]
                limited = dict(ctx)
                limited_mask = modulate_mask.clone()
                limited_mask[:n_ctx] = False
                limited_mask[selected] = True
                limited['modulate_mask'] = limited_mask
                limited['max_attn_pairs'] = max_attn_pairs
                limited['_rgip_attn_alignment_fallback'] = torch.zeros(
                    (), device=modulate_mask.device, dtype=torch.long
                )
                return limited
            ctx['_rgip_attn_alignment_fallback'] = torch.zeros(
                (), device=modulate_mask.device, dtype=torch.long
            )
            return ctx

        valid = torch.nonzero(modulate_mask, as_tuple=False).flatten()
        if valid.numel() == 0:
            return ctx

        sis = ctx.get('sis', None)
        if sis is not None and sis.numel() > 0:
            valid_sis = sis[valid].to(dtype=torch.float32)
            order = torch.argsort(valid_sis, descending=True)
            valid = valid[order]

        k = min(int(n_queries), int(valid.numel()), max_attn_pairs)
        selected = valid[:k]
        aligned = {
            'lambda_attn': ctx.get('lambda_attn', 0.0),
            'kappa': ctx.get('kappa', 1.0),
            'attn_clamp': ctx.get('attn_clamp', 5.0),
            'max_attn_pairs': max_attn_pairs,
            '_rgip_attn_alignment_fallback': torch.ones(
                (), device=modulate_mask.device, dtype=torch.long
            ),
        }
        for key in ('sis', 'confounder_weights', 'confounder_prototypes'):
            src = ctx.get(key, None)
            if src is None:
                continue
            shape = list(src.shape)
            shape[0] = k
            dst = torch.zeros(shape, device=src.device, dtype=src.dtype)
            dst[:k] = src[selected]
            aligned[key] = dst

        aligned_mask = torch.zeros(k, device=modulate_mask.device, dtype=torch.bool)
        aligned_mask[:k] = True
        aligned['modulate_mask'] = aligned_mask
        return aligned

    def compute_classification_loss(self, logits, prior, labels, pair_weights=None):
        prior = torch.cat(prior, dim=0).prod(1)
        x, y = torch.nonzero(prior).unbind(1)

        logits = logits[:, x, y]
        prior = prior[x, y]
        labels = labels[None, x, y].repeat(len(logits), 1)
        valid_weights = None
        if pair_weights is not None and len(x) > 0:
            valid_weights = pair_weights.to(device=logits.device, dtype=logits.dtype)[x]

        if valid_weights is not None:
            n_p = (labels * valid_weights[None, :]).sum()
        else:
            n_p = labels.sum()
        if self.training and dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = n_p.detach().reshape(1).to(logits.device)
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).clamp(min=1.0).item()
        else:
            n_p = n_p.clamp(min=1.0)

        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='none',
            alpha=self.alpha, gamma=self.gamma
        )
        if valid_weights is not None:
            loss = loss * valid_weights[None, :]

        return loss.sum() / n_p

    def postprocessing(self,
            boxes, paired_inds, object_types,
            logits, prior, image_sizes
        ):
        n = [len(p_inds) for p_inds in paired_inds]
        logits = logits.split(n)

        detections = []
        for bx, p_inds, objs, lg, pr, size in zip(
            boxes, paired_inds, object_types,
            logits, prior, image_sizes
        ):
            pr = pr.prod(1)
            x, y = torch.nonzero(pr).unbind(1)
            scores = lg[x, y].sigmoid() * pr[x, y].pow(self.raw_lambda)
            detections.append(dict(
                boxes=bx, pairing=p_inds[x], scores=scores,
                labels=y, objects=objs[x], size=size, x=x
            ))

        return detections

    @staticmethod
    def base_forward(ctx, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = ctx.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = ctx.transformer(ctx.input_proj(src), mask, ctx.query_embed.weight, pos[-1])[0]

        outputs_class = ctx.class_embed(hs)
        outputs_coord = ctx.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out, hs, features

    @staticmethod
    def advanced_forward(ctx, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = ctx.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(ctx.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if ctx.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, ctx.num_feature_levels):
                if l == _len_srcs:
                    src = ctx.input_proj[l](features[-1].tensors)
                else:
                    src = ctx.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = ctx.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not ctx.two_stage or ctx.mixed_selection:
            query_embeds = ctx.query_embed.weight[0 : ctx.num_queries, :]

        self_attn_mask = (
            torch.zeros([ctx.num_queries, ctx.num_queries,]).bool().to(src.device)
        )
        self_attn_mask[ctx.num_queries_one2one :, 0 : ctx.num_queries_one2one,] = True
        self_attn_mask[0 : ctx.num_queries_one2one, ctx.num_queries_one2one :,] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = ctx.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = ctx.class_embed[lvl](hs[lvl])
            tmp = ctx.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(outputs_class[:, 0 : ctx.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, ctx.num_queries_one2one :])
            outputs_coords_one2one.append(outputs_coord[:, 0 : ctx.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, ctx.num_queries_one2one :])
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }

        if ctx.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out, hs, features

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None,
        return_cross_attn: bool = False,  # 新增参数
        return_outputs=False,
        teacher_cross_attn_hint=None,
        attn_hint_alpha=0.05,
        use_vas=False, is_replay_list=None, sis_scores=None, vas_lambda=0.0,
        batch_indices=None,
        rgip_context=None,
        return_rgip_meta=False) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (M, 2) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `size`: torch.Tensor
                (2,) Image height and width
            `x`: torch.Tensor
                (M,) Index tensor corresponding to the duplications of human-objet pairs. Each
                pair was duplicated once for each valid action.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        image_sizes = torch.as_tensor([im.size()[-2:] for im in images], device=images[0].device)

        with torch.no_grad():
            # The detector is frozen in PViC training.  Keep it in eval mode even
            # when the surrounding DDP module calls train(), otherwise dropout in
            # the detector can make teacher/student HO-pair ordering diverge and
            # disable RGIP attention modulation.
            self.detector.eval()
            results, hs, features = self.od_forward(self.detector, images)
            results = self.postprocessor(results, image_sizes)

        region_props = prepare_region_proposals(
            results, hs[-1], image_sizes,
            box_score_thresh=self.box_score_thresh,
            human_idx=self.human_idx,
            min_instances=self.min_instances,
            max_instances=self.max_instances
        )
        boxes = [r['boxes'] for r in region_props]
        # Produce human-object pairs.
        (
            ho_queries,
            paired_inds, prior_scores,
            object_types, positional_embeds
        ) = self.ho_matcher(region_props, image_sizes)
        # Compute keys/values for triplet decoder.
        memory, mask = self.feature_head(features)
        b, h, w, c = memory.shape
        memory = memory.reshape(b, h * w, c)
        kv_p_m = mask.reshape(-1, 1, h * w)
        k_pos = self.kv_pe(NestedTensor(memory, mask)).permute(0, 2, 3, 1).reshape(b, h * w, 1, c)
        # Enhance visual context with triplet decoder.
        query_embeds = [None] * len(ho_queries)
        cross_attn_weights_list = [None] * len(ho_queries)
        rgip_attn_stats = {
            'hit_images': 0,
            'student_query_count_sum': 0,
            'context_mask_count': 0,
            'prepared_attn_mask_count': 0,
            'modulated_pair_count': 0,
            'attn_alignment_fallback': 0,
            'attn_bias_abs_sum': 0.0,
            'attn_bias_count': 0,
        }
        if use_vas and is_replay_list is not None and True in is_replay_list and False in is_replay_list:
            # === 开启 VAS 模式 (Batch中同时包含新旧样本) ===
            print(f"[Debug] VAS Triggered! New samples: {is_replay_list.count(False)}, Old samples: {is_replay_list.count(True)}", flush=True)
            new_indices = [i for i, r in enumerate(is_replay_list) if not r]
            old_indices = [i for i, r in enumerate(is_replay_list) if r]

            # [阶段1]：处理新任务样本，提取干涉源 M_dist
            new_attns = []
            for i in new_indices:
                out, cw = self.decoder(
                    ho_queries[i].unsqueeze(1), memory[i].unsqueeze(1),
                    kv_padding_mask=kv_p_m[i], q_pos=positional_embeds[i], k_pos=k_pos[i],
                    return_cross_attn=True
                )
                query_embeds[i] = out.squeeze(dim=2)
                cross_attn_weights_list[i] = cw

                # 提取单张图片的注意力矩阵 [1, heads, num_queries, h*w]。
                # 0 query 图片不会产生有效 cross-attention，直接跳过即可。
                if isinstance(cw, list):
                    cw_last_layer = cw[-1] if len(cw) > 0 else None
                else:
                    cw_last_layer = cw
                if cw_last_layer is not None and cw_last_layer.numel() > 0:
                    # ✅ 关键修复：先在内部沿着 batch(0), heads(1), queries(2) 维度求平均
                    # 将形状从 [1, 8, num_queries, 1050] 降维成 [1050] 的纯空间分布
                    spatial_map = cw_last_layer.detach().mean(dim=(0, 1, 2))
                    new_attns.append(spatial_map)

            # 计算 M_dist (干涉源分布)
            if len(new_attns) > 0:
                # 此时 new_attns 里的每个元素都是形状严格相同的 [1050] 张量
                stacked_attns = torch.stack(new_attns) # 形状: [新样本数量, 1050]
                # 在样本维度上取平均，得到 Batch 级别的新知识视觉干涉热区
                m_dist = stacked_attns.mean(dim=0)     # 形状: [1050]

                # 归一化到 0~1 之间，确保排斥力度 (P_bias) 的相对稳定性
                m_dist = (m_dist - m_dist.min()) / (m_dist.max() - m_dist.min() + 1e-8)
            else:
                m_dist = None

            # [阶段2]：处理旧样本，施加基于 SIS 的负向偏置 P_bias
            for i in old_indices:
                sis = sis_scores[i]
                memory_mask = None

                if m_dist is not None and sis > 0:
                    # 惩罚项: P_bias = lambda * SIS * M_dist
                    num_queries = ho_queries[i].shape[0]
                    # 扩展至 [num_queries, h*w]
                    p_bias = vas_lambda * sis * m_dist.unsqueeze(0).expand(num_queries, -1)
                    memory_mask = -p_bias # 负向偏置

                out, cw = self.decoder(
                    ho_queries[i].unsqueeze(1), memory[i].unsqueeze(1),
                    kv_padding_mask=kv_p_m[i], q_pos=positional_embeds[i], k_pos=k_pos[i],
                    qk_attn_mask=memory_mask, # ✅ 关键修复：将 memory_mask 改为 qk_attn_mask !
                    return_cross_attn=True
                )
                query_embeds[i] = out.squeeze(dim=2)
                cross_attn_weights_list[i] = cw
        else:
            for i, (ho_q, mem) in enumerate(zip(ho_queries, memory)):
                global_idx = int(batch_indices[i]) if batch_indices is not None else i
                image_rgip_context = None
                decoder_rgip_context = None
                if rgip_context is not None:
                    image_rgip_context = rgip_context.get(global_idx, None)
                    if image_rgip_context is None:
                        image_rgip_context = rgip_context.get(str(global_idx), None)
                    if image_rgip_context is None and len(rgip_context) == 1 and len(ho_queries) == 1:
                        image_rgip_context = next(iter(rgip_context.values()))
                    decoder_rgip_context = self._prepare_rgip_attn_context(
                        image_rgip_context, int(ho_q.shape[0])
                    )
                    if image_rgip_context is not None:
                        rgip_attn_stats['hit_images'] += 1
                        rgip_attn_stats['student_query_count_sum'] += int(ho_q.shape[0])
                        image_rgip_context['student_query_count'] = torch.as_tensor(
                            int(ho_q.shape[0]), device=ho_q.device
                        )
                        raw_mask = image_rgip_context.get('modulate_mask', None)
                        if raw_mask is not None:
                            rgip_attn_stats['context_mask_count'] += int(raw_mask.sum().detach().cpu())
                        image_rgip_context['attn_context_mask_count'] = torch.as_tensor(
                            0, device=ho_q.device
                        ) if raw_mask is None else raw_mask.sum().detach().to(device=ho_q.device)
                        prepared_mask = (
                            decoder_rgip_context.get('modulate_mask', None)
                            if decoder_rgip_context is not None else None
                        )
                        if prepared_mask is not None:
                            rgip_attn_stats['prepared_attn_mask_count'] += int(
                                prepared_mask.sum().detach().cpu()
                            )
                        image_rgip_context['prepared_attn_mask_count'] = torch.as_tensor(
                            0, device=ho_q.device
                        ) if prepared_mask is None else prepared_mask.sum().detach().to(
                            device=ho_q.device
                        )
                out, cross_attn_weights = self.decoder(
                    ho_q.unsqueeze(1), mem.unsqueeze(1), kv_padding_mask=kv_p_m[i],
                    q_pos=positional_embeds[i], k_pos=k_pos[i], return_cross_attn=True,
                    rgip_attn_context=decoder_rgip_context
                )
                if (
                    image_rgip_context is not None and decoder_rgip_context is not None
                    and decoder_rgip_context is not image_rgip_context
                ):
                    for stat_key in (
                        'attn_bias_abs_mean',
                        'modulated_pair_count',
                        '_rgip_attn_alignment_fallback',
                    ):
                        if stat_key in decoder_rgip_context:
                            image_rgip_context[stat_key] = decoder_rgip_context[stat_key]
                if decoder_rgip_context is not None:
                    if 'modulated_pair_count' in decoder_rgip_context:
                        rgip_attn_stats['modulated_pair_count'] += int(
                            decoder_rgip_context['modulated_pair_count'].detach().cpu()
                        )
                    if 'attn_bias_abs_mean' in decoder_rgip_context:
                        rgip_attn_stats['attn_bias_abs_sum'] += float(
                            decoder_rgip_context['attn_bias_abs_mean'].detach().cpu()
                        )
                        rgip_attn_stats['attn_bias_count'] += 1
                    if '_rgip_attn_alignment_fallback' in decoder_rgip_context:
                        rgip_attn_stats['attn_alignment_fallback'] += int(
                            decoder_rgip_context['_rgip_attn_alignment_fallback'].detach().cpu()
                        )
                out = out.squeeze(dim=2)
                # ------ 关键：Hint分支仅限shape完全对齐时融合，否则用student ------
                if (
                    teacher_cross_attn_hint is not None and
                    i < len(teacher_cross_attn_hint) and
                    teacher_cross_attn_hint[i] is not None
                ):
                    th = teacher_cross_attn_hint[i]
                    cw = cross_attn_weights
                    # 有的包实现返回list，多层，仅取最后一层
                    if isinstance(th, list):
                        th = th[-1] if len(th) > 0 else None
                    if isinstance(cw, list):
                        cw = cw[-1] if len(cw) > 0 else None
                    # 核心防呆：只有在完全shape一致时融合
                    if th is not None and cw is not None and th.shape == cw.shape:
                        cross_attn_weights = (1 - attn_hint_alpha) * cw + attn_hint_alpha * th.detach()
                        # print(f"[Hint] Blended student & teacher at batch {i}")
                    else:
                        # print(f"[Hint] SKIPPED at batch {i}: student {cw.shape}, teacher {th.shape}")
                        cross_attn_weights = cw if cw is not None else cross_attn_weights
                query_embeds[i] = out
                cross_attn_weights_list[i] = cross_attn_weights
        # 将每张图的 decoder 输出沿 HO pair 维拼接。
        # 注意：不同图片的 pair 数可以不同，甚至为 0；只能要求 decoder 层数和特征维度一致。
        query_embeds_to_cat = []
        base_layers = None
        base_dim = None
        for idx, q in enumerate(query_embeds):
            if q is None:
                continue
            if q.dim() == 4 and q.shape[2] == 1:
                q = q.squeeze(dim=2)
            if q.dim() != 3:
                pair_count = len(paired_inds[idx]) if idx < len(paired_inds) else -1
                if pair_count == 0:
                    continue
                raise RuntimeError(f"Unexpected query embedding shape at image {idx}: {tuple(q.shape)}")
            # 兼容旧 decoder 在 0 query + return_cross_attn=True 时被错误拆包后留下的畸形空张量。
            if q.shape[0] == 0:
                pair_count = len(paired_inds[idx]) if idx < len(paired_inds) else -1
                if pair_count == 0:
                    continue
                raise RuntimeError(f"Unexpected empty decoder-layer dimension at image {idx}: {tuple(q.shape)}")
            if base_layers is None:
                base_layers = q.shape[0]
                base_dim = q.shape[2]
            if q.shape[0] == base_layers and q.shape[2] == base_dim:
                query_embeds_to_cat.append(q)
            else:
                pair_count = len(paired_inds[idx]) if idx < len(paired_inds) else -1
                if pair_count == 0:
                    continue
                raise RuntimeError(
                    f"Query embedding layer/feature mismatch at image {idx}: "
                    f"{tuple(q.shape)} vs expected (*,{base_layers},*,{base_dim})"
                )
        if len(query_embeds_to_cat) == 0:
            num_layers = int(getattr(self.decoder, 'num_layers', 1))
            if not getattr(self.decoder, 'return_intermediate', True):
                num_layers = 1
            num_layers = max(num_layers, 1)
            query_embeds = torch.zeros(
                (num_layers, 0, self.repr_size),
                device=memory.device,
                dtype=memory.dtype,
            )
        else:
            query_embeds = torch.cat(query_embeds_to_cat, dim=1)
        expected_pairs = sum(len(p_inds) for p_inds in paired_inds)
        if query_embeds.shape[1] != expected_pairs:
            raise RuntimeError(
                f"Concatenated query count mismatch: got {query_embeds.shape[1]}, "
                f"expected {expected_pairs}"
            )
        # query_embeds = torch.cat(query_embeds, dim=1)
        logits = self.binary_classifier(query_embeds)
        pred_logits = logits      # [num_decoder_layers, N_pairs, num_verbs]
        pred_boxes = None         # <-- If you have box regression, add here
        if query_embeds.shape[0] == 0:
            # fallback/dummy，建议和你的repr_size一致
            feat = torch.zeros((1, query_embeds.shape[-1]), device=query_embeds.device, dtype=query_embeds.dtype)
            print("[Warning] Query_embeds is empty, fallback to zero feature.", flush=True)
        elif query_embeds.shape[0] == 1:
            feat = query_embeds[-1]
        else:
            feat = query_embeds[-2]
        # feat = query_embeds[-2] if query_embeds.shape[0] > 1 else query_embeds[-1]  # [N_pairs, repr_dim]

        if self.training or (targets is not None and return_outputs):
            labels = associate_with_ground_truth(
                boxes, paired_inds, targets, self.num_verbs
            )
            pair_weights = torch.ones(labels.shape[0], device=logits.device, dtype=logits.dtype)
            if rgip_context is not None and len(paired_inds) > 0:
                offset = 0
                for local_img_idx, p_inds in enumerate(paired_inds):
                    n_pairs = len(p_inds)
                    global_idx = int(batch_indices[local_img_idx]) if batch_indices is not None else local_img_idx
                    ctx = rgip_context.get(global_idx, None)
                    if ctx is not None and 'pair_weights' in ctx and n_pairs > 0:
                        src = ctx['pair_weights'].to(device=pair_weights.device, dtype=pair_weights.dtype)
                        pair_weights[offset:offset + min(n_pairs, len(src))] = src[:n_pairs]
                    offset += n_pairs

            cls_loss = self.compute_classification_loss(
                logits, prior_scores, labels, pair_weights=pair_weights
            )
                # 组装用于蒸馏的输出

            # 计算pair_image_indices
            pair_image_indices = []
            pair_idx_in_image = []
            for img_idx, p_inds in enumerate(paired_inds):
                pair_image_indices.extend([img_idx] * len(p_inds))
                pair_idx_in_image.extend(list(range(len(p_inds))))
            pair_image_indices = torch.tensor(pair_image_indices, device=feat.device)  # [num_pairs]
            pair_idx_in_image = torch.tensor(pair_idx_in_image, device=feat.device)    # [num_pairs]
            if batch_indices is not None and len(pair_image_indices) > 0:
                pair_global_indices = torch.as_tensor(
                    [int(batch_indices[int(img_idx)]) for img_idx in pair_image_indices.tolist()],
                    device=feat.device,
                    dtype=torch.long
                )
            else:
                pair_global_indices = pair_image_indices.clone()

            pair_objects = torch.cat(object_types, dim=0) if len(object_types) > 0 else torch.zeros(
                0, device=feat.device, dtype=torch.long
            )
            pair_prior = torch.cat(prior_scores, dim=0).prod(1) if len(prior_scores) > 0 else torch.zeros(
                0, self.num_verbs, device=feat.device, dtype=logits.dtype
            )


            output_dict = {
                'cls_loss': cls_loss,
                'pred_logits': pred_logits,
                'feat': feat,
                'pair_image_indices': pair_image_indices,  # 新增
                'pair_idx_in_image': pair_idx_in_image,    # 新增
                'pair_labels': labels,
                'pair_objects': pair_objects,
                'pair_prior': pair_prior,
                'pair_global_indices': pair_global_indices,
                'paired_inds': paired_inds,

            }
            if return_rgip_meta:
                output_dict['rgip_pair_weights'] = pair_weights
                output_dict['rgip_attn_stats'] = rgip_attn_stats
                rgip_sis = torch.zeros_like(pair_weights)
                rgip_confounder_ids = []
                rgip_confounder_weights = []
                offset = 0
                for local_img_idx, p_inds in enumerate(paired_inds):
                    n_pairs = len(p_inds)
                    global_idx = int(batch_indices[local_img_idx]) if batch_indices is not None else local_img_idx
                    ctx = rgip_context.get(global_idx, None) if rgip_context is not None else None
                    if ctx is not None:
                        if 'sis' in ctx:
                            src = ctx['sis'].to(device=rgip_sis.device, dtype=rgip_sis.dtype)
                            rgip_sis[offset:offset + min(n_pairs, len(src))] = src[:n_pairs]
                        if 'confounder_hoi_ids' in ctx:
                            rgip_confounder_ids.append(ctx['confounder_hoi_ids'][:n_pairs].to(feat.device))
                        if 'confounder_weights' in ctx:
                            rgip_confounder_weights.append(ctx['confounder_weights'][:n_pairs].to(feat.device))
                    offset += n_pairs
                output_dict['rgip_sis'] = rgip_sis
                if rgip_confounder_ids:
                    output_dict['rgip_confounder_ids'] = torch.cat(rgip_confounder_ids, dim=0)
                if rgip_confounder_weights:
                    output_dict['rgip_confounder_weights'] = torch.cat(rgip_confounder_weights, dim=0)
            #loss_dict = dict(cls_loss=cls_loss)
                    # pred_boxes（如果有）也可以加入
            if pred_boxes is not None:
                output_dict['pred_boxes'] = pred_boxes
            if return_outputs:
                if return_cross_attn:
                    return output_dict, cross_attn_weights_list  # 若你已经构造过cross_attn_weights_list
                    # 或 return output_dict, None
                else:
                    return output_dict
            else:
                if return_cross_attn:
                    return {'cls_loss': cls_loss}, cross_attn_weights_list  # 或None
                else:
                    return {'cls_loss': cls_loss}

        detections = self.postprocessing(
            boxes, paired_inds, object_types,
            logits[-1], prior_scores, image_sizes
        )
        if return_cross_attn:
            return logits, cross_attn_weights_list
        else:
            # 旧逻辑
            pass
        return detections

def build_detector(args, obj_to_verb):
    if args.detector == "base":
        detr, _, postprocessors = build_base_detr(args)
    elif args.detector == "advanced":
        detr, _, postprocessors = build_advanced_detr(args)

    if os.path.exists(args.pretrained):
        if dist.is_initialized():
            print(f"Rank {dist.get_rank()}: Load weights for the object detector from {args.pretrained}")
        else:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])

    ho_matcher = HumanObjectMatcher(
        repr_size=args.repr_dim,
        num_verbs=args.num_verbs,
        obj_to_verb=obj_to_verb,
        dropout=args.dropout
    )
    decoder_layer = TransformerDecoderLayer(
        q_dim=args.repr_dim, kv_dim=args.hidden_dim,
        ffn_interm_dim=args.repr_dim * 4,
        num_heads=args.nheads, dropout=args.dropout
    )
    triplet_decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=args.triplet_dec_layers
    )
    return_layer = {"C5": -1, "C4": -2, "C3": -3}[args.kv_src]
    if isinstance(detr.backbone.num_channels, list):
        num_channels = detr.backbone.num_channels[-1]
    else:
        num_channels = detr.backbone.num_channels
    feature_head = FeatureHead(
        args.hidden_dim, num_channels,
        return_layer, args.triplet_enc_layers
    )
    model = PViC(
        (detr, args.detector), postprocessors['bbox'],
        feature_head=feature_head,
        ho_matcher=ho_matcher,
        triplet_decoder=triplet_decoder,
        num_verbs=args.num_verbs,
        repr_size=args.repr_dim,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        raw_lambda=args.raw_lambda,
    )
    return model
