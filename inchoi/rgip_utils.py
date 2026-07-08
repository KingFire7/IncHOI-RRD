"""Rarity-Guided Interaction Preservation utilities.

This module keeps the RGIP method-specific state out of the original PViC
entrypoints.  The classifier still predicts predicates/verbs; every HOI-level
operation below is therefore explicitly routed through object+verb mappings.
"""

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def unwrap_dataset(dataset):
    """Return the underlying dataset behind torch Subset/DataFactory wrappers."""
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    return dataset


def build_hico_object_verb_mappings(dataset):
    """Build HOI/object/verb maps from a HICO-style dataset object."""
    dataset = unwrap_dataset(dataset)
    hoi_to_obj = {}
    hoi_to_verb = {}
    obj_verb_to_hoi = {}

    if hasattr(dataset, 'class_corr'):
        correspondence = dataset.class_corr
    else:
        correspondence = []

    for hoi_id, obj_id, verb_id in correspondence:
        hoi_id, obj_id, verb_id = int(hoi_id), int(obj_id), int(verb_id)
        hoi_to_obj[hoi_id] = obj_id
        hoi_to_verb[hoi_id] = verb_id
        obj_verb_to_hoi[(obj_id, verb_id)] = hoi_id

    hoi_frequency = None
    if hasattr(dataset, 'anno_interaction'):
        hoi_frequency = dataset.anno_interaction

    return hoi_to_obj, hoi_to_verb, obj_verb_to_hoi, hoi_frequency


class RarityGuidedState:
    """State and computations for RGIP training.

    The class is intentionally defensive: if a batch has no replay image, no old
    positive pair, or no current prototype yet, the corresponding RGIP term
    cleanly returns zero and the normal PViC training path remains valid.
    """

    def __init__(
        self,
        num_verbs,
        hoi_to_obj,
        hoi_to_verb,
        obj_verb_to_hoi,
        object_to_verb,
        old_hoi_classes,
        current_hoi_classes,
        args,
        device,
        hoi_frequency=None,
    ):
        self.num_verbs = int(num_verbs)
        self.hoi_to_obj = {int(k): int(v) for k, v in (hoi_to_obj or {}).items()}
        self.hoi_to_verb = {int(k): int(v) for k, v in (hoi_to_verb or {}).items()}
        self.obj_verb_to_hoi = self._normalise_obj_verb_map(obj_verb_to_hoi)
        self.object_to_verb = object_to_verb
        self.old_hoi_classes = {int(c) for c in (old_hoi_classes or [])}
        self.current_hoi_classes = {int(c) for c in (current_hoi_classes or [])}
        self.args = args
        self.device = torch.device(device)
        self.topk = max(int(getattr(args, 'rgip_topk', 3)), 0)

        self.prototypes: Dict[int, torch.Tensor] = {}
        self.prototype_counts: Dict[int, int] = defaultdict(int)
        self.rarity_risk = self._build_rarity_risk(hoi_frequency)
        self.predicate_embeddings = self._load_predicate_embeddings()
        self.last_stats = {}

    @staticmethod
    def _normalise_obj_verb_map(obj_verb_to_hoi):
        mapping = {}
        if obj_verb_to_hoi is None:
            return mapping
        if isinstance(obj_verb_to_hoi, dict):
            iterator = obj_verb_to_hoi.items()
        else:
            iterator = []
        for key, value in iterator:
            if value is None:
                continue
            if isinstance(key, str):
                key = key.strip('()').replace(',', ' ').split()
            obj_id, verb_id = key
            mapping[(int(obj_id), int(verb_id))] = int(value)
        return mapping

    def _build_rarity_risk(self, hoi_frequency):
        if hoi_frequency is None:
            n = max(max(self.hoi_to_obj.keys(), default=-1) + 1, 600)
            return torch.zeros(n, dtype=torch.float32, device=self.device)

        freq = torch.as_tensor(hoi_frequency, dtype=torch.float32, device=self.device)
        if freq.numel() == 0:
            return freq
        log_freq = torch.log1p(freq.clamp_min(0))
        nonzero = freq > 0
        if torch.any(nonzero):
            min_v = log_freq[nonzero].min()
            max_v = log_freq[nonzero].max()
            norm = (log_freq - min_v) / (max_v - min_v + 1e-6)
            risk = 1.0 - norm
            risk = risk.clamp(0.0, 1.0)
            risk[~nonzero] = 1.0
            return risk
        return torch.ones_like(freq)

    def _load_predicate_embeddings(self):
        if not getattr(self.args, 'rgip_use_semantic_confounder', False):
            return None
        path = getattr(self.args, 'rgip_predicate_embedding_path', '')
        if not path:
            return None
        try:
            if path.endswith('.npy'):
                emb = torch.from_numpy(np.load(path))
            else:
                obj = torch.load(path, map_location='cpu')
                emb = obj['embeddings'] if isinstance(obj, dict) and 'embeddings' in obj else obj
            emb = emb.float().to(self.device)
            if emb.ndim != 2 or emb.shape[0] < self.num_verbs:
                return None
            return F.normalize(emb[:self.num_verbs], dim=-1)
        except Exception as exc:
            if getattr(self.args, 'rgip_debug', False):
                print(f"[RGIP] Failed to load predicate embeddings from {path}: {exc}")
            return None

    def _lookup_hoi(self, obj_id: int, verb_id: int) -> Optional[int]:
        return self.obj_verb_to_hoi.get((int(obj_id), int(verb_id)), None)

    def _compatible_verbs_for_object(self, obj_id: int) -> List[int]:
        obj_id = int(obj_id)
        verbs = []
        if isinstance(self.object_to_verb, dict):
            verbs = self.object_to_verb.get(obj_id, [])
        elif self.object_to_verb is not None and 0 <= obj_id < len(self.object_to_verb):
            verbs = self.object_to_verb[obj_id]
        return sorted({int(v) for v in verbs if 0 <= int(v) < self.num_verbs})

    def _old_compatible_verbs_for_object(self, obj_id: int) -> List[int]:
        verbs = []
        for verb_id in self._compatible_verbs_for_object(obj_id):
            hoi_id = self._lookup_hoi(obj_id, verb_id)
            if hoi_id in self.old_hoi_classes:
                verbs.append(verb_id)
        return verbs

    def _risk_for_hoi(self, hoi_id: int, dtype, device):
        if 0 <= int(hoi_id) < self.rarity_risk.numel():
            return self.rarity_risk[int(hoi_id)].to(device=device, dtype=dtype)
        return torch.tensor(0.0, device=device, dtype=dtype)

    def compute_sis_for_teacher_pairs(self, teacher_output):
        """Compute SIS and batch-normalised replay weights for teacher pairs."""
        labels = teacher_output.get('pair_labels')
        objects = teacher_output.get('pair_objects')
        logits = teacher_output.get('pred_logits')
        if labels is None or objects is None or logits is None:
            return None

        logits = logits[-1].detach()
        device, dtype = logits.device, logits.dtype
        num_pairs = labels.shape[0]
        sis = torch.zeros(num_pairs, device=device, dtype=dtype)
        pair_weights = torch.ones(num_pairs, device=device, dtype=dtype)
        old_predicate_masks = torch.zeros(num_pairs, self.num_verbs, device=device, dtype=torch.bool)
        old_positive_verbs: List[List[int]] = [[] for _ in range(num_pairs)]
        old_positive_hoi: List[List[int]] = [[] for _ in range(num_pairs)]
        selected_old_positive_verb = torch.full((num_pairs,), -1, device=device, dtype=torch.long)

        alpha = float(getattr(self.args, 'rgip_alpha', 0.4))
        beta = float(getattr(self.args, 'rgip_beta', 0.3))
        entropy_weight = max(0.0, 1.0 - alpha - beta)
        temperature = max(float(getattr(self.args, 'rgip_temperature', 2.0)), 1e-6)

        for pair_idx in range(num_pairs):
            obj_id = int(objects[pair_idx].item())
            old_compatible = self._old_compatible_verbs_for_object(obj_id)
            if old_compatible:
                old_predicate_masks[pair_idx, old_compatible] = True

            positives = torch.nonzero(labels[pair_idx] > 0.5, as_tuple=False).flatten().tolist()
            positive_infos = []
            for verb_id in positives:
                hoi_id = self._lookup_hoi(obj_id, int(verb_id))
                if hoi_id in self.old_hoi_classes:
                    positive_infos.append((int(verb_id), int(hoi_id)))

            if not positive_infos:
                continue

            old_positive_verbs[pair_idx] = [v for v, _ in positive_infos]
            old_positive_hoi[pair_idx] = [c for _, c in positive_infos]

            compatible = self._compatible_verbs_for_object(obj_id)
            entropy_risk = torch.tensor(0.0, device=device, dtype=dtype)
            if len(compatible) > 1:
                comp_logits = logits[pair_idx, compatible] / temperature
                prob = F.softmax(comp_logits, dim=0)
                entropy = -(prob * (prob + 1e-8).log()).sum()
                entropy_risk = entropy / math.log(len(compatible))

            pair_risks = []
            for verb_id, hoi_id in positive_infos:
                freq_risk = self._risk_for_hoi(hoi_id, dtype=dtype, device=device)
                negatives = [v for v in old_compatible if v != verb_id]
                if negatives:
                    z_pos = logits[pair_idx, verb_id]
                    z_neg = logits[pair_idx, negatives].max()
                    margin_risk = torch.sigmoid(-(z_pos - z_neg))
                else:
                    margin_risk = torch.tensor(0.0, device=device, dtype=dtype)
                risk = alpha * freq_risk + beta * margin_risk + entropy_weight * entropy_risk
                pair_risks.append((risk.clamp(0.0, 1.0), verb_id))

            if pair_risks:
                risk, selected_verb = max(pair_risks, key=lambda item: float(item[0].detach().cpu()))
                sis[pair_idx] = risk
                selected_old_positive_verb[pair_idx] = int(selected_verb)

        valid = sis > 0
        if torch.any(valid):
            raw_weights = 1.0 + float(getattr(self.args, 'rgip_gamma', 2.0)) * sis[valid]
            pair_weights[valid] = (raw_weights / (raw_weights.mean() + 1e-6)).clamp(0.1, 10.0)

        return {
            'sis': sis,
            'pair_weights': pair_weights,
            'old_predicate_masks': old_predicate_masks,
            'old_positive_verbs': old_positive_verbs,
            'old_positive_hoi': old_positive_hoi,
            'selected_old_positive_verb': selected_old_positive_verb,
            'num_old_positive_pairs': int(valid.sum().item()),
        }

    def _select_confounders(self, pair_feat, pair_obj, selected_old_verb, dtype, device):
        if self.topk == 0 or not self.prototypes or selected_old_verb < 0:
            return None

        eta_object = float(getattr(self.args, 'rgip_eta_object', 0.4))
        eta_query = float(getattr(self.args, 'rgip_eta_query', 0.4))
        eta_sem = float(getattr(self.args, 'rgip_eta_sem', 0.2))
        if self.predicate_embeddings is None:
            eta_sem = 0.0
        eta_sum = max(eta_object + eta_query + eta_sem, 1e-6)
        eta_object, eta_query, eta_sem = eta_object / eta_sum, eta_query / eta_sum, eta_sem / eta_sum

        pair_feat = F.normalize(pair_feat.detach().to(device=device, dtype=dtype), dim=-1)
        scores = []
        hoi_ids = []
        verbs = []
        protos = []
        for hoi_id in sorted(self.current_hoi_classes):
            proto = self.prototypes.get(hoi_id, None)
            if proto is None:
                continue
            obj_id = self.hoi_to_obj.get(hoi_id, None)
            verb_id = self.hoi_to_verb.get(hoi_id, None)
            if obj_id is None or verb_id is None:
                continue
            proto = proto.to(device=device, dtype=dtype)
            same_object = torch.tensor(1.0 if int(obj_id) == int(pair_obj) else 0.0, device=device, dtype=dtype)
            query_sim = F.cosine_similarity(pair_feat.unsqueeze(0), proto.unsqueeze(0), dim=-1).squeeze(0)
            sem_sim = torch.tensor(0.0, device=device, dtype=dtype)
            if self.predicate_embeddings is not None:
                e_old = self.predicate_embeddings[int(selected_old_verb)].to(device=device, dtype=dtype)
                e_new = self.predicate_embeddings[int(verb_id)].to(device=device, dtype=dtype)
                sem_sim = F.cosine_similarity(e_old.unsqueeze(0), e_new.unsqueeze(0), dim=-1).squeeze(0)
            score = eta_object * same_object + eta_query * query_sim + eta_sem * sem_sim
            scores.append(score)
            hoi_ids.append(int(hoi_id))
            verbs.append(int(verb_id))
            protos.append(proto)

        if not scores:
            return None

        scores = torch.stack(scores)
        k = min(self.topk, scores.numel())
        top_scores, top_idx = torch.topk(scores, k=k)
        tau = max(float(getattr(self.args, 'rgip_confounder_tau', 0.2)), 1e-6)
        weights = F.softmax(top_scores / tau, dim=0)
        return {
            'hoi_ids': torch.tensor([hoi_ids[i] for i in top_idx.tolist()], device=device, dtype=torch.long),
            'verbs': torch.tensor([verbs[i] for i in top_idx.tolist()], device=device, dtype=torch.long),
            'weights': weights,
            'prototypes': torch.stack([protos[i] for i in top_idx.tolist()]).to(device=device, dtype=dtype),
            'same_object_ratio': float(np.mean([self.hoi_to_obj.get(hoi_ids[i]) == int(pair_obj) for i in top_idx.tolist()])),
        }

    def build_replay_pair_context(self, teacher_output, selected_global_indices=None):
        sis_meta = self.compute_sis_for_teacher_pairs(teacher_output)
        if sis_meta is None:
            return {}

        feat = teacher_output['feat'].detach()
        device, dtype = feat.device, feat.dtype
        pair_global_indices = teacher_output.get('pair_global_indices')
        pair_idx_in_image = teacher_output.get('pair_idx_in_image')
        pair_objects = teacher_output.get('pair_objects')
        if pair_global_indices is None or pair_idx_in_image is None:
            return {}

        selected_set = None
        if selected_global_indices is not None:
            selected_set = {int(idx) for idx in selected_global_indices}

        counts = defaultdict(int)
        for gid, local_idx in zip(pair_global_indices.tolist(), pair_idx_in_image.tolist()):
            if selected_set is not None and int(gid) not in selected_set:
                continue
            counts[int(gid)] = max(counts[int(gid)], int(local_idx) + 1)

        context = {}
        for gid, n_pairs in counts.items():
            context[gid] = self._empty_image_context(n_pairs, feat.shape[-1], device, dtype)

        confounder_valid = 0
        same_object_values = []
        selected_pair_indices = []
        for pair_idx, (gid, local_idx) in enumerate(zip(pair_global_indices.tolist(), pair_idx_in_image.tolist())):
            gid, local_idx = int(gid), int(local_idx)
            if selected_set is not None and gid not in selected_set:
                continue
            if gid not in context:
                continue
            selected_pair_indices.append(pair_idx)
            ctx = context[gid]
            ctx['pair_weights'][local_idx] = sis_meta['pair_weights'][pair_idx]
            ctx['sis'][local_idx] = sis_meta['sis'][pair_idx]
            ctx['old_predicate_masks'][local_idx] = sis_meta['old_predicate_masks'][pair_idx]
            ctx['old_positive_verbs'][local_idx] = sis_meta['old_positive_verbs'][pair_idx]
            ctx['old_positive_hoi'][local_idx] = sis_meta['old_positive_hoi'][pair_idx]
            ctx['selected_old_positive_verb'][local_idx] = sis_meta['selected_old_positive_verb'][pair_idx]

            selected_verb = int(sis_meta['selected_old_positive_verb'][pair_idx].item())
            if sis_meta['sis'][pair_idx] <= 0:
                continue
            confounders = self._select_confounders(
                pair_feat=feat[pair_idx],
                pair_obj=int(pair_objects[pair_idx].item()),
                selected_old_verb=selected_verb,
                dtype=dtype,
                device=device,
            )
            if confounders is None:
                continue
            k = confounders['hoi_ids'].numel()
            ctx['confounder_hoi_ids'][local_idx, :k] = confounders['hoi_ids']
            ctx['confounder_verbs'][local_idx, :k] = confounders['verbs']
            ctx['confounder_weights'][local_idx, :k] = confounders['weights']
            ctx['confounder_prototypes'][local_idx, :k] = confounders['prototypes']
            ctx['modulate_mask'][local_idx] = getattr(self.args, 'rgip_lambda_attn', 0.5) > 0
            confounder_valid += 1
            same_object_values.append(confounders['same_object_ratio'])

        if selected_pair_indices:
            selected_sis = sis_meta['sis'][selected_pair_indices]
            selected_pair_weights = sis_meta['pair_weights'][selected_pair_indices]
            selected_old_positive_pairs = int((selected_sis > 0).sum().item())
        else:
            selected_sis = sis_meta['sis'].new_zeros((0,))
            selected_pair_weights = sis_meta['pair_weights'].new_ones((0,))
            selected_old_positive_pairs = 0

        total_pairs = int(sum(counts.values()))
        self.last_stats = {
            'rgip/num_replay_images': len(context),
            'rgip/num_replay_pairs': total_pairs,
            'rgip/num_old_positive_pairs': selected_old_positive_pairs,
            'rgip/sis_mean': float(selected_sis.mean().detach().cpu()) if selected_sis.numel() else 0.0,
            'rgip/sis_max': float(selected_sis.max().detach().cpu()) if selected_sis.numel() else 0.0,
            'rgip/sis_min': float(selected_sis.min().detach().cpu()) if selected_sis.numel() else 0.0,
            'rgip/pair_weight_mean': float(selected_pair_weights.mean().detach().cpu()) if selected_pair_weights.numel() else 1.0,
            'rgip/pair_weight_max': float(selected_pair_weights.max().detach().cpu()) if selected_pair_weights.numel() else 1.0,
            'rgip/prototype_count': len(self.prototypes),
            'rgip/confounder_valid_ratio': confounder_valid / max(selected_old_positive_pairs, 1),
            'rgip/same_object_ratio': float(np.mean(same_object_values)) if same_object_values else 0.0,
            'rgip/context_modulate_pairs': int(sum(
                int(ctx['modulate_mask'].sum().detach().cpu())
                for ctx in context.values()
            )),
        }
        return context

    def _empty_image_context(self, n_pairs, repr_dim, device, dtype):
        topk = self.topk
        return {
            'pair_weights': torch.ones(n_pairs, device=device, dtype=dtype),
            'sis': torch.zeros(n_pairs, device=device, dtype=dtype),
            'modulate_mask': torch.zeros(n_pairs, device=device, dtype=torch.bool),
            'old_positive_verbs': [[] for _ in range(n_pairs)],
            'old_positive_hoi': [[] for _ in range(n_pairs)],
            'selected_old_positive_verb': torch.full((n_pairs,), -1, device=device, dtype=torch.long),
            'confounder_hoi_ids': torch.full((n_pairs, topk), -1, device=device, dtype=torch.long),
            'confounder_verbs': torch.full((n_pairs, topk), -1, device=device, dtype=torch.long),
            'confounder_weights': torch.zeros(n_pairs, topk, device=device, dtype=dtype),
            'confounder_prototypes': torch.zeros(n_pairs, topk, repr_dim, device=device, dtype=dtype),
            'old_predicate_masks': torch.zeros(n_pairs, self.num_verbs, device=device, dtype=torch.bool),
            'lambda_attn': float(getattr(self.args, 'rgip_lambda_attn', 0.5)),
            'kappa': float(getattr(self.args, 'rgip_kappa', 1.0)),
            'attn_clamp': float(getattr(self.args, 'rgip_attn_clamp', 5.0)),
            'max_attn_pairs': int(getattr(self.args, 'rgip_max_attn_pairs', 16)),
        }

    @staticmethod
    def build_student_batch_context(replay_context, batch_indices):
        if not replay_context:
            return None
        batch_set = {int(idx) for idx in batch_indices}
        context = {int(k): v for k, v in replay_context.items() if int(k) in batch_set}
        return context or None

    @staticmethod
    def _pair_ids(output) -> List[Tuple[int, int]]:
        global_indices = output.get('pair_global_indices')
        local_indices = output.get('pair_idx_in_image')
        if global_indices is None or local_indices is None:
            return []
        return [(int(g), int(p)) for g, p in zip(global_indices.tolist(), local_indices.tolist())]

    def compute_interaction_distillation(self, student_output, teacher_output, replay_context):
        device = student_output['cls_loss'].device
        dtype = student_output['cls_loss'].dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        if not replay_context:
            return zero, {}

        student_ids = self._pair_ids(student_output)
        teacher_ids = self._pair_ids(teacher_output)
        if not student_ids or not teacher_ids:
            return zero, {}

        student_id2idx = {pid: i for i, pid in enumerate(student_ids)}
        teacher_id2idx = {pid: i for i, pid in enumerate(teacher_ids)}
        common_ids = sorted(set(student_id2idx) & set(teacher_id2idx))
        if not common_ids:
            return zero, {}

        s_feat = student_output.get('feat')
        t_feat = teacher_output.get('feat')
        s_pred_logits = student_output.get('pred_logits')
        t_pred_logits = teacher_output.get('pred_logits')
        if (
            s_feat is None or t_feat is None
            or s_pred_logits is None or t_pred_logits is None
            or s_pred_logits.dim() < 2 or t_pred_logits.dim() < 2
            or s_pred_logits.shape[0] == 0 or t_pred_logits.shape[0] == 0
            or s_feat.shape[0] == 0 or t_feat.shape[0] == 0
        ):
            return zero, {}

        s_logits = s_pred_logits[-1]
        t_feat = t_feat.detach().to(device)
        t_logits = t_pred_logits[-1].detach().to(device)
        s_pair_count = min(s_feat.shape[0], s_logits.shape[0])
        t_pair_count = min(t_feat.shape[0], t_logits.shape[0])
        common_ids = [
            pid for pid in common_ids
            if student_id2idx[pid] < s_pair_count and teacher_id2idx[pid] < t_pair_count
        ]
        if not common_ids:
            return zero, {}

        pair_prior = student_output.get('pair_prior')

        total = zero.clone()
        weight_sum = zero.clone()
        feat_sum = zero.clone()
        logit_sum = zero.clone()
        hardneg_sum = zero.clone()
        used_pairs = 0
        temperature = max(float(getattr(self.args, 'rgip_temperature', 2.0)), 1e-6)
        margin = float(getattr(self.args, 'rgip_margin', 0.2))

        for pid in common_ids:
            gid, local_pair_idx = pid
            ctx = replay_context.get(int(gid), None)
            if ctx is None or local_pair_idx >= len(ctx['pair_weights']):
                continue
            if len(ctx['old_positive_verbs'][local_pair_idx]) == 0:
                continue

            si = student_id2idx[pid]
            ti = teacher_id2idx[pid]
            weight = ctx['pair_weights'][local_pair_idx].to(device=device, dtype=dtype)

            feat_loss = F.mse_loss(s_feat[si], t_feat[ti], reduction='mean')
            old_mask = ctx['old_predicate_masks'][local_pair_idx].to(device=device)
            if old_mask.sum() > 1:
                teacher_prob = F.softmax(t_logits[ti, old_mask] / temperature, dim=-1)
                student_logp = F.log_softmax(s_logits[si, old_mask] / temperature, dim=-1)
                logit_loss = F.kl_div(student_logp, teacher_prob, reduction='sum') * temperature * temperature
            else:
                logit_loss = zero.clone()

            hardneg_loss = zero.clone()
            pos_verb = int(ctx['selected_old_positive_verb'][local_pair_idx].item())
            conf_verbs = ctx['confounder_verbs'][local_pair_idx].to(device=device)
            conf_weights = ctx['confounder_weights'][local_pair_idx].to(device=device, dtype=dtype)
            valid_terms = []
            if pos_verb >= 0 and conf_verbs.numel() > 0:
                for conf_verb, conf_weight in zip(conf_verbs.tolist(), conf_weights):
                    conf_verb = int(conf_verb)
                    if conf_verb < 0 or conf_verb >= self.num_verbs or conf_weight <= 0:
                        continue
                    if (
                        pair_prior is not None
                        and (
                            si >= pair_prior.shape[0]
                            or conf_verb >= pair_prior.shape[1]
                            or pair_prior[si, conf_verb] <= 0
                        )
                    ):
                        continue
                    valid_terms.append(conf_weight * F.relu(margin + s_logits[si, conf_verb] - s_logits[si, pos_verb]))
            if valid_terms:
                hardneg_loss = torch.stack(valid_terms).sum()

            loss_i = (
                float(getattr(self.args, 'rgip_feat_weight', 1.0)) * feat_loss
                + float(getattr(self.args, 'rgip_logit_weight', 1.0)) * logit_loss
                + float(getattr(self.args, 'rgip_hardneg_weight', 0.5)) * hardneg_loss
            )
            total = total + weight * loss_i
            weight_sum = weight_sum + weight
            feat_sum = feat_sum + feat_loss.detach()
            logit_sum = logit_sum + logit_loss.detach()
            hardneg_sum = hardneg_sum + hardneg_loss.detach()
            used_pairs += 1

        if used_pairs == 0:
            return zero, {}

        loss = total / (weight_sum + 1e-6)
        stats = {
            'rgip/feat_kd': float((feat_sum / used_pairs).detach().cpu()),
            'rgip/logit_kd': float((logit_sum / used_pairs).detach().cpu()),
            'rgip/hardneg_loss': float((hardneg_sum / used_pairs).detach().cpu()),
            'rgip/int_loss': float(loss.detach().cpu()),
            'rgip/distilled_pairs': used_pairs,
        }
        return loss, stats

    def update_current_task_prototypes(self, student_output):
        labels = student_output.get('pair_labels')
        objects = student_output.get('pair_objects')
        feat = student_output.get('feat')
        if labels is None or objects is None or feat is None:
            return

        momentum = float(getattr(self.args, 'rgip_proto_momentum', 0.9))
        batch_features = defaultdict(list)
        with torch.no_grad():
            for pair_idx in range(labels.shape[0]):
                obj_id = int(objects[pair_idx].item())
                positives = torch.nonzero(labels[pair_idx] > 0.5, as_tuple=False).flatten().tolist()
                for verb_id in positives:
                    hoi_id = self._lookup_hoi(obj_id, int(verb_id))
                    if hoi_id in self.current_hoi_classes:
                        batch_features[int(hoi_id)].append(feat[pair_idx].detach())

            for hoi_id, features in batch_features.items():
                proto = F.normalize(torch.stack(features).mean(dim=0), dim=0)
                if hoi_id in self.prototypes:
                    old = self.prototypes[hoi_id].to(proto.device, proto.dtype)
                    proto = F.normalize(momentum * old + (1.0 - momentum) * proto, dim=0)
                self.prototypes[hoi_id] = proto.detach()
                self.prototype_counts[hoi_id] += len(features)
