"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import time
import torch
import pickle
import numpy as np
import scipy.io as sio
import copy

try:
    import wandb
except ImportError:
    pass

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, Subset
import torch.distributed as dist

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

from ops import recover_boxes
from detr.datasets import transforms as T
import torch.nn.functional as F

def custom_collate(batch):
    images, targets, indices = [], [], []
    for item in batch:
        if item is None: continue
        im, tar, idx = item   # DataFactory.__getitem__需返回(image, target, global_idx)
        if tar['labels'].numel() == 0: continue
        images.append(im)
        targets.append(tar)
        indices.append(idx)
    if len(images) == 0:
        print("custom_collate: All images in batch are filtered out, returning None batch.")
        return None, None, None
    return images, targets, indices

def get_base_dataset(ds):
    while hasattr(ds, 'dataset'):
        ds = ds.dataset
    return ds

def dict_to_device(d, device):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class DataFactory(Dataset):
    """
    支持增量学习的DataFactory，新增filter_classes参数用于只加载指定HOI类别的数据
    修正：确保indices为HICODet有效索引（非annotation索引）
    """
    def __init__(self, name, partition, data_root, filter_classes=None, max_per_class=None):
        """
        filter_classes: list[int]，仅加载这些HOI类别的样本（如不设则为全量）
        max_per_class: int，每个类别最多采样多少个（用于replay重放），如不设则为全部
        """
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, "hico_20160224_det/images", partition),
                anno_file=os.path.join(data_root, f"instances_{partition}.json"),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, f"instances_vcoco_{partition}.json"),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ), normalize,
            ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        self.name = name

        # 增量学习核心：只保留指定类别的数据，并支持按类别采样
        if filter_classes is not None:
            filtered_indices = []
            if name == 'hicodet':
                # 关键修正：遍历HICODet有效索引而非annotation索引
                class_to_indices = {cls: [] for cls in filter_classes}
                for i in range(len(self.dataset)):
                    ann_idx = self.dataset._idx[i]  # 真实annotation索引
                    ann = self.dataset.annotations[ann_idx]
                    hois = ann.get('hoi', [])
                    inter = set(hois) & set(filter_classes)
                    if inter:
                        filtered_indices.append(i)
                        for c in inter:
                            if c in class_to_indices:
                                class_to_indices[c].append(i)
                # 若有max_per_class，重采样
                if max_per_class is not None:
                    sampled = set()
                    for c, idxs in class_to_indices.items():
                        if len(idxs) > max_per_class:
                            idxs = np.random.choice(idxs, max_per_class, replace=False)
                        sampled.update(idxs)
                    filtered_indices = list(sampled)
                self.indices = filtered_indices
                self.dataset = Subset(self.dataset, self.indices)
            else:
                # TODO: VCOCO支持
                raise NotImplementedError("VCOCO filter_classes not implemented")
        else:
            self.indices = None  # 全量

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        image, target = self.transforms(image, target)
            # 如果没有有效labels，返回None
        if target['labels'].numel() == 0:
            # print(f"Warning: No valid labels for image {i} in dataset {self.name}.")
            return None
        global_idx = self.indices[i] if hasattr(self, "indices") and self.indices is not None else i
        return image, target, global_idx# 新增索引

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

class CustomisedDLE(DistributedLearningEngine):
    #新增修改
    def __init__(self, net, train_dataloader, test_dataloader,
                 config, filter_classes=None, teacher_model=None,
                 replay_indices=None, mir_dict=None, rare_set=None,
                 mir_min=0.0, mir_max=1.0):
        super().__init__(
            net, None, train_dataloader,
            print_interval=config.print_interval,
            cache_dir=config.output_dir,
            find_unused_parameters=True
        )
        device = None
        self.config = config
        self.max_norm = config.clip_max_norm
        self.test_dataloader = test_dataloader
        self.filter_classes = filter_classes  # 仅评估这些类别
        self._rank = dist.get_rank()
        self._device = torch.device(device) if device is not None else torch.device(self._rank)
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.cuda()
            # self.teacher_model = torch.nn.parallel.DistributedDataParallel(
            #     self.teacher_model, device_ids=[self._device],
            #     find_unused_parameters=False
            # )
        self.replay_indices = set(replay_indices) if replay_indices is not None else set()
        self.mir_dict = mir_dict if mir_dict is not None else {}
        self.rare_set = rare_set if rare_set is not None else set()
        self.mir_min = mir_min
        self.mir_max = mir_max

    def __call__(self, n: int) -> None:
        self.epochs = n
        # Train for a specified number of epochs
        self._on_start()
        for _ in range(n):
            self._on_start_epoch()

            timestamp = time.time()
            for batch in self._train_loader:
                if batch is None or batch[0] is None:
                    print(f"=> Rank {self._rank}: Skipping empty batch.")
                    continue
                self._state.inputs = batch[0]  # images
                self._state.targets = batch[1] # targets
                self._state.batch_indices = batch[2] # indices (全局图片索引)
                self._on_start_iteration()
                self._state.t_data.append(time.time() - timestamp)


                self._on_each_iteration()
                if self._state.loss is not None:
                    self._state.running_loss.append(self._state.loss.item())
                self._on_end_iteration()
                self._state.t_iteration.append(time.time() - timestamp)
                timestamp = time.time()

            self._on_end_epoch()
        self._on_end()

    def _on_start(self):
        if self._train_loader.dataset.name == "hicodet":
            ap = self.test_hico()
            trained_classes = self.filter_classes  # 仅评估这些类别
            if self._rank == 0:
                 # 获取全量 rare/non_rare 类别索引
                rare_all = get_base_dataset(self.test_dataloader.dataset).rare
                non_rare_all = get_base_dataset(self.test_dataloader.dataset).non_rare
                # 只保留当前task中的 rare/non_rare 类别
                rare = [i for i, c in enumerate(trained_classes) if c in rare_all]
                non_rare = [i for i, c in enumerate(trained_classes) if c in non_rare_all]
                perf = [ap.mean().item(),
                        ap[rare].mean().item() if rare else 0,
                        ap[non_rare].mean().item() if non_rare else 0]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
                )
                self.best_perf = perf[0]
                wandb.init(config=self.config)
                wandb.watch(self._state.net.module)
                wandb.define_metric("epochs")
                wandb.define_metric("mAP full", step_metric="epochs", summary="max")
                wandb.define_metric("mAP rare", step_metric="epochs", summary="max")
                wandb.define_metric("mAP non_rare", step_metric="epochs", summary="max")

                wandb.define_metric("training_steps")
                wandb.define_metric("elapsed_time", step_metric="training_steps", summary="max")
                wandb.define_metric("loss", step_metric="training_steps", summary="min")

                wandb.log({
                    "epochs": self._state.epoch, "mAP full": perf[0],
                    "mAP rare": perf[1], "mAP non_rare": perf[2]
                })
        else:
            ap = self.test_vcoco()
            if self._rank == 0:
                perf = [ap.mean().item(),]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}."
                )
                self.best_perf = perf[0]
                """
                NOTE wandb was not setup for V-COCO as the dataset was only used for evaluation
                """
                wandb.init(config=self.config)

    def _on_end(self):
        if self._rank == 0:
            wandb.finish()

    def _on_each_iteration(self):
        torch.autograd.set_detect_anomaly(True)
            # 跳过无效batch
        if self._state.inputs is None or self._state.targets is None or len(self._state.targets) == 0:
            print("无有效HOI标注，跳过该batch")
            return
            # 一次性forward

        # Attention Hint相关参数
        use_attn_hint = getattr(self.config, 'use_attn_hint', False)
        attn_hint_alpha = getattr(self.config, 'attn_hint_alpha', 0.05)
        attn_hint_epochs = getattr(self.config, 'attn_hint_epochs', 0)
        cur_epoch = getattr(self._state, 'epoch', 0)

        teacher_cross_attn_hint = None
        if use_attn_hint and self.teacher_model is not None and (attn_hint_epochs == 0 or cur_epoch < attn_hint_epochs):
            with torch.no_grad():
                _, teacher_cross_attn = self.teacher_model(
                    self._state.inputs, targets=self._state.targets, return_cross_attn=True
                )
            teacher_cross_attn_hint = teacher_cross_attn

        # 1. 判断 Batch 中各个样本的身份 (新任务 vs 回放任务)
        batch_indices = self._state.batch_indices
        is_replay_list = [idx in self.replay_indices for idx in batch_indices]

        # 2. 获取脆弱性标签 SIS (模块1的功能，这里做简化示例)
        # 实际中你应该根据 self.mir_dict 或其他指标实时计算 SIS
        sis_scores = []
        for img_idx, is_replay in zip(batch_indices, is_replay_list):
            if is_replay:
                # 假设通过 mir_dict 获取 SIS，数值在 0~1 之间，越大越脆弱
                raw_sis = float(self.mir_dict.get(str(img_idx), 0.5))
                sis_scores.append(raw_sis)
            else:
                sis_scores.append(0.0) # 新样本无需排斥，SIS 为 0

        # 3. 将 VAS 控制信号传入模型前向传播
        use_vas = getattr(self.config, 'use_vas', False)
        vas_lambda = getattr(self.config, 'vas_lambda', 0.0)

        # 将hint参数传递给学生模型
        outputs = self._state.net(
            self._state.inputs,
            targets=self._state.targets,
            return_outputs=True,
            # 仅当 use_attn_hint 开启时传入 hint（即便 teacher_cross_attn_hint 可能为 None）
            teacher_cross_attn_hint=teacher_cross_attn_hint if use_attn_hint else None,
            # 仅在真正有 teacher_hint 时传 alpha（否则传 None，模型按自身设置或禁用）
            attn_hint_alpha=attn_hint_alpha if (use_attn_hint and teacher_cross_attn_hint is not None) else None,
            # === 新增 VAS 参数 ===
            use_vas=use_vas,
            is_replay_list=is_replay_list,
            sis_scores=sis_scores,
            vas_lambda=vas_lambda
        )
        # outputs = self._state.net(self._state.inputs, targets=self._state.targets, return_outputs=True)

        # device = torch.device('cuda', torch.cuda.current_device())  # 或 local_rank
        # outputs = dict_to_device(outputs, device)
        loss_dict = {'cls_loss': outputs['cls_loss']}

        # ==================== 1. 获取教师模型的全局输出 ====================
        standard_distill_loss = 0.
        teacher_outputs = None
        if self.config.use_distill and self.teacher_model is not None:
            with torch.no_grad():
                self.teacher_model.train()
                # 教师模型直接对当前batch的所有图片做前向传播
                teacher_outputs = self.teacher_model(self._state.inputs, targets=self._state.targets, return_outputs=True)

            if 'cls_loss' in outputs and 'cls_loss' in teacher_outputs:
                standard_distill_loss += F.mse_loss(outputs['cls_loss'], teacher_outputs['cls_loss'])
            standard_distill_loss *= self.config.distill_loss_weight

        # ==================== 2. 全样本特征与 Logits 蒸馏 ====================
        # 去除了仅针对回放样本的限制，利用成对编码对 batch 内所有 common pairs 蒸馏
        feature_distill_loss = 0.
        if self.config.use_distill and self.config.replay_distill and teacher_outputs is not None:
            images, targets, batch_indices = self._state.inputs, self._state.targets, self._state.batch_indices
            key = self.config.replay_distill_layer

            if batch_indices is not None and key in ['pred_logits', 'feat']:
                # 筛 pair 时，获取全局唯一标识 (Global Image Index, Pair Index in Image)
                student_pair_image_indices = outputs['pair_image_indices'].tolist()
                student_pair_idx_in_image = outputs['pair_idx_in_image'].tolist()
                student_pair_global_indices = [batch_indices[i] for i in student_pair_image_indices]
                student_pair_ids = [(img_idx, pidx) for img_idx, pidx in zip(student_pair_global_indices, student_pair_idx_in_image)]

                teacher_pair_image_indices = teacher_outputs['pair_image_indices'].tolist()
                teacher_pair_idx_in_image = teacher_outputs['pair_idx_in_image'].tolist()
                teacher_pair_global_indices = [batch_indices[i] for i in teacher_pair_image_indices]
                teacher_pair_ids = [(img_idx, pidx) for img_idx, pidx in zip(teacher_pair_global_indices, teacher_pair_idx_in_image)]

                # 构造 pair id 到索引的映射字典
                student_id2idx = {pid: i for i, pid in enumerate(student_pair_ids)}
                teacher_id2idx = {pid: i for i, pid in enumerate(teacher_pair_ids)}

                # 严格有序交集（提取教师和学生都成功预测的有效配对）
                common_ids = sorted(set(student_pair_ids) & set(teacher_pair_ids))

                # 一一对应索引
                student_indices = [student_id2idx[pid] for pid in common_ids]
                teacher_indices = [teacher_id2idx[pid] for pid in common_ids]

                # ============ 关键鲁棒保护 ============
                skip_KD = False
                if outputs['pred_logits'].shape[0] == 0 or teacher_outputs['pred_logits'].shape[0] == 0:
                    print("[Feature Distill] Warning: empty pred_logits, skipping KD.")
                    skip_KD = True
                if len(student_indices) == 0 or len(teacher_indices) == 0:
                    print("[Feature Distill] Warning: empty distill indices, skipping KD.")
                    skip_KD = True

                if not skip_KD:
                    student_feat = outputs['feat'][student_indices]
                    student_logits = outputs['pred_logits'][-1][student_indices]
                    teacher_feat = teacher_outputs['feat'][teacher_indices]
                    teacher_logits = teacher_outputs['pred_logits'][-1][teacher_indices]

                    if student_feat.shape != teacher_feat.shape or student_logits.shape != teacher_logits.shape:
                        print(f"Shape mismatch in feature distill: student {student_feat.shape}, teacher {teacher_feat.shape}. Skipping feature distill.")
                        feature_distill_loss = 0.0
                    else:
                        # 所有 pair 的类别标签
                        all_pair_labels = [
                            targets[student_pair_image_indices[i]]['labels'][0].item()
                            if targets[student_pair_image_indices[i]]['labels'].numel() > 0 else -1
                            for i in range(len(student_pair_image_indices))
                        ]

                        # 使用 student_indices 对齐标签，准备计算每个pair的权重
                        student_pair_labels = [all_pair_labels[i] for i in student_indices]

                        # 权重计算 (依然保留基于 MIR 和 rare 的类别级加权保护)
                        weights = []
                        for cls in student_pair_labels:
                            mir_score = float(self.mir_dict.get(str(cls), 0.0))
                            mir_score = (mir_score - self.mir_min) / (self.mir_max - self.mir_min + 1e-6) if self.mir_max > self.mir_min else 0.0
                            is_rare = 1 if cls in self.rare_set else 0
                            w = self.config.replay_distill_loss_weight * (1 + self.config.replay_distill_mir_factor * mir_score + self.config.replay_distill_rare_factor * is_rare)
                            weights.append(w)
                        weights = torch.tensor(weights, dtype=student_feat.dtype, device=student_feat.device)

                        # 计算特征蒸馏损失
                        feat_distill = F.mse_loss(student_feat, teacher_feat, reduction='none').mean(dim=1)
                        final_feat_loss = (feat_distill * weights).mean()

                        # 计算 Logit 蒸馏损失
                        logits_distill = F.mse_loss(student_logits, teacher_logits, reduction='none').mean(dim=1)
                        final_logits_loss = (logits_distill * weights).mean()

                        if key == 'pred_logits':
                            feature_distill_loss = final_feat_loss + 0.5 * final_logits_loss
                        elif key == 'feat':
                            feature_distill_loss = final_feat_loss
                        else:
                            raise ValueError("Unknown distill_layer " + key)

        # ==================== 3. 汇总总损失 ====================
        total_loss = sum(loss for loss in loss_dict.values()) + standard_distill_loss + feature_distill_loss

        if loss_dict['cls_loss'].isnan() or loss_dict['cls_loss'].isinf() or total_loss.isnan() or total_loss.isinf():
            # print("当前输入数据:", self._state.inputs)
            # print("当前标签:", self._state.targets)
            print("当前损失:", loss_dict)
            if total_loss.isnan() or total_loss.isinf():
                print("standard_distill_loss:", standard_distill_loss)
                print("replay_distill_loss:", replay_distill_loss)
            self._state.loss = None  # 明确标记无效
            return
            #raise ValueError(f"The HOI loss is NaN or Inf for rank {self._rank}")

        self._state.loss = total_loss
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
                self._state.epoch, self.epochs,
                str(self._state.iteration - num_iter * (self._state.epoch - 1)).zfill(n_d),
                num_iter, running_loss, t_data, t_iter
            ))
            # wandb.log({
            #     "elapsed_time": (time.time() - self._dawn) / 3600,
            #     "training_steps": self._state.iteration,
            #     "loss": running_loss
            # })
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_end_epoch(self):
        if self._train_loader.dataset.name == "hicodet":
            ap = self.test_hico()
            trained_classes = self.filter_classes  # 仅评估这些类别
            if self._rank == 0:
                 # 获取全量 rare/non_rare 类别索引
                rare_all = get_base_dataset(self.test_dataloader.dataset).rare
                non_rare_all = get_base_dataset(self.test_dataloader.dataset).non_rare
                # 只保留当前task中的 rare/non_rare 类别
                rare = [i for i, c in enumerate(trained_classes) if c in rare_all]
                non_rare = [i for i, c in enumerate(trained_classes) if c in non_rare_all]
                perf = [ap.mean().item(),
                        ap[rare].mean().item() if rare else 0,
                        ap[non_rare].mean().item() if non_rare else 0]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
                )
                # wandb.log({
                #     "epochs": self._state.epoch, "mAP full": perf[0],
                #     "mAP rare": perf[1], "mAP non_rare": perf[2]
                # })
        else:
            ap = self.test_vcoco()
            if self._rank == 0:
                perf = [ap.mean().item(),]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}."
                )
                """
                NOTE wandb was not setup for V-COCO as the dataset was only used for evaluation
                """

        if self._rank == 0:
            # Save checkpoints
            checkpoint = {
                'iteration': self._state.iteration,
                'epoch': self._state.epoch,
                'performance': perf,
                'model_state_dict': self._state.net.module.state_dict(),
                'optim_state_dict': self._state.optimizer.state_dict(),
                'scaler_state_dict': self._state.scaler.state_dict()
            }
            if self._state.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
            torch.save(checkpoint, os.path.join(self._cache_dir, "latest.pth"))
            if perf[0] > self.best_perf:
                self.best_perf = perf[0]
                torch.save(checkpoint, os.path.join(self._cache_dir, "best.pth"))
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()


    @torch.no_grad()
    def test_hico(self, return_per_sample_scores=False):
        dataloader = self.test_dataloader
        net = self._state.net; net.eval()

        dataset = get_base_dataset(dataloader.dataset)
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        # ...existing code...
        trained_classes = self.filter_classes  # 仅评估这些类别
        if trained_classes is None:
            trained_classes = list(range(600))  # 默认全量
        print(f"Evaluating on {len(trained_classes)} classes")

        if self._rank == 0:
            meter = DetectionAPMeter(
                600, nproc=1, algorithm='11P',
                num_gt=dataset.anno_interaction,
            )

        per_sample_scores = []  # 新增
        print("testing hico...")
        for batch_idx, batch in enumerate(dataloader):
            if batch is None or batch[0] is None:
                print("[def test_hico]:无有效HOI标注，跳过该batch")
                continue
            images, targets, batch_indices = batch
            images = pocket.ops.relocate_to_cuda(images)
            outputs = net(images)
            outputs = pocket.ops.relocate_to_cpu(outputs, ignore=True)

            scores_clt = []; preds_clt = []; labels_clt = []
            for i, (output, target) in enumerate(zip(outputs, targets)):
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
                scores = output['scores']
                verbs = output['labels']
                objects = output['objects']
                interactions = conversion[objects, verbs]
                # Recover target box scale
                gt_bx_h = recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = recover_boxes(target['boxes_o'], target['size'])

                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()
                for hoi_idx in unique_hoi:
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )

                scores_clt.append(scores)
                preds_clt.append(interactions)
                labels_clt.append(labels)
                # ... 原有评估流程 ...
                # 记录per-sample分数
                # 这里以GT类别的预测分数为例（假设sigmoid分数，或你可以改为softmax）
                gt_classes = target['labels']

                # 假设有多query，取最大分数
                sample_scores = {}
                for gt in gt_classes:
                    gt = int(gt.item())
                    # 取该gt类别的分数（假设output['scores'] shape为[num_queries]）
                    # 你也可以改为output['pred_logits'][..., gt]等
                    if 'pred_logits' in output:
                        prob = torch.sigmoid(output['pred_logits'][..., gt]).max().item()
                    else:
                        prob = scores.max().item()
                    sample_scores[gt] = prob
                per_sample_scores.append({
                    'local_idx': batch_idx * dataloader.batch_size + i,
                    'gt_classes': [int(g.item()) for g in gt_classes],
                    'scores': sample_scores
                })
            # Collate results into one tensor
            scores_clt = torch.cat(scores_clt)
            preds_clt = torch.cat(preds_clt)
            labels_clt = torch.cat(labels_clt)

            # Gather data from all processes
            scores_ddp = pocket.utils.all_gather(scores_clt)
            preds_ddp = pocket.utils.all_gather(preds_clt)
            labels_ddp = pocket.utils.all_gather(labels_clt)

            if self._rank == 0:
                meter.append(torch.cat(scores_ddp), torch.cat(preds_ddp), torch.cat(labels_ddp))

        if return_per_sample_scores:
            if self._rank == 0:
                ap = meter.eval()
                ap_trained = ap[trained_classes]
            else:
                ap_trained = -1
            return ap_trained, per_sample_scores

        if self._rank == 0:
            ap = meter.eval()
            ap_trained = ap[trained_classes]
            return ap_trained
        else:
            return -1

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, (image, target) in enumerate(tqdm(dataloader.dataset)):
            inputs = pocket.ops.relocate_to_cuda([image,])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num

        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def test_vcoco(self):
        dataloader = self.test_dataloader
        net = self._state.net; net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)

        if self._rank == 0:
            meter = DetectionAPMeter(
                24, nproc=1, algorithm='11P',
                num_gt=dataset.num_instances,
            )
        for batch in tqdm(dataloader, disable=(self._world_size != 1)):
            inputs = pocket.ops.relocate_to_cuda(batch[:-1])
            outputs = net(*inputs)
            outputs = pocket.ops.relocate_to_cpu(outputs, ignore=True)
            targets = batch[-1]

            scores_clt = []; preds_clt = []; labels_clt = []
            for output, target in zip(outputs, targets):
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
                scores = output['scores']
                actions = output['labels']
                gt_bx_h = recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = recover_boxes(target['boxes_o'], target['size'])

                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_actions = actions.unique()
                for act_idx in unique_actions:
                    gt_idx = torch.nonzero(target['actions'] == act_idx).squeeze(1)
                    det_idx = torch.nonzero(actions == act_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )

                scores_clt.append(scores)
                preds_clt.append(actions)
                labels_clt.append(labels)
            # Collate results into one tensor
            scores_clt = torch.cat(scores_clt)
            preds_clt = torch.cat(preds_clt)
            labels_clt = torch.cat(labels_clt)
            # Gather data from all processes
            scores_ddp = pocket.utils.all_gather(scores_clt)
            preds_ddp = pocket.utils.all_gather(preds_clt)
            labels_ddp = pocket.utils.all_gather(labels_clt)

            if self._rank == 0:
                meter.append(torch.cat(scores_ddp), torch.cat(preds_ddp), torch.cat(labels_ddp))

        if self._rank == 0:
            ap = meter.eval()
            return ap
        else:
            return -1

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, (image, target) in enumerate(tqdm(dataloader.dataset)):
            inputs = pocket.ops.relocate_to_cuda([image,])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)