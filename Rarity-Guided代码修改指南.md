# Rarity-Guided Interaction Preservation 代码修改指南

本文档用于指导服务器端另一个 Codex agent 修改 `hoi-pvic` 代码，使代码实现与最新论文方案 **Rarity-Guided Interaction Preservation for Class-Incremental Human--Object Interaction Detection** 保持一致。

本指南只描述代码修改方案，不要求修改论文正文。实现时请优先保证方法闭环和实验可运行，再逐步增加可视化或额外分析。

---

## 0. 目标和边界

最新论文方法的代码侧目标不是“给现有 replay/KD 加几个权重”，而是实现一个围绕稀有交互遗忘风险展开的训练框架：

1. **Rarity-Guided Sample Evaluation and Replay**
   - 使用冻结教师模型在回放样本进入训练前评估旧交互 pair 的遗忘风险。
   - 得到 Sample Interaction Susceptibility，简称 `SIS`。
   - 用 `SIS` 对旧交互 replay pair 的监督损失和蒸馏损失加权。

2. **Predicate-Interference Attention Modulation**
   - 维护当前任务新 HOI 类的轻量原型。
   - 对旧 replay pair 选择最可能造成混淆的新类 confounder。
   - 在 decoder cross-attention 中只调节 content evidence 对应的注意力 logits，不改变 position logits。

3. **Rarity- and Attention-Guided Interaction Distillation**
   - 在 replay old pair 上进行 pair feature、old predicate distribution 和 hard new-predicate negative margin 蒸馏。
   - 蒸馏权重由 `SIS` 和 attention/confounder 权重共同决定。

代码实现应满足：

- 训练阶段使用 RGIP，推理阶段不增加额外模块。
- replay 样本可以继续沿用当前数据混合方式，但 replay pair 的训练权重要改为 `SIS` 引导。
- 对 object-disjoint 的 New Concept protocol 不能失效：confounder 选择不能只依赖 shared object。
- 现有 `DCA-HOI`、`attention hint`、普通 loss-MSE 蒸馏等旧机制不能作为最终主方法。

---

## 1. 当前代码现状和必须修正的问题

当前主要代码位于 `hoi-pvic/`。

### 1.1 `pvic.py`

关键位置：

- `PViC.compute_classification_loss`：当前只接收 `logits, prior, labels`，不能对 replay high-SIS pair 加权。
- `PViC.forward`：当前返回 `cls_loss, pred_logits, feat, pair_image_indices, pair_idx_in_image`，缺少 RGIP 所需的 pair metadata。
- `teacher_cross_attn_hint`：当前只是把 `cross_attn_weights` 返回值做线性融合，但 decoder 输出 `out` 已经计算完成，因此这个 hint 不会影响预测和 loss，不能作为论文中的 attention modulation。

必须修改：

- `compute_classification_loss` 增加 `pair_weights=None`。
- `forward` 增加 `batch_indices=None, rgip_context=None, return_rgip_meta=False`。
- `forward` 输出增加：
  - `pair_labels`
  - `pair_objects`
  - `pair_prior`
  - `pair_global_indices`
  - `paired_inds`
  - 可选 `rgip_sis`
  - 可选 `rgip_confounder_ids`
  - 可选 `rgip_confounder_weights`

### 1.2 `transformers.py`

关键位置：

- `TransformerDecoderLayer.forward` 当前将 content query/key 和 positional query/key 拼接后整体送入 `MultiheadAttention`。
- 论文方法要求区分 content logits 与 position logits，并只对 content route 加负向调制。

必须修改：

- 不要继续依赖 `teacher_cross_attn_hint`。
- 给 decoder 增加 `rgip_attn_context`。
- 在 cross-attention 中显式计算：
  - `E_content`
  - `E_position`
  - `E_total = E_content + E_position + content_bias`
- `content_bias` 由 confounder prototype、SIS 和 positional support 产生。

### 1.3 `utils_incremental.py`

关键位置：

- `_on_each_iteration` 当前流程是：
  1. 可选 teacher attention hint。
  2. student forward。
  3. 普通 distill：`F.mse_loss(outputs['cls_loss'], teacher_outputs['cls_loss'])`。
  4. replay distill：feature/logit MSE，权重来自 `rare_set` 和 `mir_dict`。

必须修改：

- `cls_loss` 之间做 MSE 没有意义，应删除或在 `--use-rgip` 时禁用。
- replay distill 应替换为 RGIP interaction distillation。
- teacher forward 必须 `eval()`，不能在 no-grad 中调用 `train()`。
- RGIP 下训练循环顺序应变为：
  1. 找出本 batch 中 replay images。
  2. teacher 对 replay images 前向，返回 pair metadata。
  3. 计算 SIS、confounders 和 per-pair weights。
  4. 构造 `rgip_context` 传给 student forward。
  5. student 前向计算 weighted classification loss。
  6. student/teacher replay pair 对齐，计算 RGIP interaction distillation。
  7. 用当前任务正样本更新 prototype bank。

### 1.4 `main_incremental.py`

关键位置：

- 当前 teacher 只在 rank 0 加载，其他 rank 为 `None`。如果使用 DDP，这会导致非 rank0 不执行 teacher 相关 RGIP loss，训练不一致。
- replay 样本当前由 `get_samples_by_class` 从旧类抽取，可作为最低成本 replay 入口继续使用。

必须修改：

- 每个 rank 都加载 teacher model，并设置：
  - `teacher_model.eval()`
  - `requires_grad_(False)`
  - 移动到当前 rank 对应 GPU。
- 增加 RGIP 相关命令行参数。
- 将 old/current task classes、HOI-object-verb 映射、object-compatible predicate 映射传入 engine 或 RGIP state。

---

## 2. 新增文件建议

建议新增一个文件：

```text
hoi-pvic/rgip_utils.py
```

该文件集中放置 RGIP 状态和工具函数，避免把所有逻辑塞进 `utils_incremental.py`。

建议包含以下类和函数：

```python
class RarityGuidedState:
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
    ):
        ...

    def compute_sis_for_teacher_pairs(self, teacher_output):
        ...

    def build_replay_pair_context(self, teacher_output, selected_global_indices):
        ...

    def build_student_batch_context(self, replay_context, batch_indices):
        ...

    def compute_interaction_distillation(self, student_output, teacher_output, replay_context):
        ...

    def update_current_task_prototypes(self, student_output):
        ...
```

也可以拆成更小的函数，但需要保证调用关系清晰。

---

## 3. 命令行参数建议

在 `main_incremental.py` 的 argparse 中新增：

```python
parser.add_argument('--use-rgip', action='store_true',
                    help='Enable Rarity-Guided Interaction Preservation')

parser.add_argument('--rgip-alpha', default=0.4, type=float,
                    help='SIS frequency-risk weight')
parser.add_argument('--rgip-beta', default=0.3, type=float,
                    help='SIS margin-risk weight')
parser.add_argument('--rgip-gamma', default=2.0, type=float,
                    help='SIS-to-training-weight strength')

parser.add_argument('--rgip-topk', default=3, type=int,
                    help='Number of current-task confounders per replay pair')
parser.add_argument('--rgip-eta-object', default=0.4, type=float,
                    help='Confounder score weight for shared object')
parser.add_argument('--rgip-eta-query', default=0.4, type=float,
                    help='Confounder score weight for query/prototype similarity')
parser.add_argument('--rgip-eta-sem', default=0.2, type=float,
                    help='Confounder score weight for predicate semantic similarity')
parser.add_argument('--rgip-confounder-tau', default=0.2, type=float,
                    help='Temperature for confounder softmax weights')

parser.add_argument('--rgip-lambda-attn', default=0.5, type=float,
                    help='Strength of content-attention modulation')
parser.add_argument('--rgip-kappa', default=1.0, type=float,
                    help='Exponent for positional support in attention modulation')
parser.add_argument('--rgip-attn-clamp', default=5.0, type=float,
                    help='Clamp magnitude for negative attention bias')

parser.add_argument('--rgip-proto-momentum', default=0.9, type=float,
                    help='EMA momentum for current-task interaction prototypes')

parser.add_argument('--rgip-int-loss-weight', default=1.0, type=float,
                    help='Overall weight for RGIP interaction distillation')
parser.add_argument('--rgip-feat-weight', default=1.0, type=float,
                    help='Feature distillation weight')
parser.add_argument('--rgip-logit-weight', default=1.0, type=float,
                    help='Old predicate distribution distillation weight')
parser.add_argument('--rgip-hardneg-weight', default=0.5, type=float,
                    help='Hard new-predicate negative margin weight')
parser.add_argument('--rgip-margin', default=0.2, type=float,
                    help='Margin for hard negative suppression')
parser.add_argument('--rgip-temperature', default=2.0, type=float,
                    help='Temperature for predicate distribution KD')

parser.add_argument('--rgip-use-semantic-confounder', action='store_true',
                    help='Use predicate semantic similarity when embeddings are available')
parser.add_argument('--rgip-predicate-embedding-path', default='', type=str,
                    help='Optional path to predicate embeddings')

parser.add_argument('--rgip-debug', action='store_true',
                    help='Log SIS, confounder and attention statistics')
```

建议保留旧参数用于 baseline，但当 `--use-rgip` 开启时：

- 禁用 `use_attn_hint` 的旧逻辑。
- 禁用 `standard_distill_loss = MSE(cls_loss)`。
- 用 RGIP interaction distillation 替代旧 `replay_distill`。

---

## 4. 数据结构和映射关系

当前代码需要特别注意一个问题：模型分类头输出的是 predicate/verb logits，而不是 600 维 HOI logits。

具体表现：

- `args.num_verbs = 117`
- `binary_classifier = nn.Linear(repr_size, num_verbs)`
- `associate_with_ground_truth` 返回 `[num_pairs, num_verbs]` 多标签矩阵。
- evaluation 通过 `object_n_verb_to_interaction[object, verb]` 将 object+verb 转换为 HOI id。

因此 RGIP 里所有 HOI 类操作都要显式转换：

```text
HOI class c -> object id o_c + verb id v_c
pair i -> detected object id o_i + positive verb(s) v_i
pair i + verb v -> HOI id c_i = obj_verb_to_hoi[o_i, v]
```

建议在 `main_incremental.py` 读取 `hoi_correspondence.json` 后构造：

```python
hoi_to_obj = {}
hoi_to_verb = {}
obj_verb_to_hoi = {}

for hoi_id, obj_id, verb_id in correspondence:
    hoi_to_obj[int(hoi_id)] = int(obj_id)
    hoi_to_verb[int(hoi_id)] = int(verb_id)
    obj_verb_to_hoi[(int(obj_id), int(verb_id))] = int(hoi_id)
```

如果服务器代码中 `object_types` 使用 COCO object id，而 `hoi_correspondence.json` 使用 HICO object id，需要先确认数据集已有的映射。不要直接混用两套 object id。当前 evaluation 使用 `dataset.object_n_verb_to_interaction[objects, verbs]`，因此最安全做法是复用这个矩阵进行 object+verb 到 HOI 的映射。

---

## 5. Module 1：SIS 和 rarity-guided replay

### 5.1 SIS 定义

代码中建议用下面三个风险量：

```text
s_i = alpha * r_freq + beta * r_margin + (1 - alpha - beta) * r_entropy
w_i = (1 + gamma * s_i) / mean_j(1 + gamma * s_j)
```

其中：

- `r_freq`：HOI 类样本数量越少，风险越大。
- `r_margin`：教师模型对旧正类和最强混淆类的 logit margin 越小，风险越大。
- `r_entropy`：教师模型在 object-compatible predicate 集合上的分布越不确定，风险越大。
- `w_i`：batch 内归一化后的 replay pair 权重。

### 5.2 频次风险 `r_freq`

建议在每个任务开始时统计训练集或旧任务样本中的 HOI frequency。

```python
freq[c] = number_of_training_instances_for_hoi_c
r_freq[c] = 1.0 - normalize(log(1 + freq[c]))
```

最低成本实现：

- 如果已有 `rare.json`，可先将 rare 类设为 `1.0`，non-rare 类设为 `0.0`。
- 更推荐统计真实频次，这样比二值 rare/non-rare 更适合论文中的 “rarity-guided”。

### 5.3 margin 风险 `r_margin`

对 replay pair `i`，先找到它的正 old verb 或 old HOI：

```python
positive_verbs = where(pair_labels[i] == 1)
old_positive_verbs = verbs whose (object_i, verb) belongs to old_hoi_classes
```

如果多个 old positive verbs，建议选择风险最大的一个，或取平均。为了更强调 vulnerable sample，建议默认用最大风险：

```python
z_pos = teacher_logits[i, v_pos]
z_neg = max teacher_logits[i, v] for v in compatible_old_or_seen_verbs and v != v_pos
margin = z_pos - z_neg
r_margin = sigmoid(-margin)
```

注意：

- `compatible_old_or_seen_verbs` 应该由 pair object 的有效 predicate 集合和 old task classes 共同确定。
- 如果集合为空，`r_margin = 0`。

### 5.4 entropy 风险 `r_entropy`

在 object-compatible predicate 集合上计算教师分布熵：

```python
p = softmax(teacher_logits[i, compatible_verbs] / T)
r_entropy = entropy(p) / log(len(compatible_verbs))
```

如果 `len(compatible_verbs) <= 1`，设为 0。

### 5.5 pair 权重如何接入分类损失

修改 `PViC.compute_classification_loss`：

```python
def compute_classification_loss(self, logits, prior, labels, pair_weights=None):
    prior = torch.cat(prior, dim=0).prod(1)   # [num_pairs, num_verbs]
    x, y = torch.nonzero(prior).unbind(1)

    logits = logits[:, x, y]
    labels = labels[None, x, y].repeat(len(logits), 1)

    elem_loss = binary_focal_loss_with_logits(..., reduction='none')
    # elem_loss shape should align with [num_layers, num_valid_entries]

    if pair_weights is not None:
        valid_weights = pair_weights[x]       # [num_valid_entries]
        elem_loss = elem_loss * valid_weights[None, :]

    # denominator should remain stable
    denom = (labels * (valid_weights[None, :] if pair_weights is not None else 1)).sum()
    denom = denom.clamp(min=1.0)
    return elem_loss.sum() / denom
```

如果当前 `binary_focal_loss_with_logits` 不支持 `reduction='none'`，需要检查其实现并增加 none reduction。不要用 `reduction='sum'` 后再乘权重，因为此时已经丢失 pair 粒度。

### 5.6 replay 采样是否必须改

最低成本方案：

- 保留现有 `get_samples_by_class` 选 replay images。
- 用 `SIS` 对 replay pair loss 加权。

增强方案：

- 在若干 epoch 后缓存每个 replay image 的最大/平均 `SIS`。
- 使用 `WeightedRandomSampler` 或重新排序，让 high-SIS replay images 更频繁出现。

建议先实现最低成本方案，保证论文核心实验能跑通。若时间允许，再加 risk-biased sampling 作为 ablation。

---

## 6. Module 2：Predicate-Interference Attention Modulation

### 6.1 模块目标

该模块不是普通 attention distillation，也不是 teacher attention blending。

目标是：

- 对 replay old pair 找到当前任务中最容易干扰它的新 HOI 类。
- 将新类 prototype 投影到旧图像的 content keys 上，估计新谓词可能抢占的 visual evidence。
- 对旧 pair 的 content-attention logits 加负向 bias，降低其被新谓词干扰区域牵引的程度。
- position logits 不改，用作 human-object geometry anchor。

### 6.2 当前任务 prototype bank

建议在 `RarityGuidedState` 中维护：

```python
self.prototypes: Dict[int, Tensor]  # hoi_id -> [repr_dim]
self.prototype_counts: Dict[int, int]
```

更新规则：

```python
proto[c] = momentum * proto[c] + (1 - momentum) * mean(normalized_pair_features_for_c)
proto[c] = normalize(proto[c])
```

prototype 来源：

- 只使用当前任务 new classes 的正样本。
- 不使用 replay old classes 更新当前任务 prototype。
- pair feature 建议先用 `outputs['feat']`，即当前代码已有的 pair representation。

如何把 pair 映射到 HOI id：

```python
for each positive verb v in pair_labels[i]:
    c = obj_verb_to_hoi[(pair_object[i], v)]
    if c in current_hoi_classes:
        update prototype[c] with feat[i]
```

如果一个 pair 有多个 current positives，可分别更新对应 prototype。

DDP 注意：

- 最低成本：每个 rank 维护本地 prototype bank。
- 更稳定：每 N 次 iteration all-reduce batch prototype sum/count，再统一 EMA 更新。论文实验建议采用同步版本，避免 rank 间行为差异。

### 6.3 hard confounder 选择

对 replay old pair `i` 和当前任务 prototype `c`，计算：

```text
h_ic = eta_object * same_object(o_i, o_c)
     + eta_query  * cosine(q_i, p_c)
     + eta_sem    * cosine(e_v_i, e_v_c)
```

其中：

- `o_i`：replay pair 的 object。
- `o_c`：当前任务 HOI class c 的 object。
- `q_i`：replay old pair 的 feature，建议用 teacher feature。
- `p_c`：当前任务 prototype。
- `e_v_i, e_v_c`：predicate semantic embedding。

然后：

```python
confounder_ids = topk current classes by h_ic
confounder_weights = softmax(h_ic / tau)
```

重要：不能只用 `same_object`。在 New Concept object-disjoint protocol 中，不同任务间可能没有 shared object。如果只依赖 shared object，模块会失效。

因此默认推荐：

```text
eta_object = 0.4
eta_query  = 0.4
eta_sem    = 0.2
```

当 object-disjoint 时，`same_object=0`，但 `query similarity` 和 `semantic similarity` 仍然可工作。

语义 embedding 的最低成本实现：

1. 优先支持从 `--rgip-predicate-embedding-path` 读取 `[num_verbs, dim]` tensor。
2. 如果没有 embedding：
   - `--rgip-use-semantic-confounder` 未开启时，设 `eta_sem=0` 并把剩余权重归一化。
   - 或使用 `binary_classifier.weight` 的 cosine 作为 model-induced predicate similarity，但这要在文档/实验里说明。

### 6.4 attention modulation 应该施加在哪些样本上

只对 replay old positive pairs 施加。

不要对以下样本施加：

- 当前任务 new samples。
- replay 图像中没有 old positive label 的 pair。
- SIS 无效或没有 confounder prototype 的 pair。

这样可以降低伤害新类学习的风险，实验上更可能提升 class-incremental old/rare mAP。

### 6.5 修改 decoder cross-attention

当前 `TransformerDecoderLayer.forward` 的 cross attention 大致是：

```python
q = qk_attn_q_proj(queries)
k = qk_attn_k_proj(features)
v = qk_attn_v_proj(features)
q_p = qk_attn_qpos_proj(q_pos["centre"])
k_p = qk_attn_kpos_proj(k_pos)
q = cat([q, q_p])
k = cat([k, k_p])
qk_attn, weights = self.qk_attn(query=q, key=k, value=v, ...)
```

建议改成显式版本：

```python
q_c = self.qk_attn_q_proj(queries)      # [nq, bs, q_dim]
k_c = self.qk_attn_k_proj(features)     # [hw, bs, q_dim]
v   = self.qk_attn_v_proj(features)     # [hw, bs, q_dim]

q_p = self.qk_attn_qpos_proj(q_pos["centre"])
k_p = self.qk_attn_kpos_proj(k_pos)

q_c_h = reshape_to_heads(q_c)           # [bs*num_heads, nq, d_h]
k_c_h = reshape_to_heads(k_c)           # [bs*num_heads, hw, d_h]
q_p_h = reshape_to_heads(q_p)
k_p_h = reshape_to_heads(k_p)
v_h   = reshape_value_to_heads(v)

scale = 1.0 / sqrt(2 * d_h)
E_content  = bmm(q_c_h, k_c_h.transpose(1, 2)) * scale
E_position = bmm(q_p_h, k_p_h.transpose(1, 2)) * scale

content_bias = build_rgip_content_bias(
    rgip_attn_context,
    layer=self,
    q_content_projector=self.qk_attn_q_proj,
    k_content_heads=k_c_h,
    E_position=E_position,
)

E_total = E_content + E_position + content_bias
apply qk_attn_mask and kv_padding_mask to E_total
A = softmax(E_total - E_total.max(dim=-1, keepdim=True)[0], dim=-1)
A = dropout(A, p=self.qk_attn.dropout, training=self.training)
attn_output = bmm(A, v_h)
attn_output = merge_heads(attn_output)
attn_output = self.qk_attn.out_proj(attn_output)
```

注意保持原模型缩放一致。因为原代码把 content 和 position 拼接成 `2*d_h`，所以 split 后建议使用 `sqrt(2*d_h)` 作为 scale。

### 6.6 content bias 计算

对每个被调制的 replay pair：

```text
M_i = sum_c a_ic * softmax( projected_proto_c dot content_keys )
content_bias_i = -lambda_attn * SIS_i * Norm( M_i * softmax(E_position_i)^kappa )
```

代码建议：

```python
def build_content_bias(...):
    # output shape: [bs*num_heads, nq, hw]
    bias = torch.zeros_like(E_position)

    if rgip_attn_context is None:
        return bias

    sis = rgip_attn_context['sis']                         # [nq]
    mask = rgip_attn_context['modulate_mask']              # [nq]
    proto = rgip_attn_context['confounder_prototypes']     # [nq, K, q_dim]
    a = rgip_attn_context['confounder_weights']            # [nq, K]

    # For each query i with mask=True:
    # 1. project prototype by qk_attn_q_proj if prototype is in query feature space.
    # 2. reshape to heads.
    # 3. compute proto-key attention over current image content keys.
    # 4. weighted sum over K confounders.
    # 5. multiply by positional support.
    # 6. normalize and clamp.

    bias = bias.clamp(min=-attn_clamp, max=0.0)
    return bias.detach()
```

`Norm` 推荐实现：

```python
field = field / (field.mean(dim=-1, keepdim=True) + 1e-6)
field = field.clamp(max=5.0)
```

这样不会因为某些 attention map 极端值导致训练不稳定。

---

## 7. Module 3：Rarity- and Attention-Guided Interaction Distillation

### 7.1 替换当前蒸馏逻辑

当前代码有两类蒸馏：

1. `standard_distill_loss = MSE(student_cls_loss, teacher_cls_loss)`
2. replay feature/logit MSE

RGIP 下应改为：

```text
L_int = weighted sum over replay old positive pairs:
    feat_weight   * pair feature distillation
  + logit_weight  * old predicate distribution distillation
  + hardneg_weight * hard new-predicate margin
```

### 7.2 replay pair 对齐

可以复用当前 pair id 对齐思想：

```python
pair_id = (global_image_idx, pair_idx_in_image)
```

要求：

- student output 和 teacher output 都返回 `pair_global_indices` 与 `pair_idx_in_image`。
- teacher 只 forward replay images，global index 使用 `selected_indices`。
- student forward full batch，global index 使用 `batch_indices`。

对齐伪代码：

```python
student_id2idx = {pair_id: i for i, pair_id in enumerate(student_pair_ids)}
teacher_id2idx = {pair_id: i for i, pair_id in enumerate(teacher_pair_ids)}
common_ids = sorted(set(student_id2idx) & set(teacher_id2idx))

s_idx = [student_id2idx[x] for x in common_ids]
t_idx = [teacher_id2idx[x] for x in common_ids]
```

### 7.3 feature distillation

```python
feat_loss_i = mse(student_feat[i], teacher_feat[i].detach()).mean()
```

注意：

- `teacher_feat` 必须 detach。
- 只对 replay old positive pairs 计算。
- 如果没有 positive old label，则跳过。

### 7.4 old predicate distribution distillation

对每个 replay pair，构造 old predicate mask：

```python
old_predicate_mask_i[v] = True
if pair object o_i and verb v can map to an old HOI class
```

然后用 KL：

```python
T = args.rgip_temperature
t_prob = softmax(teacher_logits[i, mask] / T)
s_logp = log_softmax(student_logits[i, mask] / T)
kd_i = kl_div(s_logp, t_prob, reduction='sum') * T * T
```

如果 `mask.sum() <= 1`，跳过 KL 或设为 0。

由于 HOI 是 multi-label，如果 KL 训练不稳定，可以改为 BCE KD：

```python
t_prob = sigmoid(teacher_logits[i, mask] / T)
kd_i = binary_cross_entropy_with_logits(student_logits[i, mask] / T, t_prob)
```

但如果论文正文写 KL，优先实现 KL，并在必要时用 BCE 作为工程 fallback。

### 7.5 hard new-predicate negative margin

对 replay old positive pair，使用 confounder ids 和 weights：

```python
margin_i = sum_c a_ic * relu(margin + z_student[v_c] - z_student[v_pos])
```

其中：

- `v_pos` 是旧正类 verb。
- `v_c` 是当前任务 confounder HOI 的 verb。
- `a_ic` 是 confounder 权重。

重要约束：

- 默认只对 `v_c` 在当前 pair object 下有效的情况应用 margin，即 `pair_prior[i, v_c] > 0`。
- 如果 object-disjoint protocol 下没有有效 `v_c`，margin term 可以为 0，但 attention modulation 和 feature/logit KD 仍然有效。
- 不建议对 object-incompatible verb 强制 margin，否则可能压制同一 predicate 在其他 object 上的泛化能力。

### 7.6 SIS 加权

最终：

```python
loss_i = feat_weight * feat_loss_i + logit_weight * kd_i + hardneg_weight * margin_i
L_int = sum(w_i * loss_i) / sum(w_i)
```

其中 `w_i` 是 Module 1 得到的 batch-normalized SIS weight。

---

## 8. `PViC.forward` 具体修改建议

新增函数签名：

```python
def forward(
    self,
    images,
    targets=None,
    return_cross_attn=False,
    return_outputs=False,
    teacher_cross_attn_hint=None,   # 保留兼容，但 RGIP 不用
    attn_hint_alpha=0.05,
    batch_indices=None,
    rgip_context=None,
    return_rgip_meta=False,
):
```

### 8.1 传递 per-image attention context

当前循环：

```python
for i, (ho_q, mem) in enumerate(zip(ho_queries, memory)):
    out, cross_attn_weights = self.decoder(...)
```

改为：

```python
global_idx = batch_indices[i] if batch_indices is not None else i
image_rgip_context = None
if rgip_context is not None:
    image_rgip_context = rgip_context.get(global_idx, None)

out, cross_attn_weights = self.decoder(
    ho_q.unsqueeze(1),
    mem.unsqueeze(1),
    kv_padding_mask=kv_p_m[i],
    q_pos=positional_embeds[i],
    k_pos=k_pos[i],
    return_cross_attn=True,
    rgip_attn_context=image_rgip_context,
)
```

### 8.2 构造 pair weights

在 `labels = associate_with_ground_truth(...)` 后：

```python
pair_weights = torch.ones(labels.shape[0], device=labels.device, dtype=logits.dtype)

if rgip_context is not None:
    offset = 0
    for local_img_idx, p_inds in enumerate(paired_inds):
        global_idx = batch_indices[local_img_idx] if batch_indices is not None else local_img_idx
        ctx = rgip_context.get(global_idx, None)
        n = len(p_inds)
        if ctx is not None and 'pair_weights' in ctx:
            # ctx['pair_weights'] shape: [n]
            pair_weights[offset:offset+n] = ctx['pair_weights'].to(pair_weights.device)
        offset += n
```

然后：

```python
cls_loss = self.compute_classification_loss(
    logits, prior_scores, labels, pair_weights=pair_weights
)
```

### 8.3 输出 RGIP metadata

训练时 output_dict 增加：

```python
pair_objects = torch.cat(object_types, dim=0)
pair_prior = torch.cat(prior_scores, dim=0).prod(1)

pair_global_indices = None
if batch_indices is not None:
    pair_global_indices = torch.tensor(
        [batch_indices[i] for i in pair_image_indices.tolist()],
        device=feat.device
    )

output_dict.update({
    'pair_labels': labels,
    'pair_objects': pair_objects,
    'pair_prior': pair_prior,
    'pair_global_indices': pair_global_indices,
    'paired_inds': paired_inds,
})
```

如果 `return_rgip_meta=True`，再额外返回 context 中实际使用的：

```python
output_dict['rgip_pair_weights'] = pair_weights
output_dict['rgip_sis'] = ...
output_dict['rgip_confounder_ids'] = ...
output_dict['rgip_confounder_weights'] = ...
```

---

## 9. `utils_incremental.py` 训练循环建议

当 `args.use_rgip` 为 True 时，建议 `_on_each_iteration` 改成下面结构。

```python
def _on_each_iteration(self):
    if invalid batch:
        return

    images = self._state.inputs
    targets = self._state.targets
    batch_indices = self._state.batch_indices

    rgip_context = None
    teacher_replay_output = None
    selected_indices = []

    if self.config.use_rgip and self.teacher_model is not None and len(self.replay_indices) > 0:
        replay_mask = [idx in self.replay_indices for idx in batch_indices]
        replay_images = [img for img, flag in zip(images, replay_mask) if flag]
        replay_targets = [tar for tar, flag in zip(targets, replay_mask) if flag]
        selected_indices = [idx for idx, flag in zip(batch_indices, replay_mask) if flag]

        if len(replay_images) > 0:
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_replay_output = self.teacher_model(
                    replay_images,
                    targets=replay_targets,
                    return_outputs=True,
                    batch_indices=selected_indices,
                    return_rgip_meta=True,
                )

            replay_context = self.rgip_state.build_replay_pair_context(
                teacher_replay_output,
                selected_global_indices=selected_indices,
            )

            rgip_context = self.rgip_state.build_student_batch_context(
                replay_context,
                batch_indices=batch_indices,
            )

    outputs = self._state.net(
        images,
        targets=targets,
        return_outputs=True,
        batch_indices=batch_indices,
        rgip_context=rgip_context,
        return_rgip_meta=True,
    )

    loss_dict = {'cls_loss': outputs['cls_loss']}

    rgip_int_loss = 0.0
    if self.config.use_rgip and teacher_replay_output is not None:
        rgip_int_loss = self.rgip_state.compute_interaction_distillation(
            student_output=outputs,
            teacher_output=teacher_replay_output,
            replay_context=rgip_context,
        )

    total_loss = loss_dict['cls_loss'] + self.config.rgip_int_loss_weight * rgip_int_loss

    self.rgip_state.update_current_task_prototypes(outputs)

    backward + optimizer step
```

注意：

- `update_current_task_prototypes(outputs)` 建议放在 `optimizer.step()` 前后都可以，但必须对 feature detach，不能让 prototype bank 参与梯度。
- 如果当前 batch 没有 replay images，`rgip_context=None`，代码应正常退化为普通增量训练。
- 如果 prototype bank 为空，attention modulation 和 hard negative margin 应跳过，不应报错。

---

## 10. `main_incremental.py` 修改建议

### 10.1 teacher 每个 rank 都加载

将当前逻辑：

```python
if dist.get_rank() == 0:
    teacher_model = build_detector(...)
else:
    teacher_model = None
```

改为：

```python
teacher_model = None
if (args.use_distill or args.use_rgip) and task_idx > 0 and prev_ckpt is not None and os.path.exists(prev_ckpt):
    teacher_model = build_detector(args, object_to_target)
    checkpoint = torch.load(prev_ckpt, map_location='cpu')
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.to(rank)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
```

如果 `CustomisedDLE.__init__` 内部仍然调用 `teacher_model.cuda()`，注意不要重复搬错设备。建议改成：

```python
self.teacher_model = teacher_model
if self.teacher_model is not None:
    self.teacher_model.to(self._device)
    self.teacher_model.eval()
```

### 10.2 传入 RGIP 所需类信息

创建 engine 时增加：

```python
engine = CustomisedDLE(
    model,
    train_loader,
    test_loader,
    args,
    filter_classes=eval_classes,
    teacher_model=teacher_model,
    replay_indices=replay_indices if args.use_replay and task_idx > 0 else None,
    mir_dict=mir_dict,
    rare_set=rare_set,
    mir_min=mir_min,
    mir_max=mir_max,
    old_hoi_classes=sum(tasks[:task_idx], []),
    current_hoi_classes=task_classes,
    hoi_to_obj=hoi_to_obj,
    hoi_to_verb=hoi_to_verb,
    obj_verb_to_hoi=obj_verb_to_hoi,
    object_to_verb=object_to_target,
)
```

如果不想改 `CustomisedDLE` 参数过多，也可以把这些放入 `args`，但不如显式参数清晰。

---

## 11. RGIP context 推荐格式

为了让 `PViC.forward` 和 decoder 使用方便，建议 `rgip_context` 是一个 dict：

```python
rgip_context = {
    global_image_idx_1: {
        'pair_weights': Tensor[num_pairs_in_image],
        'sis': Tensor[num_pairs_in_image],
        'modulate_mask': BoolTensor[num_pairs_in_image],
        'old_positive_verbs': List[List[int]],
        'old_positive_hoi': List[List[int]],
        'confounder_hoi_ids': LongTensor[num_pairs_in_image, K],
        'confounder_verbs': LongTensor[num_pairs_in_image, K],
        'confounder_weights': Tensor[num_pairs_in_image, K],
        'confounder_prototypes': Tensor[num_pairs_in_image, K, repr_dim],
        'old_predicate_masks': BoolTensor[num_pairs_in_image, num_verbs],
    },
    ...
}
```

对于没有 RGIP 的 image，不放入 dict 即可。

对于没有被调制的 pair：

- `pair_weights = 1`
- `sis = 0`
- `modulate_mask = False`
- confounder tensors 可填 0，但计算时必须由 mask 跳过。

---

## 12. 关键鲁棒性处理

实现时必须处理以下情况：

1. **无 replay image**
   - 跳过 teacher forward、SIS、RGIP distillation。
   - student 正常训练。

2. **无 old positive pair**
   - `SIS=0`，`pair_weight=1`。
   - 不参与 RGIP distillation。

3. **无 current prototype**
   - attention modulation 跳过。
   - hard negative margin 跳过。
   - feature/logit KD 仍可执行。

4. **object-disjoint split**
   - `same_object` 全为 0 时，confounder 仍由 query similarity 和 semantic similarity 选择。
   - 如果 hard negative verb 对 old object 不兼容，margin 跳过。

5. **多标签 HOI pair**
   - SIS 对每个 old positive label 计算，默认取最大风险作为 pair SIS。
   - distillation 的 old predicate mask 包含该 object 下所有 old-compatible verbs。
   - margin 可对多个 positive old verbs 取最小 loss 或平均；最低成本用 teacher confidence 最高的 positive verb。

6. **DDP**
   - teacher 每个 rank 都要有。
   - prototype bank 最好同步；如果不同步，至少在日志中说明 local bank。

7. **数值稳定**
   - 所有权重加 `clamp`。
   - `sum(weights)` 分母加 `1e-6`。
   - attention negative bias clamp 到 `[-attn_clamp, 0]`。
   - KL mask 为空时跳过。

---

## 13. 最小可行实现顺序

建议服务器端按以下阶段实现，不要一次性改所有内容后再调试。

### Stage 1：metadata、teacher 和 RGIP 框架

目标：

- 加入 `--use-rgip` 等参数。
- 每个 rank 正确加载 teacher。
- `PViC.forward` 能返回 RGIP metadata。
- 新增 `rgip_utils.py` 和空的 `RarityGuidedState`。

验收：

- `--use-rgip` 开启但所有 RGIP loss 为 0 时，训练能跑通。
- output 中包含 `pair_labels/pair_objects/pair_prior/pair_global_indices`。

### Stage 2：SIS + weighted classification loss

目标：

- teacher replay forward。
- 计算 replay pair SIS。
- 将 `pair_weights` 传入 `compute_classification_loss`。

验收：

- 日志输出：
  - `rgip/num_replay_images`
  - `rgip/num_replay_pairs`
  - `rgip/num_old_positive_pairs`
  - `rgip/sis_mean`
  - `rgip/sis_max`
  - `rgip/weight_mean`
- 训练 loss 无 NaN。
- 关闭 attention 和 RGIP distill 时，只测试 weighted replay 的 ablation。

### Stage 3：interaction distillation

目标：

- 实现 student/teacher replay pair 对齐。
- 实现 feature KD 和 old predicate distribution KD。
- 实现 hard negative margin，但允许无 prototype 时跳过。

验收：

- 日志输出：
  - `rgip/feat_kd`
  - `rgip/logit_kd`
  - `rgip/hardneg_loss`
  - `rgip/int_loss`
- 与旧 replay distill 比较，至少训练稳定。

### Stage 4：prototype bank + confounder selection

目标：

- 用当前任务正样本更新 prototypes。
- 对 replay old pair 选择 Top-K current confounders。

验收：

- 日志输出：
  - `rgip/prototype_count`
  - `rgip/confounder_valid_ratio`
  - `rgip/same_object_ratio`
- 在 object-disjoint split 下 `same_object_ratio` 可以接近 0，但 `confounder_valid_ratio` 不应为 0。

### Stage 5：content-only attention modulation

目标：

- 改 decoder cross-attention，显式拆分 content/position logits。
- 用 confounder field 产生 content bias。

验收：

- 无 RGIP context 时，模型输出维度与旧代码一致。
- 开启 RGIP 后：
  - `rgip/modulated_pair_count > 0`
  - `rgip/attn_bias_norm` 合理非零
  - loss 不 NaN
- 推理阶段不需要 prototype bank，也不需要 teacher。

---

## 14. 推荐日志

如果使用 wandb 或普通 print，建议记录：

```text
rgip/sis_mean
rgip/sis_max
rgip/sis_min
rgip/pair_weight_mean
rgip/pair_weight_max
rgip/num_replay_images
rgip/num_replay_pairs
rgip/num_old_positive_pairs
rgip/prototype_count
rgip/confounder_valid_ratio
rgip/same_object_ratio
rgip/modulated_pair_count
rgip/attn_bias_abs_mean
rgip/feat_kd
rgip/logit_kd
rgip/hardneg_loss
rgip/int_loss
```

实验指标建议至少保留：

- Full mAP
- Rare mAP
- Non-Rare mAP
- old mAP
- new mAP
- average forgetting
- high-SIS group AP drop

---

## 15. 训练命令模板

根据服务器原有脚本调整路径。下面只是参数结构模板：

```bash
python main_incremental.py \
  --use-replay \
  --use-distill \
  --use-rgip \
  --n-replay 50 \
  --rgip-alpha 0.4 \
  --rgip-beta 0.3 \
  --rgip-gamma 2.0 \
  --rgip-topk 3 \
  --rgip-eta-object 0.4 \
  --rgip-eta-query 0.4 \
  --rgip-eta-sem 0.2 \
  --rgip-lambda-attn 0.5 \
  --rgip-int-loss-weight 1.0 \
  --rgip-feat-weight 1.0 \
  --rgip-logit-weight 1.0 \
  --rgip-hardneg-weight 0.5 \
  --rgip-debug
```

如果暂时不实现 semantic embedding：

```bash
--rgip-eta-object 0.5 --rgip-eta-query 0.5 --rgip-eta-sem 0.0
```

如果先跑不带 attention 的消融：

```bash
--rgip-lambda-attn 0.0
```

如果先跑不带 hard negative 的消融：

```bash
--rgip-hardneg-weight 0.0
```

---

## 16. 必做消融实验对应的开关

为了后续论文实验方便，建议实现以下 ablation 控制项：

| 实验 | 参数设置 |
|---|---|
| Replay baseline | `--use-replay`，不加 `--use-rgip` |
| Weighted replay only | `--use-rgip --rgip-int-loss-weight 0 --rgip-lambda-attn 0` |
| + Interaction KD | `--use-rgip --rgip-lambda-attn 0 --rgip-int-loss-weight 1` |
| + Attention modulation | 完整 RGIP |
| Frequency-only SIS | `--rgip-alpha 1 --rgip-beta 0` |
| Confusion-only SIS | `--rgip-alpha 0 --rgip-beta 0.5`，entropy 占 0.5 |
| No hard negative | `--rgip-hardneg-weight 0` |
| Same-object confounder only | `--rgip-eta-object 1 --rgip-eta-query 0 --rgip-eta-sem 0` |
| Hybrid confounder | 默认完整设置 |

---

## 17. 最容易出错的点

1. **不要把 600 HOI id 直接当成 classifier 输出维度。**
   当前 classifier 是 117 verb logits。所有 HOI-level 操作必须经过 object+verb 映射。

2. **不要继续使用当前 attention hint 作为主方法。**
   它只改返回的 attention weight，没有重新计算 decoder output，不会改变训练目标。

3. **不要对 `cls_loss` 做 teacher/student MSE。**
   loss 标量不是知识表示，不能支撑论文的 interaction distillation claim。

4. **teacher 不要 `train()`。**
   SIS 和 distillation target 必须来自稳定的 frozen teacher。

5. **teacher 不要只在 rank0 加载。**
   DDP 下每个 rank 都要能计算 RGIP。

6. **hard negative margin 不要盲目压制 object-incompatible verb。**
   默认只在该 verb 对当前 old object 有效时应用 margin。

7. **attention modulation 必须真实影响 decoder output。**
   只返回或保存 attention map 不等于调节模型。

8. **object-disjoint protocol 下不能依赖 shared object。**
   confounder 选择必须保留 query similarity 和 semantic similarity。

---

## 18. 完成标准

代码修改完成后，应满足：

1. `--use-rgip` 关闭时，原训练/测试流程仍可运行。
2. `--use-rgip` 开启时：
   - teacher 每个 rank 正常加载；
   - replay batch 能计算 SIS；
   - classification loss 使用 pair-level SIS weight；
   - interaction distillation loss 非零且稳定；
   - prototype bank 能更新；
   - attention modulation 在有 prototype 和 replay old pair 时真实生效；
   - inference 不需要 teacher/prototype/RGIP context。
3. 至少能跑通以下三组实验：
   - Replay baseline；
   - Weighted replay + interaction KD；
   - Full RGIP。

如果时间有限，优先完成 Stage 1--3。这已经可以支撑 “rarity-guided replay + interaction distillation” 的核心实验。Stage 4--5 是论文创新性最关键的 attention/confounder 部分，应作为最终投稿版本完成。

