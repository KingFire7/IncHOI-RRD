# 代码结构

## 主流程

`main_incremental.py` 是增量实验入口，流程如下：

```text
任务划分
  -> 构造当前类训练集
  -> 可选加入历史样本回放
  -> 从上一任务 checkpoint 初始化学生/教师模型
  -> 训练当前任务
  -> 保存 checkpoint_taskN.pth
  -> 在指定类别集合上评估
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `main_incremental.py` | 任务划分、数据加载、模型衔接、逐任务训练与保存 |
| `utils_incremental.py` | `DataFactory`、训练引擎、蒸馏/提示/评估逻辑 |
| `mir_utils.py` | 历史样本读取、按置信度排序和新旧样本交错 |
| `pvic.py` | PViC 检测器、特征融合及前向计算 |
| `transformers.py` | 编解码器和注意力模块 |
| `configs.py` | base/advanced 检测器公共参数 |
| `evaluate_forgetting.py` | 读取各任务 checkpoint，计算逐类 AP 与遗忘量 |
| `evaluate_forgetting_viz.py` | 绘制单次实验的学习/遗忘曲线 |
| `evaluate_forgetting_comp.py` | 比较多次实验的 Top-K 遗忘结果 |
| `inference_demo.py` | 比较两个模型的预测及交叉注意力图 |

## 增量机制开关

| 参数 | 作用 |
| --- | --- |
| `--use-replay` | 加入历史类别样本 |
| `--dynamic-replay` | 按上一阶段置信度排列回放样本 |
| `--use-distill` | 使用上一任务模型作为教师 |
| `--replay-distill` | 对回放样本施加特征或 logits 蒸馏 |
| `--use-attn-hint` | 使用教师注意力作为提示 |
| `--use-vas` | 启用脆弱性感知抑制 |

## 外部依赖

五个上游组件以 Git 子模块保留，避免复制第三方代码历史：`detr`、`h_detr`、`pocket`、`hicodet`、`vcoco`。本仓库只提交源码与小型元数据；数据、权重和实验产物留在本地。

## 历史文件处理

根目录中的 `* copy.py` 是手工迭代副本，不参与当前入口的 import，已由 `.gitignore` 排除但未从本地删除。当前有效版本是不带 `copy` 后缀的文件。
