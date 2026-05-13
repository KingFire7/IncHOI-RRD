# Towards Class-Incremental Human-Object Interaction Detection via Rarity-Aware Relational Distillation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-class-incremental-human-object)](https://paperswithcode.com/sota)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> This is the official PyTorch implementation of the paper **"Towards Class-Incremental Human-Object Interaction Detection via Rarity-Aware Relational Distillation"**.
> 
> *The paper is currently under review at Computer Vision and Image Understanding (CVIU).*

## 📢 News
* **[2026-05]** Our paper has been submitted to CVIU and is currently under review. 
* **[2026-05]** We have created this repository. The source code, pre-trained weights, and dataset splits will be released here upon acceptance or reasonable request.

## 📝 TODO List
- [x] Create repository and provide overview.
- [ ] Release the core implementation of Rarity-Aware Relational Distillation (RRD).
- [ ] Release the training and evaluation scripts for incremental HOI scenarios.
- [ ] Upload pre-trained models and checkpoints.
- [ ] Provide comprehensive setup instructions for Conda environments.

## 💡 Abstract
Class-incremental Human-Object Interaction (HOI) detection requires models to continuously learn novel interaction categories in dynamic environments. While existing methods rely on conventional experience replay or global distillation, they often succumb to catastrophic forgetting due to the inherent long-tailed distribution of HOI data. To tackle this issue, we reframe the incremental learning process to prioritize rarity-aware knowledge preservation, ensuring the stability of infrequent historical interactions. Specifically, we develop a unified teacher-student framework named Rarity-Aware Relational Distillation (RRD) to systematically mitigate the relational drift across evolving interaction concepts. Recognizing the structural complexity of modern Transformer-based HOI detectors, we introduce a dynamic replay sampling strategy governed by a Sample Importance Score (SIS) to prioritize the retention of highly vulnerable historical instances. To anchor global relational dependencies and maintain fine-grained interaction semantics, we design a Hint-Guided Relational Topology Alignment module to preserve the established spatial structures, alongside pairwise distillation with adaptive reweighting for rare classes. Extensive experiments on the HICO-DET benchmark demonstrate that our RRD framework robustly acquires new categories while shielding fragile historical HOI knowledge, achieving state-of-the-art performance and superior stability on rare interaction categories across both the standard multi-task incremental setting and the rigorous New Concept discovery protocols.

## 🛠️ Requirements
The code is built and tested on the following environment:
* Ubuntu 
* Python >= 3.8
* PyTorch >= 1.10
* CUDA >= 11.3 (Tested on NVIDIA RTX 3090)
* *Detailed `requirements.txt` and Conda environment setup instructions will be provided soon.*

## 🚀 Usage
*(Coming soon)*

## ✉️ Contact
If you have any questions about the code or paper, please feel free to open an issue or contact [hujiaming1214@stu.xjtu.edu.cn].

## 🔗 Citation
If you find our work useful in your research, please consider citing:
```bibtex
@article{hu2026inchoi,
  title={Towards Class-Incremental Human-Object Interaction Detection via Rarity-Aware Relational Distillation},
  author={Hu, Jiaming and others},
  journal={Computer Vision and Image Understanding},
  note={Under Review},
  year={2026}
}
