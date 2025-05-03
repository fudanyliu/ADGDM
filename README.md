# [ADGDM](TBD)

<a href="https://arxiv.org/pdf/2501.11430.pdf" alt="paper"><img src="https://img.shields.io/badge/ArXiv-2501.11430-FAA41F.svg?style=flat" /></a>
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/Siriussee/adgdm) 
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/Siriussee/adgdm) 
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FSiriussee%2Fadgdm&countColor=%23263759&style=flat-square)


Welcome to the official repository for "Anomaly Detection and Generation with Diffusion Models: A Survey", submitted to [Journal](TBD).
In this survey, we comprehensively review anomaly detection and generation with diffusion models (ADGDM), presenting a tutorial-style analysis of the theoretical foundations and practical implementations and spanning images, videos, time series, tabular, and multimodal data. Crucially, we reveal how DMs create a synergistic cycle where generation addresses data scarcity challenges while detection provides feedback for refined generation strategies, advancing both capabilities beyond their individual potential. A detailed taxonomy categorizes ADGDM methods based on anomaly scoring mechanisms, conditioning strategies, and architectural designs, analyzing their strengths and limitations. Key challenges, such as computational efficiency and the identity shortcut problem, are discussed alongside promising future directions, including efficient architectures, novel conditioning strategies, and integration with large AI models (e.g., LLMs). By synthesizing recent advancements and outlining open research questions, this survey aims to guide researchers and practitioners in leveraging DMs for innovative AD solutions across diverse applications.

<img style="margin-bottom: -50px;" src="https://raw.githubusercontent.com/Siriussee/adgdm/main/assets/f1.svg">
<!-- <b align="center" style="margin-top: -10px;">Taxonomy of diffusion models for anomaly detection</b> -->
<p align="center" style="margin-top: -10px;"><b align="center" style="margin-top: -10px;">Taxonomy of diffusion models for anomaly detection</b></p>

✨ If you found this survey and repository useful, please consider to star this repository and cite our survey paper:

```bib
@misc{liu2025anomaly,
    title = {Anomaly Detection and Generation with Diffusion Models: A Survey},
    author = {Liu, Jing and Ma, Zhenchao and Wang, Zepu and Wang, and Zou, Chenxuanyin and Ren, Jiayang and Zehua and Liu, Yang and Song, Liang and Hu, Bo and Leung, Victor C. M.},
    year = {2025},
    month = jan,
    number = {arXiv:2501.11430},
    eprint = {2501.11430},
    primaryclass = {cs},
    doi = {10.48550/arXiv.2501.11430},
}
```

## Table of Contents

TBD

## Image Anomaly Detection

| Paper Title                                                                                                           | Year | Venue    | Imaging Domains              | DM  | Code                                                                                                  |
|-----------------------------------------------------------------------------------------------------------------------|------|----------|------------------------------|-----|-------------------------------------------------------------------------------------------------------|
| Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models                                      | 2022 | MICCAI   | Brain CT and MRI             | ✓   | -                                                                                                     |
| Unsupervised 3D Out-of-Distribution Detection with Latent Diffusion Models                                             | 2023 | MICCAI   | 3D medical data              | ✓   | [Link](https://github.com/marksgraham/ddpm-ood)                                                       |
| Fast Non-Markovian Diffusion Model for Weakly Supervised Anomaly Detection in Brain MR Images                           | 2023 | MICCAI   | Brain MR images              | ✓   | -                                                                                                     |
| Cold Diffusion: Inverting Arbitrary Image Transforms without Noise                                                    | 2023 | NeurIPS  | General image                | ✓   | -                                                                                                     |
| Guided Image Synthesis via Initial Image Editing in Diffusion Model                                                    | 2023 | ACM MM   | Text-image pairs             | ✓   | -                                                                                                     |
| Diffusion Models with Implicit Guidance for Medical Anomaly Detection                                                  | 2024 | MICCAI   | Brain MRI, Wrist X-rays      | ✓   | [Link](https://github.com/compai-lab/2024-miccai-bercea-thor.git)                                     |
| Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images                                   | 2024 | IEEE TMI | Brain images                 | ✓   | -                                                                                                     |
| Unsupervised Anomaly Detection in Medical Images Using Masked Diffusion Model                                          | 2024 | MLMI     | Brain MRI                    | ✓   | [Link](https://mddpm.github.io/)                                                                      |
| Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI                           | 2024 | MICCAI   | Multimodal MRI               | ✓   | -                                                                                                     |
| Ensembled Cold-Diffusion Restorations for Unsupervised Anomaly Detection                                               | 2024 | MICCAI   | Brain MRI                    | ✓   | -                                                                                                     |
| IgCONDA-PET: Implicitly-Guided Counterfactual Diffusion for Detecting Anomalies in PET Images                           | 2024 | ArXiv    | Medical imaging (PET)        | ✓   | [Link](https://github.com/igcondapet/IgCONDA-PET.git)                                                 |
| Detecting Out-of-Distribution Earth Observation Images with Diffusion Models                                           | 2024 | CVPRW    | Remote sensing               | ✓   | -                                                                                                     |
| AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model                                               | 2024 | AAAI     | Industrial images            | ✓   | [Link](https://github.com/sjtuplayer/anomalydiffusion)                                                |
| DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation                                     | 2024 | ArXiv    | Industrial images            | ✓   | -                                                                                                     |
| Pancreatic Tumor Segmentation as Anomaly Detection in CT Images Using Denoising Diffusion Models                         | 2024 | ArXiv    | Medical imaging              | ✓   | -                                                                                                     |
| UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion                                             | 2024 | ACL      | Text-image pairs             | ✓   | -                                                                                                     |
| Image-Conditioned Diffusion Models for Medical Anomaly Detection                                                       | 2025 | USMLMI   | Medical imaging              | ✓   | [Link](https://github.com/matt-baugh/img-cond-diffusion-model-ad)                                     |

## 
