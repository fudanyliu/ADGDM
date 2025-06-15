# [Anomaly Detection and Generation with Diffusion Models: A Survey](./blob/main/main.pdf)

<a href="https://arxiv.org/pdf/2506.09368.pdf" alt="paper"><img src="https://img.shields.io/badge/ArXiv-2506.09368-FAA41F.svg?style=flat" /></a>
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/Siriussee/adgdm) 
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/Siriussee/adgdm) 
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FSiriussee%2Fadgdm&countColor=%23263759&style=flat-square)

Welcome to the official repository for "Anomaly Detection and Generation with Diffusion Models: A Survey", submitted to [IEEE TPAMI](https://arxiv.org/pdf/2506.09368). In this survey, we comprehensively review anomaly detection and generation with diffusion models (ADGDM), presenting a tutorial-style analysis of the theoretical foundations and practical implementations and spanning images, videos, time series, tabular, and multimodal data. Crucially, we reveal how DMs create a synergistic cycle where generation addresses data scarcity challenges while detection provides feedback for refined generation strategies, advancing both capabilities beyond their individual potential. A detailed taxonomy categorizes ADGDM methods based on anomaly scoring mechanisms, conditioning strategies, and architectural designs, analyzing their strengths and limitations. We final discuss key challenges including scalability and computational efficiency, and outline promising future directions such as efficient architectures, conditioning strategies, and integration with foundation models (e.g., visual-language models and large language models). By synthesizing recent advances and outlining open research questions, this survey aims to guide researchers and practitioners in leveraging DMs for innovative AD solutions across diverse applications.

<img style="margin-bottom: -50px;" src="https://github.com/user-attachments/assets/8008325b-0316-4abf-a938-20703d64ddde">
<!-- <b align="center" style="margin-top: -10px;">Taxonomy of diffusion models for anomaly detection</b> -->

<p align="center"><b align="center">Fig.1: Publication and citation trends in anomaly-related research topic from 2021 to 2025</b></p>

✨ If you found this survey and repository useful, please consider to star this repository and cite our survey paper:

```bib
@misc{liu2025anomaly,
  title = {Anomaly Detection and Generation with Diffusion Models: A Survey},
  shorttitle = {Anomaly Detection and Generation with Diffusion Models},
  author = {Liu, Yang and Liu, Jing and Li, Chengfang and Xi, Rui and Li, Wenchao and Cao, Liang and Wang, Jin and Yang, Laurence T. and Yuan, Junsong and Zhou, Wei},
  year = {2025},
  month = jun,
  number = {arXiv:2506.09368},
  eprint = {2506.09368},
  primaryclass = {cs},
  doi = {10.48550/arXiv.2506.09368}
}

```

## Table of Contents
<!-- no toc -->

* [Recently AD Surveys](#recently-ad-surveys)
* [Image Anomaly Detection](#image-anomaly-detection)
* [Video Anomaly Detection](#video-anomaly-detection)
* [Time Series Anomaly Detection](#time-series-anomaly-detection)
* [Tabular Anomaly Detection](#tabular-anomaly-detection)
* [Multimodal Anomaly Detection](#multimodal-anomaly-detection)
* [Anomaly Generation](#anomaly-generation)
* [Performance Evaluation](#performance-evaluation)
  * [Evaluation Metrics](#evaluation-metrics)
  * [Public Datasets](#public-datasets)


## Recently AD Surveys

<p align="center"><b align="center">TABLE1: Summary of our survey with recently AD surveys.</b></p>

| Paper Title                                                                                                                                         | Year | Main Focus                               | Domain Specific | Domain General | IAD  | VAD  | TSAD | TAD  | MAD  | AG   | DMs  | Dataset | Metric |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------|-----------------|----------------|------|------|------|------|------|------|------|---------|--------|
| Anomaly Detection for IoT Time-Series Data: A Survey                                                                                                | 2020 | Time series AD in IoT                    | ✓               | -              | N/A  | N/A  | ●    | ○    | ○    | N/A  | ○    | ○       | ◐      |
| Deep Learning for Anomaly Detection: A Review                                                                                                       | 2021 | General AD with DL                       | -               | ✓              | ●    | ◐    | ●    | ◐    | ◐    | ○    | ○    | ◐       | ◐      |
| Deep Learning for Medical Anomaly Detection - a Survey                                                                                             | 2022 | Medical AD                               | ✓               | -              | ●    | ○    | ◐    | ◐    | ○    | ○    | ○    | ○       | ●      |
| A Survey of Single-Scene Video Anomaly Detection                                                                                                    | 2022 | Single-scene video AD                    | ✓               | -              | ○    | ●    | N/A  | N/A  | ○    | ○    | ○    | ●       | ●      |
| Anomaly Detection in Surveillance Videos: A Thematic Taxonomy of Deep Models, Review and Performance Analysis                                        | 2023 | DL for surveillance VAD                  | ✓               | -              | ○    | ●    | ○    | ○    | ○    | ○    | ○    | ●       | ◐      |
| A Comprehensive Survey on Graph Anomaly Detection with Deep Learning                                                                                 | 2023 | Graph AD with DL                         | -               | ✓              | ○    | ○    | ○    | ◐    | ◐    | ○    | ◐    | ●       | ●      |
| Anomaly Detection in Blockchain Networks: A Comprehensive Survey                                                                                    | 2023 | Blockchain AD                            | ✓               | -              | N/A  | N/A  | ◐    | ◐    | ○    | ○    | ○    | ◐       | ●      |
| A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection                                    | 2024 | GNN for time series analytics            | -               | ✓              | ○    | ○    | ●    | ◐    | ○    | ○    | ○    | ●       | ●      |
| Generalized Video Anomaly Event Detection: Systematic Taxonomy and Comparison of Deep Models                                                          | 2024 | Generalized VAD taxonomy                 | -               | ✓              | ◐    | ●    | ○    | ○    | ◐    | ○    | ○    | ●       | ●      |
| Networking Systems for Video Anomaly Detection: A Tutorial and Survey                                                                               | 2025 | Networking systems for VAD               | ✓               | -              | ◐    | ●    | ○    | ○    | ◐    | ○    | ◐    | ●       | ●      |
| Deep Learning for Time Series Anomaly Detection: A Survey                                                                                           | 2025 | Time series AD with DL                   | -               | ✓              | ○    | ○    | ●    | ◐    | ○    | ◐    | ○    | ●       | ●      |
| Ours                                                                                                                                                  | 2025 | AD & AG with DM                          | -               | ✓              | ●    | ●    | ●    | ●    | ●    | ●    | ●    | ●       | ●      |

> Notes: Specifically, ○: "Not Covered", ◐: "Partially Covered", N/A: "Not applicable", and ●: "Fully Covered". The table highlights the main focus, domain specificity, and scope across various AD tasks, including image AD (IAD), video AD (VAD), time series AD (TSAD), tabular AD (TAD), multimodal AD (MAD), anomaly generation (AG), and diffusion models (DMs).

## Image Anomaly Detection

<p align="center"><b align="center">TABLE 2: Summary of IAD methods across various imaging domains with implementations.</b></p>

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


<p align="center"><b align="center">TABLE 3: Performance comparison of key IAD methods on VisA and BTAD datasets.</b></p>

| Paper Title                                                                                                                     | Year | Venue  | VisA (I-AUROC)  | BTAD (PRO)    |
|---------------------------------------------------------------------------------------------------------------------------------|------|--------|-----------------|---------------|
| DRAEM - a Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection                                       | 2021 | ICCV   | 88.7, 73.1      | -             |
| Sub-Image Anomaly Detection with Deep Pyramid Correspondences                                                                   | 2021 | Arxiv  | 82.1, 65.9      | -             |
| PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization                                           | 2021 | ICPR   | 89.1, 85.9      | -             |
| Anomaly Detection via Reverse Distillation from One-Class Embedding                                                             | 2022 | CVPR   | 96.0, 70.9      | 94.3, 77.1    |
| Towards Total Recall in Industrial Anomaly Detection                                                                            | 2022 | CVPR   | 95.1, 91.2      | 92.7, 77.3    |
| Removing Anomalies as Noises for Industrial Defect Localization                                                                 | 2023 | ICCV   | -               | 93.4, 78.7    |
| Dynamic Addition of Noise in a Diffusion Model for Anomaly Detection                                                              | 2024 | CVPRW  | 96.0, 94.1      | 95.2, 83.2    |
| GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection           | 2025 | ECCV   | 99.5, 98.6      | -             |

## Video Anomaly Detection

<p align="center"><b align="center">TABLE 4: Summary of VAD methods across various domains.</b></p>

| Paper Title                                                                                                                          | Year | Venue           | Type   | Domain                                      | DM      |
|--------------------------------------------------------------------------------------------------------------------------------------|------|-----------------|--------|---------------------------------------------|---------|
| Exploring Diffusion Models for Unsupervised Video Anomaly Detection                                                                  | 2023 | IEEE ICIP       | F      | Video surveillance                          | ✓       |
| Anomaly Detection in Satellite Videos Using Diffusion Models                                                                         | 2023 | Arxiv           | F      | Satellite imagery, disaster detection       | ✓       |
| Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models                                                       | 2023 | CVPR            | S, M   | Driving simulation                          | ✓       |
| Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation                                                                    | 2023 | Arxiv           | S      | Text-to-video generation                    | ✓       |
| Feature Prediction Diffusion Model for Video Anomaly Detection                                                                       | 2023 | ICCV            | F      | Video surveillance                          | ✓       |
| Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets                                                        | 2023 | Arxiv           | S, M   | Text/Image-to-video generation              | ✓       |
| Diversity-Measurable Anomaly Detection                                                                                                | 2023 | CVPR            | F      | Surveillance VAD                            | ✕       |
| Ensemble Anomaly Score for Video Anomaly Detection Using Denoise Diffusion Model and Motion Filters                                    | 2023 | Neurocomputing  | F      | Video surveillance                          | ✓       |
| Masked Diffusion with Task-Awareness for Procedure Planning in Instructional Videos                                                    | 2023 | Arxiv           | C      | Video procedure planning                    | ✓       |
| GD-VDM: Generated Depth for Better Diffusion-Based Video Generation                                                                    | 2023 | Arxiv           | S      | Complex scene video generation              | ✓       |
| AADiff: Audio-Aligned Video Synthesis with Text-to-Image Diffusion                                                                      | 2023 | Arxiv           | C      | Audio-aligned video synthesis               | ✓       |
| Diffusion-Based Normality Pre-Training for Weakly Supervised Video Anomaly Detection                                                   | 2024 | ESWA            | F      | Video surveillance                          | ✓       |
| Denoising Diffusion-Augmented Hybrid Video Anomaly Detection via Reconstructing Noised Frames                                            | 2024 | IJCAI           | F      | Security and surveillance                   | ✓       |
| Safeguarding Sustainable Cities: Unsupervised Video Anomaly Detection through Diffusion-Based Latent Pattern Learning                    | 2024 | IJCAI           | F      | Sustainable cities management               | ✓       |
| Graph-Jigsaw Conditioned Diffusion Model for Skeleton-Based Video Anomaly Detection                                                    | 2024 | Arxiv           | M      | Skeleton-based VAD                          | ✓       |
| VADiffusion: Compressed Domain Information Guided Conditional Diffusion for Video Anomaly Detection                                       | 2024 | Arxiv           | F      | Security surveillance                       | ✓       |
| Unsupervised Conditional Diffusion Models in Video Anomaly Detection for Monitoring Dust Pollution                                       | 2024 | Sensors         | F      | Dust pollution monitoring                   | ✓       |
| FedDiff: Diffusion Model Driven Federated Learning for Multi-Modal and Multi-Clients                                                   | 2024 | IEEE TCSVT      | C      | Multi-modal remote sensing                  | ✓       |

Notes: "Type" refers to Learning Paradigm: F = Frame-level, S = Sequence-level, M = Motion Modeling, C = Conditioning Strategies.

<p align="center"><b align="center">TABLE 5: Performance (AUC) comparison of key VAD methods on UCSD Ped2, CUHK Avenue, and ShanghaiTech.</b></p>

| Paper Title                                                                                                                          | Year | Venue                      | UCSD Ped2 | CUHK Avenue | ShanghaiTech |
|--------------------------------------------------------------------------------------------------------------------------------------|------|----------------------------|-----------|-------------|--------------|
| Weakly-Supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning                                             | 2021 | ICCV                       | 96.3      | 85.1        | 73           |
| Diversity-Measurable Anomaly Detection                                                                                                | 2023 | CVPR                       | 99.7      | 92.8        | 78.8         |
| Learnable Locality-Sensitive Hashing for Video Anomaly Detection                                                                       | 2023 | IEEE TCSVT | 91.3      | 87.4        | 77.6         |
| Boosting Variational Inference with Margin Learning for Few-Shot Scene-Adaptive Anomaly Detection                                        | 2023 | IEEE TCSVT | -         | 87.3        | 75.2         |
| Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection                                                      | 2023 | CVPR                       | -         | 88.3        | -            |
| Dual Memory Units with Uncertainty Regulation for Weakly Supervised Video Anomaly Detection                                              | 2023 | AAAI                       | -         | 88.9        | 97.92        |
| VADiffusion: Compressed Domain Information Guided Conditional Diffusion for Video Anomaly Detection                                      | 2024 | IEEE TCSVT | 98.2      | 87.2        | 71.7         |
| PLOVAD: Prompting Vision-Language Models for Open Vocabulary Video Anomaly Detection                                                     | 2025 | IEEE TCSVT | -         | -           | 97.98        |

Notes: "Performance (AUC) scores are reported on UCSD Ped2, CUHK Avenue, and ShanghaiTech datasets."

## Time Series Anomaly Detection

<p align="center"> <img src="https://github.com/user-attachments/assets/3a2ee11e-ead0-453f-a90c-1d3fe48085fb" alt=""> </p>
<p align="center"><b align="center">Fig. 6: TSAD with reconstruction and imputation paths.</b></p>

<p align="center"><b align="center">TABLE 6: Summary of TSAD with learning paradigms</b></p>

| Paper Title                                                                                                                                                 | Year | Venue                                                             | Learning Paradigm                   | DM   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------|-------------------------------------------------------------------|-------------------------------------|------|
| Drift Doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection                      | 2023 | NeurIPS                                                           | Decompo. and recon.                 | ✓    |
| DDMT: Denoising Diffusion Mask Transformer Models for Multivariate Time Series Anomaly Detection                                                               | 2023 | ArXiv                                                             | Mask-based                          | ✓    |
| Imputation-Based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models                                                              | 2023 | KDD                                                               | Imputation-based                    | ✓    |
| Diffusion-Based Time Series Imputation and Forecasting with Structured State Space Models                                                                       | 2023 | ArXiv                                                             | Imputation and forecasting          | ✓    |
| Diffusion-Based Time Series Data Imputation for Cloud Failure Prediction at Microsoft 365                                                                         | 2023 | ESEC/FSE                                                          | Imputation-based                    | ✓    |
| SaSDim: Self-Adaptive Noise Scaling Diffusion Model for Spatial Time Series Imputation                                                                            | 2023 | ArXiv                                                             | Noise-scaling diffusion             | ✓    |
| Time Series Anomaly Detection Using Diffusion-Based Models                                                                                                     | 2023 | ICDMW                                                             | Diffusion-based                     | ✓    |
| NetDiffus: Network Traffic Generation by Diffusion Models through Time-Series Imaging                                                                            | 2023 | ArXiv                                                             | Time-series imaging                 | ✓    |
| Diffusion Model in Normal Gathering Latent Space for Time Series Anomaly Detection                                                                               | 2024 | ECML PKDD                                                         | Latent space diffusion              | ✓    |
| Unsupervised Anomaly Detection for Multivariate Time Series Using Diffusion Model                                                                                | 2024 | ICASSP                                                            | Diffusion-based                     | ✓    |
| Unsupervised Diffusion Based Anomaly Detection for Time Series                                                                                                 | 2024 | APIN                                                              | Reconstruction-based                | ✓    |
| TimeDiT: General-Purpose Diffusion Transformers for Time Series Foundation Model                                                                               | 2024 | ICMLW                                                             | Foundation model                    | ✓    |
| Self-Supervised Learning of Time Series Representation via Diffusion Process and Imputation-Interpolation-Forecasting Mask                                      | 2024 | ArXiv                                                             | Self-supervised learning            | ✓    |
| Dynamic Splitting of Diffusion Models for Multivariate Time Series Anomaly Detection in a JointCloud Environment                                                  | 2024 | IEEE TCC                                                          | Reinforcement learning              | ✕    |
| ProDiffAD: Progressively Distilled Diffusion Models for Multivariate Time Series Anomaly Detection in JointCloud Environment                                        | 2024 | IJCNN                                                             | Progressive distillation            | ✓    |
| Contaminated Multivariate Time-Series Anomaly Detection with Spatio-Temporal Graph Conditional Diffusion Models                                                 | 2024 | ArXiv                                                             | Graph conditional diffusion         | ✓    |

<p align="center"><b align="center">TABLE 7: Performance comparison of key TSAD methods on SWaT, WADI, MSL, and SMD datasets.</b></p>

| Paper Title                                                                                                                                                 | Year | Venue    | SWaT (P, R, F1, F1_PA)          | WADI (P, R, F1, F1_PA)          | MSL (P, R, F1, F1_PA)           | SMD (P, R, F1, F1_PA)           |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------|----------|---------------------------------|--------------------------------|---------------------------------|--------------------------------|
| Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection                                                                                 | 2018 | ICLR     | 27.46, 69.52, 39.37, 85.33       | 54.44, 26.99, 36.09, 61.65       | 25.91, 62.86, 36.69, 70.09       | 42.59, 50.46, 26.85, 72.29       |
| Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network                                                               | 2019 | ACM KDD  | 98.25, 64.97, 78.22, 86.61       | 49.47, 12.98, 22.96, 41.72       | 16.19, 84.66, 27.18, 89.94       | 20.61, 46.73, 28.20, 75.29       |
| Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy                                                                                | 2021 | ICLR     | 12.00, 100.00, 21.43, 94.07      | 5.79, 43.43, 10.21, 89.10        | -, -, 2.10, 93.59               | -, -, 2.12, 92.33               |
| TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data                                                                      | 2022 | VLDB     | 97.94, 60.52, 74.16, 91.04       | 86.88, 15.50, 26.00, 42.04       | 29.06, 75.96, 42.04, 94.94       | 26.95, 57.36, 37.16, 76.95       |
| Drift Doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection                             | 2023 | NeurIPS  | 12.04, 99.59, 21.49, 90.55       | 6.23, 18.93, 11.75, 35.46        | 11.04, 93.01, 19.74, 87.44       | 23.70, 52.63, 26.12, 95.09       |
| Prototype-Oriented Unsupervised Anomaly Detection for Multivariate Time Series                                                                                | 2023 | ICML     | 97.94, 60.52, 74.16, 91.04       | 86.88, 15.50, 26.00, 42.04       | 29.06, 75.96, 42.04, 94.94       | 26.95, 57.36, 37.16, 76.95       |
| Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction                                                                  | 2023 | NeurIPS  | 93.42, 75.52, 83.52, 91.07       | 78.43, 50.33, 61.31, 75.22       | 24.03, 83.92, 37.37, 96.45       | 26.58, 62.36, 37.27, 76.95       |
| SensitiveHUE: Multivariate Time Series Anomaly Detection by Enhancing the Sensitivity to Normal Patterns                                                         | 2024 | ACM KDD  | 94.68, 87.74, 91.08, 96.75       | 86.51, 58.73, 69.96, 92.25       | 33.05, 71.26, 45.16, 98.42       | 29.54, 60.80, 39.76, 96.33       |

Notes: P, R, F1, and F1_PA refer to Precision, Recall, F1-score, and Point-Adjusted F1-score respectively.

## Tabular Anomaly Detection

<p align="center"> <img src="https://github.com/user-attachments/assets/bab762e5-ec5d-4b57-ab49-4acc873a02d" alt=""> </p>
<p align="center"><b align="center">Fig. 7: TAD handling mixed data types.</b></p>

<p align="center"><b align="center">TABLE 8: Summary of TAD methods with type and metrics</b></p>

| Paper Title                                                                                                              | Year | Venue     | Type                                | Performance Metrics                   | DM   |
|--------------------------------------------------------------------------------------------------------------------------|------|-----------|-------------------------------------|---------------------------------------|------|
| TabADM: Unsupervised Tabular Anomaly Detection with Diffusion Models                                                     | 2023 | ArXiv     | Diffusion-based                     | Detection accuracy                    | ✓    |
| Energy-Based Models for Anomaly Detection: A Manifold Diffusion Recovery Approach                                         | 2023 | NeurIPS   | Energy-based                        | AUPR, AUROC                           | ✕    |
| Generative Inpainting for Shapley-Value-Based Anomaly Explanation                                                         | 2024 | xAI       | Generative inpainting               | Explanation quality                   | ✓    |
| TimeAutoDiff: Combining Autoencoder and Diffusion Model for Time Series Tabular Data Synthesizing                          | 2024 | ArXiv     | VAE + DDPM hybrid                   | Fidelity and utility metrics          | ✓    |
| Unsupervised Anomaly Detection for Tabular Data Using Noise Evaluation                                                    | 2025 | AAAI      | Noise evaluation                    | AUC score                             | ✕    |
| CoDi: Co-Evolving Contrastive Diffusion Models for Mixed-Type Tabular Synthesis                                            | 2023 | ICML      | Co-evolving diffusion               | Synthesis quality                     | ✓    |
| Self-Supervision Improves Diffusion Models for Tabular Data Imputation                                                     | 2024 | CIKM      | Self-supervised diffusion           | Imputation accuracy                   | ✓    |
| FinDiff: Diffusion Models for Financial Tabular Data Generation                                                          | 2023 | ICAIF     | Diffusion-based                     | Fidelity, privacy, utility            | ✓    |
| Prototype-Oriented Hypergraph Representation Learning for Anomaly Detection in Tabular Data                                | 2025 | IPM       | Hypergraph representation           | Detection accuracy                    | ✓    |
| Retrieval Augmented Deep Anomaly Detection for Tabular Data                                                              | 2024 | CIKM      | Retrieval-augmented                 | Detection performance                 | ✕    |

<p align="center"><b align="center">TABLE 9: Performance comparison of key TAD models on 47 real-world tabular datasets, including domains such as healthcare, image processing, and finance.</b></p>

| Paper Title                                                                                                                         | Year | Venue            | AUC (%) ± Std. Dev.    | Mean Rank | p-value  |
|-------------------------------------------------------------------------------------------------------------------------------------|------|------------------|------------------------|-----------|----------|
| Neural Transformation Learning for Deep Anomaly Detection beyond Images                                                             | 2021 | ICML             | 71.45 ± 22.6           | 9.16      | 0        |
| Anomaly Detection for Tabular Data with Internal Contrastive Learning                                                                 | 2022 | ICLR             | 69.15 ± 15.3           | 6         | 0.0002   |
| ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions                                               | 2023 | IEEE TKDE        | 52.77 ± 11.7           | 10.12     | 0        |
| Perturbation Learning Based Anomaly Detection                                                                                         | 2022 | NeurIPS          | 67.42 ± 19.6           | 9.16      | 0        |
| D-PAD: Deep-Shallow Multi-Frequency Patterns Disentangling for Time Series Forecasting                                                 | 2024 | Arxiv            | 88.05 ± 12.4           | 4.96      | 0.0003   |
| AutoUAD: Hyper-Parameter Optimization for Unsupervised Anomaly Detection                                                                | 2024 | Arxiv            | 92.68 ± 11.8           | 2.04      | 0.47     |
| Unsupervised Anomaly Detection for Tabular Data Using Noise Evaluation                                                                  | 2025 | AAAI             | 92.27 ± 11.1           | 1.68      | 0.47     |

Notes: AUC (%) denotes the average area under the ROC curve over 47 datasets; Mean Rank is the average ranking position of the method; p-value indicates the statistical significance.

## Multimodal Anomaly Detection

<p align="center"><b align="center">TABLE 10: Summary of MAD methods on different datasets.</b></p>

| Paper Title                                                                                                                               | Year | Venue   | Datasets                                     | DM  | Code                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------|------|---------|----------------------------------------------|-----|--------------------------------------------------------|
| Exploiting Multimodal Latent Diffusion Models for Accurate Anomaly Detection in Industry 5.0                                               | 2024 | Ital-IA | KSDD2                                        | ✓   | -                                                      |
| Counterfactual Condition Diffusion with Continuous Prior Adaptive Correction for Anomaly Detection in Multimodal Brain MRI                 | 2024 | ESWA    | Brain glioma datasets                        | ✓   | [Link](https://github.com/Snow1949/Ano-cDiff)           |
| Multimodal Motion Conditioned Diffusion Model for Skeleton-Based Video Anomaly Detection                                                   | 2023 | ICCV    | UBnormal, HR-UBnormal, etc.                   | ✓   | -                                                      |
| AnomalyXFusion: Multi-Modal Anomaly Synthesis with Diffusion                                                                              | 2024 | ArXiv   | MVTec AD, LOCO, MVTec Caption                 | ✓   | [Link](http://github.com/hujiecpp/MVTec-Caption)        |
| Collaborative Diffusion for Multi-Modal Face Generation and Editing                                                                     | 2023 | CVPR    | CelebAMask-HQ, CelebA-Dialog                  | ✓   | -                                                      |


## Anomaly Generation

<p align="center"><b align="center">TABLE 11: Summary of AG methods across various domains with implementations</b></p>

| Paper Title                                                                                                                           | Year | Venue   | Domain                                   | DM  | Code                                                   |
|---------------------------------------------------------------------------------------------------------------------------------------|------|---------|------------------------------------------|-----|--------------------------------------------------------|
| HumanRefiner: Benchmarking Abnormal Human Generation and Refining with Coarse-to-Fine Pose-Reversible Guidance                      | 2024 | ECCV    | Human image generation                   | ✓   | [Link](https://github.com/Enderfga/HumanRefiner)         |
| AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model                                                              | 2024 | AAAI    | Industrial inspection                    | ✓   | [Link](https://github.com/sjtuplayer/anomalydiffusion)   |
| DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation                                                    | 2024 | ArXiv   | Industrial inspection                    | ✓   | [Link](https://doi.org/10.48550/arXiv.2408.13509)         |
| A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation                         | 2024 | ArXiv   | Industrial defect generation             | ✓   | [Link](https://github.com/GrandpaXun242/AdaBLDM.git)     |
| CUT: A Controllable, Universal, and Training-Free Visual Anomaly Generation Framework                                                  | 2024 | ArXiv   | Visual anomaly detection                 | ✓   | [Link](https://doi.org/10.48550/arXiv.2406.01078)         |
| Video Anomaly Detection via Spatio-Temporal Pseudo-Anomaly Generation: A Unified Approach                                             | 2024 | CVPR    | Video anomaly detection                  | ✓   | -                                                      |
| Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization                                                     | 2022 | ECCV    | Manufacturing, Medical imaging           | ✕   | [Link](https://github.com/hmsch/natural-synthetic-anomalies) |
| CutPaste: Self-Supervised Learning for Anomaly Detection and Localization                                                              | 2021 | CVPR    | Industrial inspection                    | ✕   | -                                                      |
| Prototypical Residual Networks for Anomaly Detection and Localization                                                                  | 2023 | CVPR    | Industrial manufacturing                 | ✕   | -                                                      |
| Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images                                                   | 2024 | IEEE TMI| Medical imaging (Brain)                  | ✓   | -                                                      |
| FinDiff: Diffusion Models for Financial Tabular Data Generation                                                                        | 2023 | ACM ICAIF | Financial data                          | ✓   | -                                                      |
| NetDiffus: Network Traffic Generation by Diffusion Models through Time-Series Imaging                                                   | 2023 | Arxiv   | Network traffic analysis                 | ✓   | -                                                      |

<p align="center"><b align="center">TABLE 12: Performance comparison of key AG models on the MVTec dataset using IS and IC-LPIPS metrics.</b></p>

| Paper Title                                                                                                                        | Year | Venue   | IS ↑  | IC-L ↑ |
|------------------------------------------------------------------------------------------------------------------------------------|------|---------|-------|--------|
| Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection                                                          | 2021 | WACV    | 1.69  | 0.15   |
| Defect Image Sample Generation with GAN for Improving Defect Recognition                                                            | 2020 | IEEE TASE | 1.71  | 0.13   |
| Differentiable Augmentation for Data-Efficient GAN Training                                                                          | 2020 | NeurIPS | 1.58  | 0.09   |
| Few-Shot Defect Segmentation Leveraging Abundant Defect-Free Training Samples through Normal Background Regularization and Crop-and-Paste Operation | 2021 | IEEE ICME | 1.51  | 0.14   |
| Few-Shot Defect Image Generation via Defect-Aware Feature Manipulation                                                               | 2023 | AAAI    | 1.72  | 0.20   |
| Few-Shot Image Generation via Cross-Domain Correspondence                                                                            | 2021 | CVPR    | 1.65  | 0.07   |
| AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model                                                             | 2024 | AAAI    | 1.80  | 0.32   |

Notes: IS denotes Inception Score and IC-L denotes Inception-based LPIPS; higher values indicate better performance.


## Performance Evaluation

### Evaluation Metrics

<p align="center"> <img src="https://github.com/user-attachments/assets/c4ac692b-aedc-46ce-8226-6dc241a32e61" alt=""> </p>
<p align="center"><b align="center">Fig. 11: Comprehensive evaluation metrics for ADGDM across different data modalities.</b></p>

### Public Datasets

<p align="center"><b align="center">TABLE 13: Summary of benchmark datasets for IAD, TSAD, TAD, and VAD across industrial, medical, surveillance, and cybersecurity domains.</b></p>

| Paper Title                                                                                                                             | Task | Year | Venue       | Real/Synth. | #Samples      | #Subjects | Domain                             | HomePage                                                                                                               |
|-----------------------------------------------------------------------------------------------------------------------------------------|------|------|-------------|-------------|---------------|-----------|------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| MVTec AD --- a Comprehensive Real-World Dataset for Unsupervised Anomaly Detection                                                       | IAD  | 2019 | CVPR        | Real        | 5,354         | 15        | Industrial anomaly detection       | [Link](https://www.mvtec.com/company/research/datasets/mvtec-ad/)                                                      |
| The Brain Tumor Segmentation (BraTS) Challenge 2023: Focus on Pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs)                       | IAD  | 2020 | -           | Real        | 5,000+        | 1,250+    | Medical Imaging (Brain Tumor)      | [Link](https://www.med.upenn.edu/cbica/brats2020/data.html)                                                            |
| The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization                                                           | IAD  | 2021 | IJCVIPA     | Real        | 4,433         | 10        | Industrial anomaly detection       | [Link](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)                                                      |
| Deep Learning-Based Defect Detection of Metal Parts: Evaluating Current Methods in Complex Conditions                                      | IAD  | 2021 | ICUMT       | Real        | 4,568         | 3         | Metal part defect detection        | [Link](https://github.com/stepanje/MPDD)                                                                               |
| VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization                                                        | IAD  | 2021 | ISIE        | Real        | 2,830         | 3         | Industrial Inspection              | [Link](https://github.com/pankajmishra000/VT-ADL)                                                                      |
| Mixed Supervision for Surface-Defect Detection: From Weakly to Fully Supervised Learning                                                  | IAD  | 2021 | Comput. Ind.| Real        | 3,420         | 1         | Industrial (Surface) Inspection    | [Link](https://www.vicos.si/resources/kolektorsdd2/)                                                                 |
| Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization                                        | IAD  | 2022 | IJCV        | Real        | 1,772         | 5         | Industrial anomaly detection       | [Link](https://www.mvtec.com/company/research/datasets/mvtec-loco)                                                      |
| SPot-the-Difference Self-Supervised Pre-Training for Anomaly Detection and Segmentation                                                  | IAD  | 2022 | ECCV        | Real        | 10,821        | 12        | Visual anomaly detection           | [Link](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)                                    |
| GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection                     | IAD  | 2024 | ECCV        | Real        | -             | -         | Circuit board defect detection     | [Link](https://github.com/SSRheart/industrial-anomaly-detection-dataset)                                               |
| A Dataset to Support Research in the Design of Secure Water Treatment Systems                                                            | TSAD | 2016 | CRITIS      | Real        | 950,000       | 1         | Industrial Control Systems         | [Link](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)                                                    |
| Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding                                                        | TSAD | 2018 | KDD         | Real        | 73,729        | 27        | Aerospace Telemetry                | [Link](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)                           |
| Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding                                                        | TSAD | 2018 | KDD         | Real        | 427,617       | 55        | Satellite Monitoring               | [Link](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)                           |
| Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network                                          | TSAD | 2019 | KDD         | Real        | 20M           | 28        | AIOps Server Monitoring            | [Link](https://github.com/smallcowbaby/OmniAnomaly)                                                                    |
| Anomaly Detection in Crowded Scenes                                                                                                      | VAD  | 2010 | CVPR        | Real        | 28            | -         | Video Surveillance                 | [Link](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)                                                          |
| Abnormal Event Detection at 150 FPS in MATLAB                                                                                            | VAD  | 2013 | ICCV        | Real        | 30,652        | 16        | Crowd Behavior                     | [Link](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)                                          |
| Single-Image Crowd Counting via Multi-Column Convolutional Neural Network                                                                  | VAD  | 2016 | CVPR        | Real        | 1,198         | 13        | Crowd Counting                     | [Link](https://github.com/desenzhou/ShanghaiTechDataset)                                                               |
| Real-World Anomaly Detection in Surveillance Videos                                                                                      | VAD  | 2018 | CVPR        | Real        | 1,900         | 13        | Surveillance Video                 | [Link](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)                           |
| Cross-Task Weakly Supervised Learning from Instructional Videos                                                                            | VAD  | 2019 | CVPR        | Real        | 4,700         | 83        | Weakly Supervised AD               | [Link](https://github.com/DmZhukov/CrossTask)                                                                          |
| COIN: A Large-Scale Dataset for Comprehensive Instructional Video Analysis                                                                 | VAD  | 2019 | CVPR        | Real        | 11,827        | 180       | Instructional Video Analysis       | [Link](https://coin-dataset.github.io/)                                                                               |
| UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection                                                                    | VAD  | 2022 | CVPR        | Synth       | 236,902       | 29        | Open-set Video                     | [Link](https://github.com/lilygeorgescu/UBnormal/)                                                                     |
| Fault Detection and Diagnosis in Industrial Systems                                                                                      | TAD  | 1993 | Springer    | Synth       | -             | 21        | Industrial Process                 | [Link](https://web.mit.edu/braatzgroup/links.html)                                                                    |
| An Extensive Reference Dataset for Fault Detection and Identification in Batch Processes                                                 | TAD  | 2016 | CILS        | Synth       | 500           | 5         | Chemical Process                   | [Link](https://cit.kuleuven.be/biotec/software/batchbenchmark)                                                         |
| ADBench: Anomaly Detection Benchmark                                                                                                     | TAD  | 2022 | NeurlPS     | Real        | 57            | -         | Tabular AD                         | [Link](https://github.com/Minqi824/ADBench)                                                                            |
