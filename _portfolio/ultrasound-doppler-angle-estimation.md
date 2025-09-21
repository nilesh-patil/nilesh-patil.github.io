---
title: "Ultrasound Doppler angle from B-mode with deep learning"
collection: portfolio
permalink: /portfolio/ultrasound-doppler-angle-estimation/
date: 2021-06-12
excerpt: "A deep-learning project that reads the Doppler angle from a single grayscale carotid B-mode image. It extends my EMBC 2019 paper with grid pooling on a frozen backbone, Optuna-tuned estimators under two sampling protocols (down to ~2° MAE), and a calibrated clinical evaluation."
repo: https://github.com/nilesh-patil/ultrasound-doppler-angle-estimation
header:
  teaser: /images/blog/ultrasound-doppler-angle/header.png
  overlay_image: /images/blog/ultrasound-doppler-angle/header.png
  overlay_filter: "rgba(15, 23, 34, 0.55)"
tags: [deep-learning, medical-imaging, ultrasound, keras, jax, reproducibility]
math: true
---

## Problem

Spectral-Doppler velocity depends on the beam-to-vessel angle $\theta$ through $f_d = 2 f_0 v \cos\theta / c$, and angle correction is set by hand. I first-authored [Patil & Anand (EMBC 2019)](https://doi.org/10.1109/EMBC.2019.8857587), where a convolutional network learns $\theta$ directly from a single grayscale B-mode carotid image, with no color Doppler and no segmentation. This project rebuilds that pipeline from scratch on current tooling and goes after three questions: why a frozen backbone works here, how accurate the estimator gets once it is tuned, and what a clinic would still need. The interactive write-up, with a live beam-angle demo, is on the **[project site](https://www.nilesh42.science/ultrasound-doppler-angle-estimation/)**.

## Approach

- **One typed, test-first library** (Keras 3 / JAX, `pixi`), with the model written once and the backend chosen per machine.
- **Orientation-preserving grid pooling** instead of global average pooling. Global pooling is partly rotation-invariant, which throws away the signal an orientation target needs. Without this change the *frozen* backbone does not work.
- **Two sampling protocols behind a config flag**: *image-level* sampling (the paper's standard augmented-corpus protocol) and *patient-level* sampling (cross-subject, holding out whole volunteers). I report both and tune each separately.
- **Optuna TPE** hyperparameter search against cached frozen features (each trial a shallow head fit; one extraction per backbone serves both protocols), then a stacked ensemble of the tuned backbones.
- **Post-hoc clinical evaluation**, all Keras-free: split-conformal intervals, Bland-Altman, calibration curves, patient-level nested CV, test-time augmentation, a classical structure-tensor prior + fusion, and Grad-CAM.
- Every figure is regenerated from `results/` by script, and `pixi run all` reproduces the whole project.

## Headline results

- **The core model:** a frozen DenseNet201 + grid pooling lands at **5.84% MAPE** (3.77° MAE), the paper's best single-model regime. The pooling is responsible: it takes the same frozen model from ~14% to 5.84%.
- **Best estimator, image-level sampling:** an Optuna-tuned 5-model ensemble reaches **2.79% MAPE / 1.96° MAE** ($R^2$ 0.995), better than the paper's best single model.
- **Best estimator, patient-level sampling:** the tuned ensemble reaches **8.53% MAPE / 5.93° MAE** ($R^2$ 0.952) on the stricter cross-subject regime.
- **Architecture bake-off:** frozen DenseNet201 beats ConvNeXt and EfficientNetV2 on this small dataset.
- **Clinical evaluation:** split-conformal 90% intervals of **±20.5°** at **95.2%** coverage. On Bland-Altman the model reads about **4.3° below** the reference, which is a *single* reading, so this measures method against reference and says nothing about inter-observer agreement. Test-time augmentation cuts per-image MAE **7.8° → 4.7°**.
- **Not done:** end-to-end fine-tuning and self-supervised encoders (DINOv2, USFM) need a CUDA machine I did not have for this run. The repo documents both as open items.

## Links

- **Project site**: [the interactive write-up](https://www.nilesh42.science/ultrasound-doppler-angle-estimation/), with overview, method, results, clinical, and reproduce pages, and a live beam-angle demo.
- **Repo**: [github.com/nilesh-patil/ultrasound-doppler-angle-estimation](https://github.com/nilesh-patil/ultrasound-doppler-angle-estimation)
- **Blog post**: [Grid pooling on a frozen backbone - Doppler angle estimation from B-mode ultrasound]({{ '/posts/ultrasound-doppler-angle-deep-learning/' | relative_url }}), with the pooling analysis and the figures.
- **Prior work**: [EMBC 2019 paper](https://doi.org/10.1109/EMBC.2019.8857587) · extended preprint [arXiv:2508.04243](https://arxiv.org/abs/2508.04243).
