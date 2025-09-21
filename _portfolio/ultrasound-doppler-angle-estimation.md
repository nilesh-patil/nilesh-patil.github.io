---
title: "Ultrasound Doppler angle from B-mode with deep learning"
collection: portfolio
permalink: /portfolio/ultrasound-doppler-angle-estimation/
date: 2021-06-12
excerpt: "An end-to-end deep-learning project that reads the Doppler angle from a single grayscale carotid B-mode image — built on my EMBC 2019 paper and carried well past it: the grid-pooling insight that makes a frozen backbone work, Optuna-tuned best estimators under two sampling protocols (down to ~2° MAE), and a clinical-grade calibrated evaluation."
repo: https://github.com/nilesh-patil/ultrasound-doppler-angle-estimation
header:
  teaser: /images/blog/ultrasound-doppler-angle/header.png
  overlay_image: /images/blog/ultrasound-doppler-angle/header.png
  overlay_filter: "rgba(15, 23, 34, 0.55)"
tags: [deep-learning, medical-imaging, ultrasound, keras, jax, reproducibility]
math: true
---

## Problem

Spectral-Doppler velocity depends on the beam-to-vessel angle $\theta$ through $f_d = 2 f_0 v \cos\theta / c$, and angle correction is set by hand. I first-authored [Patil & Anand (EMBC 2019)](https://doi.org/10.1109/EMBC.2019.8857587), where a convolutional network learns $\theta$ directly from a single grayscale B-mode carotid image — no color Doppler, no segmentation. This project is that pipeline rebuilt from scratch on modern infra and carried end to end: *why* a frozen backbone works at all, how far the estimator climbs once it is tuned, and what a clinic would still need. The full interactive write-up — overview, method, results, a clinical evaluation, and a live beam-angle demo — lives on the **[project site](https://www.nilesh42.science/ultrasound-doppler-angle-estimation/)**.

## Approach

- **One typed, test-first library** (Keras 3 / JAX, `pixi`), with the model written once and the backend chosen per machine.
- **Orientation-preserving grid pooling** instead of global average pooling (global pooling is partly rotation-invariant — wrong for an orientation target). This is the load-bearing design choice that makes a *frozen* backbone work at all.
- **Two sampling protocols behind a config flag**: *image-level* sampling (the paper's standard augmented-corpus protocol) and *patient-level* sampling (cross-subject, holding out whole volunteers) — two complementary lenses, each reported and each tuned to its own best.
- **Optuna TPE** hyperparameter search against cached frozen features (each trial a shallow head fit; one extraction per backbone serves both protocols), then a stacked ensemble of the tuned backbones.
- **Clinical-grade, post-hoc evaluation**, all Keras-free: split-conformal intervals, Bland–Altman, calibration curves, patient-level nested CV, test-time augmentation, a classical structure-tensor prior + fusion, and Grad-CAM.
- Every figure is regenerated from `results/` by script; the whole thing is reproducible with `pixi run all`.

## Headline results

- **The core model:** a frozen DenseNet201 + grid pooling lands at **5.84% MAPE** (3.77° MAE), the paper's best single-model regime — and it's the pooling, not the backbone, that gets there (grid pooling lifts the frozen model from ~14% to 5.84%).
- **Best estimator, image-level sampling:** an Optuna-tuned 5-model ensemble reaches **2.79% MAPE / 1.96° MAE** ($R^2$ 0.995) — better than the paper's best single model.
- **Best estimator, patient-level sampling:** the tuned ensemble reaches **8.53% MAPE / 5.93° MAE** ($R^2$ 0.952) on the stricter cross-subject regime.
- **Architecture bake-off:** frozen DenseNet201 beats ConvNeXt and EfficientNetV2 — newer is not better for small-data frozen transfer.
- **Clinical-grade:** split-conformal 90% intervals of **±20.5°** at **95.2%** coverage; on Bland–Altman the model reads about **4.3° below** the *single* reference reading (method-vs-reference, **not** inter-observer — honestly flagged); test-time augmentation cuts per-image MAE **7.8° → 4.7°**.
- **Honest about the ceiling:** end-to-end fine-tuning and modern self-supervised encoders (DINOv2, USFM) are deferred to a CUDA box — documented, not hidden.

## Links

- **Project site**: [the interactive write-up](https://www.nilesh42.science/ultrasound-doppler-angle-estimation/) — overview, method, results, clinical, and reproduce pages, with a live beam-angle demo.
- **Repo**: [github.com/nilesh-patil/ultrasound-doppler-angle-estimation](https://github.com/nilesh-patil/ultrasound-doppler-angle-estimation)
- **Blog post**: [Reading the Doppler angle off a single B-mode image]({{ '/posts/ultrasound-doppler-angle-deep-learning/' | relative_url }}) — the full write-up, the pooling insight, and the figures.
- **Prior work**: [EMBC 2019 paper](https://doi.org/10.1109/EMBC.2019.8857587) · extended preprint [arXiv:2508.04243](https://arxiv.org/abs/2508.04243).
