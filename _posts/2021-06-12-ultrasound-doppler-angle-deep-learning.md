---
layout: single
title: "Grid pooling on a frozen backbone - Doppler angle estimation from B-mode ultrasound"
date: 2021-06-12T20:00:00+05:30
last_modified_at: 2026-01-25T20:00:00+05:30
categories: [blog]
tags: [deep-learning, medical-imaging, ultrasound, keras, jax, reproducibility]
excerpt: "My EMBC 2019 paper learned the carotid Doppler angle from one grayscale B-mode frame. This is the project rebuilt end to end in Keras 3 / JAX: one pooling choice decides whether a frozen backbone works at all, and a tuned ensemble reads the angle to about 2 degrees within-population, with calibrated bands for the clinic."
math: true
header:
  overlay_image: /images/blog/headers/ultrasound-doppler-angle-deep-learning.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/ultrasound-doppler-angle-deep-learning.jpg
---

<p class="post-meta-note">Published June 2021 &middot; revised January 2026 for a from-scratch re-run in Keras&nbsp;3 / JAX. The tooling described below (Keras&nbsp;3, JAX, Optuna, split-conformal intervals) reflects that 2026 revision, not what was on hand in 2021.</p>

<div class="post-links" markdown="1">
[Code](https://github.com/nilesh-patil/ultrasound-doppler-angle-estimation) &middot; [Preprint](https://arxiv.org/abs/2508.04243) &middot; [EMBC 2019 paper](https://doi.org/10.1109/EMBC.2019.8857587) &middot; [Project site]({{ site.baseurl }}/ultrasound-doppler-angle-estimation/)
</div>

## A few degrees decide the diagnosis

A spectral-Doppler study reports a blood velocity. That number is only as good as one angle a sonographer sets by hand on every single exam: the angle between the ultrasound beam and the direction of flow. Get it a few degrees off near the steep part of the curve and the reported velocity is wrong by tens of percent, enough to push a borderline carotid stenosis into the wrong grade. It is an operator-dependent source of velocity error. Angle correction is one of the things vascular-lab accreditation reviews scrutinize.

The cost of that hand-set angle is not linear. Blood velocity comes out of the Doppler equation,

$$f_d = \frac{2 f_0 v \cos\theta}{c}$$

where $\theta$ is the angle between the ultrasound beam and the direction of flow. The $\cos\theta$ sits on top of the velocity, so a small error in $\theta$ rides straight through: $v$ scales as $1/\cos\theta$, and the fractional velocity error per unit angle error is $\tan\theta$, which is why it explodes past 70 degrees. The sonographer sets that angle by hand on the scanner. A degree or two of angle error sounds negligible. Move the angle yourself and watch it stop being negligible.

<figure>
 <svg id="dop-svg" viewBox="0 0 640 320" role="img" aria-labelledby="dop-t dop-d" style="width:100%;height:auto;max-width:640px;color:inherit" xmlns="http://www.w3.org/2000/svg">
 <title id="dop-t">How reported velocity scales with the Doppler angle</title>
 <desc id="dop-d">Left: a carotid vessel drawn as a horizontal lozenge with an ultrasound beam crossing it at an angle theta. Right: the curve of velocity multiplier one over cosine theta, with a moving dot at the current angle and a dashed reference line at sixty degrees. As theta rises toward seventy to seventy-eight degrees the curve climbs steeply, so the reported velocity runs away from the true value.</desc>
 <g stroke="currentColor" stroke-width="1.4" fill="none">
 <!-- vessel lozenge -->
 <rect x="34" y="150" width="210" height="36" rx="18" opacity="0.55"></rect>
 <!-- beam line, rotated about pivot (244,168) via JS -->
 <line id="dop-beam" x1="244" y1="168" x2="120" y2="46"></line>
 <!-- flow axis reference -->
 <line x1="34" y1="168" x2="244" y2="168" stroke-dasharray="4 4" opacity="0.45"></line>
 <!-- angle arc -->
 <path id="dop-arc" d="" opacity="0.7"></path>
 <!-- plot axes -->
 <line x1="330" y1="288" x2="624" y2="288" opacity="0.6"></line>
 <line x1="330" y1="40" x2="330" y2="288" opacity="0.6"></line>
 <!-- reference dashed line at theta = 60 -->
 <line id="dop-ref" x1="330" y1="0" x2="624" y2="0" stroke-dasharray="5 4" opacity="0.4"></line>
 <!-- the 1/cos curve -->
 <path id="dop-curve" d="" stroke-width="1.8"></path>
 </g>
 <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif" aria-hidden="true">
 <text id="dop-theta-lbl" x="150" y="120" font-size="13" font-style="italic" text-anchor="middle">&#952;</text>
 <text x="139" y="200" font-size="11" opacity="0.7" text-anchor="middle">flow axis</text>
 <text x="139" y="142" font-size="11" opacity="0.7" text-anchor="middle">beam</text>
 <text x="477" y="306" font-size="11" opacity="0.7" text-anchor="middle">beam-to-flow angle &#952; (degrees)</text>
 <text x="318" y="48" font-size="11" opacity="0.7" text-anchor="end">5&#215;</text>
 <text x="318" y="288" font-size="11" opacity="0.7" text-anchor="end">1&#215;</text>
 </g>
 </svg>
 <div class="dop-controls" style="margin:0">
 <label for="dop-range" style="display:block;font-size:0.85rem;opacity:0.75;margin:0.2rem 0">Drag the beam-to-flow angle</label>
 <input id="dop-range" type="range" min="30" max="78" step="1" value="60" style="width:100%;max-width:640px;accent-color:var(--global-base-color)">
 <p id="dop-readout" style="font-size:0.95rem;margin:0.35rem 0 0" aria-live="polite">&#952; = 60&deg; &nbsp;|&nbsp; velocity &#215;2.00 &nbsp;|&nbsp; error vs 60&deg; reference = +0%</p>
 </div>
 <figcaption>Drag toward 60-70 degrees and watch the dot climb the curve as reported velocity runs away from the true value. Near 60 degrees each extra degree moves reported velocity by roughly 3%. Past 70 the curve goes near-vertical, which is why vascular labs are told to keep $\theta$ at or below 60.</figcaption>
</figure>

<div style="overflow-x:auto" markdown="1">

| $\theta$ | velocity multiplier $\frac{1}{cos\ \theta}$ | error vs the 60&deg; reading |
|---|---|---|
| 45&deg; | 1.41&times; | &minus;29% |
| 60&deg; | 2.00&times; | 0% |
| 70&deg; | 2.92&times; | +46% |
| 80&deg; | 5.76&times; | +188% |

</div>

Near 60 degrees, each extra degree of angle moves the reported velocity by about 3%. So a model that reads the angle off the image has to be accurate to within a degree or so to matter at all. That is the bar.

Read the angle off the image instead of asking the operator to set it. [Patil &amp; Anand (EMBC 2019)](https://doi.org/10.1109/EMBC.2019.8857587) showed a convolutional network can regress the Doppler angle straight from a single grayscale B-mode frame of the carotid, with no color Doppler, no segmentation, and no hand-placed landmarks, and reported under 3&deg; mean error. I first-authored that paper. Seven years on, I rebuilt the pipeline from scratch on modern infra and tooling. I carried it end to end, past where the original two pages had to stop.

What follows is that build, stage by stage, with each stage a labeled component bolted onto a running pipeline: the data and labels, the one pooling choice that makes a frozen backbone work at all, two evaluation protocols, how far tuning and ensembling climb, and what a clinic would still need. Three questions anchor it. Does a from-scratch build hold the paper's accuracy? *Why* does a frozen ImageNet backbone work at all on 84 images? And how far does the estimator go once it is tuned and calibrated? Short answers: yes, once the pooling is fixed and the head is tuned; because the vessel's orientation is already sitting in the frozen features if you do not average it away; and down to about 2&deg; within-population, with calibrated intervals for the clinic. The interactive write-up lives on the [project site]({{ site.baseurl }}/ultrasound-doppler-angle-estimation/). The rest of this post walks the build that gets there.

<div class="notice--info" markdown="1">
**Results at a glance.** 84 carotid images, about 10 volunteers, one scanner; Apple M4 Max; Keras 3 / JAX; built test-first. *Image-level* holds out random augmented rows (interpolating across orientations); *patient-level* holds out whole volunteers (cross-subject).

<div style="overflow-x:auto" markdown="1">

| | |
|---|---|
| The core model | A *frozen* DenseNet201 plus a small head lands at 5.84% MAPE (3.77&deg; MAE, $R^2$ 0.982) once the pooling is fixed, with no fine-tuning - close to the EMBC-2019 single model (4.03% / 2.87&deg;), which the tuned model below actually matches. |
| The pooling insight | Global average pooling is approximately rotation-invariant, the wrong bias for an orientation target. An orientation-preserving grid pooling head lifts the same frozen backbone from about 14% to 5.84% MAPE with no fine-tuning (single-split figures; the matched cross-validated pair is 10.85% GAP vs 4.58% grid image-level, in the figure below). |
| Best estimator | An Optuna-tuned 5-model stacked ensemble reaches **2.79% MAPE / 1.96&deg;** image-level (whole volunteers in play), and **8.53% / 5.93&deg;** patient-level (whole volunteers held out). |
| Clinical-grade | Split-conformal 90% bands of &plusmn;20.5&deg; at 95.2% coverage; on Bland-Altman the model reads about 4.3&deg; below the single reference reading; rotation test-time augmentation cuts base-image error 7.8&deg; &rarr; 4.7&deg;. |

</div>
</div>

## Labels from rotation {#labels}

The pipeline starts before the model. The cohort has no clinical angle annotation, so $\theta$ was hand-drawn once per image in a MATLAB GUI. That single reading per base image is the only ground truth. To turn 84 readings into a regression corpus, each base image is rotated through $[-60°, +60°]$ in 5-degree steps, 25 oriented views apiece, for $84 \times 25 = 2{,}100$ labelled views. The label of a rotated view is exact: the new angle is the base angle plus the rotation. Because the increment is exact, the network learns to read relative image orientation. The rotation sweep does double duty as augmentation and label source.

Those 2,100 views are not 2,100 independent observations. They are 25 geometrically coupled copies of 84 base images drawn from about 10 volunteers, a three-level hierarchy of volunteer, base image, and rotated view. That coupling matters later. It is why there are two ways to score the model.

## The pooling head {#pooling}

The whole project rests on one head choice, so look at it fail first. My first frozen-feature runs landed near 14% MAPE (mean absolute percentage error) on a single augmented split. The obvious read is that frozen features simply cannot reach fine-tuned accuracy, and that read is wrong.

<figure>
 <svg viewBox="0 0 680 300" role="img" aria-labelledby="gap-t gap-d" style="width:100%;height:auto;max-width:680px;color:inherit" xmlns="http://www.w3.org/2000/svg">
 <title id="gap-t">Why global average pooling destroys the orientation signal and grid pooling keeps it</title>
 <desc id="gap-d">A six by six feature map carries a bright diagonal stripe of activation standing for the vessel. Global average pooling collapses it to a single short row of bars that barely changes when the vessel rotates. Grid pooling instead averages onto a two by two block whose populated cells move to the mirror corner when the vessel rotates, so the layout still carries orientation.</desc>
 <defs>
 <marker id="gap-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
 <path d="M0,0 L10,5 L0,10 z" fill="currentColor"></path>
 </marker>
 </defs>
 <!-- feature map cells filled by JS -->
 <g id="gap-map" stroke="currentColor" stroke-width="0.8"></g>
 <!-- GAP bars filled by JS -->
 <g id="gap-bars" stroke="currentColor" stroke-width="0.8"></g>
 <!-- grid pooling 2x2 filled by JS -->
 <g id="gap-grid" stroke="currentColor" stroke-width="0.8"></g>
 <g stroke="currentColor" stroke-width="1.4" fill="none">
 <path d="M196,150 L236,150" marker-end="url(#gap-ah)"></path>
 <path d="M196,150 L236,150" marker-end="url(#gap-ah)" opacity="0"></path>
 <path d="M450,150 L490,150" marker-end="url(#gap-ah)"></path>
 </g>
 <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif" text-anchor="middle" aria-hidden="true">
 <text x="100" y="36" font-size="13" font-weight="600">feature map</text>
 <text x="100" y="278" font-size="11" opacity="0.7">vessel = diagonal stripe</text>
 <text x="338" y="36" font-size="13" font-weight="600">GAP</text>
 <text x="338" y="278" font-size="11" opacity="0.7">rotation-insensitive</text>
 <text x="585" y="36" font-size="13" font-weight="600">grid pooling (G&#215;G)</text>
 <text x="585" y="278" font-size="11" opacity="0.7">keeps coarse location</text>
 </g>
 </svg>
 <div class="gap-controls" style="margin:0">
 <button id="gap-btn" type="button" aria-pressed="false" style="font:inherit;font-size:0.9rem;padding:0.4em 0.9em;border-radius:8px;border:1px solid color-mix(in srgb, currentColor 30%, transparent);background:color-mix(in srgb, currentColor 6%, transparent);color:inherit;cursor:pointer">Rotate vessel</button>
 </div>
 <figcaption>Rotate the vessel and the GAP vector barely moves; the grid block rearranges, and that rearrangement is the orientation signal. Brighter cells mean stronger activation. Frozen DenseNet201: GAP scores 10.85% image-level MAPE, grid pooling 4.58%; patient-level, 18.70% versus 12.59%.</figcaption>
</figure>

The culprit was the pooling. Global average pooling collapses an $(H, W, C)$ feature map to a length-$C$ vector by averaging each channel over every spatial position; average out *where* a channel fired and you keep only *how much* it fired, which is roughly invariant to rotating the image. For classification you want that invariance. Here the target *is* the vessel's orientation, encoded in the spatial layout of the activations, so GAP averages away exactly the signal the model has to read. It was throwing out the answer.

Grid pooling keeps it. Instead of collapsing to one vector, average-pool the final feature map onto a coarse $G \times G$ grid, then flatten; now a top-left cell is no longer interchangeable with a bottom-right one, so the flattened vector carries coarse spatial location and the head can read orientation off it. On a single augmented split, frozen DenseNet201 with grid pooling reaches 5.84% MAPE, 3.77&deg; MAE, $R^2$ 0.982, with no fine-tuning at all - close to the EMBC-2019 best single model (about 2.87&deg; / 4.03%), though it does not match that number until the head is tuned.[^grid] The work is in not pooling it away. A frozen ImageNet backbone already sees the vessel orientation.

## Two ways to score {#two-protocols}

That core model now has a number on it. Which number depends on how you split the corpus. The hierarchy from the label step forces the question.

<figure>
 <svg viewBox="0 0 380 212" role="img" aria-labelledby="prot-t prot-d" style="width:100%;height:auto;max-width:420px;color:inherit" xmlns="http://www.w3.org/2000/svg">
 <title id="prot-t">Image-level versus patient-level splitting of the augmented corpus</title>
 <desc id="prot-d">The corpus is drawn as ten volunteer blocks, each holding a few base images and their rotation views. Under image-level splitting the held-out cells scatter across every block, including blocks that also contribute training rows. Under patient-level splitting whole volunteer blocks are held out, so no rotation of a test patient appears in training.</desc>
 <g id="prot-cells" stroke="currentColor" stroke-width="0.7"></g>
 <g id="prot-blocks" stroke="currentColor" stroke-width="1.2" fill="none"></g>
 <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif" aria-hidden="true">
 <text id="prot-vol-lbl" x="0" y="0" font-size="10.5" opacity="0.7" text-anchor="start"></text>
 </g>
 </svg>
 <div class="prot-controls" style="margin:0" role="radiogroup" aria-label="Splitting protocol">
 <button id="prot-btn-img" type="button" role="radio" aria-checked="true" style="font:inherit;font-size:0.9rem;padding:0.4em 0.9em;border-radius:8px 0 0 8px;border:1px solid color-mix(in srgb, currentColor 30%, transparent);background:color-mix(in srgb, currentColor 22%, transparent);color:inherit;cursor:pointer">Image-level split</button><button id="prot-btn-pat" type="button" role="radio" aria-checked="false" style="font:inherit;font-size:0.9rem;padding:0.4em 0.9em;border-radius:0 8px 8px 0;border:1px solid color-mix(in srgb, currentColor 30%, transparent);border-left:none;background:color-mix(in srgb, currentColor 6%, transparent);color:inherit;cursor:pointer">Patient-level split</button>
 <p id="prot-readout" style="font-size:0.95rem;margin:0.35rem 0 0" aria-live="polite">Image-level: held-out rows scatter across every volunteer, interpolating across orientations. Tuned ensemble: 2.79% MAPE / 1.96&deg;.</p>
 </div>
 <figcaption>Held-out cells are drawn at lower opacity with a dashed outline. Image-level scatters them everywhere, including inside blocks whose siblings are still in training; patient-level shades whole blocks out. Same corpus, two questions: the tuned ensemble reads 2.79% MAPE image-level and 8.53% patient-level. The ten blocks are schematic; the real patient split uses 12 time-clustered proxy groups.</figcaption>
</figure>

Image-level sampling is the paper's protocol: split at random over the 2,100 augmented rows, which measures how well the estimator interpolates across the full population of orientations. Patient-level sampling holds out whole volunteers using GroupKFold over patient id. The test anatomy is then unseen at any rotation. It is the stricter, cross-subject lens, and the one a clinic cares about.

That gap is the measurement. The tuned ensemble scores 2.79% image-level and 8.53% patient-level, about a 3-fold spread that quantifies how much of the within-population accuracy is anatomy-specific rather than general. One caveat stays attached. The patient groups are 12 time-clustered proxies recovered by clustering acquisition timestamps, not verified subject identities.[^groups]

## The climb: tuning, then ensembling {#climb}

With the core model fixed and both protocols in place, I went looking for accuracy. First, a reasonable thing to try and a quick negative result: newer encoders. They are not better here. A frozen-feature bake-off across the modern zoo, run at a common grid so the encoders compare like for like, leaves DenseNet201 (14.13% patient MAPE) below ConvNeXt-Base (15.65%), ConvNeXt-Tiny (16.07%), and the EfficientNet / V2 family (roughly 17-21%). That 14.13% is the common-grid bake-off entry; DenseNet201's own tuned 3&times;3 grid reads 12.59% patient-level in the pooling and climb figures, which is why its two patient numbers differ. With only 84 base images, the ranking is empirical: these frozen feature spaces happen to encode the orientation cue better, and the newer encoders do not buy anything here.

<figure>
 <svg viewBox="0 0 680 365" role="img" aria-labelledby="bake-t bake-d" style="width:100%;height:auto;max-width:680px;color:inherit" xmlns="http://www.w3.org/2000/svg">
 <title id="bake-t">Frozen-backbone bake-off: patient-level MAPE per encoder</title>
 <desc id="bake-d">Horizontal bar chart of patient-level five-fold MAPE for eleven frozen backbones, lower is better. DenseNet201 is shortest at 14.13 percent and is highlighted; ConvNeXt-Base 15.65, ConvNeXt-Tiny 16.07, and the EfficientNet and EfficientNetV2 family sit between 17 and 21 percent.</desc>
 <line x1="168" y1="42" x2="168" y2="319" stroke="currentColor" stroke-width="1" opacity="0.35"></line>
 <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif">
 <text x="160" y="61.0" font-size="11.5" text-anchor="end" font-weight="600">DenseNet201</text>
 <rect x="168" y="50" width="301.9" height="15" rx="2" fill="currentColor" fill-opacity="0.85"></rect>
 <text x="475.9" y="61.0" font-size="11" opacity="0.75">14.13</text>
 <text x="160" y="86.0" font-size="11.5" text-anchor="end">ConvNeXt-Base</text>
 <rect x="168" y="75" width="334.3" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="508.3" y="86.0" font-size="11" opacity="0.75">15.65</text>
 <text x="160" y="111.0" font-size="11.5" text-anchor="end">ConvNeXt-Tiny</text>
 <rect x="168" y="100" width="343.3" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="517.3" y="111.0" font-size="11" opacity="0.75">16.07</text>
 <text x="160" y="136.0" font-size="11.5" text-anchor="end">EfficientNetV2-B1</text>
 <rect x="168" y="125" width="368.1" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="542.1" y="136.0" font-size="11" opacity="0.75">17.23</text>
 <text x="160" y="161.0" font-size="11.5" text-anchor="end">EfficientNetV2-B2</text>
 <rect x="168" y="150" width="383.3" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="557.3" y="161.0" font-size="11" opacity="0.75">17.94</text>
 <text x="160" y="186.0" font-size="11.5" text-anchor="end">EfficientNet-B3</text>
 <rect x="168" y="175" width="384.8" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="558.8" y="186.0" font-size="11" opacity="0.75">18.01</text>
 <text x="160" y="211.0" font-size="11.5" text-anchor="end">EfficientNetV2-B0</text>
 <rect x="168" y="200" width="391.2" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="565.2" y="211.0" font-size="11" opacity="0.75">18.31</text>
 <text x="160" y="236.0" font-size="11.5" text-anchor="end">EfficientNet-B0</text>
 <rect x="168" y="225" width="393.1" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="567.1" y="236.0" font-size="11" opacity="0.75">18.40</text>
 <text x="160" y="261.0" font-size="11.5" text-anchor="end">EfficientNet-B1</text>
 <rect x="168" y="250" width="412.7" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="586.7" y="261.0" font-size="11" opacity="0.75">19.32</text>
 <text x="160" y="286.0" font-size="11.5" text-anchor="end">EfficientNetV2-B3</text>
 <rect x="168" y="275" width="417.2" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="591.2" y="286.0" font-size="11" opacity="0.75">19.53</text>
 <text x="160" y="311.0" font-size="11.5" text-anchor="end">EfficientNet-B2</text>
 <rect x="168" y="300" width="454.0" height="15" rx="2" fill="currentColor" fill-opacity="0.32"></rect>
 <text x="628.0" y="311.0" font-size="11" opacity="0.75">21.25</text>
 <text x="168" y="347" font-size="11" opacity="0.7">patient-level MAPE (%), frozen features, 5-fold CV - lower is better</text>
 </g>
 </svg>
 <figcaption>DenseNet201 (highlighted) is the shortest bar; every ConvNeXt and EfficientNet/V2 encoder sits above it, even though all are far larger and more recent. With 84 base images, the older feature space simply encodes the orientation cue better.</figcaption>
</figure>

So the gains had to come from the head, not the encoder. An Optuna TPE search over the head and optimizer, run on cached frozen features, moves single-model DenseNet201 from 4.58 to 4.03% image-level and 12.59 to 10.80% patient-level in the search's own validation (10.14% when re-scored on pooled out-of-fold predictions, the protocol the figure below uses): a real but small gain, and 4.03% is where the from-scratch build finally matches the paper. These tuned single-model figures are in-sample to the search, so read them as optimistic. Each trial is a shallow head fit. One feature extraction per backbone serves both protocols, and no GPU is needed.

Stacking the five tuned backbones (DenseNet201, VGG19, ResNet50, Xception, InceptionV3) is where the accuracy actually moves. A Ridge meta-learner on pooled out-of-fold predictions reaches 2.79% / 1.96&deg; image-level and 8.53% / 5.93&deg; patient-level. That is the first configuration below 10% MAPE cross-patient. Read against its like-for-like predecessor, the single tuned DenseNet201 on pooled OOF (7.80&deg; / 10.14% patient-level), the genuine ensembling gain is about 1.9&deg;. The plain mean of the same five lands at 9.89% patient-level. Before tuning, that mean was a useless 21.9%. Tuning calibrated the members just enough for averaging to help at all.

<figure>
 <svg viewBox="0 0 600 300" role="img" aria-labelledby="climb-t climb-d" style="width:100%;height:auto;max-width:600px;color:inherit" xmlns="http://www.w3.org/2000/svg">
 <title id="climb-t">MAPE descending from the core model to tuned single model to stacked ensemble</title>
 <desc id="climb-d">Two descending lines on a MAPE axis across three stages. The solid line is image-level: core model 5.84, tuned 4.03, ensemble 2.79. The dashed line is patient-level: core model 12.59, tuned 10.14, ensemble 8.53. The patient line stays well above the image line, and the vertical gap at the ensemble stage is the roughly three-fold spread.</desc>
 <g stroke="currentColor" stroke-width="1" fill="none" opacity="0.5">
 <line x1="70" y1="250" x2="560" y2="250"></line>
 <line x1="70" y1="40" x2="70" y2="250"></line>
 </g>
 <!-- patient line (dashed): rep 12.59 tuned 10.14 ens 8.53 ; map MAPE 14->40px, 0->250px linear: y=250-(v/14)*210 -->
 <g stroke="currentColor" fill="none">
 <polyline points="160,61.1 340,97.9 520,122.0" stroke-width="2" stroke-dasharray="6 4"></polyline>
 <polyline points="160,162.4 340,189.5 520,208.2" stroke-width="2"></polyline>
 </g>
 <g fill="currentColor" stroke="none">
 <circle cx="160" cy="61.1" r="3.2"></circle><circle cx="340" cy="97.9" r="3.2"></circle><circle cx="520" cy="122.0" r="3.2"></circle>
 <circle cx="160" cy="162.4" r="3.2"></circle><circle cx="340" cy="189.5" r="3.2"></circle><circle cx="520" cy="208.2" r="3.2"></circle>
 </g>
 <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif" font-size="11.5" text-anchor="middle" aria-hidden="true">
 <text x="160" y="280">Core model</text>
 <text x="340" y="280">Tuned</text>
 <text x="520" y="280">Ensemble</text>
 <text x="160" y="50">12.59</text><text x="340" y="87">10.14</text><text x="520" y="111">8.53</text>
 <text x="160" y="178">5.84</text><text x="340" y="205">4.03</text><text x="520" y="224">2.79</text>
 <text x="540" y="116" font-size="10.5" opacity="0.75" text-anchor="start">patient</text>
 <text x="540" y="203" font-size="10.5" opacity="0.75" text-anchor="start">image</text>
 <text x="38" y="46" font-size="10.5" opacity="0.7" text-anchor="end">MAPE</text>
 <text x="520" y="146" font-size="10" opacity="0.75">below 10% cross-patient</text>
 </g>
 </svg>
 <figcaption>The dashed line is patient-level, the solid line image-level (lower is better). Tuning shaves a little; ensembling does most of the work, and the standing gap between the lines is the same 3-fold image-versus-patient spread.[^anchor]</figcaption>
</figure>

## A number needs a band {#clinical}

An accuracy number is not yet an instrument. A clinic acts on a number with a band. Three post-hoc checks on the held-out patient-level predictions say how far this is from usable.

Split-conformal calibration, with a patient-disjoint calibration set, gives a 90% band of &plusmn;20.5&deg; at 95.2% empirical coverage. That is wide, roughly 41 degrees of total width. It is the price of distribution-free validity on a cohort this size. A conformal interval covers the truth at a stated rate without assuming a noise model, and here coverage sits at or above nominal. So the band is not optimistic. That 95.2% is one seed-42 patient-disjoint split. The marginal-versus-conditional-coverage caveats live on the project site.

<figure>
 <svg viewBox="0 0 460 380" role="img" aria-labelledby="cal-t cal-d" style="width:100%;height:auto;max-width:460px;color:inherit" xmlns="http://www.w3.org/2000/svg">
 <title id="cal-t">Split-conformal calibration: empirical vs nominal coverage</title>
 <desc id="cal-d">Empirical coverage plotted against nominal level. The dashed diagonal is perfect calibration. Three points sit on or above it: nominal 80 gives 89.7 percent empirical, nominal 90 gives 95.2 percent, nominal 95 gives 97.8 percent, so the intervals are valid rather than optimistic.</desc>
 <rect x="74" y="44" width="346" height="282" fill="currentColor" opacity="0.04"></rect>
 <line x1="74.0" y1="326.0" x2="420.0" y2="44.0" stroke="currentColor" stroke-dasharray="5 4" opacity="0.5"></line>
 <polyline points="143.2,160.2 281.6,98.1 350.8,68.8" fill="none" stroke="currentColor" stroke-width="1.6" opacity="0.55"></polyline>
 <g stroke="currentColor">
 <line x1="74" y1="44" x2="74" y2="326" opacity="0.5"></line>
 <line x1="74" y1="326" x2="420" y2="326" opacity="0.5"></line>
 </g>
 <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif"><line x1="143.2" y1="326" x2="143.2" y2="330" stroke="currentColor" opacity="0.5"></line><text x="143.2" y="344" font-size="10.5" text-anchor="middle" opacity="0.7">80</text><line x1="212.4" y1="326" x2="212.4" y2="330" stroke="currentColor" opacity="0.5"></line><text x="212.4" y="344" font-size="10.5" text-anchor="middle" opacity="0.7">85</text><line x1="281.6" y1="326" x2="281.6" y2="330" stroke="currentColor" opacity="0.5"></line><text x="281.6" y="344" font-size="10.5" text-anchor="middle" opacity="0.7">90</text><line x1="350.8" y1="326" x2="350.8" y2="330" stroke="currentColor" opacity="0.5"></line><text x="350.8" y="344" font-size="10.5" text-anchor="middle" opacity="0.7">95</text><line x1="420.0" y1="326" x2="420.0" y2="330" stroke="currentColor" opacity="0.5"></line><text x="420.0" y="344" font-size="10.5" text-anchor="middle" opacity="0.7">100</text><line x1="70" y1="269.6" x2="74" y2="269.6" stroke="currentColor" opacity="0.5"></line><text x="66" y="273.1" font-size="10.5" text-anchor="end" opacity="0.7">80</text><line x1="70" y1="213.2" x2="74" y2="213.2" stroke="currentColor" opacity="0.5"></line><text x="66" y="216.7" font-size="10.5" text-anchor="end" opacity="0.7">85</text><line x1="70" y1="156.8" x2="74" y2="156.8" stroke="currentColor" opacity="0.5"></line><text x="66" y="160.3" font-size="10.5" text-anchor="end" opacity="0.7">90</text><line x1="70" y1="100.4" x2="74" y2="100.4" stroke="currentColor" opacity="0.5"></line><text x="66" y="103.9" font-size="10.5" text-anchor="end" opacity="0.7">95</text><line x1="70" y1="44.0" x2="74" y2="44.0" stroke="currentColor" opacity="0.5"></line><text x="66" y="47.5" font-size="10.5" text-anchor="end" opacity="0.7">100</text><circle cx="143.2" cy="160.2" r="4.2" fill="currentColor"></circle><text x="151.2" y="154.2" font-size="11" font-weight="600">89.7%</text><circle cx="281.6" cy="98.1" r="4.2" fill="currentColor"></circle><text x="289.6" y="92.1" font-size="11" font-weight="600">95.2%</text><circle cx="350.8" cy="68.8" r="4.2" fill="currentColor"></circle><text x="358.8" y="62.8" font-size="11" font-weight="600">97.8%</text>
 <text x="247" y="360" font-size="11.5" text-anchor="middle" opacity="0.8">nominal coverage (%)</text>
 <text x="18" y="185" font-size="11.5" text-anchor="middle" opacity="0.8" transform="rotate(-90 18 185)">empirical coverage (%)</text>
 <text x="171" y="292" font-size="10.5" opacity="0.6">diagonal = perfect calibration</text>
 </g>
 </svg>
 <figcaption>Every point sits on or above the diagonal: empirical coverage meets or beats the nominal level at 80, 90, and 95%. The bands are wide (the 90% band is &plusmn;20.5&deg;), and that width is the price of distribution-free validity on a cohort this size.</figcaption>
</figure>

Agreement comes from a Bland-Altman plot against the single MATLAB-GUI reference reading: the model reads roughly 4.3 degrees lower, a per-sample bias magnitude of about 4.3&deg;. With exactly one human reading per image, this is method-versus-reference, not inter-observer agreement, and I will not invent a second reader to make the plot look more clinical.

<figure>
 <img src="/images/blog/ultrasound-doppler-angle/figure_bland_altman.png" alt="Bland-Altman plot of model minus reference angle against their mean, with a small negative bias line and ninety-five percent limits of agreement." loading="lazy">
 <figcaption>The model reads about 4.3&deg; below the reference, with the 95% limits of agreement shown. One reference reading exists per image, so this is method-versus-reference.</figcaption>
</figure>

Rotation test-time augmentation buys accuracy without retraining. The 25 views of a frame get de-rotated back to base and reduced circularly (180-periodic, seam-safe so a straddle of the 0/180 wrap does not corrupt the average), which cuts base-image MAE from 7.80&deg; to 4.72&deg; with the circular median, about a 40% cut. It costs 25 forward passes per image. That 7.80&deg; base-image footing sits on a different split from the 5.93&deg; patient headline, so the two do not line up directly. The project site works it through.

<figure>
 <img src="/images/blog/ultrasound-doppler-angle/gradcam.png" alt="Grad-CAM heatmap overlaid on a carotid B-mode frame, with warm activation along the longitudinal vessel wall." loading="lazy">
 <figcaption>The warm band hugs the longitudinal vessel wall, the boundary that fixes the flow axis, rather than the surrounding speckle. Three hand-picked frames, illustrative rather than a cohort-wide attribution study.</figcaption>
</figure>

Grad-CAM, finally, lands on the carotid wall, the boundary that fixes the flow axis. The network is reading anatomy, not label burn-in. The pipeline is now whole: rotation-built labels, a frozen backbone read through grid pooling, scored two ways, tuned and stacked, then wrapped in a calibrated band and an attribution check.

## Where it reaches next, and where the ceiling is {#limits}

So where does the method go from here, and what stops it? Two answers, and they point in opposite directions.

The next reach is concrete. Modern self-supervised encoders, DINOv2 or a medical-ultrasound foundation model, are the obvious successors to a frozen ImageNet backbone, but they need a CUDA box and a reworked dependency stack. A hand-crafted structure-tensor baseline (MAE 3.16&deg;) with a learned-plus-classical circular fusion (2.72&deg;) suggests the learned and geometric cues capture partly complementary signal, which is a thread worth pulling, though both figures are in-sample on a narrow base-angle band.

The ceiling I actually hit is hardware, not tuning. End-to-end fine-tuning of the strong backbones OOMs the Apple GPU even at batch 16. A from-scratch CNN fails outright (MAPE around 101%, negative $R^2$). For 84 images, read those two failures as positive evidence that frozen transfer is the right tool here. The small-data regime was telling me what it wanted.

Everything traces back to `results/`: the library is typed and test-driven, and every figure here regenerates from script at seed 42, nothing hand-typed. What began as one hand-drawn angle per image is now an estimator that reads it back to within a couple of degrees, wrapped in a calibrated band you could hand a clinic.

<style>
.page__content figure { text-align: center; display: block; }
.page__content figure svg,
.page__content figure img { display: block; margin-left: auto; margin-right: auto; max-width: 100%; }
.page__content figure figcaption { text-align: center; max-width: 44rem; margin-left: auto; margin-right: auto; }
.page__content .dop-controls,
.page__content .gap-controls,
.page__content .prot-controls,
.page__content .post-cta { text-align: center; }
.page__content table { margin-left: auto; margin-right: auto; }
.page__content .post-tagline { text-align: center; letter-spacing: .09em; text-transform: uppercase; font-size: .76rem; opacity: .58; margin: 0 0 .25rem; }
.page__content .post-links { text-align: center; font-size: .92rem; margin: 0 0 1.5rem; }
.page__content .post-meta-note { font-size: .8rem; line-height: 1.55; opacity: .6; font-style: italic; margin: 0 0 1.2rem; }
a.btn-soft {
 display: inline-block;
 margin: 0 0.5rem 0.55rem 0;
 padding: 0.5em 1.05em;
 font-size: 0.92rem;
 font-weight: 500;
 line-height: 1.3;
 color: inherit;
 text-decoration: none;
 border-radius: 8px;
 border: 1px solid rgba(128, 128, 128, 0.30);
 border: 1px solid color-mix(in srgb, currentColor 22%, transparent);
 background: rgba(128, 128, 128, 0.06);
 background: color-mix(in srgb, currentColor 5%, transparent);
 transition: background-color .18s ease, border-color .18s ease, color .18s ease;
}
a.btn-soft:hover {
 text-decoration: none;
 border-color: rgba(128, 128, 128, 0.55);
 border-color: color-mix(in srgb, currentColor 42%, transparent);
 background: rgba(128, 128, 128, 0.12);
 background: color-mix(in srgb, currentColor 11%, transparent);
}
a.btn-soft--primary {
 font-weight: 600;
 border-color: rgba(128, 128, 128, 0.55);
 border-color: color-mix(in srgb, currentColor 42%, transparent);
}
</style>

<div class="post-cta" markdown="1">
[Try the angle dial](#dop-svg){: .btn-soft .btn-soft--primary} [Project site]({{ site.baseurl }}/ultrasound-doppler-angle-estimation/){: .btn-soft} [Project write-up]({{ '/portfolio/ultrasound-doppler-angle-estimation/' | relative_url }}){: .btn-soft} [Browse the code](https://github.com/nilesh-patil/ultrasound-doppler-angle-estimation){: .btn-soft}
</div>

[^grid]: DenseNet201 uses a 3&times;3 grid; the other four classic backbones use 2&times;2. The choice falls out of each backbone's tuned config.
[^groups]: GroupKFold then guarantees no proxy group straddles the train/test boundary, which is the property that matters even if the group labels are imperfect.
[^anchor]: Both lines mix estimators by stage: the 5.84% and 12.59% core-model points are single-split and per-fold-mean figures, while the tuned (4.03% / 10.14%) and ensemble (2.79% / 8.53%) points are CV or pooled-OOF means. The project site details why stacking on pooled OOF over 12 groups is itself mildly optimistic.

<script>
(function () {
 "use strict";

 /* === VISUAL 1: Doppler dial === */
 (function () {
 var svg = document.getElementById("dop-svg");
 var range = document.getElementById("dop-range");
 var readout = document.getElementById("dop-readout");
 var beam = document.getElementById("dop-beam");
 var curve = document.getElementById("dop-curve");
 var ref = document.getElementById("dop-ref");
 var arc = document.getElementById("dop-arc");
 var thetaLbl = document.getElementById("dop-theta-lbl");
 if (!svg || !range || !readout || !beam || !curve || !ref || !arc) return;

 var pivotX = 244, pivotY = 168, beamLen = 150;
 var px0 = 330, px1 = 624, py0 = 288, pyTop = 40;
 var tMin = 30, tMax = 78;
 var mAt60 = 1 / Math.cos(60 * Math.PI / 180);

 function mult(t) { return 1 / Math.cos(t * Math.PI / 180); }
 function xOf(t) { return px0 + (t - tMin) / (tMax - tMin) * (px1 - px0); }
 function yOf(m) {
 var top = 5;
 var clamped = Math.min(m, top);
 return py0 - (clamped - 1) / (top - 1) * (py0 - pyTop);
 }

 var d = "";
 for (var t = tMin; t <= tMax; t += 1) {
 d += (t === tMin ? "M" : "L") + xOf(t).toFixed(1) + "," + yOf(mult(t)).toFixed(1) + " ";
 }
 curve.setAttribute("d", d.trim());

 var y60 = yOf(mAt60);
 ref.setAttribute("y1", y60.toFixed(1));
 ref.setAttribute("y2", y60.toFixed(1));
 ref.setAttribute("x1", px0);
 ref.setAttribute("x2", px1);

 var dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
 dot.setAttribute("r", "4");
 dot.setAttribute("fill", "currentColor");
 svg.appendChild(dot);

 function update() {
 var t = parseFloat(range.value);
 var rad = t * Math.PI / 180;
 var ex = pivotX - beamLen * Math.cos(rad);
 var ey = pivotY - beamLen * Math.sin(rad);
 beam.setAttribute("x2", ex.toFixed(1));
 beam.setAttribute("y2", ey.toFixed(1));
 var r = 34;
 var ax = pivotX - r, ay = pivotY;
 var bx = pivotX - r * Math.cos(rad), by = pivotY - r * Math.sin(rad);
 arc.setAttribute("d", "M" + ax.toFixed(1) + "," + ay.toFixed(1) + " A" + r + "," + r + " 0 0 1 " + bx.toFixed(1) + "," + by.toFixed(1));
 if (thetaLbl) {
 var half = rad / 2, lr = r + 14;
 thetaLbl.setAttribute("x", (pivotX - lr * Math.cos(half)).toFixed(1));
 thetaLbl.setAttribute("y", (pivotY - lr * Math.sin(half) + 4).toFixed(1));
 }
 var m = mult(t);
 dot.setAttribute("cx", xOf(t).toFixed(1));
 dot.setAttribute("cy", yOf(m).toFixed(1));
 var err = (m / mAt60 - 1) * 100;
 var sign = err >= 0 ? "+" : "";
 var weight = err > 60 ? 700 : (err > 20 ? 600 : 400);
 readout.style.fontWeight = weight;
 readout.innerHTML = "&#952; = " + t.toFixed(0) + "&#176; &nbsp;|&nbsp; velocity &#215;" + m.toFixed(2) + " &nbsp;|&nbsp; error vs 60&#176; reference = " + sign + err.toFixed(0) + "%";
 }

 range.addEventListener("input", update);
 update();
 })();

 /* === VISUAL 2: grid pooling vs GAP === */
 (function () {
 var btn = document.getElementById("gap-btn");
 var mapG = document.getElementById("gap-map");
 var barsG = document.getElementById("gap-bars");
 var gridG = document.getElementById("gap-grid");
 if (!btn || !mapG || !barsG || !gridG) return;

 var SVGNS = "http://www.w3.org/2000/svg";
 var N = 6, cell = 24, ox = 28, oy = 54;
 var state = 0;

 function rect(parent, x, y, w, h, op) {
 var r = document.createElementNS(SVGNS, "rect");
 r.setAttribute("x", x); r.setAttribute("y", y);
 r.setAttribute("width", w); r.setAttribute("height", h);
 r.setAttribute("fill", "currentColor");
 r.setAttribute("fill-opacity", op);
 parent.appendChild(r);
 return r;
 }
 function active(i, j, s) {
 if (s === 0) return Math.abs((i + j) - (N - 1)) <= 1;
 return Math.abs(i - j) <= 1;
 }

 function draw() {
 mapG.innerHTML = ""; barsG.innerHTML = ""; gridG.innerHTML = "";
 for (var i = 0; i < N; i++) {
 for (var j = 0; j < N; j++) {
 rect(mapG, ox + j * cell, oy + i * cell, cell - 2, cell - 2, active(i, j, state) ? 1 : 0.16);
 }
 }
 var bx = 300, by = 60, bw = 12, gap = 5, maxh = 120;
 var heights = [0.42, 0.55, 0.38, 0.5, 0.46, 0.4];
 for (var b = 0; b < heights.length; b++) {
 var h = heights[b] * maxh;
 rect(barsG, bx + b * (bw + gap), by + (maxh - h), bw, h, 0.85);
 }
 var gx = 520, gy = 96, gc = 34;
 var pop = state === 0 ? [[0, 1], [1, 0]] : [[0, 0], [1, 1]];
 for (var r = 0; r < 2; r++) {
 for (var c = 0; c < 2; c++) {
 var on = pop.some(function (p) { return p[0] === r && p[1] === c; });
 rect(gridG, gx + c * gc, gy + r * gc, gc - 3, gc - 3, on ? 1 : 0.16);
 }
 }
 }

 btn.addEventListener("click", function () {
 state = state ? 0 : 1;
 btn.setAttribute("aria-pressed", state ? "true" : "false");
 draw();
 });
 draw();
 })();

 /* === VISUAL 3: two-protocol split === */
 (function () {
 var cellsG = document.getElementById("prot-cells");
 var blocksG = document.getElementById("prot-blocks");
 var volLbl = document.getElementById("prot-vol-lbl");
 var btnImg = document.getElementById("prot-btn-img");
 var btnPat = document.getElementById("prot-btn-pat");
 var readout = document.getElementById("prot-readout");
 if (!cellsG || !blocksG || !btnImg || !btnPat || !readout) return;

 var SVGNS = "http://www.w3.org/2000/svg";
 var mode = "image";
 var blocksPerRow = 5, blockRows = 2;
 var cw = 13, gap = 3, bx0 = 30, by0 = 60;
 var blockW = 3 * (cw + gap) + 10, blockH = 3 * (cw + gap) + 10;
 var bgapX = 12, bgapY = 30;

 function prand(n) { var x = Math.sin(n * 12.9898) * 43758.5453; return x - Math.floor(x); }

 function rect(x, y, w, h, op, dash) {
 var r = document.createElementNS(SVGNS, "rect");
 r.setAttribute("x", x); r.setAttribute("y", y);
 r.setAttribute("width", w); r.setAttribute("height", h);
 r.setAttribute("rx", "2");
 r.setAttribute("fill", "currentColor");
 r.setAttribute("fill-opacity", op);
 r.setAttribute("stroke", "currentColor");
 r.setAttribute("stroke-opacity", dash ? 0.9 : 0.2);
 if (dash) r.setAttribute("stroke-dasharray", "2 2");
 cellsG.appendChild(r);
 }

 function draw() {
 cellsG.innerHTML = ""; blocksG.innerHTML = "";
 var heldBlocks = [3, 7];
 for (var b = 0; b < blocksPerRow * blockRows; b++) {
 var br = Math.floor(b / blocksPerRow), bc = b % blocksPerRow;
 var bx = bx0 + bc * (blockW + bgapX);
 var by = by0 + br * (blockH + bgapY);
 var outline = document.createElementNS(SVGNS, "rect");
 outline.setAttribute("x", bx - 5); outline.setAttribute("y", by - 5);
 outline.setAttribute("width", blockW); outline.setAttribute("height", blockH);
 outline.setAttribute("rx", "5");
 outline.setAttribute("stroke", "currentColor");
 outline.setAttribute("fill", "none");
 outline.setAttribute("stroke-opacity", (mode === "patient" && heldBlocks.indexOf(b) >= 0) ? 0.85 : 0.3);
 blocksG.appendChild(outline);
 for (var cc = 0; cc < 9; cc++) {
 var cr = Math.floor(cc / 3), ccol = cc % 3;
 var x = bx + ccol * (cw + gap), y = by + cr * (cw + gap);
 var held;
 if (mode === "image") {
 held = prand(b * 9 + cc + 1) < 0.22;
 } else {
 held = heldBlocks.indexOf(b) >= 0;
 }
 rect(x, y, cw, cw, held ? 0.18 : 0.7, held);
 }
 }
 }

 volLbl.setAttribute("x", 2);
 volLbl.setAttribute("y", by0 - 12);
 volLbl.textContent = "one volunteer";

 function setMode(m) {
 mode = m;
 var onBg = "color-mix(in srgb, currentColor 22%, transparent)";
 var offBg = "color-mix(in srgb, currentColor 6%, transparent)";
 btnImg.style.background = m === "image" ? onBg : offBg;
 btnPat.style.background = m === "patient" ? onBg : offBg;
 btnImg.setAttribute("aria-checked", m === "image" ? "true" : "false");
 btnPat.setAttribute("aria-checked", m === "patient" ? "true" : "false");
 readout.innerHTML = m === "image"
 ? "Image-level: held-out rows scatter across every volunteer, interpolating across orientations. Tuned ensemble: 2.79% MAPE / 1.96&#176;."
 : "Patient-level: whole volunteers held out, generalizing to an unseen patient. Tuned ensemble: 8.53% MAPE / 5.93&#176;.";
 draw();
 }

 btnImg.addEventListener("click", function () { setMode("image"); });
 btnPat.addEventListener("click", function () { setMode("patient"); });
 setMode("image");
 })();

})();
</script>
