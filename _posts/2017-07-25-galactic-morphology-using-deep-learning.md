---
layout: single
title: "Convolutional networks - regressing Galaxy Zoo morphology vectors from raw pixels"
date: 2017-07-25T15:39:55-04:00
last_modified_at: 2017-07-25T19:48:19-04:00
categories: [blog]
tags: [deep learning, galaxy, computer-vision, machine learning, neural networks]
excerpt: "Six weeks of training runs against Galaxy Zoo's 37-value vote vectors: a classification dead end, the switch to regression, and the DenseNet checkpoint that landed at an MSE of 0.0142 on a validation set that reused the training images."
math: true
redirect_from:
  - /blog/galactic-morphology-using-deep-learning/
header:
  overlay_image: /images/blog/headers/galactic-morphology-using-deep-learning.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/galactic-morphology-using-deep-learning.jpg
---

Galaxy Zoo has been running since 2007. Volunteers look at a survey image and walk a tree of questions about it. Smooth or featured? Is there a spiral? How many arms? By the time the crowd is done with a galaxy, its answers have been collapsed into 37 numbers between 0 and 1, each one a weighted share of what people saw.

I spent six weeks this spring, April 13 through May 27, trying to get a convolutional network to reproduce those 37 numbers from raw pixels on our university's GPU cluster. The wreckage is in the repo: twenty-six training logs, TensorBoard folders numbered v1 through v6 plus a v5.5 wedged in between, one saved checkpoint. Most of April went to training the wrong kind of model.

<figure>
<svg viewBox="0 0 760 240" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="pipeline-title pipeline-desc">
<title id="pipeline-title">A galaxy image flowing through convolutional and pooling blocks into a 37-length output vector</title>
<desc id="pipeline-desc">An input image block on the left passes through a sequence of shrinking convolution and pooling blocks, then into a dense head that emits a vertical strip of 37 numbers.</desc>
<rect x="20" y="70" width="90" height="90" fill="none" stroke="currentColor" stroke-width="2"></rect>
<text x="65" y="180" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">input image</text>
<rect x="150" y="82" width="60" height="66" fill="currentColor" opacity="0.12" stroke="currentColor" stroke-width="2"></rect>
<text x="180" y="166" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">conv + pool</text>
<rect x="250" y="92" width="44" height="46" fill="currentColor" opacity="0.18" stroke="currentColor" stroke-width="2"></rect>
<text x="272" y="166" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">conv + pool</text>
<rect x="334" y="100" width="30" height="30" fill="currentColor" opacity="0.24" stroke="currentColor" stroke-width="2"></rect>
<text x="349" y="166" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">conv + pool</text>
<path d="M118 115 L148 115 M218 115 L248 115 M302 115 L332 115 M372 115 L420 115" stroke="currentColor" stroke-width="2" fill="none" opacity="0.6"></path>
<rect x="420" y="78" width="70" height="74" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3"></rect>
<text x="455" y="120" text-anchor="middle" font-size="12" fill="currentColor">dense</text>
<text x="455" y="166" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">regression head</text>
<path d="M498 115 L548 115" stroke="currentColor" stroke-width="2" fill="none" opacity="0.6"></path>
<rect x="560" y="20" width="34" height="190" fill="none" stroke="currentColor" stroke-width="2"></rect>
<line x1="560" y1="40" x2="594" y2="40" stroke="currentColor" stroke-width="1" opacity="0.5"></line>
<line x1="560" y1="60" x2="594" y2="60" stroke="currentColor" stroke-width="1" opacity="0.5"></line>
<line x1="560" y1="80" x2="594" y2="80" stroke="currentColor" stroke-width="1" opacity="0.5"></line>
<line x1="560" y1="100" x2="594" y2="100" stroke="currentColor" stroke-width="1" opacity="0.5"></line>
<line x1="560" y1="120" x2="594" y2="120" stroke="currentColor" stroke-width="1" opacity="0.5"></line>
<line x1="560" y1="140" x2="594" y2="140" stroke="currentColor" stroke-width="1" opacity="0.5"></line>
<line x1="560" y1="160" x2="594" y2="160" stroke="currentColor" stroke-width="1" opacity="0.5"></line>
<line x1="560" y1="180" x2="594" y2="180" stroke="currentColor" stroke-width="1" opacity="0.5"></line>
<text x="620" y="120" text-anchor="middle" font-size="13" fill="currentColor">37</text>
<text x="620" y="138" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">values</text>
</svg>
<figcaption>The whole pipeline: stacked conv and pool blocks, then a dense head emits a 37-length morphology vector.</figcaption>
</figure>

## One jpeg in, 37 numbers out

The images come from the [Sloan Digital Sky Survey](http://www.sdss.org/), and the labels from [Galaxy Zoo](https://www.galaxyzoo.org), packaged for the Kaggle Galaxy Challenge as 61,578 training jpegs plus a `training_solutions_rev1.csv` keyed by GalaxyID. To quote the SDSS site:

> The Sloan Digital Sky Survey has created the most detailed three-dimensional maps of the Universe ever made, with deep multi-color images of one third of the sky, and spectra for more than three million astronomical objects.

Each label row is the crowd's verdict on one galaxy, produced by volunteers walking this decision tree:

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/00.galaxyzoo-tree.png" alt="Galaxy Zoo decision tree of classification questions" class="center-image">
<figcaption>The Galaxy Zoo decision tree, from Galaxy Zoo 2 (Willett et al. 2013).</figcaption>
</figure>

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/01.galaxies.png" alt="A grid of sample galaxy images from the dataset" class="center-image">
<figcaption>Sample galaxies from the dataset, SDSS image cutouts.</figcaption>
</figure>

Every jpeg is 424x424x3. The loader goes through `cv2.resize`, and over the six weeks the working resolution wandered from 32x32 sanity checks up to 256x256.

The runs themselves happened on BlueHive, the university cluster, through SLURM: the gpu partition, two GPUs and 55 GB of memory per job, `module load tensorflow`, and a time limit a shade under five days. Everything is Keras on the TensorFlow backend. The 55 GB is a real requirement, because the scripts load the whole training set into one array before fitting starts, and at 256x256 that int16 array is 24 GB on its own. The second GPU is a decoration. Training sits pinned to one device; multi-GPU Keras is a hand-rolled affair, and I never rolled it.

## April: the classification dead end

My first models were classifiers, because that is what a stack of image tutorials trains you to reach for. `01.GettingStarted.ipynb` feeds the 37 vote fractions straight to a categorical crossentropy, with a sigmoid on the 37-unit output layer. `model.py` ends in a sigmoid 1x1 convolution head. The accuracy the runs logged is Keras's `categorical_accuracy`: a galaxy counts as correct when the largest of the 37 outputs is the answer with the largest vote fraction.

One convnet jumped to 0.61 validation accuracy almost immediately and stayed there for 64 epochs. The next climbed from 0.59 to 0.75 over 33 epochs; another took 57 epochs to reach the same 0.75 and stopped. The longest run recast the answers as separate categories and needed 190 epochs to crawl from under one percent to 0.47. For the two runs that reached 0.75 the crossentropy column barely moved, easing from 14.4 to 13.8 while accuracy climbed.

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/02.classification-runs.png" alt="Validation accuracy and loss curves for four classification runs, two plateauing near 0.75" class="center-image">
<figcaption>The four classification runs.</figcaption>
</figure>

When two runs stall at the same number, the ceiling probably belongs to the question rather than the network. I say probably because the head was its own mess, a sigmoid output trained under categorical crossentropy. Still, a cleaner classifier would only have answered the wrong question more sharply. Argmax is first-past-the-post voting. A galaxy the crowd split 51-49 lands in the same class as one they called 99-1. Volunteers disagree about ambiguous smudges of light, that disagreement is graded and it is informative, and it is the entire reason the label is 37 numbers instead of one.

## The head the data wanted

So the target stays a vector and the head becomes a regression. All 37 outputs stay, the crossentropy goes, and the loss is mean squared error against the vote fractions themselves. This also happens to be exactly what the challenge scores, RMSE over all 37 values. It is on the evaluation page, which I evidently did not read carefully enough in April.

There is a trap sitting right next to that fix, though. Reflex says put a `softmax` on a 37-way output. Softmax renormalizes the vector to sum to 1, and these vectors do not sum to 1, because Galaxy Zoo scales each question's answers by the share of volunteers who actually reached that question in the tree. The structured thing to do would be a per-question softmax rescaled by the parent weight. My heads stay unnormalized, with a `relu` on the convnet's output so it cannot predict a negative vote share the way a plain linear unit sometimes will.

## The convnet

The convnet is seven blocks, filter counts stepping 32, 64, 128, 256, 512, 768, 1024, kernels of 4x4 in the first block and 2x2 after, all same-padded so the grid size survives the convolution. A `BatchNormalization` layer (Ioffe and Szegedy 2015) opens every block. Each block ends in a `relu`, a max pool that halves the grid (the first block's 4x4 pool quarters it), and a `Dropout(0.2)` (Srivastava et al. 2014). After the last block a dense layer of 1024 units feeds the 37-value head.

In a classic astronomy pipeline someone hand-designs the features: concentration, asymmetry, the Gini coefficient of the light distribution (Conselice 2003; Lotz, Primack and Madau 2004). Here the 4x4 and 2x2 kernels are fit against the vote target, and whatever transforms best predict the crowd's answers are the features.

## Regression runs

The plain convnet's best regression run flattened at a validation MSE of 0.031 and would not go lower. The jump came from swapping the backbone for a DenseNet, a Keras implementation of last year's densely connected networks (Huang et al. 2016), configured with depth 64, growth rate 10, bottleneck layers, and a 0.5 compression at each transition. The DenseNet runs also picked up augmentation the earlier ones lacked, `rotation_range=180` with both flips, free here because a galaxy has no right way up. Both changes landed in the same step, so I cannot split the 0.031-to-0.015 drop between them.

Six of the regression runs, plotted from the loss CSVs:

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/03.regression-runs.png" alt="Validation MSE curves for six regression runs on a log scale, DenseNet reaching 0.0142" class="center-image">
<figcaption>Validation MSE for the regression runs, log scale. The dashed grey line at the bottom comes from a target-loading bug.</figcaption>
</figure>

The run that was both smallest and lowest-resolution, 32x32 on a thousand samples, never left an MSE of 0.37. The fast DenseNet variant at 64x64 never got below 0.045; 128x128 was the one genuine divergence, ending its eighth epoch with a validation loss above 0.6, almost four times where it began, and I still do not know why. A 256x256 sibling bottomed out at 0.025 and then spiked to 0.53 on its final epoch, which is the kind of exit that makes you grateful for `ModelCheckpoint(save_best_only=True)`. One run's very first epoch logged a training loss just over 950,000 before settling under 0.07.

The run that mattered, densenet 01, walked down from 0.16 to 0.0152 over 54 epochs. Its continuation logged 0.0142 at the end of its first epoch and stayed there: `ReduceLROnPlateau` cut the learning rate from 1e-6 to 1e-9 in three steps across fourteen epochs, and validation MSE moved from 0.01421 to 0.01420.

Those digits come with a caveat. The DenseNet script's validation generator draws from the same arrays its training generator does. The convnet script had done this correctly, holding out a `validation_split=0.1`; the split never made it across the rewrite. Every DenseNet validation number in this chart is therefore optimistic, closer to a training fit than a held-out error.

The dashed grey line was briefly the most exciting number of the spring: a convnet variant reporting 0.0027 from its first epoch, five times better than anything else ever managed. It is fake. The script allocates the target array as `int16`, so every vote fraction below 1.0 truncated to zero as the array filled, and the model earned its score by learning to predict near-zero against targets that mostly were zero. A first epoch that starts where other runs end is not a breakthrough; it is a dtype.

## Where it landed

Six weeks reduce to one checkpoint. `densenet_02_best.h5`, validation MSE 0.0142, which on the vote fractions is an RMSE of 0.119. For scale I have two anchors. Kaggle's central-pixel benchmark, basically averaging the votes of training galaxies with similar core colors, scored 0.16194. Sander Dieleman won the 2014 challenge at a private-leaderboard RMSE of 0.07492 (Dieleman, Willett and Dambre 2015).

All of it is bounded by the validation leak, so porting the convnet's `validation_split` back over is the first thing on the list. And 0.119 is a validation number in any case; lining it up against 0.07492 would take the test-set submission I have not made.

<style>
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

[Browse the code and training runs](https://github.com/nilesh-patil/galaxy-classification-using-deep-learning){: .btn-soft .btn-soft--primary}

## References

1. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). *Journal of Machine Learning Research*, 15.
2. Ioffe, S. and Szegedy, C. (2015). [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). *Proceedings of the 32nd International Conference on Machine Learning*.
3. Willett, K. W., Lintott, C. J., Bamford, S. P., et al. (2013). [Galaxy Zoo 2: detailed morphological classifications for 304,122 galaxies from the Sloan Digital Sky Survey](https://arxiv.org/abs/1308.3496). *Monthly Notices of the Royal Astronomical Society*, 435.
4. Conselice, C. J. (2003). [The Relationship between Stellar Light Distributions of Galaxies and Their Formation Histories](https://arxiv.org/abs/astro-ph/0303065). *The Astrophysical Journal Supplement Series*, 147.
5. Lotz, J. M., Primack, J., and Madau, P. (2004). [A New Nonparametric Approach to Galaxy Morphological Classification](https://arxiv.org/abs/astro-ph/0311352). *The Astronomical Journal*, 128.
6. Huang, G., Liu, Z., Weinberger, K. Q., and van der Maaten, L. (2016). [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993). arXiv:1608.06993.
7. Dieleman, S., Willett, K. W., and Dambre, J. (2015). [Rotation-invariant convolutional neural networks for galaxy morphology prediction](https://arxiv.org/abs/1503.07077). *Monthly Notices of the Royal Astronomical Society*, 450.
