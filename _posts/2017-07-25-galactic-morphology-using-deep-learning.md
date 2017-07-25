---
layout: single
title: "Convolutional networks - regressing Galaxy Zoo morphology vectors from raw pixels"
date: 2017-07-25T15:39:55-04:00
last_modified_at: 2017-07-25T19:48:19-04:00
categories: [blog]
tags: [deep learning, galaxy, computer-vision, machine learning, neural networks]
excerpt: "Six weeks of training runs against Galaxy Zoo's 37-value vote vectors: a classification dead end, the switch to regression, and the DenseNet checkpoint that landed at validation MSE 0.0142."
math: true
redirect_from:
  - /blog/galactic-morphology-using-deep-learning/
header:
  overlay_image: /images/blog/headers/galactic-morphology-using-deep-learning.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/galactic-morphology-using-deep-learning.jpg
---

Galaxy Zoo has been running since 2007. Volunteers look at a survey image and walk a tree of questions about it. Smooth or featured? Is there a spiral? How many arms? By the time the crowd is done with a galaxy, its answers have been collapsed into 37 numbers between 0 and 1, each one a weighted share of what people saw.

I spent six weeks this spring, April 13 through May 27, trying to get a convolutional network to reproduce those 37 numbers from raw pixels on our university's GPU cluster. The wreckage is all still in the repo: twenty-four training logs, TensorBoard folders numbered v1 through v6 plus a v5.5 wedged in between, one saved checkpoint. Also preserved in there, unfortunately, is the fact that most of April went to training the wrong kind of model.

Every run shared the same pipeline, drawn below. The two ends stayed fixed the whole spring. What changed was the middle, and the loss.

<figure>
<svg viewBox="0 0 760 240" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="pipeline-title pipeline-desc">
<title id="pipeline-title">A 424x424x3 galaxy image flowing through convolutional and pooling blocks into a 37-length output vector</title>
<desc id="pipeline-desc">An input image block on the left, labeled 424 by 424 by 3, passes through a sequence of shrinking convolution and pooling blocks, then into a dense head that emits a vertical strip of 37 numbers.</desc>
<rect x="20" y="70" width="90" height="90" fill="none" stroke="currentColor" stroke-width="2"></rect>
<text x="65" y="180" text-anchor="middle" font-size="13" fill="currentColor">424x424x3</text>
<text x="65" y="196" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">input image</text>
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
<figcaption>The whole pipeline on one substrate: a 424x424x3 image is compressed through stacked conv and pool blocks, then a dense head emits a 37-length morphology vector. The depth of the conv stack is what I vary across experiments; the input and the 37-value head stay fixed.</figcaption>
</figure>

## One jpeg in, 37 numbers out

The images come from the [Sloan Digital Sky Survey](http://www.sdss.org/), and the labels from [Galaxy Zoo](https://www.galaxyzoo.org), packaged for the Kaggle Galaxy Challenge as 61,578 training jpegs plus a `training_solutions_rev1.csv` keyed by GalaxyID. To quote the SDSS site:

> The Sloan Digital Sky Survey has created the most detailed three-dimensional maps of the Universe ever made, with deep multi-color images of one third of the sky, and spectra for more than three million astronomical objects.

Each label row is the crowd's verdict on one galaxy, produced by volunteers walking this decision tree:

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/00.galaxyzoo-tree.png" alt="Galaxy Zoo decision tree of classification questions" class="center-image">
<figcaption>The Galaxy Zoo decision tree, from Galaxy Zoo 2 (Willett et al. 2013). Each volunteer walks these branching questions for an image, and the weighted answers become the 37-value target the network has to predict.</figcaption>
</figure>

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/01.galaxies.png" alt="A grid of sample galaxy images from the dataset" class="center-image">
<figcaption>Sample galaxies from the dataset, SDSS image cutouts. Each is a single 424x424x3 jpeg that has to come out the other end as 37 numbers.</figcaption>
</figure>

Every jpeg is 424x424x3. None of my runs consumed it at that size: the loader goes through `cv2.resize`, and over the six weeks the working resolution wandered from 32x32 sanity checks up to 256x256. The one preprocessing step every run shares is a rescale by 1/255, done inside the model in a `Lambda` layer, so the files on disk stay untouched and anything that loads the checkpoint gets the scaling for free. Beyond that the array goes in raw.

<figure>
<svg viewBox="0 0 520 320" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="numpy-title numpy-desc">
<title id="numpy-title">A galaxy image as a 424 by 424 by 3 numpy array of integer pixel values</title>
<desc id="numpy-desc">Three grids are stacked front to back for the red, green, and blue channels. The front grid shows sample integer pixel values from 0 to 255 with ellipses standing in for the full 424 by 424 extent. Brackets label the height, the width, and the depth of three channels.</desc>
<rect x="130" y="55" width="170" height="170" fill="currentColor" fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.5"></rect>
<rect x="110" y="75" width="170" height="170" fill="currentColor" fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"></rect>
<path d="M90 95 L130 55 M260 95 L300 55 M260 265 L300 225" stroke="currentColor" stroke-width="1.5" fill="none" opacity="0.4"></path>
<rect x="90" y="95" width="170" height="170" fill="none" stroke="currentColor" stroke-width="2"></rect>
<path d="M132.5 95 L132.5 265 M175 95 L175 265 M217.5 95 L217.5 265 M90 137.5 L260 137.5 M90 180 L260 180 M90 222.5 L260 222.5" stroke="currentColor" stroke-width="1" fill="none" opacity="0.4"></path>
<g font-size="12" text-anchor="middle" fill="currentColor">
<text x="111" y="121">32</text><text x="154" y="121">41</text><text x="196" y="121">58</text><text x="239" y="121" opacity="0.6">…</text>
<text x="111" y="163">27</text><text x="154" y="163">60</text><text x="196" y="163">91</text><text x="239" y="163" opacity="0.6">…</text>
<text x="111" y="206">44</text><text x="154" y="206">88</text><text x="196" y="206">150</text><text x="239" y="206" opacity="0.6">…</text>
<text x="111" y="248" opacity="0.6">…</text><text x="154" y="248" opacity="0.6">…</text><text x="196" y="248" opacity="0.6">…</text><text x="239" y="248" opacity="0.6">…</text>
</g>
<text x="245" y="90" font-size="12" fill="currentColor">R</text>
<text x="265" y="70" font-size="12" fill="currentColor" opacity="0.6">G</text>
<text x="285" y="50" font-size="12" fill="currentColor" opacity="0.4">B</text>
<path d="M72 95 L72 265 M72 95 L80 95 M72 265 L80 265" stroke="currentColor" stroke-width="1.5" fill="none" opacity="0.7"></path>
<text transform="rotate(-90 56 180)" x="56" y="180" text-anchor="middle" font-size="12" fill="currentColor">424</text>
<path d="M90 282 L260 282 M90 282 L90 274 M260 282 L260 274" stroke="currentColor" stroke-width="1.5" fill="none" opacity="0.7"></path>
<text x="175" y="299" text-anchor="middle" font-size="12" fill="currentColor">424</text>
<path d="M300 55 L345 55 M260 95 L345 95 M345 55 L345 95" stroke="currentColor" stroke-width="1.5" fill="none" opacity="0.6"></path>
<text x="355" y="72" font-size="12" fill="currentColor">3 channels</text>
<text x="355" y="88" font-size="11" fill="currentColor" opacity="0.75">(R, G, B)</text>
</svg>
<figcaption>The same galaxy as a numpy array: a 424x424x3 block of uint8 values, three color channels stacked, each pixel from 0 to 255. From here on, every layer is a transform on this block of numbers.</figcaption>
</figure>

The runs themselves happened on BlueHive, the university cluster, through SLURM: the gpu partition, two GPUs and 55 GB of memory per job, `module load tensorflow`, and a time limit a shade under five days. Everything is Keras on the TensorFlow backend. The 55 GB is a real requirement, because the scripts load the whole training set into one array before fitting starts, and at 256x256 that int16 array is 24 GB on its own. The second GPU is a decoration. Training sits pinned to one device; multi-GPU Keras in 2017 is a hand-rolled affair, and I never rolled it.

## April: the classification dead end

My first models were classifiers, because that is what a stack of image tutorials trains you to reach for. Take the 37-value vector, argmax it into a single winning answer, one-hot the winner, minimize categorical crossentropy. The `model.py` from that era is still in the repo, ending in a sigmoid 1x1 convolution head.

The logs say exactly how that went. One convnet jumped to 0.61 validation accuracy almost immediately and stayed there for 64 epochs. The next climbed from 0.59 to 0.75 over 33 epochs; another took 57 epochs to reach the same 0.75 and stopped. The longest run of the era recast the answers as separate categories and needed 190 epochs to crawl from under one percent to 0.47. For the three argmax runs the crossentropy column barely breathed, easing from 14.4 to 13.8 while accuracy climbed sixteen points.

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/02.classification-runs.png" alt="Validation accuracy and loss curves for four classification runs, two plateauing near 0.75" class="center-image">
<figcaption>The four classification-era runs, replotted from the committed training logs. Two runs hit the same ceiling near 0.75 accuracy and flatline there; a third stalls at 0.61.</figcaption>
</figure>

When three architectures stall at the same number, the ceiling probably belongs to the question rather than the network. I say probably because the head of that era was its own mess, a sigmoid output trained under categorical crossentropy, and that pairing alone will pin a loss column near its clipping ceiling no matter what accuracy does. Still, a cleaner classifier would only have answered the wrong question more sharply. Argmax is first-past-the-post voting. A galaxy the crowd split 51-49 lands in the same class as one they called 99-1. Volunteers disagree about ambiguous smudges of light, that disagreement is graded and it is informative, and it is the entire reason the label is 37 numbers instead of one. My argmax threw it away before training even started.

## The head the data wanted

So the target stays a vector and the head becomes a regression. All 37 outputs stay, the one-hot goes, and the loss is mean squared error against the vote fractions themselves. Instead of calling the election I am now predicting the poll. This also happens to be exactly what the challenge scores, RMSE over all 37 values. It is on the evaluation page, which I evidently did not read carefully enough in April.

There is a trap sitting right next to that fix, though. Reflex says put a `softmax` on a 37-way output. Softmax renormalizes the vector to sum to 1, and these vectors do not sum to 1, because Galaxy Zoo scales each question's answers by the share of volunteers who actually reached that question in the tree. The structured thing to do would be a per-question softmax rescaled by the parent weight. I never built it. My heads stay unnormalized, with a `relu` on the convnet's output so it cannot predict a negative vote share the way a plain linear unit sometimes will.

## The stack between the two ends

The middle of the pipeline is the standard alternation, and the committed convnet spells out one concrete instance of it: seven blocks, filter counts stepping 32, 64, 128, 256, 512, 768, 1024, kernels of 4x4 in the first block and 2x2 after, all same-padded so the grid size survives the convolution. Each block ends in a `relu`, a max pool that halves the grid (4x4 in the first block), and a `Dropout(0.2)`. After the last block a dense layer of 1024 units feeds the 37-value head.

The convolutional layers are the part I actually care about. In a classic astronomy pipeline someone hand-designs the features: concentration, asymmetry, the Gini coefficient of the light distribution (Conselice 2003; Lotz, Primack and Madau 2004). Here the 4x4 and 2x2 kernels are fit against the vote target, and whatever transforms best predict the crowd's answers are the features. Nobody tells the network what a spiral arm looks like, and by the end nothing in the weights is guaranteed to correspond to one.

## Batch normalization between the blocks

In the convnet script, a `BatchNormalization` layer opens every one of the seven blocks. The reason comes from Ioffe and Szegedy's 2015 paper: as earlier layers update, the distribution of activations arriving at each later layer keeps sliding, a moving target they name *internal covariate shift*, and it is what forces small learning rates. Their fix normalizes each channel's activations across the current mini-batch, then lets a learned affine transform undo exactly as much of that as the layer finds useful:

$$X_\text{out} = \gamma \cdot \frac{X_\text{in} - \mu_X}{\sigma_X} + \beta$$

Both statistics, $\mu_X$ and $\sigma_X$, are channelwise, computed over the mini-batch. Ioffe and Szegedy report reaching their ImageNet baseline's accuracy in roughly 14 times fewer steps. Did batch norm buy me anything comparable on galaxies? I never ran the ablation, so I cannot say. The layers went in on the paper's word and stayed in because the deeper regression runs trained without any of the learning-rate babysitting April had needed.

<figure>
<svg viewBox="0 0 620 220" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="bn-title bn-desc">
<title id="bn-title">An offset activation distribution being re-centered to mean zero and unit variance</title>
<desc id="bn-desc">On the left, a lopsided bump sits to one side of a baseline axis. An arrow points right to a symmetric bump centered on the axis, labeled mu equals zero and sigma equals one. Schematic curves with no numeric axis ticks.</desc>
<line x1="40" y1="170" x2="270" y2="170" stroke="currentColor" stroke-width="1.5" opacity="0.6"></line>
<path d="M55 170 C120 170 130 70 175 70 C210 70 210 170 255 170" fill="currentColor" opacity="0.15" stroke="currentColor" stroke-width="2"></path>
<line x1="155" y1="170" x2="155" y2="50" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3" opacity="0.5"></line>
<text x="155" y="200" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">offset, wide</text>
<path d="M300 110 L360 110 M345 100 L362 110 L345 120" stroke="currentColor" stroke-width="2" fill="none"></path>
<line x1="390" y1="170" x2="600" y2="170" stroke="currentColor" stroke-width="1.5" opacity="0.6"></line>
<path d="M410 170 C460 170 470 60 495 60 C520 60 530 170 580 170" fill="currentColor" opacity="0.2" stroke="currentColor" stroke-width="2"></path>
<line x1="495" y1="170" x2="495" y2="40" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"></line>
<text x="495" y="200" text-anchor="middle" font-size="12" fill="currentColor">mu = 0, sigma = 1</text>
</svg>
<figcaption>Batch norm takes the drifting, offset activation distribution on the left and re-centers it to mean zero and unit variance on the right before the learned affine transform restores whatever scale the next layer actually needs.</figcaption>
</figure>

## Dropout, and where mine actually went

Dropout, from Srivastava et al. in 2014, zeroes a random fraction of a layer's units at every training step, so no unit gets to be indispensable; at test time everything stays on, scaled by the keep-probability `p`. Going in, my guess was simple. Overfitting bites where the parameters are, the parameters are densest in the 1024-unit layer feeding the head, therefore the dropout belongs there.

The committed script disagrees with my guess. Dropout sits at 0.2 after every conv block and nowhere on the dense layer at all. I do not have a clean record of why it migrated; the configuration that survived is the one that spread small dropout through the stack. Whether the head would have overfit without it is an ablation the record does not contain.

<figure>
<svg viewBox="0 0 620 320" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="dropout-title dropout-desc">
<title id="dropout-title">A fully connected network before and after dropout</title>
<desc id="dropout-desc">Panel a shows a small fully connected network with an input layer, two hidden layers, and an output, all units active and fully wired. Panel b shows the same network with a random subset of hidden units faded out and their edges removed, illustrating dropout with keep-probability p.</desc>
<g stroke="currentColor" fill="currentColor">
<text x="150" y="24" text-anchor="middle" font-size="14" stroke="none">(a) standard network</text>
<line x1="60" y1="90" x2="150" y2="70" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="60" y1="90" x2="150" y2="130" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="60" y1="90" x2="150" y2="190" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="60" y1="170" x2="150" y2="70" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="60" y1="170" x2="150" y2="130" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="60" y1="170" x2="150" y2="190" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="60" y1="250" x2="150" y2="70" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="60" y1="250" x2="150" y2="130" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="60" y1="250" x2="150" y2="190" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="150" y1="70" x2="240" y2="100" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="150" y1="70" x2="240" y2="170" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="150" y1="130" x2="240" y2="100" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="150" y1="130" x2="240" y2="170" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="150" y1="190" x2="240" y2="100" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="150" y1="190" x2="240" y2="170" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="240" y1="100" x2="270" y2="135" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="240" y1="170" x2="270" y2="135" stroke-width="1" opacity="0.5" fill="none"></line>
<circle cx="60" cy="90" r="8"></circle>
<circle cx="60" cy="170" r="8"></circle>
<circle cx="60" cy="250" r="8"></circle>
<circle cx="150" cy="70" r="8"></circle>
<circle cx="150" cy="130" r="8"></circle>
<circle cx="150" cy="190" r="8"></circle>
<circle cx="240" cy="100" r="8"></circle>
<circle cx="240" cy="170" r="8"></circle>
<circle cx="270" cy="135" r="8"></circle>
</g>
<g stroke="currentColor" fill="currentColor">
<text x="460" y="24" text-anchor="middle" font-size="14" stroke="none">(b) with dropout</text>
<line x1="370" y1="90" x2="460" y2="70" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="370" y1="90" x2="460" y2="190" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="370" y1="170" x2="460" y2="70" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="370" y1="170" x2="460" y2="190" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="370" y1="250" x2="460" y2="70" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="370" y1="250" x2="460" y2="190" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="460" y1="70" x2="550" y2="170" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="460" y1="190" x2="550" y2="170" stroke-width="1" opacity="0.5" fill="none"></line>
<line x1="550" y1="170" x2="580" y2="135" stroke-width="1" opacity="0.5" fill="none"></line>
<circle cx="370" cy="90" r="8"></circle>
<circle cx="370" cy="170" r="8"></circle>
<circle cx="370" cy="250" r="8"></circle>
<circle cx="460" cy="70" r="8"></circle>
<circle cx="460" cy="130" r="8" opacity="0.25"></circle>
<circle cx="460" cy="190" r="8"></circle>
<circle cx="550" cy="100" r="8" opacity="0.25"></circle>
<circle cx="550" cy="170" r="8"></circle>
<circle cx="580" cy="135" r="8"></circle>
</g>
<text x="310" y="305" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75" stroke="none">faded units are dropped this step; a different subset is kept with probability p each time</text>
</svg>
<figcaption>(a) the full network. (b) the same network with a random subset of hidden units zeroed and their edges gone for this training step, so no single unit can become indispensable. Layout after Figure 1 of Srivastava et al. 2014.</figcaption>
</figure>

## Six weeks of regression runs

With the head fixed, May belonged to architecture sweeps. The plain convnet's best regression run flattened at a validation MSE of 0.031 and would not go lower. The jump came from swapping the backbone for a DenseNet, a checked-in Keras implementation of last year's densely connected networks, configured with depth 64, growth rate 10, bottleneck layers, and a 0.5 compression at each transition. The DenseNet runs also picked up augmentation the earlier ones lacked, `rotation_range=180` with both flips, free here because a galaxy has no right way up. Both changes landed in the same step, so the logs cannot split the 0.031-to-0.015 drop between them; a DenseNet run without augmentation is simply not in the record. Batches stayed small, 25 for the DenseNets against the convnet's 50, with Adam starting at 1e-5 and early stopping at a patience of 25.

The record of the sweep, replotted straight from the loss CSVs:

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/03.regression-runs.png" alt="Validation MSE curves for six regression runs on a log scale, DenseNet reaching 0.0142" class="center-image">
<figcaption>Validation MSE for the regression-era runs, log scale. The plain convnet and two DenseNet configurations flatten near 0.03 to 0.045; densenet 01 descends to 0.015 over 54 epochs, and its fine-tune holds 0.0142. The dashed grey line at the bottom comes from a target-loading bug.</figcaption>
</figure>

The chart compresses a month of runs, so a few of its stories deserve prose. The run that was both smallest and lowest-resolution, 32x32 on a thousand samples, never left an MSE of 0.37; two variables moved at once there too, but it sent me up the resolution ladder first. The fast DenseNet variant at 64x64 settled at 0.045; 128x128 was the one genuine divergence, ending its eighth epoch with a validation loss above 0.6, almost four times where it began, and I still do not know why. A 256x256 sibling bottomed out at 0.025 and then spiked to 0.53 on its final epoch, which is the kind of exit that makes you grateful for `ModelCheckpoint(save_best_only=True)`. One run's very first epoch logged a training loss just over 950,000 before settling under 0.07 by epoch seven, a reminder of what an unlucky initialization does to squared error.

The run that mattered, densenet 01, walked down from 0.16 to 0.0152 over 54 epochs. Its continuation picked up from the checkpoint at 0.0142, and the log of that fine-tune is mostly a portrait of diminishing returns: `ReduceLROnPlateau` cut the learning rate from 1e-6 to 1e-9 in three steps across fourteen epochs, and validation MSE moved from 0.01421 to 0.01420.

One caveat belongs next to those digits rather than buried at the end. The DenseNet script's validation generator draws from the same arrays its training generator does. The convnet script had done this correctly, holding out a `validation_split=0.1`; the split never made it across the rewrite. Every DenseNet validation number in this chart is therefore optimistic, closer to a training fit than a held-out error.

And the dashed grey line? That was briefly the most exciting number of the spring: a convnet variant reporting 0.0027 from its first epoch, five times better than anything else ever managed. It is fake. The committed script allocates the target array as `int16`, so every vote fraction below 1.0 truncated to zero as the array filled, and the model earned its score by learning to predict near-zero against targets that mostly were zero. A first epoch that starts where other runs end is not a breakthrough; it is a dtype.

## Where it landed

Six weeks reduce to one checkpoint. `densenet_02_best.h5`, validation MSE 0.0142, which on the vote fractions is an RMSE of 0.119. For scale I have two anchors. Kaggle's central-pixel benchmark, basically averaging the votes of training galaxies with similar core colors, scored 0.16194. Sander Dieleman won the 2014 challenge at a private-leaderboard RMSE of 0.07492. My checkpoint sits between those two, knowing more about a galaxy than its central colors and a great deal less than the winner did.

All of it is bounded by the validation leak above, so porting the convnet's `validation_split` back over is the first thing on the list. And 0.119 is a validation number in any case; lining it up against 0.07492 honestly would take the test-set submission I have not made. Here is the narrow thing the six weeks actually bought. The same pixels that spent all of April refusing to classify past 0.75 will happily regress a 37-value crowd vote, as soon as the head stops demanding a single winner.

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
