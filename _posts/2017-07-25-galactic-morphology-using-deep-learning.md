---
layout: single
title: "Convolutional networks - regressing Galaxy Zoo morphology vectors from raw pixels"
date: 2017-07-25T15:39:55-04:00
last_modified_at: 2017-07-25T19:48:19-04:00
categories: [blog]
tags: [deep learning, galaxy, computer-vision, machine learning, neural networks]
excerpt: "A build writeup: setting up a CNN to regress a 37-value Galaxy Zoo morphology vector straight from raw pixels."
math: true
redirect_from:
  - /blog/galactic-morphology-using-deep-learning/
header:
  overlay_image: /images/blog/headers/galactic-morphology-using-deep-learning.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/galactic-morphology-using-deep-learning.jpg
---

Since 2007, thousands of volunteers have looked at 100k+ galaxy images and answered a tree of questions about each one: is it smooth or does it have features, is there a spiral, how many arms. The answers for each galaxy get collapsed into 37 numbers, a record of what a crowd of human eyes saw. Those 37 numbers are the catch. They are not one label but a graded vote: each value sits between 0 and 1, the answers deeper in the tree are scaled by the ones above them, so the values are correlated and the sub-scores for a single question need not sum to 1. Reach for the obvious image classifier and you throw all of that away. An argmax picks a single winning class and discards the graded, multi-answer structure that was the entire reason for asking volunteers 37 questions instead of one. The build question I want to settle is narrow: can a convolutional net reproduce those same 37 numbers from raw pixels, with no astronomer hand-crafting a single feature along the way? I assemble the network one piece at a time and explain what each piece is for as it goes in. There is no trained accuracy or loss here; the content is the architecture and the data. The training run and its loss curve live in the companion code.

I want the finished machine on the table before we open it up. Here is the whole pipeline; each section below opens up one box of it.

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

That is the whole of it. Each box below gets its own section, left to right, from the input block to the strip of 37 numbers. One image runs through all of them as the thread, so before I touch any layer I should say exactly what that image is and where it comes from.

The data comes from the [Sloan Digital Sky Survey](http://www.sdss.org/), the imaging project behind the citizen-science effort. To quote the SDSS site:

> The Sloan Digital Sky Survey has created the most detailed three-dimensional maps of the Universe ever made, with deep multi-color images of one third of the sky, and spectra for more than three million astronomical objects.

The volunteer scoring happened through [Galaxy Zoo](https://www.galaxyzoo.org), launched in 2007, where thousands of people classified 100k+ images of galaxies by walking a decision tree of questions.

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/00.galaxyzoo-tree.png" alt="Galaxy Zoo decision tree of classification questions" class="center-image">
<figcaption>The Galaxy Zoo decision tree, from Galaxy Zoo 2 (Willett et al. 2013). Each volunteer walks these branching questions for an image, and the weighted answers become the 37-value target the network has to predict.</figcaption>
</figure>

## The input box: a 424x424x3 jpeg and a 37-vote vector

The leftmost block in the pipeline is the input, and the rightmost strip is the target. Fix both ends first, before anything in between exists. The training set for the Galaxy Zoo challenge is roughly 60k labeled jpegs. Each image carries a score vector of 37 values, where each value is the weighted score from volunteers on one question in the tree.

Each value is a weighted vote score between 0 and 1. Because the answers deeper in the tree are down-weighted by the ones above them, the sub-scores for a single question do not have to sum to 1 the way a probability distribution would. That detail decides the loss and the output head later, and I will come back to it once the diagram has grown that far.

Every image is `424×424×3` and each value sits between `0` and `255`. Before the image hits the network I rescale it, computing $\mu_\text{channel}$ and $\sigma_\text{channel}$ over the full dataset and then normalizing each channel using its own $\mu$ and $\sigma$. The pixels carry no physical units to preserve, so this normalization is a gradient-update convenience and nothing more.

<figure>
<img src="{{ site.baseurl }}/images/blog/galactic-morphology-using-deep-learning/01.galaxies.png" alt="A grid of sample galaxy images from the dataset" class="center-image">
<figcaption>Sample galaxies from the dataset, SDSS image cutouts. Pick any one of these as the image we follow through the network: a single 424x424x3 jpeg that has to come out the other end as 37 numbers.</figcaption>
</figure>

Once I pick one of those galaxies, it is read into Python as a `numpy` array, and that array is the literal substrate every box downstream operates on. Here is what the input block in the pipeline diagram unfolds into.

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

## The output box: why regression, not classification

The head is the box whose shape the input data dictated, so I settle it before filling in the conv stack between the two ends. A classifier would use one-vs-rest encoding: the correct class gets `1`, every other class gets `0`. The output is a vector of length `c` for `c` classes. Clean, but it assumes there is one right answer per image.

That assumption does not hold. The ground truth is a weighted vote vector, so I set the head up as a 37-dimensional regression: 37 continuous outputs, linear, no softmax, trained to match the human vote scores directly, sub-1 sums and all. My first pass reflexively reached for a `softmax` over the 37 outputs. That is exactly wrong here. Softmax does not leave the votes alone; it renormalizes the whole vector to sum to 1, and the votes do not.

Both ends are now specified: a 424x424x3 array on the left, 37 linear outputs on the right. What is left is the middle, the stack of layers that carries one end into the other.

## The conv and pool blocks: what each block does to the array

Now the array actually moves, and it is worth watching one block do it rather than trusting the diagram's shrinking boxes. Follow our galaxy in as 424x424x3. A block is two operations. The **convolutional layer** slides learned `3×3` kernels across the array (same-padded, so height and width survive) and stacks their responses into channels, turning 424x424x3 into 424x424xC for a block with C filters: same grid, more depth. Those kernels are the point. In a classic pipeline an astronomer hand-picks the features; here the kernels are fit against the 37-value target, so nobody encodes what a spiral arm looks like and the filters learn whichever transforms best drive the prediction.

The **pooling layer** then halves the grid. A `2×2` max-pool keeps the largest value in each window and drops the other three, taking 424x424xC down to 212x212xC; that halving is why the boxes shrink left to right, and it hands the next block a quarter as many positions to scan while the strongest responses survive. Stack a few blocks and 424 falls to 212 to 106 to 53, usually with more channels as the grid contracts.

Between the layers sits an elementwise **activation**, `relu` inside the stack to keep gradients flowing; the head is the exception, kept linear for the reason the output box already settled.

That is the visible skeleton. Two more pieces ride inside these blocks without ever getting their own box, dropout and batch normalization, and each earns its own section.

## Batch normalization between the blocks

This piece only shows up once training is underway. As the parameters of earlier layers update, the distribution of inputs landing on each later layer keeps sliding out from under it, and Ioffe and Szegedy's 2015 batch-norm paper names that moving target *internal covariate shift*. On their account it is what forces you into small learning rates to keep training stable, and batch normalization is the fix: normalize each channel's activations to $\mu_x = 0$ and $\sigma_x = 1$ across the current mini-batch, then apply a learned affine transform.

$$X_\text{out} = \gamma \cdot \frac{X_\text{in} - \mu_X}{\sigma_X} + \beta$$

Here $\mu_X$ and $\sigma_X$ are computed channelwise, the same axis I standardized the raw pixels along at the input, except now it happens between conv blocks in this stack and the statistics come from the live mini-batch instead of the whole dataset. With each channel re-centered, the next conv block stops chasing a target that slides every time the layers below it update; it sees a stable distribution and can commit to sharper weights instead of re-adapting each step. That is what buys back the larger learning rates. On their ImageNet network, Ioffe and Szegedy report reaching the same accuracy in roughly 14 times fewer training steps with batch normalization than without. In this net the same normalization sits between the conv blocks that feed the 37-value head, where the vote scores have to stay resolvable through the whole depth of the stack.

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

## Dropout on the dense head

My working guess is that overfitting will bite first where the parameters are densest. In this net that is the fully connected head just before the 37-value strip, so that is where the dropout goes. Whether the guess holds is something only the train-versus-validation gap can settle later. The mechanism is Srivastava et al.'s 2014 dropout: at training time it sets a randomly chosen fraction of the head's units to `zero` each step; at test time every unit is kept, its output scaled by the keep-probability `p` so the expected magnitude matches training. A different subset drops each step. No single unit can carry the representation on its own, so the head is pushed to spread the 37-value prediction across many units.

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
<figcaption>(a) the full network. (b) the same network with a random subset of hidden units zeroed and their edges gone for this training step, so no single unit can become indispensable.</figcaption>
</figure>

## The assembled network and what I would watch next

Every piece now has a name: the input array on the left, the conv-and-pool blocks shrinking through the middle, batch norm sitting between them, dropout on the dense head, and the 37-value regression head on the right. Together they stack into a module I can sweep across several network structures, starting from a VGG-style stack of small 3x3 convolutions and moving toward residual ResNet-style blocks. The bet with the deeper residual variants is that skip connections let me add depth without the gradient dying on the way back, which matters when the head has to resolve 37 separate vote scores at once. The final layers are `Dense` (fully connected), and their output is compared against the expected 37-value vector to compute a per-observation loss, back-propagated to update both the convolutional kernels and the dense weights.

The limit is just as plain: everything I have drawn here is the data and the architecture, the setup for the experiment, and it deliberately stops short of a trained accuracy or a final loss. Once the net is trained, the first number I want is the gap between train and validation MSE on the 37-value vector, which is where the dropout on the dense head earns its place or does not. What the whole diagram leaves on the table is the thing that drew me in: 37 numbers a crowd of humans voted on, predicted straight from a 424x424x3 jpeg.

The network defined here, the data loaders, and the training runs that produce the loss curve live in the companion repo.

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
