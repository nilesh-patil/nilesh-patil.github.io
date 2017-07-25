---
layout: single
title: "Classifying galaxy morphology with a CNN"
date: 2017-07-25T15:39:55-04:00
last_modified_at: 2017-07-25T19:48:19-04:00
categories: [blog]
tags: [deep learning, galaxy, computer-vision, machine learning, neural networks]
excerpt: "A build writeup: setting up a CNN to regress a 37-value Galaxy Zoo morphology vector straight from raw pixels."
math: true
redirect_from:
  - /blog/galactic-morphology-using-deep-learning/
header:
  overlay_image: /images/blog/headers/galaxy.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/galaxy.jpg
---

Since 2007, thousands of volunteers have looked at 100k+ galaxy images and answered a tree of questions about each one: is it smooth or does it have features, is there a spiral, how many arms. The answers for each galaxy get collapsed into 37 numbers, a record of what a crowd of human eyes saw. That record is what makes this a tractable supervised-learning problem in the first place. The build question I want to settle here is narrow and concrete: can a convolutional net reproduce those same 37 numbers from raw pixels, with no astronomer hand-crafting a single feature along the way? I am going to assemble the network one piece at a time and explain what each piece is for as it goes in. There is no trained accuracy or loss in this post. The content is the architecture and the data, not a benchmark, and the loss curve is the next post.

So rather than walk a single image straight through and narrate whatever it hits, I want to put the finished machine on the table first and then take it apart. Here is the whole pipeline, the thing every later section will be adding a labeled part to.

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

That is the whole of it. Every section below picks one labeled box out of this diagram and explains what it does and why it has to be there, starting from the block on the far left and ending at the strip of 37 numbers on the far right. The single image I follow through the network is the running thread that ties those boxes together, so before I touch any layer I should be precise about what that image actually is and where it comes from.

The data comes from the [Sloan Digital Sky Survey](http://www.sdss.org/), the imaging project behind the citizen-science effort. To quote the SDSS site:

> The Sloan Digital Sky Survey has created the most detailed three-dimensional maps of the Universe ever made, with deep multi-color images of one third of the sky, and spectra for more than three million astronomical objects.

The volunteer scoring happened through [Galaxy Zoo](https://www.galaxyzoo.org), launched in 2007, where thousands of people classified 100k+ images of galaxies by walking a decision tree of questions.

<figure>
<img src="/images/blog/galaxyzoo/00.galaxyzoo-tree.png" alt="Galaxy Zoo decision tree of classification questions" class="center-image">
<figcaption>The Galaxy Zoo decision tree. Each volunteer walks these branching questions for an image, and the weighted answers become the 37-value target the network has to predict.</figcaption>
</figure>

## The input box: a 424x424x3 jpeg and a 37-vote vector

The leftmost block in the pipeline is the input, and the rightmost strip is the target. Pin down both ends first, before anything in between exists. The dataset is 100k+ jpeg images plus, for each image, a score vector with 37 values, where each value is the weighted score from volunteers on one question in the tree.

These targets are not probabilities, which is the first thing that shapes the design of the box at the far right. They are weighted vote scores, so each value runs from 0 to 1, but the sub scores for a single question do not necessarily sum up to 1. That detail decides the loss and the output head later, and I will come back to it once the diagram has grown that far.

Every image is `424×424×3` and each value sits between `0` and `255`. Before the image hits the network I rescale it, computing $\mu_\text{channel}$ and $\sigma_\text{channel}$ over the full dataset and then normalizing each channel using its own $\mu$ and $\sigma$. The channels and cell values here do not carry physical meaning. They are standardized to the `0`-`255` image range, so this normalization is a gradient-update convenience, not a domain-informed transform.

<figure>
<img src="/images/blog/galaxyzoo/01.galaxies.png" alt="A grid of sample galaxy images from the dataset" class="center-image">
<figcaption>Sample galaxies from the dataset. Pick any one of these as the image we follow through the network: a single 424x424x3 jpeg that has to come out the other end as 37 numbers.</figcaption>
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

The rightmost box in the pipeline is a regression head rather than the classifier you might expect, and it is the box whose shape the input data dictated. Pin it down before I fill in the conv stack between the two ends. A convolutional network reads the image array, builds up features from it, and emits an output. The usual classification setup uses a one-vs-rest encoding, where the correct class is assigned `1`, every other class is assigned `0`, and the output is a vector of length `c` for `c` classes. That is clean, but it assumes there is one right answer per image.

That assumption does not hold here, because the ground truth is a weighted vote vector rather than a single label, so I set this up as a 37-dimensional regression instead. The network outputs a vector of length 37 and is trained to match the human vote scores directly, sub-1 sums and all, which means the target shape dictates the head: 37 continuous outputs, linear, no softmax. My first pass reflexively reached for a softmax over the 37 outputs. That is exactly wrong here: softmax does not leave the votes alone, it renormalizes the whole vector to sum to 1, and the votes do not.

With both ends of the pipeline now nailed down, a 424x424x3 array on the left and 37 linear outputs on the right, the rest of the diagram is the stack of layers that carries one into the other.

## The conv and pool blocks: what each one does to our image

The normalized array flows through the chain of conv-and-pool blocks in the middle of the diagram, shrinking as it goes, and each block is really two distinct operations worth separating. Here is what each one does to the array as it passes through.

The **convolutional layer** is where the learned features live. In a classic modeling pipeline the features are hand-crafted, with an expert deciding which transforms of the raw input matter, and the convolution kernels replace that step entirely. They are fit against the target, so instead of an astronomer choosing what counts as a spiral arm, the kernels learn whichever filters best drive the 37-value prediction.

The **pooling layer** shrinks the representation, which is the part of the block that accounts for the diagram's boxes getting smaller from left to right. `max-pooling` keeps the largest value in each window as the summary of that region and `average-pooling` keeps the mean, but either way the spatial footprint goes down while the salient signal survives, which leaves fewer spatial positions for the next conv block to scan over.

The **activation** introduces nonlinearity by applying a function elementwise to the array, and which function it is depends on where it sits in the diagram. I use `relu` between conv blocks to keep gradients flowing, while the output head stays linear because this is regression onto continuous vote scores rather than a `softmax` over mutually exclusive classes. Softmax would force the 37 outputs to sum to 1, and these vote scores do not.

That is the visible skeleton of the pipeline. Two more pieces ride inside these blocks without showing up as their own boxes in the diagram, dropout and batch normalization, and each of them earns its own section.

## Batch normalization between the blocks

The first of the two hidden pieces sits between the conv blocks, and the reason it needs to be there is something that only shows up once training is underway. As training proceeds, the parameters of earlier layers keep changing, so the distribution of inputs landing on each later layer keeps moving underneath it. Ioffe and Szegedy called that moving target *internal covariate shift*, and argued it pushes you toward small learning rates to keep things stable. Whether that is the precise reason batch norm helps has since been debated, but the fix is the same: batch normalization normalizes each channel's activations to $\mu_x = 0$ and $\sigma_x = 1$ across the current mini-batch, then applies a learned affine transform.

$$X_\text{out} = \gamma \cdot \frac{X_\text{in} - \mu_X}{\sigma_X} + \beta$$

Here $\mu_X$ and $\sigma_X$ are computed channelwise, the same axis I standardized the raw pixels along at the input, except now it happens between conv blocks in this stack and the statistics come from the live mini-batch instead of the whole dataset. With the input distribution re-centered, downstream layers see something stable to learn against, which is what buys back the larger learning rates and faster training than the same network without batch normalization.

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

The second hidden piece lives at the far end of the diagram, on the dashed dense box just before the 37-value strip, which is where the parameter count is highest and overfitting bites first. Dropout is a regularizer for overfitting, and in this build it sits exactly there. At training time, the activations of a randomly chosen fraction of units are set to `zero`, and at test time every unit is kept, with its activations scaled by the keep-probability `p` so the expected magnitude matches what later layers saw during training. Because a different random subset is dropped at each step, no single unit can carry the representation on its own, so the dense head is pushed to spread the 37-value prediction across many units.

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

With every box in the diagram now labeled, the input array on the left, the conv-and-pool blocks shrinking through the middle, batch norm sitting between them, dropout on the dense head, and the 37-value regression head on the right, the layers stack into a module that I can sweep across several network structures. I start from a VGG-style stack of small 3x3 convolutions and move toward residual ResNet-style blocks. The hope with the deeper residual variants is that skip connections let me add depth without the gradient dying on the way back, which matters when the head has to resolve 37 separate vote scores rather than pick one class. The final layers are `Dense` (fully connected), and their output is compared against the expected 37-value vector to compute a per-observation loss, which is back-propagated to update both the convolutional kernels and the dense weights.

Looking back over the assembled diagram, the wider point is that the convolution kernels in the middle stand in for an astronomer's hand-crafted features: nobody encodes what a spiral arm looks like, and the filters are fit against the votes instead. The limit is just as plain, because everything I have drawn here is the data and the architecture, the setup for the experiment, and it deliberately stops short of a trained accuracy or a final loss. The number I want to watch first, once the diagram is trained, is the gap between train and validation MSE on the 37-value vector, which is where the dropout on the dense head earns its place or does not. What this whole diagram leaves on the table is the framing itself, 37 numbers that humans voted on, predicted straight from a 424x424x3 jpeg.

## References

1. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). *Journal of Machine Learning Research*, 15.
2. Ioffe, S. and Szegedy, C. (2015). [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). *Proceedings of the 32nd International Conference on Machine Learning*.
