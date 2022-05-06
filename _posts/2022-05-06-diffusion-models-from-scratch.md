---
layout: single
title: "Diffusion models - generating images by learning to remove noise"
date: 2022-05-06T21:38:47+05:30
categories: [blog]
tags: [machine-learning, generative-models, diffusion, pytorch, deep-learning]
excerpt: "A from-scratch denoising diffusion model that turns pure static into MNIST digits on a laptop, then the exact ladder of papers - DDIM, guidance, latent diffusion - that turned the same idea into DALL-E 2 three weeks ago."
math: true
header:
  overlay_image: /images/blog/headers/diffusion-models-from-scratch.jpg
  overlay_filter: 0.5
  teaser: /images/blog/headers/diffusion-models-from-scratch.jpg
---

Every digit in the banner above started as pure static. No digit was retrieved, copied, or stitched together from training images. A small neural network looked at random noise and, in `~50` steps, removed its way to a handwritten number. You can replicate the full exercise locally, the whole thing is about three hundred lines of PyTorch.

A diffusion model is a [generative model](https://openai.com/index/generative-models/) that learns to reverse a gradual noising process. You take real data, destroy it by adding noise in small steps until nothing is left, and train a network to undo one step of that destruction. To generate something new, you hand the network pure noise and let it walk the destruction backwards. That is the entire idea, and it now produces the best image generation that the field has managed till now!

Three weeks ago OpenAI showed [DALL·E 2](https://arxiv.org/abs/2204.06125), where you write a caption like "Picard riding on Voyager, through Delta Quadrant's unique planets" and the model "paints" it. Underneath the headline results sits the same denoising process loop I am about to build using MNIST. So this post has two parts :

- First I build a denoising diffusion model from scratch and watch it hallucinate digits and clothing
- Then I walk the short, fast ladder of papers, all published between 2020 and last month, that carried this idea from "interesting on CIFAR" to DALL·E 2.

The companion code is at [`ddpm-from-scratch`](https://github.com/nilesh-patil/ddpm-from-scratch). Every figure below comes out of it.

## The whole idea in one picture

There are two processes - One is deterministic noise addition and the other one is learned denoising.

The **forward process** takes a clean image and adds a small amount of Gaussian noise, then adds a little more, and a little more, for $T$ steps, until the image is indistinguishable from static. This process has nothing to learn. It is a deterministic recipe with a fixed schedule of how much noise to add at each step, chosen before training started.

The **reverse process** is a neural network. It learns to look at a noisy image and undo one step of the forward process, nudging it back toward something slightly cleaner. Stack a thousand of these nudges and we walk all the way from static back to a plausible image.

<figure>
  <svg viewBox="0 0 640 214" role="img" aria-labelledby="chain-t chain-d" style="width:100%;height:auto;max-width:640px;color:inherit" xmlns="http://www.w3.org/2000/svg">
    <title id="chain-t">The forward and reverse diffusion chains</title>
    <desc id="chain-d">Five states in a row, from a clean image x-zero on the left to pure noise x-T on the right. Forward arrows along the top add a little Gaussian noise at each step and are fixed, with no learned parameters. Reverse arrows along the bottom remove noise one step at a time and are produced by a trained network.</desc>
    <defs>
      <marker id="chain-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="currentColor"></path>
      </marker>
    </defs>
    <g fill="none" stroke="currentColor" stroke-width="1.6">
      <rect x="36" y="74" width="56" height="56" rx="8"></rect>
      <rect x="164" y="74" width="56" height="56" rx="8" stroke-dasharray="7 3"></rect>
      <rect x="292" y="74" width="56" height="56" rx="8" stroke-dasharray="4 4"></rect>
      <rect x="420" y="74" width="56" height="56" rx="8" stroke-dasharray="2 5"></rect>
      <rect x="548" y="74" width="56" height="56" rx="8" stroke-dasharray="1 6"></rect>
    </g>
    <g fill="none" stroke="currentColor" stroke-width="1.4">
      <path d="M96,90 L160,90" marker-end="url(#chain-ah)"></path>
      <path d="M224,90 L288,90" marker-end="url(#chain-ah)"></path>
      <path d="M352,90 L416,90" marker-end="url(#chain-ah)"></path>
      <path d="M480,90 L544,90" marker-end="url(#chain-ah)"></path>
      <path d="M160,114 L96,114" marker-end="url(#chain-ah)" opacity="0.65"></path>
      <path d="M288,114 L224,114" marker-end="url(#chain-ah)" opacity="0.65"></path>
      <path d="M416,114 L352,114" marker-end="url(#chain-ah)" opacity="0.65"></path>
      <path d="M544,114 L480,114" marker-end="url(#chain-ah)" opacity="0.65"></path>
    </g>
    <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif" text-anchor="middle" aria-hidden="true">
      <text x="320" y="26" font-size="13" font-weight="600">forward: add a little Gaussian noise &#183; fixed, no parameters</text>
      <text x="64" y="156" font-size="13">x&#8320;</text>
      <text x="192" y="156" font-size="13">x&#8321;</text>
      <text x="320" y="156" font-size="13">x&#8348;</text>
      <text x="448" y="156" font-size="13">&#8230;</text>
      <text x="576" y="156" font-size="13">x&#8348;</text>
      <text x="64" y="174" font-size="10.5" opacity="0.7">clean image</text>
      <text x="576" y="174" font-size="10.5" opacity="0.7">pure noise</text>
      <text x="320" y="202" font-size="13" font-weight="600">reverse: a network predicts the noise and removes it &#183; learned</text>
    </g>
  </svg>
  <figcaption>The two chained processes : 
  
  - Going right (top arrows) is the fixed forward process: each step adds a touch of Gaussian noise, and the box outlines dissolve from solid to dotted to stand in for the image dissolving into static. 
  - Going left (bottom arrows) is the learned reverse process: a single network, applied over and over, predicts the noise in its input & we subtract it from the input. Out training teaches this network to undo one step; sampling chains the step a thousand times.</figcaption>
</figure>

The trick that makes this trainable with relatively small compute is that we don't ***have to*** run the forward process step-by-step during training. Since each step adds Gaussian noise, and Gaussians stack into Gaussians, we can jump straight to the noise level at any step `t` in one shot.

## Step 1: destroy an image by design

Let the noise schedule be a sequence of small numbers $\beta_1, \dots, \beta_T$ (I used $T = 1000$). Define $\alpha_t = 1 - \beta_t$, and let $\bar{\alpha}_t$ be the running product of every $\alpha$ up to step `t`:

$$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

Then the noised image at step `t`, given the clean image $x_0$, has a closed form:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\; \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

Visualize it as a slider, i.e. $\bar{\alpha}_t$ starts near 1 and decays toward 0 as `t` grows, so early on you keep most of the image and add a whisper of noise, and late on you keep almost none of the image and it is nearly all noise. There is no network in this equation. It is arithmetic, and it is the whole forward process. In code it is two lines:

```python
def q_sample(self, x0, t, noise):

    # jump straight to the noise level at step t ( reparameterization )

    A = extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
    B = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

    return ( A + B )
```

Here is a real `3` from MNIST run through `q_sample` at increasing `t`. Watch it dissolve step by step, as we add more noise.

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/forward_strip_mnist.png" alt="A handwritten 3 shown at seven noise levels, crisp at t=0 and pure static at t=999." loading="lazy">
  <figcaption>
  The forward process on one digit :

  - At t=0 it is the clean image
  - by t=999 it is indistinguishable from Gaussian noise. Notice how much structure survives even at t=400
  - this gradual change is a deliberate choice of our chosen schedule
  </figcaption>
</figure>

That schedule deserves a second look, because the first thing the field improved after the original model was exactly this curve. The original [DDPM](https://arxiv.org/abs/2006.11239) paper used a **linear** schedule, where $\beta_t$ grows linearly. A year later, [Improved DDPM](https://arxiv.org/abs/2102.09672) pointed out that on small images the linear schedule destroys the picture too fast, so the last few hundred steps are nearly pure noise and teach the network almost nothing. Their **cosine** schedule keeps signal around for longer. I use the cosine schedule, and you can see why below.

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/schedule.svg" alt="Plot of signal retained against diffusion step for linear and cosine schedules; the cosine curve stays high much longer." loading="lazy">
  <figcaption>How fast each schedule destroys the image, measured as $\bar{\alpha}_t$, the fraction of the original signal retained. The linear schedule (DDPM, 2020) collapses to near zero by the two-thirds mark; the cosine schedule (Improved DDPM, 2021) keeps a usable amount of signal almost the whole way. More of the chain stays informative, so more of it contributes a useful training signal.</figcaption>
</figure>

## Step 2: train a network to undo one step

Now the only learned part. We want a network that, given a noisy image $x_t$ and the step `t`, removes noise. There is a choice to make about what exactly it should output, the clean image or the noise that was added, and DDPM picks **the noise**. The network $\epsilon_\theta(x_t, t)$ guesses the $\epsilon$ that was added, and the loss is just the mean squared error between the true noise and the guess.

$$L = \mathbb{E}_{t,\, x_0,\, \epsilon}\, \big\lVert\, \epsilon - \epsilon_\theta(x_t,\, t)\, \big\rVert^2$$

This drops out of a variational bound on the data likelihood, with a few weighting terms thrown away because the simplified version trains better. I am skipping that derivation on purpose; it is in the DDPM paper if you want it, and the practical loss is what matters here. The training step is short:

```python
def p_losses(self, model, x0, t):
    noise = torch.randn_like(x0)            # the target
    x_t = self.q_sample(x0, t, noise)       # noise the image to level t
    predicted = model(x_t, t)               # ask the network to guess the noise
    return F.mse_loss(predicted, noise)     # how wrong was it
```

The full training loop is the loop you already know from any supervised model, with one extra line that picks a random noise level per image:

```python
for x0, _ in dataloader:
    t = torch.randint(0, T, (x0.size(0),))   # a different noise level per image
    loss = diffusion.p_losses(model, x0, t)
    opt.zero_grad(); loss.backward(); opt.step()
    ema.update(model)                        # keep a moving average of the weights
```

The `ema.update` line earns its keep. Sampling from an exponential moving average of the weights rather than the live weights noticeably cleans up the samples in my runs, and it costs nothing but a copy of the parameters.

What about the network itself?

It is a **U-Net**, the same encoder-decoder-with-skip-connections shape used everywhere in image-to-image projects, with one diffusion-specific addition: the step `t` is turned into a sinusoidal embedding and injected into every block, so the same weights can behave differently at high noise and low noise. The multi-scale shape is deliberate: denoising is at once a global-structure and a local-texture job, and the skip connections let the network settle the broad shape in the bottleneck while carrying fine detail straight across, the coarse-to-fine behavior the sampling figures will show. Mine is about 10M parameters, small by any standard. We're not going to walk through U-Net line by line; check it out in [`unet.py`](https://github.com/nilesh-patil/ddpm-from-scratch/blob/main/ddpm/unet.py)

<figure>
  <svg viewBox="0 0 720 364" role="img" aria-labelledby="unet-t unet-d" style="width:100%;height:auto;max-width:720px;color:inherit" xmlns="http://www.w3.org/2000/svg">
    <title id="unet-t">A U-Net for diffusion: encoder, bottleneck, decoder, skip connections, and timestep injection</title>
    <desc id="unet-d">A symmetric funnel read left to right. On the left a tall box is the noisy input image; boxes shrink in height step by step through the encoder as spatial resolution drops, reaching the smallest box at the centre, the bottleneck. Boxes then grow back to full height through the decoder, ending in the predicted-noise output on the right. Three arcs over the top are skip connections that link each encoder box to the decoder box at the same resolution, carrying fine detail straight across. At the bottom a single node holds the timestep t turned into a sinusoidal embedding, with faint lines fanning up into every box.</desc>
    <defs>
      <marker id="unet-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="currentColor"></path>
      </marker>
      <marker id="unet-ah-sm" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="currentColor"></path>
      </marker>
    </defs>
    <g fill="none" stroke="currentColor" stroke-width="0.75">
      <path d="M60,110 C60,26 660,26 660,110" marker-end="url(#unet-ah)"></path>
      <path d="M160,129 C160,52 560,52 560,129" marker-end="url(#unet-ah)"></path>
      <path d="M260,145 C260,80 460,80 460,145" marker-end="url(#unet-ah)"></path>
    </g>
    <g fill="none" stroke="currentColor" stroke-width="1.6">
      <rect x="38" y="110" width="44" height="120" rx="5"></rect>
      <rect x="138" y="129" width="44" height="82" rx="5"></rect>
      <rect x="238" y="145" width="44" height="50" rx="5"></rect>
      <rect x="338" y="156" width="44" height="28" rx="5"></rect>
      <rect x="438" y="145" width="44" height="50" rx="5"></rect>
      <rect x="538" y="129" width="44" height="82" rx="5"></rect>
      <rect x="638" y="110" width="44" height="120" rx="5"></rect>
    </g>
    <g fill="none" stroke="currentColor" stroke-width="1.4">
      <path d="M84,170 L134,170" marker-end="url(#unet-ah)"></path>
      <path d="M184,170 L234,170" marker-end="url(#unet-ah)"></path>
      <path d="M284,170 L334,170" marker-end="url(#unet-ah)"></path>
      <path d="M384,170 L434,170" marker-end="url(#unet-ah)"></path>
      <path d="M484,170 L534,170" marker-end="url(#unet-ah)"></path>
      <path d="M584,170 L634,170" marker-end="url(#unet-ah)"></path>
    </g>
    <g fill="none" stroke="currentColor" stroke-width="0.8" opacity="0.34">
      <path d="M360,306 L60,231" marker-end="url(#unet-ah-sm)"></path>
      <path d="M360,306 L160,212" marker-end="url(#unet-ah-sm)"></path>
      <path d="M360,306 L260,196" marker-end="url(#unet-ah-sm)"></path>
      <path d="M360,306 L360,185" marker-end="url(#unet-ah-sm)"></path>
      <path d="M360,306 L460,196" marker-end="url(#unet-ah-sm)"></path>
      <path d="M360,306 L560,212" marker-end="url(#unet-ah-sm)"></path>
      <path d="M360,306 L660,231" marker-end="url(#unet-ah-sm)"></path>
    </g>
    <g fill="none" stroke="currentColor" stroke-width="1">
      <rect x="294" y="306" width="120" height="26" rx="12"></rect>
      <path d="M306,319 q5,-8 10,0 q5,8 10,0" stroke-width="1"></path>
    </g>
    <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif" text-anchor="middle" aria-hidden="true">
      <text x="360" y="15" font-size="12.5" font-weight="600">skip connection &#183; fine detail carried straight across</text>
      <text x="210" y="161" font-size="10.5" opacity="0.72">encoder</text>
      <text x="510" y="161" font-size="10.5" opacity="0.72">decoder</text>
      <text x="360" y="126" font-size="11.5" font-weight="600">bottleneck</text>
      <text x="360" y="139" font-size="10" opacity="0.72">coarsest scale &#183; global structure</text>
      <text x="60" y="251" font-size="11.5">noisy image x&#8348;</text>
      <text x="60" y="265" font-size="10" opacity="0.7">input</text>
      <text x="660" y="251" font-size="11.5">predicted noise &#949;&#770;</text>
      <text x="660" y="265" font-size="10" opacity="0.7">output</text>
      <text x="334" y="323" font-size="11" text-anchor="start">step t</text>
      <text x="360" y="350" font-size="10.5" opacity="0.78">sinusoidal embedding, added into every block</text>
    </g>
  </svg>
  <figcaption>U-Net structure - read left to right :

  - **Encoder** shrinks the image down to the **bottleneck**, where the network settles the coarse, global shape
  - **Decoder** grows it back to a full-resolution noise prediction
  - The three arcs on top are **skip connections**. Each hands the decoder the same-resolution feature map from the encoder, so fine detail travels straight across instead of being squeezed through the bottleneck.
  - The timestep `t` becomes a sinusoidal embedding and is added into every block (faint lines), which is what lets one set of weights behave differently at high noise and at low noise.</figcaption>
</figure>

Trained on MNIST for forty epochs, the loss falls fast and then crawls. The crawl is fine; diffusion loss values are a poor proxy for sample quality, and the samples keep improving long after the number stops moving.

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/loss.svg" alt="Training loss against step for MNIST and Fashion-MNIST, both falling quickly then flattening." loading="lazy">
  <figcaption>Noise-prediction MSE during training, smoothed. The curve flattens early, but the sample grids keep sharpening for many epochs after that, one of the awkward things about training generative models by this objective.</figcaption>
</figure>

## Step 3: sample by denoising pure noise

Training taught the network to undo one step. Sampling chains it. Start from pure Gaussian noise $x_T$, ask the network for the noise, form a slightly cleaner mean by subtracting it, add back fresh randomness scaled by the step's variance, and step down to $x_{t-1}$. (That variance is fixed by the schedule here; Improved DDPM later made it learnable, which is the `log_var` term in the code below.) That last bit of randomness is what keeps each run different; drop it and the reverse process turns deterministic, which is exactly the trick DDIM turns into a feature later. The reverse step in code, where the subtraction lives inside `p_mean_variance`:

```python
@torch.no_grad()
def p_sample(self, model, x_t, t):
    mean, log_var, _ = self.p_mean_variance(model, x_t, t)  # uses eps_theta
    noise = torch.randn_like(x_t)
    nonzero = (t != 0).float().reshape(-1, 1, 1, 1)          # no noise on the last step
    return mean + nonzero * (0.5 * log_var).exp() * noise
```

Run that from `t = 999` down to `t = 0` and an image condenses out of the static. This is the figure that made diffusion click for me: a single sample, photographed every hundred steps as it resolves from noise into a digit.

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/reverse_trajectory_mnist.png" alt="A strip showing pure noise on the left gradually resolving into a clean handwritten 6 on the right." loading="lazy">
  <figcaption>The reverse process, one sample over the full chain. The left frame is the noise we started from; each frame to the right is a hundred denoising steps later. A digit's rough shape commits surprisingly early, around the middle of the chain, and the late steps are mostly cleanup. **The final frame is not real data, it was generated** :) </figcaption>
</figure>

Do this for sixty-four independent noise samples and you get a sheet of digits, none of which exist in MNIST.

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/samples_mnist.png" alt="An 8 by 8 grid of varied, legible generated handwritten digits." loading="lazy">
  <figcaption>Sixty-four samples generated using our trained model</figcaption>
</figure>

## Retraining on clothes

A fair worry at this point is that MNIST is a pushover and the model memorized ten shapes. So I changed one command-line flag, `--dataset fashion`, retrained the identical architecture on Fashion-MNIST, and changed nothing else. Same schedule, same loss, same sampler.

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/samples_fashion.png" alt="An 8 by 8 grid of generated Fashion-MNIST images: shirts, trousers, bags, shoes, and coats." loading="lazy">
  <figcaption>Fashion-MNIST samples, produced by the unchanged code with only the dataset flag flipped. Shirts, trousers, sneakers, bags, and ankle boots, all generated from noise. The textures (ribbing on a pullover, the sole of a shoe) are harder than digit strokes, and the model mostly gets them.</figcaption>
</figure>

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/reverse_trajectory_fashion.png" alt="A strip showing noise resolving into a piece of clothing over the reverse diffusion chain." loading="lazy">
  <figcaption>The reverse process on clothing. The silhouette of each item commits early and the details fill in late, similar to digits.</figcaption>
</figure>

## Making it usable: DDIM and fewer steps

There is a catch I have been quiet about. Sampling ran the network a thousand times for a single image. That is fine for a blog post and ruinous once you need more than a handful of images. The first fix arrived almost immediately, in [DDIM](https://arxiv.org/abs/2010.02502) (Song, Meng, and Ermon, late 2020).

DDIM reuses the exact same trained weights, with no retraining. It reinterprets the reverse process so that the steps are no longer required to be a Markov chain, which lets you skip most of them and, in its deterministic ($\eta = 0$) setting, makes the same starting noise always map to the same image. The per-step logic is "guess the clean image, then jump partway back toward it" (schematic):

```python
x0  = predict_clean_image(img, t, eps)                   # invert the forward equation
img = sqrt(abar_prev) * x0 + sqrt(1 - abar_prev) * eps   # re-noise to an earlier step
```

The practical payoff: I can sample in 50 steps instead of 1000, a 20x speedup, with barely any loss in quality. Below, the same network sampled with 10, 50, and 1000 steps.

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/ddim_vs_ddpm_mnist.png" alt="Three grids of digits sampled with DDIM 10 steps, DDIM 50 steps, and full DDPM 1000 steps; all are legible." loading="lazy">
  <figcaption>DDIM sampling at 10, 50, and 1000 steps, all on the one trained network. The two DDIM panels share a starting seed, so notice the digit identities line up between them - that is the determinism. Ten steps is already legible; fifty is clean; the thousand-step DDPM run (a different, stochastic path) is no better to the eye.</figcaption>
</figure>

This is the part of the from-scratch model that points straight at the rest of the field. Once sampling is cheap and the objective is this stable, the obvious next questions are: can we make the samples sharper, can we steer what gets generated, and can we afford to do it at megapixel resolution.

One note before the papers, because the model I built has a gap the next section leans on. It draws *a* digit, never *the* digit you ask for, because it never saw a label. Conditioning fixes that with a small change: you hand the network a class label alongside the timestep, as one more embedding added in exactly like `t`, and train it to denoise with the label in view. Every kind of steering below is built on that one hook, so I added it to my MNIST model and trained the conditional version, which the guidance section puts to work.

## How this became DALL·E 2

Everything above is the 2020 core: DDPM plus DDIM. What turned it into the system that drew the astronaut on the horse is a short ladder of papers, each fixing one specific limitation. Here is that ladder, in order.

<figure>
  <svg viewBox="0 0 720 264" role="img" aria-labelledby="line-t line-d" style="width:100%;height:auto;max-width:720px;color:inherit" xmlns="http://www.w3.org/2000/svg">
    <title id="line-t">A timeline of diffusion model milestones from 2015 to 2022</title>
    <desc id="line-d">A horizontal time axis. Key milestones: 2015 the origin paper; 2019 score matching; 2020 DDPM makes it work and DDIM makes sampling fast; 2021 classifier guidance beats GANs, then classifier-free guidance, latent diffusion, and GLIDE arrive late in the year; April 2022 DALL-E 2. The events cluster more densely toward 2021 and 2022.</desc>
    <defs>
      <marker id="line-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0,0 L10,5 L0,10 z" fill="currentColor"></path>
      </marker>
    </defs>
    <line x1="32" y1="138" x2="694" y2="138" stroke="currentColor" stroke-width="1.6" marker-end="url(#line-ah)"></line>
    <g stroke="currentColor" stroke-width="1.3" fill="currentColor">
      <circle cx="64" cy="138" r="3.5"></circle>
      <circle cx="190" cy="138" r="3.5"></circle>
      <circle cx="300" cy="138" r="3.5"></circle>
      <circle cx="350" cy="138" r="3.5"></circle>
      <circle cx="470" cy="138" r="3.5"></circle>
      <circle cx="560" cy="138" r="3.5"></circle>
      <circle cx="650" cy="138" r="4.5" stroke-width="2"></circle>
    </g>
    <g stroke="currentColor" stroke-width="1" opacity="0.5">
      <line x1="64" y1="138" x2="64" y2="104"></line>
      <line x1="190" y1="138" x2="190" y2="172"></line>
      <line x1="300" y1="138" x2="300" y2="104"></line>
      <line x1="350" y1="138" x2="350" y2="172"></line>
      <line x1="470" y1="138" x2="470" y2="104"></line>
      <line x1="560" y1="138" x2="560" y2="172"></line>
      <line x1="650" y1="138" x2="650" y2="92"></line>
    </g>
    <g fill="currentColor" font-family="-apple-system, system-ui, sans-serif" text-anchor="middle" aria-hidden="true">
      <text x="64" y="96" font-size="11.5" font-weight="600">2015</text>
      <text x="64" y="83" font-size="10" opacity="0.75">origin</text>
      <text x="190" y="188" font-size="11.5" font-weight="600">2019</text>
      <text x="190" y="201" font-size="10" opacity="0.75">score matching</text>
      <text x="300" y="96" font-size="11.5" font-weight="600">2020 DDPM</text>
      <text x="300" y="83" font-size="10" opacity="0.75">it works</text>
      <text x="350" y="188" font-size="11.5" font-weight="600">2020 DDIM</text>
      <text x="350" y="201" font-size="10" opacity="0.75">fast sampling</text>
      <text x="470" y="96" font-size="11.5" font-weight="600">2021 guidance</text>
      <text x="470" y="83" font-size="10" opacity="0.75">beats GANs</text>
      <text x="560" y="188" font-size="11.5" font-weight="600">Dec 2021</text>
      <text x="560" y="201" font-size="10" opacity="0.75">CFG &#183; latents &#183; text</text>
      <text x="650" y="84" font-size="11.5" font-weight="700">Apr 2022</text>
      <text x="650" y="71" font-size="10" opacity="0.75">DALL&#183;E 2</text>
    </g>
    <text x="32" y="232" fill="currentColor" font-family="-apple-system, system-ui, sans-serif" font-size="11" opacity="0.7" text-anchor="start" aria-hidden="true">Two years from "it works" to text-to-image. The cluster on the right is late 2021.</text>
  </svg>
  <figcaption>The ladder, compressed. The interesting feature is the spacing: a slow start, then a dense burst through 2021 as the pieces (faster sampling, sharper samples, steering, cheaper resolution) landed in quick succession, and text-to-image fell out the far end in early 2022.</figcaption>
</figure>

**Sampling speed: DDIM (2020).** Covered above. The line to remember: a thousand steps became fifty, which is what made everything downstream practical to iterate on.

**Sharper samples and a better schedule: Improved DDPM (2021).** [Nichol and Dhariwal](https://arxiv.org/abs/2102.09672) contributed the cosine schedule I used, plus the idea of letting the network learn the reverse-step variance instead of fixing it. They reported better log-likelihood and fewer sampling steps. The schedule alone is the kind of change that costs one function and improves everything after it.

**The unification: score-based models and SDEs.** Running alongside diffusion was a second line of work, [score matching](https://arxiv.org/abs/1907.05600) (Song and Ermon, 2019), which learns the gradient of the data density and samples by following it. In late 2020, [Song and colleagues](https://arxiv.org/abs/2011.13456) showed that diffusion models and score-based models are the same object written two ways: predicting the noise is, up to a time-dependent scaling, estimating that gradient at each noise level, and both are discretizations of one continuous-time stochastic differential equation. Worth knowing, because the literature switches between "diffusion" and "score-based" language as if you should already know they are the same.

**Beating GANs: classifier guidance (May 2021).** [Dhariwal and Nichol](https://arxiv.org/abs/2105.05233) tuned the architecture and added **classifier guidance**: at sampling time, nudge each step with the gradient of a classifier trained on noised images toward the class you want. The result beat the best GANs on ImageNet, which is the moment the field's default flipped from GANs to diffusion. The catch in the title is real but conditional: this was one benchmark, ImageNet, with one carefully tuned setup. What guidance actually buys is a knob that trades sample diversity for fidelity, by sharpening the conditional distribution.

**Dropping the classifier: classifier-free guidance (Dec 2021).** Training a separate classifier on noisy images is awkward, and it pins you to a fixed set of labels, which is useless for free-form text. [Ho and Salimans](https://openreview.net/forum?id=qw8AKxfYbI) (NeurIPS 2021 workshop) removed it. Train one network that sometimes sees the condition and sometimes sees a blank, then at sampling time extrapolate between the two predictions:

$$\tilde{\epsilon}(x_t, c) = (1 + w)\, \epsilon_\theta(x_t, c) - w\, \epsilon_\theta(x_t, \varnothing)$$

Here $c$ is the condition (a class, a caption), $\varnothing$ is the blank, and $w$ turns the steering up. Most implementations write the same thing with a guidance scale $s = 1 + w$, where $w = 0$ (so $s = 1$) means no steering, so if you compare two codebases and the numbers look off by one, that is why. Classifier-free guidance is the workhorse of every strong text-to-image model that followed.

This is the one rung I can run myself. Classifier-free guidance needs a model that can sample both with and without the label, so I trained exactly that on MNIST: a conditional U-Net with the label dropped 15% of the time. First the basic question, does the label even steer it? Same ten starting seeds, run once ignoring the label and once told which digit to draw:

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/conditioning_mnist.png" alt="Two rows of generated digits from the same noise seeds; the top row with the label off is ten unrelated digits, the bottom row with the label on is exactly 0 through 9 in order." loading="lazy">
  <figcaption>The same noise, steered by a class label. Both rows start from the identical ten seeds. With the label off (the unconditional model) each seed draws whatever it wants; with the label on, those same seeds resolve into exactly the digits 0 through 9.</figcaption>
</figure>

Now turn the guidance dial up. More guidance trades diversity for fidelity, sharpening each sample toward its class. On a model this small, the usable range is narrow:

<figure>
  <img src="/images/blog/diffusion-models-from-scratch/guidance_mnist.png" alt="Four rows of generated digits zero through nine at increasing guidance scale; the top row is clean, and lower rows thicken, distort, and oversaturate." loading="lazy">
  <figcaption>The same conditional model swept across guidance scales, s = 1 (the plain conditional model) at the top. A little guidance bolds the strokes, but past s of about 1.5 the samples oversaturate, thicken, and slide off the manifold of real digits, a few even flipping which digit they are. This is the over-saturation the big text-to-image models spend real effort taming; here it shows up early because the model is tiny. I don't have a clean number for where it tips over - on MNIST it was somewhere under 1.5 and I was mostly eyeballing the grid.</figcaption>
</figure>

**Affording resolution: latent diffusion (Dec 2021).** Diffusing directly on megapixel pixels is brutally expensive, because the U-Net runs at full resolution a thousand times. [Rombach and colleagues](https://arxiv.org/abs/2112.10752) moved the diffusion into the compact latent space of a pretrained autoencoder, cutting the cost by roughly an order of magnitude while keeping quality, and wired in cross-attention so you can condition on text or layout. This is the architecture that, later in 2022, would become Stable Diffusion, though as I write this that release does not exist yet.

**Text-to-image: GLIDE and DALL·E 2 (Dec 2021 to Apr 2022).** [GLIDE](https://arxiv.org/abs/2112.10741) was the first strong text-to-image diffusion model and showed classifier-free guidance beats CLIP-based guidance for following captions. Then [DALL·E 2](https://arxiv.org/abs/2204.06125), three weeks ago, restructured the problem around CLIP, a model trained to match images with their captions: a **prior** turns the caption into a CLIP image embedding, then a diffusion **decoder** turns that embedding into a picture, with classifier-free guidance doing the steering. The authors call it unCLIP, and they find a diffusion prior works better than an autoregressive one, so there is diffusion on both ends. Note that this is a different lineage from latent diffusion; DALL·E 2 is not a latent-diffusion model, even though both are text-to-image systems and it is easy to blur them together.

## What is still hard

A few things that were still genuinely unsolved, from where I sat in May 2022.

Sampling is still slow next to a GAN, which generates in a single forward pass. DDIM helped a lot, fifty steps instead of a thousand, but that is still fifty forward passes to the GAN's one, and the race to cut that further has already started: [progressive distillation](https://arxiv.org/abs/2202.00512), out this February, halves the step count and then halves it again. Evaluation is shaky too. FID (Fréchet Inception Distance) is the standard number and it is a blunt instrument, and "does this image match the caption" has no clean metric at all.

The compute needed is a core gap - impressive results come from models far larger than anything that can be trained locally, and what separates "can build this on MNIST" from "can build DALL·E 2" is mostly scale & data available.

Guidance carries the cost the figure above showed in miniature: the lower rows thicken, oversaturate, and slide off the manifold of real images.

## Build it yourself

The model in this post is small enough to read in one sitting and train on a laptop, which was the whole point. If you want the denoising idea to stop being abstract, clone the repo, run `train.py`, and watch the sample grid fill in epoch by epoch. The forward process, the loss, and both samplers are each a handful of lines, exactly as shown above.

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

[Browse the code](https://github.com/nilesh-patil/ddpm-from-scratch){: .btn-soft .btn-soft--primary} [Read the DALL·E 2 paper](https://arxiv.org/abs/2204.06125){: .btn-soft} [More deep learning from scratch]({{ site.baseurl }}/posts/galactic-morphology-using-deep-learning/){: .btn-soft}
