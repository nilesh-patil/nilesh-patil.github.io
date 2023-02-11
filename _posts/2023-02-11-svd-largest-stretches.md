---
layout: single
title: "Singular value decomposition - rebuilding an image from its largest stretches"
date: 2023-02-11T17:05:00+05:30
categories: [blog]
tags: [linear-algebra, svd, low-rank, image-compression, numpy, astronomy, visualization]
excerpt: "Every matrix moves space in three steps: rotate, stretch along perpendicular axes, rotate again. One small matrix and one SDSS image of the Whirlpool galaxy carry the whole idea, from what a singular value measures to why fifteen of them rebuild a recognizable galaxy from six percent of the numbers, with the error known before the first pixel is drawn."
math: true
header:
  overlay_image: /images/blog/headers/svd-largest-stretches.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/svd-largest-stretches.jpg
---

During singular value decomposition, every matrix, whatever is in its entries, moves space in three steps : a rotation, a stretch along perpendicular axes, and a second rotation. Any real matrix factors as :

$$
A = U \Sigma V^{\mathsf{T}},
$$

where $U$ and $V$ are orthogonal (their columns are perpendicular unit vectors, so they rotate without distorting anything) and $\Sigma$ is diagonal and nonnegative. The diagonal entries of $\Sigma$, sorted so that $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$, are the singular values : the stretch factors, ranked by how much they matter.

This single, short process is enough to rebuild a galaxy from a sliver of its raw data. A `512x512` photograph of the Whirlpool galaxy, to numpy, is a matrix of 262,144 numbers. Keep its fifteen largest "stretches" and we are storing 15,375 numbers, under `6%` of the original, and the spiral arms are already unmistakably visible. A theorem from 1936 guarantees that nothing else of that size does better, and tells us the error before materializing a single pixel.

In this post, we work with two objects - a `2x2` matrix that we can watch move the plane, and one photograph of the Whirlpool galaxy. We explain two concepts, PCA and least squares, each with a short section of its own. The one thing I deliberately skip is how a cpu actually finds singular values.

## One matrix, one circle

A `2x2` matrix :

$$
A = \begin{bmatrix} 2.1907 & 1.4836 \\ 0.4483 & 1.6731 \end{bmatrix}
$$

Columns tell you where the two basis vectors land, but not what $A$ does to space as a whole, so the better probe is to feed it every direction at once : all the unit vectors, the whole unit circle, watched together as they pass through.

<figure>
  <svg viewBox="0 0 660 250" role="img" aria-labelledby="fig-ce-t fig-ce-d"
       style="width:100%; max-width:640px; height:auto; color:inherit"
       xmlns="http://www.w3.org/2000/svg">
    <title id="fig-ce-t">The unit circle maps to an ellipse</title>
    <desc id="fig-ce-d">A unit circle on the left passes through the matrix A and comes out as a tilted ellipse three times longer in one direction than the other.</desc>
    <defs>
      <marker id="fig-ce-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <line x1="150" y1="55" x2="150" y2="195" stroke="currentColor" stroke-width="1" opacity="0.18"/>
    <line x1="80" y1="125" x2="220" y2="125" stroke="currentColor" stroke-width="1" opacity="0.18"/>
    <circle cx="150" cy="125" r="40" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.8"/>
    <text x="150" y="215" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7">every unit vector</text>
    <line x1="235" y1="125" x2="330" y2="125" stroke="currentColor" stroke-width="1.6" marker-end="url(#fig-ce-ah)"/>
    <text x="282" y="112" text-anchor="middle" font-size="15" font-style="italic" fill="currentColor">A</text>
    <line x1="470" y1="55" x2="470" y2="195" stroke="currentColor" stroke-width="1" opacity="0.18"/>
    <line x1="340" y1="125" x2="600" y2="125" stroke="currentColor" stroke-width="1" opacity="0.18"/>
    <circle cx="470" cy="125" r="40" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.3"/>
    <ellipse cx="470" cy="125" rx="120" ry="40" transform="rotate(-30 470 125)"
             fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.8"/>
    <text x="470" y="215" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7">its image under A</text>
  </svg>
  <figcaption>The unit circle goes in, an ellipse comes out. The dashed circle marks where the input was.</figcaption>
</figure>

An ellipse, tilted and three times longer in one direction than the other. The only freedoms it has are which perpendicular axes to use, how hard to stretch along each, and where to point the result.

## The ellipse axes have names

Label the long semi-axis of the ellipse $\sigma_1 u_1$ : its length is $\sigma_1$, the largest stretch, and the unit vector $u_1$ is the direction it points. The short semi-axis is $\sigma_2 u_2$, perpendicular to the first. Then walk backwards : somewhere on the input circle sits the unit vector that $A$ sent to the tip of the long axis; call it $v_1$, and call the preimage of the short axis tip $v_2$.

<figure>
  <svg viewBox="0 0 660 250" role="img" aria-labelledby="fig-ax-t fig-ax-d"
       style="width:100%; max-width:640px; height:auto; color:inherit"
       xmlns="http://www.w3.org/2000/svg">
    <title id="fig-ax-t">Singular vectors on the circle, singular values on the ellipse</title>
    <desc id="fig-ax-d">The same circle and ellipse, now labeled : input directions v1 and v2 on the circle map to the ellipse semi-axes with lengths sigma 1 and sigma 2.</desc>
    <defs>
      <marker id="fig-ax-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <circle cx="150" cy="125" r="40" fill="none" stroke="currentColor" stroke-width="1.8"/>
    <line x1="150" y1="125" x2="178.3" y2="96.7" stroke="currentColor" stroke-width="1.6" stroke-dasharray="5 3" marker-end="url(#fig-ax-ah)"/>
    <line x1="150" y1="125" x2="121.7" y2="96.7" stroke="currentColor" stroke-width="1.6" stroke-dasharray="5 3" opacity="0.6" marker-end="url(#fig-ax-ah)"/>
    <text x="190" y="92" font-size="13" font-style="italic" fill="currentColor">v&#8321;</text>
    <text x="98" y="92" font-size="13" font-style="italic" fill="currentColor" opacity="0.75">v&#8322;</text>
    <text x="150" y="215" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7">the two special inputs</text>
    <line x1="235" y1="125" x2="330" y2="125" stroke="currentColor" stroke-width="1.6" marker-end="url(#fig-ax-ah)"/>
    <text x="282" y="112" text-anchor="middle" font-size="15" font-style="italic" fill="currentColor">A</text>
    <ellipse cx="470" cy="125" rx="120" ry="40" transform="rotate(-30 470 125)"
             fill="none" stroke="currentColor" stroke-width="1.8"/>
    <line x1="366.1" y1="185" x2="573.9" y2="65" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4" opacity="0.35"/>
    <line x1="450" y1="90.4" x2="490" y2="159.6" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4" opacity="0.35"/>
    <line x1="470" y1="125" x2="573.9" y2="65" stroke="currentColor" stroke-width="2" marker-end="url(#fig-ax-ah)"/>
    <line x1="470" y1="125" x2="450" y2="90.4" stroke="currentColor" stroke-width="2" opacity="0.7" marker-end="url(#fig-ax-ah)"/>
    <text x="560" y="52" font-size="13" font-style="italic" fill="currentColor">&#963;&#8321;u&#8321;</text>
    <text x="405" y="82" font-size="13" font-style="italic" fill="currentColor" opacity="0.8">&#963;&#8322;u&#8322;</text>
    <text x="470" y="215" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7">the two semi-axes they become</text>
  </svg>
  <figcaption>The inputs $v_1, v_2$ that land on the ellipse axes. $A v_1 = \sigma_1 u_1$ and $A v_2 = \sigma_2 u_2$</figcaption>
</figure>

There is a second way to arrive at $v_1$ : over all unit vectors $v$, which one does $A$ stretch the most? That is $v_1$, the achieved length $\lVert A v_1 \rVert$ is $\sigma_1$, and restricting to directions perpendicular to $v_1$ and asking again gives $v_2$. The line through $v_1$ is the best-fit line through a cloud of points in n-dimensions (the rows of $A$), the plane through $v_1, v_2$ is the best-fit plane.

## Rotate, stretch, rotate

$V$ contains $v_1, v_2$ in its columns, $U$ contains $u_1, u_2$, and $\Sigma$ contains the two stretch factors. The equation $A = U \Sigma V^{\mathsf{T}}$ then reads, right to left, as a recipe applied to any input :

<figure>
  <svg viewBox="0 0 760 220" role="img" aria-labelledby="fig-3m-t fig-3m-d"
       style="width:100%; max-width:760px; height:auto; color:inherit"
       xmlns="http://www.w3.org/2000/svg">
    <title id="fig-3m-t">The three moves of the SVD</title>
    <desc id="fig-3m-d">Four panels : the unit circle, the circle rotated so v1 lies on the x axis, the axis-aligned stretch into an ellipse, and the final rotation of the ellipse into place.</desc>
    <defs>
      <marker id="fig-3m-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <circle cx="95" cy="120" r="25" fill="none" stroke="currentColor" stroke-width="1.8"/>
    <line x1="95" y1="120" x2="112.7" y2="102.3" stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#fig-3m-ah)"/>
    <line x1="95" y1="120" x2="77.3" y2="102.3" stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.6" marker-end="url(#fig-3m-ah)"/>
    <text x="120" y="98" font-size="11" font-style="italic" fill="currentColor">v&#8321;</text>
    <text x="60" y="98" font-size="11" font-style="italic" fill="currentColor" opacity="0.75">v&#8322;</text>
    <text x="95" y="180" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.7">start</text>
    <line x1="135" y1="120" x2="238" y2="120" stroke="currentColor" stroke-width="1.5" marker-end="url(#fig-3m-ah)"/>
    <text x="186" y="106" text-anchor="middle" font-size="13" font-style="italic" fill="currentColor">V&#7488;</text>
    <circle cx="277" cy="120" r="25" fill="none" stroke="currentColor" stroke-width="1.8"/>
    <line x1="277" y1="120" x2="302" y2="120" stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#fig-3m-ah)"/>
    <line x1="277" y1="120" x2="277" y2="95" stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.6" marker-end="url(#fig-3m-ah)"/>
    <text x="305" y="112" font-size="11" font-style="italic" fill="currentColor">v&#8321;</text>
    <text x="284" y="94" font-size="11" font-style="italic" fill="currentColor" opacity="0.75">v&#8322;</text>
    <text x="277" y="180" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.7">aligned with the axes</text>
    <line x1="315" y1="120" x2="370" y2="120" stroke="currentColor" stroke-width="1.5" marker-end="url(#fig-3m-ah)"/>
    <text x="342" y="106" text-anchor="middle" font-size="14" fill="currentColor">&#931;</text>
    <ellipse cx="459" cy="120" rx="75" ry="25" fill="none" stroke="currentColor" stroke-width="1.8"/>
    <line x1="459" y1="120" x2="534" y2="120" stroke="currentColor" stroke-width="1.6" marker-end="url(#fig-3m-ah)"/>
    <line x1="459" y1="120" x2="459" y2="95" stroke="currentColor" stroke-width="1.6" opacity="0.7" marker-end="url(#fig-3m-ah)"/>
    <text x="512" y="108" font-size="11" font-style="italic" fill="currentColor">&#963;&#8321;</text>
    <text x="466" y="94" font-size="11" font-style="italic" fill="currentColor" opacity="0.75">&#963;&#8322;</text>
    <text x="459" y="180" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.7">stretched 3x and 1x</text>
    <line x1="543" y1="120" x2="583" y2="120" stroke="currentColor" stroke-width="1.5" marker-end="url(#fig-3m-ah)"/>
    <text x="563" y="106" text-anchor="middle" font-size="13" font-style="italic" fill="currentColor">U</text>
    <ellipse cx="665" cy="120" rx="75" ry="25" transform="rotate(-30 665 120)" fill="none" stroke="currentColor" stroke-width="1.8"/>
    <line x1="665" y1="120" x2="729.9" y2="82.5" stroke="currentColor" stroke-width="1.6" marker-end="url(#fig-3m-ah)"/>
    <line x1="665" y1="120" x2="652.5" y2="98.3" stroke="currentColor" stroke-width="1.6" opacity="0.7" marker-end="url(#fig-3m-ah)"/>
    <text x="700" y="70" font-size="11" font-style="italic" fill="currentColor">&#963;&#8321;u&#8321;</text>
    <text x="612" y="92" font-size="11" font-style="italic" fill="currentColor" opacity="0.8">&#963;&#8322;u&#8322;</text>
    <text x="665" y="180" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.7">rotated into place</text>
  </svg>
  <figcaption>Reading $A = U \Sigma V^{\mathsf{T}}$ right to left : $V^{\mathsf{T}}$ rotates the special directions onto the coordinate axes, $\Sigma$ stretches along them, $U$ rotates the result where it belongs.</figcaption>
</figure>

First $V^{\mathsf{T}}$ rotates the plane so that $v_1$ and $v_2$ lie along the coordinate axes, and $\Sigma$ then stretches by 3x and 1x, independently. Then $U$ rotates the stretched result into its final orientation. All the shearing, tilting messiness that $A$ seemed to have was an artifact of looking at it in the wrong coordinates; in the right input frame and the right output frame, every matrix is diagonal.

using numpy :

```python
import numpy as np

A = np.array([[2.1907, 1.4836],
              [0.4483, 1.6731]])

U, S, Vt = np.linalg.svd(A)
print(S)
```

```
[3.0000682  1.00003136]
```

Not exactly 3 and 1, because we rounded the entries of $A$ to four decimals when writing them down, and the singular values report the damage in the fifth digit. That is because this matrix was synthetic : I built it as a 45 degree rotation, then a stretch by $\mathrm{diag}(3, 1)$, then a 30 degree rotation, and the printed entries are that final output. The decomposition recovered the recipe from the entries alone.

Two conventions hide in that call : 
- numpy returns back `Vt`, the transpose of $V$, not $V$ itself
- with `full_matrices=False` (used below) it returns the thin form that skips the parts multiplied by nothing ( [numpy docs](https://numpy.org/doc/1.24/reference/generated/numpy.linalg.svd.html) )

## Dropping the smaller stretch

The matrix-of-rotations picture has an equivalent that turns out to be the useful one for data. Multiply the factorization - cell by cell and it becomes a sum:

$$
A = \sigma_1 u_1 v_1^{\mathsf{T}} + \sigma_2 u_2 v_2^{\mathsf{T}}.
$$

Each term is an outer product, a column pattern $u_i$ x a row recipe $v_i^{\mathsf{T}}$ weighted by $\sigma_i$, and I think of each one as a block. A block is the simplest matrix there is, rank one, every row a multiple of every other, and it costs only $m + n + 1$ numbers to describe no matter how large the matrix is. SVD rewrites your matrix as a sum of blocks, sorted by weight.

Sorted by weight implies a natural potential cut : keep the heavy blocks and drop the light ones.

<figure>
  <svg viewBox="0 0 660 250" role="img" aria-labelledby="fig-r1-t fig-r1-d"
       style="width:100%; max-width:640px; height:auto; color:inherit"
       xmlns="http://www.w3.org/2000/svg">
    <title id="fig-r1-t">Setting sigma 2 to zero flattens the ellipse</title>
    <desc id="fig-r1-d">The full ellipse on the left collapses onto its long axis when the smaller stretch is set to zero, leaving a line segment.</desc>
    <defs>
      <marker id="fig-r1-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <ellipse cx="165" cy="125" rx="120" ry="40" transform="rotate(-30 165 125)"
             fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.8"/>
    <line x1="165" y1="125" x2="268.9" y2="65" stroke="currentColor" stroke-width="1.6" marker-end="url(#fig-r1-ah)"/>
    <line x1="165" y1="125" x2="145" y2="90.4" stroke="currentColor" stroke-width="1.6" opacity="0.7" marker-end="url(#fig-r1-ah)"/>
    <text x="255" y="52" font-size="12" font-style="italic" fill="currentColor">&#963;&#8321;u&#8321;</text>
    <text x="100" y="82" font-size="12" font-style="italic" fill="currentColor" opacity="0.8">&#963;&#8322;u&#8322;</text>
    <text x="165" y="218" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7">both blocks</text>
    <line x1="305" y1="125" x2="385" y2="125" stroke="currentColor" stroke-width="1.5" marker-end="url(#fig-r1-ah)"/>
    <text x="345" y="110" text-anchor="middle" font-size="12" fill="currentColor">&#963;&#8322; &#8594; 0</text>
    <ellipse cx="495" cy="125" rx="120" ry="40" transform="rotate(-30 495 125)"
             fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.3"/>
    <line x1="391.1" y1="185" x2="598.9" y2="65" stroke="currentColor" stroke-width="2.6"/>
    <text x="495" y="218" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7">the heavy block alone</text>
  </svg>
  <figcaption>Rank one is a flat world : with $\sigma_2 = 0$ every input lands on the line through $u_1$.</figcaption>
</figure>

Numerically the heavy block of our matrix is :

$$
\sigma_1 u_1 v_1^{\mathsf{T}} = \begin{bmatrix} 1.8371 & 1.8372 \\ 1.0607 & 1.0607 \end{bmatrix},
$$

and comparing it against $A$ entry by entry, the worst miss is 0.61, which is exactly the largest entry of the discarded $\sigma_2 u_2 v_2^{\mathsf{T}}$. For a 2 by 2 matrix this trade is silly : the block costs five numbers to store where the matrix cost four. This turns more interesting when the matrix is large and its stretches are unequal, as is the case in most real-world datasets.

## The matrix is a galaxy

I pulled a `512x512` cutout of `M51`, the Whirlpool galaxy, from the [SDSS DR17 SkyServer](https://skyserver.sdss.org/dr17) cutout service, centered on `RA 202.4842`, Dec 47.2306 at 1.4 arcseconds per pixel, wide enough to catch both the spiral and its companion `NGC 5195`. The exact request URL sits at the end of the post, so every number below can be re-run against the same frame. Collapsed to grayscale luminance it is one matrix, 512 rows by 512 columns, 262,144 numbers between 0 and 1.

<figure>
  <img src="{{ site.baseurl }}/images/blog/svd-largest-stretches/figure-1-galaxy-as-matrix.svg" alt="The M51 grayscale cutout beside a zoomed 7 by 7 patch of its raw pixel values." loading="lazy">
  <figcaption>The working matrix. Left : the full cutout, with the row and column indices any entry can be addressed by. Right : the boxed patch at the nucleus, printed as numbers.</figcaption>
</figure>

```python
from skimage import io, color

A = color.rgb2gray(io.imread("m51_sdss_512.jpg"))   # 512 x 512, values in [0, 1]
U, s, Vt = np.linalg.svd(A, full_matrices=False)     # s has 512 entries
```

Our initial `2x2` matrix had two singular values, this one has 512 :

<figure>
  <img src="{{ site.baseurl }}/images/blog/svd-largest-stretches/figure-2-spectrum.svg" alt="Log-scale plot of the 512 singular values falling from 122 to below one ten-thousandth." loading="lazy">
  <figcaption>The spectrum spans six orders of magnitude and never quite reaches zero; sensor noise and JPEG artifacts keep the tail alive.</figcaption>
</figure>

The top value, $\sigma_1 = 122.4$, towers over $\sigma_2 = 21.8$ by a factor of 5.6, and the tail decays over six orders of magnitude. The physical reading : about half the frame is black given a small margin, and most of what is not black is smooth, the broad glow of two galaxy disks. Smooth large-scale structure is exactly what a few blocks reproduce well; the pinpoint stars scattered across the frame are another matter though.

Square the singular values and you get the "energy" most textbooks use : $\sigma_1$ alone carries `94.6%` of this image's "energy", and the first seven values carry `99%`. By that measure - seven blocks should nearly suffice, and yet at $k = 7$ the frame is still smudges. Squaring inflates the lead of the front-runner, so energy fractions crowd toward `100%` long before pictures look like anything meaningful. The yardstick that tracks what the eye sees is the relative error $\lVert A - A_k \rVert_F / \lVert A \rVert_F$, which at $k = 7$ still stands at `9.7%`.

## block by block

The rank-$k$ reconstruction keeps the $k$ heaviest blocks :

```python
def rebuild(k):
    return (U[:, :k] * s[:k]) @ Vt[:k, :]
```

<figure>
  <img src="{{ site.baseurl }}/images/blog/svd-largest-stretches/figure-3-rank-strip.svg" alt="Six panels showing the galaxy rebuilt from 1, 5, 15, 50, and 150 blocks beside the original." loading="lazy">
  <figcaption>Rank-k reconstructions vs the original : The percentage under each panel counts the stored numbers, $k(m+n+1)$ of them, against the full 262,144.</figcaption>
</figure>

Each block is a single column pattern times a single row recipe, so $k = 1$ can only paint axis-aligned bands. First panel is a bright horizontal belt across the rows where the galaxies live, crossed by a vertical one over the columns that hold them. By $k = 5$ the two disks exist as blobs. At $k = 15$, costing `5.9%` of the numbers, the spiral arms are unmistakable, though every miss still smears into horizontal and vertical streaks, the outer-product basis showing through. At $k = 50$ the streaks are gone, and from $k = 150$ eyes cannot tell the reconstruction from the original at this size, with `1.6%` relative error.

Compare the $k = 15$ panel against the original and one class of objects is conspicuously late : the foreground stars. A star is two or three bright pixels, which contributes almost nothing to the Frobenius energy the approximation is optimizing, so the top blocks spend their budget on the broad disks and leave the stars for the deep tail. The glow is cheap; the stars are expensive. If your science lives in point sources, truncated SVD is a lossy filter aimed directly at the thing you care about, which is worth knowing before you use it to denoise anything.

One caveat before anyone tries this on a photo library : this is not how images are compressed in practice. JPEG runs a fixed cosine transform on local `8x8` blocks; it has no 512-number global patterns to fit and nothing to solve, and at these quality levels it is far cheaper. The rank-$k$ ledger here, `0.4%` at $k = 1$ to `58.7%` at $k = 150$, is the pedagogical price sheet for a *global*, *image-specific* basis.

## As good as rank k gets

The guarantee is the Eckart-Young theorem, published in 1936 and extended by Mirsky in 1960. Among all matrices of rank at most $k$, with no restriction to SVD-shaped candidates, the truncated sum $A_k = \sum_{i \le k} \sigma_i u_i v_i^{\mathsf{T}}$ is the closest to $A$, in the Frobenius norm, in the spectral norm, and, after Mirsky, in every norm that is indifferent to rotations.

$$
\lVert A - A_k \rVert_F = \sqrt{\textstyle\sum_{i > k} \sigma_i^2},
\qquad
\lVert A - A_k \rVert_2 = \sigma_{k+1}.
$$

The discarded weights are the error, so the spectrum in the second figure was already a complete error catalog for every possible truncation, available before reconstructing anything. It can be checked in two lines:

```python
k = 15
measured  = np.linalg.norm(A - rebuild(k)) / np.linalg.norm(A)
predicted = np.sqrt(np.sum(s[k:] ** 2)) / np.linalg.norm(A)
```

<figure>
  <img src="{{ site.baseurl }}/images/blog/svd-largest-stretches/figure-4-error-curve.svg" alt="Log-scale error curve predicted by the tail formula with fourteen measured reconstruction errors sitting exactly on it." loading="lazy">
  <figcaption>Fourteen reconstructions measured - each landing on the curve the singular values predicted</figcaption>
</figure>

## The same picture in PCA

PCA ~ mean centered + truncated SVD output

Take $n$ data points in $d$ dimensions, subtract the column means, and call the centered matrix $X$. Principal component analysis asks for the directions along which the cloud varies most, subject to being mutually perpendicular, which is verbatim the greedy question from earlier asked of $X$. The answers are its right singular vectors : the first principal direction is $v_1$, the second is $v_2$, and the variance captured along $v_i$ is $\frac{\sigma_i^2}{(n - 1)}$.

<figure>
  <svg viewBox="0 0 640 260" role="img" aria-labelledby="fig-pca-t fig-pca-d"
       style="width:100%; max-width:620px; height:auto; color:inherit"
       xmlns="http://www.w3.org/2000/svg">
    <title id="fig-pca-t">A centered point cloud and its principal directions</title>
    <desc id="fig-pca-d">Twenty-six scattered points forming an elongated cloud, with a long dashed line along the direction of most variance and a short one perpendicular to it.</desc>
    <line x1="182.8" y1="210.1" x2="428" y2="74.9" stroke="currentColor" stroke-width="1.4" stroke-dasharray="6 4" opacity="0.55"/>
    <line x1="282.2" y1="100.5" x2="328.6" y2="184.5" stroke="currentColor" stroke-width="1.4" stroke-dasharray="6 4" opacity="0.35"/>
    <circle cx="340.3" cy="165.1" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="337.8" cy="129.8" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="308" cy="137.8" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="279" cy="151.6" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="311.9" cy="163.2" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="279.2" cy="162.5" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="330.6" cy="142.1" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="386.9" cy="106.4" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="289.3" cy="128" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="298.6" cy="157.4" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="342.3" cy="117.7" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="328.9" cy="108.4" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="329.5" cy="135.4" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="279" cy="155.8" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="317.8" cy="129.2" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="350.8" cy="111" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="269.3" cy="182.1" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="298.8" cy="140.8" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="223.4" cy="160.5" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="274.4" cy="185.1" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="230.1" cy="165.9" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="308.5" cy="134.4" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="268.1" cy="171.9" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="316.1" cy="95" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="320.9" cy="115.3" r="3.5" fill="currentColor" opacity="0.55"/>
    <circle cx="321.2" cy="151.6" r="3.5" fill="currentColor" opacity="0.55"/>
    <text x="432" y="66" font-size="12" font-style="italic" fill="currentColor">v&#8321;</text>
    <text x="334" y="196" font-size="12" font-style="italic" fill="currentColor" opacity="0.75">v&#8322;</text>
  </svg>
  <figcaption>PCA of a toy cloud of 26 points : the right singular vectors of the centered data are the principal directions, here splitting the variance 86 to 14</figcaption>
</figure>

The Gram matrix makes the link mechanical : $X^{\mathsf{T}} X = V \Sigma^2 V^{\mathsf{T}}$, so the eigenvectors of $X^{\mathsf{T}} X$ are the right singular vectors, its eigenvalues are the squared singular values, and dividing by $(n-1)$ turns it into the covariance matrix carrying exactly the variances above. This is also why we usually compute PCA through an SVD of $X$ rather than by forming $X^{\mathsf{T}} X$ : squaring the matrix squares its condition number, and the small components drown first. "keeping the components before the elbow" from an analysis of principal components is the same as keeping fifteen blocks of a galaxy here.

## Running the film backwards

What do you do with $Ax = b$ when it has no solution, the everyday situation of more equations than unknowns? Strang's answer in [18.06](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/) is to project $b$ onto the column space of $A$ and solve for the projection, which is what least squares is.

The SVD version is the three-moves picture run in reverse : $A$ rotated by $V^{\mathsf{T}}$, stretched by $\Sigma$, and rotated by $U$, so undo each move last-first, rotating back by $U^{\mathsf{T}}$, dividing by every stretch you can divide by, rotating back by $V$. The only move that can refuse to reverse is the stretch, where a $\sigma_i = 0$ flattened a whole direction to nothing, and there division is impossible; the convention is to leave those directions at zero rather than invent information the matrix destroyed. The result is the pseudoinverse

$$
A^{+} = V \Sigma^{+} U^{\mathsf{T}},
$$

with $\Sigma^{+}$ holding $\frac{1}{\sigma_i}$ for each nonzero stretch and 0 elsewhere. The vector $x = A^{+} b$ is the least-squares solution, and when even that is not unique, it is the shortest one. This is what `np.linalg.lstsq` and `np.linalg.pinv` compute underneath, and it is why they keep working on the degenerate systems that make the textbook normal equations fall over.

In the meantime, the [cutout service](https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?ra=202.4842&dec=47.2306&scale=1.4&width=512&height=512) takes any coordinates you like : pick your own galaxy, pull its 512 stretches, and count how few of them matter.

## Sources

- Gregory Gundersen, [Singular Value Decomposition as Simply as Possible](https://gregorygundersen.com/blog/2018/12/10/svd/), 2018. The geometric build-up, with the barn-shear intuition.
- John Hopcroft and Ravindran Kannan, [Computer Science Theory for the Information Age, chapter 4](https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf). The greedy best-fit-subspace development and both optimality proofs.
- Gilbert Strang, [MIT 18.06, Least Squares unit](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/), OCW 2011. Projection, normal equations, pseudoinverse.
- G. W. Stewart, [On the Early History of the Singular Value Decomposition](https://doi.org/10.1137/1035134), SIAM Review, 1993.
- Carl Eckart and Gale Young, The Approximation of One Matrix by Another of Lower Rank, Psychometrika 1(3), 1936.
- Lloyd Trefethen and David Bau, Numerical Linear Algebra, SIAM 1997, Lecture 4. The hyperellipse picture.
