---
layout: single
title: "NumPy vectorization - replacing Python loops in scientific computing"
date: 2017-03-04T05:10:55-01:00
last_modified_at: 2022-07-18T15:09:55-04:00
categories: [blog]
tags: [numpy, python, performance, vectorization, scientific-computing]
excerpt: "Six NumPy operations timed against the Python loop I would have written for each, with speedups from 3.2x to 382x. What sets the spread, and where the array call barely wins."
math: true
redirect_from:
 - /blog/working-with-numpy/
header:
 overlay_image: /images/blog/headers/working-with-numpy.jpg
 overlay_filter: 0.4
 teaser: /images/blog/headers/working-with-numpy.jpg
---

*I wrote this in 2017 and rewrote it in 2022 after re-running every benchmark. The prose, the numbers and the code below, including the `default_rng` seeding, are from 2022 pass*

I summed a million floats in a Python loop and it took 9.8 ms. The same task as `a.sum()` took 0.16 ms and matched to every digit I checked. That is 60x for deleting three lines, which made me suspect my timing code before digging into what `numpy` was really doing under the hood.

To dig deeper, I wrote the same pair for five more operations : six loops I would plausibly have typed, six one-line array calls, keeping identical data on both sides. The measured speedups came back at 3.2x, 9x, 28x, 35x, 60x and 382x!

<div class="notice--info" markdown="1">
**Six small experiments :** Each operation at $n = 10^6$, a hand-written Python loop over lists against the one-line whole-array call, timed in one script run on one machine.

| operation | loop | array | speedup |
| :--- | ---: | ---: | ---: |
| `a @ b` | 27.5 ms | 0.072 ms | 382x |
| `a.sum()` | 9.8 ms | 0.16 ms | 60x |
| `z-score` by column | 169 ms | 4.8 ms | 35x |
| `a * b + c` | | | 28x |
| `np.cumsum(a)` | 30 ms | 3.4 ms | 9x |
| `a[a > 0.0]` | 16 ms | 5.0 ms | 3.2x |
{: .table}

</div>

## Timing process

All comparisons are part of `bench.py` python script in the companion repo at the end. It seeds `np.random.default_rng(20170204)` and re-seeds before each operation, so the loop and the array call see the same numbers. Timing is a small autotimer rather than `timeit`: it doubles the batch size until a batch clears `0.05s` on `time.perf_counter`, warms both versions once, then keeps the best of 7 iterations. Top numbers are at `n = 1,000,000`; the size sweep runs `a * b + c` alone, from n of 10 up to four million.

The timed loop iterates a plain Python list, `a.tolist()`, not the array. So every pair is a list loop against one call on the array. Iterating a Numpy array in Python is a third thing, slower again, because every element comes back wrapped in a fresh scalar object.

## The price of one add

The sum loop does one add per element and there's nothing obviously wasteful in there, so the 60x has to come from somewhere other than the arithmetic. At 9.8 ms over a million elements it spends about 9.8 ns per element, and a float add is well under a nanosecond. The interpreter fetches a pointer, checks the object's type, unboxes it, sends `+` through dynamic dispatch, boxes the result into a fresh heap object, and steps the iterator. A Python list is a block of pointers to floats scattered across the heap, so each fetch risks a cache miss.

A Numpy array is one contiguous typed buffer :

```python
import numpy as np

rng = np.random.default_rng(20170204)
a = rng.standard_normal(1000000)
a_list = a.tolist()

total = 0.0
for x in a_list:      # 9.8 ms
    total += x

total = a.sum()       # 0.16 ms
```

<figure>
 <svg viewBox="0 0 600 300" role="img" aria-labelledby="lva-t lva-d" style="width:100%;height:auto;max-width:600px;color:inherit" xmlns="http://www.w3.org/2000/svg">
   <title id="lva-t">A list of boxed floats against one contiguous buffer</title>
   <desc id="lva-d">Eight list slots pointing to scattered float objects above one contiguous eight-cell buffer.</desc>
   <defs>
     <marker id="lva-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
       <path d="M0,0 L10,5 L0,10 z" fill="currentColor"></path>
     </marker>
   </defs>
   <g aria-hidden="true" font-family="-apple-system, system-ui, sans-serif">
     <text x="20" y="30" font-size="17" font-weight="600" fill="currentColor">Python list</text>
     <text x="20" y="50" font-size="11.5" fill="currentColor" opacity="0.7">boxed floats,</text>
     <text x="20" y="66" font-size="11.5" fill="currentColor" opacity="0.7">scattered on the heap</text>
     <g fill="currentColor" opacity="0.55">
       <circle cx="192" cy="60" r="6"></circle>
       <circle cx="220" cy="80" r="6"></circle>
       <circle cx="272" cy="56" r="6"></circle>
       <circle cx="301" cy="76" r="6"></circle>
       <circle cx="353" cy="58" r="6"></circle>
       <circle cx="381" cy="84" r="6"></circle>
       <circle cx="432" cy="62" r="6"></circle>
       <circle cx="461" cy="78" r="6"></circle>
     </g>
     <g stroke="currentColor" stroke-width="1" fill="none" opacity="0.4">
       <path d="M187,104 L192,66"></path>
       <path d="M227,104 L220,86"></path>
       <path d="M267,104 L272,62"></path>
       <path d="M307,104 L301,82"></path>
       <path d="M347,104 L353,64"></path>
       <path d="M387,104 L381,90"></path>
       <path d="M427,104 L432,68"></path>
       <path d="M467,104 L461,84"></path>
     </g>
     <g stroke="currentColor" stroke-width="1.5" fill="none">
       <rect x="170" y="104" width="34" height="28" rx="3"></rect>
       <rect x="210" y="104" width="34" height="28" rx="3"></rect>
       <rect x="250" y="104" width="34" height="28" rx="3"></rect>
       <rect x="290" y="104" width="34" height="28" rx="3"></rect>
       <rect x="330" y="104" width="34" height="28" rx="3"></rect>
       <rect x="370" y="104" width="34" height="28" rx="3"></rect>
       <rect x="410" y="104" width="34" height="28" rx="3"></rect>
       <rect x="450" y="104" width="34" height="28" rx="3"></rect>
     </g>
     <text x="327" y="152" font-size="11.5" fill="currentColor" opacity="0.75" text-anchor="middle">type check &#183; unbox &#183; add &#183; rebox</text>
     <path d="M484,162 C492,190 178,190 170,170" stroke="currentColor" stroke-width="1.5" fill="none" marker-end="url(#lva-ah)"></path>
     <text x="327" y="204" font-size="11.5" fill="currentColor" opacity="0.7" text-anchor="middle">repeat &#215; n</text>
     <text x="20" y="234" font-size="17" font-weight="600" fill="currentColor">NumPy array</text>
     <text x="20" y="252" font-size="11.5" fill="currentColor" opacity="0.7">one typed buffer</text>
     <g stroke="currentColor" stroke-width="1.5">
       <rect x="170" y="232" width="39" height="28" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="209" y="232" width="39" height="28" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="248" y="232" width="39" height="28" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="287" y="232" width="39" height="28" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="326" y="232" width="39" height="28" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="365" y="232" width="39" height="28" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="404" y="232" width="39" height="28" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="443" y="232" width="39" height="28" fill="currentColor" fill-opacity="0.08"></rect>
     </g>
     <path d="M170,268 L170,276 L482,276 L482,268" stroke="currentColor" stroke-width="1.5" fill="none"></path>
     <text x="326" y="294" font-size="12" fill="currentColor" text-anchor="middle">one C pass, type resolved once</text>
   </g>
 </svg>
 <figcaption>The top arrangement costs 9.8ns per element in a loop, and the bottom one is the array-struct <code>a.sum()</code> reads in 0.16 ms total</figcaption>
</figure>

## The fixed cost per call

The array call is not free either. Before Numpy touches any data it parses arguments, resolves dtypes and shapes, works out broadcasting, allocates an output buffer and sets up the ufunc. That bill is fixed per call, so there is a size below which it swamps the work. For `a * b + c` at n of 10 the array version measures 0.9x, slower than the loop.

$$\text{speedup}(n) = \frac{n \cdot c_{\text{py}}}{C_{\text{fixed}} + n \cdot c_{\text{elem}}(n)}$$

$C_{\text{fixed}}$ is that setup and $c_{\text{py}}$ is the loop's cost per element, the 9.8 ns from the last section. $c_{\text{elem}}(n)$, Numpy's cost per element, changes with $n$ : flat while the working set fits in cache, rising toward the bandwidth-bound value once the arrays spill to main memory.

<figure>
  <img src="/images/blog/numpy-thinking-in-arrays/figure-2-loop-vs-array.svg" alt="Time per call against array length. The loop line rises linearly; the Numpy line stays flat until about a thousand elements." loading="lazy">
  <figcaption>Time per call for <code>a*b + c</code> against $n$. The Numpy line is flat below roughly $n = 1000$, where the fixed per-call cost is most of the measurement.</figcaption>
</figure>

The sweep runs 0.9x at n of 10, 7.5x at 100, 44.6x at 1000, 41.3x at 10,000, a peak of 57.9x near 100,000, then 29.7x at a million and 27.4x at four million. The number I act on is the low end : below a few thousand elements the fixed cost is most of what I am paying.

<figure>
  <img src="/images/blog/numpy-thinking-in-arrays/figure-3-speedup-vs-size.svg" alt="Speedup against array length, starting below one, peaking near ten to the fifth, then settling lower." loading="lazy">
  <figcaption>Speedup of the array call over the loop for <code>a*b + c</code>, with the dashed line at parity.</figcaption>
</figure>

More data buying less speedup was not what I expected from the sweep. My best guess is cache. Near $10^5$ the operands and the temporary NumPy allocates for `a * b` still fit a fast cache level, so $c_{\text{elem}}$ sits near its floor. Past that the arrays spill, $c_{\text{elem}}$ climbs toward the rate memory can stream them at, and the ratio settles.

## Four of the six hit the same ceiling

Four of the six operations sit under one ceiling. Once the interpreter is out of the way, memory sets the pace.

`a * b + c` is the plain case at 28x, and `Numpy` does not fuse it : two ufuncs, a multiply then an add, with a full n-element temporary between them, so at a million elements the expression holds `a`, `b`, `c`, the temporary and the result at once. If the footprint bites, `np.multiply(a, b, out=a)` then `a += c`, an explicit `out=` buffer, or `numexpr`, which does fuse it. The loop allocates none of that and carries one float at a time.

That 28x and the 29.7x from the sweep are the same operation at the same $n$, two independent runs about 1.7 apart in the multiplier. I have not reconciled them beyond noting that they are two runs.

Cumulative sum is 9x, 30 ms in `for-loop` against 3.4 ms in `Numpy`. `np.cumsum` has to run left to right: `out[i]` needs `out[i-1]`, so there is one pass, no room to reorder, and a full n-element output written on top of the input read.

The smallest win, and the most instructive number of the six experiments, is the filter. The loop takes 16 ms, `numpy` takes 5.0 ms, 3.2x.

```python
mask = a > 0.0    # allocation 1 : one bool per element
a[mask]           # allocation 2, after a pass to count the True entries
```

`a > 0.0` allocates a full boolean array first, then boolean indexing then counts the `True` entries to size the output, allocates that, and makes a second pass to gather the survivors: two allocations, several passes over memory, and almost no arithmetic for a C loop to be faster. Once the mask is an array rather than a single `True`, the rest follows : `mask.sum()` counts matches, `np.where(mask, a, b)` chooses elementwise, and boolean and fancy indexing return a copy where a slice or `.T` returns a view.

## Reductions and broadcasting

The z-score is the last of the bandwidth-bound four, 35x, and it has two modes rather than just one. Reductions are the first. `sum`, `mean`, `std`, `var`, `argmax` and `argmin` are one operation with an axis argument under six names : each collapses the axis its operation runs over.

<figure>
 <svg viewBox="0 0 600 286" role="img" aria-labelledby="ax-t ax-d" style="width:100%;height:auto;max-width:560px;color:inherit" xmlns="http://www.w3.org/2000/svg">
   <title id="ax-t">Summing along an axis</title>
   <desc id="ax-d">A grid with arrows collapsing it along each of its two axes in turn.</desc>
   <defs>
     <marker id="ax-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
       <path d="M0,0 L10,5 L0,10 z" fill="currentColor"></path>
     </marker>
   </defs>
   <g aria-hidden="true" font-family="-apple-system, system-ui, sans-serif">
     <text x="182" y="52" font-size="12.5" fill="currentColor" text-anchor="middle">a, shape (3, 4)</text>
     <g stroke="currentColor" stroke-width="1.5" fill="none">
       <rect x="120" y="64" width="28" height="28" rx="3"></rect><rect x="152" y="64" width="28" height="28" rx="3"></rect><rect x="184" y="64" width="28" height="28" rx="3"></rect><rect x="216" y="64" width="28" height="28" rx="3"></rect>
       <rect x="120" y="96" width="28" height="28" rx="3"></rect><rect x="152" y="96" width="28" height="28" rx="3"></rect><rect x="184" y="96" width="28" height="28" rx="3"></rect><rect x="216" y="96" width="28" height="28" rx="3"></rect>
       <rect x="120" y="128" width="28" height="28" rx="3"></rect><rect x="152" y="128" width="28" height="28" rx="3"></rect><rect x="184" y="128" width="28" height="28" rx="3"></rect><rect x="216" y="128" width="28" height="28" rx="3"></rect>
     </g>
     <g stroke="currentColor" stroke-width="1.5" fill="none">
       <path d="M248,78 L286,78" marker-end="url(#ax-ah)"></path>
       <path d="M248,110 L286,110" marker-end="url(#ax-ah)"></path>
       <path d="M248,142 L286,142" marker-end="url(#ax-ah)"></path>
     </g>
     <g stroke="currentColor" stroke-width="1.5">
       <rect x="290" y="64" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="290" y="96" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="290" y="128" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
     </g>
     <text x="336" y="98" font-size="13" font-weight="600" fill="currentColor">sum(axis=1)</text>
     <text x="336" y="116" font-size="12" fill="currentColor">result shape (3,)</text>
     <text x="336" y="137" font-size="11" fill="currentColor" opacity="0.7">one value per row</text>
     <g stroke="currentColor" stroke-width="1.5" fill="none">
       <path d="M134,160 L134,197" marker-end="url(#ax-ah)"></path>
       <path d="M166,160 L166,197" marker-end="url(#ax-ah)"></path>
       <path d="M198,160 L198,197" marker-end="url(#ax-ah)"></path>
       <path d="M230,160 L230,197" marker-end="url(#ax-ah)"></path>
     </g>
     <g stroke="currentColor" stroke-width="1.5">
       <rect x="120" y="201" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="152" y="201" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="184" y="201" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="216" y="201" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
     </g>
     <text x="182" y="252" font-size="13" font-weight="600" fill="currentColor" text-anchor="middle">sum(axis=0) &#8594; result shape (4,)</text>
     <text x="182" y="270" font-size="11" fill="currentColor" opacity="0.7" text-anchor="middle">one value per column</text>
   </g>
 </svg>
 <figcaption>Same operation ( `sum` ) with different axis arguments</figcaption>
</figure>

The mnemonic I use is that the axis you name is the axis that vanishes.

```python
X = rng.standard_normal((1000, 50))   # (n, d)
X.sum(axis=0)   # -> (50,)
X.sum(axis=1)   # -> (1000,)
```

`dtype=np.float32` on a reduction down-casts the accumulator, which a million values will notice; integer sums up-cast to `int64` to dodge silent overflow.

Second mode is that of Broadcasting, and it is where the loop model differs : the input array shapes differ and NumPy invents the missing axis.

<figure>
 <svg viewBox="0 0 600 218" role="img" aria-labelledby="bc-t bc-d" style="width:100%;height:auto;max-width:560px;color:inherit" xmlns="http://www.w3.org/2000/svg">
   <title id="bc-t">A size-1 axis is re-read, not copied</title>
   <desc id="bc-d">Two small operands beside the larger grid they produce; dashed outlines mark positions read again rather than stored</desc>
   <g aria-hidden="true" font-family="-apple-system, system-ui, sans-serif">
     <text x="54" y="38" font-size="11" fill="currentColor" opacity="0.7" text-anchor="middle">1</text>
     <g stroke="currentColor" stroke-width="1.5" fill="none">
       <rect x="40" y="46" width="28" height="28" rx="3"></rect>
       <rect x="40" y="78" width="28" height="28" rx="3"></rect>
       <rect x="40" y="110" width="28" height="28" rx="3"></rect>
     </g>
     <g stroke="currentColor" stroke-width="1.3" fill="none" stroke-dasharray="3 3" opacity="0.55">
       <rect x="72" y="46" width="28" height="28" rx="3"></rect><rect x="104" y="46" width="28" height="28" rx="3"></rect><rect x="136" y="46" width="28" height="28" rx="3"></rect>
       <rect x="72" y="78" width="28" height="28" rx="3"></rect><rect x="104" y="78" width="28" height="28" rx="3"></rect><rect x="136" y="78" width="28" height="28" rx="3"></rect>
       <rect x="72" y="110" width="28" height="28" rx="3"></rect><rect x="104" y="110" width="28" height="28" rx="3"></rect><rect x="136" y="110" width="28" height="28" rx="3"></rect>
     </g>
     <text x="54" y="162" font-size="12" fill="currentColor" text-anchor="middle">(3, 1)</text>
     <text x="182" y="98" font-size="20" fill="currentColor" text-anchor="middle">+</text>
     <text x="192" y="64" font-size="11" fill="currentColor" opacity="0.7" text-anchor="middle">1</text>
     <g stroke="currentColor" stroke-width="1.5" fill="none">
       <rect x="206" y="46" width="28" height="28" rx="3"></rect><rect x="238" y="46" width="28" height="28" rx="3"></rect><rect x="270" y="46" width="28" height="28" rx="3"></rect><rect x="302" y="46" width="28" height="28" rx="3"></rect>
     </g>
     <g stroke="currentColor" stroke-width="1.3" fill="none" stroke-dasharray="3 3" opacity="0.55">
       <rect x="206" y="78" width="28" height="28" rx="3"></rect><rect x="238" y="78" width="28" height="28" rx="3"></rect><rect x="270" y="78" width="28" height="28" rx="3"></rect><rect x="302" y="78" width="28" height="28" rx="3"></rect>
       <rect x="206" y="110" width="28" height="28" rx="3"></rect><rect x="238" y="110" width="28" height="28" rx="3"></rect><rect x="270" y="110" width="28" height="28" rx="3"></rect><rect x="302" y="110" width="28" height="28" rx="3"></rect>
     </g>
     <text x="268" y="162" font-size="12" fill="currentColor" text-anchor="middle">(1, 4)</text>
     <text x="350" y="98" font-size="20" fill="currentColor" text-anchor="middle">=</text>
     <g stroke="currentColor" stroke-width="1.5">
       <rect x="376" y="46" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect><rect x="408" y="46" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect><rect x="440" y="46" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect><rect x="472" y="46" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="376" y="78" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect><rect x="408" y="78" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect><rect x="440" y="78" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect><rect x="472" y="78" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
       <rect x="376" y="110" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect><rect x="408" y="110" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect><rect x="440" y="110" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect><rect x="472" y="110" width="28" height="28" rx="3" fill="currentColor" fill-opacity="0.08"></rect>
     </g>
     <text x="438" y="162" font-size="12" fill="currentColor" text-anchor="middle">(3, 4)</text>
     <text x="300" y="200" font-size="11" fill="currentColor" opacity="0.7" text-anchor="middle">dashed cells are re-reads of the same memory</text>
   </g>
 </svg>
 <figcaption>Adding a $(3,1)$ column to a $(1,4)$ row => Only the $(3,4)$ result is allocated</figcaption>
</figure>

The rule : compare shapes from the trailing axis, treat a missing leading axis as length 1, and call two axes compatible when they are equal or one of them is 1. So `(2,3)` with `(3,)` works and `(2,3)` with `(2,)` fails, even though a left-to-right reading says it should work. A stretched axis is a stride-0 axis: the iterator re-reads one address as it walks that dimension instead of copying anything.

Z-scoring a matrix by column needs both shapes at once. The loop is two nested passes and easy to index wrong :

```python
# for-loop version : lists of lists
for j in range(d):
    col = [row[j] for row in rows]
    m = sum(col) / n
    s = (sum((v - m) ** 2 for v in col) / n) ** 0.5
    for i in range(n):
        out[i][j] = (rows[i][j] - m) / s


# Numpy version

(X - X.mean(0)) / X.std(0)      # (n, d) - (1, d), then / (1, d)
```

169 ms in `for` loop against 4.8 ms in vectorized version.

## Where BLAS takes over

The dot product is the outlier, 382x, 27.5 ms ( `for` loop ) against 0.072 ms ( `numpy` ), and it is fast for a different reason than everything above.

Every cheat-sheet I learned from conflated two products. `a * b` is elementwise, one ufunc writing an n-element output; `a @ b` contracts over the shared axis. I pick between them by asking what the loop body would have been : one multiply per cell means `*`, a sum of products means `@`. (`np.dot` is the legacy spelling and diverges above two dimensions.)

```python
a * b     # (n,) * (n,) -> (n,)
a @ b     # (n,) @ (n,) -> scalar
```

`@` hands that contraction to a BLAS kernel : cache tiling, SIMD, and a lot of tuning. BLAS libraries are commonly multi-threaded as well, but the repo records neither the library nor a thread count, so I would not lean on threading to explain this run. Part of the 382x is also a heavier baseline : the dot loop does a multiply and a running add per step, so it has further to fall than the elementwise loop.

Floating-point addition is not associative, so a left fold, `np.sum` and a BLAS dot all round differently. `np.sum` has been pairwise since Numpy 1.9, its error growing like $O(\varepsilon \log n)$ against $O(\varepsilon n)$ for the naive fold, so over a million standard-normal draws the two agree to about fifteen significant digits and part ways after that, the array result being the more accurate one. For a reduction or a dot, "same computation, just faster" is slightly wrong: it is the same math up to reassociation.

## One ratio behind the spread

<figure>
  <img src="/images/blog/numpy-thinking-in-arrays/figure-1-speedup-by-op.svg" alt="Horizontal bar chart on a log axis ranking six operations by speedup, from filter at the bottom to dot at the top." loading="lazy">
  <figcaption>All six operations at $n = 10^6$ on one log axis.</figcaption>
</figure>

One observation accounts for all six experiments : the win scales with how much arithmetic the operation does per byte it moves. The dot does a million multiply-adds and returns one number, so it runs compute-bound and gets 382x. The filter reads n floats, writes a mask, reads it back and writes the survivors, doing almost no arithmetic on the way, so it stays bandwidth-bound at 3.2x.

The $n$ itself is a separate lever and a larger one - a double sum over $x_i y_j$ costs $O(n^2)$ as two nested loops and $O(n)$ written as $(\sum_i x_i)(\sum_j y_j)$, because the sums factor. Vectorizing the nested loop keeps the $n^2$; while rewriting the expression removes it.

<style>
a.btn-soft {
  display: inline-block;
  margin: 0 0.5rem 0.55rem 0;
  padding: 0.5em 1.05em;
  font-size: 0.92rem;
  font-weight: 500;
  color: inherit;
  text-decoration: none;
  border-radius: 8px;
  border: 1px solid color-mix(in srgb, currentColor 22%, transparent);
  background: color-mix(in srgb, currentColor 5%, transparent);
  transition: background 0.15s ease, border-color 0.15s ease;
}
a.btn-soft:hover {
  border-color: color-mix(in srgb, currentColor 38%, transparent);
  background: color-mix(in srgb, currentColor 10%, transparent);
}
</style>

[Browse the code and reproduce the benchmarks](https://github.com/nilesh-patil/numpy-thinking-in-arrays){: .btn-soft .btn-soft--primary} [NumPy broadcasting docs](https://numpy.org/doc/stable/user/basics.broadcasting.html){: .btn-soft}
