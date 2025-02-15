---
layout: single
title: "Two-phase simplex in Python and Rust - a browser-trusted linear-programming solver"
date: 2025-02-15T19:40:00+05:30
categories: [blog]
tags: [optimization, linear-programming, simplex, rust, python, webassembly, visualization]
excerpt: "An interactive explainer of linear programming needs a solver it can trust running in the reader's browser tab. The same two-phase simplex, written twice: a 408-line Python reference and a line-accurate Rust port, plus one small program to walk through how it works and how I know it is right. The Python-versus-Rust timing is at the end."
math: true
header:
  overlay_image: /images/blog/headers/simplex-solver-twice.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/simplex-solver-twice.jpg
---

In the fall of 1947, [nine clerks at the National Bureau of Standards](https://doi.org/10.1287/opre.49.1.1.11187) spent roughly 120 person-days on desk calculators solving one linear program: Jack Laderman's team ran George Stigler's cheapest-adequate-diet problem, 77 foods against 9 nutrient minimums, through the brand-new simplex method. They landed on \$39.69 a year at 1939 prices, 24 cents under the estimate Stigler had published in 1945. Until that computation nobody could prove how close his guess had been.

Last summer I built [an interactive field guide to linear programming]({{ site.baseurl }}/feasible-region/), and the figure at the top of it re-runs that kind of computation on every mouse drag. Grab a constraint line, and a simplex solver compiled to WebAssembly re-solves the program before your finger leaves the trackpad. This post is about the engineering underneath that page, and one decision in particular: I wrote the solver twice, in two languages, and made the two implementations police each other.

If your daily work is gradient descent and scipy, linear programming can read like someone else's subject. It is closer than that. The exact Wasserstein distance is a linear program, [Kantorovich's transport problem](https://arxiv.org/abs/1803.00567), and the classical algorithm for it is a cousin of the solver in this post, and a `scipy.optimize.linprog` call in a data pipeline lands on the same mathematics. So I will follow one small program the whole way through, and the corners are where it starts. The Python-versus-Rust timing sits in its own section at the end, because it answers a different question than the rest of the post.

## A region with corners

Take the smallest worked example on the site, the sketch this post keeps returning to. A fabricator bay can pour two products, and profit is $3x_1 + 2x_2$. Feedstock allows $x_1 + x_2 \le 4$, press time allows $x_1 + 3x_2 \le 6$, and you cannot pour negative amounts. Each limit cuts the plane in half; keeping the allowed side of every cut leaves a polygon of feasible plans, four corners in this case.

<figure>
  <svg viewBox="0 0 560 360" role="img" aria-labelledby="fig-region-title fig-region-desc"
       style="width:100%; max-width:560px; height:auto; color:inherit"
       xmlns="http://www.w3.org/2000/svg">
    <title id="fig-region-title">The feasible region of the product-mix program, with the level line sweeping to a corner</title>
    <desc id="fig-region-desc">Two constraint lines carve a shaded four-corner region; dashed profit lines sweep outward until the last one touches the region only at the corner (4, 0).</desc>
    <defs>
      <marker id="fig-region-arrow" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <!-- axes -->
    <g stroke="currentColor" stroke-width="1.2" opacity="0.55" aria-hidden="true">
      <line x1="60" y1="310" x2="520" y2="310" marker-end="url(#fig-region-arrow)"/>
      <line x1="60" y1="310" x2="60" y2="40" marker-end="url(#fig-region-arrow)"/>
    </g>
    <text x="524" y="330" font-size="12" fill="currentColor" opacity="0.75">x1</text>
    <text x="42" y="38" font-size="12" fill="currentColor" opacity="0.75">x2</text>
    <!-- feasible region: (0,0) (4,0) (3,1) (0,2) -->
    <path d="M60,310 L380,310 L300,230 L60,150 Z" fill="currentColor" fill-opacity="0.10" stroke="none"/>
    <!-- constraint: x1 + x2 = 4 -->
    <line x1="100" y1="30" x2="380" y2="310" stroke="currentColor" stroke-width="1.8"/>
    <text x="136" y="50" font-size="12.5" fill="currentColor">x1 + x2 &#8804; 4</text>
    <!-- constraint: x1 + 3 x2 = 6 -->
    <line x1="60" y1="150" x2="500" y2="297" stroke="currentColor" stroke-width="1.8"/>
    <text x="430" y="270" font-size="12.5" fill="currentColor">x1 + 3x2 &#8804; 6</text>
    <!-- level lines 3x1 + 2x2 = c, dashed, sweeping right -->
    <g stroke="currentColor" stroke-dasharray="6 5" stroke-width="1.4" opacity="0.8" aria-hidden="true">
      <line x1="220" y1="310" x2="60" y2="70"/>
      <line x1="380" y1="310" x2="220" y2="70"/>
    </g>
    <text x="220" y="330" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.8">profit = 6</text>
    <text x="232" y="62" font-size="11.5" fill="currentColor" opacity="0.8">profit = 12</text>
    <line x1="192" y1="192" x2="254" y2="164" stroke="currentColor" stroke-width="1.4"
          marker-end="url(#fig-region-arrow)" opacity="0.8"/>
    <text x="186" y="212" font-size="11.5" fill="currentColor" opacity="0.8">better</text>
    <!-- corners -->
    <g fill="currentColor" aria-hidden="true">
      <circle cx="60" cy="310" r="4"/>
      <circle cx="300" cy="230" r="4"/>
      <circle cx="60" cy="150" r="4"/>
    </g>
    <circle cx="380" cy="310" r="5.5" fill="currentColor"/>
    <circle cx="380" cy="310" r="10" fill="none" stroke="currentColor" stroke-width="1.6"/>
    <text x="396" y="334" font-size="12.5" fill="currentColor" font-weight="600">(4, 0), profit 12</text>
  </svg>
  <figcaption>The product-mix program. Two limits carve a four-corner region, and the dashed profit line leaves the region for the last time at (4, 0).</figcaption>
</figure>

Every plan with the same profit sits on one straight line, and raising the profit slides that line across the plane without turning it. Slide it as far as it will go while still touching the region and it comes to rest against the boundary. A flat line leaving a polygon last touches it at a corner, or along a whole edge in the case of a tie, and an edge has corners at its ends. So if a best plan exists at all, one of the corners is a best plan, and there are only finitely many corners to check. (Regions can also be empty, or open in the direction of profit; two of the recorded fixtures exist to pin exactly those verdicts.)

Checking four corners is nothing. But the diet problem's region lives in 77 dimensions, corner counts grow exponentially with dimension, and "check the corners" stops being a plan almost immediately. What matters is how many corners a solver actually has to visit. Later on I measure that, on a family of programs built to give the worst possible answer.

## Watching one walk

The simplex method, which Dantzig devised in 1947, visits corners the way you would climb stairs: start at one, move to an adjacent corner that improves the objective, stop when no neighbor is better. The bookkeeping happens in a tableau, a matrix that holds every constraint with a slack variable added per row. One move between corners is a pivot, and a pivot makes two choices. The entering column answers "which direction improves profit fastest right now," read off the objective row. The leaving row answers "how far can I go before some limit stops me": divide each row's remaining budget by what the step costs it, and the smallest ratio names the first wall you hit. Then one round of Gauss-Jordan elimination rewrites the tableau in terms of the new corner.

The explainer's centerpiece scales the fabricator bay up to three products under five named limits: the cargo problem, maximize $x_1 + x_2 + x_3$ subject to $-x_1 + x_2 \le 5$, $x_1 + 4x_2 \le 45$, $2x_1 + x_2 \le 27$, $3x_1 - 4x_2 \le 24$, and $x_3 \le 4$. That walk is the one this post keeps re-solving. The site draws it twice at once: on the left, the feasible region as a wireframe polytope with the path traced corner to corner; on the right, the tableau after each pivot, with an objective staircase climbing 0, 8, 15, 19, 22. Both panels read from the same recorded step.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simplex-solver-twice/site-dualview-figure.png"
       alt="The explainer at step 3: the walk traced on a polytope, the matching tableau in exact fractions, and the objective staircase at 19.">
  <figcaption>One recorded step, drawn as the path on the polytope, the exact-fraction tableau, and the objective staircase at 19.</figcaption>
</figure>

Watch any one of those pivots up close and it is nothing but row reduction: divide the pivot row, clear the column above and below it. "A pivot is row reduction" is the one-line summary I wish I had been handed earlier. It is the elimination from CLRS chapter 28, and chapter 29 builds the whole simplex procedure on top of it.

## The bet: write it twice

The figures above are not videos. When a reader drags a limit line, something in the page has to actually re-solve the program, in milliseconds, offline, on whatever device the reader brought. That something runs unattended in strangers' browser tabs, where I cannot attach a debugger, read a stack trace, or ship a hotfix before the reader notices. I wanted a solver I could reason about completely. And I was not sure I could get a from-scratch simplex right on the first try.

So I wrote it twice:

- **The reference** is 408 lines of pure Python, `reference.py`, one `_Simplex` class, no numpy. Its module docstring sets the priorities: "Correctness and clarity outrank speed." It exists to be read, to be tested against textbooks, and to record traces.
- **The port** is `feasible-core`, 714 lines of Rust. It is a port by line: identical column ordering, identical `EPS = 1e-9` comparisons, identical pivot selection and tie-breaks. The crate's own header calls that fidelity load-bearing, and the test suite enforces it mechanically. It ships twice, as a 118 KB WebAssembly binary for the page and as a PyO3 extension for Python.

Both implementations are deliberately boring inside: a dense tableau and two phases (phase 1 manufactures a feasible corner when the origin is not one, phase 2 optimizes). On top of that, the Rust core follows a strict determinism regime:

- no `HashMap` iteration anywhere near the algebra
- no randomness, no time, no threads
- zero `unsafe`
- solving the same program twice must produce byte-identical output

## Traces, and the net around them

What follows is the QA under the solver: how the reference records its own work, and the blind spot the tests left open. If you came for the geometry and the benchmark, skip to [how the solver fails on demand](#two-ways-to-ruin-a-walk) without losing the thread.

The reference does one thing the port never does in production: it writes everything down. Ask it to record a solve and it emits a trace, a JSON document whose every step carries the tableau snapshot, the basis, the vertex, the entering and leaving variables, and the objective, captured just before each pivot lands. Nine fixtures get recorded, from the two-variable sketch to the cycling exhibit, and a log-barrier tracer adds four interior-point paths for the site's race figure. Those 13 files do triple duty: the golden tests demand each committed file back byte for byte, the page replays them when WASM will not load, and every figure here (the staircase, the tableau, the walk on the polytope) is a rendering of trace fields, nothing typed in by hand.

One trick I would reuse anywhere algorithm state gets recorded: the artificial variables phase 1 adds are kept as dead columns for the whole solve instead of being dropped once used. Wasteful by textbook standards, but it holds every tableau snapshot to one width, an invariant the tests check on every fixture.

Around the traces sits the test net in `tests/`: hand-derived anchors that pin the cargo walk's exact corner sequence, so a solver that reaches 22 by a different route fails; structural invariants (constant tableau width, monotone phases, a leaving variable basic before its pivot and gone after); and cross-implementation parity, where the Rust core must reproduce the reference's walk: the same step count, basis, and pivots, every float agreeing within 1e-9. But parity and my own hand arithmetic share the same blind spots. So the property suite in `tests/test_property.py` (60 seeds) and the cross-backend fuzz in `tests/test_native_parity.py` (40 more) draw random programs and hand each to scipy's `linprog`, which has shipped the [HiGHS solver since SciPy 1.6 and defaulted to it since 1.9](https://docs.scipy.org/doc/scipy/release/1.9.0-notes.html), sharing neither code nor algorithm with mine. When a program has several optimal corners the two routinely disagree on which, and both are right, so the suite compares objective values to 1e-6, then checks each answer is feasible and on a genuine vertex by matrix rank, so degenerate corners still pass.

<figure>
  <svg viewBox="0 0 680 340" role="img" aria-labelledby="fig-arch-title fig-arch-desc"
       style="width:100%; max-width:680px; height:auto; color:inherit"
       xmlns="http://www.w3.org/2000/svg">
    <title id="fig-arch-title">One solver written twice, and everything that reads from each copy</title>
    <desc id="fig-arch-desc">The Python reference and the Rust core sit side by side, parity-checked; traces recorded by the reference feed the tests and the site, while the Rust core ships to the page as WASM and to Python via PyO3.</desc>
    <defs>
      <marker id="fig-arch-arrow" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <!-- row 1: the two implementations -->
    <rect x="30" y="24" width="240" height="58" rx="7" fill="currentColor" fill-opacity="0.10" stroke="currentColor" stroke-width="1.6"/>
    <text x="150" y="48" text-anchor="middle" font-size="13.5" fill="currentColor" font-weight="600">reference.py</text>
    <text x="150" y="68" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.8">pure Python, 408 lines</text>
    <rect x="410" y="24" width="240" height="58" rx="7" fill="currentColor" fill-opacity="0.10" stroke="currentColor" stroke-width="1.6"/>
    <text x="530" y="48" text-anchor="middle" font-size="13.5" fill="currentColor" font-weight="600">feasible-core</text>
    <text x="530" y="68" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.8">Rust port, 714 lines</text>
    <line x1="274" y1="53" x2="406" y2="53" stroke="currentColor" stroke-width="1.5"
          marker-end="url(#fig-arch-arrow)" marker-start="url(#fig-arch-arrow)"/>
    <text x="340" y="42" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.85">parity-tested</text>
    <!-- row 2: traces / wasm / pyo3 -->
    <rect x="30" y="140" width="240" height="58" rx="7" fill="none" stroke="currentColor" stroke-width="1.6"/>
    <text x="150" y="164" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">traces/*.json</text>
    <text x="150" y="184" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.8">13 recorded walks, byte-stable</text>
    <rect x="352" y="140" width="150" height="58" rx="7" fill="none" stroke="currentColor" stroke-width="1.6"/>
    <text x="427" y="164" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">feasible-wasm</text>
    <text x="427" y="184" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.8">118 KB binary</text>
    <rect x="522" y="140" width="128" height="58" rx="7" fill="none" stroke="currentColor" stroke-width="1.6"/>
    <text x="586" y="164" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">feasible-py</text>
    <text x="586" y="184" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.8">PyO3 extension</text>
    <!-- row 3: consumers -->
    <rect x="30" y="256" width="240" height="58" rx="7" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="7 4"/>
    <text x="150" y="280" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">test suite + HiGHS oracle</text>
    <text x="150" y="300" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.8">goldens, parity, fuzz</text>
    <rect x="352" y="256" width="298" height="58" rx="7" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="7 4"/>
    <text x="501" y="280" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">the page</text>
    <text x="501" y="300" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.8">live figures, trace replay as fallback</text>
    <!-- arrows -->
    <g stroke="currentColor" stroke-width="1.4" fill="none">
      <line x1="150" y1="86" x2="150" y2="136" marker-end="url(#fig-arch-arrow)"/>
      <line x1="470" y1="86" x2="430" y2="136" marker-end="url(#fig-arch-arrow)"/>
      <line x1="560" y1="86" x2="584" y2="136" marker-end="url(#fig-arch-arrow)"/>
      <line x1="150" y1="202" x2="150" y2="252" marker-end="url(#fig-arch-arrow)"/>
      <line x1="427" y1="202" x2="450" y2="252" marker-end="url(#fig-arch-arrow)"/>
      <line x1="586" y1="202" x2="230" y2="256" marker-end="url(#fig-arch-arrow)"/>
      <path d="M272,180 C320,196 360,230 392,252" marker-end="url(#fig-arch-arrow)"/>
    </g>
    <text x="122" y="122" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.85">records</text>
    <text x="316" y="238" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.85">fallback</text>
    <text x="432" y="242" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.85">live</text>
    <text x="404" y="228" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.85" transform="rotate(-14 404 228)">parity</text>
  </svg>
  <figcaption>The reference records the traces and anchors the tests; the Rust port is what actually ships, to the browser and to Python.</figcaption>
</figure>

## The hole in the net

Every shipped fixture happens to be a maximization with all-below limits and non-negative right-hand sides, and the random generator stays inside that class too. So the phase-1 machinery, the minimization signs, and the row-flip dual logic had zero coverage from goldens and fuzz alike.

Two real bug classes lived in that shadow. I found them while broadening the suite beyond the fixture catalogue, before the Rust port existed. A lingering phase-1 artificial variable could leave the solver reporting an "optimal" vertex that violated its own constraints, with an objective better than the true optimum (on one two-constraint minimization: x = 2, objective -2, on a program whose only feasible point is 0), or a false unbounded verdict on a bounded program. The dual read-out came back with the wrong sign on above-limit constraints, +1 where the true shadow price is -1, which breaks strong duality on every such program.

Parity testing could not have caught either one. The port copies the reference line by line, mistakes included, so both copies would have agreed while both were wrong. Catching that kind of bug took an outside solver and test programs from outside my own habits. The regression file now reproduces both, sweeping the artificial-variable cases across all three pivot rules, and a 17-line docstring in the reference explains the failure so the next reader does not have to rediscover it.

That correctness net is 86 Python test functions under `tests/`, fanning out to roughly 260 parametrized cases, plus about 1,070 lines of Rust tests under `crates/feasible-core/tests/`, against the 714 lines of solver in `crates/feasible-core/src/lib.rs`.

## Two ways to ruin a walk

So the net proves the solver is right when it terminates. But does it always terminate? Linear programming has two famous ways to not terminate well, and the site ships both as exhibits. The solver had to be able to fail on demand.

The first is degeneracy. When a corner carries more tight constraints than the dimension needs, a pivot can move the algebra without moving the point. On the sketch it takes exactly one redundant wall to manufacture the situation:

<figure>
  <svg viewBox="0 0 560 360" role="img" aria-labelledby="fig-degen-title fig-degen-desc"
       style="width:100%; max-width:560px; height:auto; color:inherit"
       xmlns="http://www.w3.org/2000/svg">
    <title id="fig-degen-title">A redundant third wall makes the corner (3, 1) degenerate</title>
    <desc id="fig-degen-desc">Three constraint lines pass through one corner, so a pivot can swap which slack the basis names while the point itself never moves.</desc>
    <defs>
      <marker id="fig-degen-arrow" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <!-- axes -->
    <g stroke="currentColor" stroke-width="1.2" opacity="0.55" aria-hidden="true">
      <line x1="60" y1="310" x2="520" y2="310" marker-end="url(#fig-degen-arrow)"/>
      <line x1="60" y1="310" x2="60" y2="40" marker-end="url(#fig-degen-arrow)"/>
    </g>
    <text x="524" y="330" font-size="12" fill="currentColor" opacity="0.75">x1</text>
    <text x="42" y="38" font-size="12" fill="currentColor" opacity="0.75">x2</text>
    <!-- feasible region unchanged: the third wall is redundant -->
    <path d="M60,310 L380,310 L300,230 L60,150 Z" fill="currentColor" fill-opacity="0.10" stroke="none"/>
    <!-- constraint: x1 + x2 = 4 -->
    <line x1="100" y1="30" x2="380" y2="310" stroke="currentColor" stroke-width="1.8"/>
    <text x="136" y="50" font-size="12.5" fill="currentColor">x1 + x2 &#8804; 4</text>
    <!-- constraint: x1 + 3 x2 = 6 -->
    <line x1="60" y1="150" x2="500" y2="297" stroke="currentColor" stroke-width="1.8"/>
    <text x="430" y="270" font-size="12.5" fill="currentColor">x1 + 3x2 &#8804; 6</text>
    <!-- redundant third wall x1 + 2 x2 = 5, grazing (3, 1) -->
    <line x1="60" y1="110" x2="500" y2="330" stroke="currentColor" stroke-width="1.5" opacity="0.7"/>
    <text x="150" y="152" font-size="12" fill="currentColor" opacity="0.85"
          text-anchor="middle" transform="rotate(26.5 150 152)">x1 + 2x2 &#8804; 5 (redundant)</text>
    <!-- the two basis chips pointing at the same corner -->
    <rect x="330" y="64" width="176" height="32" rx="7" fill="none" stroke="currentColor" stroke-width="1.4"/>
    <text x="418" y="85" text-anchor="middle" font-size="12" fill="currentColor">basis: x1, x2, s1 = 0</text>
    <rect x="330" y="118" width="176" height="32" rx="7" fill="none" stroke="currentColor" stroke-width="1.4"/>
    <text x="418" y="139" text-anchor="middle" font-size="12" fill="currentColor">basis: x1, x2, s3 = 0</text>
    <line x1="418" y1="99" x2="418" y2="115" stroke="currentColor" stroke-width="1.3"
          marker-end="url(#fig-degen-arrow)" marker-start="url(#fig-degen-arrow)"/>
    <text x="428" y="111" font-size="11.5" fill="currentColor" opacity="0.85">one pivot</text>
    <g stroke="currentColor" stroke-width="1.3" fill="none" opacity="0.85" aria-hidden="true">
      <line x1="356" y1="150" x2="308" y2="222" marker-end="url(#fig-degen-arrow)"/>
    </g>
    <!-- the degenerate corner -->
    <circle cx="300" cy="230" r="5.5" fill="currentColor"/>
    <circle cx="300" cy="230" r="10" fill="none" stroke="currentColor" stroke-width="1.6"/>
    <text x="284" y="262" text-anchor="end" font-size="12.5" fill="currentColor" font-weight="600">(3, 1): the point never moves</text>
    <!-- other corners -->
    <g fill="currentColor" aria-hidden="true">
      <circle cx="60" cy="310" r="4"/>
      <circle cx="380" cy="310" r="4"/>
      <circle cx="60" cy="150" r="4"/>
    </g>
  </svg>
  <figcaption>Three walls meet at (3, 1), one more than two dimensions need, so a pivot can swap which slack the basis names while the point stays put. A naive rule can loop on a stall like this forever.</figcaption>
</figure>

A redundant wall through one corner is harmless on its own; the canonical demonstration of the harm is a small maximization from Chvátal's 1983 textbook, built so the stalls chain into a loop. My solver implements three pivot rules: greedy Dantzig with a sane tie-break, Bland's rule, whose 1977 paper proved cycling impossible, and a deliberately naive Dantzig that resolves ties by first row, the way a first draft would. The naive rule is the one that breaks: on Chvátal's program it never terminates, and the repo ships the recording, a 31-step trace truncated at its 30-pivot cap, as a golden fixture pinned by the same byte-exact machinery as the successes.

What I did not expect was where my safeguarded rule would land. The Rust acceptance test hedges its own claim in a comment: greedy must either reach the optimum or hit the cap, never cycle forever. Measuring for this post settled which side of that hedge Chvátal's program sits on:

<figure>
  <img src="{{ site.baseurl }}/images/blog/simplex-solver-twice/figure-1-pivots-by-rule.svg"
       alt="Pivot counts per fixture: the rules roughly tie on ordinary problems, but on the Chvatal LP only Bland terminates.">
  <figcaption>Pivots to termination per fixture, log scale. Dantzig and Bland tie on four of six fixtures and split mildly on the Klee-Minty cube; Chvátal's degenerate LP is the real divide, and only Bland exits it.</figcaption>
</figure>

Greedy cycles too, tie-break and all, spinning until the 10,000-pivot cap trips. (The golden trace stops at 30 to keep the committed artifact small; the measurement raises the cap to 10,000 and the cycle still never exits.) So the tie-break I trusted turned out to be just hygiene. With a greedy entering rule, no leaving-row tie-break alone prevents cycling on this program. Only Bland's rule, which constrains both choices, terminates, in 7 pivots.

The second failure is slowness, and it is the one Klee and Minty made famous in 1972: a family of squashed hypercubes shaped so the greedy rule visits every single corner. The site draws the three-dimensional cube and its 8-stop tour. For this post I generated the family up to dimension 12 and let both rules loose:

<figure>
  <img src="{{ site.baseurl }}/images/blog/simplex-solver-twice/figure-2-kleeminty-growth.svg"
       alt="Measured pivots on Klee-Minty cubes to dimension 12: Dantzig sits exactly on the 2 to the n minus 1 curve; Bland grows slower here.">
  <figcaption>Measured pivots on generated Klee-Minty cubes, n = 3 through 12. Dantzig lands exactly on $2^n - 1$ at every n; Bland stays far below on this family.</figcaption>
</figure>

The greedy counts are not approximately exponential, they are exactly $2^n - 1$, every corner of every cube, through all 4,095 pivots of the twelve-dimensional case. Bland takes 465 pivots at n = 12 against those 4,095, though that says nothing general in its favor. Adversarial families exist for Bland too. At n = 200 the corner count is a 61-digit number, which is the site's way of showing you cannot get there by listing corners. Simplex stays fast in practice because real programs are not adversarial.

## The price row

The walk finishes on the top of that staircase, 22: the first corner whose neighbors all sit lower, so no pivot climbs higher and the search stops, exactly the stopping rule the climb started from. By then the solver has computed more than the best plan. The objective row is also holding prices, and the sketch shows what they mean before any algebra:

<figure>
  <svg viewBox="0 0 760 330" role="img" aria-labelledby="fig-price-title fig-price-desc"
       style="width:100%; max-width:760px; height:auto; color:inherit"
       xmlns="http://www.w3.org/2000/svg">
    <title id="fig-price-title">The shadow price read directly off the sketch region</title>
    <desc id="fig-price-desc">Moving the binding feedstock wall out one unit grows the region and slides the optimum from profit 12 to 15; the slack press-time wall changes nothing.</desc>
    <defs>
      <marker id="fig-price-arrow" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
      </marker>
    </defs>
    <!-- panel A: feedstock <= 4 -->
    <text x="60" y="66" font-size="13" fill="currentColor" font-weight="600">feedstock &#8804; 4</text>
    <g stroke="currentColor" stroke-width="1.2" opacity="0.55" aria-hidden="true">
      <line x1="60" y1="260" x2="352" y2="260" marker-end="url(#fig-price-arrow)"/>
      <line x1="60" y1="260" x2="60" y2="104" marker-end="url(#fig-price-arrow)"/>
    </g>
    <path d="M60,260 L260,260 L210,210 L60,160 Z" fill="currentColor" fill-opacity="0.10" stroke="none"/>
    <line x1="135" y1="135" x2="285" y2="285" stroke="currentColor" stroke-width="1.8"/>
    <text x="232" y="176" font-size="11.5" fill="currentColor">x1 + x2 &#8804; 4</text>
    <line x1="60" y1="160" x2="345" y2="255" stroke="currentColor" stroke-width="1.8"/>
    <text x="302" y="232" font-size="11.5" fill="currentColor">x1 + 3x2 &#8804; 6</text>
    <line x1="185" y1="148" x2="260" y2="260" stroke="currentColor" stroke-width="1.4"
          stroke-dasharray="6 5" opacity="0.8" aria-hidden="true"/>
    <text x="120" y="126" font-size="11.5" fill="currentColor" opacity="0.8">profit = 12</text>
    <circle cx="260" cy="260" r="5" fill="currentColor"/>
    <circle cx="260" cy="260" r="9.5" fill="none" stroke="currentColor" stroke-width="1.6"/>
    <text x="260" y="290" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">(4, 0), profit 12</text>
    <!-- panel B: feedstock <= 5 -->
    <text x="440" y="66" font-size="13" fill="currentColor" font-weight="600">feedstock &#8804; 5</text>
    <g stroke="currentColor" stroke-width="1.2" opacity="0.55" aria-hidden="true">
      <line x1="440" y1="260" x2="732" y2="260" marker-end="url(#fig-price-arrow)"/>
      <line x1="440" y1="260" x2="440" y2="104" marker-end="url(#fig-price-arrow)"/>
    </g>
    <path d="M440,260 L690,260 L665,235 L440,160 Z" fill="currentColor" fill-opacity="0.10" stroke="none"/>
    <line x1="515" y1="135" x2="665" y2="285" stroke="currentColor" stroke-width="1.5" opacity="0.28"/>
    <line x1="565" y1="135" x2="695" y2="265" stroke="currentColor" stroke-width="1.8"/>
    <line x1="440" y1="160" x2="725" y2="255" stroke="currentColor" stroke-width="1.8"/>
    <text x="452" y="204" font-size="11.5" fill="currentColor" opacity="0.85">price 0 (slack)</text>
    <line x1="615" y1="148" x2="690" y2="260" stroke="currentColor" stroke-width="1.4"
          stroke-dasharray="6 5" opacity="0.8" aria-hidden="true"/>
    <text x="548" y="126" font-size="11.5" fill="currentColor" opacity="0.8">profit = 15</text>
    <circle cx="690" cy="260" r="5" fill="currentColor"/>
    <circle cx="690" cy="260" r="9.5" fill="none" stroke="currentColor" stroke-width="1.6"/>
    <text x="682" y="290" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">(5, 0), profit 15</text>
    <!-- the +1 unit shift -->
    <line x1="646" y1="308" x2="684" y2="308" stroke="currentColor" stroke-width="1.3"
          marker-end="url(#fig-price-arrow)"/>
    <text x="638" y="312" text-anchor="end" font-size="12" fill="currentColor">+1 unit of feedstock, +3 profit</text>
  </svg>
  <figcaption>Push the binding feedstock wall out one unit and the best corner slides from (4, 0) to (5, 0), profit 12 to 15: that wall's shadow price is 3 per unit. The press-time wall stays slack, so its price is 0.</figcaption>
</figure>

The cargo problem's tableau does the same accounting in higher dimension. When its walk finishes, the slack columns of the objective row hold the dual prices: at a non-degenerate optimum, and within the range where the optimal basis keeps holding, that is what one more unit of each limit would be worth. If you know these as Lagrange multipliers, that is exactly what they are: the shadow price on a constraint is the multiplier on it, the sensitivity of the optimum to loosening that limit. It is the same object that turns up in any KKT condition (Boyd and Vandenberghe work out the LP case in chapter 5 of Convex Optimization), and the same reason an SVM keeps only a handful of support vectors. A training point with a zero dual multiplier sits off the margin and moves the boundary not at all, exactly as a slack constraint here carries a zero price and can loosen without changing the plan; the points that set the boundary are the ones whose multiplier is nonzero, the binding constraints.

Three of the cargo problem's five limits bind at the optimum, so three slack columns of the final z-row carry a nonzero price: $-1/7$ on $x_1 + 4x_2 \le 45$, $-3/7$ on $2x_1 + x_2 \le 27$, and $-1$ on $x_3 \le 4$ (the row stores the objective negated, so flip the signs to read the prices; the explainer renders them as exact fractions instead of $-0.142857$, $-0.428571$, and $-1$). Charge each price against its limit and they stack to exactly the optimal profit: $\tfrac{1}{7}(45) + \tfrac{3}{7}(27) + 1(4) = \tfrac{45}{7} + \tfrac{81}{7} + 4 = 22$. That identity is strong duality.

Full disclosure about those tidy fractions: the solvers compute in ordinary 64-bit floats with an epsilon of 1e-9, in both languages. The fractions are a display-layer reconstruction, snapping a float to the nearest small-denominator rational when one sits within tolerance, and falling back to decimals when none does. Exact rational arithmetic in the engine was never needed at this problem scale, and floats are what keep 64-dimensional fuzz problems cheap. The golden tests would catch any float drift in CI first.

## Timing the Rust port

Porting bought trust and deployability. It also set up an unusually clean benchmark, since both engines execute the identical sequence of pivots on the identical arithmetic, and the tests prove the walks identical, so whatever gap the clock shows comes from the execution substrate alone. I timed seeded random dense programs from 4 to 512 variables, five programs per size, medians over repeated runs (nine at the small sizes, tapering to three at the largest, where a single pure-Python solve takes seconds). The harness asserts status, objective, and iteration-count parity between backends on every problem before recording a number.

<figure>
  <img src="{{ site.baseurl }}/images/blog/simplex-solver-twice/figure-3-backend-scaling.svg"
       alt="Two panels: median solve times diverging on a log-log plot, and per-size boxplots of the paired Rust-over-Python speedups.">
  <figcaption>Left: median solve time on seeded random LPs, log-log, on an Apple M4 Max. Right: the paired Rust-over-Python speedup for the five problems at each size, boxed by size. The 2x at n = 4 grows to 77x by n = 512.</figcaption>
</figure>

At n = 4 the native engine wins by 2.0x, most of which is Python's per-call machinery on both sides of a sub-microsecond computation. The gap widens with size. The paired median is 6.6x at n = 16 and 27x at n = 64, and by n = 512 it reaches 77x: the median solve there takes 3.9 seconds in pure Python and 50 milliseconds in Rust for the identical walk (paired medians, so the ratio differs slightly from dividing those two numbers). The boxes in the right panel stay narrow all the way up. Whatever problem the generator draws, the substrate gap comes out about the same. On the shipped fixtures, which are tiny, the gap is a mundane 1.3x to 4.2x, because the PyO3 boundary crossing is a fixed tax on a microsecond-scale job. For the explainer even the slow line would have been fast enough; a drag re-solve is a three-variable problem. The port's real payoff is elsewhere: the same 714 lines run in the reader's browser tab and behind the Python API, both under the same tested guarantees.


<style>
.btn-soft {
  display: inline-block;
  padding: 0.55em 1.15em;
  border: 1px solid rgba(128, 128, 128, 0.45);
  border-radius: 6px;
  text-decoration: none !important;
  font-weight: 600;
  font-size: 0.85em;
  transition: background 0.2s ease, border-color 0.2s ease;
}
.btn-soft:hover {
  background: rgba(128, 128, 128, 0.12);
  border-color: rgba(128, 128, 128, 0.7);
}
.btn-soft--primary {
  border-color: rgba(52, 120, 246, 0.55);
}
</style>

[Open the explainer]({{ site.baseurl }}/feasible-region/){: .btn-soft .btn-soft--primary}
[Browse the code](https://github.com/nilesh-patil/feasible-region){: .btn-soft}