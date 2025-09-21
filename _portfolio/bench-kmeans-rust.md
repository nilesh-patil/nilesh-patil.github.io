---
title: "Python vs Rust: k-means at scale"
collection: portfolio
permalink: /portfolio/bench-kmeans-rust/
date: 2026-03-22
excerpt: "Side-by-side performance comparison of k-means clustering implementations in Python (NumPy / scikit-learn) and Rust, on synthetic benchmark datasets."
repo: https://github.com/nilesh-patil/bench-kmeans-rust
header:
  teaser: /images/blog/bench-kmeans-rust/header.png
  overlay_image: /images/blog/bench-kmeans-rust/header.png
  overlay_filter: "rgba(15, 23, 34, 0.55)"
tags: [rust, python, machine-learning, performance, clustering]
---

## Problem

I wanted to know how much of a "Rust is faster" result survives a fair comparison. K-means is a good test for that: the kernel is small and well defined, and its cost depends on memory layout, vectorization, and parallelism all at once.

## Approach

- **Python side**: hand-rolled NumPy implementation (to control the assignment + update loops directly) plus scikit-learn as the industrial baseline.
- **Rust side**: from-scratch implementation over a `Vec<DataPoint>` array-of-structs layout, with Rayon for optional data-parallel assignment/update. I have not yet tried a flat-matrix, BLAS-backed `ndarray` version, which should help at large *n*.
- **Same algorithm everywhere**: every engine uses the same k-means++ initialization, and scikit-learn runs at `n_init=1`, so seeding cannot explain a difference between engines.
- All engines run on the same synthetic Gaussian-blob datasets across 9 sizes (1k to 256k), 3 feature counts, and 2 cluster budgets, with 3 repeats per cell. That is 162 dataset cells and 648 runs across the four engines.
- **Metric**: end-to-end CLI wall time (load CSV → fit *k* = 1..K → write CSV), sampled peak RSS, and ARI/NMI against ground-truth labels.

## Headline results (2026 rerun)

- Hand-rolled Rust is a **median ~4.5x** faster than pure-Python NumPy and **~11x leaner** in peak memory. The speed gap is widest at small and medium *n*; the memory gap holds at every size.
- scikit-learn is *slower* than NumPy until ~128k and overtakes Rust **only** where both *n* and *k* are large, the corner dominated by one big matrix multiply. Its BLAS wins there; the algorithm is the same.
- Rayon adds at most **~1.3x** over serial Rust, and over-subscribing all cores *hurts* at small *n*.
- With a matched k-means++ init, every engine **essentially ties on accuracy** (median ARI 1.0). The quality gap in my earlier run came from initialization.

## Links

- **Project site**: [K-Means, four ways](https://www.nilesh42.science/bench-kmeans-rust/), the figure-centric write-up comparing Python, serial Rust, parallel Rust, and scikit-learn, with separate [algorithms](https://www.nilesh42.science/bench-kmeans-rust/algorithms/) and [benchmarks](https://www.nilesh42.science/bench-kmeans-rust/benchmarks/) pages.
- **Live demo**: [Race WASM vs JS in-browser](https://www.nilesh42.science/bench-kmeans-rust/demo/). The Rust k-means compiled to WebAssembly, timed against a pure-JS k-means on six point distributions.
- **Repo**: [github.com/nilesh-patil/bench-kmeans-rust](https://github.com/nilesh-patil/bench-kmeans-rust)
- **Blog post**: [parallel Rust k-means - benchmarking a clustering rewrite against NumPy and scikit-learn]({{ '/posts/parallel-rust-kmeans-rewrite/' | relative_url }}), with the full figures and the crossover analysis.
- **Related**: [distributed k-means - clustering datasets that outgrow one machine]({{ '/posts/distributed-kmeans-clustering/' | relative_url }})
