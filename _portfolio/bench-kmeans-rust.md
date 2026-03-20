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

The same algorithm in two languages can have wildly different real-world cost. K-means is a small, well-defined kernel that exercises memory layout, vectorization, and parallelism — a clean target for a head-to-head.

## Approach

- **Python side**: NumPy + scikit-learn baseline, plus a hand-rolled NumPy implementation to control assignment + update loops directly.
- **Rust side**: from-scratch implementation over a flat `Vec<DataPoint>` layout, with Rayon for optional parallel assignment (a BLAS-backed `ndarray` rewrite is the obvious next step for large *n*).
- Both run on the same synthetic Gaussian-blob datasets across a range of *n*, *k*, and dimensionality.
- Measured cold-start, per-iteration cost, and steady-state throughput.

## Links

- **Repo**: [github.com/nilesh-patil/bench-kmeans-rust](https://github.com/nilesh-patil/bench-kmeans-rust)
- **Blog post**: [Just rewrite it in Rust? A three-way k-means benchmark]({{ '/posts/just-rewrite-it-in-rust-kmeans/' | relative_url }}) — the full write-up, figures, and the crossover story.
- **Related**: [Distributed K-Means Clustering in Python]({{ '/posts/distributed-kmeans-clustering/' | relative_url }})
