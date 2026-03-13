---
title: "Python vs Rust: k-means at scale"
collection: portfolio
permalink: /portfolio/pythonvsrust-kmeans/
date: 2024-08-15
excerpt: "Side-by-side performance comparison of k-means clustering implementations in Python (NumPy / scikit-learn) and Rust, on synthetic and real datasets."
header:
  teaser: /images/blog/distributed-kmeans/clusters.jpeg
  overlay_image: /images/blog/distributed-kmeans/clusters.jpeg
  overlay_filter: 0.5
tags: [rust, python, machine-learning, performance, clustering]
---

## Problem

The same algorithm in two languages can have wildly different real-world cost. K-means is a small, well-defined kernel that exercises memory layout, vectorization, and parallelism — a clean target for a head-to-head.

## Approach

- **Python side**: NumPy + scikit-learn baseline, plus a hand-rolled NumPy implementation to control assignment + update loops directly.
- **Rust side**: from-scratch implementation using `ndarray` and Rayon for parallel assignment.
- Both run on the same synthetic and real datasets across a range of *n*, *k*, and dimensionality.
- Measured cold-start, per-iteration cost, and steady-state throughput.

## Links

- **Repo**: [github.com/nilesh-patil/pythonvsrust-kmeans](https://github.com/nilesh-patil/pythonvsrust-kmeans)
- **Related blog post**: [Distributed K-Means Clustering in Python]({{ '/posts/distributed-kmeans-clustering/' | relative_url }})
