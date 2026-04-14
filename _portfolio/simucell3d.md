---
title: "SimuCell3D: HPC tissue mechanics in C++"
collection: portfolio
permalink: /portfolio/simucell3d/
date: 2026-03-02
excerpt: "Performance-oriented fork of an ETH-developed C++ framework for 3D tissue mechanics at subcellular resolution — adaptive OpenMP scheduling delivers 1.8×–4.4× speedups, growing with tissue size."
tags: [cpp, hpc, openmp, computational-biology, simulation]
---

## Problem

[SimuCell3D](https://www.nature.com/articles/s43588-024-00620-9) simulates 3D tissue mechanics — cell division, polarization, adhesion — at subcellular resolution. The reference implementation is correct but bottlenecked by uneven OpenMP work distribution: most threads sit idle while a few finish long contact-detection chunks.

## Approach

- Profiled the hot loops to localize the imbalance to contact detection and force assembly over heterogeneous cell meshes.
- Replaced the static OpenMP schedule with an adaptive scheme that sizes chunks to the per-cell work estimate.
- Kept the public simulation API unchanged so existing experiment scripts continue to run without modification.

Result: **1.8×–4.4× faster** across the benchmark tissues — the bigger and more heterogeneous the tissue, the larger the win — with no change to numerical output. Per-thread busy time during contact detection tightens from a 20–95% spread to a uniform 90–94%.

## Links

- **Repo**: [github.com/nilesh-patil/simucell3d](https://github.com/nilesh-patil/simucell3d)
- **Deep dive**: [Most threads idle, a few sprinting](/posts/simucell3d-adaptive-openmp-scheduling/) — how the adaptive scheduler was built and measured.
- **Reference paper**: Runser et al., [SimuCell3D](https://www.nature.com/articles/s43588-024-00620-9), *Nature Computational Science* (2024)
