---
title: "SimuCell3D: HPC tissue mechanics in C++"
collection: portfolio
permalink: /portfolio/simucell3d/
date: 2026-03-02
excerpt: "Performance-oriented fork of an ETH-developed C++ framework for simulating 3D tissue mechanics at subcellular resolution — adaptive OpenMP scheduling lifts parallel thread efficiency from 29% to 60%."
tags: [cpp, hpc, openmp, computational-biology, simulation]
---

## Problem

[SimuCell3D](https://www.nature.com/articles/s43588-024-00620-9) simulates 3D tissue mechanics — cell division, polarization, adhesion — at subcellular resolution. The reference implementation is correct but bottlenecked by uneven OpenMP work distribution: most threads sit idle while a few finish long contact-detection chunks.

## Approach

- Profiled the hot loops to localize the imbalance to contact detection and force assembly over heterogeneous cell meshes.
- Replaced the static OpenMP schedule with an adaptive scheme that sizes chunks to the per-cell work estimate.
- Kept the public simulation API unchanged so existing experiment scripts continue to run without modification.

Result: parallel thread efficiency moves from ~29% to ~60% on the standard benchmarks, with no change to numerical output.

## Links

- **Repo**: [github.com/nilesh-patil/simucell3d](https://github.com/nilesh-patil/simucell3d)
- **Reference paper**: Runser et al., *Nature Computational Science* (2024)
