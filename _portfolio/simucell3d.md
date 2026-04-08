---
title: "SimuCell3D: HPC tissue mechanics in C++"
collection: portfolio
permalink: /portfolio/simucell3d/
date: 2026-03-02
excerpt: "Performance-oriented fork of an ETH-developed C++ framework for 3D tissue mechanics at subcellular resolution. In two benchmark runs it ran 1.5x to 2.75x faster than v1.0 at matched cell counts and reached about 2x the cells in the same wall-clock time."
repo: https://github.com/nilesh-patil/simucell3d
tags: [cpp, hpc, openmp, computational-biology, simulation]
---

## Problem

[SimuCell3D](https://www.nature.com/articles/s43588-024-00620-9) simulates 3D tissue mechanics (cell division, polarization, adhesion) at subcellular resolution. On the reference implementation, `top` showed every core busy while a trace showed threads waiting at a barrier for one thread to finish a long contact-detection chunk. I did not measure the idle share per thread.

## Approach

- Profiled the hot loops and found the idle time concentrated in contact detection over heterogeneous cell meshes.
- Replaced the static OpenMP schedule with per-loop schedules (dynamic for contact detection, guided for time integration, static for mesh updates) and a per-cell work estimate that tracks how uneven the load is.
- A v1.0 command line and its XML parameter tags are still accepted, though parameter files have moved and some options changed in v2.1.

Result: **1.5x to 2.75x faster** than v1.0 at matched cell counts across two benchmark runs (a steadier 2.2x to 2.6x in the second), and about 2x the cells in the same wall-clock time. The speedup did not grow with tissue size, and the workload stayed too uniform to test the high-heterogeneity case. The numerical output is not identical to v1.0: the fork's code line includes physics fixes, and mean cell pressure differs by a median of 2 to 3% at matched iterations. The measured gain bundles the new schedules with a contact-grid storage change, and these runs cannot separate the two. In a later single run, a build source-equivalent to v2.1.0 was 3.9x to 7.1x faster than crash-fixed v1.0 at matched cell counts on a different configuration; the physics differs between v1.0 and 2.x, so that is not a like-for-like comparison.

## Links

- **Repo**: [github.com/nilesh-patil/simucell3d](https://github.com/nilesh-patil/simucell3d)
- **Deep dive**: [Adaptive OpenMP scheduling for tissue-mechanics simulations]({{ '/posts/simucell3d-adaptive-openmp-scheduling/' | relative_url }}) - how the adaptive scheduler was built and measured, with the November 2026 corrections.
- **Reference paper**: Runser et al., [SimuCell3D](https://www.nature.com/articles/s43588-024-00620-9), *Nature Computational Science* (2024)
