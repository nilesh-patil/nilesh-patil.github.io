---
title: "SimuCell3D: HPC tissue mechanics in C++"
collection: portfolio
permalink: /portfolio/simucell3d/
date: 2026-03-02
excerpt: "Performance-oriented fork of an ETH-developed C++ framework for 3D tissue mechanics at subcellular resolution. On eight cores v2.1.0 grows the benchmark tissue to around 145,000 cells in two days and runs about twice as fast as v1.0 on the cases both versions run."
repo: https://github.com/nilesh-patil/simucell3d
tags: [cpp, hpc, openmp, computational-biology, simulation]
---

## Problem

[SimuCell3D](https://www.nature.com/articles/s43588-024-00620-9) simulates 3D tissue mechanics (cell division, polarization, adhesion) at subcellular resolution. I picked it up to learn HPC on a real scientific code base and to grow bigger tissues with it. On my eight-core runs v1.0 kept about half of the cores busy, and I struggled to get it through its growing-aggregate benchmark: built as tagged it stopped at around 130 cells, and with two small patches at between about 23,000 and 42,000.

## Approach

- Gave the hot loops `schedule(runtime)` and set the schedule per loop group (static for the mesh loops, dynamic for contact detection, guided for polarization and internal forces), with a per-cell work estimate that tracks how uneven the load is.
- Made long runs keep going (a cell whose mesh goes unstable is removed, a division that can't be closed is refused and retried), pinned threads to physical cores, and rebuilt the contact grid.
- A v1.0 command line and its XML parameter tags are still accepted, though parameter files have moved and some options changed in v2.1.

Result: on the growth benchmark, three v2.1.0 runs were still growing at the two-day cap, at **around 145,000 cells**, passing 125,000 in about 38 hours. On the six cases both versions run, v2.1.0 is **about twice as fast** as v1.0 with two patches (between 1.8 and 2.6 times, and an hour and a quarter against two and a half hours to reach 10,000 cells), and it keeps a bit over 80% of the eight cores busy. No run isolates the scheduler from the other changes, so the speedup is all of them together. The growth benchmark runs at a quarter of the paper's time step, because none of my builds stayed stable at the paper's step. Time integration is still serial. Mean cell pressure differs from v1.0 in two cases, and I don't know why.

## Links

- **Repo**: [github.com/nilesh-patil/simucell3d](https://github.com/nilesh-patil/simucell3d)
- **Deep dive**: [OpenMP loop scheduling - a twice-as-fast tissue simulation]({{ '/posts/simucell3d-adaptive-openmp-scheduling/' | relative_url }}) - how the simulator works, what the scheduler reaches, the benchmark across eight builds, and what is still open.
- **Reference paper**: Runser et al., [SimuCell3D](https://www.nature.com/articles/s43588-024-00620-9), *Nature Computational Science* (2024)
