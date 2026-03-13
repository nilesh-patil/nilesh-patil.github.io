---
title: "Data Science Docker Environment"
collection: portfolio
permalink: /portfolio/datascience-environment/
date: 2018-07-01
excerpt: "Opinionated Dockerfile + compose setup for a reproducible Python/R data-science stack: Jupyter, common ML libraries, and a pre-wired notebook server."
header:
  teaser: /images/blog/feature/nyc_network.png
tags: [docker, devops, jupyter, reproducibility, data-science]
---

## Problem

Bootstrapping a new analyst on a Python + R data-science stack — Jupyter, scikit-learn, pandas, RStudio kernel, common viz libs — eats a day every time. The exact versions that worked last quarter rarely work today.

## Approach

- One Dockerfile that pins Python + R + library versions.
- `docker-compose` wires up a notebook server with a sane volume mount, port mapping, and an opinionated default working directory.
- A small `Makefile` of common operations (build, start, shell, kill) so day-zero is one command.

Used as a teaching environment and as the base image for several internal analytics setups.

## Links

- **Repo**: [github.com/nilesh-patil/datascience-environment](https://github.com/nilesh-patil/datascience-environment) — 18 GitHub stars
