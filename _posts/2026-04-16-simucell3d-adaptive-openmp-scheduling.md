---
layout: single
title: "OpenMP loop scheduling - a twice-as-fast tissue simulation"
date: 2026-04-16T10:00:00+05:30
categories: [blog]
tags: [cpp, hpc, openmp, simucell3d, computational-biology, profiling]
excerpt: "The adaptive scheduler I added to SimuCell3D reaches none of its hot loops. The fork still keeps a bit over 80% of eight cores busy where v1.0 kept half, and runs about twice as fast. This is how the simulator spends its time and what that factor of two includes."
measurement: "a bit over 80% of eight cores busy"
math: true
header:
  overlay_image: /images/blog/headers/simucell3d-adaptive-openmp-scheduling.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/simucell3d-adaptive-openmp-scheduling.jpg
---

<style>
/* Scoped to this post: light hero needs a dark title instead of the theme's white */
.page__hero--overlay .page__title,
.page__hero--overlay .page__lead,
.page__hero--overlay .page__meta,
.page__hero--overlay .page__meta a { color: #13233a !important; text-shadow: none !important; }
</style>

## a tissue as a mechanics problem

Whether a sheet of cells stays flat or rolls up into a tube is a mechanical question as much as a chemical one. Each cell pushes outward with some internal pressure while its surface pulls tight, and it sticks to the cells around it. The tissue settles into whatever shape balances all of that.

[SimuCell3D](https://git.bsse.ethz.ch/iber/Publications/2024_runser_simucell3d) is a simulator for this, published by Steve Runser, Roman Vetter and Dagmar Iber in Nature Computational Science in 2024. The paper's growth demonstration is the line this post keeps coming back to : a tissue grown from a single cell to 125,000 cells in one day of computing, on an eight-core CPU running 16 threads.

I picked the code up to learn HPC on a real scientific code base, and to grow bigger tissues with it. Eventually I'd like to build interactive 3D tissue simulations to play with on an Oculus headset, for fun.

So I [forked it](https://github.com/nilesh-patil/simucell3d). The fork's current release, v2.1.0, comes out roughly twice as fast as a lightly patched v1.0 on a benchmark of about 240 runs. Most of this post is the background you need to see where that factor comes from.

## a cell is a closed triangle mesh

A SimuCell3D cell is all surface : a closed mesh of triangles with nothing meshed inside it. The corner points are nodes, an edge joins two nodes, and each triangle is a face.

<figure>
<svg viewBox="0 0 720 380" role="img" aria-labelledby="fig-mesh-t fig-mesh-d" style="width:100%;height:auto;max-width:720px;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title id="fig-mesh-t">One cell as a closed surface of triangles</title>
<desc id="fig-mesh-d">A rounded cell drawn as a closed mesh of triangles, with one node, one edge and one face labelled. A second, fainter cell is pressed flat against its right side.</desc>
<g opacity="0.5">
<path d="M453.2,159.2 L504.4,109.7 L518.2,169 z M548.2,87.4 L504.4,109.7 L487.1,73.8 z M453.2,159.2 L436.8,116.6 L504.4,109.7 z M431.6,76.4 L487.1,73.8 L436.8,116.6 z M504.4,109.7 L436.8,116.6 L487.1,73.8 z M431.6,76.4 L487.7,58.8 L487.1,73.8 z M453.2,159.2 L420,187.4 L436.8,116.6 z" fill="currentColor" fill-opacity="0.03" stroke="none"></path>
<path d="M453.2,159.2 L469.5,225.1 L420,187.4 z M431.6,76.4 L420,137.4 L420.1,108.1 z M548.2,87.4 L571.9,137.3 L504.4,109.7 z M571.9,137.3 L518.2,169 L504.4,109.7 z M548.2,87.4 L487.1,73.8 L538.4,58.3 z M487.1,73.8 L487.7,58.8 L538.4,58.3 z M453.2,159.2 L518.2,169 L469.5,225.1 z M431.6,76.4 L436.8,116.6 L420,137.4 z M436.8,116.6 L420,187.4 L420,137.4 z" fill="currentColor" fill-opacity="0.052" stroke="none"></path>
<path d="M420,196 L432.1,253.1 L420,236.7 z M432.1,253.1 L420,187.4 L469.5,225.1 z M548.2,87.4 L538.4,58.3 L600.3,106.7 z M548.2,87.4 L600.3,106.7 L571.9,137.3 z" fill="currentColor" fill-opacity="0.075" stroke="none"></path>
<path d="M420,196 L420,187.4 L432.1,253.1 z M578.7,201.8 L518.2,169 L571.9,137.3 z M600.3,106.7 L615,175.3 L571.9,137.3 z M578.7,201.8 L539.2,249.4 L518.2,169 z M518.2,169 L539.2,249.4 L469.5,225.1 z" fill="currentColor" fill-opacity="0.098" stroke="none"></path>
<path d="M491.8,267.3 L474.7,275.5 L432.1,253.1 z M491.8,267.3 L540.5,276.7 L474.7,275.5 z M576.1,256.6 L540.5,276.7 L587.3,247.6 z M491.8,267.3 L539.2,249.4 L540.5,276.7 z M578.7,201.8 L587.3,247.6 L539.2,249.4 z M540.5,276.7 L539.2,249.4 L587.3,247.6 z M491.8,267.3 L432.1,253.1 L469.5,225.1 z M616.1,140.9 L615,175.3 L600.3,106.7 z M578.7,201.8 L571.9,137.3 L615,175.3 z M491.8,267.3 L469.5,225.1 L539.2,249.4 z M578.7,201.8 L615,175.3 L587.3,247.6 z M615,175.3 L613,204.6 L587.3,247.6 z" fill="currentColor" fill-opacity="0.12" stroke="none"></path>
<path d="M431.6,76.4 L420,137.4 M431.6,76.4 L436.8,116.6 M431.6,76.4 L487.1,73.8 M548.2,87.4 L571.9,137.3 M548.2,87.4 L504.4,109.7 M548.2,87.4 L487.1,73.8 M548.2,87.4 L538.4,58.3 M548.2,87.4 L600.3,106.7 M453.2,159.2 L420,187.4 M453.2,159.2 L469.5,225.1 M453.2,159.2 L518.2,169 M453.2,159.2 L504.4,109.7 M453.2,159.2 L436.8,116.6 M491.8,267.3 L432.1,253.1 M491.8,267.3 L474.7,275.5 M491.8,267.3 L540.5,276.7 M491.8,267.3 L539.2,249.4 M491.8,267.3 L469.5,225.1 M578.7,201.8 L539.2,249.4 M578.7,201.8 L587.3,247.6 M578.7,201.8 L571.9,137.3 M578.7,201.8 L518.2,169 M578.7,201.8 L615,175.3 M420,196 L432.1,253.1 M432.1,253.1 L420,187.4 M432.1,253.1 L469.5,225.1 M540.5,276.7 L539.2,249.4 M540.5,276.7 L587.3,247.6 M539.2,249.4 L587.3,247.6 M539.2,249.4 L469.5,225.1 M539.2,249.4 L518.2,169 M587.3,247.6 L615,175.3 M420,187.4 L469.5,225.1 M420,187.4 L436.8,116.6 M469.5,225.1 L518.2,169 M420,137.4 L436.8,116.6 M571.9,137.3 L518.2,169 M571.9,137.3 L504.4,109.7 M571.9,137.3 L600.3,106.7 M571.9,137.3 L615,175.3 M518.2,169 L504.4,109.7 M504.4,109.7 L436.8,116.6 M504.4,109.7 L487.1,73.8 M436.8,116.6 L487.1,73.8 M487.1,73.8 L487.7,58.8 M487.1,73.8 L538.4,58.3 M600.3,106.7 L615,175.3" stroke="currentColor" stroke-width="1.2" fill="none" opacity="0.85" stroke-linecap="round"></path>
<path d="M576.1,256.6 L540.5,276.7 M576.1,256.6 L587.3,247.6 M431.6,76.4 L420.1,108.1 M431.6,76.4 L487.7,58.8 M616.1,140.9 L600.3,106.7 M616.1,140.9 L615,175.3 M420,196 L420,236.7 M420,196 L420,187.4 M420,236.7 L432.1,253.1 M432.1,253.1 L474.7,275.5 M474.7,275.5 L540.5,276.7 M587.3,247.6 L613,204.6 M420,187.4 L420,137.4 M420.1,108.1 L420,137.4 M613,204.6 L615,175.3 M487.7,58.8 L538.4,58.3 M538.4,58.3 L600.3,106.7" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round"></path>
<g fill="currentColor" stroke="none">
<circle cx="576.1" cy="256.6" r="2.8"></circle>
<circle cx="431.6" cy="76.4" r="2.8"></circle>
<circle cx="548.2" cy="87.4" r="2.8"></circle>
<circle cx="453.2" cy="159.2" r="2.8"></circle>
<circle cx="491.8" cy="267.3" r="2.8"></circle>
<circle cx="616.1" cy="140.9" r="2.8"></circle>
<circle cx="578.7" cy="201.8" r="2.8"></circle>
<circle cx="420" cy="196" r="2.8"></circle>
<circle cx="420" cy="236.7" r="2.8"></circle>
<circle cx="432.1" cy="253.1" r="2.8"></circle>
<circle cx="474.7" cy="275.5" r="2.8"></circle>
<circle cx="540.5" cy="276.7" r="2.8"></circle>
<circle cx="539.2" cy="249.4" r="2.8"></circle>
<circle cx="587.3" cy="247.6" r="2.8"></circle>
<circle cx="420" cy="187.4" r="2.8"></circle>
<circle cx="469.5" cy="225.1" r="2.8"></circle>
<circle cx="420.1" cy="108.1" r="2.8"></circle>
<circle cx="420" cy="137.4" r="2.8"></circle>
<circle cx="613" cy="204.6" r="2.8"></circle>
<circle cx="571.9" cy="137.3" r="2.8"></circle>
<circle cx="518.2" cy="169" r="2.8"></circle>
<circle cx="504.4" cy="109.7" r="2.8"></circle>
<circle cx="436.8" cy="116.6" r="2.8"></circle>
<circle cx="487.1" cy="73.8" r="2.8"></circle>
<circle cx="487.7" cy="58.8" r="2.8"></circle>
<circle cx="538.4" cy="58.3" r="2.8"></circle>
<circle cx="600.3" cy="106.7" r="2.8"></circle>
<circle cx="615" cy="175.3" r="2.8"></circle>
</g>
</g>
<path d="M278.3,138.7 L300.4,79.4 L358.8,131.3 z M311.7,45.6 L300.4,79.4 L238.8,54.1 z M278.3,138.7 L219.5,104.9 L300.4,79.4 z M178,112.4 L238.8,54.1 L219.5,104.9 z M300.4,79.4 L219.5,104.9 L238.8,54.1 z M178,112.4 L194.5,84.4 L238.8,54.1 z M178,112.4 L219.5,104.9 L176,179.5 z" fill="currentColor" fill-opacity="0.03" stroke="none"></path>
<path d="M178,112.4 L176,179.5 L152.1,146.3 z M311.7,45.6 L355.8,39.4 L377.1,68.3 z M311.7,45.6 L377.1,68.3 L300.4,79.4 z M377.1,68.3 L358.8,131.3 L300.4,79.4 z M311.7,45.6 L238.8,54.1 L276.1,43 z M311.7,45.6 L276.1,43 L355.8,39.4 z M278.3,138.7 L232.8,196.2 L219.5,104.9 z M200.6,237.2 L176,179.5 L232.8,196.2 z M219.5,104.9 L232.8,196.2 L176,179.5 z M178,112.4 L152.1,146.3 L194.5,84.4 z" fill="currentColor" fill-opacity="0.052" stroke="none"></path>
<path d="M200.6,237.2 L232.8,196.2 L272.9,254.2 z M278.3,138.7 L318.2,199.9 L232.8,196.2 z M272.9,254.2 L232.8,196.2 L318.2,199.9 z M412,123.3 L358.8,131.3 L377.1,68.3 z M278.3,138.7 L358.8,131.3 L318.2,199.9 z" fill="currentColor" fill-opacity="0.075" stroke="none"></path>
<path d="M200.6,237.2 L176.3,230.2 L176,179.5 z M176.3,230.2 L152.1,146.3 L176,179.5 z M412,123.3 L399,194.9 L358.8,131.3 z M358.8,131.3 L399,194.9 L318.2,199.9 z" fill="currentColor" fill-opacity="0.098" stroke="none"></path>
<path d="M343.2,256.6 L398,250.5 L319.9,293.3 z M319.9,293.3 L398,250.5 L367.5,275.3 z M200.6,237.2 L239.2,279.5 L176.3,230.2 z M288.6,289.8 L239.2,279.5 L319.9,293.3 z M200.6,237.2 L272.9,254.2 L239.2,279.5 z M343.2,256.6 L319.9,293.3 L272.9,254.2 z M239.2,279.5 L272.9,254.2 L319.9,293.3 z M412,223.2 L398,250.5 L412,175.4 z M343.2,256.6 L399,194.9 L398,250.5 z M412,123.3 L412,175.4 L399,194.9 z M398,250.5 L399,194.9 L412,175.4 z M343.2,256.6 L272.9,254.2 L318.2,199.9 z M412,123.3 L377.1,68.3 L412,108.2 z M355.8,39.4 L412,108.2 L377.1,68.3 z M343.2,256.6 L318.2,199.9 L399,194.9 z" fill="currentColor" fill-opacity="0.12" stroke="none"></path>
<path d="M288.6,289.8 L367.5,275.3 M288.6,289.8 L308.7,267 M288.6,289.8 L225,267.2 M309.8,200.9 L375.3,227.7 M309.8,200.9 L308.7,267 M309.8,200.9 L242.9,213.5 M309.8,200.9 L353.7,147.3 M309.8,200.9 L279.1,138.7 M412,223.2 L367.5,275.3 M412,223.2 L375.3,227.7 M412,223.2 L412,160.9 M180.8,199.5 L242.9,213.5 M180.8,199.5 L225,267.2 M180.8,199.5 L176.3,230.2 M180.8,199.5 L152.1,146.3 M180.8,199.5 L206.2,132.7 M236.7,83.8 L279.1,138.7 M236.7,83.8 L206.2,132.7 M236.7,83.8 L194.5,84.4 M236.7,83.8 L276.1,43 M236.7,83.8 L329,67.1 M394.2,98.6 L412,160.9 M394.2,98.6 L353.7,147.3 M394.2,98.6 L355.8,39.4 M394.2,98.6 L412,108.2 M394.2,98.6 L329,67.1 M367.5,275.3 L375.3,227.7 M367.5,275.3 L308.7,267 M375.3,227.7 L308.7,267 M375.3,227.7 L412,160.9 M375.3,227.7 L353.7,147.3 M308.7,267 L242.9,213.5 M308.7,267 L225,267.2 M242.9,213.5 L225,267.2 M242.9,213.5 L279.1,138.7 M242.9,213.5 L206.2,132.7 M225,267.2 L176.3,230.2 M225,267.2 L239.2,279.5 M412,160.9 L353.7,147.3 M412,160.9 L412,175.4 M412,160.9 L412,108.2 M353.7,147.3 L279.1,138.7 M353.7,147.3 L329,67.1 M412,175.4 L412,108.2 M152.1,146.3 L206.2,132.7 M279.1,138.7 L206.2,132.7 M279.1,138.7 L329,67.1 M206.2,132.7 L194.5,84.4 M355.8,39.4 L329,67.1 M194.5,84.4 L276.1,43 M276.1,43 L329,67.1" stroke="currentColor" stroke-width="0.8" fill="none" opacity="0.2" stroke-linecap="round"></path>
<path d="M278.3,138.7 L232.8,196.2 M278.3,138.7 L318.2,199.9 M278.3,138.7 L358.8,131.3 M278.3,138.7 L300.4,79.4 M278.3,138.7 L219.5,104.9 M311.7,45.6 L355.8,39.4 M311.7,45.6 L377.1,68.3 M311.7,45.6 L300.4,79.4 M311.7,45.6 L238.8,54.1 M311.7,45.6 L276.1,43 M412,123.3 L399,194.9 M412,123.3 L377.1,68.3 M412,123.3 L358.8,131.3 M178,112.4 L152.1,146.3 M178,112.4 L176,179.5 M178,112.4 L219.5,104.9 M178,112.4 L238.8,54.1 M178,112.4 L194.5,84.4 M200.6,237.2 L176.3,230.2 M200.6,237.2 L239.2,279.5 M200.6,237.2 L272.9,254.2 M200.6,237.2 L232.8,196.2 M200.6,237.2 L176,179.5 M343.2,256.6 L319.9,293.3 M343.2,256.6 L398,250.5 M343.2,256.6 L272.9,254.2 M343.2,256.6 L399,194.9 M343.2,256.6 L318.2,199.9 M319.9,293.3 L398,250.5 M319.9,293.3 L239.2,279.5 M319.9,293.3 L272.9,254.2 M398,250.5 L399,194.9 M398,250.5 L412,175.4 M176.3,230.2 L176,179.5 M239.2,279.5 L272.9,254.2 M272.9,254.2 L232.8,196.2 M272.9,254.2 L318.2,199.9 M399,194.9 L412,175.4 M399,194.9 L318.2,199.9 M399,194.9 L358.8,131.3 M232.8,196.2 L318.2,199.9 M232.8,196.2 L176,179.5 M232.8,196.2 L219.5,104.9 M318.2,199.9 L358.8,131.3 M152.1,146.3 L176,179.5 M176,179.5 L219.5,104.9 M355.8,39.4 L377.1,68.3 M412,108.2 L377.1,68.3 M377.1,68.3 L358.8,131.3 M377.1,68.3 L300.4,79.4 M358.8,131.3 L300.4,79.4 M300.4,79.4 L219.5,104.9 M300.4,79.4 L238.8,54.1 M219.5,104.9 L238.8,54.1" stroke="currentColor" stroke-width="1.2" fill="none" opacity="0.85" stroke-linecap="round"></path>
<path d="M288.6,289.8 L319.9,293.3 M288.6,289.8 L239.2,279.5 M412,123.3 L412,175.4 M412,123.3 L412,108.2 M412,223.2 L398,250.5 M412,223.2 L412,175.4 M319.9,293.3 L367.5,275.3 M398,250.5 L367.5,275.3 M176.3,230.2 L239.2,279.5 M176.3,230.2 L152.1,146.3 M152.1,146.3 L194.5,84.4 M355.8,39.4 L412,108.2 M355.8,39.4 L276.1,43 M238.8,54.1 L194.5,84.4 M238.8,54.1 L276.1,43" stroke="currentColor" stroke-width="1.9" fill="none" stroke-linecap="round"></path>
<g fill="currentColor" stroke="none">
<circle cx="288.6" cy="289.8" r="3.4"></circle>
<circle cx="278.3" cy="138.7" r="3.4"></circle>
<circle cx="311.7" cy="45.6" r="3.4"></circle>
<circle cx="412" cy="123.3" r="3.4"></circle>
<circle cx="412" cy="223.2" r="3.4"></circle>
<circle cx="178" cy="112.4" r="3.4"></circle>
<circle cx="200.6" cy="237.2" r="3.4"></circle>
<circle cx="343.2" cy="256.6" r="3.4"></circle>
<circle cx="319.9" cy="293.3" r="3.4"></circle>
<circle cx="398" cy="250.5" r="3.4"></circle>
<circle cx="367.5" cy="275.3" r="3.4"></circle>
<circle cx="176.3" cy="230.2" r="3.4"></circle>
<circle cx="239.2" cy="279.5" r="3.4"></circle>
<circle cx="272.9" cy="254.2" r="3.4"></circle>
<circle cx="399" cy="194.9" r="3.4"></circle>
<circle cx="412" cy="175.4" r="3.4"></circle>
<circle cx="232.8" cy="196.2" r="3.4"></circle>
<circle cx="318.2" cy="199.9" r="3.4"></circle>
<circle cx="152.1" cy="146.3" r="3.4"></circle>
<circle cx="176" cy="179.5" r="3.4"></circle>
<circle cx="355.8" cy="39.4" r="3.4"></circle>
<circle cx="412" cy="108.2" r="3.4"></circle>
<circle cx="377.1" cy="68.3" r="3.4"></circle>
<circle cx="358.8" cy="131.3" r="3.4"></circle>
<circle cx="300.4" cy="79.4" r="3.4"></circle>
<circle cx="219.5" cy="104.9" r="3.4"></circle>
<circle cx="238.8" cy="54.1" r="3.4"></circle>
<circle cx="194.5" cy="84.4" r="3.4"></circle>
<circle cx="276.1" cy="43" r="3.4"></circle>
</g>
<path d="M200.6,237.2 L272.9,254.2 L239.2,279.5 z" fill="currentColor" fill-opacity="0.24" stroke="currentColor" stroke-width="1.5"></path>
<path d="M232.8,196.2 L176,179.5" stroke="currentColor" stroke-width="3.4" fill="none" stroke-linecap="round"></path>
<circle cx="238.8" cy="54.1" r="5.8" fill="currentColor" stroke="none"></circle>
<path d="M82,54 L230.8,52.6" stroke="currentColor" stroke-width="1" fill="none" opacity="0.6"></path>
<path d="M82,174 L198.4,187.9" stroke="currentColor" stroke-width="1" fill="none" opacity="0.6"></path>
<path d="M82,294 L237.6,256.9" stroke="currentColor" stroke-width="1" fill="none" opacity="0.6"></path>
<circle cx="237.6" cy="256.9" r="2.2" fill="currentColor" stroke="none" opacity="0.75"></circle>
<path d="M416,34 L416,131.2" stroke="currentColor" stroke-width="1" fill="none" opacity="0.6"></path>
<circle cx="416" cy="134.2" r="3" fill="currentColor" stroke="none"></circle>
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="28" y="60" font-size="17" font-weight="600">node</text>
<text x="28" y="180" font-size="17" font-weight="600">edge</text>
<text x="28" y="300" font-size="17" font-weight="600">face</text>
<text x="416" y="26" font-size="16" text-anchor="middle">contact</text>
<text x="590" y="40" font-size="14" text-anchor="middle" opacity="0.78">a second cell</text>
<text x="300" y="342" font-size="16" text-anchor="middle" font-weight="600">one cell: a closed surface of triangles</text>
<text x="300" y="364" font-size="14" text-anchor="middle" opacity="0.78">in the benchmark: about 120 nodes and 240 triangles</text>
</g>
</svg>
<figcaption>A cell, with a second one pressed against it.</figcaption>
</figure>

A cell in the growth benchmark has about 120 nodes and 240 triangles. The count isn't fixed : the simulator is given a minimum edge length, and a cell gets as many triangles as it takes to cover it at that size.

Every node carries a position, a momentum and a share of its cell's mass, and the rest of the simulator is there to work out the force on it and move it.

## the forces on one node

The paper writes the mechanics of one cell as an energy. It's the only equation in this post :

$$
U = K V \left( \ln\frac{V}{V_0} - 1 \right) + \frac{k_a}{2} \left( \frac{A}{A_0} - 1 \right)^2 + \int_{\partial\Omega} \left( \gamma + \frac{k_b}{2} (2H)^2 \right) dS
$$

Each term stores energy in one aspect of the cell's shape, and the force on a node points whichever way lowers the total fastest. The four terms, left to right, are panels 1 to 4 of the drawing.

<figure>
<svg viewBox="0 0 720 236" role="img" aria-labelledby="fig-forces-t fig-forces-d" style="width:100%;height:auto;max-width:720px;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title id="fig-forces-t">Five forces on the nodes of a cell</title>
<desc id="fig-forces-d">Five numbered panels in a row, each a cut through a cell outline with arrows: 1 pressure pushing every face outward, 2 area elasticity as springs in the surface, 3 surface tension pulling along the surface, 4 bending flattening a sharp fold, and 5 two cells that attract when close and repel when they overlap.</desc>
<path d="M108.2,89.4 L90.4,114.6 L63.1,123.3 L29.3,108.2 L21.6,78.5 L30.6,50.8 L54.4,35.8 L86.3,34.2 L110.7,61.6 z" stroke="currentColor" stroke-width="1.1" fill="none" stroke-dasharray="4 4" opacity="0.55" stroke-linejoin="round"></path>
<path d="M90.8,84.7 L80.3,99.5 L64.3,104.6 L44.4,95.8 L39.9,78.3 L45.1,62 L59.2,53.2 L78,52.3 L92.3,68.3 z" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06" stroke-linejoin="round"></path>
<path d="M88.4,94.1 L92.9,97.3" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M97.4,100.5 L91.1,99.1 L94,95 z" fill="currentColor" stroke="none"></path>
<path d="M73.4,105.4 L75,110.7" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M76.7,115.9 L72.5,111 L77.3,109.4 z" fill="currentColor" stroke="none"></path>
<path d="M52.9,103.4 L50.7,108.4" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M48.4,113.4 L48.6,106.9 L53.2,109 z" fill="currentColor" stroke="none"></path>
<path d="M38.8,87.9 L33.4,89.3" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M28.1,90.7 L33.3,86.7 L34.6,91.6 z" fill="currentColor" stroke="none"></path>
<path d="M39.2,69.1 L34,67.4" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M28.7,65.7 L35.2,65.2 L33.7,70 z" fill="currentColor" stroke="none"></path>
<path d="M50.3,54.6 L47.4,50" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M44.4,45.3 L49.8,49.1 L45.5,51.8 z" fill="currentColor" stroke="none"></path>
<path d="M68.4,49.2 L68.1,43.7" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M67.8,38.2 L70.6,44.1 L65.6,44.4 z" fill="currentColor" stroke="none"></path>
<path d="M87.7,58 L91.8,54.3" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M95.9,50.7 L93.1,56.5 L89.8,52.8 z" fill="currentColor" stroke="none"></path>
<path d="M95,76.8 L100.5,77.3" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M106,77.8 L99.8,79.8 L100.2,74.8 z" fill="currentColor" stroke="none"></path>
<circle cx="90.8" cy="84.7" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="80.3" cy="99.5" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="64.3" cy="104.6" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="44.4" cy="95.8" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="39.9" cy="78.3" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="45.1" cy="62" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="59.2" cy="53.2" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="78" cy="52.3" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="92.3" cy="68.3" r="2.4" fill="currentColor" stroke="none"></circle>
<path d="M222.7,86 L208.1,112.9 L176.9,119.7 L147.1,98.1 L140.4,69.2 L156.6,45.2 L191.6,37.3 L216.6,56.8 z" fill="currentColor" fill-opacity="0.06" stroke="none"></path>
<path d="M202.3,82 L195,95.4 L179.4,98.8 L164.6,88 L161.2,73.6 L169.3,61.6 L186.8,57.7 L199.3,67.4 z" stroke="currentColor" stroke-width="1.1" fill="none" stroke-dasharray="4 4" opacity="0.55" stroke-linejoin="round"></path>
<path d="M222.7,86 L219.8,91.4 L215.9,91 L220.7,97.1 L213,96.4 L217.8,102.5 L210,101.7 L214.9,107.9 L211,107.5 L208.1,112.9" stroke="currentColor" stroke-width="1.3" fill="none" stroke-linejoin="round"></path>
<path d="M217.8,85 L213.4,84.2" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M208.5,83.2 L214.3,82 L213.4,86.5 z" fill="currentColor" stroke="none"></path>
<path d="M208.1,112.9 L201.8,114.2 L199.5,111.1 L197.9,118.8 L193.3,112.4 L191.7,120.1 L187,113.8 L185.4,121.5 L183.1,118.3 L176.9,119.7" stroke="currentColor" stroke-width="1.3" fill="none" stroke-linejoin="round"></path>
<path d="M176.9,119.7 L170.9,115.4 L171.6,111.4 L164.3,115 L165.6,107.1 L158.4,110.7 L159.7,102.7 L152.5,106.4 L153.1,102.4 L147.1,98.1" stroke="currentColor" stroke-width="1.3" fill="none" stroke-linejoin="round"></path>
<path d="M177.5,114.7 L178,110.3" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M178.6,105.3 L180.3,111 L175.7,110.5 z" fill="currentColor" stroke="none"></path>
<path d="M147.1,98.1 L145.8,92.3 L149,90.1 L141.3,88.8 L147.6,84.3 L139.9,83 L146.3,78.5 L138.6,77.2 L141.8,75 L140.4,69.2" stroke="currentColor" stroke-width="1.3" fill="none" stroke-linejoin="round"></path>
<path d="M140.4,69.2 L143.7,64.4 L147.5,65.2 L143.1,58.8 L150.7,60.4 L146.4,54 L154,55.6 L149.6,49.2 L153.4,50 L156.6,45.2" stroke="currentColor" stroke-width="1.3" fill="none" stroke-linejoin="round"></path>
<path d="M145.3,70.2 L149.7,71.2" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M154.6,72.2 L148.8,73.3 L149.7,68.8 z" fill="currentColor" stroke="none"></path>
<path d="M156.6,45.2 L163.6,43.6 L166.2,46.7 L168.1,38.9 L173.2,45.1 L175.1,37.3 L180.2,43.6 L182.1,35.8 L184.6,38.9 L191.6,37.3" stroke="currentColor" stroke-width="1.3" fill="none" stroke-linejoin="round"></path>
<path d="M191.6,37.3 L196.6,41.2 L195.6,45 L202.6,41.3 L200.6,48.9 L207.6,45.2 L205.6,52.8 L212.5,49.1 L211.6,52.9 L216.6,56.8" stroke="currentColor" stroke-width="1.3" fill="none" stroke-linejoin="round"></path>
<path d="M190.5,42.2 L189.4,46.6" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M188.3,51.4 L187.3,45.5 L191.8,46.6 z" fill="currentColor" stroke="none"></path>
<path d="M216.6,56.8 L217.8,62.6 L214.6,64.8 L222.2,66.3 L215.8,70.7 L223.5,72.1 L217,76.5 L224.7,78 L221.5,80.2 L222.7,86" stroke="currentColor" stroke-width="1.3" fill="none" stroke-linejoin="round"></path>
<circle cx="222.7" cy="86" r="2.6" fill="currentColor" stroke="none"></circle>
<circle cx="208.1" cy="112.9" r="2.6" fill="currentColor" stroke="none"></circle>
<circle cx="176.9" cy="119.7" r="2.6" fill="currentColor" stroke="none"></circle>
<circle cx="147.1" cy="98.1" r="2.6" fill="currentColor" stroke="none"></circle>
<circle cx="140.4" cy="69.2" r="2.6" fill="currentColor" stroke="none"></circle>
<circle cx="156.6" cy="45.2" r="2.6" fill="currentColor" stroke="none"></circle>
<circle cx="191.6" cy="37.3" r="2.6" fill="currentColor" stroke="none"></circle>
<circle cx="216.6" cy="56.8" r="2.6" fill="currentColor" stroke="none"></circle>
<path d="M333.7,93 L309.7,114.1 L282.6,113.9 L261.5,88.8 L264.7,61.9 L282.9,43.7 L314.3,45.5 L333.4,67.5 z" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06" stroke-linejoin="round"></path>
<path d="M336.8,100.2 L332.1,104.4" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M328.3,107.7 L330.9,102.3 L334,105.8 z" fill="currentColor" stroke="none"></path>
<path d="M316.5,118.1 L321.3,113.9" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M325,110.6 L322.4,116 L319.4,112.5 z" fill="currentColor" stroke="none"></path>
<path d="M308.2,121.6 L303.3,121.5" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M298.3,121.5 L303.8,119.2 L303.8,123.8 z" fill="currentColor" stroke="none"></path>
<path d="M284,121.4 L288.9,121.5" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M293.9,121.5 L288.4,123.8 L288.4,119.2 z" fill="currentColor" stroke="none"></path>
<path d="M275,116.5 L270.9,111.7" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M267.7,107.9 L273,110.6 L269.5,113.6 z" fill="currentColor" stroke="none"></path>
<path d="M257.6,95.8 L261.7,100.6" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M264.9,104.5 L259.6,101.7 L263.1,98.8 z" fill="currentColor" stroke="none"></path>
<path d="M254.2,86.4 L254.8,81.6" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M255.4,76.6 L257,82.4 L252.5,81.8 z" fill="currentColor" stroke="none"></path>
<path d="M257.1,62.5 L256.5,67.3" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M255.9,72.3 L254.3,66.5 L258.9,67.1 z" fill="currentColor" stroke="none"></path>
<path d="M260.5,55.5 L263.4,52.6" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M267,49 L264.7,54.5 L261.4,51.3 z" fill="currentColor" stroke="none"></path>
<path d="M276.6,39.4 L273.6,42.4" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M270.1,45.9 L272.3,40.4 L275.6,43.7 z" fill="currentColor" stroke="none"></path>
<path d="M285.6,36.3 L291.8,36.7" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M296.8,36.9 L291.2,38.9 L291.5,34.3 z" fill="currentColor" stroke="none"></path>
<path d="M312.5,37.9 L306.2,37.5" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M301.2,37.2 L306.8,35.2 L306.6,39.8 z" fill="currentColor" stroke="none"></path>
<path d="M320.9,41.7 L324.8,46.1" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M328.1,49.9 L322.7,47.2 L326.2,44.2 z" fill="currentColor" stroke="none"></path>
<path d="M338.1,61.4 L334.2,57" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M330.9,53.2 L336.3,55.8 L332.8,58.8 z" fill="currentColor" stroke="none"></path>
<path d="M341,68.9 L341,72.9" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M341.1,77.9 L338.7,72.5 L343.3,72.4 z" fill="currentColor" stroke="none"></path>
<path d="M341.2,91.4 L341.1,87.3" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M341.1,82.3 L343.5,87.8 L338.8,87.9 z" fill="currentColor" stroke="none"></path>
<circle cx="333.7" cy="93" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="309.7" cy="114.1" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="282.6" cy="113.9" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="261.5" cy="88.8" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="264.7" cy="61.9" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="282.9" cy="43.7" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="314.3" cy="45.5" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="333.4" cy="67.5" r="2.4" fill="currentColor" stroke="none"></circle>
<path d="M415.5,30.8 L435.6,62.9 L446.3,83.2 L443.5,105.9 L425.4,121.5 L403,119.7 L384.9,104.4 L381.2,82.7 L392,60.7 z" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06" stroke-linejoin="round"></path>
<path d="M392,60.7 L435.6,62.9" stroke="currentColor" stroke-width="1.1" fill="none" stroke-dasharray="4 4" opacity="0.55"></path>
<path d="M392,60.7 L415.5,30.8 L435.6,62.9" stroke="currentColor" stroke-width="2.4" fill="none" stroke-linejoin="round" stroke-linecap="round"></path>
<circle cx="415.5" cy="30.8" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="435.6" cy="62.9" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="446.3" cy="83.2" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="443.5" cy="105.9" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="425.4" cy="121.5" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="403" cy="119.7" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="384.9" cy="104.4" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="381.2" cy="82.7" r="2.4" fill="currentColor" stroke="none"></circle>
<circle cx="392" cy="60.7" r="2.4" fill="currentColor" stroke="none"></circle>
<path d="M427.5,26.8 L427.5,38.3" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M427.5,43.8 L425,37.8 L430,37.8 z" fill="currentColor" stroke="none"></path>
<path d="M403.5,26.8 L403.5,38.3" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M403.5,43.8 L401,37.8 L406,37.8 z" fill="currentColor" stroke="none"></path>
<path d="M387,57.7 L387,49.7" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M387,44.7 L389.3,50.2 L384.7,50.2 z" fill="currentColor" stroke="none"></path>
<path d="M440.6,59.9 L440.6,51.9" stroke="currentColor" stroke-width="1.3" fill="none"></path>
<path d="M440.6,46.9 L442.9,52.4 L438.3,52.4 z" fill="currentColor" stroke="none"></path>
<path d="M524.5,80 L523.2,99 L519.2,106.2 L511.6,109.6 L491.4,109.6 L483.8,106.2 L479.8,99 L478.5,80 L479.8,61 L483.8,53.8 L491.4,50.4 L511.6,50.4 L519.2,53.8 L523.2,61 z" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06" stroke-linejoin="round"></path>
<path d="M589.2,93.1 L586.6,103.1 L581,108.3 L566.5,110 L552,108.3 L546.4,103.1 L543.8,93.1 L543.8,66.9 L546.4,56.9 L552,51.7 L566.5,50 L581,51.7 L586.6,56.9 L589.2,66.9 z" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06" stroke-linejoin="round"></path>
<circle cx="524.5" cy="80" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="523.2" cy="99" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="519.2" cy="106.2" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="511.6" cy="109.6" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="491.4" cy="109.6" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="483.8" cy="106.2" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="479.8" cy="99" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="478.5" cy="80" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="479.8" cy="61" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="483.8" cy="53.8" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="491.4" cy="50.4" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="511.6" cy="50.4" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="519.2" cy="53.8" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="523.2" cy="61" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="589.2" cy="93.1" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="586.6" cy="103.1" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="581" cy="108.3" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="566.5" cy="110" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="552" cy="108.3" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="546.4" cy="103.1" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="543.8" cy="93.1" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="543.8" cy="66.9" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="546.4" cy="56.9" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="552" cy="51.7" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="566.5" cy="50" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="581" cy="51.7" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="586.6" cy="56.9" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="589.2" cy="66.9" r="2.2" fill="currentColor" stroke="none"></circle>
<path d="M527.5,80 L531,80" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M535.5,80 L530.5,82.1 L530.5,77.9 z" fill="currentColor" stroke="none"></path>
<path d="M526.2,99 L529.7,99" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M534.2,99 L529.2,101.1 L529.2,96.9 z" fill="currentColor" stroke="none"></path>
<path d="M526.2,61 L529.7,61" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M534.2,61 L529.2,63.1 L529.2,58.9 z" fill="currentColor" stroke="none"></path>
<path d="M540.8,93.1 L537.3,93.1" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M532.8,93.1 L537.8,91 L537.8,95.2 z" fill="currentColor" stroke="none"></path>
<path d="M540.8,66.9 L537.3,66.9" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M532.8,66.9 L537.8,64.8 L537.8,69 z" fill="currentColor" stroke="none"></path>
<path d="M543.4,103.1 L539.9,103.1" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M535.4,103.1 L540.4,101 L540.4,105.2 z" fill="currentColor" stroke="none"></path>
<path d="M543.4,56.9 L539.9,56.9" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<path d="M535.4,56.9 L540.4,54.8 L540.4,59 z" fill="currentColor" stroke="none"></path>
<path d="M660,80 L658.7,99 L654.7,106.2 L647.1,109.6 L626.9,109.6 L619.3,106.2 L615.3,99 L614,80 L615.3,61 L619.3,53.8 L626.9,50.4 L647.1,50.4 L654.7,53.8 L658.7,61 z" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06" stroke-linejoin="round"></path>
<path d="M697.7,93.1 L695.1,103.1 L689.5,108.3 L675,110 L660.5,108.3 L654.9,103.1 L652.3,93.1 L652.3,66.9 L654.9,56.9 L660.5,51.7 L675,50 L689.5,51.7 L695.1,56.9 L697.7,66.9 z" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06" stroke-linejoin="round"></path>
<path d="M660,80 L658.7,99 L655.9,104.1 L654.9,103.1 L652.3,93.1 L652.3,66.9 L654.9,56.9 L655.9,55.9 L658.7,61 z" fill="currentColor" fill-opacity="0.3" stroke="none"></path>
<circle cx="660" cy="80" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="658.7" cy="99" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="654.7" cy="106.2" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="647.1" cy="109.6" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="626.9" cy="109.6" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="619.3" cy="106.2" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="615.3" cy="99" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="614" cy="80" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="615.3" cy="61" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="619.3" cy="53.8" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="626.9" cy="50.4" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="647.1" cy="50.4" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="654.7" cy="53.8" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="658.7" cy="61" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="697.7" cy="93.1" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="695.1" cy="103.1" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="689.5" cy="108.3" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="675" cy="110" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="660.5" cy="108.3" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="654.9" cy="103.1" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="652.3" cy="93.1" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="652.3" cy="66.9" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="654.9" cy="56.9" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="660.5" cy="51.7" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="675" cy="50" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="689.5" cy="51.7" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="695.1" cy="56.9" r="2.2" fill="currentColor" stroke="none"></circle>
<circle cx="697.7" cy="66.9" r="2.2" fill="currentColor" stroke="none"></circle>
<path d="M651,38 L639.5,38" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M634,38 L640,35.5 L640,40.5 z" fill="currentColor" stroke="none"></path>
<path d="M661,38 L672.5,38" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M678,38 L672,40.5 L672,35.5 z" fill="currentColor" stroke="none"></path>
<path d="M651,122 L639.5,122" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M634,122 L640,119.5 L640,124.5 z" fill="currentColor" stroke="none"></path>
<path d="M661,122 L672.5,122" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M678,122 L672,124.5 L672,119.5 z" fill="currentColor" stroke="none"></path>
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="66" y="83" font-size="15" text-anchor="middle" font-style="italic">V</text>
<text x="111" y="32" font-size="15" text-anchor="middle" opacity="0.85" font-style="italic">V<tspan dy="4" font-size="13" font-style="normal">0</tspan><tspan dy="-4" font-size="13"> </tspan></text>
<text x="182" y="83" font-size="15" text-anchor="middle" opacity="0.85" font-style="italic">A<tspan dy="4" font-size="13" font-style="normal">0</tspan><tspan dy="-4" font-size="13"> </tspan></text>
<text x="66" y="156" font-size="16" text-anchor="middle" font-weight="600">1 pressure, <tspan font-style="italic">K</tspan></text>
<text x="66" y="177" font-size="14" text-anchor="middle" opacity="0.78">pushes every</text>
<text x="66" y="197" font-size="14" text-anchor="middle" opacity="0.78">face out toward</text>
<text x="66" y="216" font-size="14" text-anchor="middle" opacity="0.78">a target volume</text>
<text x="182" y="156" font-size="16" text-anchor="middle" font-weight="600">2 area</text>
<text x="182" y="177" font-size="16" text-anchor="middle" font-weight="600">elasticity, <tspan font-style="italic">k<tspan dy="4" font-size="13" font-style="normal">a</tspan><tspan dy="-4" font-size="13"> </tspan></tspan></text>
<text x="182" y="197" font-size="14" text-anchor="middle" opacity="0.78">a spring around</text>
<text x="182" y="216" font-size="14" text-anchor="middle" opacity="0.78">a target area</text>
<text x="298" y="156" font-size="16" text-anchor="middle" font-weight="600">3 surface</text>
<text x="298" y="177" font-size="16" text-anchor="middle" font-weight="600">tension, <tspan font-style="italic">γ</tspan></text>
<text x="298" y="197" font-size="14" text-anchor="middle" opacity="0.78">always shrinks</text>
<text x="298" y="216" font-size="14" text-anchor="middle" opacity="0.78">the surface</text>
<text x="414" y="156" font-size="16" text-anchor="middle" font-weight="600">4 bending, <tspan font-style="italic">k<tspan dy="4" font-size="13" font-style="normal">b</tspan><tspan dy="-4" font-size="13"> </tspan></tspan></text>
<text x="414" y="177" font-size="14" text-anchor="middle" opacity="0.78">resists a sharp</text>
<text x="414" y="197" font-size="14" text-anchor="middle" opacity="0.78">fold between</text>
<text x="414" y="216" font-size="14" text-anchor="middle" opacity="0.78">two faces</text>
<text x="595" y="156" font-size="16" text-anchor="middle" font-weight="600">5 contact</text>
<text x="595" y="177" font-size="16" text-anchor="middle" font-weight="600">between two cells</text>
<text x="534" y="197" font-size="14" text-anchor="middle" opacity="0.78">inside the</text>
<text x="534" y="216" font-size="14" text-anchor="middle" opacity="0.78">cut-off: attract</text>
<text x="656" y="197" font-size="14" text-anchor="middle" opacity="0.78">on overlap:</text>
<text x="656" y="216" font-size="14" text-anchor="middle" opacity="0.78">repel</text>
</g>
</svg>
<figcaption>A cut through a cell for each term of the equation, in order. Panel 5 is two cells in contact.</figcaption>
</figure>

The first term is pressure. $V$ is the cell's volume and $V_0$ the volume it's aiming for, and the pressure works out to $p = -K \ln(V/V_0)$, so a cell smaller than its target pushes every face outward. $K$ is the bulk modulus, how hard the cell resists being squeezed.

Area elasticity (panel 2) does the same job for the membrane. There's a target area $A_0$, tied to the cell's volume, and $k_a$ sets how hard the surface is pulled back when its real area $A$ drifts from it.

The last two terms sit inside an integral, which just means they're added up over the whole surface of the cell. $\gamma$ is surface tension : the cortex, a thin contractile layer under the membrane, is always trying to shrink the surface, the way the surface of a liquid does (panel 3). The other one charges for sharp folds (panel 4). $H$ is the mean curvature, how sharply the surface bends at a point, and $k_b$ is the bending rigidity. On a triangle mesh a fold is the angle between two faces that share an edge.

Panel 5 isn't in the equation. It's contact between cells, and the code has three models for it. In the simplest one, node-face, contact is a pair of springs, which is what the panel shows : membranes closer than a short cut-off distance attract, and membranes that overlap are pushed apart. The other two couple nodes instead. The growth benchmark uses node-node, where a node pairs up with the nearest node on a neighbouring cell and the two then move as one. The compile-time default, which the code calls face-face, lets a node hold one partner per neighbouring cell.

Add all of it up and each node has one force. A node also has mass, and it feels a drag proportional to its velocity. Time moves forward in fixed steps, and in each one the simulator updates a node's momentum from the force and then its position from the new momentum. The paper's time step is `1e-7` seconds. It has to be small. Each step treats the force as fixed while the node moves, so a node that travels too far in one step gets a force on the next that overshoots, and the mesh blows up. That's what unstable means further down.

## grow, divide, remesh

Growth is one line of code : at each step a growing cell's target volume $V_0$ goes up a little. Now the cell is smaller than it wants to be, the pressure turns positive, and the faces move outward until the volume catches up.

Once the volume passes a set division volume the cell divides. The simulator finds the cell's longest axis and puts a plane through the centre, perpendicular to that axis. Each edge the plane crosses is cut, the open cross-section is filled with new triangles, and where there was one closed mesh there are now two, with the target volume halved. Cells die as well. One whose volume drops below a minimum is removed from the tissue.

All of that stretches some triangles and squashes others, so the mesh needs constant repair. On every iteration a refiner goes over each cell and keeps its edges between a minimum length $l_{\min}$ and three times that. An edge that's grown too long is split in two and one that's shrunk too short is merged away, so the triangle count goes up and down as the cell changes shape, and two cells in the same tissue can cost different amounts of work before contact even comes into it.

<figure>
<svg viewBox="0 0 720 318" role="img" aria-labelledby="fig-divide-t fig-divide-d" style="width:100%;height:auto;max-width:720px;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title id="fig-divide-t">A cell grows, divides and is remeshed</title>
<desc id="fig-divide-d">Three steps from left to right: a cell mesh swelling toward a larger dashed target outline, an elongated cell cut by a plane through its centre across its longest axis, and two daughter cells each with a fresh triangle mesh.</desc>
<circle cx="118" cy="128" r="76" stroke="currentColor" stroke-width="1.1" fill="none" stroke-dasharray="5 5" opacity="0.55"></circle>
<path d="M81.8,125.4 L111,106.7 L106.1,137.1 z M133.2,96.7 L111,106.7 L108.9,85.4 z M81.8,125.4 L82.9,99.1 L111,106.7 z M90.2,84.7 L108.9,85.4 L82.9,99.1 z M111,106.7 L82.9,99.1 L108.9,85.4 z M90.2,84.7 L114.9,75.8 L108.9,85.4 z M90.2,84.7 L82.9,99.1 L70.9,105.9 z M81.8,125.4 L68.7,122.7 L82.9,99.1 z M70.9,105.9 L82.9,99.1 L68.7,122.7 z" fill="currentColor" fill-opacity="0.03" stroke="none"></path>
<path d="M81.8,125.4 L106.1,137.1 L84.7,152.1 z M133.2,96.7 L133.5,119.1 L111,106.7 z M106.1,137.1 L111,106.7 L133.5,119.1 z M133.2,96.7 L143.1,85.1 L154,102.1 z M81.8,125.4 L84.7,152.1 L68.7,122.7 z M133.2,96.7 L108.9,85.4 L143.1,85.1 z M114.9,75.8 L143.1,85.1 L108.9,85.4 z" fill="currentColor" fill-opacity="0.052" stroke="none"></path>
<path d="M132,152.2 L106.1,137.1 L133.5,119.1 z M132,152.2 L133.5,119.1 L157,136.6 z M133.2,96.7 L154,102.1 L133.5,119.1 z M133.5,119.1 L154,102.1 L157,136.6 z M67.7,134.4 L68.7,122.7 L72.1,155.5 z M84.7,152.1 L72.1,155.5 L68.7,122.7 z M144,84.3 L143.1,85.1 L114.9,75.8 z" fill="currentColor" fill-opacity="0.075" stroke="none"></path>
<path d="M132,152.2 L110.5,164.1 L106.1,137.1 z M110.5,164.1 L84.7,152.1 L106.1,137.1 z" fill="currentColor" fill-opacity="0.098" stroke="none"></path>
<path d="M132,152.2 L157,136.6 L143.1,165 z M169.6,127.5 L163.6,150.8 L157,136.6 z M145.3,173.7 L143.1,165 L163.6,150.8 z M157,136.6 L163.6,150.8 L143.1,165 z M132,152.2 L143.1,165 L110.5,164.1 z M145.3,173.7 L119.4,177.3 L143.1,165 z M93.3,169.2 L110.5,164.1 L119.4,177.3 z M143.1,165 L119.4,177.3 L110.5,164.1 z M93.3,169.2 L84.7,152.1 L110.5,164.1 z M169.6,127.5 L157,136.6 L154,102.1 z M169.6,127.5 L154,102.1 L163.6,104.9 z M144,84.3 L163.6,104.9 L143.1,85.1 z M154,102.1 L143.1,85.1 L163.6,104.9 z M93.3,169.2 L72.1,155.5 L84.7,152.1 z" fill="currentColor" fill-opacity="0.12" stroke="none"></path>
<path d="M132,152.2 L157,136.6 M132,152.2 L143.1,165 M132,152.2 L110.5,164.1 M132,152.2 L106.1,137.1 M132,152.2 L133.5,119.1 M93.3,169.2 L110.5,164.1 M93.3,169.2 L84.7,152.1 M144,84.3 L143.1,85.1 M145.3,173.7 L143.1,165 M90.2,84.7 L82.9,99.1 M90.2,84.7 L108.9,85.4 M81.8,125.4 L84.7,152.1 M81.8,125.4 L106.1,137.1 M81.8,125.4 L111,106.7 M81.8,125.4 L82.9,99.1 M81.8,125.4 L68.7,122.7 M133.2,96.7 L111,106.7 M133.2,96.7 L133.5,119.1 M133.2,96.7 L154,102.1 M133.2,96.7 L143.1,85.1 M133.2,96.7 L108.9,85.4 M169.6,127.5 L157,136.6 M169.6,127.5 L154,102.1 M157,136.6 L163.6,150.8 M157,136.6 L143.1,165 M157,136.6 L133.5,119.1 M157,136.6 L154,102.1 M163.6,150.8 L143.1,165 M143.1,165 L119.4,177.3 M143.1,165 L110.5,164.1 M119.4,177.3 L110.5,164.1 M110.5,164.1 L84.7,152.1 M110.5,164.1 L106.1,137.1 M84.7,152.1 L106.1,137.1 M84.7,152.1 L72.1,155.5 M84.7,152.1 L68.7,122.7 M106.1,137.1 L111,106.7 M106.1,137.1 L133.5,119.1 M111,106.7 L133.5,119.1 M111,106.7 L82.9,99.1 M111,106.7 L108.9,85.4 M133.5,119.1 L154,102.1 M154,102.1 L143.1,85.1 M154,102.1 L163.6,104.9 M143.1,85.1 L163.6,104.9 M143.1,85.1 L108.9,85.4 M143.1,85.1 L114.9,75.8 M82.9,99.1 L108.9,85.4 M82.9,99.1 L68.7,122.7 M82.9,99.1 L70.9,105.9 M108.9,85.4 L114.9,75.8 M72.1,155.5 L68.7,122.7" stroke="currentColor" stroke-width="1.2" fill="none" opacity="0.85" stroke-linecap="round"></path>
<path d="M93.3,169.2 L119.4,177.3 M93.3,169.2 L72.1,155.5 M144,84.3 L163.6,104.9 M144,84.3 L114.9,75.8 M145.3,173.7 L163.6,150.8 M145.3,173.7 L119.4,177.3 M90.2,84.7 L114.9,75.8 M90.2,84.7 L70.9,105.9 M67.7,134.4 L72.1,155.5 M67.7,134.4 L68.7,122.7 M169.6,127.5 L163.6,150.8 M169.6,127.5 L163.6,104.9 M68.7,122.7 L70.9,105.9" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round"></path>
<g fill="currentColor" stroke="none">
<circle cx="132" cy="152.2" r="2.7"></circle>
<circle cx="93.3" cy="169.2" r="2.7"></circle>
<circle cx="144" cy="84.3" r="2.7"></circle>
<circle cx="145.3" cy="173.7" r="2.7"></circle>
<circle cx="90.2" cy="84.7" r="2.7"></circle>
<circle cx="81.8" cy="125.4" r="2.7"></circle>
<circle cx="67.7" cy="134.4" r="2.7"></circle>
<circle cx="133.2" cy="96.7" r="2.7"></circle>
<circle cx="169.6" cy="127.5" r="2.7"></circle>
<circle cx="157" cy="136.6" r="2.7"></circle>
<circle cx="163.6" cy="150.8" r="2.7"></circle>
<circle cx="143.1" cy="165" r="2.7"></circle>
<circle cx="119.4" cy="177.3" r="2.7"></circle>
<circle cx="110.5" cy="164.1" r="2.7"></circle>
<circle cx="84.7" cy="152.1" r="2.7"></circle>
<circle cx="106.1" cy="137.1" r="2.7"></circle>
<circle cx="111" cy="106.7" r="2.7"></circle>
<circle cx="133.5" cy="119.1" r="2.7"></circle>
<circle cx="154" cy="102.1" r="2.7"></circle>
<circle cx="143.1" cy="85.1" r="2.7"></circle>
<circle cx="163.6" cy="104.9" r="2.7"></circle>
<circle cx="82.9" cy="99.1" r="2.7"></circle>
<circle cx="108.9" cy="85.4" r="2.7"></circle>
<circle cx="72.1" cy="155.5" r="2.7"></circle>
<circle cx="68.7" cy="122.7" r="2.7"></circle>
<circle cx="114.9" cy="75.8" r="2.7"></circle>
<circle cx="70.9" cy="105.9" r="2.7"></circle>
</g>
<path d="M162.4,165.3 L168.6,170.4" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M173.2,174.3 L166.4,172.2 L169.9,168 z" fill="currentColor" stroke="none"></path>
<path d="M73.6,165.3 L67.4,170.4" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M62.8,174.3 L66.1,168 L69.6,172.2 z" fill="currentColor" stroke="none"></path>
<path d="M73.6,90.7 L67.4,85.6" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M62.8,81.7 L69.6,83.8 L66.1,88 z" fill="currentColor" stroke="none"></path>
<path d="M162.4,90.7 L168.6,85.6" stroke="currentColor" stroke-width="1.4" fill="none"></path>
<path d="M173.2,81.7 L169.9,88 L166.4,83.8 z" fill="currentColor" stroke="none"></path>
<path d="M260.8,165.3 L459.2,90.7" stroke="currentColor" stroke-width="1.1" fill="none" stroke-dasharray="9 3 1.5 3" opacity="0.6"></path>
<path d="M324.2,98.6 L320.4,124 L293.3,117.7 z M288.9,153.1 L293.3,117.7 L320.4,124 z M396.1,72.8 L401.8,92.1 L365.5,74.9 z M324.2,98.6 L365.5,74.9 L357.2,97.5 z M401.8,92.1 L357.2,97.5 L365.5,74.9 z M324.2,98.6 L330.8,85.4 L365.5,74.9 z M324.2,98.6 L357.2,97.5 L320.4,124 z M324.2,98.6 L293.3,117.7 L330.8,85.4 z" fill="currentColor" fill-opacity="0.03" stroke="none"></path>
<path d="M316.6,153 L288.9,153.1 L320.4,124 z M282.8,137.7 L293.3,117.7 L288.9,153.1 z M396.1,72.8 L423.7,88.9 L401.8,92.1 z M391,116.8 L357.2,97.5 L401.8,92.1 z M396.1,72.8 L365.5,74.9 L369.9,71.1 z M365.5,74.9 L330.8,85.4 L369.9,71.1 z M391,116.8 L351.9,139.9 L357.2,97.5 z M316.6,153 L320.4,124 L351.9,139.9 z M357.2,97.5 L351.9,139.9 L320.4,124 z" fill="currentColor" fill-opacity="0.052" stroke="none"></path>
<path d="M282.8,137.7 L288.9,153.1 L287.3,167.5 z M316.6,153 L351.9,139.9 L338.8,172.4 z M391,116.8 L401.8,92.1 L418.8,114.2 z M423.7,88.9 L418.8,114.2 L401.8,92.1 z" fill="currentColor" fill-opacity="0.075" stroke="none"></path>
<path d="M391,116.8 L390,144.6 L351.9,139.9 z M338.8,172.4 L351.9,139.9 L390,144.6 z" fill="currentColor" fill-opacity="0.098" stroke="none"></path>
<path d="M322.7,184.4 L287.3,167.5 L313.2,175.1 z M316.6,153 L313.2,175.1 L288.9,153.1 z M287.3,167.5 L288.9,153.1 L313.2,175.1 z M322.7,184.4 L313.2,175.1 L345.5,183.7 z M316.6,153 L338.8,172.4 L313.2,175.1 z M384.6,168.7 L345.5,183.7 L338.8,172.4 z M313.2,175.1 L338.8,172.4 L345.5,183.7 z M384.6,168.7 L386.9,170.3 L345.5,183.7 z M384.6,168.7 L418.5,144.4 L386.9,170.3 z M438.9,118.6 L422.2,144.6 L418.5,144.4 z M386.9,170.3 L418.5,144.4 L422.2,144.6 z M384.6,168.7 L338.8,172.4 L390,144.6 z M438.9,118.6 L418.8,114.2 L423.7,88.9 z M391,116.8 L418.8,114.2 L390,144.6 z M438.9,118.6 L418.5,144.4 L418.8,114.2 z M384.6,168.7 L390,144.6 L418.5,144.4 z M418.8,114.2 L418.5,144.4 L390,144.6 z" fill="currentColor" fill-opacity="0.12" stroke="none"></path>
<path d="M322.7,184.4 L313.2,175.1 M282.8,137.7 L288.9,153.1 M438.9,118.6 L418.5,144.4 M438.9,118.6 L418.8,114.2 M396.1,72.8 L401.8,92.1 M396.1,72.8 L365.5,74.9 M391,116.8 L351.9,139.9 M391,116.8 L390,144.6 M391,116.8 L418.8,114.2 M391,116.8 L401.8,92.1 M391,116.8 L357.2,97.5 M316.6,153 L288.9,153.1 M316.6,153 L313.2,175.1 M316.6,153 L338.8,172.4 M316.6,153 L351.9,139.9 M316.6,153 L320.4,124 M324.2,98.6 L293.3,117.7 M324.2,98.6 L320.4,124 M324.2,98.6 L357.2,97.5 M324.2,98.6 L365.5,74.9 M324.2,98.6 L330.8,85.4 M384.6,168.7 L338.8,172.4 M384.6,168.7 L345.5,183.7 M384.6,168.7 L386.9,170.3 M384.6,168.7 L418.5,144.4 M384.6,168.7 L390,144.6 M287.3,167.5 L288.9,153.1 M287.3,167.5 L313.2,175.1 M288.9,153.1 L313.2,175.1 M288.9,153.1 L293.3,117.7 M288.9,153.1 L320.4,124 M313.2,175.1 L338.8,172.4 M313.2,175.1 L345.5,183.7 M338.8,172.4 L345.5,183.7 M338.8,172.4 L351.9,139.9 M338.8,172.4 L390,144.6 M386.9,170.3 L418.5,144.4 M418.5,144.4 L422.2,144.6 M418.5,144.4 L390,144.6 M418.5,144.4 L418.8,114.2 M351.9,139.9 L390,144.6 M351.9,139.9 L320.4,124 M351.9,139.9 L357.2,97.5 M390,144.6 L418.8,114.2 M293.3,117.7 L320.4,124 M320.4,124 L357.2,97.5 M369.9,71.1 L365.5,74.9 M423.7,88.9 L418.8,114.2 M423.7,88.9 L401.8,92.1 M418.8,114.2 L401.8,92.1 M401.8,92.1 L357.2,97.5 M401.8,92.1 L365.5,74.9 M357.2,97.5 L365.5,74.9 M365.5,74.9 L330.8,85.4" stroke="currentColor" stroke-width="1.2" fill="none" opacity="0.85" stroke-linecap="round"></path>
<path d="M322.7,184.4 L287.3,167.5 M322.7,184.4 L345.5,183.7 M282.8,137.7 L287.3,167.5 M282.8,137.7 L293.3,117.7 M438.9,118.6 L422.2,144.6 M438.9,118.6 L423.7,88.9 M396.1,72.8 L369.9,71.1 M396.1,72.8 L423.7,88.9 M345.5,183.7 L386.9,170.3 M386.9,170.3 L422.2,144.6 M293.3,117.7 L330.8,85.4 M369.9,71.1 L330.8,85.4" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round"></path>
<g fill="currentColor" stroke="none">
<circle cx="322.7" cy="184.4" r="2.7"></circle>
<circle cx="282.8" cy="137.7" r="2.7"></circle>
<circle cx="438.9" cy="118.6" r="2.7"></circle>
<circle cx="396.1" cy="72.8" r="2.7"></circle>
<circle cx="391" cy="116.8" r="2.7"></circle>
<circle cx="316.6" cy="153" r="2.7"></circle>
<circle cx="324.2" cy="98.6" r="2.7"></circle>
<circle cx="384.6" cy="168.7" r="2.7"></circle>
<circle cx="287.3" cy="167.5" r="2.7"></circle>
<circle cx="288.9" cy="153.1" r="2.7"></circle>
<circle cx="313.2" cy="175.1" r="2.7"></circle>
<circle cx="338.8" cy="172.4" r="2.7"></circle>
<circle cx="345.5" cy="183.7" r="2.7"></circle>
<circle cx="386.9" cy="170.3" r="2.7"></circle>
<circle cx="418.5" cy="144.4" r="2.7"></circle>
<circle cx="422.2" cy="144.6" r="2.7"></circle>
<circle cx="351.9" cy="139.9" r="2.7"></circle>
<circle cx="390" cy="144.6" r="2.7"></circle>
<circle cx="293.3" cy="117.7" r="2.7"></circle>
<circle cx="320.4" cy="124" r="2.7"></circle>
<circle cx="369.9" cy="71.1" r="2.7"></circle>
<circle cx="423.7" cy="88.9" r="2.7"></circle>
<circle cx="418.8" cy="114.2" r="2.7"></circle>
<circle cx="401.8" cy="92.1" r="2.7"></circle>
<circle cx="357.2" cy="97.5" r="2.7"></circle>
<circle cx="365.5" cy="74.9" r="2.7"></circle>
<circle cx="330.8" cy="85.4" r="2.7"></circle>
</g>
<g style="color:var(--global-accent, currentColor)">
<path d="M332.9,55.9 L387.1,200.1" stroke="currentColor" stroke-width="2.6" fill="none" stroke-linecap="round"></path>
</g>
<path d="M538.9,127 L557.5,118.2 L556,148.8 z M580.2,112.9 L557.5,118.2 L564.5,98.6 z M538.9,127 L540.6,106.5 L557.5,118.2 z M547.3,89.7 L564.5,98.6 L540.6,106.5 z M557.5,118.2 L540.6,106.5 L564.5,98.6 z M547.3,89.7 L540.6,106.5 L531.1,108.9 z M538.9,127 L526.3,127 L540.6,106.5 z M531.1,108.9 L540.6,106.5 L526.3,127 z" fill="currentColor" fill-opacity="0.03" stroke="none"></path>
<path d="M538.9,127 L556,148.8 L534.5,157.2 z M580.2,112.9 L580.3,135.3 L557.5,118.2 z M556,148.8 L557.5,118.2 L580.3,135.3 z M538.9,127 L534.5,157.2 L526.3,127 z M547.3,89.7 L574.1,91.2 L564.5,98.6 z" fill="currentColor" fill-opacity="0.052" stroke="none"></path>
<path d="M559.6,176.1 L534.5,157.2 L556,148.8 z M576.3,161.3 L556,148.8 L580.3,135.3 z M534.5,157.2 L529.1,156.6 L526.3,127 z M580.2,112.9 L564.5,98.6 L585.4,103.7 z M574.1,91.2 L585.4,103.7 L564.5,98.6 z" fill="currentColor" fill-opacity="0.075" stroke="none"></path>
<path d="M576.3,161.3 L559.6,176.1 L556,148.8 z M576.3,161.3 L580.3,135.3 L599.5,160.8 z M580.2,112.9 L595.6,130.8 L580.3,135.3 z M580.3,135.3 L595.6,130.8 L599.5,160.8 z M580.2,112.9 L585.4,103.7 L595.6,130.8 z" fill="currentColor" fill-opacity="0.098" stroke="none"></path>
<path d="M576.3,161.3 L584.3,184.6 L559.6,176.1 z M587.1,191.6 L564.9,192.3 L584.3,184.6 z M541.9,174.7 L559.6,176.1 L564.9,192.3 z M584.3,184.6 L564.9,192.3 L559.6,176.1 z M541.9,174.7 L534.5,157.2 L559.6,176.1 z M602.8,150 L599.5,160.8 L595.6,130.8 z M576.3,161.3 L599.5,160.8 L584.3,184.6 z M602.8,150 L603,176.8 L599.5,160.8 z M587.1,191.6 L584.3,184.6 L603,176.8 z M599.5,160.8 L603,176.8 L584.3,184.6 z M541.9,174.7 L529.1,156.6 L534.5,157.2 z M602.8,150 L605.2,156.4 L603,176.8 z" fill="currentColor" fill-opacity="0.12" stroke="none"></path>
<path d="M576.3,161.3 L584.3,184.6 M576.3,161.3 L559.6,176.1 M576.3,161.3 L556,148.8 M576.3,161.3 L580.3,135.3 M576.3,161.3 L599.5,160.8 M538.9,127 L534.5,157.2 M538.9,127 L556,148.8 M538.9,127 L557.5,118.2 M538.9,127 L526.3,127 M538.9,127 L540.6,106.5 M541.9,174.7 L559.6,176.1 M541.9,174.7 L534.5,157.2 M580.2,112.9 L557.5,118.2 M580.2,112.9 L580.3,135.3 M580.2,112.9 L595.6,130.8 M580.2,112.9 L585.4,103.7 M580.2,112.9 L564.5,98.6 M547.3,89.7 L540.6,106.5 M547.3,89.7 L564.5,98.6 M602.8,150 L599.5,160.8 M602.8,150 L603,176.8 M587.1,191.6 L584.3,184.6 M584.3,184.6 L564.9,192.3 M584.3,184.6 L559.6,176.1 M584.3,184.6 L599.5,160.8 M584.3,184.6 L603,176.8 M564.9,192.3 L559.6,176.1 M559.6,176.1 L534.5,157.2 M559.6,176.1 L556,148.8 M534.5,157.2 L556,148.8 M534.5,157.2 L529.1,156.6 M534.5,157.2 L526.3,127 M556,148.8 L557.5,118.2 M556,148.8 L580.3,135.3 M557.5,118.2 L580.3,135.3 M557.5,118.2 L540.6,106.5 M557.5,118.2 L564.5,98.6 M580.3,135.3 L595.6,130.8 M580.3,135.3 L599.5,160.8 M595.6,130.8 L599.5,160.8 M599.5,160.8 L603,176.8 M526.3,127 L540.6,106.5 M585.4,103.7 L564.5,98.6 M540.6,106.5 L564.5,98.6 M540.6,106.5 L531.1,108.9 M564.5,98.6 L574.1,91.2" stroke="currentColor" stroke-width="1.2" fill="none" opacity="0.85" stroke-linecap="round"></path>
<path d="M541.9,174.7 L564.9,192.3 M541.9,174.7 L529.1,156.6 M547.3,89.7 L574.1,91.2 M547.3,89.7 L531.1,108.9 M602.8,150 L595.6,130.8 M602.8,150 L605.2,156.4 M587.1,191.6 L564.9,192.3 M587.1,191.6 L603,176.8 M595.6,130.8 L585.4,103.7 M603,176.8 L605.2,156.4 M529.1,156.6 L526.3,127 M526.3,127 L531.1,108.9 M585.4,103.7 L574.1,91.2" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round"></path>
<g fill="currentColor" stroke="none">
<circle cx="576.3" cy="161.3" r="2.7"></circle>
<circle cx="538.9" cy="127" r="2.7"></circle>
<circle cx="541.9" cy="174.7" r="2.7"></circle>
<circle cx="580.2" cy="112.9" r="2.7"></circle>
<circle cx="547.3" cy="89.7" r="2.7"></circle>
<circle cx="602.8" cy="150" r="2.7"></circle>
<circle cx="587.1" cy="191.6" r="2.7"></circle>
<circle cx="584.3" cy="184.6" r="2.7"></circle>
<circle cx="564.9" cy="192.3" r="2.7"></circle>
<circle cx="559.6" cy="176.1" r="2.7"></circle>
<circle cx="534.5" cy="157.2" r="2.7"></circle>
<circle cx="556" cy="148.8" r="2.7"></circle>
<circle cx="557.5" cy="118.2" r="2.7"></circle>
<circle cx="580.3" cy="135.3" r="2.7"></circle>
<circle cx="595.6" cy="130.8" r="2.7"></circle>
<circle cx="599.5" cy="160.8" r="2.7"></circle>
<circle cx="603" cy="176.8" r="2.7"></circle>
<circle cx="529.1" cy="156.6" r="2.7"></circle>
<circle cx="526.3" cy="127" r="2.7"></circle>
<circle cx="605.2" cy="156.4" r="2.7"></circle>
<circle cx="585.4" cy="103.7" r="2.7"></circle>
<circle cx="540.6" cy="106.5" r="2.7"></circle>
<circle cx="564.5" cy="98.6" r="2.7"></circle>
<circle cx="574.1" cy="91.2" r="2.7"></circle>
<circle cx="531.1" cy="108.9" r="2.7"></circle>
</g>
<path d="M624.6,70.2 L649.7,80.2 L621.8,92 z M624.6,70.2 L613,68 L634.7,63.2 z M624.6,70.2 L621.8,92 L606.7,80.2 z M621.8,92 L607.9,111.6 L606.7,80.2 z M624.6,70.2 L606.7,80.2 L613,68 z M596.4,84.2 L613,68 L606.7,80.2 z" fill="currentColor" fill-opacity="0.03" stroke="none"></path>
<path d="M630,118.7 L621.8,92 L649.9,108.3 z M649.9,108.3 L621.8,92 L649.7,80.2 z M624.6,70.2 L634.7,63.2 L649.7,80.2 z M630,118.7 L607.9,111.6 L621.8,92 z M600.2,103.2 L606.7,80.2 L607.9,111.6 z M600.2,103.2 L596.4,84.2 L606.7,80.2 z" fill="currentColor" fill-opacity="0.052" stroke="none"></path>
<path d="M630,118.7 L649.9,108.3 L645.1,141 z M649.7,80.2 L634.7,63.2 L664.7,82.3 z M630,118.7 L645.1,141 L618.7,136.7 z M630,118.7 L618.7,136.7 L607.9,111.6 z" fill="currentColor" fill-opacity="0.075" stroke="none"></path>
<path d="M669.6,100.2 L669.8,127.3 L649.9,108.3 z M669.8,127.3 L645.1,141 L649.9,108.3 z M669.6,100.2 L649.9,108.3 L649.7,80.2 z M669.6,100.2 L649.7,80.2 L664.7,82.3 z M620.5,157.2 L608.4,125.1 L618.7,136.7 z M600.2,103.2 L607.9,111.6 L608.4,125.1 z M608.4,125.1 L607.9,111.6 L618.7,136.7 z" fill="currentColor" fill-opacity="0.098" stroke="none"></path>
<path d="M669.6,100.2 L664.7,82.3 L678.5,112.7 z M664.7,82.3 L671.2,93.6 L678.5,112.7 z M669.6,100.2 L678.5,112.7 L669.8,127.3 z M661.8,154 L669.8,127.3 L676.4,146 z M678.5,112.7 L676.4,146 L669.8,127.3 z M661.8,154 L645.1,141 L669.8,127.3 z M661.8,154 L676.4,146 L652.8,161.1 z M661.8,154 L636.6,161.9 L645.1,141 z M620.5,157.2 L618.7,136.7 L636.6,161.9 z M645.1,141 L636.6,161.9 L618.7,136.7 z M620.5,157.2 L636.6,161.9 L628.9,166 z M661.8,154 L652.8,161.1 L636.6,161.9 z M628.9,166 L636.6,161.9 L652.8,161.1 z" fill="currentColor" fill-opacity="0.12" stroke="none"></path>
<path d="M669.6,100.2 L664.7,82.3 M669.6,100.2 L678.5,112.7 M669.6,100.2 L669.8,127.3 M669.6,100.2 L649.9,108.3 M669.6,100.2 L649.7,80.2 M661.8,154 L676.4,146 M661.8,154 L669.8,127.3 M661.8,154 L645.1,141 M661.8,154 L652.8,161.1 M661.8,154 L636.6,161.9 M600.2,103.2 L607.9,111.6 M600.2,103.2 L606.7,80.2 M630,118.7 L645.1,141 M630,118.7 L649.9,108.3 M630,118.7 L621.8,92 M630,118.7 L607.9,111.6 M630,118.7 L618.7,136.7 M620.5,157.2 L636.6,161.9 M620.5,157.2 L618.7,136.7 M624.6,70.2 L621.8,92 M624.6,70.2 L649.7,80.2 M624.6,70.2 L634.7,63.2 M624.6,70.2 L613,68 M624.6,70.2 L606.7,80.2 M664.7,82.3 L678.5,112.7 M664.7,82.3 L649.7,80.2 M678.5,112.7 L669.8,127.3 M676.4,146 L669.8,127.3 M669.8,127.3 L645.1,141 M669.8,127.3 L649.9,108.3 M645.1,141 L649.9,108.3 M645.1,141 L636.6,161.9 M645.1,141 L618.7,136.7 M649.9,108.3 L621.8,92 M649.9,108.3 L649.7,80.2 M621.8,92 L649.7,80.2 M621.8,92 L607.9,111.6 M621.8,92 L606.7,80.2 M649.7,80.2 L634.7,63.2 M652.8,161.1 L636.6,161.9 M613,68 L606.7,80.2 M607.9,111.6 L606.7,80.2 M607.9,111.6 L618.7,136.7 M607.9,111.6 L608.4,125.1 M606.7,80.2 L596.4,84.2 M636.6,161.9 L618.7,136.7 M636.6,161.9 L628.9,166 M618.7,136.7 L608.4,125.1" stroke="currentColor" stroke-width="1.2" fill="none" opacity="0.85" stroke-linecap="round"></path>
<path d="M600.2,103.2 L596.4,84.2 M600.2,103.2 L608.4,125.1 M620.5,157.2 L608.4,125.1 M620.5,157.2 L628.9,166 M664.7,82.3 L671.2,93.6 M664.7,82.3 L634.7,63.2 M671.2,93.6 L678.5,112.7 M678.5,112.7 L676.4,146 M676.4,146 L652.8,161.1 M634.7,63.2 L613,68 M652.8,161.1 L628.9,166 M613,68 L596.4,84.2" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round"></path>
<g fill="currentColor" stroke="none">
<circle cx="669.6" cy="100.2" r="2.7"></circle>
<circle cx="661.8" cy="154" r="2.7"></circle>
<circle cx="600.2" cy="103.2" r="2.7"></circle>
<circle cx="630" cy="118.7" r="2.7"></circle>
<circle cx="620.5" cy="157.2" r="2.7"></circle>
<circle cx="624.6" cy="70.2" r="2.7"></circle>
<circle cx="664.7" cy="82.3" r="2.7"></circle>
<circle cx="671.2" cy="93.6" r="2.7"></circle>
<circle cx="678.5" cy="112.7" r="2.7"></circle>
<circle cx="676.4" cy="146" r="2.7"></circle>
<circle cx="669.8" cy="127.3" r="2.7"></circle>
<circle cx="645.1" cy="141" r="2.7"></circle>
<circle cx="649.9" cy="108.3" r="2.7"></circle>
<circle cx="621.8" cy="92" r="2.7"></circle>
<circle cx="649.7" cy="80.2" r="2.7"></circle>
<circle cx="634.7" cy="63.2" r="2.7"></circle>
<circle cx="652.8" cy="161.1" r="2.7"></circle>
<circle cx="613" cy="68" r="2.7"></circle>
<circle cx="607.9" cy="111.6" r="2.7"></circle>
<circle cx="606.7" cy="80.2" r="2.7"></circle>
<circle cx="636.6" cy="161.9" r="2.7"></circle>
<circle cx="618.7" cy="136.7" r="2.7"></circle>
<circle cx="596.4" cy="84.2" r="2.7"></circle>
<circle cx="608.4" cy="125.1" r="2.7"></circle>
<circle cx="628.9" cy="166" r="2.7"></circle>
</g>
<path d="M208,128 L234.5,128" stroke="currentColor" stroke-width="1.6" fill="none" opacity="0.7"></path>
<path d="M242,128 L234,131.4 L234,124.6 z" fill="currentColor" stroke="none" opacity="0.7"></path>
<path d="M478,128 L504.5,128" stroke="currentColor" stroke-width="1.6" fill="none" opacity="0.7"></path>
<path d="M512,128 L504,131.4 L504,124.6 z" fill="currentColor" stroke="none" opacity="0.7"></path>
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="118" y="42" font-size="14" text-anchor="middle" opacity="0.78">target volume</text>
<text x="465.2" y="81.7" font-size="14" text-anchor="middle" opacity="0.78">longest axis</text>
<text x="118" y="246" font-size="17" text-anchor="middle" font-weight="600">1  grow</text>
<text x="360" y="246" font-size="17" text-anchor="middle" font-weight="600">2  divide</text>
<text x="602" y="246" font-size="17" text-anchor="middle" font-weight="600">3  remesh</text>
<text x="118" y="268" font-size="14" text-anchor="middle" opacity="0.78">the target volume rises</text>
<text x="118" y="287" font-size="14" text-anchor="middle" opacity="0.78">and the cell follows</text>
<text x="360" y="268" font-size="14" text-anchor="middle" opacity="0.78">past the division volume, a plane</text>
<text x="360" y="287" font-size="14" text-anchor="middle" opacity="0.78">cuts through the centre,</text>
<text x="360" y="306" font-size="14" text-anchor="middle" opacity="0.78">across the longest axis</text>
<text x="602" y="268" font-size="14" text-anchor="middle" opacity="0.78">two daughters, each with a new</text>
<text x="602" y="287" font-size="14" text-anchor="middle" opacity="0.78">mesh; every edge stays between</text>
<text x="602" y="306" font-size="14" text-anchor="middle" opacity="0.78"><tspan font-style="italic">l<tspan dy="4" font-size="13" font-style="normal">min</tspan><tspan dy="-4" font-size="13"> </tspan></tspan>and 3 <tspan font-style="italic">l<tspan dy="4" font-size="13" font-style="normal">min</tspan><tspan dy="-4" font-size="13"> </tspan></tspan></text>
</g>
<g style="color:var(--global-accent, currentColor)">
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="326.9" y="45.9" font-size="15" text-anchor="middle" font-weight="600">division plane</text>
</g>
</g>
</svg>
<figcaption>The dashed outline in step 1 is the target volume.</figcaption>
</figure>

## one iteration, phase by phase

One iteration moves every node forward by one time step. In v2.1.0 it goes in this order.

1. Divide the cells that are ready (checked every 5 iterations).
2. Reset the face types. A face's type says what it borders, a neighbouring cell for instance, and the types can have different tension and stiffness. It's worked out again after contact.
3. Refine the meshes, and drop any cell whose refinement failed.
4. Find the contacts between cells and add the contact forces.
5. Work the face types out again (the code calls this the polarization update), then add the internal forces from the energy equation.
6. Time integration, which moves the nodes.
7. Every 50 iterations, write statistics and re-estimate what each cell costs.
8. Remove cells that have died or gone unstable (a node position that's no longer a finite number, say, or a volume that's gone negative).

Nearly all of these are loops over cells, and a loop over cells is the natural thing to split across cores : this thread takes these cells, that thread takes those. The code does it with OpenMP, where a `#pragma omp parallel for` line above a loop tells the compiler to share the loop's cells out among threads.

Not all of it gets shared out. Time integration runs on one thread (its library is built without OpenMP in every version, mine included). So do removing cells, the statistics pass, and a few passes inside the contact phase.

<figure>
<svg viewBox="0 0 720 371" role="img" aria-labelledby="fig-iteration-t fig-iteration-d" style="width:100%;height:auto;max-width:720px;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title id="fig-iteration-t">One iteration, phase by phase</title>
<desc id="fig-iteration-d">A strip of eight thread rows with time running left to right, cut into seven phases in order. Divide, reset face types, refine meshes, contact and polarization plus internal forces fill all eight rows; time integration and remove cells fill one row. Contact takes three quarters of the width and starts with a short grid build on one row.</desc>
<g stroke="currentColor" stroke-width="1.1" fill="currentColor" fill-opacity="0.09">
<rect x="88" y="90" width="25" height="15" rx="2"></rect>
<rect x="88" y="108" width="25" height="15" rx="2"></rect>
<rect x="88" y="126" width="25" height="15" rx="2"></rect>
<rect x="88" y="144" width="25" height="15" rx="2"></rect>
<rect x="88" y="162" width="25" height="15" rx="2"></rect>
<rect x="88" y="180" width="25" height="15" rx="2"></rect>
<rect x="88" y="198" width="25" height="15" rx="2"></rect>
<rect x="88" y="216" width="25" height="15" rx="2"></rect>
<rect x="116" y="90" width="25" height="15" rx="2"></rect>
<rect x="116" y="108" width="25" height="15" rx="2"></rect>
<rect x="116" y="126" width="25" height="15" rx="2"></rect>
<rect x="116" y="144" width="25" height="15" rx="2"></rect>
<rect x="116" y="162" width="25" height="15" rx="2"></rect>
<rect x="116" y="180" width="25" height="15" rx="2"></rect>
<rect x="116" y="198" width="25" height="15" rx="2"></rect>
<rect x="116" y="216" width="25" height="15" rx="2"></rect>
<rect x="144" y="90" width="25" height="15" rx="2"></rect>
<rect x="144" y="108" width="25" height="15" rx="2"></rect>
<rect x="144" y="126" width="25" height="15" rx="2"></rect>
<rect x="144" y="144" width="25" height="15" rx="2"></rect>
<rect x="144" y="162" width="25" height="15" rx="2"></rect>
<rect x="144" y="180" width="25" height="15" rx="2"></rect>
<rect x="144" y="198" width="25" height="15" rx="2"></rect>
<rect x="144" y="216" width="25" height="15" rx="2"></rect>
<rect x="625" y="90" width="25" height="15" rx="2"></rect>
<rect x="625" y="108" width="25" height="15" rx="2"></rect>
<rect x="625" y="126" width="25" height="15" rx="2"></rect>
<rect x="625" y="144" width="25" height="15" rx="2"></rect>
<rect x="625" y="162" width="25" height="15" rx="2"></rect>
<rect x="625" y="180" width="25" height="15" rx="2"></rect>
<rect x="625" y="198" width="25" height="15" rx="2"></rect>
<rect x="625" y="216" width="25" height="15" rx="2"></rect>
<rect x="653" y="90" width="25" height="15" rx="2"></rect>
<rect x="681" y="90" width="25" height="15" rx="2"></rect>
</g>
<g style="color:var(--global-accent, currentColor)" stroke="currentColor" stroke-width="1.2" fill="currentColor" fill-opacity="0.2">
<rect x="172" y="90" width="26" height="15" rx="2"></rect>
<rect x="201" y="90" width="421" height="15" rx="2"></rect>
<rect x="201" y="108" width="421" height="15" rx="2"></rect>
<rect x="201" y="126" width="421" height="15" rx="2"></rect>
<rect x="201" y="144" width="421" height="15" rx="2"></rect>
<rect x="201" y="162" width="421" height="15" rx="2"></rect>
<rect x="201" y="180" width="421" height="15" rx="2"></rect>
<rect x="201" y="198" width="421" height="15" rx="2"></rect>
<rect x="201" y="216" width="421" height="15" rx="2"></rect>
</g>
<g stroke="currentColor" stroke-width="0.9" fill="none" opacity="0.34" stroke-dasharray="2.5 2.5">
<rect x="653" y="108" width="25" height="15" rx="2"></rect>
<rect x="653" y="126" width="25" height="15" rx="2"></rect>
<rect x="653" y="144" width="25" height="15" rx="2"></rect>
<rect x="653" y="162" width="25" height="15" rx="2"></rect>
<rect x="653" y="180" width="25" height="15" rx="2"></rect>
<rect x="653" y="198" width="25" height="15" rx="2"></rect>
<rect x="653" y="216" width="25" height="15" rx="2"></rect>
<rect x="681" y="108" width="25" height="15" rx="2"></rect>
<rect x="681" y="126" width="25" height="15" rx="2"></rect>
<rect x="681" y="144" width="25" height="15" rx="2"></rect>
<rect x="681" y="162" width="25" height="15" rx="2"></rect>
<rect x="681" y="180" width="25" height="15" rx="2"></rect>
<rect x="681" y="198" width="25" height="15" rx="2"></rect>
<rect x="681" y="216" width="25" height="15" rx="2"></rect>
</g>
<g style="color:var(--global-accent, currentColor)" stroke="currentColor" stroke-width="0.9" fill="none" opacity="0.5" stroke-dasharray="2.5 2.5">
<rect x="172" y="108" width="26" height="15" rx="2"></rect>
<rect x="172" y="126" width="26" height="15" rx="2"></rect>
<rect x="172" y="144" width="26" height="15" rx="2"></rect>
<rect x="172" y="162" width="26" height="15" rx="2"></rect>
<rect x="172" y="180" width="26" height="15" rx="2"></rect>
<rect x="172" y="198" width="26" height="15" rx="2"></rect>
<rect x="172" y="216" width="26" height="15" rx="2"></rect>
</g>
<path d="M100.5,87 L100.5,23 L107.5,23 M128.5,87 L128.5,45 L135.5,45 M156.5,87 L156.5,67 L163.5,67 M637.5,234 L637.5,252 L630.5,252 M665.5,234 L665.5,274 L658.5,274 M693.5,234 L693.5,296 L686.5,296" stroke="currentColor" stroke-width="1" fill="none" opacity="0.6"></path>
<path d="M82,90 L78,90 L78,231 L82,231" stroke="currentColor" stroke-width="1" fill="none" opacity="0.6"></path>
<path d="M88,321 L699.5,321" stroke="currentColor" stroke-width="1.1" fill="none" opacity="0.6"></path>
<path d="M706,321 L699,323.9 L699,318.1 z" fill="currentColor" stroke="none" opacity="0.6"></path>
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="8" y="165.5" font-size="14" opacity="0.78">8 threads</text>
<text x="111.5" y="28" font-size="16"><tspan font-weight="600">divide</tspan><tspan dx="6" font-size="14">(every 5 iterations)</tspan><tspan dx="9" font-size="14" fill-opacity="0.78">all threads</tspan></text>
<text x="139.5" y="50" font-size="16"><tspan font-weight="600">reset face types</tspan><tspan dx="9" font-size="14" fill-opacity="0.78">all threads</tspan></text>
<text x="167.5" y="72" font-size="16"><tspan font-weight="600">refine meshes</tspan><tspan dx="9" font-size="14" fill-opacity="0.78">all threads</tspan></text>
<text x="626.5" y="257" font-size="16" text-anchor="end"><tspan font-weight="600">polarization + internal forces</tspan><tspan dx="9" font-size="14" fill-opacity="0.78">all threads</tspan></text>
<text x="654.5" y="279" font-size="16" text-anchor="end"><tspan font-weight="600">time integration</tspan><tspan dx="9" font-size="14" fill-opacity="0.78">one thread</tspan></text>
<text x="682.5" y="301" font-size="16" text-anchor="end"><tspan font-weight="600">remove cells</tspan><tspan dx="9" font-size="14" fill-opacity="0.78">one thread</tspan></text>
<text x="706" y="340" font-size="14" text-anchor="end" opacity="0.78">time</text>
<text x="88" y="340" font-size="14" opacity="0.78">short phases and the grid build are drawn equal, not to scale</text>
<text x="88" y="359" font-size="14" opacity="0.78">not shown: a statistics pass every 50 iterations, on one thread</text>
</g>
<g style="color:var(--global-accent, currentColor)">
<path d="M430,87 L430,23 L437,23 M185,234 L185,274 L192,274" stroke="currentColor" stroke-width="1.1" fill="none"></path>
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="441" y="28" font-size="17"><tspan font-weight="700">contact</tspan><tspan dx="9" font-size="14">all threads</tspan></text>
<text x="441" y="50" font-size="14">about three quarters of an</text>
<text x="441" y="72" font-size="14">iteration at 100,000 cells</text>
<text x="196" y="279" font-size="14" font-weight="600">grid build: one thread</text>
</g>
</g>
</svg>
<figcaption>One iteration of v2.1.0 with the node-node contact model. Time runs left to right and each row is a thread.</figcaption>
</figure>

The 2.x builds time the four main phases (mesh refinement, contact, internal forces and time integration), and by 100,000 cells contact takes about three quarters of an iteration, a share that rises as the tissue grows. So what is contact detection doing with all that time?

## finding contacts with a grid

A tissue of 100,000 cells has about 24 million faces. Testing every node against every face of every other cell would never finish, so the simulator sorts the faces into a grid first.

Space is cut into equal cubes called voxels, and the grid is rebuilt from scratch every iteration. Each face gets a bounding box, padded by the contact cut-off distance so that a node just across a voxel boundary still finds it, and is entered into every voxel that box overlaps. After that a node only has to look up the one voxel it sits in and test itself against the faces listed there. (For two of the three contact models, node-node among them, building this grid is one of the single-thread passes.)

<figure>
<svg viewBox="0 0 720 376" role="img" aria-labelledby="fig-grid-t fig-grid-d" style="width:100%;height:auto;max-width:720px;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title id="fig-grid-t">Finding contacts with a grid of voxels</title>
<desc id="fig-grid-d">A slice through the edge of a tissue laid over a square grid of voxels, with labels to the right. One highlighted voxel deep inside the tissue holds a node and many faces of other cells, drawn thick. One highlighted voxel on the free surface holds a node and no faces of other cells.</desc>
<path d="M12,12 L12,306 M54,12 L54,306 M96,12 L96,306 M138,12 L138,306 M180,12 L180,306 M222,12 L222,306 M264,12 L264,306 M306,12 L306,306 M348,12 L348,306 M390,12 L390,306 M432,12 L432,306 M474,12 L474,306 M12,12 L474,12 M12,54 L474,54 M12,96 L474,96 M12,138 L474,138 M12,180 L474,180 M12,222 L474,222 M12,264 L474,264 M12,306 L474,306" stroke="currentColor" stroke-width="0.8" fill="none" opacity="0.26"></path>
<g style="color:var(--global-accent, currentColor)">
<rect x="54" y="264" width="42" height="42" fill="currentColor" fill-opacity="0.14" stroke="none"></rect>
<rect x="306" y="138" width="42" height="42" fill="currentColor" fill-opacity="0.14" stroke="none"></rect>
</g>
<path d="M33.3,26.2 L23.9,16.6 L21.7,12 L104.1,12 L97.6,20.6 L87.7,28.8 L71.9,36.4 L62.5,37.3 L47.4,34.4 z M133.9,22 L125.2,14.2 L124,12 L207,12 L202.2,18 L187.9,27.7 L172.9,32.8 L161.9,33.2 L148.2,30 z M317.1,12 L307.9,20.6 L296.3,26.7 L282.9,33.3 L268.4,37.6 L256.5,37.9 L240.9,32.1 L229.8,22.2 L224.7,14.7 L224,12 z M48,107.5 L39.4,112.6 L22.3,122.3 L12.7,121.7 L12,121.3 L12,25.8 L21.9,27.1 L37.9,34.7 L45.3,40.1 L55.4,52 L61.2,67.7 L61.1,81 L61.1,94.1 z M94.2,115.5 L82,108.9 L70.1,102.1 L65.9,87.5 L66,74.9 L67.5,62.3 L73.6,48.9 L85.3,36.2 L97.6,28.2 L107,24.6 L125.2,24.7 L139,30.7 L148,37.8 L156.6,49.6 L160.1,61.1 L160.6,77.5 L156.6,91.1 L147.8,103.1 L139.5,109.8 L120.9,117.8 L109.4,118.9 z M241.5,113.8 L230.6,120.5 L218.4,126.6 L204.5,119.8 L195.3,114.9 L180.8,105.7 L172,95.6 L165.7,79.3 L165.6,65.2 L168.2,55.4 L176.1,42.1 L185.3,34.3 L200.1,27.8 L216.6,26.2 L230.6,30.5 L240,37.4 L248.6,47.5 L253.7,59.9 L255.5,76.7 L255.1,93 L254.8,103.9 z M358.9,92.4 L355.1,101.1 L346,113.4 L334.7,122.3 L318.8,128.6 L307.5,126.1 L294.2,120.4 L279.6,114.1 L269.5,109.6 L259.9,97.6 L260.1,87.1 L260.4,70.5 L264.2,58.4 L272.3,47.2 L285.4,37.5 L297.6,31.4 L310.2,25.5 L323.6,21.7 L337.1,28.3 L346.2,35.9 L354.9,47.5 L360.7,63.3 L361.8,74.4 z M12,135.7 L12,135.7 L16,149.6 L16,163.4 L12,175.3 z M34.2,191.9 L25.2,179.2 L21.2,165.5 L20.8,156.1 L22.5,141.7 L32.1,124 L41.2,117 L55.9,108.9 L66.9,106.8 L80.9,113.8 L94.1,121.2 L103.6,131.6 L109.6,146.2 L110.8,157 L108.2,175.9 L99.5,188.9 L91.7,195.7 L77,203 L60.6,205.2 L49.7,202.4 z M136.9,196 L126.2,186.2 L118.2,171.4 L115.7,160.2 L117.4,139.9 L123.8,127.4 L134.4,118.1 L148.6,111.5 L161.6,109.5 L174.7,111.2 L189.5,117.2 L201,125.1 L211.9,139.1 L215.3,152.2 L214.5,163.3 L208.6,181.2 L200.4,192 L188.5,200.9 L174.6,206.3 L162.6,206.7 L148.4,202.9 z M322,169.4 L319,183.2 L313.1,195 L303.5,206.1 L291.6,214.1 L280.4,218.2 L265,216.6 L256.3,211.3 L242.8,202.6 L231.3,193 L224.6,183.8 L219.5,168.3 L220.1,152.3 L221.2,138.1 L226.8,128.6 L242.7,118.7 L253.5,112.2 L264.8,113.1 L279.9,119.4 L292.5,124.9 L305.3,130.8 L315.6,138.6 L321.4,155.7 z M12,196.9 L18.4,195.8 L33.5,198.7 L43.1,204.4 L56.5,219.7 L60.6,230.3 L62.2,245.6 L59.7,260.6 L53,272 L42,281.1 L26.4,288.2 L12.5,290.6 L12,290.5 z M83.3,279 L75.3,270.6 L69.6,259.5 L67,244 L68.2,231 L73.9,217.2 L86.5,204.4 L96.9,198.4 L114,195.2 L127.1,197.4 L136.2,201.2 L152.3,213.4 L160.2,224.9 L162.7,233.4 L162.1,252.7 L158,263.2 L149,274.7 L135,284.9 L126,288.7 L110.2,290.9 L97.8,287.9 z M270,248.5 L267.2,261.3 L259.7,271.1 L244.6,280.4 L232.9,286.7 L220.1,289.4 L208.9,288.5 L192.5,282 L181.6,274.3 L173.4,264.6 L168.6,252.1 L169,234.8 L172.9,223.6 L183.1,210.8 L196.2,201.8 L208.6,198 L221.8,198.3 L237,204.6 L248.3,211.9 L262.2,221.3 L268.8,231.6 z M22.7,306 L28.2,296.9 L37.7,288.9 L55.4,281.6 L64.5,280.8 L78.5,283.4 L93.7,291.5 L105.4,298.7 L108.7,306 z M121.8,306 L124.2,301.3 L132.4,292.9 L143.4,285.3 L158,279.8 L174.5,280.2 L185.9,283.8 L197,290.1 L209.1,303.3 L210.3,306 z M225.7,306 L234.5,293.6 L244.7,286 L255,279.9 L271.5,272.7 L286.3,276.1 L295.9,280.7 L306,288.6 L316.4,302.7 L317.6,306 z" fill="currentColor" fill-opacity="0.055" stroke="none"></path>
<path d="M33.3,26.2 L23.9,16.6 M23.9,16.6 L21.7,12 M104.1,12 L97.6,20.6 M97.6,20.6 L87.7,28.8 M87.7,28.8 L71.9,36.4 M71.9,36.4 L62.5,37.3 M62.5,37.3 L47.4,34.4 M47.4,34.4 L33.3,26.2 M133.9,22 L125.2,14.2 M125.2,14.2 L124,12 M207,12 L202.2,18 M202.2,18 L187.9,27.7 M187.9,27.7 L172.9,32.8 M172.9,32.8 L161.9,33.2 M161.9,33.2 L148.2,30 M148.2,30 L133.9,22 M317.1,12 L307.9,20.6 M307.9,20.6 L296.3,26.7 M296.3,26.7 L282.9,33.3 M282.9,33.3 L268.4,37.6 M268.4,37.6 L256.5,37.9 M256.5,37.9 L240.9,32.1 M240.9,32.1 L229.8,22.2 M229.8,22.2 L224.7,14.7 M224.7,14.7 L224,12 M48,107.5 L39.4,112.6 M39.4,112.6 L22.3,122.3 M22.3,122.3 L12.7,121.7 M12.7,121.7 L12,121.3 M12,25.8 L21.9,27.1 M21.9,27.1 L37.9,34.7 M37.9,34.7 L45.3,40.1 M45.3,40.1 L55.4,52 M55.4,52 L61.2,67.7 M61.2,67.7 L61.1,81 M61.1,81 L61.1,94.1 M61.1,94.1 L48,107.5 M94.2,115.5 L82,108.9 M82,108.9 L70.1,102.1 M70.1,102.1 L65.9,87.5 M65.9,87.5 L66,74.9 M66,74.9 L67.5,62.3 M67.5,62.3 L73.6,48.9 M73.6,48.9 L85.3,36.2 M85.3,36.2 L97.6,28.2 M97.6,28.2 L107,24.6 M107,24.6 L125.2,24.7 M125.2,24.7 L139,30.7 M139,30.7 L148,37.8 M148,37.8 L156.6,49.6 M156.6,49.6 L160.1,61.1 M160.1,61.1 L160.6,77.5 M160.6,77.5 L156.6,91.1 M156.6,91.1 L147.8,103.1 M147.8,103.1 L139.5,109.8 M139.5,109.8 L120.9,117.8 M120.9,117.8 L109.4,118.9 M109.4,118.9 L94.2,115.5 M241.5,113.8 L230.6,120.5 M230.6,120.5 L218.4,126.6 M218.4,126.6 L204.5,119.8 M204.5,119.8 L195.3,114.9 M195.3,114.9 L180.8,105.7 M180.8,105.7 L172,95.6 M172,95.6 L165.7,79.3 M165.7,79.3 L165.6,65.2 M165.6,65.2 L168.2,55.4 M168.2,55.4 L176.1,42.1 M176.1,42.1 L185.3,34.3 M185.3,34.3 L200.1,27.8 M200.1,27.8 L216.6,26.2 M216.6,26.2 L230.6,30.5 M230.6,30.5 L240,37.4 M240,37.4 L248.6,47.5 M248.6,47.5 L253.7,59.9 M253.7,59.9 L255.5,76.7 M255.5,76.7 L255.1,93 M255.1,93 L254.8,103.9 M254.8,103.9 L241.5,113.8 M358.9,92.4 L355.1,101.1 M355.1,101.1 L346,113.4 M346,113.4 L334.7,122.3 M334.7,122.3 L318.8,128.6 M318.8,128.6 L307.5,126.1 M307.5,126.1 L294.2,120.4 M294.2,120.4 L279.6,114.1 M279.6,114.1 L269.5,109.6 M269.5,109.6 L259.9,97.6 M259.9,97.6 L260.1,87.1 M260.1,87.1 L260.4,70.5 M260.4,70.5 L264.2,58.4 M264.2,58.4 L272.3,47.2 M272.3,47.2 L285.4,37.5 M285.4,37.5 L297.6,31.4 M297.6,31.4 L310.2,25.5 M310.2,25.5 L323.6,21.7 M323.6,21.7 L337.1,28.3 M337.1,28.3 L346.2,35.9 M346.2,35.9 L354.9,47.5 M354.9,47.5 L360.7,63.3 M360.7,63.3 L361.8,74.4 M361.8,74.4 L358.9,92.4 M12,135.7 L12,135.7 M12,135.7 L16,149.6 M16,149.6 L16,163.4 M16,163.4 L12,175.3 M34.2,191.9 L25.2,179.2 M25.2,179.2 L21.2,165.5 M21.2,165.5 L20.8,156.1 M20.8,156.1 L22.5,141.7 M22.5,141.7 L32.1,124 M32.1,124 L41.2,117 M41.2,117 L55.9,108.9 M55.9,108.9 L66.9,106.8 M66.9,106.8 L80.9,113.8 M80.9,113.8 L94.1,121.2 M94.1,121.2 L103.6,131.6 M103.6,131.6 L109.6,146.2 M109.6,146.2 L110.8,157 M110.8,157 L108.2,175.9 M108.2,175.9 L99.5,188.9 M99.5,188.9 L91.7,195.7 M91.7,195.7 L77,203 M77,203 L60.6,205.2 M60.6,205.2 L49.7,202.4 M49.7,202.4 L34.2,191.9 M136.9,196 L126.2,186.2 M126.2,186.2 L118.2,171.4 M118.2,171.4 L115.7,160.2 M115.7,160.2 L117.4,139.9 M117.4,139.9 L123.8,127.4 M123.8,127.4 L134.4,118.1 M134.4,118.1 L148.6,111.5 M148.6,111.5 L161.6,109.5 M161.6,109.5 L174.7,111.2 M174.7,111.2 L189.5,117.2 M189.5,117.2 L201,125.1 M201,125.1 L211.9,139.1 M211.9,139.1 L215.3,152.2 M215.3,152.2 L214.5,163.3 M214.5,163.3 L208.6,181.2 M208.6,181.2 L200.4,192 M200.4,192 L188.5,200.9 M188.5,200.9 L174.6,206.3 M174.6,206.3 L162.6,206.7 M162.6,206.7 L148.4,202.9 M148.4,202.9 L136.9,196 M322,169.4 L319,183.2 M319,183.2 L313.1,195 M313.1,195 L303.5,206.1 M303.5,206.1 L291.6,214.1 M291.6,214.1 L280.4,218.2 M280.4,218.2 L265,216.6 M265,216.6 L256.3,211.3 M256.3,211.3 L242.8,202.6 M242.8,202.6 L231.3,193 M231.3,193 L224.6,183.8 M224.6,183.8 L219.5,168.3 M219.5,168.3 L220.1,152.3 M220.1,152.3 L221.2,138.1 M221.2,138.1 L226.8,128.6 M226.8,128.6 L242.7,118.7 M242.7,118.7 L253.5,112.2 M253.5,112.2 L264.8,113.1 M264.8,113.1 L279.9,119.4 M279.9,119.4 L292.5,124.9 M292.5,124.9 L305.3,130.8 M305.3,130.8 L315.6,138.6 M315.6,138.6 L321.4,155.7 M321.4,155.7 L322,169.4 M12,196.9 L18.4,195.8 M18.4,195.8 L33.5,198.7 M33.5,198.7 L43.1,204.4 M43.1,204.4 L56.5,219.7 M56.5,219.7 L60.6,230.3 M60.6,230.3 L62.2,245.6 M62.2,245.6 L59.7,260.6 M59.7,260.6 L53,272 M53,272 L42,281.1 M42,281.1 L26.4,288.2 M26.4,288.2 L12.5,290.6 M12.5,290.6 L12,290.5 M83.3,279 L75.3,270.6 M75.3,270.6 L69.6,259.5 M69.6,259.5 L67,244 M67,244 L68.2,231 M68.2,231 L73.9,217.2 M73.9,217.2 L86.5,204.4 M86.5,204.4 L96.9,198.4 M96.9,198.4 L114,195.2 M114,195.2 L127.1,197.4 M127.1,197.4 L136.2,201.2 M136.2,201.2 L152.3,213.4 M152.3,213.4 L160.2,224.9 M160.2,224.9 L162.7,233.4 M162.7,233.4 L162.1,252.7 M162.1,252.7 L158,263.2 M158,263.2 L149,274.7 M149,274.7 L135,284.9 M135,284.9 L126,288.7 M126,288.7 L110.2,290.9 M110.2,290.9 L97.8,287.9 M97.8,287.9 L83.3,279 M270,248.5 L267.2,261.3 M267.2,261.3 L259.7,271.1 M259.7,271.1 L244.6,280.4 M244.6,280.4 L232.9,286.7 M232.9,286.7 L220.1,289.4 M220.1,289.4 L208.9,288.5 M208.9,288.5 L192.5,282 M192.5,282 L181.6,274.3 M181.6,274.3 L173.4,264.6 M173.4,264.6 L168.6,252.1 M168.6,252.1 L169,234.8 M169,234.8 L172.9,223.6 M172.9,223.6 L183.1,210.8 M183.1,210.8 L196.2,201.8 M196.2,201.8 L208.6,198 M208.6,198 L221.8,198.3 M221.8,198.3 L237,204.6 M237,204.6 L248.3,211.9 M248.3,211.9 L262.2,221.3 M262.2,221.3 L268.8,231.6 M268.8,231.6 L270,248.5 M22.7,306 L28.2,296.9 M28.2,296.9 L37.7,288.9 M37.7,288.9 L55.4,281.6 M55.4,281.6 L64.5,280.8 M64.5,280.8 L78.5,283.4 M78.5,283.4 L93.7,291.5 M93.7,291.5 L105.4,298.7 M105.4,298.7 L108.7,306 M121.8,306 L124.2,301.3 M124.2,301.3 L132.4,292.9 M132.4,292.9 L143.4,285.3 M143.4,285.3 L158,279.8 M158,279.8 L174.5,280.2 M174.5,280.2 L185.9,283.8 M185.9,283.8 L197,290.1 M197,290.1 L209.1,303.3 M209.1,303.3 L210.3,306 M225.7,306 L234.5,293.6 M234.5,293.6 L244.7,286 M244.7,286 L255,279.9 M255,279.9 L271.5,272.7 M271.5,272.7 L286.3,276.1 M286.3,276.1 L295.9,280.7 M295.9,280.7 L306,288.6 M306,288.6 L316.4,302.7 M316.4,302.7 L317.6,306" stroke="currentColor" stroke-width="1.3" fill="none" stroke-linecap="round"></path>
<g fill="currentColor" stroke="none" opacity="0.85">
<circle cx="33.3" cy="26.2" r="1.8"></circle>
<circle cx="23.9" cy="16.6" r="1.8"></circle>
<circle cx="97.6" cy="20.6" r="1.8"></circle>
<circle cx="87.7" cy="28.8" r="1.8"></circle>
<circle cx="71.9" cy="36.4" r="1.8"></circle>
<circle cx="62.5" cy="37.3" r="1.8"></circle>
<circle cx="47.4" cy="34.4" r="1.8"></circle>
<circle cx="133.9" cy="22" r="1.8"></circle>
<circle cx="125.2" cy="14.2" r="1.8"></circle>
<circle cx="202.2" cy="18" r="1.8"></circle>
<circle cx="187.9" cy="27.7" r="1.8"></circle>
<circle cx="172.9" cy="32.8" r="1.8"></circle>
<circle cx="161.9" cy="33.2" r="1.8"></circle>
<circle cx="148.2" cy="30" r="1.8"></circle>
<circle cx="307.9" cy="20.6" r="1.8"></circle>
<circle cx="296.3" cy="26.7" r="1.8"></circle>
<circle cx="282.9" cy="33.3" r="1.8"></circle>
<circle cx="268.4" cy="37.6" r="1.8"></circle>
<circle cx="256.5" cy="37.9" r="1.8"></circle>
<circle cx="240.9" cy="32.1" r="1.8"></circle>
<circle cx="229.8" cy="22.2" r="1.8"></circle>
<circle cx="224.7" cy="14.7" r="1.8"></circle>
<circle cx="48" cy="107.5" r="1.8"></circle>
<circle cx="39.4" cy="112.6" r="1.8"></circle>
<circle cx="22.3" cy="122.3" r="1.8"></circle>
<circle cx="21.9" cy="27.1" r="1.8"></circle>
<circle cx="37.9" cy="34.7" r="1.8"></circle>
<circle cx="45.3" cy="40.1" r="1.8"></circle>
<circle cx="55.4" cy="52" r="1.8"></circle>
<circle cx="61.2" cy="67.7" r="1.8"></circle>
<circle cx="61.1" cy="81" r="1.8"></circle>
<circle cx="61.1" cy="94.1" r="1.8"></circle>
<circle cx="94.2" cy="115.5" r="1.8"></circle>
<circle cx="82" cy="108.9" r="1.8"></circle>
<circle cx="70.1" cy="102.1" r="1.8"></circle>
<circle cx="65.9" cy="87.5" r="1.8"></circle>
<circle cx="66" cy="74.9" r="1.8"></circle>
<circle cx="67.5" cy="62.3" r="1.8"></circle>
<circle cx="73.6" cy="48.9" r="1.8"></circle>
<circle cx="85.3" cy="36.2" r="1.8"></circle>
<circle cx="97.6" cy="28.2" r="1.8"></circle>
<circle cx="107" cy="24.6" r="1.8"></circle>
<circle cx="125.2" cy="24.7" r="1.8"></circle>
<circle cx="139" cy="30.7" r="1.8"></circle>
<circle cx="148" cy="37.8" r="1.8"></circle>
<circle cx="156.6" cy="49.6" r="1.8"></circle>
<circle cx="160.1" cy="61.1" r="1.8"></circle>
<circle cx="160.6" cy="77.5" r="1.8"></circle>
<circle cx="156.6" cy="91.1" r="1.8"></circle>
<circle cx="147.8" cy="103.1" r="1.8"></circle>
<circle cx="139.5" cy="109.8" r="1.8"></circle>
<circle cx="120.9" cy="117.8" r="1.8"></circle>
<circle cx="109.4" cy="118.9" r="1.8"></circle>
<circle cx="241.5" cy="113.8" r="1.8"></circle>
<circle cx="230.6" cy="120.5" r="1.8"></circle>
<circle cx="218.4" cy="126.6" r="1.8"></circle>
<circle cx="204.5" cy="119.8" r="1.8"></circle>
<circle cx="195.3" cy="114.9" r="1.8"></circle>
<circle cx="180.8" cy="105.7" r="1.8"></circle>
<circle cx="172" cy="95.6" r="1.8"></circle>
<circle cx="165.7" cy="79.3" r="1.8"></circle>
<circle cx="165.6" cy="65.2" r="1.8"></circle>
<circle cx="168.2" cy="55.4" r="1.8"></circle>
<circle cx="176.1" cy="42.1" r="1.8"></circle>
<circle cx="185.3" cy="34.3" r="1.8"></circle>
<circle cx="200.1" cy="27.8" r="1.8"></circle>
<circle cx="216.6" cy="26.2" r="1.8"></circle>
<circle cx="230.6" cy="30.5" r="1.8"></circle>
<circle cx="240" cy="37.4" r="1.8"></circle>
<circle cx="248.6" cy="47.5" r="1.8"></circle>
<circle cx="253.7" cy="59.9" r="1.8"></circle>
<circle cx="255.5" cy="76.7" r="1.8"></circle>
<circle cx="255.1" cy="93" r="1.8"></circle>
<circle cx="254.8" cy="103.9" r="1.8"></circle>
<circle cx="358.9" cy="92.4" r="1.8"></circle>
<circle cx="355.1" cy="101.1" r="1.8"></circle>
<circle cx="346" cy="113.4" r="1.8"></circle>
<circle cx="334.7" cy="122.3" r="1.8"></circle>
<circle cx="318.8" cy="128.6" r="1.8"></circle>
<circle cx="307.5" cy="126.1" r="1.8"></circle>
<circle cx="294.2" cy="120.4" r="1.8"></circle>
<circle cx="279.6" cy="114.1" r="1.8"></circle>
<circle cx="269.5" cy="109.6" r="1.8"></circle>
<circle cx="259.9" cy="97.6" r="1.8"></circle>
<circle cx="260.1" cy="87.1" r="1.8"></circle>
<circle cx="260.4" cy="70.5" r="1.8"></circle>
<circle cx="264.2" cy="58.4" r="1.8"></circle>
<circle cx="272.3" cy="47.2" r="1.8"></circle>
<circle cx="285.4" cy="37.5" r="1.8"></circle>
<circle cx="297.6" cy="31.4" r="1.8"></circle>
<circle cx="310.2" cy="25.5" r="1.8"></circle>
<circle cx="323.6" cy="21.7" r="1.8"></circle>
<circle cx="337.1" cy="28.3" r="1.8"></circle>
<circle cx="346.2" cy="35.9" r="1.8"></circle>
<circle cx="354.9" cy="47.5" r="1.8"></circle>
<circle cx="360.7" cy="63.3" r="1.8"></circle>
<circle cx="361.8" cy="74.4" r="1.8"></circle>
<circle cx="16" cy="149.6" r="1.8"></circle>
<circle cx="16" cy="163.4" r="1.8"></circle>
<circle cx="34.2" cy="191.9" r="1.8"></circle>
<circle cx="25.2" cy="179.2" r="1.8"></circle>
<circle cx="21.2" cy="165.5" r="1.8"></circle>
<circle cx="20.8" cy="156.1" r="1.8"></circle>
<circle cx="22.5" cy="141.7" r="1.8"></circle>
<circle cx="32.1" cy="124" r="1.8"></circle>
<circle cx="41.2" cy="117" r="1.8"></circle>
<circle cx="55.9" cy="108.9" r="1.8"></circle>
<circle cx="66.9" cy="106.8" r="1.8"></circle>
<circle cx="80.9" cy="113.8" r="1.8"></circle>
<circle cx="94.1" cy="121.2" r="1.8"></circle>
<circle cx="103.6" cy="131.6" r="1.8"></circle>
<circle cx="109.6" cy="146.2" r="1.8"></circle>
<circle cx="110.8" cy="157" r="1.8"></circle>
<circle cx="108.2" cy="175.9" r="1.8"></circle>
<circle cx="99.5" cy="188.9" r="1.8"></circle>
<circle cx="91.7" cy="195.7" r="1.8"></circle>
<circle cx="77" cy="203" r="1.8"></circle>
<circle cx="60.6" cy="205.2" r="1.8"></circle>
<circle cx="49.7" cy="202.4" r="1.8"></circle>
<circle cx="136.9" cy="196" r="1.8"></circle>
<circle cx="126.2" cy="186.2" r="1.8"></circle>
<circle cx="118.2" cy="171.4" r="1.8"></circle>
<circle cx="115.7" cy="160.2" r="1.8"></circle>
<circle cx="117.4" cy="139.9" r="1.8"></circle>
<circle cx="123.8" cy="127.4" r="1.8"></circle>
<circle cx="134.4" cy="118.1" r="1.8"></circle>
<circle cx="148.6" cy="111.5" r="1.8"></circle>
<circle cx="161.6" cy="109.5" r="1.8"></circle>
<circle cx="174.7" cy="111.2" r="1.8"></circle>
<circle cx="189.5" cy="117.2" r="1.8"></circle>
<circle cx="201" cy="125.1" r="1.8"></circle>
<circle cx="211.9" cy="139.1" r="1.8"></circle>
<circle cx="215.3" cy="152.2" r="1.8"></circle>
<circle cx="214.5" cy="163.3" r="1.8"></circle>
<circle cx="208.6" cy="181.2" r="1.8"></circle>
<circle cx="200.4" cy="192" r="1.8"></circle>
<circle cx="188.5" cy="200.9" r="1.8"></circle>
<circle cx="174.6" cy="206.3" r="1.8"></circle>
<circle cx="162.6" cy="206.7" r="1.8"></circle>
<circle cx="148.4" cy="202.9" r="1.8"></circle>
<circle cx="322" cy="169.4" r="1.8"></circle>
<circle cx="319" cy="183.2" r="1.8"></circle>
<circle cx="313.1" cy="195" r="1.8"></circle>
<circle cx="303.5" cy="206.1" r="1.8"></circle>
<circle cx="291.6" cy="214.1" r="1.8"></circle>
<circle cx="280.4" cy="218.2" r="1.8"></circle>
<circle cx="265" cy="216.6" r="1.8"></circle>
<circle cx="256.3" cy="211.3" r="1.8"></circle>
<circle cx="242.8" cy="202.6" r="1.8"></circle>
<circle cx="231.3" cy="193" r="1.8"></circle>
<circle cx="224.6" cy="183.8" r="1.8"></circle>
<circle cx="219.5" cy="168.3" r="1.8"></circle>
<circle cx="220.1" cy="152.3" r="1.8"></circle>
<circle cx="221.2" cy="138.1" r="1.8"></circle>
<circle cx="226.8" cy="128.6" r="1.8"></circle>
<circle cx="242.7" cy="118.7" r="1.8"></circle>
<circle cx="253.5" cy="112.2" r="1.8"></circle>
<circle cx="264.8" cy="113.1" r="1.8"></circle>
<circle cx="279.9" cy="119.4" r="1.8"></circle>
<circle cx="292.5" cy="124.9" r="1.8"></circle>
<circle cx="305.3" cy="130.8" r="1.8"></circle>
<circle cx="315.6" cy="138.6" r="1.8"></circle>
<circle cx="321.4" cy="155.7" r="1.8"></circle>
<circle cx="18.4" cy="195.8" r="1.8"></circle>
<circle cx="33.5" cy="198.7" r="1.8"></circle>
<circle cx="43.1" cy="204.4" r="1.8"></circle>
<circle cx="56.5" cy="219.7" r="1.8"></circle>
<circle cx="60.6" cy="230.3" r="1.8"></circle>
<circle cx="62.2" cy="245.6" r="1.8"></circle>
<circle cx="59.7" cy="260.6" r="1.8"></circle>
<circle cx="53" cy="272" r="1.8"></circle>
<circle cx="42" cy="281.1" r="1.8"></circle>
<circle cx="26.4" cy="288.2" r="1.8"></circle>
<circle cx="83.3" cy="279" r="1.8"></circle>
<circle cx="75.3" cy="270.6" r="1.8"></circle>
<circle cx="69.6" cy="259.5" r="1.8"></circle>
<circle cx="67" cy="244" r="1.8"></circle>
<circle cx="68.2" cy="231" r="1.8"></circle>
<circle cx="73.9" cy="217.2" r="1.8"></circle>
<circle cx="86.5" cy="204.4" r="1.8"></circle>
<circle cx="96.9" cy="198.4" r="1.8"></circle>
<circle cx="114" cy="195.2" r="1.8"></circle>
<circle cx="127.1" cy="197.4" r="1.8"></circle>
<circle cx="136.2" cy="201.2" r="1.8"></circle>
<circle cx="152.3" cy="213.4" r="1.8"></circle>
<circle cx="160.2" cy="224.9" r="1.8"></circle>
<circle cx="162.7" cy="233.4" r="1.8"></circle>
<circle cx="162.1" cy="252.7" r="1.8"></circle>
<circle cx="158" cy="263.2" r="1.8"></circle>
<circle cx="149" cy="274.7" r="1.8"></circle>
<circle cx="135" cy="284.9" r="1.8"></circle>
<circle cx="126" cy="288.7" r="1.8"></circle>
<circle cx="110.2" cy="290.9" r="1.8"></circle>
<circle cx="97.8" cy="287.9" r="1.8"></circle>
<circle cx="270" cy="248.5" r="1.8"></circle>
<circle cx="267.2" cy="261.3" r="1.8"></circle>
<circle cx="259.7" cy="271.1" r="1.8"></circle>
<circle cx="244.6" cy="280.4" r="1.8"></circle>
<circle cx="232.9" cy="286.7" r="1.8"></circle>
<circle cx="220.1" cy="289.4" r="1.8"></circle>
<circle cx="208.9" cy="288.5" r="1.8"></circle>
<circle cx="192.5" cy="282" r="1.8"></circle>
<circle cx="181.6" cy="274.3" r="1.8"></circle>
<circle cx="173.4" cy="264.6" r="1.8"></circle>
<circle cx="168.6" cy="252.1" r="1.8"></circle>
<circle cx="169" cy="234.8" r="1.8"></circle>
<circle cx="172.9" cy="223.6" r="1.8"></circle>
<circle cx="183.1" cy="210.8" r="1.8"></circle>
<circle cx="196.2" cy="201.8" r="1.8"></circle>
<circle cx="208.6" cy="198" r="1.8"></circle>
<circle cx="221.8" cy="198.3" r="1.8"></circle>
<circle cx="237" cy="204.6" r="1.8"></circle>
<circle cx="248.3" cy="211.9" r="1.8"></circle>
<circle cx="262.2" cy="221.3" r="1.8"></circle>
<circle cx="268.8" cy="231.6" r="1.8"></circle>
<circle cx="28.2" cy="296.9" r="1.8"></circle>
<circle cx="37.7" cy="288.9" r="1.8"></circle>
<circle cx="55.4" cy="281.6" r="1.8"></circle>
<circle cx="64.5" cy="280.8" r="1.8"></circle>
<circle cx="78.5" cy="283.4" r="1.8"></circle>
<circle cx="93.7" cy="291.5" r="1.8"></circle>
<circle cx="105.4" cy="298.7" r="1.8"></circle>
<circle cx="124.2" cy="301.3" r="1.8"></circle>
<circle cx="132.4" cy="292.9" r="1.8"></circle>
<circle cx="143.4" cy="285.3" r="1.8"></circle>
<circle cx="158" cy="279.8" r="1.8"></circle>
<circle cx="174.5" cy="280.2" r="1.8"></circle>
<circle cx="185.9" cy="283.8" r="1.8"></circle>
<circle cx="197" cy="290.1" r="1.8"></circle>
<circle cx="209.1" cy="303.3" r="1.8"></circle>
<circle cx="234.5" cy="293.6" r="1.8"></circle>
<circle cx="244.7" cy="286" r="1.8"></circle>
<circle cx="255" cy="279.9" r="1.8"></circle>
<circle cx="271.5" cy="272.7" r="1.8"></circle>
<circle cx="286.3" cy="276.1" r="1.8"></circle>
<circle cx="295.9" cy="280.7" r="1.8"></circle>
<circle cx="306" cy="288.6" r="1.8"></circle>
<circle cx="316.4" cy="302.7" r="1.8"></circle>
</g>
<g style="color:var(--global-accent, currentColor)">
<path d="M59.7,260.6 L53,272 M37.7,288.9 L55.4,281.6 M55.4,281.6 L64.5,280.8 M64.5,280.8 L78.5,283.4 M78.5,283.4 L93.7,291.5 M93.7,291.5 L105.4,298.7" stroke="currentColor" stroke-width="3.4" fill="none" stroke-linecap="round"></path>
<rect x="54" y="264" width="42" height="42" fill="none" stroke="currentColor" stroke-width="2"></rect>
<rect x="306" y="138" width="42" height="42" fill="none" stroke="currentColor" stroke-width="2"></rect>
<circle cx="83.3" cy="279" r="5" fill="currentColor" stroke="none"></circle>
<circle cx="322" cy="169.4" r="5" fill="currentColor" stroke="none"></circle>
<path d="M348,159 L484,159 M75,306 L75,326 L484,326" stroke="currentColor" stroke-width="1.2" fill="none"></path>
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="490" y="144" font-size="16" font-weight="700">a node on the</text>
<text x="490" y="164" font-size="16" font-weight="700">free surface</text>
<text x="490" y="186" font-size="14">few or no faces of other</text>
<text x="490" y="204" font-size="14">cells in its voxel</text>
<text x="490" y="331" font-size="16" font-weight="700">a node deep inside</text>
<text x="490" y="352" font-size="14">many faces of other cells</text>
<text x="490" y="370" font-size="14">in its voxel (drawn thick)</text>
</g>
</g>
<path d="M484,33 L453,33" stroke="currentColor" stroke-width="1" fill="none" opacity="0.6"></path>
<circle cx="453" cy="33" r="2.2" fill="currentColor" stroke="none" opacity="0.75"></circle>
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="490" y="38" font-size="14" opacity="0.78">one voxel</text>
<text x="12" y="352" font-size="14" opacity="0.78">2D slice: a face shows as a line segment</text>
</g>
</svg>
<figcaption>A 2D slice through the edge of a tissue, on the voxel grid.</figcaption>
</figure>

The work for one node is the number of other cells' faces in its voxel. As I read the code, a node on the free surface of the tissue finds few, and one deep inside, with neighbours packed in on all sides, finds many. I haven't counted faces per voxel in a run.

Each voxel needs a container for its faces, and in v1.0 that's a linked list. A linked list keeps each entry in its own small allocation with a pointer to the next, so walking a voxel means hopping around memory, and the processor can't fetch an entry until it has read the pointer to it. A `std::vector` keeps the entries side by side in one block that the processor can read straight through.

## where half the cores go

I first noticed the idle cores by watching `top` while a run was going. The benchmark later put a number on it : v1.0 with two patches (a build I'll get to) keeps about half of the eight cores busy.

So where does the other half go?

Start with what `#pragma omp parallel for` does when it's given nothing else. OpenMP lets you choose how a loop's work is dealt out by adding a schedule clause to the pragma, and the loops in v1.0 don't have one. With no clause, GCC and Clang cut the index range into one contiguous block per thread. With 8 threads the first thread gets the first eighth of the cells, the second gets the next eighth, and so on. This is the static schedule. It costs nothing to set up, and it's the right choice when the cells all cost about the same.

When the loop ends, every thread waits until all of them are done, so the loop takes as long as its slowest block.

But cells don't cost the same, in face count or in how many neighbours' faces their nodes have to test. If the expensive cells sit together in the list, the block that holds them keeps its thread busy long after the others have finished, and the finished ones wait.

There's a catch in that "if". Blocks only come out uneven when the expensive cells are bunched together along the list. Scattered at random through a block of thousands, they'd average out. The usual single number for unevenness is the coefficient of variation (the standard deviation of the per-cell cost over its mean), and it can't tell those two cases apart. I haven't measured how cost is laid out along the cell list in a real run.

The other schedule kinds deal work out while the loop is running. Under `dynamic`, a thread takes a small batch of cells called a chunk, finishes it and comes back for another, so a thread that drew cheap cells just takes more of them, at the price of some bookkeeping per chunk. `guided` does the same with chunks that start large and shrink as the loop runs down, which saves part of that bookkeeping.

<figure>
<svg viewBox="0 0 720 484" role="img" aria-labelledby="fig-threads-t fig-threads-d" style="width:100%;height:auto;max-width:720px;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title id="fig-threads-t">Static blocks against dynamic scheduling</title>
<desc id="fig-threads-d">Two timelines of the same loop over cells on four threads. With static blocks one thread gets all the expensive cells and runs long while the other three finish early and wait. With dynamic scheduling the expensive cells are spread over the threads and all four finish close together, sooner.</desc>
<g stroke="currentColor" stroke-width="1.1">
<rect x="98" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="121" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="144" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="167" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="190" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="213" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="236" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="259" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="282" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="305" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="328" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="351" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="374" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="397" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="420" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="443" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="466" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="489" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="512" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="535" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="558" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="581" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="604" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="627" y="36" width="19" height="19" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
</g>
<path d="M235,59 L235,64 L371,64 L371,59" stroke="currentColor" stroke-width="1" fill="none" opacity="0.6"></path>
<g stroke="currentColor" stroke-width="1.1">
<rect x="98.8" y="130" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="130" y="130" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="176.7" y="130" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="207.9" y="130" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="239.1" y="130" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="285.9" y="130" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
</g>
<g stroke="currentColor" stroke-width="1.1">
<rect x="98.8" y="159" width="91.9" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="192.3" y="159" width="107.5" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="301.5" y="159" width="76.3" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="379.4" y="159" width="123.1" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="504.1" y="159" width="91.9" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="597.7" y="159" width="107.5" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
</g>
<g stroke="currentColor" stroke-width="1.1">
<rect x="98.8" y="188" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="145.6" y="188" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="176.7" y="188" width="60.8" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="239.1" y="188" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="285.9" y="188" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="317.1" y="188" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
</g>
<g stroke="currentColor" stroke-width="1.1">
<rect x="98.8" y="217" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="130" y="217" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="176.7" y="217" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="207.9" y="217" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="239.1" y="217" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="285.9" y="217" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
</g>
<g style="color:var(--global-accent, currentColor)">
<rect x="317.1" y="130" width="388.9" height="22" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-width="1"></rect>
<path d="M317.1,133.5 L320.6,130 M317.1,140.5 L327.6,130 M317.1,147.5 L334.6,130 M319.6,152 L341.6,130 M326.6,152 L348.6,130 M333.6,152 L355.6,130 M340.6,152 L362.6,130 M347.6,152 L369.6,130 M354.6,152 L376.6,130 M361.6,152 L383.6,130 M368.6,152 L390.6,130 M375.6,152 L397.6,130 M382.6,152 L404.6,130 M389.6,152 L411.6,130 M396.6,152 L418.6,130 M403.6,152 L425.6,130 M410.6,152 L432.6,130 M417.6,152 L439.6,130 M424.6,152 L446.6,130 M431.6,152 L453.6,130 M438.6,152 L460.6,130 M445.6,152 L467.6,130 M452.6,152 L474.6,130 M459.6,152 L481.6,130 M466.6,152 L488.6,130 M473.6,152 L495.6,130 M480.6,152 L502.6,130 M487.6,152 L509.6,130 M494.6,152 L516.6,130 M501.6,152 L523.6,130 M508.6,152 L530.6,130 M515.6,152 L537.6,130 M522.6,152 L544.6,130 M529.6,152 L551.6,130 M536.6,152 L558.6,130 M543.6,152 L565.6,130 M550.6,152 L572.6,130 M557.6,152 L579.6,130 M564.6,152 L586.6,130 M571.6,152 L593.6,130 M578.6,152 L600.6,130 M585.6,152 L607.6,130 M592.6,152 L614.6,130 M599.6,152 L621.6,130 M606.6,152 L628.6,130 M613.6,152 L635.6,130 M620.6,152 L642.6,130 M627.6,152 L649.6,130 M634.6,152 L656.6,130 M641.6,152 L663.6,130 M648.6,152 L670.6,130 M655.6,152 L677.6,130 M662.6,152 L684.6,130 M669.6,152 L691.6,130 M676.6,152 L698.6,130 M683.6,152 L705.6,130 M690.6,152 L706,136.6 M697.6,152 L706,143.6 M704.6,152 L706,150.6" stroke="currentColor" stroke-width="1" fill="none"></path>
<rect x="363.8" y="188" width="342.2" height="22" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-width="1"></rect>
<path d="M363.8,191.5 L367.3,188 M363.8,198.5 L374.3,188 M363.8,205.5 L381.3,188 M366.3,210 L388.3,188 M373.3,210 L395.3,188 M380.3,210 L402.3,188 M387.3,210 L409.3,188 M394.3,210 L416.3,188 M401.3,210 L423.3,188 M408.3,210 L430.3,188 M415.3,210 L437.3,188 M422.3,210 L444.3,188 M429.3,210 L451.3,188 M436.3,210 L458.3,188 M443.3,210 L465.3,188 M450.3,210 L472.3,188 M457.3,210 L479.3,188 M464.3,210 L486.3,188 M471.3,210 L493.3,188 M478.3,210 L500.3,188 M485.3,210 L507.3,188 M492.3,210 L514.3,188 M499.3,210 L521.3,188 M506.3,210 L528.3,188 M513.3,210 L535.3,188 M520.3,210 L542.3,188 M527.3,210 L549.3,188 M534.3,210 L556.3,188 M541.3,210 L563.3,188 M548.3,210 L570.3,188 M555.3,210 L577.3,188 M562.3,210 L584.3,188 M569.3,210 L591.3,188 M576.3,210 L598.3,188 M583.3,210 L605.3,188 M590.3,210 L612.3,188 M597.3,210 L619.3,188 M604.3,210 L626.3,188 M611.3,210 L633.3,188 M618.3,210 L640.3,188 M625.3,210 L647.3,188 M632.3,210 L654.3,188 M639.3,210 L661.3,188 M646.3,210 L668.3,188 M653.3,210 L675.3,188 M660.3,210 L682.3,188 M667.3,210 L689.3,188 M674.3,210 L696.3,188 M681.3,210 L703.3,188 M688.3,210 L706,192.3 M695.3,210 L706,199.3 M702.3,210 L706,206.3" stroke="currentColor" stroke-width="1" fill="none"></path>
<rect x="317.1" y="217" width="388.9" height="22" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-width="1"></rect>
<path d="M317.1,220.5 L320.6,217 M317.1,227.5 L327.6,217 M317.1,234.5 L334.6,217 M319.6,239 L341.6,217 M326.6,239 L348.6,217 M333.6,239 L355.6,217 M340.6,239 L362.6,217 M347.6,239 L369.6,217 M354.6,239 L376.6,217 M361.6,239 L383.6,217 M368.6,239 L390.6,217 M375.6,239 L397.6,217 M382.6,239 L404.6,217 M389.6,239 L411.6,217 M396.6,239 L418.6,217 M403.6,239 L425.6,217 M410.6,239 L432.6,217 M417.6,239 L439.6,217 M424.6,239 L446.6,217 M431.6,239 L453.6,217 M438.6,239 L460.6,217 M445.6,239 L467.6,217 M452.6,239 L474.6,217 M459.6,239 L481.6,217 M466.6,239 L488.6,217 M473.6,239 L495.6,217 M480.6,239 L502.6,217 M487.6,239 L509.6,217 M494.6,239 L516.6,217 M501.6,239 L523.6,217 M508.6,239 L530.6,217 M515.6,239 L537.6,217 M522.6,239 L544.6,217 M529.6,239 L551.6,217 M536.6,239 L558.6,217 M543.6,239 L565.6,217 M550.6,239 L572.6,217 M557.6,239 L579.6,217 M564.6,239 L586.6,217 M571.6,239 L593.6,217 M578.6,239 L600.6,217 M585.6,239 L607.6,217 M592.6,239 L614.6,217 M599.6,239 L621.6,217 M606.6,239 L628.6,217 M613.6,239 L635.6,217 M620.6,239 L642.6,217 M627.6,239 L649.6,217 M634.6,239 L656.6,217 M641.6,239 L663.6,217 M648.6,239 L670.6,217 M655.6,239 L677.6,217 M662.6,239 L684.6,217 M669.6,239 L691.6,217 M676.6,239 L698.6,217 M683.6,239 L705.6,217 M690.6,239 L706,223.6 M697.6,239 L706,230.6 M704.6,239 L706,237.6" stroke="currentColor" stroke-width="1" fill="none"></path>
</g>
<path d="M706.5,124 L706.5,245" stroke="currentColor" stroke-width="1.6" fill="none"></path>
<g stroke="currentColor" stroke-width="1.1">
<rect x="98.8" y="315" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="130" y="315" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="176.7" y="315" width="123.1" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="301.5" y="315" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="332.6" y="315" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="363.8" y="315" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="395" y="315" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
</g>
<g stroke="currentColor" stroke-width="1.1">
<rect x="98.8" y="344" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="145.6" y="344" width="107.5" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="254.7" y="344" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="301.5" y="344" width="60.8" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="363.8" y="344" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
</g>
<g stroke="currentColor" stroke-width="1.1">
<rect x="98.8" y="373" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="130" y="373" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="161.2" y="373" width="76.3" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="239.1" y="373" width="107.5" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="348.2" y="373" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="395" y="373" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
</g>
<g stroke="currentColor" stroke-width="1.1">
<rect x="98.8" y="402" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="130" y="402" width="91.9" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="223.5" y="402" width="91.9" height="22" rx="2.5" fill="currentColor" fill-opacity="0.26"></rect>
<rect x="317.1" y="402" width="45.2" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="363.8" y="402" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="395" y="402" width="29.6" height="22" rx="2.5" fill="currentColor" fill-opacity="0.06"></rect>
</g>
<g style="color:var(--global-accent, currentColor)">
<rect x="426.2" y="315" width="14.8" height="22" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-width="1"></rect>
<path d="M426.2,318.5 L429.7,315 M426.2,325.5 L436.7,315 M426.2,332.5 L441,317.7 M428.7,337 L441,324.7 M435.7,337 L441,331.7" stroke="currentColor" stroke-width="1" fill="none"></path>
<rect x="410.6" y="344" width="30.4" height="22" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-width="1"></rect>
<path d="M410.6,347.5 L414.1,344 M410.6,354.5 L421.1,344 M410.6,361.5 L428.1,344 M413.1,366 L435.1,344 M420.1,366 L441,345.1 M427.1,366 L441,352.1 M434.1,366 L441,359.1" stroke="currentColor" stroke-width="1" fill="none"></path>
<rect x="426.2" y="402" width="14.8" height="22" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-width="1"></rect>
<path d="M426.2,405.5 L429.7,402 M426.2,412.5 L436.7,402 M426.2,419.5 L441,404.7 M428.7,424 L441,411.7 M435.7,424 L441,418.7" stroke="currentColor" stroke-width="1" fill="none"></path>
</g>
<path d="M441.5,309 L441.5,430" stroke="currentColor" stroke-width="1.6" fill="none"></path>
<path d="M706.5,267 L706.5,430" stroke="currentColor" stroke-width="1" fill="none" stroke-dasharray="3 4" opacity="0.45"></path>
<path d="M98,458 L699.5,458" stroke="currentColor" stroke-width="1.1" fill="none" opacity="0.6"></path>
<path d="M706,458 L699,460.9 L699,455.1 z" fill="currentColor" stroke="none" opacity="0.6"></path>
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="14" y="51" font-size="16" font-weight="600">the loop</text>
<text x="98" y="24" font-size="14" opacity="0.78">cells in index order</text>
<text x="303" y="82" font-size="14" text-anchor="middle" opacity="0.78">expensive cells</text>
<text x="14" y="113" font-size="16"><tspan font-weight="700">static:</tspan><tspan dx="8">one contiguous block of cells per thread</tspan></text>
<text x="14" y="298" font-size="16"><tspan font-weight="700">dynamic:</tspan><tspan dx="8">a free thread takes the next cell</tspan></text>
<text x="14" y="146" font-size="14" opacity="0.78">thread 1</text>
<text x="14" y="175" font-size="14" opacity="0.78">thread 2</text>
<text x="14" y="204" font-size="14" opacity="0.78">thread 3</text>
<text x="14" y="233" font-size="14" opacity="0.78">thread 4</text>
<text x="14" y="331" font-size="14" opacity="0.78">thread 1</text>
<text x="14" y="360" font-size="14" opacity="0.78">thread 2</text>
<text x="14" y="389" font-size="14" opacity="0.78">thread 3</text>
<text x="14" y="418" font-size="14" opacity="0.78">thread 4</text>
<text x="698" y="262" font-size="14" text-anchor="end" opacity="0.78">loop ends</text>
<text x="450" y="447" font-size="14" text-anchor="start" opacity="0.78">loop ends</text>
<text x="706" y="477" font-size="14" text-anchor="end" opacity="0.78">time</text>
</g>
<g style="color:var(--global-accent, currentColor)">
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="702" y="113" font-size="16" text-anchor="end" font-weight="600">hatched: waiting</text>
</g>
</g>
</svg>
<figcaption>The same loop under two schedules. Four threads are drawn where the runs use eight, with chunks of one cell. It's a sketch of the mechanism, nothing in it is measured.</figcaption>
</figure>

Schedules can't help with the second place the cores go. During time integration, cell removal and the serial passes inside contact, one core works and seven have nothing to do.

And v1.0 doesn't pin its threads, so the operating system is free to move them between cores, and a thread that gets moved leaves behind whatever it had loaded into that core's cache. Pinning ties each thread to one core for the whole run. (v1.0's default is also one thread per logical CPU, and each of this machine's 8 physical cores shows up as two. I gave every build 8 threads in the benchmark, so that part is the same on both sides.)

I can't tell you how the idle half splits between uneven blocks and the single-thread stretches.

## what changed between v1.0 and v2.1.0

I've made two releases of the fork. v2.0 is where the scheduler arrived, along with the first change to the contact grid. v2.1.0 is the current one, and a lot of the work in it went into keeping a long run going when a cell's mesh goes bad or a division can't be completed.

Three changes are about speed.

| | v1.0 | v2.1.0 | Measured effect |
|---|---|---|---|
| Loop schedules | bare `parallel for`, one block of cells per thread | `schedule(runtime)` on most hot loops, one fixed kind per group of loops in the default mode (exceptions below) | not separated |
| Threads | free to move between cores | each pinned to one physical core (Linux) | not separated |
| Per-voxel container in the contact grid | linked list (`std::forward_list`) | `std::vector`, plus a compact grid and work units for the face-face model | not separated |

All three say "not separated" because I compared whole builds, so I can't tell you what any one of them is worth alone.

The rest is what the fork adds to keep a run going.

| Added in the fork | What it does | Measured effect |
|---|---|---|
| Unstable-cell removal | a cell whose mesh has blown up is removed and logged, and the run carries on | none removed on the growth case at the benchmark's time step |
| Division refusal | if the new triangles don't close the cut, the division is called off, and the mother stays whole and tries again at the next check | about 25,000 refused in the run that passed 150,000 cells (outside the benchmark, see below) |
| Cell ids re-synced | after every removal, so ids keep matching the cell list | ported back to v1.0 together with the next row, it takes that build from around 130 cells to between about 23,000 and 42,000 |
| Winding kept on an edge split | new faces keep the winding of the face they replace, and it's checked after every refinement | see the row above |

Memory isn't in either table because I can't tie it to one change. It comes up in the benchmark.

The contact grid changed in two steps. Linked list to vector happened in v2.0, for every contact model. For the face-face model v2.1.0 goes further : the grid is packed into flat arrays, and a cell that would take much more than its share of the contact work is cut into ranges of nodes, so the loop hands out work units of similar size.

## the adaptive part reaches no loop

In v2.0 I put a `schedule(runtime)` clause on the hot loops. That tells OpenMP to look the schedule up when the loop starts, and the solver sets it by calling `omp_set_schedule()` before each group of loops. You launch a run in one of four modes. Three force a single kind on all of those loops. The fourth, `adaptive`, is the default and gives each group of loops its own kind.

| Loop | What it runs under in v2.1.0, `adaptive` mode |
|---|---|
| Face-type reset, mesh refinement | runtime schedule, set to `static` |
| Contact, node-node and node-face models | runtime schedule, set to `dynamic` |
| Polarization update, internal forces | runtime schedule, set to `guided` |
| Contact, face-face model | written into the source, `static` for the grid build and `dynamic` over work units, one at a time |
| Time integration | one thread, in every version |
| Cell division | bare `parallel for` with no schedule clause, in every version |

Beside that sits the machinery the name comes from. Every 50 iterations a phase detector labels the run `INITIALIZATION` (fewer than 10 cells), `GROWTH` (more than 1% of the cells divided in the last 50 iterations) or `HOMEOSTASIS`. In the same pass a cost estimate gives each cell a modelled cost from its face count and its contacts, and takes the coefficient of variation across cells. Phase, coefficient and cell count together choose what the code calls the global schedule.

I traced which loop runs under which setting. Turns out that in `adaptive` mode none of the hot loops runs under the global schedule. Each group's kind is fixed when the solver starts and set again before the group runs, on every iteration, so whatever the detector and the cost estimate decide lands in a schedule that no hot loop reads. What changed for the loops between v1.0 and v2.1.0 is static blocks becoming a fixed kind per loop, picked by me in advance. The adaptive part doesn't reach them.

<figure>
<svg viewBox="0 0 720 340" role="img" aria-labelledby="fig-scheduler-t fig-scheduler-d" style="width:100%;height:auto;max-width:720px;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title id="fig-scheduler-t">What the scheduler reaches in adaptive mode</title>
<desc id="fig-scheduler-d">On the left, three groups of loops are each wired to a fixed schedule kind: mesh loops to static, contact for the node-node and node-face models to dynamic, polarization and internal forces to guided. On the right, the cell count and the divisions in the last 50 iterations set a phase; the phase, the cost spread and the cell count then set a global schedule, which is drawn dashed and is wired to no loop.</desc>
<rect x="16" y="62" width="196" height="60" rx="8" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="268" y="71" width="104" height="42" rx="8" stroke="currentColor" stroke-width="1.5" fill="none"></rect>
<path d="M215,92 L256.5,92" stroke="currentColor" stroke-width="1.5" fill="none"></path>
<path d="M264,92 L256,95.4 L256,88.6 z" fill="currentColor" stroke="none"></path>
<rect x="16" y="150" width="196" height="60" rx="8" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="268" y="159" width="104" height="42" rx="8" stroke="currentColor" stroke-width="1.5" fill="none"></rect>
<path d="M215,180 L256.5,180" stroke="currentColor" stroke-width="1.5" fill="none"></path>
<path d="M264,180 L256,183.4 L256,176.6 z" fill="currentColor" stroke="none"></path>
<rect x="16" y="238" width="196" height="60" rx="8" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="268" y="247" width="104" height="42" rx="8" stroke="currentColor" stroke-width="1.5" fill="none"></rect>
<path d="M215,268 L256.5,268" stroke="currentColor" stroke-width="1.5" fill="none"></path>
<path d="M264,268 L256,271.4 L256,264.6 z" fill="currentColor" stroke="none"></path>
<rect x="430" y="62" width="104" height="50" rx="8" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="548" y="62" width="158" height="50" rx="8" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="474" y="156" width="100" height="42" rx="8" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06"></rect>
<rect x="592" y="156" width="114" height="42" rx="8" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.06"></rect>
<path d="M504,115 L504,144.5" stroke="currentColor" stroke-width="1.5" fill="none"></path>
<path d="M504,152 L500.6,144 L507.4,144 z" fill="currentColor" stroke="none"></path>
<path d="M560,115 L560,144.5" stroke="currentColor" stroke-width="1.5" fill="none"></path>
<path d="M560,152 L556.6,144 L563.4,144 z" fill="currentColor" stroke="none"></path>
<path d="M524,201 L524,232.5" stroke="currentColor" stroke-width="1.5" fill="none"></path>
<path d="M524,240 L520.6,232 L527.4,232 z" fill="currentColor" stroke="none"></path>
<path d="M649,201 L649,232.5" stroke="currentColor" stroke-width="1.5" fill="none"></path>
<path d="M649,240 L645.6,232 L652.4,232 z" fill="currentColor" stroke="none"></path>
<path d="M448,115 L448,268" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round"></path>
<path d="M448,268 L462.5,268" stroke="currentColor" stroke-width="1.5" fill="none"></path>
<path d="M470,268 L462,271.4 L462,264.6 z" fill="currentColor" stroke="none"></path>
<g style="color:var(--global-accent, currentColor)">
<rect x="474" y="244" width="232" height="48" rx="8" stroke="currentColor" stroke-width="2" fill="currentColor" fill-opacity="0.07" stroke-dasharray="7 5"></rect>
</g>
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="16" y="28" font-size="17" font-weight="700">per-loop schedule</text>
<text x="16" y="48" font-size="14" opacity="0.78">fixed when the solver starts</text>
<text x="430" y="28" font-size="17" font-weight="700">detector</text>
<text x="430" y="48" font-size="14" opacity="0.78">runs every 50 iterations</text>
<text x="114" y="97.5" font-size="16" text-anchor="middle">mesh loops</text>
<text x="320" y="97.5" font-size="16" text-anchor="middle" font-weight="600" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">static</text>
<text x="114" y="177" font-size="16" text-anchor="middle">contact</text>
<text x="114" y="196" font-size="14" text-anchor="middle" opacity="0.78">(node-node, node-face)</text>
<text x="320" y="185.5" font-size="16" text-anchor="middle" font-weight="600" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">dynamic</text>
<text x="114" y="264" font-size="16" text-anchor="middle">polarization +</text>
<text x="114" y="284" font-size="16" text-anchor="middle">internal forces</text>
<text x="320" y="273.5" font-size="16" text-anchor="middle" font-weight="600" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">guided</text>
<text x="482" y="92" font-size="15" text-anchor="middle">cell count</text>
<text x="627" y="83" font-size="14" text-anchor="middle">divisions in the</text>
<text x="627" y="101" font-size="14" text-anchor="middle">last 50 iterations</text>
<text x="524" y="182.5" font-size="16" text-anchor="middle">phase</text>
<text x="649" y="182" font-size="15" text-anchor="middle">cost spread</text>
</g>
<g style="color:var(--global-accent, currentColor)">
<g font-family="-apple-system, system-ui, sans-serif" aria-hidden="true" fill="currentColor" stroke="none">
<text x="590" y="273.5" font-size="16" text-anchor="middle" font-weight="600">global schedule</text>
<text x="590" y="318" font-size="16" text-anchor="middle" font-weight="600">no loop runs under it</text>
</g>
</g>
</svg>
<figcaption>v2.1.0 in adaptive mode.</figcaption>
</figure>

Three rows of that loop table get nothing from the scheduler in any mode. Time integration is on one thread. The division loop has no schedule clause, so whatever the scheduler sets for it never arrives. And the face-face contact loop, which did follow the runtime schedule in v2.0, has its schedules written into the source on its default path in v2.1.0. The growth benchmark uses the node-node model, where the contact loop still takes its kind from the scheduler.

## one machine type, eight builds

I ran everything on GCP c2-standard-16 instances, which have 8 physical cores, with 8 threads. It's about 240 runs across eight builds, named like this.

- **v1.0 as tagged** is the published release, built unchanged.
- **v1.0 with two patches** adds two small fixes ported back from the fork (cell ids re-synced after a removal, and a split edge keeping the winding of the face it replaces).
- **v2.0** is the fork's first release.
- **The four development builds** sit between v2.0 and v2.1.0.
- **v2.1.0** is the fork's current release.

The main case is the lab's own growing-aggregate file from the v1.0 release. It starts from one cell and uses the node-node contact model, and a run goes on until it reaches 150,000 cells or two days (48 hours), unless it stops by itself first. The other cases are small tissues, vesicles and a tube, each run with the paper's parameters (arm A in the charts) and with a second parameter set taken from the literature (arm B).

I ran the growth case at a quarter of the paper's time step, because none of my builds stayed stable on it at the paper's step (more on that at the end). That's four times as many iterations for the same simulated time, so the hours below don't compare with the paper's one day.

I thought 150,000 cells was a day's run, since the paper grows 125,000 in a day. It took two days and more on eight cores. My runs are on a different machine, with 8 threads where the paper used 16, doing four times the iterations.

I also struggled to get v1.0 to run cleanly, and that cost me more time than anything else in this study. Built as tagged, it kept stopping at around 130 cells on the growth case, where the mesh refiner gives up on a cell (130 to 170 across the four time steps I tried). The paper grows 125,000 cells with this code, so it plainly can. Porting two small fixes back from the fork got it going on my machine, and that build is the v1.0 I compare against below.

## 145,000 cells in two days

<figure>
<img src="{{ site.baseurl }}/images/blog/simucell3d/bench2-reach.svg" alt="Bar chart of the final cell count of every run on the growth case, on a log scale, grouped by build." loading="lazy">
<figcaption>Final cell count of every run on the growth case at the benchmark's time step. The axis is logarithmic.</figcaption>
</figure>

v1.0 as tagged stops at around 130 cells. v2.0, my own release, stops sooner, at about 60, where it segfaults within a dozen iterations of the first cell death. The first development build re-syncs cell ids after a removal, and from there on every 2.x build runs to the two-day cap.

The three runs of v1.0 with two patches stopped between about 23,000 and 42,000 cells.

The development builds ended around 140,000 cells (136,000 to 143,000) and v2.1.0 around 145,000 (142,000 to 149,000), still growing when the time ran out. None of them removed an unstable cell at this time step.

None of the release runs got to the 150,000-cell stop at the benchmark's time step, they ran out of time at around 145,000. A run on a build from just before the v2.1.0 tag did go past it, outside this benchmark, on my reconstruction of the paper's configuration with the face-face contact model : just over 150,000 cells in about 53 hours, with 50 cells removed and about 25,000 divisions refused on the way. Its stop rule ended it there, so I don't know how far it would have gone. (At the paper's time step v2.1.0 gets to 150,000 on the growth case in under a day, but it removes over 90,000 unstable cells on the way, and I don't count that.)

<figure>
<img src="{{ site.baseurl }}/images/blog/simucell3d/bench2-growth.svg" alt="Cell count against wall-clock hours for v1.0 with two patches, the development builds and v2.1.0, log cell axis." loading="lazy">
<figcaption>Cells against wall-clock hours, with the cell axis logarithmic.</figcaption>
</figure>

The same runs as hours to reach a given size.

| Cells | v1.0 with two patches | v2.1.0 |
|---|---|---|
| 10,000 | 2½ h | 1¼ h |
| 30,000 | 11 h (one run) | 5½ h |
| 60,000 | stopped before this | 14 h |
| 100,000 | stopped before this | 28 h |
| 125,000 | stopped before this | 38 h |

## about twice as fast on six cases

On the growth case the cleanest comparison is hours to 10,000 cells, the last milestone that every run of both builds reaches : two and a half hours for v1.0 with two patches, an hour and a quarter for v2.1.0. Five other cases were timed on both, vesicles and tubes under the two parameter sets.

<figure>
<img src="{{ site.baseurl }}/images/blog/simucell3d/bench2-matched-cases.svg" alt="Dumbbell chart of wall-clock hours for six cases on v1.0 with two patches and on v2.1.0, with each ratio." loading="lazy">
<figcaption>The six cases timed on both versions. Arm A is the paper's parameter set, arm B the one from the literature.</figcaption>
</figure>

v2.1.0 is roughly twice as fast on all six, between 1.8 and 2.6 times. Most of these timings are single runs, so I wouldn't read much into which case got which.

On the growth case v1.0 with two patches keeps half of the eight cores busy and v2.1.0 a bit over 80%. The single-thread phases are all still there in v2.1.0.

## scaling with cell count

<figure>
<img src="{{ site.baseurl }}/images/blog/simucell3d/bench2-sec-per-iter.svg" alt="Seconds per iteration against cell count on log-log axes for v1.0 with two patches, the development builds and v2.1.0." loading="lazy">
<figcaption>Seconds per iteration against cell count, both axes logarithmic.</figcaption>
</figure>

The cost of one iteration grows a little faster than linearly in the number of cells $N$, about $N^{1.1}$. Both versions sit on about that slope, and v2.1.0 is the lower line.

The total wall-clock time to reach $N$ cells is a different number. It grows as about $N^{1.3}$, which is the quantity the paper reports, with an exponent of 4/3. (If the tissue kept doubling at a fixed pace the two exponents would be the same. They aren't, so by my arithmetic each doubling takes more iterations than the one before.)

## what the factor of two includes

So v2.1.0 is about twice as fast as v1.0 with two patches, and I can't split that factor. No two runs in the study differ only in the schedule. The pair that would settle it is one build run twice, once in `adaptive` mode and once with the schedule forced to static, and I haven't run it. The factor of two is the schedules, the pinning, the vector grid and the fixes together.

v2.0 has the scheduler and the vector grid, but on the growth case it stops too early to time, so the first build I can time is the first development one, and that also has pinning and the fixes that keep a run going. By then nearly 80% of the cores are busy, and all four development builds reach 10,000 cells in the same hour and a quarter as v2.1.0.

From there to v2.1.0 is a small step, a few percent in hours to 60,000 cells (a bit over 14 hours against about 15), with three runs of each and a run-to-run spread of a percent or two. Up to that point I can't tell the development builds apart.

Three of the four leak memory, about 140 KB each time a division fails partway, and late in a two-day run they fill the machine's 60-odd GB and start swapping (that's the upward bend at the right edge of the scaling chart). The fourth development build and v2.1.0 don't show the leak. They hold about 130 MB per thousand cells and end the run under 20 GB, and I haven't worked out which change made the difference. It costs hours, too : to 125,000 cells the three leaking builds take about 42 to 43 hours, the fourth 39, and v2.1.0 37 to 39.

Where both versions complete a case the results agree within run-to-run noise, with two exceptions, both in mean cell pressure. The vesicle case is a sweep over a parameter $\varepsilon$, timed above at $\varepsilon = 2$. At $\varepsilon = 3$, v1.0 with two patches gives about 445 Pa against 350 for v2.1.0, roughly a quarter higher. In a monolayer vesicle it's 20 against 26 Pa. I don't know why, in either case. The two versions don't compute exactly the same thing, since a handful of the changes between them touch the forces themselves (how the closest point on a triangle is found, how bending is handled on very thin triangles, how coupled nodes find their partners after a division), and I haven't traced the difference to any one of them. (Mesh resolution on the growth case is the same in both, about 120 nodes per cell.)

## still open

These are the problems I know about in the fork, v2.1.0 included.

- **The paper's time step.** On the growth benchmark my builds don't hold it. v2.1.0 starts removing unstable cells in bulk from around 14,000 cells, and the development builds from a few hundred to about 4,000. Unless it says otherwise, every growth-case number above is at a quarter of that step.
- **Time integration is serial.** In the review of the v2.1 release candidate that phase took about 25 ms per iteration with the program set to one thread, and about 31 with it set to four, on a vesicle of about 430 cells.
- **A race in the node-node contact model.** It adds forces to a neighbouring cell's nodes inside the parallel loop without an atomic update, so a force can get lost when two threads write to the same node. I haven't measured how large the error is. Every growth-case number in this post carries it.
- **Cell deaths aren't counted.** A cell that shrinks below the minimum volume is deleted with no log line and no count. The benchmark's estimate is about 9,000 deaths by the time the tissue holds 10,000 cells and over 200,000 by 100,000.

<style>
a.btn-soft {
  display: inline-block;
  margin: 0 0.5rem 0.55rem 0;
  padding: 0.5em 1.05em;
  font-size: 0.92rem;
  font-weight: 500;
  line-height: 1.3;
  color: inherit;
  text-decoration: none;
  border-radius: 8px;
  border: 1px solid rgba(128, 128, 128, 0.30);
  border: 1px solid color-mix(in srgb, currentColor 22%, transparent);
  background: rgba(128, 128, 128, 0.06);
  background: color-mix(in srgb, currentColor 5%, transparent);
  transition: background-color .18s ease, border-color .18s ease, color .18s ease;
}
a.btn-soft:hover {
  text-decoration: none;
  border-color: rgba(128, 128, 128, 0.55);
  border-color: color-mix(in srgb, currentColor 42%, transparent);
  background: rgba(128, 128, 128, 0.12);
  background: color-mix(in srgb, currentColor 11%, transparent);
}
a.btn-soft--primary {
  font-weight: 600;
  border-color: rgba(128, 128, 128, 0.55);
  border-color: color-mix(in srgb, currentColor 42%, transparent);
}
</style>

[Fork on GitHub](https://github.com/nilesh-patil/simucell3d){: .btn-soft .btn-soft--primary} [Project page](/portfolio/simucell3d/){: .btn-soft} [SimuCell3D paper (Nature Computational Science, 2024)](https://www.nature.com/articles/s43588-024-00620-9){: .btn-soft}
