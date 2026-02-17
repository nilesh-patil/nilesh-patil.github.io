---
layout: single
title: "Flow-network analysis - functional city structure from NYC yellow-taxi trips"
date: 2017-03-14T15:39:55-04:00
last_modified_at: 2026-02-17T22:14:07+05:30
categories: [blog]
tags: [graph, network, visualization, nyc, transportation]
excerpt: "146 million taxi trips collapse into a 262-node graph. A neighborhood the gravity model scores as mildly under-connected is the city's busiest late-night taxi destination, and the shape survives a six-fold swing in volume."
redirect_from:
  - /blog/transportation-graph-nyc-taxi-data/
header:
  overlay_image: /images/blog/headers/transportation-graph-nyc-taxi-data.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/transportation-graph-nyc-taxi-data.jpg
---

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
.page__content { overflow-wrap: break-word; }
</style>

*I first published this post in March 2017 and rewrote it in February 2026 after re-running the whole pipeline. Every number and figure below comes from that re-run, as do the tools it names (DuckDB, pixi, Leiden).*

I took the 146 million yellow-taxi trips New York logged in 2015 and collapsed them into a directed graph of 262 taxi zones, then read the city off that graph, its hubs, its districts, and how much of the flow a gravity model can predict.

## From trips to a graph

A trip becomes a directed edge from its pickup node to its dropoff node, and every trip between the same pair collapses into one weighted edge whose weight is the trip count<sup>[1]</sup>.

What a node is took two tries, years apart. In 2017 the TLC files shipped raw pickup and dropoff coordinates, and my first graph put them on a grid of roughly 200m cells : 40,000+ nodes, too many to partition or even draw cleanly, and one real place kept splintering into several nodes. Penn Station has multiple entrances, each entrance collected its own cluster of pickups, and the busiest location in the city showed up in my graph as a handful of medium nodes. The fix then was joining the coordinates onto census tracts, which brought the graph down to about 580 nodes. The re-run of the pipeline in February 2026 hit a surprise in the opposite direction. TLC has since re-coded the historical data : the 2015 parquet now ships zone ids instead of coordinates, the original lat/long files are no longer distributed, the old s3 bucket that held them returns HTTP 403, and trips arrive pre-aggregated to 263 taxi zones, unions of census tracts built for exactly this purpose. The problem I spent my 2017 cleanup on is solved upstream now, and the data it lived in is gone from distribution. So the node is a taxi zone. The zone-to-tract crosswalk I built to compare the two graphs says one zone overlaps about 19 census tracts on average, which puts a number on how much resolution the move gave up.

Cleaning is conservative. Of 146,039,231 raw rows across the twelve monthly files, 142,199,201 survive (97.4%), after dropping invalid zone ids (anything outside 1 to 263, which covers TLC's Unknown and Outside of NYC codes), non-positive durations, durations outside one to 360 minutes, fares outside a cent to a thousand dollars, and distances above a hundred miles. One count I cannot reconcile : my 2017 post reported 146,112,990 trips, the re-coded parquet holds 146,039,231 rows, and I do not know which 73,759 rows TLC dropped in the re-code. Intra-zone trips, pickup and dropoff in the same zone, are 4.54% of the total; they stay in the edge table but sit out every network and spatial metric, because a self-loop has zero distance and breaks the distance-based methods.

Plot fare against duration over the same trips and there is a flat strip, rides whose cost does not move with how long they took. My 2017 guess was tips rounded to the nearest five or ten dollars. The rate-code coloring says otherwise.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/04_cost_vs_duration.png" alt="Scatter of fare against duration, colored by rate code, with the flat band near 52 dollars highlighted as JFK flat-fare trips" />
<figcaption>Fare against duration, colored by rate code. The flat band is a fixed $52.</figcaption>
</figure>

It is the JFK flat fare : RatecodeID 2, a fixed fare regardless of how long the ride takes. 2.19% of all trips fall in the 49-to-53-dollar band, and 92.7% of those are flat-fare trips.

The ingest is a single DuckDB pass over about 2 GB of parquet, the graph work is NetworkX and igraph, and the whole pipeline is pixi-locked, one `pixi run pipeline` from raw parquet to every figure below. What comes out carries about 135.7M trips, the 142.2M minus the intra-zone ones, across 262 active nodes and 42,347 directed edges, 7,945 of which carry more than 500 trips in the year, with a mean trip distance of 2.25 miles, measured centroid to centroid. Density is about 0.62, so most zone pairs exchange at least some taxis.

Rank the zones by in-flow and by out-flow and the two orderings do not match.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/02_degree_distribution.png" alt="Rank plots of per-zone in-strength and out-strength, both heavy-tailed but with distinct shapes" />
<figcaption>Per-zone in-strength and out-strength in rank order.</figcaption>
</figure>

Both curves are heavy-tailed and they are not mirror images of each other, so the flow is directed and a handful of zones dominate it.

## Hubs

Drawn on real geography, with node size set by out-strength and edge width by annual trips, the graph sorts the city into tiers<sup>[2]</sup>.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/05_hub_map_geographic.png" alt="The flow network drawn on Manhattan-core geography, node size set by out-strength and edge width by annual trips" />
<figcaption>The flow network on its real geography.</figcaption>
</figure>

Trips per distinct source : in-strength divided by the number of distinct origins a zone draws from, cheap to compute off the weighted edge table. Midtown Center pulls 22,929 trips per distinct origin, Times Square 18,664, Penn Station 17,399, the East Village 16,143. LaGuardia pulls 7,289 trips per source and JFK only 4,510. The gap is mostly volume. Roughly 240 of the 260-odd origin zones feed each airport, and 219 to 233 feed each of the four Manhattan zones.

## Four districts

Community detection is the part I trust least. The algorithm is Leiden. It gives four districts<sup>[3]</sup>.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/07_community_map.png" alt="The network colored into four Leiden communities, node size set by outgoing trips, districts cutting across borough boundaries" />
<figcaption>Leiden communities on the 2015 graph, node size set by trips leaving each zone over the year.</figcaption>
</figure>

How much should you believe four? The resolution parameter decides the count. At 1.0, the value I used, you get four; my Louvain sweep over other resolutions came back with anything from four to seventeen. So the claim I can defend is smaller : at this resolution, the partition is stable. 100 random Louvain seeds, count stays three or four, mode four, mean Adjusted Rand Index between seed pairs 0.751.

Q = 0.189 is the modularity, low as modularity numbers go. The reshuffle null puts it z = 13.8 out, though, and about half the trips never leave their district. Borough labels do not predict the partition well (ARI 0.25). Service-zone labels do not either.

## Where Manhattan ends

The East Village looks central on a geographic map. In the 2015 flow it kept showing up with the connectivity of somewhere much farther out, and I wanted to know whether that read was real.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/11_distance_decay.png" alt="Scatter of annual trips per origin-destination pair against centroid distance, with a fitted decay curve falling off steeply" />
<figcaption>Annual trips per origin-destination pair against centroid distance, with the fitted decay curve.</figcaption>
</figure>

Flow falls off steeply with distance<sup>[4]</sup>. For the model I used the doubly-constrained gravity form, the Furness version, because it conditions on what each zone sends and receives plus the pairwise distances and nothing else. Whatever it cannot predict is, by construction, the part volume and geography do not explain. The decay exponent calibrates to 1.146. I also fit an unconstrained Poisson version, which returns a gentler 0.72.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/12_gravity_fit.png" alt="Scatter of observed origin-destination flows against doubly-constrained gravity-predicted flows, clustered along the diagonal" />
<figcaption>Observed against predicted flows in the high-coverage core.</figcaption>
</figure>

Observed against predicted, the common part of commuters, the share of trips the model puts on the right zone pair, comes out at 0.816.

Yellow-taxi pickups concentrate in the Manhattan core plus the airports, the zones TLC labels Yellow Zone, Airports or EWR. The residuals mean something inside that core and nothing outside it, which is why the rest of the map is greyed out. Even in the core there is a catch I cannot check from this data : yellow cabs are one way of getting around among several, and if a good share of the East Village's night trips in 2015 were riding something else, ride-hail for one, part of what reads as under-connected would just be invisible here.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/08_where_manhattan_ends.png" alt="Manhattan-core taxi zones colored by gravity-model deviance residual, under-connected zones blue and over-connected red" />
<figcaption>Per-zone deviance residual from the doubly-constrained gravity model, scoped to the high-coverage core. Blue is under-connected given volume and location, red is over-connected, grey is too thinly observed to read.</figcaption>
</figure>

The residuals span -20.3 to +9.4 across the core. The East Village comes in at -5.1, the 29th percentile, and Alphabet City at -4.3. The Lower East Side complicated the story for me : it lands at +0.8, right about where the model expects it, even though I would have lumped it with the East Village by eye. The most under-connected zones are uptown residential, Upper East Side North at -20.3, Upper West Side South at -16.3. The most over-connected are the commercial spine, Times Square at +9.4, Midtown South at +6.3.

## Day and night trade places

Build the same graph on morning trips and evening trips separately (the weekday peaks, 06 to 10 and 16 to 20) and the directional structure flips sign. A net-flow index, trips out minus trips in, normalised, runs Midtown Center at -0.567 in the morning and +0.128 in the evening : a sink at 9am, a source at 6pm, the classic central business district. The East Village is the mirror image, +0.492 in the morning and -0.168 in the evening.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/10_day_night_reversal.png" alt="Two-panel figure showing the morning-evening net-flow sign flip and the East Village rising from 42nd destination by day to 1st at night" />
<figcaption>Net-flow index in the AM and PM peaks (left), and destination rank by day against late night (right).</figcaption>
</figure>

From daytime (the same weekday 06 to 10 peak) to late night (weekends, 22 to 04) the East Village climbs from the 42nd-busiest destination to the 1st, the Lower East Side from 51st to 6th, and Clinton East from 22nd to 2nd, while the AM and late-night top-15 hub sets overlap at about 0.20 by Jaccard.

That is also how the East Village can score as under-connected on the residual map and still top the late-night ranking. The gravity model is handed each zone's yearly totals before it predicts anything, so -5.1 makes no statement about how busy the zone is, only that its trips are not spread over partner zones the way volume and distance alone would spread them, while the ranking is raw arrivals inside one daypart and late night is the daypart the East Village fills in. How much of the -5.1 the night trips account for, these figures do not say.

## Stress tests

Most of the network-science toolbox assumes sparse graphs. Mine sits at density 0.62, close to complete, so before trusting any headline claims I spent a while checking which standard results even apply here.

The strength plots practically beg the scale-free question. I followed the Clauset-Shalizi-Newman recipe : discrete maximum-likelihood fits, a goodness-of-fit bootstrap, Vuong comparisons against alternative distributions.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/17_degree_distribution_loglog.png" alt="Log-log complementary-CDF fits of in, out, and total strength with fitted power law and lognormal overlays" />
<figcaption>Complementary CDFs of in-, out- and total strength on log-log axes.</figcaption>
</figure>

The fitted exponents come out at 1.23, 1.35 and 1.33 for out-, in- and total strength, all below 2. Then the bootstrap rejects the power law outright, p_gof = 0, for all three, and each one ties with a lognormal under Vuong. What survives is the boring version : the tails beat an exponential decisively, and nothing stronger. With 262 nodes the tail is too short to call anything scale-free anyway.

Removing nodes was the next thing I tried. Molloy-Reed kappa comes out at 365.5 against a threshold of 2, which on a near-complete graph is a degenerate baseline and says nothing about resilience, and the graph basically cannot be disconnected by deleting zones.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/18_attack_vs_failure.png" alt="Robustness curves: largest weakly-connected component, weighted efficiency, and surviving-trip fraction under random failure versus targeted attack" />
<figcaption>Largest component, flow-weighted efficiency and surviving-trip fraction as nodes are removed, random order against strength-targeted.</figcaption>
</figure>

Take out a random 10% of zones and 82.2% of trips survive, with about 96% of weighted efficiency. Target the same 10% by strength instead, recomputing after each removal, and surviving trips drop to 13.0%, efficiency to about 17%.

Of the null models, the useful one keeps the observed topology fixed and only permutes trip weights across the existing edges. The observed weighted reciprocity, 0.818, lands at z = +217 against that null, and the cost-weighted efficiency (edge cost 1/trips) at z = -54. I also ran Erdos-Renyi and configuration nulls; on a graph this dense their z-scores are enormous and mean nothing.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/19_nullmodel_overlay.png" alt="Null-model benchmarking: weight-permuted null on the left with observed reciprocity and efficiency far in the tail, topology nulls on the right" />
<figcaption>Observed weighted reciprocity and cost-weighted efficiency against the weight-permuted null (left) and the topology nulls (right).</figcaption>
</figure>

Corridor circuity, network distance over straight-line distance, has a mean of 1.006 and a median of 1.000. Edge length is centroid-to-centroid distance, so a directly connected pair scores exactly 1. At density 0.62 that is most pairs. Global efficiency is 0.173, which is 0.997 of the straight-line ideal<sup>[5]</sup>.

Betweenness load is concentrated, Gini 0.89 with the top-10 zones carrying 54.7%, but it is not spatially clustered (Moran's I = 0.028, p = 0.238); the load sits in scattered hubs. The Motter-Lai cascade I triggered at JFK does collapse the giant component at every tolerance, and I discount my own result here : 149 of the 259 zones that have geometry (three of the 262 do not, since TLC's shapefile stores ids 57, 104 and 105 under 56 and 103) carry zero initial load, so any rerouting fails them by construction.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/20_spatial_efficiency_cascade.png" alt="Spatial efficiency diagnostics, a Motter-Lai cascade triggered at JFK, and a betweenness load map" />
<figcaption>Circuity and straightness diagnostics (straightness is straight-line over network distance, the inverse of circuity, averaged per zone), the betweenness load map, and the Motter-Lai cascade triggered at JFK.</figcaption>
</figure>

## Ten years of the same graph

This last section is new in the February 2026 revision : I came back and re-ran the identical pipeline on every year from 2015 to 2024.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/24_volume_timeline_multiyear.png" alt="Monthly yellow-taxi trip volume stitched into one continuous 2015 to 2024 timeline, with a steady pre-COVID decline and a sharp 2020 cliff" />
<figcaption>Monthly yellow-taxi trips, 2015 to 2024.</figcaption>
</figure>

The mean is 69.8M trips a year, the low 22.8M, the high 135.7M, a coefficient of variation of 0.61. There is a 42% secular decline from 2015 to 2019 as ride-hailing eats the yellow-cab share, a 70.9% collapse into 2020 when COVID empties the streets, and a recovery to only 48.6% of the 2019 level by 2024. Trips per year change by a factor of about six.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/26_metric_evolution_panel.png" alt="Panel of per-year network metrics from 2015 to 2024 showing volume cratering while global efficiency, gravity fit, and modularity stay nearly flat" />
<figcaption>Per-year network metrics, 2015 to 2024.</figcaption>
</figure>

Against that, the gravity-fit quality has a CV of 0.007, and the community count stays between three and four throughout. This is a different measure than the seed test from the districts section : that one shuffled random seeds on the single 2015 graph, this one compares each year's partition to the next year's. Consecutive partitions, each run with the same fixed seed, align at an average ARI of 0.883, and the 2019-to-2020 COVID boundary itself scores 0.943, meaning the functional districts did not re-draw under the shock.

The exception is the marquee. From 2020 on, Upper East Side North overtakes Midtown Center as the city's top arrival hub, 976,160 arrivals against 797,677 in 2020 and 1,581,109 against 1,429,662 in 2024.

[Explore the interactive project]({{ site.baseurl }}/transportation-flow-network/){: .btn-soft .btn-soft--primary} [Browse the code on GitHub](https://github.com/nilesh-patil/transportation-flow-network){: .btn-soft}

## References

1. NYC taxi data: http://www.nyc.gov/html/tlc/html/about/trip\_record\_data.shtml
2. Peng C, Jin X, Wong K-C, Shi M, Lio P (2012) Collective Human Mobility Pattern from Taxi Trips in Urban Area. PLoS ONE 7(4): e34487. doi:10.1371/journal.pone.0034487
3. Dash Nelson G, Rae A (2016) An Economic Geography of the United States: From Commutes to Megaregions. PLoS ONE 11(11): e0166083. doi:10.1371/journal.pone.0166083
4. Tobler W. A computer movie simulating urban growth in the Detroit region. Economic Geography 1970;46: 234-240.
5. P. Crucitti, V. Latora, and S. Porta. Centrality measures in spatial networks of urban streets. Physical Review E, 73(3):036125, 2006.
