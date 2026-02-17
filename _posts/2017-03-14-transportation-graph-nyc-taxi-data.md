---
layout: single
title: "Flow-network analysis - functional city structure from NYC yellow-taxi trips"
date: 2017-03-14T15:39:55-04:00
last_modified_at: 2026-02-17T22:14:07+05:30
categories: [blog]
tags: [graph, network, visualization, nyc, transportation]
excerpt: "146 million taxi trips collapse into a 262-node graph. The neighborhood the gravity model calls a suburb turns out to be a nightlife district, and the shape survives a six-fold swing in volume."
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

[Explore the interactive project]({{ site.baseurl }}/transportation-flow-network/){: .btn-soft .btn-soft--primary} [Browse the code on GitHub](https://github.com/nilesh-patil/transportation-flow-network){: .btn-soft}

## From trips to a graph

A trip becomes a directed edge from its pickup node to its dropoff node, and every trip between the same pair collapses into one weighted edge whose weight is the trip count<sup>[1]</sup>. That is the whole move. The rest of the post is reading things off the object it produces.

<figure>
<svg viewBox="0 0 640 260" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="t2g-title t2g-desc">
<title id="t2g-title">A taxi trip becomes a directed edge between two location nodes</title>
<desc id="t2g-desc">On the left, several faint individual taxi trips run from a pickup area to a dropoff area. On the right, those trips collapse into one node-to-node arrow weighted by trip count.</desc>
<g fill="none" stroke="currentColor">
<g opacity="0.35" stroke-width="1.5">
<line x1="60" y1="70" x2="230" y2="60"/>
<line x1="55" y1="95" x2="235" y2="100"/>
<line x1="62" y1="120" x2="228" y2="130"/>
<line x1="58" y1="150" x2="232" y2="155"/>
</g>
<g opacity="0.35" fill="currentColor" stroke="none">
<circle cx="60" cy="70" r="4"/><circle cx="55" cy="95" r="4"/><circle cx="62" cy="120" r="4"/><circle cx="58" cy="150" r="4"/>
<circle cx="230" cy="60" r="4"/><circle cx="235" cy="100" r="4"/><circle cx="228" cy="130" r="4"/><circle cx="232" cy="155" r="4"/>
</g>
<text x="58" y="40" fill="currentColor" stroke="none" font-size="13" opacity="0.8">pickups</text>
<text x="200" y="40" fill="currentColor" stroke="none" font-size="13" opacity="0.8">dropoffs</text>
<text x="150" y="210" fill="currentColor" stroke="none" font-size="13" opacity="0.7" text-anchor="middle">raw trips</text>
<line x1="300" y1="110" x2="350" y2="110" stroke-width="2" opacity="0.7"/>
<polygon points="350,104 364,110 350,116" fill="currentColor" stroke="none" opacity="0.7"/>
<circle cx="430" cy="110" r="22" stroke-width="2"/>
<circle cx="580" cy="110" r="22" stroke-width="2"/>
<text x="430" y="114" fill="currentColor" stroke="none" font-size="13" text-anchor="middle">A</text>
<text x="580" y="114" fill="currentColor" stroke="none" font-size="13" text-anchor="middle">B</text>
<line x1="452" y1="110" x2="552" y2="110" stroke-width="5"/>
<polygon points="552,101 572,110 552,119" fill="currentColor" stroke="none"/>
<text x="505" y="155" fill="currentColor" stroke="none" font-size="13" text-anchor="middle" opacity="0.85">one weighted edge</text>
</g>
</svg>
<figcaption>Many individual taxi runs between the same two places collapse into one directed edge from A to B, weighted by the trip count.</figcaption>
</figure>

What a node is took two tries, years apart. In 2017 the TLC files shipped raw pickup and dropoff coordinates, and my first graph put them on a grid of roughly 200m cells : 40,000+ nodes, too many to partition or even draw cleanly, and one real place kept splintering into several nodes. Penn Station has multiple entrances, each entrance collected its own cluster of pickups, and the busiest location in the city showed up in my graph as a handful of medium nodes. The fix then was joining the coordinates onto census tracts, which brought the graph down to about 580 nodes. The re-run of the pipeline years later, which is where every number in this post comes from, hit a surprise in the opposite direction. TLC has since re-coded the historical data : the 2015 parquet now ships zone ids instead of coordinates, the original lat/long files are no longer distributed, the old s3 bucket that held them returns HTTP 403, and trips arrive pre-aggregated to 263 taxi zones, unions of census tracts built for exactly this purpose. The problem I spent my 2017 cleanup on is solved upstream now, and the data it lived in is gone from distribution. So the node is a taxi zone. The zone-to-tract crosswalk I built to compare the two graphs says one zone overlaps about 19 census tracts on average, which puts a number on how much resolution the move gave up.

Cleaning is conservative. Of 146,039,231 raw rows across the twelve monthly files, 142,199,201 survive (97.4%), after dropping invalid zone ids, non-positive durations, durations outside one to 360 minutes, fares outside a cent to a thousand dollars, and distances above a hundred miles. One count I cannot reconcile : my 2017 post reported 146,112,990 trips, the re-coded parquet holds 146,039,231 rows, and I do not know which 73,759 rows TLC dropped in the re-code. Intra-zone trips, pickup and dropoff in the same zone, are 4.54% of the total; they stay in the edge table but sit out every network and spatial metric, because a self-loop has zero distance and breaks the distance-based methods.

The ingest is a single DuckDB pass over about 2 GB of parquet, the graph work is NetworkX and igraph, and the whole pipeline is pixi-locked, one `pixi run pipeline` from raw parquet to every figure below. What comes out carries about 135.7M trips across 262 active nodes and 42,347 directed edges, 7,945 of which carry more than 500 trips in the year, with a mean trip distance of 2.25 miles. Density is about 0.62, so most zone pairs exchange at least some taxis.

Rank the zones by in-flow and by out-flow and the two orderings do not match.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/02_degree_distribution.png" alt="Rank plots of per-zone in-strength and out-strength, both heavy-tailed but with distinct shapes" />
<figcaption>Per-zone in-strength and out-strength in rank order.</figcaption>
</figure>

Both curves are heavy-tailed and they are not mirror images of each other, so the flow is genuinely directed and a handful of zones dominate it.

## Two kinds of hub

Drawn on real geography, with node size set by out-strength and edge width by annual trips, the graph sorts the city into tiers<sup>[2]</sup>.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/05_hub_map_geographic.png" alt="The flow network drawn on Manhattan-core geography, node size set by out-strength and edge width by annual trips" />
<figcaption>The flow network on its real geography. Node size encodes out-strength, edge width annual trips.</figcaption>
</figure>

The biggest nodes on the map are the transport hubs, office districts next. Two zones can also pull similar volume in completely different ways, and what separates them is trips per distinct source : in-strength divided by the number of distinct origins a zone draws from, cheap to compute off the weighted edge table.

<figure>
<svg viewBox="0 0 640 280" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" aria-labelledby="hub-title hub-desc">
<title id="hub-title">An inner-city hub versus an airport-style node</title>
<desc id="hub-desc">Left: many arrows converge from a few clustered places into one high-in-degree hub. Right: arrows arrive from many scattered places into one node with a lower trips-per-source ratio.</desc>
<g stroke="currentColor" fill="none">
<g fill="currentColor" stroke="none" opacity="0.7">
<circle cx="40" cy="110" r="5"/><circle cx="45" cy="135" r="5"/><circle cx="38" cy="160" r="5"/>
</g>
<g stroke-width="4" opacity="0.85">
<line x1="45" y1="110" x2="150" y2="140"/>
<line x1="50" y1="135" x2="150" y2="140"/>
<line x1="43" y1="160" x2="150" y2="140"/>
</g>
<circle cx="175" cy="140" r="26" stroke-width="2"/>
<text x="175" y="144" fill="currentColor" stroke="none" font-size="12" text-anchor="middle">hub</text>
<text x="120" y="225" fill="currentColor" stroke="none" font-size="13" text-anchor="middle" opacity="0.85">few sources, huge in-degree</text>
<text x="120" y="245" fill="currentColor" stroke="none" font-size="11" text-anchor="middle" opacity="0.65">Penn Station / MSG</text>
<g fill="currentColor" stroke="none" opacity="0.7">
<circle cx="400" cy="50" r="4"/><circle cx="370" cy="90" r="4"/><circle cx="420" cy="100" r="4"/>
<circle cx="380" cy="140" r="4"/><circle cx="410" cy="180" r="4"/><circle cx="375" cy="210" r="4"/>
<circle cx="430" cy="220" r="4"/>
</g>
<g stroke-width="1.5" opacity="0.7">
<line x1="404" y1="52" x2="540" y2="135"/>
<line x1="374" y1="92" x2="540" y2="135"/>
<line x1="424" y1="100" x2="540" y2="135"/>
<line x1="384" y1="140" x2="540" y2="135"/>
<line x1="414" y1="180" x2="540" y2="135"/>
<line x1="379" y1="210" x2="540" y2="135"/>
<line x1="434" y1="220" x2="540" y2="135"/>
</g>
<circle cx="565" cy="135" r="22" stroke-width="2"/>
<text x="565" y="139" fill="currentColor" stroke="none" font-size="11" text-anchor="middle">airport</text>
<text x="470" y="250" fill="currentColor" stroke="none" font-size="13" text-anchor="middle" opacity="0.85">many scattered sources, lower ratio</text>
</g>
</svg>
<figcaption>An inner-city draw pulls from a few nearby places and trips per source runs high. An airport pulls from everywhere and trips per source stays low.</figcaption>
</figure>

Midtown Center pulls 22,929 trips per distinct origin, Times Square 18,664, Penn Station 17,399, the East Village 16,143. These are the inner-city draws, enormous volume converging from a limited, mostly Manhattan set of origins. The airports invert the ratio, LaGuardia at 7,289 trips per source and JFK at only 4,510, because roughly 240 of the 260-odd origin zones feed each airport.

## Four districts

Community detection was the part I was most excited about in 2017, and it is the part I trust least now. The algorithm on the re-run is Leiden, which did not exist when the original post went up (it was published in 2019, after the original analysis), so this partition is a later-re-run product like every other number here. It gives four districts<sup>[3]</sup>.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/07_community_map.png" alt="The network colored into four Leiden communities, node size set by outgoing trips, districts cutting across borough boundaries" />
<figcaption>Leiden communities on the 2015 graph, node size set by trips leaving each zone over the year.</figcaption>
</figure>

How much should you believe four? The resolution parameter decides the count. At 1.0, the value I used, you get four; my sweep over other resolutions came back with anything from four to seventeen. So the honest claim is smaller : at this resolution, the partition is stable. 100 random seeds, count stays three or four, mode four, mean Adjusted Rand Index between seed pairs 0.751.

Q = 0.189 is the modularity, low as modularity numbers go. The reshuffle null puts it z = 13.8 out, though, and about half the trips never leave their district. Borough labels do not predict the partition well. Service-zone labels do not either. Whatever the taxi flow is organized around, it is not the administrative map.

## Where Manhattan ends

The East Village looks central on a geographic map. In the 2015 flow it kept showing up with the connectivity of somewhere much farther out, and I wanted to know whether that read was real. Getting there takes a distance-decay curve, then a gravity model, then a residual map.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/11_distance_decay.png" alt="Scatter of annual trips per origin-destination pair against centroid distance, with a fitted decay curve falling off steeply" />
<figcaption>Annual trips per origin-destination pair against centroid distance, with the fitted decay curve.</figcaption>
</figure>

Flow falls off steeply with distance<sup>[4]</sup>. For the model I used the doubly-constrained gravity form, the Furness version, because it conditions on what each zone sends and receives plus the pairwise distances and nothing else. Whatever it cannot predict is, by construction, the part volume and geography do not explain. The decay exponent calibrates to 1.146. I also fit an unconstrained Poisson version, which returns a gentler 0.72, but it leans on the very flows it is supposed to explain, so that number gets reported here and then ignored.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/12_gravity_fit.png" alt="Scatter of observed origin-destination flows against doubly-constrained gravity-predicted flows, clustered along the diagonal" />
<figcaption>Observed against predicted flows in the high-coverage core.</figcaption>
</figure>

Observed against predicted, the common part of commuters comes out at 0.816. As a one-number description of the city, that is fair.

Yellow-taxi pickups concentrate in the Manhattan core plus the airports. The residuals mean something inside that core and nothing outside it, which is why the rest of the map is greyed out. Even in the core there is a catch : by 2015 the East Village's nightlife crowd was increasingly in green cabs and ride-hail, so part of what reads as under-connected might just be invisible to yellow taxis.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/08_where_manhattan_ends.png" alt="Manhattan-core taxi zones colored by gravity-model deviance residual, under-connected zones blue and over-connected red" />
<figcaption>Per-zone deviance residual from the doubly-constrained gravity model, scoped to the high-coverage core. Blue is under-connected given volume and location, red is over-connected, grey is too thinly observed to read.</figcaption>
</figure>

The residuals span -20.3 to +9.4 across the core. The East Village comes in at -5.1, the 29th percentile, and Alphabet City at -4.3. Both sit well inside the under-connected tail, so the functional edge of Manhattan really does cut through neighborhoods the geographic map counts as central. The Lower East Side complicated the story for me : it lands at +0.8, right about where the model expects it, even though I would have lumped it with the East Village by eye. And the extreme ends of the scale were not where I looked for them. The most under-connected zones are uptown residential, Upper East Side North at -20.3, Upper West Side South at -16.3. The most over-connected are the commercial spine, Times Square at +9.4, Midtown South at +6.3.

## The flat-fare band

The points that miss the gravity diagonal have a mundane explanation. Plot fare against duration over the same trips and there is a flat strip, rides whose cost does not move with how long they took. My 2017 guess was tips rounded to the nearest five or ten dollars. The rate-code coloring says otherwise.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/04_cost_vs_duration.png" alt="Scatter of fare against duration, colored by rate code, with the flat band near 52 dollars highlighted as JFK flat-fare trips" />
<figcaption>Fare against duration, colored by rate code. The flat band is the JFK flat fare (RatecodeID 2), a fixed $52 whatever the duration.</figcaption>
</figure>

It is the JFK flat fare : RatecodeID 2, a fixed fare regardless of how long the ride takes. 2.19% of all trips fall in the 49-to-53-dollar band, and 92.7% of those are flat-fare trips.

## Day and night trade places

Build the same graph on morning trips and evening trips separately and the directional structure flips sign. A net-flow index, trips out minus trips in, normalised, runs Midtown Center at -0.567 in the morning and +0.128 in the evening : a sink at 9am, a source at 6pm, the classic central business district. The East Village is the mirror image, +0.492 in the morning and -0.168 in the evening.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/10_day_night_reversal.png" alt="Two-panel figure showing the morning-evening net-flow sign flip and the East Village rising from 42nd destination by day to 1st at night" />
<figcaption>Net-flow index by daypart. Midtown fills in the morning and drains in the evening, the East Village does the opposite, and by late night the East Village is the city's busiest taxi destination.</figcaption>
</figure>

From daytime to late night the East Village climbs from the 42nd-busiest destination to the 1st, the Lower East Side from 51st to 6th, and Clinton East from 22nd to 2nd, while the AM and late-night top-15 hub sets overlap at about 0.20 by Jaccard. The city runs two largely different sets of busiest places depending on the hour, and the suburb reading from the residual map turns out to be a daytime effect, written by where the nightlife is.

## Stress tests

Most of the network-science toolbox assumes sparse graphs. Mine sits at density 0.62, close to complete, so before trusting any headline claims I spent a while checking which standard results even apply here. Four of those checks are worth writing down.

First the scale-free question, because the strength plots practically beg it. I followed the Clauset-Shalizi-Newman recipe : discrete maximum-likelihood fits, a goodness-of-fit bootstrap, Vuong comparisons against alternative distributions.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/17_degree_distribution_loglog.png" alt="Log-log complementary-CDF fits of in, out, and total strength with fitted power law and lognormal overlays" />
<figcaption>Complementary CDFs of in-, out- and total strength on log-log axes, with the fitted power-law and lognormal overlays.</figcaption>
</figure>

The fitted exponents come out at 1.23, 1.35 and 1.33 for out-, in- and total strength, all below 2. Then the bootstrap rejects the power law outright, p_gof = 0, for all three, and each one ties with a lognormal under Vuong. What survives is the boring version : the tails beat an exponential decisively, and nothing stronger. With 262 nodes the tail is too short to call anything scale-free anyway, however tempting the label.

Second, node removal. Molloy-Reed kappa comes out at 365.5 against a threshold of 2, which on a near-complete graph is a degenerate baseline rather than a resilience finding, and the graph basically cannot be disconnected by deleting zones. Trips are the axis where removal actually bites.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/18_attack_vs_failure.png" alt="Robustness curves: largest weakly-connected component, weighted efficiency, and surviving-trip fraction under random failure versus targeted attack" />
<figcaption>Largest component, flow-weighted efficiency and surviving-trip fraction as nodes are removed, random order against strength-targeted.</figcaption>
</figure>

Take out a random 10% of zones and 82.2% of trips survive, with about 96% of weighted efficiency. Target the same 10% by strength instead, recomputing after each removal, and surviving trips drop to 13.0%, efficiency to about 17%. Meanwhile the unweighted largest-component curve barely moves off 0.90 in either case, separating nothing.

Third, null models. The useful one keeps the observed topology fixed and only permutes trip weights across the existing edges.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/19_nullmodel_overlay.png" alt="Null-model benchmarking: weight-preserving null on the left with observed reciprocity and efficiency far in the tail, topology nulls on the right" />
<figcaption>Observed weighted reciprocity and cost-weighted efficiency against the weight-preserving null (left) and the topology nulls (right).</figcaption>
</figure>

The observed weighted reciprocity, 0.818, lands at z = +217 against that null, and the cost-weighted efficiency at z = -54. The heavy mutual corridors, Midtown and the airports, are about as far from random as a number can be. I also ran Erdos-Renyi and configuration nulls; on a graph this dense their z-scores are enormous and mean nothing.

Last, geometry. Corridor circuity, network distance over straight-line distance, has a mean of 1.006 and a median of 1.000. Global efficiency is 0.173, which is 0.997 of the straight-line ideal, so the flow layout is about as direct as the map permits<sup>[5]</sup>.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/20_spatial_efficiency_cascade.png" alt="Spatial efficiency diagnostics, a Motter-Lai cascade triggered at JFK, and a betweenness load map" />
<figcaption>Circuity and straightness diagnostics, the betweenness load map, and the Motter-Lai cascade triggered at JFK.</figcaption>
</figure>

Betweenness load is concentrated, Gini 0.89 with the top-10 zones carrying 54.7%, but it is not spatially clustered (Moran's I = 0.028, p = 0.238); the load sits in scattered hubs, not one patch. The Motter-Lai cascade I triggered at JFK does collapse the giant component at every tolerance, and I discount my own result here : 149 of 259 nodes carry zero initial load, so any rerouting fails them by construction.

## Ten years of the same graph

This last section is a later revision : I came back years afterward and re-ran the identical pipeline on every year from 2015 to 2024, so the COVID collapse sits here on purpose, well past the year the original graph was built.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/24_volume_timeline_multiyear.png" alt="Monthly yellow-taxi trip volume stitched into one continuous 2015 to 2024 timeline, with a steady pre-COVID decline and a sharp 2020 cliff" />
<figcaption>Monthly yellow-taxi trips, 2015 to 2024, stitched into one continuous timeline.</figcaption>
</figure>

The mean is 69.7M trips a year, the low 22.8M, the high 135.7M, a coefficient of variation of 0.61. There is a 42% secular decline from 2015 to 2019 as ride-hailing eats the yellow-cab share, a 70.9% collapse into 2020 when COVID empties the streets, and a recovery to only 48.6% of the 2019 level by 2024. Trips per year change by a factor of about six.

<figure>
<img src="{{ site.baseurl }}/images/blog/transportation-graph-nyc-taxi-data/26_metric_evolution_panel.png" alt="Panel of per-year network metrics from 2015 to 2024 showing volume cratering while global efficiency, gravity fit, modularity, and community count stay nearly flat" />
<figcaption>Per-year network metrics, 2015 to 2024. Volume craters while global efficiency, the gravity fit, modularity and the community count stay nearly flat.</figcaption>
</figure>

Against that, global efficiency has a CV of 0.004 and the gravity-fit quality a CV of 0.007, the two most stable quantities measured anywhere in this project, and the community count stays between three and four throughout. This is a different measure than the seed test from the districts section : that one shuffled random seeds on the single 2015 graph, this one compares each year's partition to the next year's. Consecutive partitions align at an average ARI of 0.883, and the 2019-to-2020 COVID boundary itself scores 0.943, meaning the functional districts did not re-draw under the shock. This is still yellow taxis only, TLC re-coded the trip schema along the way, and the 70.9% and the 42% are movements in yellow-taxi volume specifically, with a changing observation process folded in.

The exception is the marquee. From 2020 on, the Upper East Side overtakes Midtown as the city's top arrival hub, as commuting and tourism fall away faster than residential travel. Everything else, the districts, the distance law, the efficiency, holds through a six-fold swing in the volume that built it.

[Explore the interactive project]({{ site.baseurl }}/transportation-flow-network/){: .btn-soft .btn-soft--primary} [Browse the code on GitHub](https://github.com/nilesh-patil/transportation-flow-network){: .btn-soft}

## References

1. NYC taxi data: http://www.nyc.gov/html/tlc/html/about/trip\_record\_data.shtml
2. Peng C, Jin X, Wong K-C, Shi M, Lio P (2012) Collective Human Mobility Pattern from Taxi Trips in Urban Area. PLoS ONE 7(4): e34487. doi:10.1371/journal.pone.0034487
3. Dash Nelson G, Rae A (2016) An Economic Geography of the United States: From Commutes to Megaregions. PLoS ONE 11(11): e0166083. doi:10.1371/journal.pone.0166083
4. Tobler W. A computer movie simulating urban growth in the Detroit region. Economic Geography 1970;46: 234-240.
5. P. Crucitti, V. Latora, and S. Porta. Centrality measures in spatial networks of urban streets. Physical Review E, 73(3):036125, 2006.
