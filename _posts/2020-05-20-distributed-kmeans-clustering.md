---
layout: single
title: "When K-Means outgrows one machine"
date: 2020-05-20T10:00:00-00:00
last_modified_at: 2020-05-20T10:00:00-00:00
categories: [blog]
tags: [machine-learning, clustering, distributed-computing, python, pyspark, dask]
excerpt: "A mental model for when k-means outgrows one machine, and how the data actually moves when it does"
math: true
redirect_from:
  - /blog/distributed-kmeans-clustering/
header:
  overlay_image: /images/blog/headers/distributed-kmeans.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/distributed-kmeans.jpg
---

The first time k-means broke for me, it did not get slow. It refused to start. A billion 8-dimensional points sat in object storage, and `sklearn.cluster.KMeans` never reached its first iteration: it tried to materialize the array, hit the memory ceiling, and died. The algorithm was fine. The machine was the problem.

That is the stake. The interesting failure is not "my clustering takes too long," it is "the loop cannot run at all, because the data does not fit." Hold that distinction, because everything that follows is about it.

So a warning up front: this post is not a benchmark and it has no speedup numbers, because those would be specific to one machine and one dataset and would not transfer to yours. What transfers is the reasoning, namely when you should distribute k-means and, once you do, exactly how the data moves.

I will carry one example the whole way: a billion 8-dimensional points in object storage, far too large for your laptop. That is the dataset, and every section below is what happens to it.

## What distributing actually changes

Standard k-means is a short loop:

1. **Initialize:** pick `k` cluster centroids.
2. **Assign:** put each point with its nearest centroid.
3. **Update:** move each centroid to the mean of its assigned points.
4. **Check:** test the stop condition (centroids stopped moving, or you hit the iteration cap).
5. **Repeat** steps 2 and 3 until then.

On one machine that is a few lines, and for moderate data it is genuinely fast:

```python
import numpy as np
from sklearn.cluster import KMeans

X = np.random.randn(1000, 2)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
```

The cost of that loop is $O(n \times k \times d \times i)$, where `n` is the number of points, `k` is the number of clusters, `d` is the dimensionality, and `i` is the number of iterations. The term that hurts is `n`, because every one of those `n` points gets touched `k` times on every iteration. When `n` is a billion and the array will not fit in memory, the loop does not slow down, it stops being runnable. That is the wall.

Now look at where the work lives, because one fact decides everything else: the assignment step is the expensive one, and each point is assigned independently of every other point. To decide which centroid a point belongs to, you need that point and the current `k` centroids, and nothing else. That independence is the entire reason distributing k-means works.

So you pin the data where it already lives. You split the billion points across workers, each worker holding a partition that never moves, and broadcast the `k` centroids to every worker. Each worker assigns its own partition locally, which is embarrassingly parallel, and instead of shipping points anywhere, each worker emits a tiny summary per cluster: the sum of its points and a count. A tree-reduce combines those partial sums and counts into `k` updated centroids, those centroids broadcast back out, and the next iteration begins.

The thing that crosses the wire each iteration is `k` centroids, not `n` points. For our billion points in 8 dimensions, that is a handful of tiny vectors moving instead of the entire dataset. The data stays pinned. Only the summaries travel.

<figure>
<svg viewBox="0 0 640 320" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" fill="none" stroke="currentColor" stroke-width="2">
<title>One distributed k-means iteration</title>
<desc>Three worker partitions stay pinned in place. Each assigns its own points locally and emits per-cluster partial sums and counts. A tree-reduce combines them into k updated centroids, which are broadcast back to every worker for the next iteration. Only the k centroids cross the wire.</desc>
<text x="20" y="28" font-size="14" stroke="none" fill="currentColor" font-weight="bold">data stays pinned</text>
<rect x="20" y="48" width="120" height="84" rx="6" opacity="0.9"/>
<rect x="20" y="148" width="120" height="84" rx="6" opacity="0.9"/>
<rect x="20" y="248" width="120" height="56" rx="6" opacity="0.9"/>
<text x="80" y="86" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">partition 1</text>
<text x="80" y="104" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">assign locally</text>
<text x="80" y="186" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">partition 2</text>
<text x="80" y="204" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">assign locally</text>
<text x="80" y="280" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">partition 3</text>
<line x1="140" y1="90" x2="250" y2="150" opacity="0.6"/>
<line x1="140" y1="190" x2="250" y2="175" opacity="0.6"/>
<line x1="140" y1="276" x2="250" y2="200" opacity="0.6"/>
<text x="170" y="120" font-size="10" stroke="none" fill="currentColor" opacity="0.7">sums + counts</text>
<rect x="250" y="138" width="120" height="74" rx="6"/>
<text x="310" y="170" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">tree-reduce</text>
<text x="310" y="190" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">combine partials</text>
<line x1="370" y1="175" x2="470" y2="175"/>
<rect x="470" y="138" width="150" height="74" rx="6"/>
<text x="545" y="170" font-size="13" stroke="none" fill="currentColor" text-anchor="middle" font-weight="bold">k centroids</text>
<text x="545" y="190" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">updated</text>
<path d="M545 212 C545 296 230 300 90 90" stroke-dasharray="5 4" opacity="0.6"/>
<path d="M545 212 C500 296 230 300 90 190" stroke-dasharray="5 4" opacity="0.6"/>
<path d="M545 212 C470 300 250 304 90 276" stroke-dasharray="5 4" opacity="0.6"/>
<text x="250" y="296" font-size="11" stroke="none" fill="currentColor" opacity="0.7">broadcast k centroids back to every worker</text>
</svg>
<figcaption>One iteration of distributed k-means. Partitions never move. Each worker assigns its own points and emits per-cluster sums and counts, a tree-reduce folds them into k updated centroids, and those k centroids broadcast back out to all three workers. Tiny summaries flow up, a tiny model flows down.</figcaption>
</figure>

## The real bottleneck is initialization

Once you see the iteration that way, the loop is cheap to distribute and initialization becomes where the cost moves.

Classic k-means++ picks seeds one at a time. Each new centroid is chosen with probability proportional to its squared distance from the seeds chosen so far, so you have to scan the whole dataset before you can pick the next seed, and that is `k` sequential passes over the data. On one machine, fine. On a cluster, each of those passes is a full distributed scan, effectively `k` full distributed passes stacked back to back, all of them before the actual clustering loop begins. For a large `k` over a billion points, you can spend longer seeding than clustering.

The fix is **k-means||** (k-means parallel). Instead of drawing one seed per pass, it oversamples: each pass draws several candidate centers at once, building up a candidate pool in **O(log k) passes rather than k sequential passes**. Then it reduces that pool down to `k` final seeds with one local reduction, no further distributed scans. Same goal as k-means++, good well-spread seeds, but the number of passes over the distributed data drops from `k` to `O(log k)`.

<figure>
<svg viewBox="0 0 640 340" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" fill="none" stroke="currentColor" stroke-width="2">
<title>k-means++ versus k-means parallel seeding</title>
<desc>On the left, k-means++ runs k sequential full passes over distributed data, one seed per pass, drawn as a tall stack of repeated scans. On the right, k-means parallel runs a few oversampling passes labeled O of log k that gather a candidate pool, then one local reduction down to k seeds.</desc>
<text x="20" y="26" font-size="14" stroke="none" fill="currentColor" font-weight="bold">k-means++</text>
<text x="20" y="44" font-size="11" stroke="none" fill="currentColor" opacity="0.7">k sequential passes</text>
<rect x="20" y="56" width="220" height="26" rx="4" opacity="0.9"/>
<rect x="20" y="90" width="220" height="26" rx="4" opacity="0.8"/>
<rect x="20" y="124" width="220" height="26" rx="4" opacity="0.7"/>
<rect x="20" y="158" width="220" height="26" rx="4" opacity="0.55"/>
<rect x="20" y="192" width="220" height="26" rx="4" opacity="0.4"/>
<text x="130" y="74" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">full scan, +1 seed</text>
<text x="130" y="108" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">full scan, +1 seed</text>
<text x="130" y="142" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">full scan, +1 seed</text>
<text x="130" y="240" font-size="13" stroke="none" fill="currentColor" text-anchor="middle">... k times</text>
<line x1="320" y1="40" x2="320" y2="300" stroke-dasharray="4 5" opacity="0.4"/>
<text x="400" y="26" font-size="14" stroke="none" fill="currentColor" font-weight="bold">k-means||</text>
<text x="400" y="44" font-size="11" stroke="none" fill="currentColor" opacity="0.7">O(log k) oversampling passes</text>
<rect x="400" y="56" width="220" height="30" rx="4" opacity="0.9"/>
<rect x="400" y="94" width="220" height="30" rx="4" opacity="0.8"/>
<rect x="400" y="132" width="220" height="30" rx="4" opacity="0.7"/>
<text x="510" y="76" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">pass: oversample candidates</text>
<text x="510" y="114" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">pass: oversample candidates</text>
<text x="510" y="152" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">pass: oversample candidates</text>
<path d="M510 162 L510 196" opacity="0.7"/>
<rect x="430" y="200" width="160" height="36" rx="4"/>
<text x="510" y="222" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">candidate pool</text>
<path d="M510 236 L510 262" opacity="0.7"/>
<rect x="430" y="266" width="160" height="36" rx="4"/>
<text x="510" y="282" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">local reduce to k seeds</text>
<text x="510" y="296" font-size="9" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">no distributed scan</text>
</svg>
<figcaption>Why k-means|| wins at scale. k-means++ needs k full distributed passes, one seed each. k-means|| oversamples in O(log k) passes into a candidate pool, then reduces locally to k seeds with no further scan.</figcaption>
</figure>

This is why Spark's MLlib defaults to k-means||, and it is the one distributed-specific lever you actually need to know:

```python
from pyspark.ml.clustering import KMeans as SparkKMeans

# k-means|| (k-means parallel) is the scalable seeding, not k-means++.
# Both aim for good seeds, but k-means|| runs in O(log k) passes
# rather than k sequential passes.
kmeans = SparkKMeans(k=5, initMode="k-means||", initSteps=2)
kmeans.setTol(1e-4)
```

## Two engines, one algorithm

PySpark MLlib and Dask-ML both implement the data-movement model from the first section, and they differ in which world they live in, not in what k-means does.

Reach for **Spark** when the data already lives in the Spark or JVM ecosystem and is huge. You load it, pack the columns into a single `features` vector with `VectorAssembler`, and fit, and Spark handles the partitioning and the reduce:

```python
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.feature import VectorAssembler

df = spark.read.option("header", "true").csv(data_path)
features = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df_vec = features.transform(df)

model = SparkKMeans(k=5, maxIter=50, featuresCol="features").fit(df_vec)
predictions = model.transform(df_vec)
```

Reach for **Dask** when the workflow is already NumPy and Pandas native. The API mirrors scikit-learn, so the jump is small: read into a Dask dataframe, hand it a Dask array, fit.

```python
import dask.dataframe as dd
from dask_ml.cluster import KMeans as DaskKMeans

df = dd.read_csv(data_path)
X = df[numeric_cols].to_dask_array(lengths=True)

kmeans = DaskKMeans(n_clusters=5, max_iter=100, init_max_iter=3)
kmeans.fit(X)
labels = kmeans.predict(X)
```

The original version of this post carried a rule of thumb that Spark earns its keep on very large datasets, on the order of >1TB, but treat that as a heuristic, not a measured threshold. The real decision is ecosystem: if your pipeline is JVM and the data is already in Spark, stay in Spark, and if your pipeline is Python and the data is already in arrays, stay in Dask. The friction of crossing that boundary usually outweighs any per-iteration difference.

## When the data never stops: streaming

The two engines above assume the data, however large, eventually lands somewhere and you fit over all of it. There is a third regime: data that never fully lands, a stream you see one batch at a time. For that you want `partial_fit`, which updates the centroids from a single mini-batch and then forgets it.

Here is a sharp edge worth knowing, because the obvious move does not work. `dask_ml.cluster.KMeans` does **not** expose `partial_fit`, so call it and you get an `AttributeError`. The streaming path is to wrap scikit-learn's `MiniBatchKMeans`, which does support `partial_fit`, in `dask_ml.wrappers.Incremental`:

```python
from sklearn.cluster import MiniBatchKMeans
from dask_ml.wrappers import Incremental

def incremental_kmeans_dask(data_stream, k=3):
    base = MiniBatchKMeans(n_clusters=k, random_state=42)
    kmeans = Incremental(base)
    for batch in data_stream:
        # We drive the stream ourselves, feeding explicit mini-batches.
        # Incremental delegates each call to partial_fit on the wrapped
        # estimator (here it is not mapping over Dask-array blocks).
        kmeans.partial_fit(batch)
    return kmeans
```

Each batch nudges the centroids a little, then is discarded, and you never hold the whole stream. That is the entire point. This is the regime where even "fits across the cluster" stops being true, because there is no fixed dataset to fit, only an arrival rate.

<figure>
<svg viewBox="0 0 640 300" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" fill="none" stroke="currentColor" stroke-width="2">
<title>Batch fit versus streaming partial_fit</title>
<desc>On the top, batch fit ingests the whole dataset at once into a set of centroids. On the bottom, an unbounded stream of mini-batches flows one at a time into a single evolving set of centroids, each batch nudging the centroids in place before being discarded.</desc>
<text x="20" y="26" font-size="14" stroke="none" fill="currentColor" font-weight="bold">batch fit</text>
<rect x="20" y="40" width="180" height="54" rx="6" opacity="0.9"/>
<text x="110" y="72" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">whole dataset</text>
<line x1="200" y1="67" x2="300" y2="67"/>
<circle cx="350" cy="67" r="34"/>
<text x="350" y="64" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">k</text>
<text x="350" y="80" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">centroids</text>
<text x="20" y="156" font-size="14" stroke="none" fill="currentColor" font-weight="bold">streaming partial_fit</text>
<rect x="20" y="172" width="44" height="44" rx="4" opacity="0.5"/>
<rect x="78" y="172" width="44" height="44" rx="4" opacity="0.65"/>
<rect x="136" y="172" width="44" height="44" rx="4" opacity="0.8"/>
<rect x="194" y="172" width="44" height="44" rx="4" opacity="0.9"/>
<text x="42" y="198" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">b1</text>
<text x="100" y="198" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">b2</text>
<text x="158" y="198" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">b3</text>
<text x="216" y="198" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">b4</text>
<text x="262" y="198" font-size="13" stroke="none" fill="currentColor">...</text>
<line x1="290" y1="194" x2="320" y2="194"/>
<circle cx="360" cy="194" r="36"/>
<text x="360" y="190" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">k centroids</text>
<text x="360" y="206" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">nudged</text>
<path d="M392 176 A30 30 0 1 1 396 200" stroke-dasharray="4 4" opacity="0.6"/>
<text x="430" y="172" font-size="11" stroke="none" fill="currentColor" opacity="0.8">update in place</text>
<text x="430" y="206" font-size="11" stroke="none" fill="currentColor" opacity="0.8">each batch updates, then is discarded</text>
</svg>
<figcaption>Two regimes. Batch fit pulls the whole dataset into one fit. Streaming partial_fit lets an unbounded sequence of mini-batches each nudge the same evolving centroids in place, holding only one batch at a time.</figcaption>
</figure>

## Choosing k without eyeballing

One practical note survives the move to a cluster: you still have to pick `k`. The elbow method (sweep `k`, plot the within-cluster sum of squares, look for the bend) usually means squinting at a chart, but you can automate the squint. For each `k`, record the inertia, then pick the `k` whose point on the curve sits farthest, in perpendicular distance, from the straight line joining the first and last points. A no-eyeball elbow detector:

```python
import numpy as np

def find_elbow_point(k_range, inertias):
    points = np.array(list(zip(k_range, inertias)), dtype=float)
    line_vec = points[-1] - points[0]
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_first = points - points[0]
    scalar_proj = vec_from_first @ line_vec_norm
    proj_points = np.outer(scalar_proj, line_vec_norm) + points[0]
    distances = np.linalg.norm(points - proj_points, axis=1)
    return int(k_range[int(np.argmax(distances))])
```

One storage habit matters far more on a cluster than on a laptop: store features as `float32` rather than `float64` when you can. Halving the per-point footprint is often the difference between a partition fitting in a worker's memory and spilling, which ties straight back to the stake at the top, because the whole reason you reached for a cluster was that the data stopped fitting. Standardizing and dropping outliers first (z-score, keep `z_scores < 3`) still helps the clustering itself, the same way it does on one machine.

## The pattern, and the limit

Strip away the engines and distributed k-means is one idea: figure out which step is embarrassingly parallel, run that step where the data already sits, and ship as little as possible per round. Here the assignment step parallelizes per partition, the update step is a tree-reduce of partial sums, and the only thing crossing the wire each iteration is the `k` centroids.

<figure>
<svg viewBox="0 0 640 300" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;color:inherit" fill="none" stroke="currentColor" stroke-width="2">
<title>Decision: distribute k-means or not</title>
<desc>A flow that starts by asking whether the data fits in one machine's RAM. If yes, use single-node MiniBatchKMeans and the cluster almost always loses. If no, distribute, then branch on Spark for JVM or huge data, Dask for NumPy or Pandas native workflows, or Incremental for streaming data that never fully lands.</desc>
<rect x="220" y="20" width="200" height="46" rx="6"/>
<text x="320" y="40" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">does it fit in one</text>
<text x="320" y="56" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">machine's RAM?</text>
<path d="M220 43 L120 43 L120 96" opacity="0.8"/>
<text x="150" y="36" font-size="11" stroke="none" fill="currentColor">yes</text>
<rect x="20" y="100" width="200" height="56" rx="6" opacity="0.9"/>
<text x="120" y="124" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">single-node</text>
<text x="120" y="142" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">MiniBatchKMeans</text>
<text x="120" y="178" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">the cluster almost always loses</text>
<path d="M420 43 L520 43 L520 96" opacity="0.8"/>
<text x="480" y="36" font-size="11" stroke="none" fill="currentColor">no: distribute</text>
<rect x="420" y="100" width="200" height="40" rx="6"/>
<text x="520" y="124" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">which engine?</text>
<path d="M460 140 L420 200" opacity="0.7"/>
<path d="M520 140 L520 200" opacity="0.7"/>
<path d="M580 140 L620 200" opacity="0.7"/>
<rect x="350" y="204" width="100" height="56" rx="5" opacity="0.85"/>
<text x="400" y="228" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">Spark</text>
<text x="400" y="246" font-size="9" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">JVM / huge</text>
<rect x="470" y="204" width="100" height="56" rx="5" opacity="0.85"/>
<text x="520" y="228" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">Dask</text>
<text x="520" y="246" font-size="9" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">NumPy native</text>
<rect x="588" y="204" width="44" height="56" rx="5" opacity="0.85"/>
<text x="610" y="226" font-size="10" stroke="none" fill="currentColor" text-anchor="middle">stream</text>
<text x="610" y="244" font-size="8" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">Incremental</text>
</svg>
<figcaption>The decision that matters. The first question is whether the data fits in one machine. If it does, a single-node MiniBatchKMeans almost always beats a cluster. Only when it does not do you distribute, and then the choice is ecosystem: Spark, Dask, or Incremental for streams.</figcaption>
</figure>

That same shape, parallelize the per-record step, reduce small summaries, broadcast a small model, recurs across a lot of distributed machine learning. Gradient descent over sharded data does it, distributed gradient boosting does it, and federated training does it. K-means is just the version where the model is small enough that you can watch the trick happen.

And none of it is free. You pay shuffle cost on every reduce and coordination cost across workers, and that overhead is real, so for anything that fits on one machine, a single-node `MiniBatchKMeans` will beat a cluster, because the cluster spends its first seconds just agreeing on who holds what. Distribution is not a default. It is what you reach for when the data has already won the argument against your RAM. Data stays pinned, and the centroids are the only thing that crosses the wire.
