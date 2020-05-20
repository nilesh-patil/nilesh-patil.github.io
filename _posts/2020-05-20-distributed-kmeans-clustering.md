---
layout: single
title: "distributed k-means - clustering datasets that outgrow one machine"
date: 2020-05-20T10:00:00+05:30
last_modified_at: 2020-05-20T10:00:00+05:30
categories: [blog]
tags: [machine-learning, clustering, distributed-computing, python, pyspark, dask]
excerpt: "A mental model for when k-means outgrows one machine, and how the data moves when it does"
math: true
redirect_from:
  - /blog/distributed-kmeans-clustering/
header:
  overlay_image: /images/blog/headers/distributed-kmeans-clustering.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/distributed-kmeans-clustering.jpg
---

<style>
a.btn-soft {
  display: inline-block;
  padding: 0.5em 1.1em;
  margin: 0.2em 0.3em 0.2em 0;
  border: 1px solid currentColor;
  border-radius: 6px;
  font-size: 0.85em;
  line-height: 1.4;
  text-decoration: none;
  opacity: 0.85;
  transition: opacity 0.15s ease;
}
a.btn-soft:hover {
  opacity: 1;
  text-decoration: none;
}
</style>

One point in this post is eight `float64` numbers, which is 64 bytes, and a billion of them is 64 GB. The dataset is one I constructed on paper (a billion points, eight dimensions each, sitting in object storage), so every large number below is arithmetic you can redo by hand.

I use it to ask one thing: what changes about k-means when the array will not fit in one machine's memory? `sklearn.cluster.KMeans` wants a materialized array, and a 16 GB laptop cannot hold a 64 GB one, so the fit never reaches its first assignment step.

## Three points and two centroids

Three points on a line: `p1` at 1, `p2` at 3, `p3` at 10. Two centroids on the same line: `c1` at 0, `c2` at 8. That is the entire dataset, picked because it fits in a sentence.

k-means tries to place those centroids so that the total squared distance from every point to its own centroid is as small as it can get. That total is called inertia.

The assignment step asks, for each point, which centroid is nearer.

Take `p2` at 3. Its distance to `c1` is 3 and its distance to `c2` is 5. Three is smaller, so `p2` joins `c1`. Run the same comparison twice more: `p1` joins `c1`, and `p3` joins `c2`.

The comparison for `p2` used `p2` and the two centroids. It never looked at `p1` or `p3`, and in the billion-point version it would never look at the other 999,999,997 points either.

The update step then moves each centroid to the mean of the points that chose it. `c1` becomes the mean of 1 and 3, which is 2. `c2` becomes the mean of 10, which is 10. Both moved, and inertia fell.

## Two machines and the same three points

Put `p1` and `p2` on machine A, put `p3` on machine B, and leave them where they are.

Send both centroids to both machines. That is two numbers, 0 and 8. Machine A assigns its two points using its two points and those two numbers; machine B does the same for its one.

The update is harder. The obvious move is to have each machine compute its own mean and average the results, and the toy already breaks it. Machine A's cluster-1 mean is 2, taken over two points. Machine B holds no cluster-1 points, so its cluster-1 mean does not exist. Even in the friendlier case where both machines hold some, averaging the two means weights a machine holding one point the same as a machine holding a million.

The pair of a sum and a count is composable. Machine A emits, for cluster 1, the sum 4 and the count 2. Machine B emits, for cluster 2, the sum 10 and the count 1. Add the sums, add the counts, divide once at the top: `c1` is 4/2, which is 2, and `c2` is 10/1, which is 10. Those are the two numbers a single machine produced a moment ago.

<figure>
<svg viewBox="0 0 640 356" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;height:auto;color:inherit" aria-labelledby="split-title split-desc" fill="none" stroke="currentColor" stroke-width="1.8">
<title id="split-title">The same three points, split across two machines</title>
<desc id="split-desc">Shows how splitting the points across two machines still yields the same centroids, by combining per-cluster sums and counts.</desc>
<rect x="220" y="16" width="200" height="42" rx="6"/>
<text x="320" y="42" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">send c1 = 0, c2 = 8</text>
<path d="M296 58 L150 100" opacity="0.65"/>
<path d="M344 58 L490 100" opacity="0.65"/>
<rect x="30" y="100" width="230" height="88" rx="6" opacity="0.9"/>
<text x="145" y="124" font-size="12" stroke="none" fill="currentColor" text-anchor="middle" font-weight="bold">machine A</text>
<text x="145" y="146" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">p1 = 1, p2 = 3</text>
<text x="145" y="166" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">both nearer c1</text>
<rect x="380" y="100" width="230" height="88" rx="6" opacity="0.9"/>
<text x="495" y="124" font-size="12" stroke="none" fill="currentColor" text-anchor="middle" font-weight="bold">machine B</text>
<text x="495" y="146" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">p3 = 10</text>
<text x="495" y="166" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">nearer c2</text>
<path d="M160 188 L262 242" opacity="0.65"/>
<path d="M480 188 L378 242" opacity="0.65"/>
<text x="34" y="222" font-size="10" stroke="none" fill="currentColor" opacity="0.85">cluster 1: sum 4, count 2</text>
<text x="606" y="222" font-size="10" stroke="none" fill="currentColor" text-anchor="end" opacity="0.85">cluster 2: sum 10, count 1</text>
<rect x="230" y="244" width="180" height="42" rx="6"/>
<text x="320" y="270" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">add sums, add counts</text>
<path d="M320 286 L320 306" opacity="0.65"/>
<rect x="186" y="306" width="268" height="40" rx="6" opacity="0.9"/>
<text x="320" y="331" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">c1 = 4/2 = 2, c2 = 10/1 = 10</text>
<path d="M454 326 C600 318 626 120 420 44" stroke-dasharray="5 4" opacity="0.55"/>
<text x="545" y="80" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">next iteration</text>
</svg>
<figcaption>The same three points, split across two machines.</figcaption>
</figure>

Each of those steps has a standard name. The assignment step is **embarrassingly parallel**: the work splits with no communication across the splits. Pushing the centroids out to both machines is a **broadcast**. Folding the per-machine sums and counts into one pair per cluster is a **tree-reduce**, which meets partial results in pairs the way a knockout bracket does, so its depth grows with the logarithm of the worker count. A reduce whose result immediately broadcasts back out is an **all-reduce**.

## Now make it a billion points

Split the billion points across workers and pin them there. Each worker holds a partition, assigns its own partition against the current centroids, and emits per-cluster sums and counts.

The pinned data is a billion points times 8 dimensions times 8 bytes per `float64`, which is 64 GB. The broadcast per iteration, with `k` set to 5, is five centroids times 8 dimensions times 8 bytes, which is 320 bytes. The ratio between them is $2 \times 10^{8}$.

Coming back up, each worker emits `k` sums and `k` counts. Five sums of eight `float64` is 320 bytes, five `int64` counts is 40 bytes, so 360 bytes per worker per iteration. At a hundred workers that is roughly 36 KB climbing the reduce tree.

The per-iteration cost:

$$
n \times k \quad \text{distance computations per iteration, each over } d \text{ dimensions}
$$

`k` and `d` are small, and `n` is the only term that grew.

Store the features as `float32` rather than `float64` and the dataset goes from 64 GB to 32 GB. Across a cluster, halving the bytes can decide whether a partition sits in a worker's memory or spills to that worker's disk.

## Seeding

Initialization, in its classic form, is sequential: k-means++ picks seeds one at a time. Each new centroid is drawn with probability proportional to its squared distance from the nearest seed already chosen, so the probabilities depend on the current seed set. So the data has to be scanned before the next one can be drawn. That is `k` sequential passes, and where one machine makes `k` passes over an array, a cluster makes `k` full distributed scans, all of them before the first assignment step of the actual loop.

Seeding costs `k` distributed passes and the loop that follows costs `i` iterations. At the `k` of 5 used above, that is five passes: a uniform random draw for the first seed, then four distance scans. With `k` of 5000 and `i` of 30, seeding is about 167 times as many distributed passes as the clustering it was preparing for. That ratio counts passes. The floating-point work per pass is much lighter in seeding, since a seeding pass scores every point against one new center where a Lloyd iteration scores every point against all `k`. On a cluster the pass count still matters, because each pass is a scheduled job over the whole dataset and has to wait for its slowest worker.

The fix is **k-means||**, from [Bahmani et al. 2012](https://arxiv.org/abs/1203.6402), the "Scalable K-Means++" paper. It draws several seeds per pass: each pass oversamples candidate centers, still with probability proportional to squared distance. The paper's argument is that a pass of that kind cuts the expected remaining cost by roughly a constant factor. A quantity shrinking by a constant factor per round reaches its target in a logarithmic number of rounds. The logarithm is of the starting clustering cost, which is inertia again, measured against the single random center the algorithm opens with. It is tempting to remember the bound as log `k`, but neither `k` nor `n` is the quantity inside the logarithm.

That leaves a candidate pool bigger than `k`, and collapsing it costs one more distributed pass. Every point in the dataset has to be scored against the pool so that each candidate can be weighted by how many points it is nearest to. Only then is the weighted pool, which is small, reclustered down to `k` seeds on a single machine. The full count is a logarithmic number of oversampling passes plus one weighting pass, against `k` sequential passes for k-means++. At large `k` that is far fewer passes, as long as the weighting pass is counted with the rest.

Spark MLlib fixes the number of oversampling steps at two by default, whatever the bound would give for the data.

<figure>
<svg viewBox="0 0 640 350" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;height:auto;color:inherit" aria-labelledby="seed-title seed-desc" fill="none" stroke="currentColor" stroke-width="1.8">
<title id="seed-title">Seeding passes over distributed data</title>
<desc id="seed-desc">Compares seeding cost: one seed per full scan against a few oversampling passes, a weighting pass, and a local reduction.</desc>
<text x="20" y="24" font-size="13" stroke="none" fill="currentColor" font-weight="bold">k-means++</text>
<text x="20" y="42" font-size="10" stroke="none" fill="currentColor" opacity="0.75">one seed per full scan</text>
<rect x="20" y="54" width="250" height="26" rx="4" opacity="0.9"/>
<rect x="20" y="88" width="250" height="26" rx="4" opacity="0.75"/>
<rect x="20" y="122" width="250" height="26" rx="4" opacity="0.6"/>
<rect x="20" y="156" width="250" height="26" rx="4" opacity="0.4"/>
<circle cx="34" cy="67" r="3" fill="currentColor" stroke="none"/>
<circle cx="44" cy="67" r="3" fill="currentColor" stroke="none"/>
<circle cx="54" cy="67" r="3" fill="currentColor" stroke="none"/>
<circle cx="34" cy="101" r="3" fill="currentColor" stroke="none"/>
<circle cx="44" cy="101" r="3" fill="currentColor" stroke="none"/>
<circle cx="54" cy="101" r="3" fill="currentColor" stroke="none"/>
<circle cx="34" cy="135" r="3" fill="currentColor" stroke="none"/>
<circle cx="44" cy="135" r="3" fill="currentColor" stroke="none"/>
<circle cx="54" cy="135" r="3" fill="currentColor" stroke="none"/>
<text x="165" y="71" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">full scan, +1 seed</text>
<text x="165" y="105" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">full scan, +1 seed</text>
<text x="165" y="139" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">full scan, +1 seed</text>
<text x="145" y="205" font-size="12" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.8">and so on, k times</text>
<line x1="320" y1="40" x2="320" y2="330" stroke-dasharray="4 5" opacity="0.35"/>
<text x="360" y="24" font-size="13" stroke="none" fill="currentColor" font-weight="bold">k-means||</text>
<text x="360" y="42" font-size="10" stroke="none" fill="currentColor" opacity="0.75">several candidates per pass</text>
<rect x="360" y="54" width="250" height="30" rx="4" opacity="0.9"/>
<rect x="360" y="92" width="250" height="30" rx="4" opacity="0.75"/>
<circle cx="374" cy="69" r="3" fill="currentColor" stroke="none"/>
<circle cx="384" cy="69" r="3" fill="currentColor" stroke="none"/>
<circle cx="394" cy="69" r="3" fill="currentColor" stroke="none"/>
<circle cx="374" cy="107" r="3" fill="currentColor" stroke="none"/>
<circle cx="384" cy="107" r="3" fill="currentColor" stroke="none"/>
<circle cx="394" cy="107" r="3" fill="currentColor" stroke="none"/>
<text x="505" y="73" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">pass: oversample</text>
<text x="505" y="111" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">pass: oversample</text>
<path d="M485 122 L485 148" opacity="0.65"/>
<rect x="390" y="152" width="190" height="32" rx="4"/>
<text x="485" y="172" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">candidate pool</text>
<path d="M485 184 L485 210" opacity="0.65"/>
<rect x="360" y="214" width="250" height="44" rx="4" opacity="0.75"/>
<circle cx="374" cy="229" r="3" fill="currentColor" stroke="none"/>
<circle cx="384" cy="229" r="3" fill="currentColor" stroke="none"/>
<circle cx="394" cy="229" r="3" fill="currentColor" stroke="none"/>
<text x="500" y="233" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">pass: weight the pool</text>
<text x="500" y="249" font-size="9" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">still a full distributed scan</text>
<path d="M485 258 L485 284" opacity="0.65"/>
<rect x="390" y="288" width="190" height="44" rx="4"/>
<text x="485" y="308" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">recluster to k seeds</text>
<text x="485" y="324" font-size="9" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">on one machine</text>
</svg>
<figcaption>Dots mark a full pass over the distributed data. The left stack has one pass per seed, so it grows with k.</figcaption>
</figure>

```python
from pyspark.ml.clustering import KMeans as SparkKMeans

# initMode picks the seeder. initSteps is the number of
# oversampling passes k-means|| makes before the weighting pass.
kmeans = SparkKMeans(k=5, initMode="k-means||", initSteps=2)
```

## PySpark and Dask-ML

PySpark MLlib and Dask-ML both implement this data movement. I pick by ecosystem.

Use Spark when the data already lives in the JVM ecosystem. The `inferSchema` option is required here: without it every column reads back as a string and `VectorAssembler` refuses the frame. It costs one extra pass over the data.

```python
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.feature import VectorAssembler

df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(data_path))
features = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df_vec = features.transform(df)

model = SparkKMeans(k=5, maxIter=50, featuresCol="features").fit(df_vec)
predictions = model.transform(df_vec)
```

Dask fits when the workflow is already NumPy and pandas native. The API mirrors scikit-learn closely enough that the diff against single-machine code is a couple of lines. Both libraries made the same seeding choice: `dask_ml.cluster.KMeans` defaults to `init="k-means||"` as Spark does. I wrote up the delayed-graph side of Dask in [an earlier post]({{ site.baseurl }}/posts/dask-parallel-light-curves/); this is the array side of the same library.

```python
import dask.dataframe as dd
from dask_ml.cluster import KMeans as DaskKMeans

# persist, so later passes read worker memory instead of re-parsing the CSV
df = dd.read_csv(data_path)[numeric_cols].persist()
X = df.to_dask_array(lengths=True)

# init_max_iter is Dask-ML's equivalent of initSteps: a cap on the
# number of k-means|| oversampling rounds.
kmeans = DaskKMeans(n_clusters=5, max_iter=100, init_max_iter=3)
kmeans.fit(X)
labels = kmeans.predict(X)
```

Moving from one to the other means serializing the whole dataset out of one runtime and into the other, which is a full extra pass over the data before any clustering starts. I would stay where the data already is.

Everything so far assumes the data eventually lands somewhere and you fit over the whole of it. Drop that assumption and you have a stream, where batches arrive and each one is seen once.

The tool for that is `partial_fit`, which updates the centroids from one mini-batch and then discards the batch. On Dask-ML's own KMeans, though, that call does not exist:

```python
>>> from dask_ml.cluster import KMeans   # dask-ml 1.4.0
>>> KMeans(n_clusters=3).partial_fit(X)
AttributeError: 'KMeans' object has no attribute 'partial_fit'
```

That class implements only the full-dataset k-means|| fit. For a stream you drive yourself, use scikit-learn's `MiniBatchKMeans` directly.

```python
from sklearn.cluster import MiniBatchKMeans

def streaming_kmeans(data_stream, k=3):
    model = MiniBatchKMeans(n_clusters=k, random_state=42)
    for batch in data_stream:
        model.partial_fit(batch)   # this batch moves the centroids, then is dropped
    return model
```

When you own that loop, wrapping it in `dask_ml.wrappers.Incremental` adds nothing. `Incremental` is for the other case, where the data is already one large Dask array and you want `partial_fit` mapped across its blocks without writing the loop yourself.

```python
from sklearn.cluster import MiniBatchKMeans
from dask_ml.wrappers import Incremental   # dask-ml 1.4.0

est = Incremental(MiniBatchKMeans(n_clusters=3, random_state=42))
est.fit(X)   # X is a Dask array; Incremental calls partial_fit block by block
```

## What carries over

Data-parallel SGD has the same shape as the two-machine toy. Each worker computes gradients on its own shard, an all-reduce sums them, the summed gradient goes back out, and the next step happens. The payload is what differs. For k-means it is `k` times `d` floats plus `k` counts, and it stays that size whether the cluster holds a million points or a billion. For SGD it is the model's full gradient vector, which grows with the model. A framework like Horovod moves gradients with a ring all-reduce, where each worker only ever talks to its two neighbours and the bytes each one sends stay flat as the cluster grows.

[Part one: parallelizing the feature loop]({{ site.baseurl }}/posts/dask-parallel-light-curves/){: .btn-soft} [Scalable K-Means++, Bahmani et al.](https://arxiv.org/abs/1203.6402){: .btn-soft} [Spark MLlib clustering](https://spark.apache.org/docs/latest/ml-clustering.html){: .btn-soft}
