---
layout: single
title: "distributed k-means - clustering datasets that outgrow one machine"
date: 2020-05-20T10:00:00+05:30
last_modified_at: 2020-05-20T10:00:00+05:30
categories: [blog]
tags: [machine-learning, clustering, distributed-computing, python, pyspark, dask]
excerpt: "A mental model for when k-means outgrows one machine, and how the data actually moves when it does"
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

One point in this post is eight `float64` numbers, which is 64 bytes. A billion of them is 64 GB. That number is the whole stake of the article, and it is arithmetic, not a measurement.

I want to be exact about that up front, because everything below leans on it. There is no billion-point job behind this post. I constructed one on paper: a billion points, eight dimensions each, `float64`, sitting in object storage. Every quantity I give for that dataset is multiplication that can be redone by hand, and I flag it as arithmetic each time. No stopwatch appears anywhere below.

What the constructed dataset is good for is one question. What changes about k-means when the array will not fit in one machine's memory? `sklearn.cluster.KMeans` wants a materialized array. Hand it 64 GB on a 16 GB laptop and the fit call fails inside the allocation, ahead of the first assignment step. The algorithm is unchanged. The machine is the constraint.

If you have called KMeans on a dataframe that fit in RAM, that is the only prerequisite. If you have not, the first section rebuilds the loop out of three points and two centroids, so this reads cold either way. After that I add exactly one new thing per section, and the machinery gets its standard names only once you have watched it work.

## One point, one centroid, one distance

Three points on a line: `p1` at 1, `p2` at 3, `p3` at 10. Two centroids on the same line: `c1` at 0, `c2` at 8. That is the entire dataset. It fits in a sentence, which is why I picked it.

k-means is trying to do one thing with those centroids: place them so that the total squared distance from every point to its own centroid is as small as it can get. That total has a name, inertia, and it is the number the algorithm is grinding down. Hold onto it, because it is the only scoreboard in the room.

The assignment step asks one question per point. Which centroid is nearer?

Take `p2` at 3. Its distance to `c1` is 3 and its distance to `c2` is 5. Three is smaller, so `p2` joins `c1`. Run the same comparison twice more. `p1` at 1 sits 1 from `c1` and 7 from `c2`, so it joins `c1`. `p3` at 10 sits 10 from `c1` and 2 from `c2`, so it joins `c2`.

<figure>
<svg viewBox="0 0 640 210" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;height:auto;color:inherit" aria-labelledby="assign-title assign-desc" fill="none" stroke="currentColor" stroke-width="1.8">
<title id="assign-title">Assigning one point by comparing two distances</title>
<desc id="assign-desc">Shows which centroid the middle point is closer to, by comparing two distances drawn above a number line.</desc>
<text x="20" y="24" font-size="13" stroke="none" fill="currentColor" font-weight="bold">the whole dataset</text>
<line x1="50" y1="140" x2="610" y2="140" opacity="0.55"/>
<rect x="43" y="133" width="14" height="14" rx="2"/>
<rect x="411" y="133" width="14" height="14" rx="2"/>
<circle cx="96" cy="140" r="5" fill="currentColor" stroke="none"/>
<circle cx="188" cy="140" r="5" fill="currentColor" stroke="none"/>
<circle cx="510" cy="140" r="5" fill="currentColor" stroke="none"/>
<text x="96" y="124" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">p1 = 1</text>
<text x="188" y="124" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">p2 = 3</text>
<text x="510" y="124" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">p3 = 10</text>
<text x="50" y="168" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">c1 = 0</text>
<text x="418" y="168" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">c2 = 8</text>
<path d="M188 128 Q119 70 50 128" opacity="0.75"/>
<path d="M188 128 Q303 44 418 128" opacity="0.75"/>
<text x="119" y="60" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">3</text>
<text x="303" y="38" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">5</text>
<text x="330" y="192" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">3 is smaller, so p2 joins c1</text>
</svg>
<figcaption>Three points and two centroids on a line, with both distances drawn for the middle point.</figcaption>
</figure>

Now notice what that comparison needed. It needed `p2`, it needed the two centroids, and it needed nothing further. Not `p1`. Not `p3`. Not the other 999,999,997 points in the constructed version. The assignment of a point is a function of the point, the `k` centroids, and nothing else in the dataset.

That independence is the load-bearing fact of this post. Nothing has been distributed yet, and the fact does not have a name yet either.

The update step then moves each centroid to the mean of the points that chose it. `c1` becomes the mean of 1 and 3, which is 2. `c2` becomes the mean of 10, which is 10. Both moved, and inertia fell. Repeat the assignment against the new centroids until nothing moves, which for this toy takes one more round.

```python
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1.0], [3.0], [10.0]])
km = KMeans(n_clusters=2, init=np.array([[0.0], [8.0]]), n_init=1)
km.fit(X)

km.cluster_centers_   # array([[ 2.], [10.]])
km.labels_            # array([0, 0, 1])
```

## Two machines and the same three points

One rung up. Put `p1` and `p2` on machine A, put `p3` on machine B, and leave them where they are.

Send both centroids to both machines. That is two numbers, 0 and 8. Machine A assigns its two points using its two points and those two numbers; machine B does the same for its one. Neither machine needs anything the other is holding, which is the fact from the previous section doing its job.

Then the update, which is where the interesting failure lives. The obvious move is to have each machine compute its own mean and average the results. That is wrong, and the toy shows why in one line. Machine A's cluster-1 mean is 2, taken over two points. Machine B holds no cluster-1 points, so its cluster-1 mean does not exist. Even in the friendlier case where both machines hold some, averaging the two means weights a machine holding one point the same as a machine holding a million.

The mean is not composable, but the pair of a sum and a count is. Machine A emits, for cluster 1, the sum 4 and the count 2. Machine B emits, for cluster 2, the sum 10 and the count 1. Add the sums, add the counts, divide once at the top: `c1` is 4/2, which is 2, and `c2` is 10/1, which is 10. Those are the two numbers a single machine produced a moment ago.

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
<figcaption>Split the three points and the answer does not move: c1 = 2 and c2 = 10, exactly what one machine computes. Four summary numbers travel, and none of the points do.</figcaption>
</figure>

That figure is the previous one cut down the middle, with a reduce bolted underneath. Nothing else about the algorithm changed.

Now the names, which I have been holding back. The assignment step is **embarrassingly parallel**: the work splits with no communication across the splits. Pushing the centroids out to both machines is a **broadcast**. Folding the per-machine sums and counts into one pair per cluster is a **tree-reduce**, which meets partial results in pairs the way a knockout bracket does, so its depth grows with the logarithm of the worker count rather than with the count itself. A reduce whose result immediately broadcasts back out is an **all-reduce**.

Four terms, and they describe the two paragraphs above without adding anything to them.

## Now make it a billion points

Another rung up, and the procedure does not change. There is just three orders of magnitude more of it.

Split the billion points across workers and pin them there. Each worker holds a partition, assigns its own partition against the current centroids, and emits per-cluster sums and counts. The partition itself does not travel, in any iteration.

Here is the arithmetic that makes the arrangement worth its complexity. Going out: a billion points times 8 dimensions times 8 bytes per `float64` is 64 GB. Crossing the wire per iteration, with `k` set to 5: five centroids times 8 dimensions times 8 bytes is 320 bytes. The ratio between them is $2 \times 10^{8}$.

Coming back up, each worker emits `k` sums and `k` counts. Five sums of eight `float64` is 320 bytes, five `int64` counts is 40 bytes, so 360 bytes per worker per iteration. At a hundred workers that is roughly 36 KB climbing the reduce tree. Set against 64 GB of pinned points, both directions round to zero. A broadcast is cheap the way a memo is cheap: one small thing copied onto many desks, when the alternative on the table was moving the filing cabinets.

The reason the ratio is that lopsided is the shape of the per-iteration cost:

$$
n \times k \times d \quad \text{distance computations per iteration}
$$

`k` and `d` are small and you chose them. `n` is the one you were handed, and it is the only term that grew.

One storage decision belongs here, because it is the same multiplication. Store the features as `float32` rather than `float64` and the constructed dataset goes from 64 GB to 32 GB. On a laptop that is a nicety. Across a cluster it decides whether a partition sits in a worker's memory or spills to that worker's disk, and spilling is the thing the cluster was supposed to avoid.

If the mechanism is what you came for, that is all of it, and this is a fine place to stop. The three sections below are the parts that bite once you run it.

## The step that did not get cheaper

The loop distributes cleanly. Initialization does not, and that asymmetry is the main practical surprise in distributed k-means.

Classic k-means++ picks seeds one at a time. Each new centroid is drawn with probability proportional to its squared distance from the nearest seed already chosen, so the probabilities depend on the current seed set, so the data has to be scanned before the next one can be drawn. That is `k` sequential passes. On one machine it is `k` passes over an array. On a cluster each pass is a full distributed scan, and the whole stack of them happens before the first assignment step of the actual loop.

Line the two costs up. Seeding costs `k` distributed passes. The loop that follows costs `i` iterations, and `i` is usually a few tens. At `k` of 5000 and `i` of 30, seeding is about 167 times as many distributed passes as the clustering it was preparing for. Be careful what that ratio counts. It is passes, not floating-point work: a seeding pass scores every point against one new center, where a Lloyd iteration scores every point against all `k`, so the arithmetic per seeding pass is much lighter. On a cluster the pass is still the unit that hurts, because each one is a scheduled job over the whole dataset with its own stragglers to wait on.

The fix is **k-means||**, from [Bahmani et al. 2012](https://arxiv.org/abs/1203.6402), the "Scalable K-Means++" paper. It does not draw one seed per pass, it draws several. Each pass oversamples candidate centers, still with probability proportional to squared distance. The paper's argument is that a pass of that kind cuts the expected remaining cost by roughly a constant factor, and a quantity shrinking by a constant factor per round reaches its target in a logarithmic number of rounds. Worth being precise about what that logarithm is in: the bound is logarithmic in the starting clustering cost, not in `k` and not in `n`. That starting cost is inertia again, the same scoreboard from the first section, measured against the single random center the algorithm opens with. The intuition most people carry away is log `k`, and that is not what the paper proves.

That leaves a candidate pool bigger than `k`, and collapsing it is not free. Every point in the dataset has to be scored against the pool so that each candidate can be weighted by how many points it is nearest to. That is one more distributed pass, not zero. Only then is the weighted pool, which is small, reclustered down to `k` seeds on a single machine. So the honest count is a logarithmic number of oversampling passes plus one weighting pass, set against `k` sequential ones. Still the trade you want at large `k`, and still not the free lunch the summaries usually imply.

In practice the round count is smaller than the theory has to allow for. Spark MLlib defaults to two oversampling steps, which is a pragmatic setting rather than a reading of the bound.

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
<figcaption>Dots mark a full pass over the distributed data. The left stack grows with k; the right column does not, and its true cost is the oversampling passes plus the weighting pass, not the oversampling passes alone.</figcaption>
</figure>

```python
from pyspark.ml.clustering import KMeans as SparkKMeans

# initMode picks the seeder. initSteps is the number of
# oversampling passes k-means|| makes before the weighting pass.
kmeans = SparkKMeans(k=5, initMode="k-means||", initSteps=2)
```

## Two libraries that already do this

PySpark MLlib and Dask-ML both implement the data movement from the three sections above. Picking between them is an ecosystem question, and I would treat any per-iteration performance argument between the two with suspicion unless it arrives with the hardware attached.

Reach for Spark when the data already lives in the JVM ecosystem. Load it, assemble the feature column, fit. The `inferSchema` option is load-bearing: without it every column reads back as a string and `VectorAssembler` refuses the frame.

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

Reach for Dask when the workflow is already NumPy and pandas native. The API mirrors scikit-learn closely enough that the diff against single-machine code is a couple of lines. Both libraries made the same seeding choice, incidentally: `dask_ml.cluster.KMeans` defaults to `init="k-means||"` exactly as Spark does, so the previous section describes what either one is doing before your first iteration. I wrote up the delayed-graph side of Dask in [an earlier post]({{ site.baseurl }}/posts/dask-parallel-light-curves/); this is the array side of the same library.

```python
import dask.dataframe as dd
from dask_ml.cluster import KMeans as DaskKMeans

df = dd.read_csv(data_path)
X = df[numeric_cols].to_dask_array(lengths=True)

# init_max_iter is Dask-ML's equivalent of initSteps: the number of
# k-means|| oversampling rounds before the weighting pass.
kmeans = DaskKMeans(n_clusters=5, max_iter=100, init_max_iter=3)
kmeans.fit(X)
labels = kmeans.predict(X)
```

If the pipeline is JVM and the data is already in Spark, stay in Spark. If the pipeline is Python and the data is already in arrays, stay in Dask. The cost of crossing that boundary, which means serializing a large dataset out of one runtime and into the other, tends to exceed whatever difference was being chased across it.

## When there is no dataset to fit

Everything so far assumes the data eventually lands somewhere and you fit over the whole of it. Drop that assumption and you have a stream, where batches arrive and each one is seen once.

The tool for that is `partial_fit`, which updates the centroids from one mini-batch and then forgets the batch. It treats data the way a running average treats a sensor feed. Keep the estimate, drop the reading.

There is a sharp edge here, because the obvious call does not exist:

```python
>>> from dask_ml.cluster import KMeans   # dask-ml 1.4.0
>>> KMeans(n_clusters=3).partial_fit(X)
AttributeError: 'KMeans' object has no attribute 'partial_fit'
```

That class implements the full-dataset k-means|| fit, not the incremental API. For a stream you drive yourself, use scikit-learn's `MiniBatchKMeans` directly.

```python
from sklearn.cluster import MiniBatchKMeans

def streaming_kmeans(data_stream, k=3):
    model = MiniBatchKMeans(n_clusters=k, random_state=42)
    for batch in data_stream:
        model.partial_fit(batch)   # this batch moves the centroids, then is dropped
    return model
```

When you own that loop, wrapping it in `dask_ml.wrappers.Incremental` adds nothing. `Incremental` is for the other case, where the data is already one large Dask array and you want `partial_fit` mapped across its blocks rather than written out by hand.

```python
from sklearn.cluster import MiniBatchKMeans
from dask_ml.wrappers import Incremental   # dask-ml 1.4.0

est = Incremental(MiniBatchKMeans(n_clusters=3, random_state=42))
est.fit(X)   # X is a Dask array; Incremental calls partial_fit block by block
```

<figure>
<svg viewBox="0 0 640 292" role="img" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;height:auto;color:inherit" aria-labelledby="stream-title stream-desc" fill="none" stroke="currentColor" stroke-width="1.8">
<title id="stream-title">Fitting a whole dataset against fitting an arriving stream</title>
<desc id="stream-desc">Contrasts fitting a whole dataset at once with updating the same centroids from one arriving batch at a time.</desc>
<text x="20" y="24" font-size="13" stroke="none" fill="currentColor" font-weight="bold">batch fit</text>
<rect x="20" y="40" width="200" height="54" rx="6" opacity="0.9"/>
<text x="120" y="72" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">p1, p2, p3 at once</text>
<line x1="220" y1="67" x2="300" y2="67" opacity="0.75"/>
<circle cx="346" cy="67" r="34"/>
<text x="346" y="64" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">c1, c2</text>
<text x="346" y="80" font-size="9" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">one fit</text>
<text x="400" y="71" font-size="11" stroke="none" fill="currentColor" opacity="0.8">the dataset has to exist</text>
<text x="20" y="160" font-size="13" stroke="none" fill="currentColor" font-weight="bold">streaming partial_fit</text>
<rect x="20" y="176" width="48" height="44" rx="4" opacity="0.45"/>
<rect x="80" y="176" width="48" height="44" rx="4" opacity="0.7"/>
<rect x="140" y="176" width="48" height="44" rx="4" opacity="0.9"/>
<text x="44" y="203" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">p1</text>
<text x="104" y="203" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">p2</text>
<text x="164" y="203" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">p3</text>
<text x="206" y="204" font-size="13" stroke="none" fill="currentColor">...</text>
<line x1="232" y1="198" x2="300" y2="198" opacity="0.75"/>
<circle cx="346" cy="198" r="36"/>
<text x="346" y="195" font-size="11" stroke="none" fill="currentColor" text-anchor="middle">c1, c2</text>
<text x="346" y="211" font-size="9" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">nudged</text>
<path d="M380 178 A30 30 0 1 1 384 206" stroke-dasharray="4 4" opacity="0.6"/>
<text x="400" y="174" font-size="11" stroke="none" fill="currentColor" opacity="0.8">one batch updates,</text>
<text x="400" y="192" font-size="11" stroke="none" fill="currentColor" opacity="0.8">then is discarded</text>
<text x="400" y="222" font-size="11" stroke="none" fill="currentColor" opacity="0.8">nothing is held but the estimate</text>
</svg>
<figcaption>Two things live in memory in the lower path: the batch currently in hand, and k centroids. Everything else has already been thrown away.</figcaption>
</figure>

This is the regime where "it fits across the cluster" stops meaning anything, because there is no fixed dataset, no fixed size, only an arrival rate.

## What the ladder was for

Read backwards, the post is one move shown at three sizes: one machine, two machines, a thousand. Find the step whose work splits per record. Run it where the records already sit. Ship a summary rather than the records.

Data-parallel SGD has the same shape with a different payload. Each worker computes gradients on its own shard, an all-reduce sums them, the summed gradient goes back out, and the next step happens. What differs is the size of the thing crossing the wire. For k-means it is `k` times `d` floats, and it stays that size whether the cluster holds a million points or a billion. For SGD it is the model's full gradient vector, which grows with the model, which is why a framework like Horovod moves gradients with a ring all-reduce, where each worker only ever talks to its two neighbours and the bytes each one sends stay flat as the cluster grows. Nobody has ever needed that for five centroids. K-means is the version of the pattern where the model is small enough to watch.

Two things I left out on purpose. There is nothing here about choosing `k`: the elbow sweep, the silhouette score, and the automated versions of both are their own subject, and they mostly do not change under distribution, beyond the obvious point that a sweep multiplies everything above by the number of values tried. There is also nothing about skewed partitions, where one worker ends up holding most of a cluster's points and the reduce sits waiting on it. That one I would want measurements for before writing about it, and I do not have any.

What is left to pay for is the coordination. Workers have to agree on who holds what, the broadcast has to land, and the reduce has to complete before the next iteration can start. That is a fixed cost, paid before any clustering happens. Below some size it is a cost with nothing to show for it, and a single-node `MiniBatchKMeans` will finish first, though that one is trading inertia for speed and is not solving quite the same problem. The gap between a constructed 64 GB and a 16 GB laptop is what pays for the machinery. Underneath that gap, the machinery is the problem.

[Part one: parallelizing the feature loop]({{ site.baseurl }}/posts/dask-parallel-light-curves/){: .btn-soft} [Scalable K-Means++, Bahmani et al.](https://arxiv.org/abs/1203.6402){: .btn-soft} [Spark MLlib clustering](https://spark.apache.org/docs/latest/ml-clustering.html){: .btn-soft}
