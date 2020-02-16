---
layout: single
title: "Introduction to dask - parallel feature extraction in variable-star data"
date: 2020-02-16T11:30:00+05:30
last_modified_at: 2020-02-16T11:30:00+05:30
categories: [blog]
tags: [python, dask, parallel-computing, astronomy, machine-learning]
excerpt: "I had 5,204 light curves, one slow for-loop, and eight cores watching one of them work. Here is how dask.delayed turned that loop into a graph the scheduler could spread out, and the one scheduler choice that actually mattered."
header:
  overlay_image: /images/blog/headers/dask-parallel-light-curves.jpg
  overlay_filter: 0.5
  teaser: /images/blog/headers/dask-parallel-light-curves.jpg
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
</style>

Dask is a library for running generic Python & its data ecosystem libraries in parallel. The core advantage is that you do not rewrite your code as a parallel program. You keep NumPy, pandas, the plain functions you already have, and Dask works out how to spread the work across your cores, or across a whole cluster, and runs it for you.

I went looking for it the day a feature-extraction script pinned one CPU core at 100% while the other seven sat at zero, for almost five minutes, on work where every star was independent of every other. Embarrassingly parallel, and I was grinding through it one core at a time.

It comes in a few pieces, and the first job was sorting out which one I actually needed. `dask.array` and `dask.dataframe` stand in for NumPy arrays and pandas frames when the data is too big to hold in memory. That was not my first problem. My data fit fine on a single instance. My problem was a slow loop over things that fit, and the piece for that is `dask.delayed`: it takes the function calls you would have written in the loop and records them as a graph of tasks the scheduler can run in parallel. No out-of-core machinery, no rewriting the science. That small corner of Dask is the whole of this post, and the arrays and bigger-than-memory tricks can wait for another day.

To learn it properly instead of skimming the docs and forgetting them, I took a real problem: about 5,200 light curves, one CSV per star, each turned into a row of features. Nothing about star number 5,000 depends on star number 4,999 - no ordering, no shared state, no reason to do them one at a time except that a `for` loop is the first thing your fingers reach for.

The fix comes to two lines. The one decision in them is which Dask scheduler you hand the work to.

## What the loop was doing

The light curves are real, from the LINEAR survey, packaged by [astroML](https://www.astroml.org/) with hand-checked class labels. Five kinds of star are in here: RR Lyrae of both the ab and c types, the two flavours of eclipsing binary (close contact pairs and the deeper Algol type), and a thin scattering of Delta Scuti. Each star is a string of brightness measurements on the irregular cadence a real survey actually produces, a hundred to a few hundred points spread across years, with the gaps you get from weather and daylight.

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/lightcurve_example.png" alt="A raw RR Lyrae light curve on the left, the same data folded at its recovered period on the right.">
  <figcaption>One RR Lyrae star from the sample. On the left, the raw light curve as observed: magnitude (brighter is up) against time, and it looks close to noise. On the right, the same points folded at the period the periodogram recovers, which snaps them into the characteristic fast-rise, slow-fade sawtooth. Finding that period is the expensive step.</figcaption>
</figure>

The per-star job is to read the file and boil the light curve down to a fixed vector of features: a period, an amplitude, a few shape and scatter statistics. The slow part is the period. To find it you run a Lomb-Scargle periodogram, which scans close to a hundred thousand trial frequencies across the multi-year baseline and asks how well each one explains the data, so that scan is most of the per-star cost, and it is pure CPU.

Does the expensive step even produce the right answer? The LINEAR catalogue ships its own validated periods, so I can hold mine up against them:

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/period_recovery.png" alt="Recovered period versus catalogue period, log-log: most points on the 1:1 line, a band of binaries on the half-period line.">
  <figcaption>My Lomb-Scargle period against the catalogue value, one point per star, log-log. About 53% of the 5,204 land within 2% of the catalogue, right on the 1:1 line. The dense band below it is not random failure: it is the eclipsing binaries, sitting at half the catalogue period because an eclipsing binary dips twice per orbit, so the periodogram locks onto the doubled frequency. That consistent half-period is a stable geometric alias, and the classifier in the next post is happy to use it as just another feature.</figcaption>
</figure>

Here is the unit of work, in full. It is not interesting, and that is the point:

```python
def features_for_file(path):
    """Read one light-curve file and return its features as a dict."""
    t, m, e = read_light_curve(path)
    row = {"star_id": star_id_from_path(path)}
    row.update(extract_features(t, m, e))
    return row
```

And here is the loop I started with, the baseline:

```python
rows = [features_for_file(p) for p in paths]
```

On my machine that list comprehension is about 56 ms per star, and across all 5,204 it takes 294 seconds, just under five minutes. One core the whole time.

## dask.delayed: stop calling the function

`dask.delayed` is a decorator that turns a normal function call into a *promise* to call it. When you wrap `features_for_file` in `delayed` and call it, nothing runs. Instead Dask records a little node: "here is a function, here is the argument, here is where the result will go." Do that for every file and you have not computed anything yet. You have built a graph.

```python
import dask

tasks = [dask.delayed(features_for_file)(p) for p in paths]
```

`tasks` is now 5,204 unstarted nodes. The function inside them is byte-for-byte the same `features_for_file` from above. I did not rewrite the science to parallelize it. I wrapped the call and stopped running the loop myself.

The shape of this particular graph is the simplest one there is: a row of independent leaves with nothing connecting them, because no star needs any other star's result.

<figure>
  <svg viewBox="0 0 640 250" role="img" aria-labelledby="graph-t graph-d" style="width:100%;max-width:640px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="1.6">
    <title id="graph-t">A directory of files becomes a graph of independent delayed tasks</title>
    <desc id="graph-d">On the left, a stack of light-curve files. Each one feeds a separate features_for_file task node, and the nodes do not connect to each other. On the right, the scheduler hands those independent tasks to eight worker processes, which return rows into one table.</desc>
    <text x="60" y="24" font-size="13" stroke="none" fill="currentColor" text-anchor="middle" font-weight="bold">files</text>
    <rect x="20" y="40" width="80" height="26" rx="4" opacity="0.85"/>
    <rect x="20" y="74" width="80" height="26" rx="4" opacity="0.85"/>
    <rect x="20" y="108" width="80" height="26" rx="4" opacity="0.85"/>
    <rect x="20" y="142" width="80" height="26" rx="4" opacity="0.5"/>
    <text x="60" y="196" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">5,204 of them</text>
    <text x="250" y="24" font-size="13" stroke="none" fill="currentColor" text-anchor="middle" font-weight="bold">delayed tasks</text>
    <rect x="200" y="40" width="100" height="26" rx="4"/>
    <rect x="200" y="74" width="100" height="26" rx="4"/>
    <rect x="200" y="108" width="100" height="26" rx="4"/>
    <rect x="200" y="142" width="100" height="26" rx="4" opacity="0.5"/>
    <text x="250" y="57" font-size="10" stroke="none" fill="currentColor" text-anchor="middle">features_for_file</text>
    <text x="250" y="91" font-size="10" stroke="none" fill="currentColor" text-anchor="middle">features_for_file</text>
    <text x="250" y="125" font-size="10" stroke="none" fill="currentColor" text-anchor="middle">features_for_file</text>
    <line x1="100" y1="53" x2="200" y2="53" opacity="0.6"/>
    <line x1="100" y1="87" x2="200" y2="87" opacity="0.6"/>
    <line x1="100" y1="121" x2="200" y2="121" opacity="0.6"/>
    <rect x="380" y="84" width="110" height="46" rx="6"/>
    <text x="435" y="104" font-size="12" stroke="none" fill="currentColor" text-anchor="middle">scheduler</text>
    <text x="435" y="120" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">keeps workers fed</text>
    <line x1="300" y1="87" x2="380" y2="100" opacity="0.6"/>
    <line x1="300" y1="121" x2="380" y2="110" opacity="0.6"/>
    <rect x="540" y="40" width="80" height="24" rx="4" opacity="0.9"/>
    <rect x="540" y="70" width="80" height="24" rx="4" opacity="0.9"/>
    <rect x="540" y="100" width="80" height="24" rx="4" opacity="0.9"/>
    <rect x="540" y="130" width="80" height="24" rx="4" opacity="0.9"/>
    <text x="580" y="30" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">8 workers</text>
    <line x1="490" y1="100" x2="540" y2="64" opacity="0.6"/>
    <line x1="490" y1="105" x2="540" y2="112" opacity="0.6"/>
    <text x="320" y="235" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">independent leaves, no edges between them - the whole reason this parallelizes</text>
  </svg>
  <figcaption>The same loop, drawn as a graph. Each file becomes one delayed features_for_file task, and because the stars are independent the tasks have no edges between them. The scheduler is then free to hand any task to any idle worker, in any order. Nothing here forces a sequence.</figcaption>
</figure>

You can see this for real on a handful of files. `dask.visualize` renders the graph Dask built:

```python
dask.visualize(*tasks[:6])
```

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/task_graph.png" alt="Dask task graph for six files: six identical, disconnected branches, each a circle feeding a rectangle, with no links between the branches.">
  <figcaption>The actual graph Dask drew for six files. Six identical, disconnected branches. The circles are the input file paths, the rectangles are the features_for_file calls, and nothing crosses between branches. A graph with no edges is a graph the scheduler can run in any order it likes, which is exactly what makes it fan out.</figcaption>
</figure>

## A cluster on one machine and a dashboard to monitor it

A graph is just a plan. To run it you hand it to a scheduler. The one I want is the distributed scheduler, because it comes with two things I care about: real worker processes, and a live dashboard.

```python
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=8, threads_per_worker=1)
client = Client(cluster)
```

`LocalCluster` starts eight worker processes on this one machine. `Client` connects to them. As a side effect, it prints a dashboard URL. Open it and you get the single most useful thing about Dask: a real-time picture of what the workers are doing.

```python
results = dask.compute(*tasks)
```

`compute` is where those promises actually run. It ships the graph to the scheduler, which starts handing tasks to whichever workers are free, and the dashboard fills in.

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/dashboard.png" alt="The Dask dashboard scheduler page: a table of eight workers, each processing about a hundred tasks at once.">
  <figcaption>The Dask dashboard's scheduler view during the run. Eight workers, each on its own core, and the two columns that matter are Processing and Occupancy: every worker holds about a hundred tasks (Processing), and Occupancy, Dask's estimate of how long that queue will take, sits near eleven seconds for all eight. Eleven runs above a naive hundred-times-56-ms because each task costs roughly double its serial time on the cluster once the file read, result serialization, and scheduler handling are counted, the same overhead that later holds the run to 3.6x rather than 8x. That is what a healthy parallel run looks like, all eight busy at once and none sitting idle waiting on the scheduler. The live task-stream plot lives behind the Bokeh button; this table is the at-a-glance version.</figcaption>
</figure>

The first time I opened the dashboard mid-run, Dask stopped being something I had to take on faith, because eight Processing counters sitting near a hundred is the whole story: the workers are busy, in parallel, and I can watch it happen instead of inferring it from a stopwatch at the end. When a run goes wrong the same view shows it in a couple of seconds, one worker buried and the other seven idle.

## Eight workers, 3.6x faster

Here is the full run, the same 5,204 stars, three ways:

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/speedup.png" alt="Wall time over 5,204 light curves: serial 294 seconds, threaded 82, distributed 81, the two parallel bars nearly equal.">
  <figcaption>Wall time to extract features from all 5,204 light curves. Serial on one core is the baseline at 294 seconds. Both eight-worker schedulers cut it to a bit over a quarter of that, and they land on top of each other: threads at 82 seconds, processes at 81. Same graph in all three cases; only the scheduler changed.</figcaption>
</figure>

The serial baseline was 294 seconds. The distributed scheduler on eight worker processes did it in 81, which is 3.6x. The first thing to say about that number is that 3.6 is not 8. Eight workers did not come close to making it eight times faster. Where the rest went is the more interesting part of this section.

But the number I did not expect was the middle one. Dask has a threaded scheduler too, and switching to it is a one-line change:

```python
dask.compute(*tasks, scheduler="threads", num_workers=8)
```

Eight threads instead of eight processes. I went in assuming Python's global interpreter lock would wreck it. The GIL lets only one thread execute Python at a time, and "use threads for CPU work in Python" is usually a trap. It came in at 82 seconds, 3.6x, a single second behind the worker processes. One second out of 82 is well inside the run-to-run noise, so I read threads and processes as a flat tie, on a workload I had written off as hopeless for threads.

The reason is that almost none of the per-star cost is Python the GIL can block: the Lomb-Scargle periodogram, and the astropy and NumPy underneath it, are compiled code that releases the GIL while it runs, so eight threads genuinely compute eight periodograms at the same time. The only thing left holding the GIL is the `pandas` read for each file and the small dict I build at the end, and on these light curves, where each periodogram chews through a few hundred points, that residue is too small to measure. So worker processes, which do not share a GIL at all, have essentially nothing to win back. Threads and processes tie.

<figure>
  <svg viewBox="0 0 640 250" role="img" aria-labelledby="gil-t gil-d" style="width:100%;max-width:640px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="1.5">
    <title id="gil-t">On threads, the GIL-releasing periodogram runs in parallel while only the Python glue serializes</title>
    <desc id="gil-d">Two stacked groups for four worker threads. In the top group the long periodogram bars all overlap in time because that compiled code releases the GIL. In the bottom group the short pandas-read bars are staggered so none overlap, because they hold the GIL and must take turns.</desc>
    <line x1="150" y1="32" x2="150" y2="226" opacity="0.25" stroke-dasharray="3 4"/>
    <text x="150" y="20" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.6">time -&gt;</text>
    <text x="12" y="58" font-size="11" stroke="none" fill="currentColor" font-weight="bold">periodogram</text>
    <text x="12" y="72" font-size="9.5" stroke="none" fill="currentColor" opacity="0.65">GIL released</text>
    <rect x="150" y="44" width="430" height="16" rx="3" fill="currentColor" fill-opacity="0.18"/>
    <rect x="150" y="64" width="430" height="16" rx="3" fill="currentColor" fill-opacity="0.18"/>
    <rect x="150" y="84" width="430" height="16" rx="3" fill="currentColor" fill-opacity="0.18"/>
    <rect x="150" y="104" width="430" height="16" rx="3" fill="currentColor" fill-opacity="0.18"/>
    <text x="365" y="138" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">overlap = parallel</text>
    <text x="12" y="168" font-size="11" stroke="none" fill="currentColor" font-weight="bold">pandas read</text>
    <text x="12" y="182" font-size="9.5" stroke="none" fill="currentColor" opacity="0.65">GIL held</text>
    <rect x="150" y="150" width="48" height="16" rx="3" fill="currentColor" fill-opacity="0.85" stroke="none"/>
    <rect x="198" y="170" width="48" height="16" rx="3" fill="currentColor" fill-opacity="0.85" stroke="none"/>
    <rect x="246" y="190" width="48" height="16" rx="3" fill="currentColor" fill-opacity="0.85" stroke="none"/>
    <rect x="294" y="210" width="48" height="16" rx="3" fill="currentColor" fill-opacity="0.85" stroke="none"/>
    <text x="430" y="190" font-size="10" stroke="none" fill="currentColor" opacity="0.7">staggered = taking turns</text>
  </svg>
  <figcaption>Why threads tied processes. Each star's work splits into a long periodogram (compiled code that releases the GIL, so the threads run these at the same time) and a short pandas read (real Python that holds the GIL, so the threads must take turns). The long part dominates and runs in parallel, so threads reach the full 3.6x. The short serial part is all that worker processes could have recovered, and here it is too small to see: the two schedulers finish a second apart.</figcaption>
</figure>

So neither half of my mental model survived the stopwatch: for this workload the two schedulers tie, and the only way I learned that was the dashboard and a stopwatch.

That still leaves the bigger gap. Eight workers, 3.6x. I did not profile where the missing speedup went, but the shape is familiar: 5,204 files means 5,204 tiny reads, which is I/O rather than CPU and does not split cleanly across cores, and each task runs for only about 56 ms, short enough that the scheduler's own bookkeeping starts to show against it. This is not the long, pure, CPU-bound work that scales to a clean 8x.

That last part has a knob I left alone here but would reach for next. When tasks are too small, you batch them. Hand each task a list of files instead of one, and let it loop internally, so the per-task overhead is amortised over fifty stars rather than paid for each.

```python
def features_for_batch(paths):
    return [features_for_file(p) for p in paths]

chunks = [paths[i:i + 50] for i in range(0, len(paths), 50)]
tasks = [dask.delayed(features_for_batch)(c) for c in chunks]
```

The arithmetic does not change and neither does the worker count, but now the scheduler tracks a hundred fat tasks instead of five thousand thin ones. On work this short that is the difference between fighting the overhead and ignoring it.

## What this bought, and what it did not

The accounting: I turned a five-minute serial loop into an 81-second one by changing two things. I wrapped one function in `dask.delayed`, and I picked the distributed scheduler. The science code did not change at all.

What Dask did *not* do is make anything faster on a single core, or rescue an algorithm that was badly written. The per-star work is identical. There is just more of it happening at once. If `features_for_file` had been slow because of a bad inner loop, Dask would have faithfully run that bad loop on eight workers. I would have learned nothing. Parallelism is a multiplier on the work you have, not a fix for the work being wrong.

And it is not free. The distributed scheduler spends real time spinning up workers, pickling your functions, and shipping results back. On 5,204 stars that overhead disappears into the win. On a handful it would dominate, and the plain `for` loop would beat all of this. The break-even is real: reach for a cluster when the pile of independent work is genuinely large.

That is the embarrassingly-parallel case in one move: the unit of work does not depend on its neighbors, so you wrap it and hand the scheduler the pile. The code that does this lives in the companion repo, the same one these light curves come from.

The next thing I wanted was to actually *use* these features: feed them to a classifier and see whether a machine can tell an RR Lyrae from an eclipsing binary the way the periodogram quietly already did. That turned into [a second post]({{ site.baseurl }}/posts/dask-hyperband-variable-stars/), where the data fits in memory easily and the model trains in seconds, so the thing Dask speeds up is the hyperparameter search around it, most of which turns out to be wasted on configurations that lost in the first few epochs.

[dask-experiments on GitHub](https://github.com/nilesh-patil/dask-experiments){: .btn-soft .btn-soft--primary} [Part two: tuning the classifier]({{ site.baseurl }}/posts/dask-hyperband-variable-stars/){: .btn-soft} [Dask project site](https://www.dask.org){: .btn-soft}
