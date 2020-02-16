---
layout: single
title: "Introduction to dask - parallel feature extraction in variable-star data"
date: 2020-02-16T11:30:00+05:30
last_modified_at: 2020-06-08T10:00:00+05:30
categories: [blog]
tags: [python, dask, parallel-computing, astronomy, machine-learning]
excerpt: "I had 5,204 light curves and one slow for-loop. Wrapping the per-star function in dask.delayed and running it on eight workers took the feature extraction from 294 seconds to 81, and eight threads did just as well as eight processes."
measurement: "294 seconds to 81"
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

Dask is a library for running generic Python & its data ecosystem libraries in parallel. You keep NumPy, pandas and the plain functions you already have, and Dask works out how to spread the work across your cores, or across a cluster, and runs it for you.

It comes in a few pieces. `dask.array` and `dask.dataframe` stand in for NumPy arrays and pandas frames when the data is too big to hold in memory. My data fit fine on a single instance. What I had was a slow loop over things that fit, and the piece for that is `dask.delayed`: it takes the function calls you would have written in the loop and records them as a graph of tasks a scheduler can run in parallel.

To check it out properly instead of just skimming the docs and forgetting, I took a real working problem: about 5,200 light curves from variable stars, each turned into a row of features. Star number 5,000 needs nothing from star number 4,999. There is no ordering and no shared state, and they were being processed one at a time only because a `for` loop is the first thing most folks would reach for.

## What the loop was doing

The light curves are real, from the LINEAR survey, packaged by [astroML](https://www.astroml.org/) with hand-checked class labels. Five kinds of star are in here: RR Lyrae of both the ab and c types, the two flavours of eclipsing binary (close contact pairs and the deeper Algol type), and a thin scattering of Delta Scuti. Each star is a string of brightness measurements on an irregular cadence, a hundred to a few hundred points spread across years, with the gaps you get from weather and daylight.

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/lightcurve_example.png" alt="A raw RR Lyrae light curve on the left, the same data folded at its recovered period on the right.">
  <figcaption>One RR Lyrae star from the sample. Left: magnitude against time as observed, brighter is up. Right: the same points folded at the period the periodogram recovers.</figcaption>
</figure>

Plotted against time, that star looks close to noise. Folded at the right period, the same points line up into a fast rise and a slow fade, which is why the period is the feature everything else hangs off. The per-star job is to read the file and boil the light curve down to a fixed vector of features: a period, an amplitude, a few shape and scatter statistics. The slow part is the period. To find it you run a Lomb-Scargle periodogram, which scans close to a hundred thousand trial frequencies across the multi-year baseline and asks how well each one explains the data, so that scan is most of the per-star cost, and it is pure CPU.

The LINEAR catalogue ships its own validated periods, so I can hold mine up against them:

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/period_recovery.png" alt="Recovered period versus catalogue period, log-log: most points on the 1:1 line, a band of binaries on the half-period line.">
  <figcaption>My Lomb-Scargle period against the catalogue period, one point per star, log-log axes. The diagonal is agreement.</figcaption>
</figure>

About 53% of the 5,204 land within 2% of the catalogue value. The dense band below the diagonal is the eclipsing binaries: an eclipsing binary dips twice per orbit, so the periodogram locks onto the doubled frequency and reports half the period. The offset is the same for the whole band, so a half period still works as a feature.

The unit of work is one function:

```python
def features_for_file(path):
    """Read one light-curve file and return its features as a dict."""
    t, m, e = read_light_curve(path)
    row = {"star_id": star_id_from_path(path)}
    row.update(extract_features(t, m, e))
    return row
```

The loop I started with calls it once per path:

```python
rows = [features_for_file(p) for p in paths]
```

Run this way, all 5,204 stars take 294 seconds, just under five minutes, which works out to about 56 ms per star, all of it on one core.

## dask.delayed: stop calling the function

`dask.delayed` is a decorator that turns a normal function call into a *promise* to call it. Wrap `features_for_file` in `delayed` and call it, and nothing runs. Dask records a small node holding the function, its argument and a place for the result. After doing that for every file I have a graph and no features yet.

```python
import dask

tasks = [dask.delayed(features_for_file)(p) for p in paths]
```

`tasks` is now 5,204 unstarted nodes. The function inside them is byte-for-byte the same `features_for_file` from above, and all I edited was the line that calls it.

The graph is a row of independent leaves with nothing connecting them, because no star needs any other star's result. The scheduler can hand any task to any idle worker, in any order.

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
  <figcaption>The loop drawn as a graph: one delayed features_for_file task per file, no edges between tasks, eight workers fed by the scheduler.</figcaption>
</figure>

`dask.visualize` renders the graph Dask built. For the first six files:

```python
dask.visualize(*tasks[:6])
```

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/task_graph.png" alt="Dask task graph for six files: six identical, disconnected branches, each a circle feeding a rectangle, with no links between the branches.">
  <figcaption>The graph Dask drew for six files. Circles are the input file paths, rectangles are the features_for_file calls, and no line crosses between the six branches.</figcaption>
</figure>

## A cluster on one machine and a dashboard to monitor it

To run the graph I hand it to a scheduler. I picked the distributed scheduler because it brings two things I care about: separate worker processes, and a live dashboard.

```python
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=8, threads_per_worker=1)
client = Client(cluster)
```

`LocalCluster` starts eight worker processes on this one machine and `Client` connects to them. The client also prints a dashboard URL, and that page shows what each worker is doing while the run is in progress.

```python
results = dask.compute(*tasks)
```

`compute` is where those promises run. It ships the graph to the scheduler, which starts handing tasks to whichever workers are free, and the dashboard fills in.

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/dashboard.png" alt="The Dask dashboard scheduler page: a table of eight workers, each processing about a hundred tasks at once.">
  <figcaption>The dashboard's scheduler page during the run, one row per worker. Processing is the number of tasks a worker currently holds, and Occupancy is Dask's estimate of how long that queue will take.</figcaption>
</figure>

Every worker holds about a hundred tasks, and Occupancy sits near eleven seconds for all eight. A hundred tasks at the serial 56 ms would be under six seconds, so on the cluster each task is costing roughly double its serial time once the file read, result serialization and scheduler handling are counted. The live task-stream plot lives behind the Bokeh button; this table is the at-a-glance version.

The first time I opened the dashboard mid-run, Dask stopped being something I had to take on faith. Eight Processing counters sitting near a hundred means the workers are busy at the same time, and I can watch that happen instead of inferring it from a stopwatch at the end. When a run goes wrong the same view shows it in a couple of seconds, one worker buried and the other seven idle.

## Eight workers, 3.6x faster

The same 5,204 stars, run three ways:

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/speedup.png" alt="Wall time over 5,204 light curves: serial 294 seconds, threaded 82, distributed 81, the two parallel bars nearly equal.">
  <figcaption>Wall time to extract features from all 5,204 light curves: serial 294 s, eight threads 82 s, eight worker processes 81 s. The task graph is the same in all three.</figcaption>
</figure>

These are the wall times recorded in `results/timings.json` in the companion repo (293.79, 82.18 and 81.41 seconds), rounded. The serial loop took 294 seconds and the distributed scheduler on eight worker processes took 81. That is 3.6x from eight workers, well short of 8x.

The middle bar is the one I did not expect. Dask has a threaded scheduler too, and switching to it is a one-line change:

```python
dask.compute(*tasks, scheduler="threads", num_workers=8)
```

Eight threads instead of eight processes. I went in assuming Python's global interpreter lock would wreck it. The GIL lets only one thread execute Python at a time, and "use threads for CPU work in Python" is usually a trap. It came in at 82 seconds, 3.6x, a single second behind the worker processes. I would not call a one-second gap on an 82-second run a difference, so I read threads and processes as a tie, on a workload I had written off as hopeless for threads.

Almost none of the per-star cost is Python the GIL can block. The Lomb-Scargle periodogram, and the astropy and NumPy underneath it, are compiled code that releases the GIL while it runs, so eight threads compute eight periodograms at the same time. What still holds the GIL is the `pandas` read for each file and the small dict I build at the end, and on these light curves, where each periodogram chews through a few hundred points, that remainder did not show up in the wall time.

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
  <figcaption>Four threads, one star each. The long periodogram bars overlap in time because that compiled code releases the GIL. The short pandas-read bars are staggered because they hold the GIL and have to take turns.</figcaption>
</figure>

That still leaves the bigger gap, eight workers and only 3.6x. I did not profile where the missing speedup went, but I have two suspects. 5,204 files means 5,204 tiny reads, which is I/O and does not split across cores the way the periodogram does. And each task runs for only about 56 ms, short enough that the scheduler's own bookkeeping shows against it, which is what the doubled per-task cost on the dashboard was already saying.

The second suspect has a knob I left alone here but would reach for next. When tasks are too small, you batch them. Hand each task a list of files and let it loop internally, so the per-task overhead is paid once per fifty stars.

```python
def features_for_batch(paths):
    return [features_for_file(p) for p in paths]

chunks = [paths[i:i + 50] for i in range(0, len(paths), 50)]
tasks = [dask.delayed(features_for_batch)(c) for c in chunks]
```

The arithmetic and the worker count stay the same, but the scheduler now tracks about a hundred fat tasks instead of five thousand thin ones. I have not timed this version.

## What changed in the code

A five-minute serial loop became an 81-second run with two changes: I wrapped one function in `dask.delayed`, and I picked the distributed scheduler.

Nothing got faster on a single core. The per-star work is identical, there is just more of it happening at once, and if `features_for_file` had been slow because of a bad inner loop, Dask would have faithfully run that bad loop on eight workers.

The distributed scheduler also spends time spinning up workers, pickling functions and shipping results back. On 5,204 stars that cost is absorbed; on a handful I would expect the plain `for` loop to win.

The next thing I wanted was to use these features: feed them to a classifier and see whether it can tell an RR Lyrae from an eclipsing binary, which the half-period band in the periodogram plot already hints at.

*Update, June 2020: the classifier became [a second post]({{ site.baseurl }}/posts/dask-hyperband-variable-stars/). There the data fits in memory easily and the model trains in seconds, so the thing Dask speeds up is the hyperparameter search around it, most of which is wasted on configurations that lost in the first few epochs.*

[dask-experiments on GitHub](https://github.com/nilesh-patil/dask-experiments){: .btn-soft .btn-soft--primary} [Part two: tuning the classifier]({{ site.baseurl }}/posts/dask-hyperband-variable-stars/){: .btn-soft} [Dask project site](https://www.dask.org){: .btn-soft}
