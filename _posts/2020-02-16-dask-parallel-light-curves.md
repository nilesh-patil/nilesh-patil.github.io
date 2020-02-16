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

It comes in a few pieces. `dask.array` and `dask.dataframe` stand in for NumPy arrays and pandas frames when the data is too big to hold in memory. My data fit fine on a single instance. What I had was a slow loop, and the piece for that is `dask.delayed`: it takes the function calls you would have written in the loop and records them as a graph of tasks a scheduler can run in parallel.

To check it out properly, I took a real working problem: about 5,200 light curves from variable stars, each turned into a row of features. Star number 5,000 needs nothing from star number 4,999.

## What the loop was doing

The light curves are real, from the LINEAR survey, packaged by [astroML](https://www.astroml.org/) with hand-checked class labels. Five kinds of star are in here: RR Lyrae of both the ab and c types, the two flavours of eclipsing binary (close contact pairs and the deeper Algol type), and a thin scattering of Delta Scuti. Each star is a string of brightness measurements on an irregular cadence, a hundred to a few hundred points spread across years, with the gaps you get from weather and daylight.

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/lightcurve_example.png" alt="A raw RR Lyrae light curve on the left, the same data folded at its recovered period on the right.">
  <figcaption>One RR Lyrae star from the sample, brighter is up.</figcaption>
</figure>

Plotted against time, that star looks close to noise. Folded at the right period, the same points line up into a fast rise and a slow fade, which is why the period is the feature everything else hangs off. The per-star job is to read the file and boil the light curve down to a fixed vector of features: a period, an amplitude, a few shape and scatter statistics. To find the period you run a Lomb-Scargle periodogram, which scans close to a hundred thousand trial frequencies across the multi-year baseline and asks how well each one explains the data, so that scan is most of the per-star cost, and it is pure CPU.

The LINEAR catalogue ships its own validated periods, so I can hold mine up against them:

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/period_recovery.png" alt="Recovered period versus catalogue period, log-log: most points on the 1:1 line, a band of binaries on the half-period line.">
  <figcaption>One point per star, log-log axes.</figcaption>
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

Run this way, all 5,204 stars take 294 seconds, about 56 ms per star, all of it on one core.

## dask.delayed: stop calling the function

`dask.delayed` is a decorator that turns a normal function call into a *promise* to call it. Wrap `features_for_file` in `delayed` and call it, and nothing runs. Dask records a small node holding the function, its argument and a place for the result.

```python
import dask

tasks = [dask.delayed(features_for_file)(p) for p in paths]
```

`tasks` is now 5,204 unstarted nodes. The function inside them is the same `features_for_file` from above, and all I edited was the line that calls it.

The graph is a row of independent leaves with nothing connecting them, because no star needs any other star's result. The scheduler can hand any task to any idle worker, in any order.

`dask.visualize` renders the graph Dask built. For the first six files:

```python
dask.visualize(*tasks[:6])
```

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/task_graph.png" alt="Dask task graph for six files: six identical, disconnected branches, each a circle feeding a rectangle, with no links between the branches.">
  <figcaption>Circles are the features_for_file calls, rectangles are their results.</figcaption>
</figure>

## A cluster on one machine and a dashboard to monitor it

To run the graph I hand it to a scheduler. I picked the distributed scheduler because it brings separate worker processes and a live dashboard.

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
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/dashboard.png" alt="The Dask dashboard scheduler page: a table of eight workers, each with one core, 89 to 105 tasks under Processing and an Occupancy between 9.9 and 11.7 seconds.">
  <figcaption>The dashboard's scheduler page during the run, one row per worker. Processing is the number of tasks a worker currently holds, and Occupancy is Dask's estimate of how long that queue will take.</figcaption>
</figure>

Every worker holds about a hundred tasks, and Occupancy sits near eleven seconds for all eight. A hundred tasks at the serial 56 ms would be under six seconds, so on the cluster each task is costing roughly double its serial time. The live task-stream plot lives behind the Bokeh button; this table is the at-a-glance version.

Eight Processing counters sitting near a hundred means the workers are busy at the same time, and I can watch that happen instead of inferring it from a stopwatch at the end.

## Eight workers, 3.6x faster

The same 5,204 stars, run three ways:

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-parallel-light-curves/speedup.png" alt="Wall time over 5,204 light curves: serial 294 seconds, threaded 82, distributed 81, the two parallel bars nearly equal.">
  <figcaption>Wall time to extract features.</figcaption>
</figure>

The serial loop took 294 seconds and the distributed scheduler on eight worker processes took 81. That is 3.6x from eight workers, well short of 8x.

Dask has a threaded scheduler too, and switching to it is a one-line change:

```python
dask.compute(*tasks, scheduler="threads", num_workers=8)
```

Eight threads instead of eight processes. I went in assuming Python's global interpreter lock would wreck it. The GIL lets only one thread execute Python at a time, and "use threads for CPU work in Python" is usually a trap. It came in at 82 seconds, 3.6x, a single second behind the worker processes. I would not call a one-second gap on an 82-second run a difference, so I read threads and processes as a tie.

Almost none of the per-star cost is Python the GIL can block. The Lomb-Scargle periodogram is astropy's `LombScargle`, which runs a scan this size as FFTs inside NumPy, compiled code that releases the GIL while it runs, so eight threads compute eight periodograms at the same time. What still holds the GIL is the Python around those calls and the small dict I build at the end, and on these light curves, where each periodogram chews through a few hundred points, that remainder did not show up in the wall time.

That still leaves the bigger gap, eight workers and only 3.6x. It matches what the dashboard showed, each task costing roughly double its serial time. I did not profile where the missing speedup went.

The next thing I wanted was to use these features: feed them to a classifier and see whether it can tell an RR Lyrae from an eclipsing binary.

*Update, June 2020: the classifier became [a second post]({{ site.baseurl }}/posts/dask-hyperband-variable-stars/).*

[dask-experiments on GitHub](https://github.com/nilesh-patil/dask-experiments){: .btn-soft .btn-soft--primary} [Part two: tuning the classifier]({{ site.baseurl }}/posts/dask-hyperband-variable-stars/){: .btn-soft} [Dask project site](https://www.dask.org){: .btn-soft}
