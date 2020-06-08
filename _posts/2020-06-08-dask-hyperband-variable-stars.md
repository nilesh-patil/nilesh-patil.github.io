---
layout: single
title: "Hyperband on Dask - hyperparameter tuning for a variable-star classifier"
date: 2020-06-08T10:00:00+05:30
last_modified_at: 2020-06-08T10:00:00+05:30
categories: [blog]
tags: [python, dask, machine-learning, hyperparameter-tuning, astronomy]
excerpt: "A random forest sorts these variable stars at 96.8% with no tuning at all. So I tuned a model that does need it, an SGD classifier, in two ways: 143 configurations trained to the end, and 143 under Hyperband on a Dask cluster, which spent 7.3x fewer training passes."
redirect_from:
  - /posts/dask-random-forest-variable-stars/
header:
  overlay_image: /images/blog/headers/dask-hyperband-variable-stars.jpg
  overlay_filter: 0.5
  teaser: /images/blog/headers/dask-hyperband-variable-stars.jpg
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

In the [first post]({{ site.baseurl }}/posts/dask-parallel-light-curves/) I turned about 5,200 LINEAR light curves into a feature table, one row per star, twelve numbers each. The astroML catalogue also ships four colours per star (ug, gi, iK, JK), so I bolted those on, for sixteen features in total. Hand that table to a random forest and it sorts the five star types at 96.8% on a held-out quarter with no tuning whatsoever. Drop the colours and accuracy falls by about a point, to 95.7%.

That leaves nothing to search, so for this post I swapped in a model whose settings do matter, a linear classifier trained by SGD, and tuned it twice on an eight-worker Dask cluster. Random search trained 143 configurations to the end and spent 11,583 training passes doing it, while Hyperband, the early-stopping search in dask-ml, went through 143 configurations in 1,581 passes and picked a model that was within a point of the other one on the test set.

## A model that needs tuning

```python
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="log", penalty="elasticnet")
# learns one pass at a time:
for _ in range(n_epochs):
 clf.partial_fit(X_train, y_train, classes=classes)
```

This is logistic regression with an elastic-net penalty, one-vs-rest over the five classes. The features are standardised first, with a `StandardScaler` fit on the training rows. Tuned, it reaches 91.6% to 92.5% on the test set across the two searches below, a few points under the forest, but its two regularisation settings move it a lot. Among the ten configurations that Hyperband later trained for the full 81 epochs, validation accuracy runs from 88.6% to 94.1%, and `alpha` and `l1_ratio` are the only settings that differ between them ([`results/hyperband_history.json`](https://github.com/nilesh-patil/dask-experiments/blob/master/results/hyperband_history.json) in the companion repo). It also trains through `partial_fit`, one pass over the data per call, so I can stop after any pass and decide whether it gets more.

## The baseline

The usual way to set `alpha` and `l1_ratio` is random search. Every fit is independent of the others, the same embarrassingly parallel shape as the feature loop in the first post, so I ran it the same way. 143 configurations went out as `dask.delayed` tasks across eight workers, 81 epochs each.

That is 143 x 81 = 11,583 training passes.

A configuration with far too much regularisation underfits, and it shows early: in the Hyperband run below, validation scores after a single epoch already spread from 75% to 92%. Random search would train the 75% one for all 81 epochs anyway, then throw it away.

## Successive halving

Successive halving gives every configuration a small budget, say one epoch, scores them all, keeps the top third and drops the rest. The survivors get three times the epochs, then you score and cut again, and so on until the budget reaches the maximum.

Start with 81 configurations and that goes: 81 at 1 epoch, 27 at 3 epochs, 9 at 9, 3 at 27, and 1 trained to the full 81.

The scheme has a setting of its own, the number of configurations to start with. Start with many on one epoch each, and a slow starter that would have come good is cut before it has the epochs to prove it. Hyperband runs several successive-halving brackets side by side, from many configurations on short budgets to a few on the full budget, and takes the best model out of all of them.

## Hyperband on the cluster

dask-ml ships this as `HyperbandSearchCV`, with one constraint: the estimator has to implement `partial_fit`, which is why the search runs over the SGD classifier and the forest sits out. The brackets, and the models inside them, are independent of one another, so the scheduler spreads them over the workers.

```python
from dask_ml.model_selection import HyperbandSearchCV
from scipy.stats import loguniform, uniform

params = {"alpha": loguniform(1e-6, 1e-1), "l1_ratio": uniform(0, 1)}
search = HyperbandSearchCV(clf, params, max_iter=81, aggressiveness=3)

with Client(cluster):
 search.fit(X_train, y_train, classes=classes)
```

`max_iter` is the most epochs any single model can earn, and `aggressiveness` is the cut factor. The search works out the bracket schedule from those two numbers. For 81 and 3 it made five brackets, starting with 81, 34, 15, 8 and 5 configurations, which add up to the 143. The bracket of 81 is the schedule from the previous section, while the bracket of 5 does no cutting at all and trains all five to 81 epochs.

<figure>
 <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/hyperband_tournament.png" alt="Each line is one configuration, validation score against epochs trained.">
 <figcaption>Grey lines were dropped, 95 of the 133 after one to three epochs. Blue lines are the ten that reached 81.</figcaption>
</figure>

## Results

<figure>
 <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/search_compute.png" alt="Two bars of total training passes: random search uses far more than Hyperband.">
 <figcaption>Total training passes for each search, with the test accuracy of the configuration it picked printed on the bar. The title says "same 143 configs", but each search drew its own 143.</figcaption>
</figure>

For 143 configurations, random search used 11,583 passes and Hyperband 1,581, which is 7.3x less. The two searches drew different samples of 143, random search from seed 43 and Hyperband from seed 42. On the test set Hyperband's pick scored 91.6% against random search's 92.5%. That gap is about a dozen of the 1,301 test stars, and it comes from one run. Random search's pick was refit on all 3,903 training rows before the test, and Hyperband's was tested as the search left it, trained on 80% of them. Hyperband's pick had scored 94.1% on the other 20%, which the search held out for validation.

Random search took 8.3 seconds on eight workers. I timed the Hyperband search in a separate sweep over cluster sizes ([`results/search_metrics.json`](https://github.com/nilesh-patil/dask-experiments/blob/master/results/search_metrics.json)): 7.7 seconds on one worker, 4.2 on two, 3.1 on four and about 2.7 on eight. That last doubling buys 0.4 seconds, because each fit runs on a single worker, so extra workers only help while there are more models alive than workers to run them, and the later rounds have few models left.

<figure>
 <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/search_scaling.png" alt="Hyperband wall time falls steeply to four workers, then flattens.">
 <figcaption>The same Hyperband search timed on one, two, four and eight workers.</figcaption>
</figure>

The saving here is a few seconds, because an SGD epoch on 3,903 rows is nearly free. I would expect it to matter when an epoch is expensive, a neural net for instance, where one bad configuration trained to the end costs minutes.

[dask-experiments on GitHub](https://github.com/nilesh-patil/dask-experiments){: .btn-soft .btn-soft--primary} [Part one: the feature loop]({{ site.baseurl }}/posts/dask-parallel-light-curves/){: .btn-soft} [When the model itself outgrows one machine]({{ site.baseurl }}/posts/distributed-kmeans-clustering/){: .btn-soft}
