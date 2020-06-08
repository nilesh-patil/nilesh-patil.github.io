---
layout: single
title: "Hyperband on Dask - hyperparameter tuning for a variable-star classifier"
date: 2020-06-08T10:00:00+05:30
last_modified_at: 2020-06-08T10:00:00+05:30
categories: [blog]
tags: [python, dask, machine-learning, hyperparameter-tuning, astronomy]
excerpt: "A random forest sorts these variable stars at 97% with no tuning at all. So I tuned a model that does need it, an SGD classifier, in two ways: 143 configurations trained to the end, and 143 under Hyperband on a Dask cluster, which spent 7.3x fewer training passes."
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

In the [first post]({{ site.baseurl }}/posts/dask-parallel-light-curves/) I turned about 5,200 LINEAR light curves into a feature table, one row per star, twelve numbers each. The astroML catalogue also ships four colours per star (ug, gi, iK, JK), so I bolted those on, for sixteen features in total. Hand that table to a random forest and it sorts the five star types at 96.8% on a held-out quarter with no tuning whatsoever. The colours earn their place: drop them and accuracy falls by about a point, to 95.7%.

That leaves nothing to search, so for this post I swapped in a model whose settings do matter, a linear classifier trained by SGD, and tuned it twice on an eight-worker Dask cluster. Random search trained 143 configurations to the end and spent 11,583 training passes doing it, while Hyperband, the early-stopping search in dask-ml, went through 143 configurations in 1,581 passes and picked a model that was within a point of the other one on the test set.

## A model that needs tuning

The forest's defaults are already close to its best, and it trains in one shot, so there is no partial result to look at. A linear classifier trained by stochastic gradient descent has neither problem.

```python
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="log", penalty="elasticnet", tol=1e-3)
# learns one pass at a time:
for _ in range(n_epochs):
 clf.partial_fit(X_train, y_train, classes=classes)
```

This is logistic regression with an elastic-net penalty, one-vs-rest over the five classes. Tuned, it reaches 91.6% to 92.5% on the test set across the two searches below, a few points under the forest, but its two regularisation settings move it a lot. Among the ten configurations that Hyperband later trained for the full 81 epochs, validation accuracy runs from 88.6% to 94.1%, and `alpha` and `l1_ratio` are the only things that differ between them (`results/hyperband_history.json` in the companion repo). It also trains through `partial_fit`, one pass over the data per call, so I can stop after any pass, score it, and decide whether it gets more.

## The default: train everyone to the end

The usual way to set `alpha` and `l1_ratio` is random search: sample a few hundred pairs, train each one to convergence, keep whichever scored best. Every fit is independent of the others, the same embarrassingly parallel shape as the feature loop in the first post, so I ran it the same way. 143 configurations went out as `dask.delayed` tasks across eight workers, 81 epochs each.

That is 143 x 81 = 11,583 training passes, finished in 8.3 seconds on the cluster, with the best configuration scoring 92.5% on the test set.

A configuration with far too much regularisation underfits, and it shows early: in the Hyperband run below, validation scores after a single epoch already spread from 75% to 92%. Random search trains the 75% one for all 81 epochs anyway, then throws it away.

## Give everyone a little, then cut

Successive halving gives every configuration a small budget, say one epoch, scores them all, keeps the top third and drops the rest. The survivors get three times the epochs, then you score and cut again, and so on until the budget reaches the maximum. A configuration costs only the epochs it was given before it was dropped.

Start with 81 configurations and that goes: 81 at 1 epoch, 27 at 3 epochs, 9 at 9, 3 at 27, and 1 trained to the full 81.

<figure>
 <svg viewBox="0 0 640 300" role="img" aria-labelledby="sh-t sh-d" style="width:100%;max-width:640px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="1.4">
 <title id="sh-t">Successive halving funnels many cheap configurations down to a few well-trained ones</title>
 <desc id="sh-d">Four rounds left to right. The first round has many short bars, each a configuration trained for one epoch. At each round most bars are dropped and the survivors get taller, meaning more epochs, until the last round has a few full-height bars.</desc>
 <text x="70" y="20" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" font-weight="bold">many configs</text>
 <text x="70" y="288" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">1 epoch each</text>
 <text x="560" y="20" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" font-weight="bold">a few configs</text>
 <text x="560" y="288" font-size="10" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.7">full budget</text>
 <g opacity="0.85">
 <rect x="24" y="232" width="9" height="34" rx="2"/><rect x="37" y="232" width="9" height="34" rx="2"/><rect x="50" y="232" width="9" height="34" rx="2"/><rect x="63" y="232" width="9" height="34" rx="2"/><rect x="76" y="232" width="9" height="34" rx="2"/><rect x="89" y="232" width="9" height="34" rx="2"/><rect x="102" y="232" width="9" height="34" rx="2"/>
 </g>
 <g opacity="0.85">
 <rect x="190" y="196" width="13" height="70" rx="2"/><rect x="210" y="196" width="13" height="70" rx="2"/><rect x="230" y="196" width="13" height="70" rx="2"/>
 </g>
 <g opacity="0.85">
 <rect x="350" y="140" width="18" height="126" rx="2"/><rect x="376" y="140" width="18" height="126" rx="2"/>
 </g>
 <g opacity="0.9">
 <rect x="520" y="70" width="26" height="196" rx="3"/>
 </g>
 <g opacity="0.28" stroke-dasharray="3 3">
 <rect x="24" y="244" width="9" height="22" rx="2"/><rect x="50" y="244" width="9" height="22" rx="2"/><rect x="76" y="244" width="9" height="22" rx="2"/><rect x="102" y="244" width="9" height="22" rx="2"/>
 <rect x="210" y="216" width="13" height="50" rx="2"/>
 <rect x="376" y="170" width="18" height="96" rx="2"/>
 </g>
 <path d="M120 250 L185 230" opacity="0.5"/>
 <path d="M250 215 L345 200" opacity="0.5"/>
 <path d="M398 175 L515 150" opacity="0.5"/>
 <text x="320" y="290" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">keep the top third each round, give survivors 3x the epochs</text>
 </svg>
 <figcaption>Successive halving. Each bar is one configuration and its height is the epochs it has trained. Faded bars are dropped at that round; the others move right with three times the budget.</figcaption>
</figure>

The scheme has a setting of its own, the number of configurations to start with. Start with many on one epoch each, and a slow starter that would have come good is cut before it has the epochs to prove it. Start with a few on long budgets and you are back to random search. Hyperband runs several successive-halving brackets side by side, from many configurations on short budgets to a few on the full budget, and takes the best model out of all of them.

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

`max_iter` is the most epochs any single model can earn, and `aggressiveness` is the cut factor. The search works out the bracket schedule from those two numbers. For 81 and 3 it made five brackets, starting with 81, 34, 15, 8 and 5 configurations, which add up to the 143. The bracket of 81 is the schedule from the previous section, while the bracket of 5 does no cutting at all and trains all five to 81 epochs. On the dashboard while it runs, models appear, most of them train for a few epochs and vanish, and a thinning set keeps going.

<figure>
 <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/hyperband_tournament.png" alt="Each line is one configuration, validation score against epochs trained: most stop short at low scores, a few run to the full budget.">
 <figcaption>One line per configuration: validation accuracy against epochs trained. Grey lines were dropped, 95 of the 133 after one to three epochs. Blue lines are the ten that reached 81.</figcaption>
</figure>

## What the early stopping bought

<figure>
 <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/search_compute.png" alt="Two bars of total training passes: random search uses far more than Hyperband, with both reaching about the same test accuracy noted above them.">
 <figcaption>Total training passes for each search, with the test accuracy of the configuration it picked printed on the bar.</figcaption>
</figure>

For the same 143 configurations, random search used 11,583 passes and Hyperband 1,581, which is 7.3x less. On the test set Hyperband's pick scored 91.6% against random search's 92.5%. That gap is about a dozen of the 1,301 test stars, and it comes from one run with every `random_state` set to 42, so I read it as a tie rather than a loss. On validation, the score the search optimises, Hyperband's pick was the higher of the two, 94.1% against 92.8%, and its wider drop from there to 91.6% on test is the cost of picking a winner on a small validation split. The two searches held out different slices of the training set for validation, though, so that pair of numbers is not a like-for-like comparison.

<figure>
 <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/accuracy_vs_budget.png" alt="Accuracy against training passes for both searches: Hyperband sits left at far less compute, tied with random search on test and higher on validation.">
 <figcaption>Both searches, on the test set and on validation, placed by the training passes they spent. The shaded band is the 0.9-point gap on test.</figcaption>
</figure>

The saving in passes shows up in wall time too. Random search took 8.3 seconds on eight workers. I timed the Hyperband search in a separate sweep over cluster sizes (`results/search_metrics.json`): 7.7 seconds on one worker, 4.2 on two, 3.1 on four and about 2.7 on eight. That last doubling buys 0.4 seconds, because each fit runs on a single worker, so extra workers only help while there are more models alive than workers to run them, and the later rounds have few models left.

<figure>
 <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/search_scaling.png" alt="Hyperband wall time against worker count: it falls steeply to four workers, then flattens.">
 <figcaption>The same Hyperband search timed on one, two, four and eight workers.</figcaption>
</figure>

## Where this stops being the answer

Hyperband needs a model it can pause and score partway through training. The random forest has no `partial_fit` and no half-trained state to judge after one epoch, so if you want to tune one you are back to plain random search. The forest is also still the better classifier on this table, 96.8% against the tuned linear model's 91.6%, and the only reason there was a search to write about is that I picked the weaker model.

The other limit is scale. The saving here is a few seconds, because an SGD epoch on 3,903 rows is nearly free. I would expect it to matter when an epoch is expensive, a neural net for instance, where one bad configuration trained to the end costs minutes, but I have not run that case.

[dask-experiments on GitHub](https://github.com/nilesh-patil/dask-experiments){: .btn-soft .btn-soft--primary} [Part one: the feature loop]({{ site.baseurl }}/posts/dask-parallel-light-curves/){: .btn-soft} [When the model itself outgrows one machine]({{ site.baseurl }}/posts/distributed-kmeans-clustering/){: .btn-soft}
