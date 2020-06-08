---
layout: single
title: "Hyperband on Dask - hyperparameter tuning for a variable-star classifier"
date: 2020-06-08T10:00:00+05:30
last_modified_at: 2020-06-08T10:00:00+05:30
categories: [blog]
tags: [python, dask, machine-learning, hyperparameter-tuning, astronomy]
excerpt: "A random forest sorts these variable stars at 97% with no tuning at all. So I went looking for a model that does need tuning, and watched the usual way to tune it throw most of its compute at configurations that were dead after five epochs. Hyperband, on a Dask cluster, spends that compute where it counts."
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

Training a model once is cheap. The expense is in tuning it: you do not fit one model, you fit a few hundred, one for every combination of settings, and keep whichever scored best. Those fits do not depend on each other. The search is embarrassingly parallel, the same shape as the feature loop in the [first post]({{ site.baseurl }}/posts/dask-parallel-light-curves/), and it runs on the same Dask cluster the same way.

The interesting part is that you do not have to run all of it. dask-ml, the machine-learning toolkit built on Dask, has a search for exactly this, called Hyperband: it spends most of its budget on the configurations that are doing well and pulls the plug early on the ones that are not. That is what this post is about, not making any single fit faster, but skipping the fits that were never going to win.

This is the second half of a small Dask project, and it opens with the model that nearly convinced me Dask had no place here.

In the first post I turned about 5,200 LINEAR light curves into a feature table, one row per star, twelve numbers each. The astroML catalogue also ships four colours per star (ug, gi, iK, JK), so I bolted those on too, for sixteen features total. Hand that table to a random forest and it sorts the five star types at 96.8% on a held-out quarter, with no tuning whatsoever. The colours earn their place: drop them and accuracy falls by about a point, from 96.8% to 95.7%.

So the classifier is done. A random forest with default settings is hard to beat on a table this size, and it has no knobs I need to touch. Which leaves the actual question this post is about, the one the forest is too easy to show: when a model *does* need its knobs set, how do you set them without wasting most of your compute? Because the standard answer wastes a lot.

This is not about making the forest faster (it fits in two seconds, leave it alone) or about data that does not fit in memory (it fits). It is about the search, run on a model that genuinely needs searching, with a Dask-native method that gives up on the bad configurations early.

## A model that actually needs tuning

The forest is the wrong thing to study tuning on, for two reasons: it barely needs it, and it trains in one shot, so there is nothing to watch. I want the opposite. A plain linear classifier trained by stochastic gradient descent gives me both.

```python
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="log", penalty="elasticnet", tol=1e-3)
# learns one pass at a time:
for _ in range(n_epochs):
    clf.partial_fit(X_train, y_train, classes=classes)
```

This is multinomial logistic regression, fit by SGD. On the scaled sixteen features it reaches about 92%, a few points under the forest, but it has the two properties the forest lacks. It is genuinely sensitive to its regularisation: the `alpha` and `l1_ratio` knobs move its accuracy around by several points, so the tuning is not cosmetic. And it learns *incrementally*, one `partial_fit` pass at a time, so I can train it a little, look at how it is doing, and decide whether to keep going. That second property is the whole game here.

## The default: train everyone to the end

The standard way to set `alpha` and `l1_ratio` is to sample a few hundred combinations, train each one to convergence, and keep whichever scored best. Random search. It is trivially parallel, one independent fit per configuration, exactly the shape from the first post, so I ran it the same way, 143 configurations as `dask.delayed` tasks across eight workers.

Here is what that costs. Each of the 143 configurations trains for 81 epochs. That is 143 x 81 = 11,583 training passes. On the cluster the whole search finished in 8.3 seconds, at a best test accuracy of 92.5%.

The number that bothers me is 11,583. A configuration with a hopeless learning rate, or so much regularisation that it underfits, is recognisable as a loser after five or ten epochs: its validation score is bad and not moving. Random search trains it for all 81 anyway, then throws the result away. Most of those 11,583 passes were spent finishing models that were already out of contention.

## Give everyone a little, then cut

The fix is an old idea with a good name: successive halving. Give every configuration a small budget, say one epoch. Look at the scores. Keep the top third, throw the rest away, and give the survivors three times the budget. Look again, cut again. After a few rounds, only a handful of configurations are still standing, and those are the ones getting trained to the full 81. The bad configurations died cheap.

<figure>
  <svg viewBox="0 0 640 300" role="img" aria-labelledby="sh-t sh-d" style="width:100%;max-width:640px;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="1.4">
    <title id="sh-t">Successive halving funnels many cheap configurations down to a few well-trained ones</title>
    <desc id="sh-d">Four rungs left to right. The first rung has many short bars, each a configuration trained for one epoch. At each rung most bars are dropped and the survivors get taller, meaning more epochs, until the last rung has a few full-height bars.</desc>
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
    <text x="320" y="290" font-size="11" stroke="none" fill="currentColor" text-anchor="middle" opacity="0.75">keep the top third each rung, give survivors 3x the epochs</text>
  </svg>
  <figcaption>Successive halving. Every configuration starts with one epoch (left). After each rung the worst two thirds are dropped (the faded bars) and the survivors are trained three times longer (the taller bars), until a few configurations reach the full budget on the right. The compute spent on any configuration is proportional to how long it stayed alive.</figcaption>
</figure>

Successive halving has one awkward knob of its own: how aggressively to cut, and how many configurations to start with. Cut too hard and a slow starter that would have come good never gets the epochs to prove it; cut too gently and you have just rebuilt random search with extra steps. Hyperband is the trick that removes that knob: it runs several successive-halving brackets side by side, each with a different starting count and cut rate, so it hedges across the aggressive and the cautious schedules instead of betting on one. You get the early-stopping savings without having to guess the schedule.

## Hyperband on the cluster

dask-ml ships Hyperband as a drop-in search, and on a cluster it has a nice property: the brackets, and the models within them, are independent, so the scheduler runs them across the workers the same way it ran the feature loop in the first post. The only constraint is the one from earlier, the estimator has to support `partial_fit`, which is why this is the SGD classifier and not the forest.

```python
from dask_ml.model_selection import HyperbandSearchCV
from scipy.stats import loguniform, uniform

params = {"alpha": loguniform(1e-6, 1e-1), "l1_ratio": uniform(0, 1)}
search = HyperbandSearchCV(clf, params, max_iter=81, aggressiveness=3)

with Client(cluster):
    search.fit(X_train, y_train, classes=classes)
```

`max_iter` is the most epochs any single model can earn, and `aggressiveness` sets the cut rate. The search figures out the bracket schedule from those two numbers. Watching the dashboard while it runs is the clearest picture of what it does: models appear, most of them run for a few epochs and vanish, and a thinning set keeps going.

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/hyperband_tournament.png" alt="Each line is one configuration, validation score against epochs trained: most stop short at low scores, a few run to the full budget.">
  <figcaption>Every configuration Hyperband tried, validation accuracy against how many epochs it was allowed. The grey lines are the models it cut: most ran one to three epochs, scored somewhere mediocre, and were dropped. The blue lines are the survivors, promoted rung by rung to the full 81 epochs. Almost all the compute lives in the handful of blue lines stretching to the right; the grey crowd on the left cost a few epochs each and nothing more.</figcaption>
</figure>

## What the early stopping bought

Same 143 configurations explored, two ways to spend the epochs:

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/search_compute.png" alt="Two bars of total training passes: random search uses far more than Hyperband, with both reaching about the same test accuracy noted above them.">
  <figcaption>The same 143 configurations, costed two ways. Random search trains every one to the full 81 epochs, 11,583 training passes. Hyperband explores the same 143 but stops the losers early, 1,581 passes, 7.3x less. The test accuracy printed on each bar is within a point either way, so the cheaper search did not land a meaningfully worse model.</figcaption>
</figure>

Random search spent 11,583 training passes to find its best configuration. Hyperband explored the same 143 configurations in 1,581 passes, 7.3x less compute, because it stopped the losers after a rung or two instead of training them to the end. On the test set it landed at 91.6% against random search's 92.5%. That gap is about a dozen of the 1,301 test stars, and it is one run on one seed, so I read it as a tie rather than a loss. On the validation score the search was actually optimising, Hyperband's pick was the better of the two.

On eight workers that compute saving is a wall-clock saving too: the Hyperband search finished in about 2.7 seconds against random search's 8.3. The search itself parallelises across configurations, so adding workers helps until there are no more models to run at once:

<figure>
  <img src="{{ site.baseurl }}/images/blog/dask-hyperband-variable-stars/search_scaling.png" alt="Hyperband wall time against worker count: it falls steeply to four workers, then flattens.">
  <figcaption>Hyperband wall time against worker count, on the same search. It drops steeply from one worker to four, then flattens: the search parallelises across configurations, not within a fit, so once there are enough workers to run every model alive at a given rung, the extra ones sit idle.</figcaption>
</figure>

## Where this stops being the answer

Hyperband only helps a model you can stop and restart, one that learns a bit at a time. The random forest has no `partial_fit`: there is no half-trained forest to judge after one epoch, so none of this applies to it, and you are back to plain random search if you want to tune one. And the forest is still the better classifier here, 96.8% against the tuned linear model's 91.6%. For a table of sixteen features, the forest wins and barely needs tuning; the search lesson only shows up because I picked the model that does.

The other limit is scale. On this problem the absolute saving is seconds, because an SGD epoch on four thousand rows is nearly free. The reason to care is what happens when an epoch is *not* free: tuning a gradient-boosted model with hundreds of rounds, or a neural net with real epochs, where a single bad configuration trained to the end costs minutes. There the 7.3x is minutes saved per search, and the same `HyperbandSearchCV` call scales out across a cluster without changing a line. That is the case I would actually reach for it, and the one I will try next.

<a href="https://github.com/nilesh-patil/dask-experiments" class="btn-soft btn-soft--primary">dask-experiments on GitHub</a>
<a href="{{ site.baseurl }}/posts/dask-parallel-light-curves/" class="btn-soft">Part one: the feature loop</a>
<a href="{{ site.baseurl }}/posts/distributed-kmeans-clustering/" class="btn-soft">When the model itself outgrows one machine</a>
