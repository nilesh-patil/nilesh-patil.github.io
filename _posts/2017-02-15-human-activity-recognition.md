---
layout: single
title: "Five numbers tell your phone what you're doing"
date: 2017-02-15T15:39:55-04:00
last_modified_at: 2017-02-15T14:19:19-04:00
categories: [blog]
tags: [sensors, model, RandomForest, project]
excerpt: "A waist-mounted phone classifies 6 daily activities at 94.37% test accuracy, and it gets there on 5 of 561 engineered features."
redirect_from:
  - /blog/human-activity-recognition/
header:
  overlay_image: /images/blog/headers/human-activity.jpg
  overlay_filter: 0.4
  teaser: /images/blog/headers/human-activity.jpg
---

<style>
.btn-soft{display:inline-block;padding:.5em 1.1em;border:1px solid currentColor;border-radius:6px;font-size:.85em;line-height:1.4;text-decoration:none;opacity:.85;transition:opacity .15s ease}
.btn-soft:hover{opacity:1;text-decoration:none}
.btn-soft--primary{font-weight:600}
</style>

When I started this project I assumed I would need most of the 561 features the dataset hands you. I was wrong about how few it takes. A phone clipped to your waist can tell walking-upstairs from sitting at 94.37% accuracy, on people it never saw during training, and the final model gets there on 5 of those 561 numbers. The other 556 I threw away, and the accuracy barely moved. I did not expect that going in. So I worked my way down the dependency chain to understand it: what the raw signal actually carries, then which of its features survive a hard cut, then which model reads the survivors best, and finally where the whole thing would stop holding.

The project and code live here:

[Companion repo on GitHub](https://github.com/nilesh-patil/HumanActivityRecognition){: .btn-soft .btn-soft--primary}

## The signal: 30 people, 6 activities, 561 features

Before any of the modeling, I wanted to understand what was actually being measured. The data comes from the UCI Human Activity Recognition project. 30 volunteers, aged between 19-48 years, each wore a waist-mounted smartphone with embedded inertial sensors while going about ordinary daily living. The raw signal is acceleration along the x, y, and z axes, sampled over time. From those raw streams the project derives 561 attributes per record: time- and frequency-domain summaries alongside orientation angles like `angle(X,gravityMean)` and the `tGravityAcc` family. Two extra columns name the subject and the activity. The full data and the related papers live at the [UCI ML repository page](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).

There are 6 categories of activity to tell apart:

1. `standing`
2. `sitting`
3. `laying`
4. `walking`
5. `walking-downstairs`
6. `walking-upstairs`

I used the dataset packaged as a single `.RData` file rather than stitching together the separate raw text files. A `subject` column identifies the user, the last column names the activity, and every other column holds one of the 561 derived features. The values are already normalized, so feature scales are comparable out of the box.

The first thing I noticed once I started poking at the columns is that 561 features do not span 561 independent dimensions. They are all derived from a handful of raw acceleration streams, so a lot of them move together. To see that structure I built a correlation matrix across all 561 variables, and it came back dense with high-correlation blocks. When a cluster of features carries the same signal, you can keep one of them and drop the rest without losing what it tells you.

<center>
<figure class="full">
<img src="/images/blog/activityRecognition/image8.png" alt="Correlation matrix for all 561 sensor features">
<figcaption>Correlation matrix across all 561 features. The bright square blocks along the diagonal are the gravity and body-acceleration families correlating tightly within themselves, so a single member stands in for the whole cluster.</figcaption>
</figure>
</center>

A couple of other checks pointed the same way. Some variables barely change across records, and a feature with near-zero variance carries almost nothing for a classifier, so it is a safe drop. There were no missing values anywhere in the columns, which meant I could work with the complete dataset and not worry about imputation. I tried plotting per-category distributions as well, but with this many features that quickly turns into a wall of charts. The one thing that does jump out of those distributions is a split into two broad groups of activity, which felt like a clue worth holding onto.

<center>
<figure class="full">
<img src="/images/blog/activityRecognition/image9.png" alt="Representative distribution of a sensor feature across activity categories">
<figcaption>A representative feature distribution. Even one feature separates the activities into two broad groups, a hint that a small handful of features might carry the classification.</figcaption>
</figure>
</center>

## The features: letting the forest pick them

So the signal was smaller than its dimension count, but I still did not know which features to keep. I did not want to pick them by hand. My hunches about which sensor readings mattered would just be hunches, so I let the selection run on Random Forest importance scores rather than domain intuition.

First I had to split the data, and I wanted to get that right before anything else touched it. I divided it into train and test sets in a 7:3 ratio by random sampling without replacement, so each set stays representative of the whole. Sampling per output class instead made no meaningful difference here. To keep everything reproducible I set `RandomState=42` at the top, which fixes the train and test sets and the feature draws on every run.

Then the test set goes in a drawer. It plays no part in feature selection, training, or tuning. For tuning I lean on the out-of-bag (OOB) score instead. A Random Forest trains each tree on a bootstrap sample, and the rows left out of a given tree's bootstrap act as a built-in held-out set for that tree. Averaged across the forest, the OOB score gives me a validation signal for free during training, without ever touching the real test set. That property is what let me tune freely without worrying I was quietly leaking the test set into my decisions.

<center>
<figure class="full">
<svg viewBox="0 0 640 250" role="img" style="width:100%;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title>OOB score as an internal validation loop</title>
<desc>The full dataset splits 70 percent train and 30 percent frozen test. Inside the training set, out-of-bag rows act as a validation signal during training, while the test set stays untouched until the final check.</desc>
<rect x="20" y="100" width="160" height="50" rx="6" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="100" y="122" text-anchor="middle" font-size="13" fill="currentColor">All records</text>
<text x="100" y="139" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">561 features</text>
<line x1="180" y1="125" x2="230" y2="70" stroke="currentColor" stroke-width="1.5"/>
<line x1="180" y1="125" x2="230" y2="190" stroke="currentColor" stroke-width="1.5"/>
<rect x="230" y="40" width="180" height="120" rx="6" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="1.5"/>
<text x="320" y="60" text-anchor="middle" font-size="12" fill="currentColor">Train (70%)</text>
<rect x="250" y="74" width="140" height="34" rx="4" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-width="1"/>
<text x="320" y="95" text-anchor="middle" font-size="11" fill="currentColor">bootstrap trees</text>
<rect x="250" y="116" width="140" height="34" rx="4" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3"/>
<text x="320" y="137" text-anchor="middle" font-size="11" fill="currentColor">OOB rows = validation</text>
<rect x="230" y="180" width="180" height="40" rx="6" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.55"/>
<text x="320" y="205" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7">Test (30%), frozen</text>
<line x1="410" y1="100" x2="470" y2="100" stroke="currentColor" stroke-width="1.5"/>
<polygon points="470,96 478,100 470,104" fill="currentColor"/>
<text x="555" y="96" text-anchor="middle" font-size="12" fill="currentColor">tune on OOB</text>
<line x1="410" y1="200" x2="470" y2="200" stroke="currentColor" stroke-width="1.5" opacity="0.55" stroke-dasharray="4 3"/>
<polygon points="470,196 478,200 470,204" fill="currentColor" opacity="0.55"/>
<text x="555" y="196" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7">touched once, last</text>
</svg>
<figcaption>How OOB score lets you tune without leaking the test set. The data splits 70/30, the test set stays frozen, and out-of-bag rows inside the training set act as an internal validation loop until the very last accuracy check.</figcaption>
</figure>
</center>

To rank the features I trained a forest on the training set using all 561 variables and read off the importance scores. Most features sit near zero. A short head of them carries the signal, which matched what the correlation blocks had already suggested.

<center>
<figure class="full">
<img src="/images/blog/activityRecognition/image10.png" alt="Random forest variable importance scores for all 561 features">
<figcaption>Importance scores across all 561 features. The long flat tail is the redundancy from the correlation matrix showing up again: most features add almost nothing once a few are in.</figcaption>
</figure>
</center>

With the features ranked, two knobs were left to turn: how many top features to keep, and how many trees to grow. I swept them both, iterating over 0-150 trees and for 1-25 variables, and watched the OOB score to see where it flattened out.

<center>
<figure class="half">
<img src="/images/blog/activityRecognition/image2.png" alt="OOB error against number of variables selected">
<img src="/images/blog/activityRecognition/image1.png" alt="OOB error against number of trees">
<figcaption>OOB score against number of features (left) and number of trees (right). Both curves are flat by the low single digits of features and a few dozen trees, which is the whole argument for a tiny model.</figcaption>
</figure>
</center>

The curves plateau fast. By the time you reach a few features and a few dozen trees, adding more buys almost nothing. So I kept 5 features, set the number of trees to 25, and froze the model there, leaving the held-out test accuracy for the very last step.

<center>
<figure class="full">
<svg viewBox="0 0 640 200" role="img" style="width:100%;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title>Funnel from 561 features down to 5</title>
<desc>561 engineered features narrow through Random Forest importance ranking and OOB-guided iteration over feature count and tree count down to the 5 features that survive in the final model.</desc>
<polygon points="20,40 200,40 150,160 70,160" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-width="1.5"/>
<text x="110" y="95" text-anchor="middle" font-size="15" fill="currentColor">561</text>
<text x="110" y="114" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">features</text>
<line x1="210" y1="100" x2="262" y2="100" stroke="currentColor" stroke-width="1.5"/>
<polygon points="262,96 270,100 262,104" fill="currentColor"/>
<text x="240" y="88" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">RF importance</text>
<polygon points="282,55 420,55 392,145 310,145" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-width="1.5"/>
<text x="351" y="104" text-anchor="middle" font-size="12" fill="currentColor">short head</text>
<line x1="424" y1="100" x2="476" y2="100" stroke="currentColor" stroke-width="1.5"/>
<polygon points="476,96 484,100 476,104" fill="currentColor"/>
<text x="450" y="88" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">OOB sweep</text>
<polygon points="496,72 600,72 580,128 516,128" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-width="1.5"/>
<text x="548" y="98" text-anchor="middle" font-size="15" fill="currentColor">5</text>
<text x="548" y="116" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">+ 25 trees</text>
</svg>
<figcaption>The core arc of the project. 561 engineered features narrow through Random Forest importance and OOB-guided iteration down to the 5 that survive, with the tree count set to 25.</figcaption>
</figure>
</center>

Here are the 5 that made the final model:

- `angle(X,gravityMean)`
- `tGravityAcc-mean()-Y`
- `tGravityAcc-min()-X`
- `tGravityAcc-max()-X`
- `tBodyAcc-mad()-X`

Four of the five are gravity-orientation features, and the fifth is body-acceleration spread along X. That lines up with intuition once you say it out loud. Gravity points a fixed way relative to the phone, so the angle between the device and gravity is a clean read on posture. Sitting, standing, and laying are mostly posture. Walking and its stair variants are mostly motion. A few orientation features plus one motion-spread feature is enough to put a record on the right side of that divide.

<center>
<figure class="full">
<img src="/images/blog/activityRecognition/image3.png" alt="Variable importance scores for the final 5 selected features">
<figcaption>Importance scores for the final 5 features. Gravity-orientation dominates, which fits the posture-versus-motion split.</figcaption>
</figure>
</center>

I want to be careful about what I am claiming here. The forest picked these features, not a biomechanics argument. The posture-versus-motion reading is a story I tell after the fact about why the selection makes sense, and it holds up, but the selection itself was algorithmic.

## The model: RF versus SVM

With the 5 features chosen, the next question was which model reads them best. I trained both a Support Vector Machine and a Random Forest on the same setup and compared them on the held-out test set.

|         | Random-Forest | SVM     |
| :-----: | :-----------: | :-----: |
| *Train* | 94.50% (oob)  | 83.48%  |
| *Test*  | 94.37%        | 82.37%  |
{: .table}

Random Forest wins by about 12 points on the test set. The part I found more reassuring than the win is the consistency: the Random Forest OOB validation estimate (94.50%) almost exactly matches the held-out test accuracy (94.37%), which is the signal that the 5-feature model is not memorizing the training set.

<center>
<figure class="full">
<svg viewBox="0 0 640 240" role="img" style="width:100%;height:auto;color:inherit" xmlns="http://www.w3.org/2000/svg">
<title>Random Forest versus SVM, train and test accuracy</title>
<desc>Four horizontal bars whose lengths are proportional to the four accuracies from the table: Random Forest train 94.50 percent, Random Forest test 94.37 percent, SVM train 83.48 percent, SVM test 82.37 percent. Bar length equals accuracy times 5.6 pixels, so 100 percent maps to the 560-pixel axis.</desc>
<line x1="150" y1="20" x2="150" y2="220" stroke="currentColor" stroke-width="1" opacity="0.4"/>
<text x="150" y="236" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">0%</text>
<text x="612" y="236" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">100%</text>
<rect x="150" y="30" width="529.2" height="28" rx="3" fill="currentColor" fill-opacity="0.22" stroke="currentColor" stroke-width="1"/>
<text x="146" y="49" text-anchor="end" font-size="12" fill="currentColor">RF train</text>
<text x="676" y="49" text-anchor="end" font-size="11" fill="currentColor">94.50%</text>
<rect x="150" y="72" width="528.5" height="28" rx="3" fill="currentColor" fill-opacity="0.35" stroke="currentColor" stroke-width="1"/>
<text x="146" y="91" text-anchor="end" font-size="12" fill="currentColor">RF test</text>
<text x="675" y="91" text-anchor="end" font-size="11" fill="currentColor">94.37%</text>
<rect x="150" y="130" width="467.5" height="28" rx="3" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-width="1"/>
<text x="146" y="149" text-anchor="end" font-size="12" fill="currentColor">SVM train</text>
<text x="623" y="149" text-anchor="end" font-size="11" fill="currentColor">83.48%</text>
<rect x="150" y="172" width="461.3" height="28" rx="3" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-width="1"/>
<text x="146" y="191" text-anchor="end" font-size="12" fill="currentColor">SVM test</text>
<text x="617" y="191" text-anchor="end" font-size="11" fill="currentColor">82.37%</text>
</svg>
<figcaption>Random Forest against SVM, train and test. Bar lengths are proportional to the four exact accuracies from the table above (length equals accuracy times 5.6 pixels). The gap between the two models is wide, the gap between Random Forest train and test is barely there.</figcaption>
</figure>
</center>

To see where the remaining errors actually live, I looked at the confusion matrices. The motion activities are clean to tell from the still ones, since orientation alone separates posture from movement. The confusions that survive appear to sit within a group, among the activities that look alike to the sensors. Sitting versus standing is the pair you would expect to blur, since both are upright and still.

<center>
<figure class="half">
<img src="/images/blog/activityRecognition/08.a.ConfusionMatrix-Test_RF.png" alt="Confusion matrix for Random Forest on the test set">
<img src="/images/blog/activityRecognition/08.b.ConfusionMatrix-Test_SVM.png" alt="Confusion matrix for SVM on the test set">
<figcaption>Confusion matrices on the test set, Random Forest (left) and SVM (right). The off-diagonal mass appears to cluster among the similar activities rather than scattering across all six.</figcaption>
</figure>
</center>

The same separability shows up if you go back to raw feature space. Plotting the distribution of `tBodyAccJerk-std()-X` colored by activity, some categories fall into tidy clusters while others overlap, which is exactly the pattern the confusion matrices show. It was reassuring to find the model's errors and the raw data telling the same story.

<center>
<figure class="full">
<img src="/images/blog/activityRecognition/image11.png" alt="Distribution of tBodyAccJerk-std()-X across all six activity categories">
<figcaption>Distribution of `tBodyAccJerk-std()-X` colored by activity. Some categories separate cleanly, others overlap, which mirrors where the confusion matrices put the errors.</figcaption>
</figure>
</center>

## What breaks: the bias caveat

The collapse from 561 to 5 is the part of this that I keep coming back to. On a phone it matters in a concrete way. The whole point of activity recognition on a device is that the device has to do it, on its own battery, while doing everything else. A model that reads 5 features and runs 25 small trees is cheap enough to run continuously without anyone noticing the drain. So the dimensionality collapse is not a tidiness win, it is the thing that makes on-device inference practical at all.

The real limit is the one I built in by choosing Random Forest. I leaned on the assumption that random forests do not usually overfit the training set, and the tight 94.50% to 94.37% gap is what that assumption looks like when it is holding. But it breaks down when the training data is heavily biased. This dataset is relatively balanced across activities and subjects, so the assumption holds here. Move to a population that skews one way, an age band these 30 volunteers never covered, a gait the sensors never saw, and the no-overfit comfort is the first thing I would stop trusting.

That is the part I would want to test next, and it is also the part I cannot answer from this dataset alone. The five numbers are enough to tell what these 30 people were doing. Whether they are enough to tell what you are doing is a question for data I do not have yet.

## References

*Random Forest:*

- <https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm>
- <http://scikit-learn.org/stable/modules/ensemble.html#forest>
- <https://en.wikipedia.org/wiki/Random_forest>

*SVM:*

- <http://scikit-learn.org/stable/modules/svm.html>
- <https://en.wikipedia.org/wiki/Support_vector_machine>

*OOB Score:*

- <https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr>
- <http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html>

*UCI-ML dataset location:*

- <https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones>

*Scikit-Learn:*

- <http://scikit-learn.org/stable/index.html>

*GitHub Page:*

- <https://github.com/nilesh-patil/HumanActivityRecognition>
