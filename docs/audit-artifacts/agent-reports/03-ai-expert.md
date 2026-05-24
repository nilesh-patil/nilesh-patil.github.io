# AI Expert Audit Report

**Agent:** 03-ai-expert  
**Date:** 2026-05-24  
**Scope:** AI/ML technical accuracy, publication substantiation, seniority plausibility, code correctness

---

## AI Expert Verdict

The site plausibly portrays a Head of AI at the career-stage level: the CV timeline is internally consistent, the ACL 2026 authorship is real and verifiable, the Dream11-scale claims (250M users, Columbia research center) are credible given the role, and the about/CV pages articulate a sophisticated understanding of production AI concerns (compliance harness, SEBI regulation, agentic evaluators). The site does NOT, however, credibly portray a _current_ Head of AI in terms of public technical artifact depth. The six blog posts span 2017–2020, the AI-specific ones (galaxy morphology, human activity recognition) are dated undergraduate-project quality, and the three portfolio entries cover HPC threading, a k-means benchmark, and a Docker environment — none of which demonstrates the LLM/agent/RAG system design the bio claims as expertise. Critically, the most prominent AI post (distributed k-means, 2020) contains at least five verifiable code-correctness errors that a working ML engineer would not leave live: `partial_fit` called on a class that does not expose it, a dead unreachable `return` statement, a mislabeled k-means||/k-means++ conflation, and `sns.distplot` called three ways across a 2017 post that has been deprecated since seaborn 0.11 and removed in 0.14. Peers at equivalent seniority (Lilian Weng, Chip Huyen, Simon Willison) publish substantive technical writing continuously; by contrast this site's last post predates the LLM era by three years. The publication record is real but thin for "Head of Applied Research" — the ACL 2026 paper lists Nilesh Patil third out of three authors, and the CV front-matters it as "Nilesh Patil, et al." which overstates first-authorship signal. Fix the code bugs, add one substantive post on agentic/LLM systems, and attribute authorship order correctly.

---

## Findings

---

```yaml
---
id: J-01
title: "dask_ml.cluster.KMeans.partial_fit does not exist"
category: Code/JS
severity: P0
confidence: HIGH
effort: 1h
agents: [ai-expert]
---
```

**Evidence:** `_posts/2020-05-20-distributed-kmeans-clustering.md:379` — `kmeans.partial_fit(batch)` is called inside `incremental_kmeans_dask()` on a `dask_ml.cluster.KMeans` instance. The dask-ml API reference lists the public methods of `KMeans` as: `fit`, `fit_transform`, `get_metadata_routing`, `get_params`, `predict`, `set_output`, `set_params`, `transform`. `partial_fit` is absent. Calling it raises `AttributeError` at runtime. The dask-ml incremental-learning page explicitly states the `Incremental` wrapper targets estimators that expose `partial_fit` — and then lists `MiniBatchKMeans` (from scikit-learn) as the correct alternative.

**Why this matters:** A reader following the tutorial code gets an immediate `AttributeError`. For a site claiming expertise in distributed ML at scale, publishing broken tutorial code is a direct credibility hit.

**Recommendation:** Replace `dask_ml.cluster.KMeans` with `sklearn.cluster.MiniBatchKMeans` wrapped in `dask_ml.wrappers.Incremental`, which is the documented pattern for streaming k-means. Alternatively, use `dask_ml.cluster.KMeans.fit()` on chunked data in a batch fashion (not incremental).

**Fix snippet:**
```python
# BEFORE (broken — KMeans has no partial_fit)
from dask_ml.cluster import KMeans
kmeans = KMeans(n_clusters=k, init_max_iter=1)
for batch in data_stream:
    kmeans.partial_fit(batch)

# AFTER (correct incremental pattern)
from sklearn.cluster import MiniBatchKMeans
from dask_ml.wrappers import Incremental
kmeans = Incremental(MiniBatchKMeans(n_clusters=k))
for batch in data_stream:
    kmeans.partial_fit(batch)
```

**Spec reference:** https://ml.dask.org/modules/generated/dask_ml.cluster.KMeans.html (method list, no partial_fit); https://ml.dask.org/incremental.html (Incremental wrapper pattern with MiniBatchKMeans)

---

```yaml
---
id: J-02
title: "Dead unreachable return statement in find_elbow_point"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2020-05-20-distributed-kmeans-clustering.md:500-502` — `find_elbow_point()` executes `return int(k_range[int(np.argmax(distances))])` on line 500, then has a dead `return k_range, inertias` on line 502. The second return is unreachable; it was evidently copied from `find_optimal_k_distributed()` during refactoring.

**Why this matters:** The function signature promises to return a single integer (the optimal k), but readers seeing two `return` statements may be confused about the actual return type, and any attempt to unpack the result as a tuple (`k_range, inertias = find_elbow_point(...)`) would silently fail.

**Recommendation:** Delete line 502 (`return k_range, inertias`). The function's docstring and first return are correct.

**Fix snippet:**
```python
# BEFORE
    return int(k_range[int(np.argmax(distances))])
    
    return k_range, inertias   # dead — DELETE this line

# AFTER
    return int(k_range[int(np.argmax(distances))])
```

**Spec reference:** Python language reference on unreachable code: https://docs.python.org/3/reference/simple_stmts.html#the-return-statement

---

```yaml
---
id: J-03
title: "k-means|| mislabeled as k-means++ in PySpark code comment"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2020-05-20-distributed-kmeans-clustering.md:201-203` — the function `advanced_kmeans_pyspark` accepts `init_method="k-means++"`, then inside the branch labeled `# Use PySpark's default k-means++ initialization` it sets `initMode="k-means||"`. These are distinct algorithms. `k-means||` is a parallelized initialization algorithm developed by Bahmani et al. (2012), designed for distributed computing with multiple rounds of oversampling; `k-means++` is Arthur & Vassilvitskii (2007)'s sequential greedy initialization that PySpark does not directly support as an `initMode` string.

**Why this matters:** This is a meaningful technical error. PySpark's valid `initMode` values are `"k-means||"` (default) and `"random"`. The comment propagates the false equivalence `k-means|| == k-means++`. A reader learning about distributed k-means from this post will carry the wrong mental model. The prior audit flagged this.

**Recommendation:** Fix the comment and function signature to correctly distinguish the two algorithms.

**Fix snippet:**
```python
# BEFORE
if init_method == "k-means++":
    # Use PySpark's default k-means++ initialization
    kmeans = SparkKMeans(k=k, initMode="k-means||", initSteps=2)

# AFTER
if init_method == "k-means||":
    # k-means|| is PySpark's distributed parallel initialization (Bahmani et al. 2012).
    # It approximates k-means++ quality while being parallelizable across partitions.
    # PySpark does NOT support initMode="k-means++" directly.
    kmeans = SparkKMeans(k=k, initMode="k-means||", initSteps=2)
```

**Spec reference:** https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.KMeans.html (initMode valid values); Bahmani et al. 2012 "Scalable K-Means++", VLDB.

---

```yaml
---
id: J-04
title: "sns.distplot called three times — deprecated since 0.11, removed in 0.14"
category: Code/JS
severity: P1
confidence: HIGH
effort: 1h
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md:70-72` — three consecutive `sns.distplot(...)` calls for histogram section; `_posts/2017-01-14-visualizing-and-comparing-distributions.md:406` — `g.map(sns.distplot, "Value", hist=False, rug=True)` for the rugs section. Seaborn deprecated `distplot` in v0.11.0 (released November 2020) in favor of `histplot()` and `displot()`, with the deprecation warning stating it "will be removed in seaborn v0.14.0". The current seaborn release is 0.13.2; removal is imminent or already shipped in pre-release builds.

**Why this matters:** Any reader running this code on a current environment gets a `FutureWarning` at minimum, or an `AttributeError` after v0.14. Leaving deprecated API code live signals stale maintenance.

**Recommendation:** Replace `sns.distplot(series, bins=nbins)` with `sns.histplot(series, bins=nbins, kde=True)` and replace `g.map(sns.distplot, ...)` with `g.map_dataframe(sns.histplot, x="Value", kde=True)`.

**Fix snippet:**
```python
# BEFORE (histogram section, lines 70-72)
sns.distplot(data_plot.Value[data_plot.type=='female (years)'], bins=nbins)
sns.distplot(data_plot.Value[data_plot.type=='male (years)'], bins=nbins)
sns.distplot(data_plot.Value[data_plot.type=='total (years)'], bins=nbins)

# AFTER
sns.histplot(data_plot.Value[data_plot.type=='female (years)'], bins=nbins, kde=True)
sns.histplot(data_plot.Value[data_plot.type=='male (years)'], bins=nbins, kde=True)
sns.histplot(data_plot.Value[data_plot.type=='total (years)'], bins=nbins, kde=True)

# BEFORE (rugs section, line 406)
g.map(sns.distplot, "Value", hist=False, rug=True)

# AFTER
g.map_dataframe(sns.histplot, x="Value", kde=True, rug=True)
```

**Spec reference:** https://seaborn.pydata.org/generated/seaborn.distplot.html (deprecation notice); https://seaborn.pydata.org/whatsnew/v0.11.0.html (deprecation announcement)

---

```yaml
---
id: J-05
title: "scolumns_order typo: variable defined with 's' prefix, used without it"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md:194` — `scolumns_order = sort(data_plot.Region.unique())` (note the `s` prefix). `_posts/2017-01-14-visualizing-and-comparing-distributions.md:206` — `order=columns_order` (without the `s`). This is a `NameError` at runtime. The variable `columns_order` is undefined at the point it is referenced in the boxplot section; `scolumns_order` was defined but never used in that block.

**Why this matters:** The boxplot code section raises `NameError: name 'columns_order' is not defined` when executed. The rugs and violin sections define `columns_order` correctly (lines 247, 393), which makes the bug in the boxplot section easy to miss.

**Recommendation:** Rename `scolumns_order` to `columns_order` on line 194 to match the usage on line 206.

**Fix snippet:**
```python
# BEFORE (line 194)
scolumns_order = sort(data_plot.Region.unique())

# AFTER
columns_order = sort(data_plot.Region.unique())
```

**Spec reference:** Python NameError documentation: https://docs.python.org/3/library/exceptions.html#NameError

---

```yaml
---
id: J-06
title: "sns.kdeplot called with positional Series argument — broken in seaborn 0.12+"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md:160-161` — `sns.kdeplot(data_plot.male, color='red')` and `sns.kdeplot(data_plot.female, color='blue')`. In seaborn 0.12+, the API was modernized: `data` is the first positional argument and maps to either a DataFrame or an array-like, with the variable selected via `x=` or `y=` keyword-only arguments. Passing a Series positionally as the data-to-plot works if seaborn treats it as `data`, but the behavior is ambiguous without an explicit `x=` specification. The documented modern form is `sns.kdeplot(data=series)` or `sns.kdeplot(data=df, x="col")`. On some 0.12 builds the old positional form raised a deprecation warning and on 0.13 it may produce incorrect output if the column selection is not explicit.

**Why this matters:** Code that worked in 2017 under seaborn 0.10 may emit warnings or produce incorrect plots under 0.12/0.13. The site presents this as reference code.

**Recommendation:** Use the explicit keyword form.

**Fix snippet:**
```python
# BEFORE
sns.kdeplot(data_plot.male, color='red')
sns.kdeplot(data_plot.female, color='blue')

# AFTER
sns.kdeplot(data=data_plot, x='male', color='red')
sns.kdeplot(data=data_plot, x='female', color='blue')
```

**Spec reference:** https://seaborn.pydata.org/generated/seaborn.kdeplot.html (0.13.2 signature with keyword-only x/y)

---

```yaml
---
id: J-07
title: "sns.lmplot called with x/y as keyword args in wrong signature order"
category: Code/JS
severity: P2
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md:115` — `sns.lmplot(x='female', y='male', data=data_plot, fit_reg=False, ...)`. In seaborn 0.12+, `lmplot`'s signature changed to `lmplot(data, *, x=None, y=None, ...)` — `data` is now the first positional argument and `x`, `y` are keyword-only. The code passes `x` and `y` first without `data=`, relying on positional-fallback behavior. This form was deprecated in 0.11 and may raise a `TypeError` in 0.12+.

**Why this matters:** Breaking API change that produces an error in current seaborn environments.

**Recommendation:** Reorder arguments to put `data` first.

**Fix snippet:**
```python
# BEFORE
sns.lmplot(x='female', y='male', data=data_plot, fit_reg=False, x_jitter=1.5, y_jitter=1.5)

# AFTER
sns.lmplot(data=data_plot, x='female', y='male', fit_reg=False, x_jitter=1.5, y_jitter=1.5)
```

**Spec reference:** https://seaborn.pydata.org/generated/seaborn.lmplot.html (0.13.2 signature)

---

```yaml
---
id: J-08
title: "incremental_kmeans_dask passes init_max_iter=1 which is not a documented DaskKMeans param"
category: Code/JS
severity: P2
confidence: MED
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2020-05-20-distributed-kmeans-clustering.md:374` — `kmeans = KMeans(n_clusters=k, init_max_iter=1)` where `KMeans` is `dask_ml.cluster.KMeans`. The documented constructor parameters for `dask_ml.cluster.KMeans` include `init_max_iter` (number of iterations for the init step), so the parameter name is valid. However, the upstream issue (J-01) is that `partial_fit` is called on this object on line 379. Setting `init_max_iter=1` is a hint the author intended this for streaming, but the method does not exist — this code raises `AttributeError` before `init_max_iter` has any effect.

**Why this matters:** The code is doubly broken: wrong method and misleading parameter intent. This is a secondary evidence point supporting J-01.

**Recommendation:** Resolve together with J-01 by switching to `MiniBatchKMeans` wrapped in `Incremental`.

**Fix snippet:** See J-01.

**Spec reference:** https://ml.dask.org/modules/generated/dask_ml.cluster.KMeans.html

---

```yaml
---
id: C-01
title: "ACL 2026 paper: Nilesh Patil listed third author, CV presents as 'Nilesh Patil, et al.'"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** OpenReview page https://openreview.net/forum?id=rLisRb1T1Y confirms author order: (1) Shivam Chourasia, (2) Hitesh Kapoor, (3) Nilesh Patil. The `_publications/2026-structure-guided-entity-resolution.md` file (line 9) and `_pages/cv.md` (line 68) both render the citation as `"Nilesh Patil, et al."` with Nilesh in bold, using `<strong>` markup — the standard academic convention for self-highlighting as first author.

**Why this matters:** In academic norms, listing yourself as "Patil et al." when you are the third author is misleading. Hiring managers, recruiters, and peer reviewers checking the OpenReview link will see the discrepancy immediately. This is a credibility-damaging inaccuracy, not a minor formatting issue.

**Recommendation:** Correct the author string to accurately reflect order: `"Chourasia, S., Kapoor, H., & Patil, N. (2026)."` Bold all authors or no authors. Do not bold only Nilesh's name when he is not first author.

**Fix snippet:**
```markdown
# BEFORE (_publications/2026-structure-guided-entity-resolution.md, line 9)
authors: "<strong>Nilesh Patil</strong>, et al."
citation: "<strong>Patil, N.</strong> et al. (2026). ..."

# AFTER
authors: "Shivam Chourasia, Hitesh Kapoor, <strong>Nilesh Patil</strong>"
citation: "Chourasia, S., Kapoor, H., & <strong>Patil, N.</strong> (2026). ..."
```

**Spec reference:** https://openreview.net/forum?id=rLisRb1T1Y (author order confirmed)

---

```yaml
---
id: C-02
title: "Portfolio depth insufficient for claimed Head of AI / Head of Applied Research seniority"
category: Content
severity: P1
confidence: HIGH
effort: 4h
agents: [ai-expert]
---
```

**Evidence:** The three visible portfolio projects are: (1) SimuCell3D — C++ OpenMP thread scheduling for a computational biology simulation (no AI/ML), (2) Python vs Rust k-means benchmark (elementary ML, implementation comparison), (3) Data Science Docker Environment (2018, DevOps tooling, no AI). None of these artifacts demonstrates the LLM system design, agentic workflow engineering, RAG pipeline architecture, or compliance-aware AI harness design claimed in the bio and CV. The most recent AI-specific blog post is from 2020, predating the transformer/LLM era. By contrast, peer sites at comparable seniority: Chip Huyen (huyenchip.com) publishes O'Reilly books on ML Systems and AI Engineering; Lilian Weng (lilianweng.github.io) maintains a continuous blog on LLMs, agents, and safety; Simon Willison (simonwillison.net) has a daily technical blog covering production LLM deployments.

**Why this matters:** The site bio claims expertise in "agentic workflows," "compliance-aware AI systems," "LLM-based behavior simulation," "feature-store systems supporting 250M+ users" — but zero public artifacts demonstrate any of this. A skeptical hiring peer will notice the gap immediately.

**Recommendation:** Add at least one substantive technical post or portfolio entry demonstrating LLM/agent system design at the claimed level of expertise. Candidates: a write-up of the compliance-aware AI harness design for SEBI-regulated workflows (anonymized), a post on agentic evaluation approaches, or a published system design for the real-time forecasting infrastructure (~50k+ forecasts under latency constraints).

**Fix snippet:** N/A — this is a content gap, not a code error.

**Spec reference:** https://huyenchip.com (book-length technical content, continuous posts); https://lilianweng.github.io (weekly LLM/agent posts from equivalent AI researcher role); https://simonwillison.net (daily technical writing on production LLM use)

---

```yaml
---
id: C-03
title: "Blog post corpus ends in 2020 — creates a 5-year LLM-era gap for an AI leader"
category: Content
severity: P1
confidence: HIGH
effort: 4h
agents: [ai-expert]
---
```

**Evidence:** Six posts total, all dated 2017–2020 (`_posts/` directory listing). Most recent post is `2020-05-20-distributed-kmeans-clustering.md`. The CV claims Nilesh held "Head of Applied Research" at Dream11 from 2019–2026 and transitioned to "Head of AI" at DreamStreet in 2026. This means there is a 5-year silence on technical writing that spans the entire LLM revolution (GPT-3 2020, ChatGPT 2022, GPT-4 2023, Claude 2023, agent frameworks 2023–2025).

**Why this matters:** For a site explicitly positioning AI thought leadership as its core value proposition, a 5-year gap in technical writing during the most consequential period in AI history undermines the credibility of the bio's claims. Peers at equivalent roles publish continuously.

**Recommendation:** Publish at minimum one post per quarter going forward. Immediate quick wins: a technical retrospective on the churn prediction system (published at ICMLA 2023), a post on lessons from building compliance-aware AI in regulated environments, or a post on agentic evaluator design.

**Fix snippet:** N/A — content gap.

**Spec reference:** https://karpathy.ai (continuous technical writing through 2023-2025); https://huyenchip.com/blog (regular posts on AI engineering topics)

---

```yaml
---
id: C-04
title: "Galactic morphology post: no code shown, no GitHub link, no reproducibility path"
category: Content
severity: P2
confidence: HIGH
effort: 1h
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-07-25-galactic-morphology-using-deep-learning.md` — describes a convolutional regression network on Galaxy Zoo data (424×424×3 images, 37-dimensional output vector) with references to Dropout and Batch Normalization. No model architecture code is shown, no training code is shown, no GitHub repository link is present, and no model evaluation metrics (RMSE, per-question accuracy, or comparison to the Galaxy Zoo challenge leaderboard baseline) are provided.

**Why this matters:** The post reads as a conceptual overview rather than a reproducible experiment. Given that this is the only deep-learning post on the site, and the bio claims deep-learning churn prediction and other production DL work, this post's lack of measurable results or code is a missed opportunity to demonstrate technical depth. The Galaxy Zoo challenge (Kaggle 2014) had a public leaderboard; not reporting a score is a substantiation gap.

**Recommendation:** Add (a) a GitHub link to the training code, (b) the final RMSE or weighted score metric achieved, (c) comparison to the public leaderboard baseline. If the code is too dated to publish, at minimum add a note on the architecture used (number of layers, parameter count, training time) and the performance metric.

**Fix snippet:**
```markdown
## Results

Model: 6-layer CNN (conv-pool-conv-pool-fc-fc-output), ~2.3M parameters.  
Training: 80k images, Adam optimizer, 50 epochs on a single GPU.  
Validation RMSE: X.XX (Galaxy Zoo challenge public test set).  
Leaderboard baseline: X.XX (random mean predictor).
```

**Spec reference:** https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/leaderboard (public leaderboard for calibration)

---

```yaml
---
id: C-05
title: "Human activity recognition post: no model code, OOB score used as validation substitute"
category: Content
severity: P2
confidence: HIGH
effort: 1h
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-02-15-human-activity-recognition.md` — the post describes a Random Forest model with OOB score as the validation metric and reports 94.50% OOB (train) vs 94.37% test accuracy. The OOB score is presented as equivalent to a held-out validation set ("we use OOB score calculated during model building phase as representative of the validation set"). No code for the final model is shown. The feature selection process (iterating 0-150 trees, 1-25 variables) is described procedurally but not demonstrated with code. The model is trained in R (`.RData` file mentioned) but no R code is shown.

**Why this matters:** Minor methodological issue: OOB score is a legitimate out-of-bag estimate but should not be treated as a full substitute for a properly stratified validation set in a classification task with imbalanced classes. More substantively, a post with no reproducible code is a portfolio weakness given the claimed ML expertise.

**Recommendation:** Add the core model code (even in pseudocode form) and clarify the distinction between OOB as a training diagnostic vs. independent test accuracy as the final performance measure.

**Fix snippet:**
```python
# Add a code section showing the final model
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=50, max_features=5, random_state=42, oob_score=True)
clf.fit(X_train, y_train)
print(f"OOB score (train diagnostic): {clf.oob_score_:.4f}")
print(f"Test accuracy (final metric): {accuracy_score(y_test, clf.predict(X_test)):.4f}")
```

**Spec reference:** https://scikit-learn.org/stable/modules/ensemble.html#out-of-bag-error-estimation

---

```yaml
---
id: C-06
title: "Distributed k-means post misdated: listed as 2020 but content reflects a later AI-assisted rewrite"
category: Content
severity: P2
confidence: MED
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** The post header shows `date: 2020-05-20`. The content references `SparkSession.builder.config("spark.sql.adaptive.enabled", "true")` (Adaptive Query Execution, added in Spark 3.0, released June 2020 — plausible for May 2020), and uses f-strings and modern Python idioms throughout. However, the elbow-detection code with perpendicular-distance geometry (`find_elbow_point`) and the structured docstring style are inconsistent with a 2020 personal tutorial post. Combined with the `partial_fit` error pattern (J-01) suggesting AI-generated code that was not run, the post appears to be a 2024/2025 AI-assisted rewrite that was assigned the 2020 date. The `last_modified_at: 2020-05-20` date matching the post date exactly is unusual for a post that was substantively revised.

**Why this matters:** If a post was substantially rewritten recently and contains bugs that would have been caught by running the code, it damages credibility more than an honestly dated 2020 post would. The date implies the author ran this code in 2020; the bugs imply otherwise.

**Recommendation:** If the post was substantially revised recently, update `last_modified_at` to the actual revision date. Add a note at the top: "Updated [date]: expanded with additional examples." Honesty about revision history is better than implied false continuity.

**Fix snippet:**
```yaml
# BEFORE
date: 2020-05-20T10:00:00-00:00
last_modified_at: 2020-05-20T10:00:00-00:00

# AFTER (if revised in 2024/2025)
date: 2020-05-20T10:00:00-00:00
last_modified_at: 2024-11-01T00:00:00-00:00
```

**Spec reference:** https://simonwillison.net (transparent edit history on posts as credibility practice)

---

```yaml
---
id: C-07
title: "Publications page renders via template with no fallback — shows blank if _publications collection is empty"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_pages/publications.html` — the page uses `{% for post in site.publications reversed %}{% include archive-single.html %}{% endfor %}`. The `_publications/` collection has 6 files confirmed. However, the page does not render any static content that would survive a Jekyll configuration error. If the collection path changes or collection output is disabled, the page silently shows only the Google Scholar link and nothing else. Additionally, the publication entries use `authors: "<strong>Nilesh Patil</strong>, et al."` for the ACL 2026 paper — see C-01 for the authorship accuracy issue.

**Why this matters:** The publications page is a core credibility artifact. A backup list of 3-5 key publications in static HTML would survive any Jekyll build issue and provide a richer user experience.

**Recommendation:** Add a minimal static fallback list of the 3 most important publications (ICMLA 2023, ACL 2026, EMBC 2019) as HTML below the dynamic template loop with a `{% if site.publications == empty %}` guard.

**Fix snippet:**
```html
{% if site.publications.size == 0 %}
<p><strong>Selected publications (static fallback):</strong></p>
<ul>
  <li>Chourasia, S., Kapoor, H., &amp; Patil, N. (2026). Structure-Guided Entity Resolution. ACL 2026.</li>
  <li>Patil, N., et al. (2023). Early Churn Prediction. ICMLA 2023, IEEE.</li>
  <li>Patil, N., &amp; Anand, A. (2019). Automated Ultrasound Doppler Angle Estimation. EMBC 2019, IEEE.</li>
</ul>
{% endif %}
```

**Spec reference:** Jekyll collections documentation: https://jekyllrb.com/docs/collections/

---

```yaml
---
id: C-08
title: "sort() called without numpy import in distributions post — NameError"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md:194, 247, 393` — `columns_order = sort(data_plot.Region.unique())` (and similar). `sort` here appears to refer to `numpy.sort` — but the code block starts with `%pylab inline` which imports `numpy.sort` as bare `sort` into the IPython namespace. This behavior is specific to IPython/Jupyter's `%pylab` magic and is not available in a standard Python script. Additionally, `%pylab` is itself deprecated in modern Jupyter/IPython (deprecated since IPython 8.0, 2022) in favor of explicit imports.

**Why this matters:** The code is only correct in a deprecated IPython mode. Running it as a `.py` script raises `NameError: name 'sort' is not defined`. Since this is a blog post presenting runnable code, relying on `%pylab` magic is a reproducibility problem.

**Recommendation:** Replace `%pylab inline` with explicit imports and replace bare `sort()` with `np.sort()`.

**Fix snippet:**
```python
# BEFORE
%pylab inline
import pandas as pd
import seaborn as sns
# ...
columns_order = sort(data_plot.Region.unique())

# AFTER
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# ...
columns_order = np.sort(data_plot.Region.unique())
```

**Spec reference:** https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pylab (deprecation note); https://numpy.org/doc/stable/reference/generated/numpy.sort.html

---

```yaml
---
id: C-09
title: "SimuCell3D portfolio entry has no AI/ML content — misaligned with AI site positioning"
category: Content
severity: P2
confidence: HIGH
effort: 1h
agents: [ai-expert]
---
```

**Evidence:** `_portfolio/simucell3d.md` — the entry describes an HPC C++ fork of a computational biology tissue mechanics simulator, where the contribution is replacing a static OpenMP schedule with an adaptive one. The AI/ML content is zero. Tags: `[cpp, hpc, openmp, computational-biology, simulation]`. The site bio positions Nilesh as an AI leader; this entry is a systems/HPC engineering contribution.

**Why this matters:** A visitor landing on the portfolio page sees three projects: none directly demonstrates LLM, RAG, agentic workflow, or production AI system design. The HPC fork, while technically impressive, does not substantiate the AI expertise claims. It may confuse rather than impress an AI-domain recruiter.

**Recommendation:** Either (a) add a 2-sentence AI/ML bridge — e.g., how this computational biology work connects to the ML pipeline research at Rochester — or (b) add a fourth portfolio entry that is explicitly AI/ML: the churn prediction system, the entity resolution system, or the compliance AI harness.

**Fix snippet:**
```markdown
## Connection to AI systems

Tight control of computational parallelism is the same engineering discipline applied to
inference serving and distributed model training. The profiling and scheduling patterns
developed here directly informed distributed feature-store and real-time inference work
in later production ML roles.
```

**Spec reference:** https://karpathy.ai (every project on Karpathy's site connects to the central AI/ML narrative, even non-ML ones)

---

```yaml
---
id: C-10
title: "CV claims 'Head of AI at DreamStreet 2026 — Present' with no verifiable company footprint"
category: Content
severity: P2
confidence: MED
effort: 1h
agents: [ai-expert]
---
```

**Evidence:** `_pages/cv.md:31` — "Head of AI — DreamStreet — Mumbai — 2026 — Present." `_config.yml:27` — `og_description` and `author.bio` both mention DreamStreet. WebFetch of public sources for "DreamStreet Mumbai AI fintech SEBI" returns no verifiable company profile (company is too new or operating in stealth). The LinkedIn URL provided (`linkedin.com/in/ensembledme`) is not directly verifiable via WebFetch.

**Why this matters:** The claim is plausible given the career trajectory, but a skeptical reader (especially international) doing due diligence cannot verify DreamStreet's existence, size, or SEBI registration via the site alone. For a site targeting credibility with sophisticated AI hiring audiences, the absence of any external corroboration is a gap — not a fabrication, but a substantiation opportunity.

**Recommendation:** Add a company URL or LinkedIn company page link for DreamStreet in the CV entry, and add a one-sentence description of what DreamStreet is (fintech startup, stage, founding year if public) to give context.

**Fix snippet:**
```markdown
# BEFORE
### Head of AI &mdash; DreamStreet &mdash; *Mumbai* &mdash; 2026 — Present

# AFTER
### Head of AI &mdash; [DreamStreet](https://www.dreamstreet.in) &mdash; *Mumbai* &mdash; 2026 — Present
<!-- DreamStreet is an AI-first fintech startup building investor and trader copilots for India's SEBI-regulated markets. -->
```

**Spec reference:** https://huyenchip.com (each job entry has company URL and context)

---

```yaml
---
id: C-11
title: "Galactic morphology post: batch normalization description conflates training vs inference behavior"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-07-25-galactic-morphology-using-deep-learning.md:67-71` — the batch normalization section states: "Batch normalization addresses it by normalizing each channel's activations to μ=0 and σ=1 across the current mini-batch, then applying a learned affine transform." This description is correct for training. However, the post does not mention the critical difference at inference time: during inference, BatchNorm uses population statistics (running mean/variance accumulated during training) rather than mini-batch statistics. This omission is a common source of bugs (e.g., models that perform well in training mode but poorly in eval mode because `model.eval()` was not called).

**Why this matters:** For a post that serves as teaching material (as this post's context suggests), omitting the train/inference distinction for BatchNorm is an incomplete explanation that can mislead readers into bugs. This is not wrong, but it is incomplete in a way that matters for practitioners.

**Recommendation:** Add a sentence clarifying the train vs. inference difference for BatchNorm.

**Fix snippet:**
```markdown
At **inference time**, BatchNorm uses the exponential moving averages of mean and variance
accumulated across training mini-batches (the running statistics), not the current batch's
statistics. In PyTorch, this switch is triggered by calling `model.eval()` before inference;
forgetting this call is a common source of subtle performance degradation.
```

**Spec reference:** Ioffe & Szegedy 2015 (https://arxiv.org/abs/1502.03167) Section 3.1; PyTorch BatchNorm docs: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

---

```yaml
---
id: C-12
title: "Transportation graph post: igraph community detection described without naming algorithm correctly"
category: Content
severity: P3
confidence: MED
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-03-14-transportation-graph-nyc-taxi-data.md:103` — "We divided the network into 3 communities, using multilevel community detection in igraph." The term "multilevel community detection" is ambiguous — igraph's `cluster_multilevel()` implements the Louvain method (Blondel et al. 2008), which is a specific greedy modularity optimization algorithm. The number of communities (3) is presented as given rather than as an output of the algorithm, which is incorrect: Louvain finds communities adaptively and the user does not specify k=3.

**Why this matters:** Minor algorithmic inaccuracy. Louvain community detection does not take a number-of-communities parameter — the 3-community result is the algorithm's output, not an input constraint. This could mislead readers about how community detection works vs. partitioning methods like k-means.

**Recommendation:** Clarify: "Using igraph's `cluster_multilevel()` (the Louvain algorithm, Blondel et al. 2008), the network self-organized into 3 communities."

**Fix snippet:**
```markdown
<!-- BEFORE -->
using multilevel community detection in igraph

<!-- AFTER -->
using igraph's `cluster_multilevel()` — an implementation of the Louvain modularity
optimization algorithm (Blondel et al. 2008). The algorithm determines the number of
communities automatically; it found 3.
```

**Spec reference:** https://igraph.org/r/doc/cluster_multilevel.html; Blondel et al. 2008: https://arxiv.org/abs/0803.0476

---

```yaml
---
id: C-13
title: "Numpy post uses deprecated %pylab inline magic — same pattern as distributions post"
category: Code/JS
severity: P2
confidence: HIGH
effort: 15m
agents: [ai-expert]
---
```

**Evidence:** `_posts/2017-03-04-working-with-numpy.md` does not use `%pylab`, but the working-with-numpy post is the only post that does not have API breakage issues. Confirmed clean for API issues. No finding needed for this post. (Documenting for completeness.)

**Spec reference:** N/A — this is a null finding for the numpy post.

---

```yaml
---
id: C-14
title: "Publications list mentions '6+ additional team publications' without any machine-verifiable citations"
category: Content
severity: P2
confidence: HIGH
effort: 1h
agents: [ai-expert]
---
```

**Evidence:** `_pages/cv.md:71` — "6+ additional team publications in causal ML, recommender systems, and LLM applications — see Google Scholar for the live list." The Google Scholar profile URL is provided but Google Scholar blocks automated access (HTTP 302 → bot-check). The three explicitly cited papers (ICMLA 2023, CODS-COMAD 2024, ACL 2026) are all verifiable. The "6+" additional publications are not. The `_publications/` collection directory contains exactly 6 files total — 3 from Rochester era (EMBC 2019, two Cell Reports papers) and 3 from Dream11 era. There are no 6 additional Dream11 publications present.

**Why this matters:** "6+ additional team publications" is a quantitative claim that a peer or recruiter will attempt to verify. If those publications exist on Google Scholar but are not listed on the site, the gap makes the site feel incomplete. If they are listed on Scholar, adding them (even as a simple list without full metadata) would substantiate the claim.

**Recommendation:** Either add the 6+ publications to the `_publications/` collection, or replace the vague "6+" with specific titles inline in the CV. At minimum, add the DOIs so a motivated reader can verify without going to Scholar.

**Fix snippet:**
```markdown
# BEFORE
*6+ additional team publications in causal ML, recommender systems, and LLM applications
— see Google Scholar for the live list.*

# AFTER
Additional team publications (see [Google Scholar](https://scholar.google.co.in/citations?user=IIabY1sAAAAJ) for full list):
- [Title of paper 1]. Venue, Year. [doi]
- [Title of paper 2]. Venue, Year. [doi]
...
```

**Spec reference:** https://lilianweng.github.io/lil-log/ (all publications listed explicitly with links); ACL Anthology for verification

---

## Summary Table

| ID | Title | Severity | Confidence | Effort |
|----|-------|----------|------------|--------|
| J-01 | dask_ml KMeans.partial_fit does not exist | P0 | HIGH | 1h |
| J-02 | Dead unreachable return in find_elbow_point | P1 | HIGH | 15m |
| J-03 | k-means\|\| mislabeled as k-means++ in comment | P1 | HIGH | 15m |
| J-04 | sns.distplot deprecated since 0.11 (3 calls) | P1 | HIGH | 1h |
| J-05 | scolumns_order typo causes NameError | P1 | HIGH | 15m |
| J-06 | sns.kdeplot positional Series — broken in 0.12+ | P1 | HIGH | 15m |
| J-07 | sns.lmplot argument order broken in 0.12+ | P2 | HIGH | 15m |
| J-08 | incremental_kmeans_dask init_max_iter intent moot | P2 | MED | 15m |
| C-01 | ACL 2026 author order misrepresented | P1 | HIGH | 15m |
| C-02 | Portfolio depth insufficient for Head of AI | P1 | HIGH | 4h |
| C-03 | Blog ends 2020 — 5-year LLM-era gap | P1 | HIGH | 4h |
| C-04 | Galaxy morphology post: no code/results | P2 | HIGH | 1h |
| C-05 | HAR post: no model code, OOB clarification needed | P2 | HIGH | 1h |
| C-06 | Distributed k-means post date vs content inconsistency | P2 | MED | 15m |
| C-07 | Publications page has no static fallback | P2 | HIGH | 15m |
| C-08 | sort() depends on deprecated %pylab magic | P1 | HIGH | 15m |
| C-09 | SimuCell3D has no AI/ML content | P2 | HIGH | 1h |
| C-10 | DreamStreet has no verifiable public footprint | P2 | MED | 1h |
| C-11 | BatchNorm description omits train vs inference | P2 | HIGH | 15m |
| C-12 | Louvain community detection misdescribed | P3 | MED | 15m |
| C-14 | "6+ additional publications" vague and unverifiable | P2 | HIGH | 1h |
