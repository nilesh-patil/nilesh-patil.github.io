# AI Leader Site Audit — Nilesh Patil
**Lens:** Would a peer AI practitioner / VP-Engineering / CTO take this person seriously after 60 seconds?
**Peers:** lilianweng.github.io, huyenchip.com, simonwillison.net

## [Severity: P0] — The "Recent Posts" section actively undermines seniority
**Category:** Content Freshness
**Location:** `_pages/home.md`, rendered at `/`
**Evidence:** Home page opens with "Recent posts": Distributed K-Means (May 2020), Galactic Morphology (July 2017), four other 2017 posts. The gap between stated seniority (ACL 2026 paper, Head of AI) and visible thinking output (a 2020 Spark/Dask tutorial) kills the 60-second impression.
**Recommendation:** Either (a) publish two substantive posts on current-work topics, or (b) suppress the "Recent posts" block from the home page until new content exists. The sidebar bio is strong enough to stand alone.
**Reference:** https://huyenchip.com/blog/

## [Severity: P0] — No post exists on any topic from six years of senior work
**Category:** Content Freshness
**Location:** `_posts/` directory
**Recommendation:** Five highest-ROI posts to write, ranked by signal value:
1. *"What compliance-aware AI architecture means in a SEBI-regulated environment"*
2. *"Entity resolution at production scale: lessons from fine-tuning LLMs for name matching"* (companion to the ACL 2026 paper at openreview.net/forum?id=rLisRb1T1Y)
3. *"Evaluating LLM-based persona simulators: metrics we ran, metrics we threw out"*
4. *"Lessons from a feature store for 250M users"*
5. *"Self-hosting SLMs in regulated domains: actual tradeoffs vs. frontier APIs"*
**Reference:** https://lilianweng.github.io/posts/2023-06-23-agent/

## [Severity: P1] — Positioning headline is accurate but not memorable
**Category:** Positioning & Narrative
**Location:** `_pages/home.md`, `_pages/about.md`
**Evidence:** Carry-away line: "AI systems & applied research." Category description, not positioning. No declared stance.
**Recommendation:** Lead with the regulated-domain constraint ("I build AI systems where 'it occasionally hallucinates' is not an acceptable failure mode") or the reliability angle ("Production AI at 250M users taught me that evaluation design matters more than model selection").
**Reference:** https://simonwillison.net

## [Severity: P1] — LinkedIn absent; CV entry is a live placeholder
**Category:** Positioning & Narrative
**Location:** `_pages/cv.md` line 94, `_config.yml` line 248
**Evidence:** CV has `*(see GitHub profile for current link)*` as the LinkedIn entry. `_config.yml` comment: "LinkedIn intentionally omitted until handle is provided."
**Recommendation:** Add LinkedIn URL to `_config.yml`, `_pages/about.md`, and replace CV placeholder. 10 minutes.

## [Severity: P1] — Talks page exists but is hidden; has only a 2099-dated placeholder
**Category:** Visible Thought Leadership
**Location:** `_config.yml:231`, `_talks/2099-01-01-draft-talk.md`
**Evidence:** CV mentions co-leading "Sports x AI sessions at Columbia University." None of this surfaces on the site.
**Recommendation:** Create minimal entries for Columbia sessions and major internal/external talks. Set `show_talks: true`.
**Reference:** https://huyenchip.com

## [Severity: P1] — No AI-specific content on evals, RAG, agentic reliability, model selection
**Category:** Missing Content
**Location:** `_posts/`
**Evidence:** Six posts cover distribution visualization, random forest + sensors, NumPy, taxi graph, galaxy CNN, distributed k-means. None touch any current AI leader topic.
**Recommendation:** Eval-design post and "What I mean by AI harness design" — concrete description of the abstraction layer between an LLM call and a regulated production workflow.
**Reference:** https://huyenchip.com/2023/04/11/llm-engineering.html

## [Severity: P1] — Publications page: biology paper author ordering unexplained for CS/ML audience
**Category:** Publications
**Location:** `_publications/2020-cxcl10-*.md`
**Evidence:** Both 2020 immunology papers list `"et al., <strong>Nilesh Patil</strong>"` — trailing position. CS/ML visitors don't know last-author = senior in biology.
**Recommendation:** One-sentence note in publications header: "In life-sciences papers, last author denotes senior/lead contributor; my role here was computational infrastructure."

## [Severity: P1] — Medium surfaced three times but the linked account may not support seniority
**Category:** Visible Thought Leadership
**Location:** `_pages/about.md`, `_config.yml` footer, sidebar
**Evidence:** Medium (@ensembledme) linked from sidebar, about, footer. If account has no current content, the link is a broken promise.
**Recommendation:** Verify Medium content. If not active, demote or remove. (Cross-ref profile-mgmt audit: this account is currently a food-recipe blog.)

## [Severity: P2] — "Currently exploring" section duplicates About intro verbatim
**Category:** Tone
**Location:** `_pages/about.md` lines 33–34
**Evidence:** "Particularly interested in AI harness design, developer productivity..." appears at intro and again at "Currently exploring."
**Recommendation:** Replace with a forward-looking specific entry: "Evaluating whether SLM-based compliance classifiers can achieve parity with frontier API classifiers on SEBI audit dimensions."

## [Severity: P2] — "Powered by AcademicPages" footer undercuts the positioning
**Category:** Tone
**Recommendation:** Shorten to "Powered by Jekyll" or remove entirely.

## [Severity: P2] — Portfolio has no AI-era projects; OSS signal is a 2018 Docker file
**Category:** Thought Leadership
**Location:** `_portfolio/datascience-environment.md`, `_portfolio/pythonvsrust-kmeans.md`
**Recommendation:** Open-source any internal tooling or build a small useful OSS contribution: a structured output validator for SEBI-domain entities, a compliance-constraint wrapper for LLM tool calls.
**Reference:** https://simonwillison.net — AI credibility rests on shipping/maintaining open tools.

## [Severity: P2] — CV lists "6+ additional team publications" as vague placeholder
**Category:** Publications
**Location:** `_pages/cv.md` line 71
**Recommendation:** Enumerate the additional publications as proper `_publications/` entries, or state the exact count inline with specific conference/journal names.

## Article-level audit

### `2017-01-14-visualizing-and-comparing-distributions.md`
**Deprecated APIs — P0 for code correctness.** `sns.distplot()` was deprecated in seaborn 0.11.0 and **removed in seaborn 0.12.0 (Sept 2022)**. Reader running this code today gets `AttributeError`. `%pylab inline` discouraged in current Jupyter practice. `sns.kdeplot(data_plot.male, color='red')` positional syntax deprecated since seaborn 0.12. Conceptual content (distribution types, when to use each) is correct. Code is the problem.

### `2017-02-15-human-activity-recognition.md`
**Accurate overall; one methodology gap.** UCI-HAR has official subject-stratified split designed to test generalization. Post uses random 70/30 across all 30 subjects — risks same subject in both train and test. Random Forest pipeline, OOB validation, 94.37% test accuracy are credible.

### `2017-03-04-working-with-numpy.md`
**No deprecated APIs; content correct.** `@` operator comment is dated trivia (operator is nine years old). Beginner-reference level appropriate for 2017, but being on the home page in 2026 is positioning misfire.

### `2017-03-14-transportation-graph-nyc-taxi-data.md`
**Content accurate; two dead URLs, three typos, one missing citation.** NYC TLC URL `http://www.nyc.gov/html/tlc/...` dead — current is `https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page`. Missing Blondel et al. (2008) for multilevel community detection. Typos: "straighforward," "wekday," "ahs" ("Penn station ahs multiple entries").

### `2017-07-25-galactic-morphology-using-deep-learning.md`
**The strongest 2017 post.** Dropout (Srivastava 2014) and batch norm (Ioffe & Szegedy 2015) correctly explained and cited. Honesty about normalization as "more of a hack" signals technical maturity. Missing: batch norm inference uses running statistics, not batch statistics — matters in production.

### `2020-05-20-distributed-kmeans-clustering.md`
**Three technical errors in the most production-relevant post.**
- **Error 1 (P1):** `initMode="k-means||"` is labeled "k-means++ initialization" in the comment. k-means|| (Bahmani 2012) is a distributed approximation — not k-means++.
- **Error 2 (P1):** `incremental_kmeans_dask` calls `kmeans.partial_fit(batch)`. `dask_ml.cluster.KMeans` does not implement `partial_fit`. Raises AttributeError. Use `dask_ml.cluster.MiniBatchKMeans`.
- **Error 3 (P2):** Performance comparison implies Dask speedup at 10k–100k samples without showing numbers. At those sizes scikit-learn is faster — Dask's advantage is at sizes that exceed single-machine memory.

## Five highest-impact actions
1. Publish two posts on current work within 30 days (compliance-aware AI harness + ACL 2026 companion).
2. Add LinkedIn to about, CV, and footer.
3. Fix `sns.distplot` → `sns.histplot` and the `partial_fit`/k-means|| errors before another engineer shares these posts.
4. Create real talk entries for Columbia sessions. Flip `show_talks: true`.
5. Rewrite "Currently exploring" to avoid repeating the intro paragraph.
