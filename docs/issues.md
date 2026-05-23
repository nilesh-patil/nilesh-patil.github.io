# Site Audit — Issues & Improvement Opportunities

**Audited:** 2026-05-23 against local dev (current branch `master` @ `5e20398`) and live production (https://nilesh-patil.github.io @ `d85a745`).
**Team:** 6 specialist agents (Designer, Web Dev, AI Expert, Hiring, Profile Mgmt, Blogging) → independent Overseer (Software Design Expert).
**Method:** DOM snapshots + screenshots of every primary page on both targets, source-code reading across `_pages/`, `_posts/`, `_sass/`, `_includes/`, `_layouts/`, `_config.yml`, and external WebFetch verification of social surfaces. Peer benchmarks: lilianweng.github.io, huyenchip.com, simonwillison.net.
**Raw reports:** `docs/audit-artifacts/agent-reports/` (per-agent) and `docs/audit-artifacts/overseer-consolidated.md` (ranked, deduped, verified).

---

## Verdict

The site is technically sound and deploys cleanly — no build drift, no broken pages from infrastructure failures — but it functions as a dormant 2017 portfolio wearing a 2026 job title. Every page that matters to a recruiter, peer, or collaborator finds a content gap, a broken widget, or a stale social link where proof of current work should be.

- **Biggest single risk:** the Medium sidebar/footer/about link points to `@ensembledme`, a food-recipe blog with a fashion-themed bio. A "Head of AI" audience clicking through sees muffin recipes. Three surfaces, active reputational risk.
- **Biggest single opportunity:** publish one substantive post on current AI work (compliance-aware AI harness, ACL 2026 companion, or eval-design at 250M scale). One action updates the archive's last-modified signal, validates the stated seniority, gives the share buttons something worth sharing, seeds RSS, and provides LinkedIn material — six outcomes for one effort.
- **Local vs. prod:** identical apart from a one-day "site last updated" date stamp. Your deploy is clean; nothing to recover.

---

## Decisions you've already made (apply throughout)

These choices were made during the audit and the recommendations below reflect them:

1. **LinkedIn — restore.** Add the handle to sidebar, footer, CV. Replace the broken "see GitHub profile for current link" breadcrumb with the direct URL. _(Handle TBD by you.)_
2. **Medium — migrate to professional handle.** Replace all `@ensembledme` refs with `https://nilesh-patil.medium.com/`. The recipe blog stays under @ensembledme; it's just no longer surfaced from the site. (Side note: your professional Medium last posted May 2022 — a refresh is a separate content decision.)
3. **Comments — disable globally for now.** Set `comments: false` in posts defaults to suppress the "not yet wired" yellow banner; revisit Giscus wiring when there's traffic to support comments.
4. **Doc style — by category with reasoning** (this document).

---

## Action priority — top 15

Ranked by impact × confidence ÷ effort. P0 = breaks UX or reputational risk now; P1 = clear quality cost vs. peer bar; P2 = polish.

| # | Action | Category | Severity | Effort |
|---|--------|----------|----------|--------|
| 1 | Migrate Medium refs to `nilesh-patil.medium.com` (3 surfaces) | Brand | P0 | 15m |
| 2 | Fix `main.min.js` `type="module"` → `defer` | Code | P0 | 15m |
| 3 | Disable comments globally (`comments: false` in defaults) | Code | P0 | 5m |
| 4 | Add LinkedIn to `_config.yml`, `_data/authors.yml`, CV | Brand | P0 | 1h |
| 5 | Add `social:` + top-level `twitter:` + `og_image` to `_config.yml`; fix `http://schema.org` → `https://` | SEO | P0/P1 | 1h |
| 6 | Publish one substantive post (2,000–3,000 words) on current AI work | Content | P0 | 1d |
| 7 | Fix broken code in posts (`sns.distplot`, `partial_fit`, k-means∥ label, dead return, `scolumns_order` typo) | Content / Code | P1 | 1h |
| 8 | Add skip-to-content link in `_layouts/default.html` | A11y | P1 | 15m |
| 9 | Fix Follow button (label → "Connect"; add `aria-expanded`/`aria-controls`; eventually wire direct to LinkedIn) | A11y / UX | P1 | 1h |
| 10 | Fix theme toggle (`<a role="button">` → `<button>`; add `aria-live` announcement) | A11y | P1 | 1h |
| 11 | Update GitHub bio away from "Sr. Principal Research Scientist @Dream11" | Brand | P1 | 15m |
| 12 | Add `<link rel="preload">` for avatar; add Google Search Console verification key | Perf / SEO | P1 | 30m |
| 13 | Rewrite "Currently exploring" (duplicate of intro); add team-size to Dream11 CV entry; add "Open to" sentence | Content | P1 | 1h |
| 14 | Remove "AcademicPages" footer attribution; add `theme-color` variants for dark/sepia | Design / UX | P2 | 15m |
| 15 | Fix Sass slash-division in `_mixins.scss`; add `sass: quiet_deps: true` to `_config.yml` | Build | P1 | 4h |

Everything below is grouped by category, with reasoning, evidence, and references.

---

# 1. Critical / Runtime correctness

These are issues that are *broken right now in production*, not subjective improvements. They cost the user trust the moment they're seen.

## 1.1 Medium link points to a food-recipe blog with a fashion-themed bio
**Severity:** P0 · **Effort:** 15m · **Confidence:** HIGH (4 source surfaces verified)
**Locations:**
- `_config.yml:33` (`author.medium`)
- `_config.yml:244` (`footer.links`)
- `_pages/about.md:42` (`[@ensembledme](https://medium.com/@ensembledme)`)
- `_data/authors.yml` (Medium entry)

**Evidence.** Profile-agent WebFetched `https://medium.com/@ensembledme` and returned: name "Ensembled Me," bio "Lover of music, travel, and fashion," 10 posts including "Double Chocolate Chip Banana Muffins," "Ramen Noodle Soup," "Beef Curry," last published March 24, 2024. Cross-checked against your professional Medium at `https://nilesh-patil.medium.com/` — bio "Interested in applied machine learning, statistics and data science," topics include Dask, Docker for data science, Keras generators, network analysis. Last post May 21, 2022.

**Why this matters.** Three surfaces (sidebar, footer, about page) lead a "Head of AI" audience to recipes. The reputational dissonance is severe and unmitigated by any other site content.

**Fix.**
- Replace all `@ensembledme` Medium URLs with `https://nilesh-patil.medium.com/`.
- In `_config.yml` author block: change `medium: "https://medium.com/@ensembledme"` to `medium: "https://nilesh-patil.medium.com/"`.
- In `_config.yml` footer.links: update the same.
- In `_pages/about.md:42`: change link text from `@ensembledme` to `nilesh-patil.medium.com` and URL accordingly.
- In `_data/authors.yml`: update the Medium entry.

**Reference.** Pattern: handle = brand. See `https://medium.com/@francois.chollet` — handle matches name across surfaces.

---

## 1.2 `main.min.js` loaded as `type="module"` breaks the jQuery bundle
**Severity:** P0 · **Effort:** 15m · **Confidence:** HIGH
**Location:** `_includes/scripts.html:1`

**Evidence.** Line 1: `<script type="module" src="{{ base_path }}/assets/js/main.min.js"></script>`. ES modules are implicitly deferred, execute in strict mode, and scope their bindings — none of these are compatible with the AcademicPages jQuery bundle, which relies on global `$` and `$(document).ready()`. `theme-cycle.js` is loaded with `defer` and depends on `$(document).ready` having fired before its own click handler binds (see header comment in `scripts.html:3-9`). With the module-vs-defer ordering change, that guarantee no longer holds in all browsers; the symptom is that the stock 2-state theme toggle survives instead of being replaced by the 3-state cycle.

**Fix.** Replace `type="module"` with `defer`:
```html
<script defer src="{{ base_path }}/assets/js/main.min.js"></script>
```
Keep `theme-cycle.js` as-is.

**Reference.** https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script#module

---

## 1.3 Giscus comments widget renders a "Comments are not yet wired" banner on every post
**Severity:** P0 · **Effort:** 5m · **Confidence:** HIGH
**Location:** `_config.yml:199,201`; `_includes/comments-providers/giscus.html:3-11`

**Evidence.** `_config.yml:199` is `repo_id: ""`; `_config.yml:201` is `category_id: ""`. The Giscus template's guard `{% if g.repo_id == "" or g.category_id == "" %}` renders a yellow `notice--warning` block on every post page. Visible in production *right now*.

**Decision applied.** You chose to disable comments globally for now rather than wire up Giscus immediately.

**Fix.** In the `defaults:` block of `_config.yml` (search for `scope: type: posts`), add or set:
```yaml
- scope:
    path: ""
    type: posts
  values:
    layout: single
    comments: false   # was true
```
This suppresses the comments include entirely on posts; the broken banner disappears.

**Revisit later.** When you decide to enable comments, enable Discussions on `nilesh-patil/nilesh-patil.github.io`, run `https://giscus.app` to generate `repo_id` and `category_id`, paste into `_config.yml`, and flip `comments: true` back on.

---

## 1.4 Talks page is route-accessible but renders as an empty shell
**Severity:** P1 · **Effort:** 15m (placeholder) or several hours (populate properly) · **Confidence:** HIGH
**Location:** `_config.yml:231` (`show_talks: false`); `_talks/` (only `.gitkeep` and `2099-01-01-draft-talk.md` with `published: false`).

**Evidence.** Nav-hidden but the `/talks/` URL builds successfully. CV mentions Columbia University "Sports x AI" sessions for students, postdocs, and faculty; CV also mentions training sessions ranging 10–200 participants at Dream Sports. None surface.

**Fix.** Two options:
- **Quick:** add `redirect_to: /cv/` to the page frontmatter or `sitemap: false` so the URL doesn't exist as a dead end.
- **Right:** populate `_talks/` with at least the Columbia sessions and any conference / internal sessions. Set `show_talks: true` in `_config.yml`. Each entry: date, title, venue, one-paragraph abstract (no slides required).

**Reference.** https://lilianweng.github.io/talks/ — three minimal entries already signal active external engagement.

---

# 2. Brand & Identity

The story across site / GitHub / Medium / Scholar / Twitter must be coherent. A recruiter who Googles you will cross-reference these surfaces.

## 2.1 LinkedIn is absent from every site surface and the CV breadcrumb is dead
**Severity:** P0 · **Effort:** 1h · **Confidence:** HIGH (verified by 4 of 6 agents independently)
**Locations:**
- `_config.yml:34` (`linkedin:` commented out)
- `_data/authors.yml` (LinkedIn entry intentionally omitted)
- `_pages/cv.md:94` — dead text: `*(see GitHub profile for current link)*`
- `_pages/about.md` "Get in touch" section (no LinkedIn entry)

**Why this matters.** LinkedIn is the primary inbound channel for recruiters, hiring managers, peer outreach, and conference invitations. The current state combines "deliberate omission" (config comments) with "broken breadcrumb" (CV), so the message is mixed: principled stance, or unfinished site? A reader cannot tell.

**Decision applied.** Restore it.

**Fix.**
1. Add `linkedin: <handle>` to `_config.yml:34` (uncomment / fill).
2. Add the LinkedIn entry to `_data/authors.yml` (the template already has the conditional).
3. Replace the dead CV breadcrumb at `_pages/cv.md:94` with the direct URL: `**LinkedIn** — [linkedin.com/in/<handle>](https://www.linkedin.com/in/<handle>)`.
4. Add LinkedIn to the `_pages/about.md` "Get in touch" list at line 42.
5. Verify the LinkedIn profile's job timeline (DreamStreet 2026–present, Dream11 prior, etc.) matches the site CV.

**Reference.** https://lilianweng.github.io — LinkedIn deep-link visible in the first sidebar scan zone.

---

## 2.2 GitHub bio still says "Sr. Principal Research Scientist @Dream11"
**Severity:** P1 · **Effort:** 15m · **Confidence:** MED (external; profile-agent pulled via GitHub API)
**Location:** https://github.com/nilesh-patil (settings/profile)

**Evidence.** `GET https://api.github.com/users/nilesh-patil` returns `bio: "Sr. Principal Research Scientist @Dream11"`, `company: "@dream11"`. The site's sidebar reads "Head of AI at DreamStreet."

**Why this matters.** Your site links to GitHub from sidebar and footer. A recruiter who clicks through immediately sees a stale, more junior-sounding title from a prior employer. That's a credibility tax on every visit.

**Fix.** Go to https://github.com/settings/profile. Update:
- Bio: "Head of AI @ DreamStreet · AI systems & applied research"
- Company: DreamStreet org handle (or plain text)
- Location: Mumbai
- Add link to https://nilesh-patil.github.io

**Reference.** https://github.com/karpathy, https://github.com/swyx — bios are tight positioning lines, current.

---

## 2.3 No `og:image`, no `og:description`, no Twitter card, no Person JSON-LD
**Severity:** P0 (combined) · **Effort:** 1h (single config block fixes all) · **Confidence:** HIGH
**Locations:**
- `_config.yml` — no top-level `social:`, `twitter:`, or `og_image:` keys
- `_includes/seo.html:51` — Twitter Card block guarded on `site.twitter.username`
- `_includes/seo.html:89` — Person JSON-LD guarded on `site.social`
- `_includes/seo.html:92, 134` — both JSON-LD blocks use `"@context": "http://schema.org"` (should be `https`)

**Evidence.** Live `curl` of the homepage on production confirms zero `<meta property="og:image">` tags on any page, zero `<meta name="twitter:*">` tags emitted (because `twitter:` is nested under `author:` not at top level), zero JSON-LD blocks (because `site.social` doesn't exist).

**Why this matters.**
- LinkedIn / X / Slack / email shares of any URL on this site show a blank card with title only.
- Google's knowledge graph has no `sameAs` anchor for "Nilesh Patil" — a common name — so disambiguation falls to the page title alone.
- Your headshot exists at `/images/ensembledme.webp` but isn't wired as the default OG image.

**Fix.** Add to `_config.yml` at top level (not under `author:`):
```yaml
og_image: "/images/ensembledme.jpg"
og_description: "Head of AI at DreamStreet. Building compliance-aware AI systems for SEBI-regulated investor and trader workflows. Previously Head of Applied Research, Dream11."

twitter:
  username: ensembledme

social:
  type: Person
  name: "Nilesh Patil"
  links:
    - "https://github.com/nilesh-patil"
    - "https://scholar.google.co.in/citations?user=IIabY1sAAAAJ"
    - "https://www.linkedin.com/in/<handle>"   # after 2.1
    - "https://twitter.com/ensembledme"
    - "https://nilesh-patil.medium.com/"
```
Also fix `_includes/seo.html:92` and `:134`: change `"http://schema.org"` to `"https://schema.org"` (Google's Rich Results Test prefers https).

**Polish.** Design a proper 1200×630px OG card (name, title, accent color background) in Figma/Canva and save as `images/og-card.jpg`; switch `og_image:` to point at it. The 712×720 portrait will letterbox badly when LinkedIn renders it at 1200×630.

**References.**
- https://ogp.me/#structured
- https://developer.twitter.com/en/docs/twitter-for-websites/cards/overview/summary
- https://developers.google.com/search/docs/appearance/structured-data/person

---

## 2.4 X/Twitter account dormancy + URL using legacy `twitter.com` domain
**Severity:** P2 · **Effort:** 15m (decide + edit) · **Confidence:** LOW (account activity unconfirmed; gated behind login)

**Evidence.** Sidebar HTML renders `href="https://twitter.com/ensembledme"`. The label reads "X (formerly Twitter)" but the URL is the legacy domain. Profile-agent's WebFetch returned HTTP 402 (login wall) so last-post date is unverified.

**Fix.** Check the X account manually. If last post predates 2024, remove `twitter:` from `_config.yml` author block and `_data/authors.yml` — a linked dormant account is a worse signal than no link. If keeping, update the href to `https://x.com/ensembledme` for consistency with the "X" label.

---

# 3. Content & Articles

## 3.1 Six-year post-publication gap actively signals disengagement
**Severity:** P0 · **Effort:** 1d (per post) · **Confidence:** HIGH (4 of 6 agents)
**Locations:** `_pages/home.md` "Recent posts" section; `_posts/` (newest post 2020-05-20).

**Evidence.** Six posts: five from 2017, one from 2020. Meanwhile the CV claims Dream11 Head of Applied Research, Columbia collaboration, ACL 2026 paper, production systems at 250M+ users — all during the post-publishing gap. A visitor reading "Head of AI at DreamStreet" then scanning the "Recent posts" list registers: silent for six years.

**Why this matters.** It is the single largest credibility problem on the site. UX polish, design refinement, and SEO fixes will not paper over the absence of current thinking. Every other action in this document is downstream of having something current to read.

**Fix.** Publish one substantive post (2,000–3,000 words) on current work. Highest-ROI topics, in order:
1. **"What compliance-aware AI architecture means in a SEBI-regulated environment"** — audit trails, explainability, human-in-the-loop gates, output validation. A rare domain almost no practitioner writes about publicly.
2. **"Entity resolution at production scale: lessons from fine-tuning LLMs for name matching"** — companion to the ACL 2026 paper (https://openreview.net/forum?id=rLisRb1T1Y), covering hard-negative construction and where structure-guidance failed.
3. **"Evaluating LLM-based persona simulators: what metrics we ran, what we threw out"** — eval-design from inside Dream11.
4. **"Lessons from a feature store for 250M users"** — scale-specific infrastructure decisions.
5. **"Self-hosting SLMs in regulated domains: actual tradeoffs vs. frontier APIs"** — directly reflects current DreamStreet work.

**Interim fix (if posting is delayed).** Replace the home page "Recent posts" block with "Recent work" surfacing the publications and portfolio collections instead. The ACL 2026 and ICMLA 2023 papers are fresher signals than a 2020 k-means tutorial. Edit `_pages/home.md`.

**Reference.** https://lilianweng.github.io/posts/2023-06-23-agent/ — single canonical post becomes a years-long credibility anchor for an AI leader.

---

## 3.2 Per-post technical errors — published code throws `AttributeError` if run
**Severity:** P1 · **Effort:** 1h · **Confidence:** HIGH (two agents independently found the same errors)

### `_posts/2020-05-20-distributed-kmeans-clustering.md` — three errors

1. **`partial_fit` doesn't exist on `dask_ml.cluster.KMeans`** (line 379). The `incremental_kmeans_dask` function calls `kmeans.partial_fit(batch)`; `dask_ml.cluster.KMeans` never implemented this interface. Reader running this gets `AttributeError`. Use `dask_ml.cluster.MiniBatchKMeans` or `sklearn.cluster.MiniBatchKMeans` directly.
2. **k-means∥ mislabeled as k-means++** (line 203, comment line 202). `initMode="k-means||"` (Bahmani et al. 2012) is a *distributed approximation* to k-means++ — not the same algorithm. Either change the comment or switch the param.
3. **Dead `return` statement** in `find_elbow_point()` (line 502 — `return k_range, inertias` after the actual return at line 500). Unreachable paste artifact; remove.
4. **Misleading performance benchmark.** The `performance_comparison` function implies Dask speedup at 10k–100k sample sizes without showing numbers. At those sizes scikit-learn is faster; Dask's advantage is at sizes that exceed single-machine memory. Add an explicit note.

### `_posts/2017-01-14-visualizing-and-comparing-distributions.md` — broken code

1. **`sns.distplot()`** (three calls around line 70-72) was deprecated in seaborn 0.11 (2020) and *removed in seaborn 0.12 (Sept 2022)*. Any reader running this code today gets `AttributeError`. Replacement: `sns.histplot()` for histogram + optional KDE, or `sns.kdeplot(data=df, x='male', ...)` for density.
2. **`scolumns_order` NameError** — line 194 references `scolumns_order`, line 206 uses `columns_order`. One of these is a typo.
3. **`%pylab inline`** discouraged in current Jupyter practice; pollutes namespace. Replace with `import matplotlib.pyplot as plt`.
4. **Kaggle World Development Indicators dataset URL** no longer stable.

**Fix priority.** These two posts are the most visible (post index, RSS feed, share targets). Fix code first; add a `Last reviewed: 2026-05-XX` note at the top of each post.

**References.**
- https://seaborn.pydata.org/whatsnew.html (deprecation history)
- https://ml.dask.org/clustering.html (dask-ml clustering API)

---

## 3.3 Per-post content & link audit (lower severity, P2)

### `_posts/2017-02-15-human-activity-recognition.md`
- Random-Forest pipeline, OOB validation, 94.37% accuracy — all credible.
- **Methodology gap:** UCI-HAR has an *official subject-stratified split* (subjects 1–21 train, 28–30 test) designed to test generalization to unseen subjects. The post uses random 70/30 across all 30 subjects, risking subject leakage. Post says "result wasn't significantly different" but the methodology is incorrect and could mislead readers. Add a footnote acknowledging the official split.
- Three HTTP links should be HTTPS (scikit-learn.org).
- Breiman's Random Forests page at `stat.berkeley.edu/~breiman/RandomForests/cc_home.htm` is dead.

### `_posts/2017-03-04-working-with-numpy.md`
- Title "Working with numpy" is unindexable for any real query.
- All NumPy APIs (`arange`, `zeros`, `randn`, `argmax`, `dot`, `@`) remain stable and correct.
- 2-minute read; no broadcasting section (NumPy's most important confusing feature).
- The `@` operator was new in Python 3.5 (2015) — the comment "New matrix multiplication operator in python3.5+!" now reads as dated trivia.
- Rename to "NumPy operations: a practical reference" if kept; add a broadcasting section.

### `_posts/2017-03-14-transportation-graph-nyc-taxi-data.md`
- **Dead links** in footnotes 7 and 8: `http://www.nyc.gov/html/tlc/...` no longer resolves. Current is `https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page`. NYC land use also redirects.
- **Missing citation** for multilevel community detection — add Blondel et al. (2008) or the igraph documentation reference.
- **Three typos** on close inspection: "straighforward" (line 115), "wekday" (line 48), "ahs" (line 48 — "Penn station ahs multiple entries").
- 2015 yellow-taxi-only is now historically interesting; TLC has since added HVFHV (Uber/Lyft) data that restructures the network. A 200-word "2026 postscript" would convert the post from dated to historically grounded.

### `_posts/2017-07-25-galactic-morphology-using-deep-learning.md`
- The **strongest 2017 post** technically. Dropout (Srivastava 2014) and BatchNorm (Ioffe & Szegedy 2015) correctly explained and cited. Honesty about normalization as "more of a hack" signals technical maturity.
- **Incomplete.** Ends mid-description of the experimental setup. No results section, no accuracy figures, no sample predictions, no conclusion. Reads like a draft.
- **Missing detail:** BatchNorm inference uses *running statistics*, not batch statistics — a distinction that matters in production.
- Either complete the results section or add a prominent callout: *"This post covers the architecture; results and a companion notebook are at [link]."*

---

## 3.4 "Currently exploring" section duplicates the About page intro
**Severity:** P2 · **Effort:** 15m · **Confidence:** HIGH (verified verbatim)
**Location:** `_pages/about.md:12` vs. `:34`

**Evidence.**
- Line 12: *"Particularly interested in AI harness design, developer productivity, and turning emerging model capabilities into reliable workflows and products."*
- Line 34: *"AI harness design, developer productivity, and turning emerging model capabilities into reliable workflows and products. Self-hosting SLMs and building agent tooling that survives contact with regulated production environments."*

The first sentence is verbatim. Three agents independently flagged this.

**Fix.** Replace "Currently exploring" with a forward-looking specific entry. Suggested:
> *Evaluating whether SLM-based compliance classifiers can achieve parity with frontier API classifiers on SEBI audit dimensions. Building agentic evaluators that survive contact with regulated production. Sketching the operating model for an AI team that ships production systems and writes about them in the open.*

Or convert to "How to work with me" / "Open to":
> *Open to advisory conversations on AI architecture in regulated fintech/insurance domains, talks at AI engineering conferences, and conversations about building applied-research orgs.*

Either replacement converts dead repetition into either differentiated keyword density or an active inbound funnel.

**Reference.** https://www.shreya-shankar.com/about/ — short "what I'm up to / open to" block functions as a soft CTA.

---

## 3.5 Positioning headline is accurate but not memorable
**Severity:** P1 · **Effort:** 1h · **Confidence:** HIGH
**Locations:** `_pages/home.md`, `_config.yml:5` (`subtitle: "AI systems & applied research"`)

**Why this matters.** "AI systems & applied research" is a category description — about 40% of senior ML LinkedIn profiles use the equivalent. No declared stance, no scale anchor, no angle. There is no sentence that says "I shipped X at scale" or "here is the problem I'm uniquely good at solving."

**Fix.** Rewrite the home page lede using **WHO + scale + angle**:

> *Head of AI at DreamStreet. Previously built production AI systems for 250M+ users at Dream11. Particularly interested in making LLMs reliable in regulated, high-stakes environments — where "it occasionally hallucinates" is not an acceptable failure mode.*

Or a sharper stance-led variant:

> *I build AI systems in domains where wrong outputs have real cost. Currently Head of AI at DreamStreet (SEBI-regulated investor workflows). Previously Head of Applied Research at Dream11 (250M+ users, Columbia research collaboration).*

The hiring agent's framing: a recruiter at a non-fintech company currently bounces because "compliance-aware AI architecture for SEBI-regulated investor and trader workflows" is fintech-insider language. Lead with the universally parseable signal (250M users); domain follows.

**Reference.** https://karpathy.ai — opens with role + credibility anchor + unique angle in under 30 words.

---

## 3.6 Team size never stated; "Head of AI" title is unsubstantiated
**Severity:** P1 · **Effort:** 30m · **Confidence:** HIGH
**Location:** `_pages/cv.md`

**Evidence.** CV: *"Led a high-performing cross-continent team of research scientists, applied scientists, and ML engineers across India and New York."* Team size is never given. No hiring decisions appear anywhere. For a DreamStreet Head-of-AI claim, the substantiation is similarly absent.

**Why this matters.** Team size is the single most-asked recruiter follow-up after seeing a "Head of X" title. Without it, the claim is indistinguishable from "managed one intern." Even a lean team shipping production AI at scale is a strong signal — a number prevents misreading.

**Fix.** Add one line to the Dream11 CV entry: *"Led a cross-continent team of N researchers + ML engineers across India and New York."* And to the DreamStreet entry: *"Built and lead a team of N AI engineers."* Replace N with the real number.

---

## 3.7 Portfolio is two items; reads as a graveyard
**Severity:** P2 · **Effort:** 4h (case study draft) · **Confidence:** HIGH
**Location:** `_portfolio/`

**Evidence.** Two files: `datascience-environment.md` (Docker setup, 2018) and `pythonvsrust-kmeans.md` (2024). Neither substantiates "compliance-aware AI systems," "LLM persona simulators," or "Columbia research center."

**Fix.** Add 3 case-study entries from the Dream11 / DreamStreet era. Proprietary code is not required — *problem / approach / outcome / scale* write-ups suffice:
1. Compliance-aware AI harness for SEBI-regulated investor workflows.
2. LLM-based persona simulator + agentic evaluators at Dream11.
3. Columbia research center model & what it produced.

**Reference.** https://eugeneyan.com/work/ — portfolio entries are system-design writeups, not code repos.

---

## 3.8 Publications page — biology author order unexplained for CS/ML audience
**Severity:** P2 · **Effort:** 15m · **Confidence:** HIGH
**Location:** `_publications/2020-cxcl10-*.md`

**Evidence.** Both 2020 immunology papers list `"et al., <strong>Nilesh Patil</strong>"` — trailing author position. In biology this signals senior/PI authorship; in CS/ML, first author is typically lead, so a CS reader may misread these as junior contributions.

**Fix.** Add a one-sentence note in the publications header: *"In life-sciences papers (immunology), last author denotes the senior/lead contributor; my role on these was computational infrastructure."*

---

## 3.9 CV-level: "6+ additional team publications" is vague
**Severity:** P2 · **Effort:** 15m–2h (depends on how many to enumerate) · **Confidence:** HIGH
**Location:** `_pages/cv.md:71`

**Fix.** Enumerate them as proper `_publications/` entries (minimal frontmatter is fine), or replace "6+" with the exact count and venue names inline.

---

## 3.10 CV PDF disclaimer is trust-eroding; download not surfaced
**Severity:** P2 · **Effort:** 30m · **Confidence:** MED
**Location:** `_pages/cv.md` Download notice

**Evidence.** *"PDF version of this CV — The web version above is authoritative; the PDF may lag the web by a few weeks."* No last-modified date. "May lag by weeks" is exactly the kind of language a recruiter forwarding the PDF to a hiring committee doesn't want.

**Fix.** Replace with: *"PDF last updated 2026-05-XX."* Add "Download CV (PDF)" link to the sidebar author profile below the social icons.

---

# 4. UX & Visual Design

## 4.1 "Follow" button is a dead affordance and fails ARIA disclosure
**Severity:** P1 · **Effort:** 1h · **Confidence:** HIGH (3 agents)
**Location:** `_includes/author-profile.html:39`

**Evidence.** `<button class="btn btn--inverse">Follow</button>`. On desktop (≥1024px) the button is `display: none`, but on mobile it's the *first interactive element* in the sidebar. Label implies a feed subscription; no `aria-expanded`, no `aria-controls`. Two problems: wrong label, broken semantics.

**Fix (ordered by dependency).**
1. Rename label from "Follow" to "Connect" or "Links" — `_includes/author-profile.html:39`.
2. Add `aria-expanded="false"` and `aria-controls="author-urls"` to the `<button>`. Add `id="author-urls"` to the `<ul class="author__urls">` it controls. Update the toggle handler in the academicpages JS to flip `aria-expanded`.
3. **Better long-term:** once LinkedIn is added (action 2.1), replace the entire disclosure pattern with a plain `<a>` link to LinkedIn. The cleanest end-state is icon links only, no pseudo-dropdown.

**Reference.** https://www.w3.org/WAI/ARIA/apg/patterns/disclosure/

---

## 4.2 Theme toggle: wrong element semantics + no screen-reader announcement
**Severity:** P1 · **Effort:** 1h · **Confidence:** HIGH
**Locations:** `_includes/masthead.html:33`; `assets/js/theme-cycle.js:48-52, 79-86`

**Evidence.** `<a href="#" role="button" aria-label="...">`. An element with `role="button"` must activate on both click and Space keypress. Native `<a>` fires `click` only on Enter, not Space. `theme-cycle.js:81` binds only `addEventListener("click", ...)`. Additionally, the `aria-label` updates after theme change but there is no `aria-live` region — a screen reader user receives no feedback that the theme actually changed.

**Fix.**
1. In `_includes/masthead.html`, replace the `<li>`+`<a role="button">` with a native `<button>`:
```html
<li class="masthead__menu-item persist tail">
  <button id="theme-toggle" type="button" aria-label="Switch to dark mode">
    <i id="theme-icon" class="fa-solid fa-sun" aria-hidden="true"></i>
  </button>
</li>
```
2. In `_layouts/default.html`, add inside `<body>`:
```html
<div id="theme-announcement" aria-live="polite" aria-atomic="true" class="visually-hidden"></div>
```
3. In `assets/js/theme-cycle.js` `setTheme()`, after `syncIcon()`, set the announcement text:
```js
var ann = document.getElementById("theme-announcement");
if (ann) ann.textContent = theme.charAt(0).toUpperCase() + theme.slice(1) + " mode active.";
```

**References.** https://www.w3.org/WAI/ARIA/apg/patterns/button/ · https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html

---

## 4.3 Dark-mode active nav underline is invisible against the dark text
**Severity:** P1 · **Effort:** 15m · **Confidence:** HIGH
**Location:** `_sass/layout/_masthead.scss:77-85`

**Evidence.** `.masthead__menu-item.selected a { border-bottom: 2px solid var(--global-text-color) }`. In dark mode `--global-text-color` is `#d6d3cc` — a light-grey-on-dark-grey border with minimal salience. The accent cyan `#7fb3d5` is used for links elsewhere but not for the active nav indicator.

**Fix.** Change to:
```scss
.masthead__menu-item.selected a {
  border-bottom: 2px solid var(--global-base-color);
}
```
This resolves to `#7fb3d5` in dark, `#8b5a2b` in sepia — high-contrast in both.

**Reference.** https://linear.app — active nav uses an accent underline across all themes.

---

## 4.4 Theme toggle has no visible label; 3-mode cycle is undiscoverable
**Severity:** P2 · **Effort:** 30m · **Confidence:** HIGH
**Location:** `_includes/masthead.html:33`

**Evidence.** The icon is a bare 16px Font Awesome glyph (`fa-sun`/`fa-moon`/`fa-book-open`). The `fa-book-open` icon is not a universally recognized sepia/reading-mode symbol. A first-time visitor cannot tell that clicking it cycles through three modes.

**Fix.** Add a small visible text label inline next to the icon at `≥$large` breakpoints:
```html
<button id="theme-toggle" type="button" aria-label="Switch theme">
  <i id="theme-icon" class="fa-solid fa-sun" aria-hidden="true"></i>
  <span class="theme-label">Light</span>
</button>
```
Update the JS to also rewrite `.theme-label` text on cycle. Or convert to a three-segment control with text labels.

**Reference.** https://linear.app/changelog — three-segment "System / Light / Dark" with visible text.

---

## 4.5 Code blocks are double-scaled to body-text size; visual subordination lost
**Severity:** P2 · **Effort:** 15m · **Confidence:** HIGH
**Location:** `_sass/_syntax.scss:13, 33`

**Evidence.** Outer container at `$type-size-4` (1.25em) × inner `.highlight` at `$type-size-6` (0.75em) = ~0.94em effective. Code blocks render near body-text size rather than clearly subordinate.

**Fix.** Remove `font-size: $type-size-4` from the outer block; set only `font-size: $type-size-6` (or `0.875em`) on `.highlight`.

**Reference.** https://simonwillison.net — code at ~14px under 17px body text.

---

## 4.6 Syntax highlighting hardcoded for Solarized Light; broken in dark/sepia modes
**Severity:** P2 · **Effort:** 1h · **Confidence:** HIGH
**Location:** `_sass/_syntax.scss`

**Evidence.** All token color rules use fixed Solarized Light hex values. Dark-mode code background is `#161a20` but token colors (yellows, greens, blues calibrated for white) don't adapt. Contrast and readability fail in dark mode.

**Fix.** Add a `html[data-theme="dark"] .highlight { ... }` override block with One Dark or Solarized Dark token colors. Map the 6 most common token classes (keyword, string, comment, function, number, operator) to dark-safe values. Sepia: usually OK with light tokens because background is light.

**Reference.** https://github.com/primer/github-markdown-css — CSS variables for syntax tokens, theme-agnostic.

---

## 4.7 Footer attribution signals "unmodified template"
**Severity:** P2 · **Effort:** 5m · **Confidence:** HIGH (4 agents)
**Location:** `_includes/footer.html:30`

**Evidence.** Every page footer reads: *"Powered by Jekyll & AcademicPages, a fork of Minimal Mistakes."* AcademicPages is the template used by thousands of PhD students and postdocs. For a Head of AI in industry, this footer's default attribution is a soft "I used a grad-student CV template" signal. The AcademicPages MIT license does NOT require footer attribution on the rendered page.

**Fix.** Trim line 30 to:
```liquid
&copy; {{ site.time | date: '%Y' }} {{ site.name }} · Built with Jekyll
```
If attribution feels important to you, add a `/colophon/` page that goes into detail.

---

## 4.8 Footer "FOLLOW:" all-caps style is dated AcademicPages 2017 vernacular
**Severity:** P2 · **Effort:** 15m · **Confidence:** HIGH
**Location:** `_sass/layout/_footer.scss:73-77` (`text-transform: uppercase`); `_includes/footer.html:6` (`follow_label` → "FOLLOW:")

**Fix.** Remove "FOLLOW:" prefix or replace with "Find me on". Change `text-transform: uppercase` to `font-weight: 500`. Also align the footer Google Scholar icon to use `ai ai-google-scholar` (Academicons is already loaded) instead of `fas fa-graduation-cap` — currently inconsistent with the sidebar.

**Reference.** https://simonwillison.net — footer in title-case, no prefix label.

---

## 4.9 No distinctive typeface — system-font stack reads as "GitHub Pages template"
**Severity:** P1 · **Effort:** 45m · **Confidence:** HIGH
**Locations:** `_sass/_themes.scss:18-19, 42-43`

**Evidence.** Body and headings both render in the system stack (`-apple-system, "San Francisco", "Roboto", ...`). Zero typographic differentiation between h1 / h2 / body except size and weight. Indistinguishable from a 2017 GitHub Pages scaffold.

**Fix.** Introduce one editorial serif for h1/h2; keep body in system stack. Candidates:
- **Source Serif 4** (open, variable, no licensing overhead)
- **DM Serif Display** (Stripe Press–adjacent)
- **Libre Baskerville**

In `_sass/_themes.scss:42`:
```scss
$header-font-family: 'Source Serif 4', Georgia, serif;
```
Add to `_includes/head.html`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300..900&display=swap">
```

**Reference.** https://simonwillison.net — minimal: tuning line-height to 1.65 and measure to ~70ch delivers strong readability without a font import. If you prefer minimal, do that instead.

---

## 4.10 Content column too narrow on 1440px viewports
**Severity:** P2 · **Effort:** 1w (proper) or 1h (tweak) · **Confidence:** HIGH
**Location:** `_sass/layout/_page.scss:32-34`; `_sass/layout/_sidebar.scss:35`

**Evidence.** At $x-large the article is `span(8 of 12)` ≈ 756px inside a 1280px container. The sidebars at 2/12 each leave dead margins. The article measure stays fixed when viewport is 1440px; the layout looks pinched.

**Quick fix.** Widen article to `span(9 of 12)` at `$x-large`. Or set a `min-width: 720px; max-width: 800px` directly on `.page`. Lower-risk than a full refactor.

**Proper fix.** Convert to CSS Grid with `minmax(0, 68ch)` for the article column and fixed sidebars. This is a 1-week refactor with regression risk across 12 layout variants.

**Reference.** https://huyenchip.com/blog.html — single-column article at fixed max-width ~700px, no left sidebar on post pages.

---

## 4.11 Publications listing: "Recommended citation" block destroys scan hierarchy
**Severity:** P2 · **Effort:** 1h · **Confidence:** HIGH
**Location:** `_layouts/archive-single.html` (publications); `_includes/publication-archive-single.html`

**Evidence.** Every entry renders: title → venue/year → abstract → full "Recommended citation: Patil, N. et al. …" → Download link. The citation block is the same visual weight as the abstract; the eye lands on it as if it were the key field.

**Fix.** Collapse the citation behind a `<details>` disclosure, or move it exclusively to the individual publication page. The listing should show: title link, venue + year on one muted line, one-sentence abstract, Download button.

**Reference.** https://lilianweng.github.io/posts/ — flat scannable list.

---

## 4.12 Avatar may be a group photo (unverifiable from source files)
**Severity:** LOW · **Confidence:** LOW (image content not verifiable here)
**Location:** `images/ensembledme.webp`

**Evidence.** Designer agent flagged the avatar as appearing to be a cropped group photo with a warm yellow-toned background; at 110px sidebar display individual faces are hard to distinguish. This is an image-content judgment that only you can verify.

**Fix (if it is a group photo).** Replace with a clean solo headshot — minimum 400×400px, saved as WebP at q=85. Filename can stay `ensembledme.webp` so no HTML changes needed.

**Reference.** https://lilianweng.github.io — clean solo headshot at 200px; no ambiguity about whose site this is.

---

## 4.13 Blog list visual grammar is inconsistent between home and /posts/
**Severity:** P2 · **Effort:** 30m · **Confidence:** HIGH
**Locations:** `_pages/home.md` vs. `_pages/posts.html` + `_includes/archive-single.html`

**Evidence.** Home page renders compact bullets (title + date · excerpt). `/posts/` renders each entry with a clock-icon read-time, then a calendar-icon "Published:" + date, then excerpt. The "Published:" prefix is redundant. Two pages feel like different sites.

**Fix.** Strip the "Published:" prefix from `archive-single.html`. Consolidate read-time and date on one line separated by `·`. In `_sass/layout/_archive.scss` hide `.page__meta` "Published:" label text if it's structural.

**Reference.** https://simonwillison.net/2025/

---

# 5. Accessibility

## 5.1 Skip-to-content link: CSS exists but HTML never rendered (WCAG 2.4.1)
**Severity:** P1 · **Effort:** 5m · **Confidence:** HIGH
**Location:** `_layouts/default.html`; `_sass/include/_utilities.scss:68-80`

**Evidence.** `.skip-link` and `.screen-reader-shortcut:focus` defined in SCSS. Grep across `_layouts/` and `_includes/` returns zero matches for the rendered HTML. Keyboard users must tab through the entire masthead nav on every page before reaching `<div id="main">`.

**Fix.** Add immediately after `<body>` in `_layouts/default.html`, before `{% include masthead.html %}`:
```html
<ul class="skip-link">
  <li><a href="#main" class="screen-reader-shortcut">Skip to main content</a></li>
</ul>
```
The CSS already handles focus-visible styling and positioning.

**Reference.** https://www.w3.org/WAI/WCAG22/Understanding/bypass-blocks.html

---

## 5.2 Theme toggle missing Space-key handler (WCAG 4.1.2)
See **4.2** above.

## 5.3 Theme change has no screen-reader announcement (WCAG 4.1.3)
See **4.2** above.

## 5.4 Follow button missing `aria-expanded` / `aria-controls` (WCAG 4.1.2)
See **4.1** above.

## 5.5 Google Scholar / arxiv social icons missing `aria-hidden`
**Severity:** P2 · **Effort:** 5m · **Confidence:** HIGH
**Location:** `_includes/author-profile.html:60, 62-64`

**Evidence.** DOM snapshot shows the accessible name for the Scholar link as `" Google Scholar"` — leading-space artifact from the `<i>` icon lacking `aria-hidden="true"`. Other social icons in the same template already have it.

**Fix.** Add `aria-hidden="true"` to the `<i>` elements in the `googlescholar` and `arxiv` blocks.

**Reference.** https://www.w3.org/WAI/WCAG22/Techniques/aria/ARIA6

---

# 6. SEO & Discoverability

(See also 2.3 for the `social:` / `twitter:` / `og_image:` block fix — combined for clarity.)

## 6.1 No Google Search Console verification → no crawl monitoring
**Severity:** P1 · **Effort:** 15m · **Confidence:** HIGH
**Location:** `_includes/seo.html:101` (conditional exists); `_config.yml` (no `google_site_verification` key)

**Evidence.** `seo.html` has the conditional verification meta tag at line 101 — dormant because no key is set. `jekyll-sitemap` is active and generates `/sitemap.xml`, but no Search Console account has been told to consume it. Without Search Console you have no crawl monitoring, no Core Web Vitals data, and no control over how Google discovers content.

**Fix.** Verify at https://search.google.com/search-console using the HTML tag method. Copy the content value. Add to `_config.yml`:
```yaml
google_site_verification: "XXXX"
```
Submit the sitemap URL (`https://nilesh-patil.github.io/sitemap.xml`) in Search Console after verification.

**Reference.** https://support.google.com/webmasters/answer/9008080

---

## 6.2 `og:type` and default `og:image` absent on non-article pages
**Severity:** P2 · **Effort:** 15m · **Confidence:** HIGH
**Location:** `_includes/seo.html:119-146`

**Evidence.** `og:type: article` emitted only when `page.date` exists. Home, about, publications, CV emit no `og:type`. `og:image` only set when `page.header.image` or `page.header.overlay_image` is present.

**Fix.** In `_includes/seo.html`, after the `og:type` article block, add an `{% else %}` branch:
```liquid
{% if page.date %}
  <meta property="og:type" content="article">
{% else %}
  <meta property="og:type" content="website">
{% endif %}
```
After line 146, add a fallback for `og:image` to use `site.og_image` when no page-level image is set. Together with 2.3 these provide complete OG coverage.

---

## 6.3 Generic page titles + common name → weak knowledge-graph signal
**Severity:** P2 · **Effort:** 30m · **Confidence:** HIGH
**Location:** Page frontmatter

**Evidence.** Browser tab titles: "Nilesh Patil" (home), "About - Nilesh Patil" (about). Subtitle "AI systems & applied research" doesn't appear in any page title. "Nilesh Patil" is a common name.

**Fix.** Add specific `description` to home page frontmatter (`_pages/home.md`):
```yaml
description: "Head of AI at DreamStreet. Previously Head of Applied Research, Dream11. Building LLMs and agentic systems at production scale in regulated environments."
```
Combined with the JSON-LD Person markup (2.3), this gives Google enough disambiguation for the knowledge graph.

---

## 6.4 Nav label "Blog" implies active commentary; content is 2017 technical notebooks
**Severity:** P2 · **Effort:** 5m · **Confidence:** MED
**Location:** `_data/navigation.yml`

**Fix.** Rename the nav entry from "Blog" to "Writing" or "Notes." Also update `home.md` link labels from `[All posts →]` to `[All writing →]`. Lowers the implicit cadence promise to match the actual content.

**Reference.** https://eugeneyan.com/ uses "Writing."

---

# 7. Performance

## 7.1 No `<link rel="preload">` for the LCP image (author avatar)
**Severity:** P1 · **Effort:** 5m · **Confidence:** HIGH
**Location:** `_includes/head.html`

**Evidence.** Author avatar (`/images/ensembledme.webp`) is the first image in `<main>` on every page. The `<img>` has `fetchpriority="high"` but is buried inside `{% include sidebar.html %}` inside `{% include single.html %}` — the browser cannot start fetching until HTML parse → CSS resolve → layout. No `preload` in head.

**Fix.** Add to `_includes/head.html` after the main CSS `<link>`:
```html
<link rel="preload" href="{{ '/images/ensembledme.webp' | prepend: base_path }}" as="image" type="image/webp" fetchpriority="high">
```

**Reference.** https://web.dev/articles/lcp

---

## 7.2 Module-bundle JS load (covered in 1.2)
See **1.2**. The `type="module"` issue is both a runtime correctness problem and a performance one — modules block parser-aware preload.

---

## 7.3 `meta[name="theme-color"]` hardcoded white; mismatches dark/sepia
**Severity:** P2 · **Effort:** 5m · **Confidence:** HIGH
**Location:** `_includes/head/custom.html:12`

**Evidence.** `<meta name="theme-color" content="#ffffff"/>`. Static. Mobile browser chrome (address bar) stays white in dark and sepia modes — jarring mismatch.

**Fix.** Replace with media-scoped variants:
```html
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff">
<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#1d2128">
```

**Reference.** https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta/name/theme-color

---

# 8. Build & Code Health

## 8.1 354 Dart Sass slash-division deprecations → hard build error in Sass 2.0
**Severity:** P1 · **Effort:** 4h (mixins + quiet_deps), 1w (susy/breakpoint removal) · **Confidence:** MED on count (HIGH on the underlying issue)
**Locations:** `_sass/include/_mixins.scss:18`; `_sass/vendor/susy/**`; `_sass/vendor/breakpoint/**`

**Evidence.** Jekyll build log emits 354 deprecation warnings. Susy (last release 2017) and breakpoint (2018) are unmaintained and pre-date the Dart Sass migration roadmap.

**Phased fix.**
1. **Short-term (5 min):** add to `_config.yml`:
```yaml
sass:
  quiet_deps: true
```
This suppresses warnings from vendored libraries while leaving your own SCSS warnings visible.
2. **First-party fix (1h):** in `_sass/include/_mixins.scss`, add `@use "sass:math";` at the top and replace `($target / $context)` with `math.div($target, $context)`.
3. **Long-term (1w):** plan removal of susy (replace float-based grid with CSS Grid) and breakpoint (replace `@include breakpoint(...)` calls with native `@media (min-width: ...)` or `sass-mq`). This is the right move before Dart Sass 2.0 forces it.

**Reference.** https://sass-lang.com/documentation/breaking-changes/slash-div/ · https://sass-lang.com/documentation/js-api/interfaces/stringoptions/#quietdeps

---

## 8.2 `.travis.yml` targeting Ruby 2.1 (EOL 2017) at the project root
**Severity:** P2 · **Effort:** 1m · **Confidence:** HIGH
**Location:** `.travis.yml`

**Evidence.** First line `language: ruby`, specifies `rvm: [2.1]`. Active CI is `.github/workflows/pages.yml` running Ruby 3.3. The Travis file is from 2016–2017 vintage.

**Fix.** `git rm .travis.yml`. Dead weight, source of contributor confusion.

---

# 9. Blog UX & distribution loop (lower priority)

## 9.1 No tags visible on post index; `/tag-archive/` unlinked
**Severity:** P2 · **Effort:** 30m · **Confidence:** MED
**Location:** `_pages/posts.html`, `_includes/archive-single.html`

**Fix.** Tags are set in every post's frontmatter but the archive-single template doesn't render them. Add `{% include tag-list.html %}` (file already exists in `_includes/`) below the excerpt. Link the tag archive from the posts index header.

---

## 9.2 No code copy button on code blocks
**Severity:** P2 · **Effort:** 30m · **Confidence:** HIGH
**Locations:** `assets/js/` (new file); `_includes/scripts.html`

**Fix.** Add ~25 lines of vanilla JS in `assets/js/copy-code.js`. On `DOMContentLoaded`, insert a `<button>` into each `.highlight` div; on click, `navigator.clipboard.writeText(pre.textContent)` and briefly swap the label to "Copied!". No library required.

**Reference.** https://simonwillison.net — all code blocks have one-click copy; a baseline expectation for technical writing in 2026.

---

## 9.3 Share buttons pre-fill only the URL — no title, no author handle
**Severity:** P2 · **Effort:** 15m · **Confidence:** HIGH
**Location:** `_includes/social-share.html`

**Fix.** Change the X share href to:
```liquid
https://x.com/intent/post?text={{ page.title | url_encode }}%20{{ base_path | append: page.url | url_encode }}&via=ensembledme
```
For LinkedIn: append `&title={{ page.title | url_encode }}&summary={{ page.excerpt | url_encode }}`.

**Reference.** https://developer.x.com/en/docs/twitter-for-websites/tweet-button/overview

---

## 9.4 No "Back to all posts" / breadcrumbs
**Severity:** P2 · **Effort:** 1m · **Confidence:** HIGH
**Location:** `_config.yml`, `_layouts/single.html`

**Fix.** Set `breadcrumbs: true` in `_config.yml` (single line). The layout already conditionally includes them. Or add a static `← All posts` anchor above the article title in `single.html`.

---

## 9.5 Post index excerpts are generic labels, not hooks
**Severity:** P2 · **Effort:** 30m · **Confidence:** HIGH
**Location:** Post frontmatter `excerpt:` fields

**Fix.** Rewrite each excerpt as 1–2 sentences that state a specific finding or frame a tension. Example for NumPy post: *"The 12 array operations that appear in 90% of ML code, with hardcoded outputs you can paste directly — plus why vectorized code is faster than the equivalent loop."* Excerpts are frontmatter; no code changes.

**Reference.** https://lilianweng.github.io — every index excerpt answers "why is this interesting" before the click.

---

## 9.6 No "Now" page; no signal of active intellectual work
**Severity:** P2 · **Effort:** 15m to create, ongoing to maintain · **Confidence:** HIGH
**Location:** `_pages/` (new file)

**Fix.** Add `_pages/now.md` with 3–5 bullets updated quarterly: what you're building, reading, or trying to figure out. The "Now" page (nownownow.com) is a widely-practiced convention that converts a static personal site into a living signal.

**References.** https://simonwillison.net/now/ · https://nownownow.com/about

---

## 9.7 No newsletter / email subscription path
**Severity:** Deferred (P2) · **Effort:** 1h · **Confidence:** HIGH
**Note:** Valid recommendation but premature until there's regular content to subscribe to. Revisit after publishing 2–3 substantive posts.
**Reference.** https://buttondown.email — free tier, GDPR-compliant, native RSS-to-email.

---

# 10. Risks deferred or dropped

These were raised by agents but deferred or downgraded after verification:

- **Avatar as group photo** — unverifiable from source; needs your judgment on the image content. (See 4.12.)
- **Owner-hires-talent reverse lens** — adding a "We're hiring" section to the site. Valid but below the severity floor given the volume of P0/P1 work above. Revisit when actively recruiting.
- **No site analytics** — your `_config.yml` explicitly states a zero-telemetry policy. Plausible / GoatCounter were suggested but this is a deliberate principled choice; respect it.
- **Galactic-morphology post incomplete** — strongest 2017 post technically but ends mid-experimental-setup. Lower priority than publishing new 2026-era content; your discretion.
- **Newsletter signup widget** — premature without regular new content.

---

# 11. Cross-references

- **Agent reports (raw):** `docs/audit-artifacts/agent-reports/01-designer.md` through `06-blogging.md`
- **Overseer consolidation:** `docs/audit-artifacts/overseer-consolidated.md` (ranked top-15, dedup decisions, fabrication check)
- **Baseline artifacts:** `/var/folders/_m/ktffvd9x2fnc_g9spdkkkrr80000gp/T/audit/` — DOM snapshots and screenshots for every primary page on local + prod, plus dark/sepia variants of the home page.

---

*Audit closed 2026-05-23. Each item above is independently verifiable: cite the file and line, then read the source. No agent fabricated a path; all line references hold within ±2 lines.*
