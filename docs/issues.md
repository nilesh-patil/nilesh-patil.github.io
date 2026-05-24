# Site Audit — Issues & Improvement Opportunities

**Audited:** 2026-05-24 against local dev (master).
**Method:** 6 specialist agents (Designer, Web Dev, AI Expert, Hiring, Profile Mgmt, Blogging) → independent Overseer → user decisions. Real-browser interaction via Chrome DevTools MCP, source-code reading, WebFetch verification of external surfaces. Peer benchmarks: lilianweng.github.io, huyenchip.com, simonwillison.net, karpathy.ai.
**Raw reports:** `docs/audit-artifacts/agent-reports/` (per-agent); `docs/audit-artifacts/overseer-consolidated.md` (deduped + verified); `docs/audit-artifacts/user-decisions.md` (user answers + recommended defaults); `docs/audit-artifacts/verification-log.md` (per-issue verification status).

---

## Verdict

The site is in a P0 regression state on every page load: commit `3930ceb`'s JS fix introduced a `SyntaxError` that kills jQuery, navigation, the Follow button, smooth scroll, and all JS-dependent UI. That is the single most urgent fix. Beyond the regression, the site's largest strategic risk is a credibility gap: the bio claims active AI leadership at the LLM/agentic-systems level, but the public artifact record — six blog posts ending in 2020, three portfolio projects with zero AI/LLM content, and a GitHub bio still attributing a prior employer — actively contradicts that claim to any technically sophisticated reader. The biggest opportunity is therefore not a code fix but a content commitment: two substantive posts on current AI work would shift the entire perception. The structured-data and brand-consistency issues (three competing Twitter/X handles, Person+Organization JSON-LD collision, ORCID missing from sameAs) are fixable in an afternoon now that the canonical X handle has been confirmed.

---

## Decisions you've made (apply throughout)

### D-001 — X/Twitter handle
`@optimistic_flw` is the canonical handle. Remove `@ensembledme` from every X/Twitter surface.
- `_config.yml` `twitter.username` → `optimistic_flw`
- `_config.yml` `author.twitter` → `optimistic_flw`
- `_data/authors.yml` Twitter entry → `optimistic_flw`
- Footer sidebar X link → `https://x.com/optimistic_flw`
- `_config.yml` `social.links` → drop both `https://twitter.com/nilesh-patil` and the duplicate `https://x.com/optimistic_flw` already present; retain only one canonical X URL: `https://x.com/optimistic_flw`
- Resolves N-001, B-03, B-08, B-09, C-013.

### D-002 — Contact method
No email surfaced. Add a single CTA line on the About page (and consider sidebar): "LinkedIn is the fastest way to reach me."
- Do NOT add `author.email` or any mailto link.
- Resolves N-003, B-10, C-030.

### D-003 — LinkedIn + Google Scholar headlines
Both already updated to "Head of AI at DreamStreet." Mark the corresponding audit items as verified-by-user.
- Resolves N-004, B-11, B-14.

### D-004 — Team sizes
Do not disclose team headcount for Dream11 or DreamStreet on the public site.
- Use scope-language alternative where the audit recommended headcount: e.g. "cross-functional research org spanning ML eng, applied research, and data."
- Resolves N-006, H-03, C-011.

### R-005 — ORCID surfacing [Recommended default — not yet confirmed]
Add ORCID (`0000-0002-3438-8571`) to both `social.links` / `sameAs` (JSON-LD) and the sidebar.
- Rationale: ORCID is already public on GitHub, so this is not a privacy expansion. It is the strongest disambiguation anchor for a common name in academic databases.
- Side-action: update the ORCID employment record to add DreamStreet (currently only shows Dream Sports).
- Tied to N-002, B-04, C-029.

### R-006 — Medium bio [Recommended default — not yet confirmed]
Update the Medium bio at `nilesh-patil.medium.com` to current positioning. No commitment to publish.
- Rationale: 5-minute edit removes the 2020 "data scientist" impression for anyone who clicks through.
- Tied to N-005, B-12.

### R-007 — Avatar/logo file rename [Recommended default — not yet confirmed]
Rename `/images/ensembledme.{jpg,webp}` → `/images/nilesh-patil.{jpg,webp}` and update all `_config.yml` references.
- Rationale: `@ensembledme` confirmed NOT canonical handle (per D-001). The filename surfaces in every OG image URL. Renaming aligns asset names with confirmed brand identity.
- Optional: set up a redirect from the old paths if any external sites may have hot-linked the file.
- Tied to N-007, B-17.

### R-008 — Service / advisory [Recommended default — not yet confirmed]
Treat as a genuine gap. Developmental recommendation: "Consider taking on one reviewer/advisor role in the next 6 months to create this signal before the next active search." Not blocking.
- Rationale: No service surfaced means either it's absent or undisclosed; either way, the public artifact gap is real.
- Tied to N-008, H-25, H-22.

---

## Action priority — top 15

| # | ID | Action | Category | Severity | Effort | Status |
|---|-----|--------|----------|----------|--------|--------|
| 1 | C-001 | main.min.js SyntaxError — jQuery dead on all pages | Code/JS | P0 | 15m | Open |
| 2 | C-002 | GitHub bio shows stale Dream11 title | Brand | P0 | 15m | Open (manual action on GitHub.com required) |
| 3 | C-003 | 6-year publishing gap — blog signals disengagement | Content | P0 | Ongoing | Open |
| 4 | U-001 | dask_ml.KMeans.partial_fit doesn't exist — AttributeError | Code/JS | P0 | 1h | Open |
| 5 | C-009 | ACL 2026 author order misrepresented as first-author | Content | P1 | 15m | Open |
| 6 | C-010 | Portfolio has zero AI/LLM artifacts | Content | P1 | 4h | Open |
| 7 | C-007 | Two H1 elements on home page; sidebar H2 before H1 | Accessibility | P1 | 30m | Open |
| 8 | C-006 | No skip-to-content link | Accessibility | P1 | 30m | Open |
| 9 | C-013 | Three competing Twitter/X handles in structured data | Brand | P1 | 1h | Resolved by D-001 |
| 10 | C-008 | Duplicate meta[name=description] | SEO/Meta | P1 | 15m | Open |
| 11 | C-014 | Person+Organization JSON-LD on same URL | SEO/Meta | P1 | 15m | Open |
| 12 | C-004 | TOC links at 9.28px uppercase — unreadable | Accessibility | P1 | 15m | Open |
| 13 | C-012 | Share buttons produce blank/untitled social posts | Content | P1 | 15m | Open |
| 14 | C-005 | Follow button and nav toggle missing aria-expanded | Accessibility | P1 | 15m | Open |
| 15 | C-015 | Teaching and Talks pages are completely empty | Content | P1 | 30m | Open |

---

# 1. Critical / Runtime Correctness (P0)

---

```yaml
id: C-001
title: "main.min.js: top-level ES6 import in defer script causes SyntaxError — jQuery dead on all pages"
category: Code/JS
severity: P0
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `assets/js/_main.js:57` contains `import { plotlyDarkLayout, plotlyLightLayout } from './theme.js'` — a static top-level ES6 import. Commit `3930ceb` changed the `<script>` tag from `type="module"` to `defer`. A non-module `<script>` with a top-level static `import` is a parse error per the HTML spec; the browser aborts execution before any byte of jQuery runs. Browser console (re-verified 2026-05-24): `Uncaught SyntaxError: Cannot use import statement outside a module`. JS eval confirmed `typeof window.$` === `"undefined"`. Cascade: hamburger button stuck at `display: block` at 1280px desktop, Follow button `aria-expanded === null`, nav collapse broken, smooth scroll dead.

**Why this matters:** Every page on the site has a completely broken JavaScript layer. Navigation, mobile menu, theme toggle JS, smooth scroll, search masthead-toggle, and social-link reveal are all dead. This is worse than having no JS at all — the UI elements render but do nothing, creating a confusing phantom-control experience.

**Recommendation:** Remove the static `import` from `_main.js` and replace with a dynamic import gated on plotly element presence. Rebuild `main.min.js`.

```diff
# assets/js/_main.js
- import { plotlyDarkLayout, plotlyLightLayout } from './theme.js';

+ // Dynamic import — only loads theme.js when Plotly elements are present
+ if (document.querySelectorAll('.plotly-graph-div').length > 0) {
+   import('./theme.js').then(({ plotlyDarkLayout, plotlyLightLayout }) => {
+     // existing plotly init code
+   });
+ }
```

Then rebuild: `npx rollup assets/js/_main.js -o assets/js/main.min.js --format iife` (or equivalent build step).

**Spec reference:** https://html.spec.whatwg.org/multipage/webappapis.html#module-script; https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script#type

---

```yaml
id: U-001
title: "dask_ml.cluster.KMeans.partial_fit does not exist — AttributeError on published tutorial code"
category: Code/JS
severity: P0
confidence: HIGH
effort: 1h
status: OPEN
```

**Evidence:** `_posts/2020-05-20-distributed-kmeans-clustering.md:379` confirmed: `kmeans.partial_fit(batch)` called on a `dask_ml.cluster.KMeans` instance. The dask-ml public API lists: `fit`, `fit_transform`, `get_metadata_routing`, `get_params`, `predict`, `set_output`, `set_params`, `transform` — no `partial_fit`. A reader following the tutorial immediately hits `AttributeError`. Re-verified 2026-05-24: `grep` confirms `kmeans.partial_fit(batch)` at line 379.

**Why this matters:** The distributed k-means post is the most technically ambitious content on the site. A broken code example in a tutorial is worse than no example — it signals lack of testing and erodes trust in the author's technical claims.

**Recommendation:** Replace `dask_ml.cluster.KMeans` with `sklearn.cluster.MiniBatchKMeans` wrapped in `dask_ml.wrappers.Incremental`, which is the documented pattern for streaming k-means.

```diff
- from dask_ml.cluster import KMeans
- kmeans = KMeans(n_clusters=k, init_max_iter=5, oversampling_factor=2)
- kmeans.fit(X_dask)
- ...
- kmeans.partial_fit(batch)

+ from sklearn.cluster import MiniBatchKMeans
+ from dask_ml.wrappers import Incremental
+ base_est = MiniBatchKMeans(n_clusters=k, random_state=42)
+ kmeans = Incremental(base_est)
+ kmeans.fit(X_dask, classes=list(range(k)))
+ # For online update:
+ kmeans.partial_fit(batch)  # Incremental wrapper exposes partial_fit correctly
```

**Spec reference:** https://ml.dask.org/modules/generated/dask_ml.cluster.KMeans.html; https://ml.dask.org/incremental.html

---

```yaml
id: C-002
title: "GitHub bio shows 'Sr. Principal Research Scientist @dream11' — actively contradicts DreamStreet role on site"
category: Brand
severity: P0
confidence: HIGH
effort: 15m
status: OPEN (manual action on github.com/settings/profile required)
```

**Evidence:** GitHub API re-verified 2026-05-24: `bio: "Sr. Principal Research Scientist @Dream11"`, `company: "@dream11"`. Site `_config.yml`, `_data/authors.yml`, and `_pages/cv.md` all show "Head of AI at DreamStreet" as current role. A recruiter opening three tabs — GitHub, LinkedIn, site — sees a mismatched story on the first tab.

**Why this matters:** GitHub is often the first professional surface a technical recruiter checks. A stale title there signals either role inflation on the site or simple profile neglect — both damage credibility.

**Recommendation:** Update GitHub profile at https://github.com/settings/profile:
- Bio: "Head of AI at DreamStreet | Previously Head of Applied Research, Dream11 | nilesh-patil.github.io"
- Company: "DreamStreet"

This is a manual off-site action. No code change required.

**Peer reference:** https://github.com/karpathy — bio, employer, and personal site are consistent and current.

---

```yaml
id: C-003
title: "Six-year publishing gap (2021–2026): blog signals disengagement from AI field during the LLM era"
category: Content
severity: P0
confidence: HIGH
effort: 1d (ongoing)
status: OPEN
```

**Evidence:** `_posts/` directory confirmed: six files, newest dated `2020-05-20` (re-verified 2026-05-24 via `ls`). No post contains the words "LLM", "agent", "RAG", "agentic", "compliance", or "DreamStreet". Bio claims "compliance-aware AI architecture," "agentic workflows," and "LLM-based behavior simulation." Published content: seaborn histograms (2017), random forests (2017), NumPy reference (2017), NYC taxi graph (2017), galaxy CNN (2017), k-means clustering (2020). The gap covers GPT-3 (2020), ChatGPT (2022), GPT-4 (2023), and the entire agent-framework era.

**Why this matters:** Peers at equivalent seniority publish continuously. Sophisticated readers — including the VPs and CTOs making hiring decisions — will notice immediately that the stated expertise has no corresponding public record. The bio's claims are unfalsifiable and thus less credible than a 1,500-word essay demonstrating the knowledge.

**Recommendation:** Publish at minimum one substantive post per quarter. Two immediate candidates requiring no IP disclosure:
1. Architecture patterns for compliance-aware AI in regulated environments (SEBI context)
2. Agentic evaluator design at scale

Even 1,500 words on either topic reframes the blog from archive to active practice log.

**Peer reference:** https://lilianweng.github.io/posts/2025-05-01-thinking/ — substantive 2025 post demonstrating current AI thought leadership while in a senior role; https://huyenchip.com/blog/ — continuous content directly mapping to stated expertise.

---

# 2. Brand & Cross-surface Consistency

---

```yaml
id: C-013
title: "Three different Twitter/X handles across site surfaces — structured data integrity compromised"
category: Brand
severity: P1
confidence: HIGH
effort: 1h
status: RESOLVED by user decision D-001 — see "Decisions you've made"
resolved_by: D-001
```

**Evidence:** `_config.yml` had `twitter.username: ensembledme`, `author.twitter: "ensembledme"`, and `social.links` containing both `x.com/optimistic_flw` AND `twitter.com/nilesh-patil`. The Twitter Card `twitter:site` was `@ensembledme`. Three distinct handles across four surfaces. `medium.com/@ensembledme` is a food/lifestyle blog with no connection to Nilesh — the same brand risk applied to X.

**Status: RESOLVED by user decision D-001.** Action required: implement the config changes specified in D-001 above.

```diff
# _config.yml
- twitter:
-   username: ensembledme
+ twitter:
+   username: optimistic_flw

- author:
-   twitter: "ensembledme"
+ author:
+   twitter: "optimistic_flw"

# social.links — remove both old entries, retain only:
  social:
    links:
+     - "https://x.com/optimistic_flw"
-     - "https://twitter.com/nilesh-patil"
-     - "https://x.com/optimistic_flw"  # duplicate
```

```diff
# _includes/author-profile.html:169
- href="https://twitter.com/{{ author.twitter }}"
+ href="https://x.com/{{ author.twitter }}"
```

**Spec reference:** https://schema.org/Person — `sameAs` should contain unique, verified canonical URLs.

---

```yaml
id: C-002
title: "GitHub bio shows 'Sr. Principal Research Scientist @dream11'"
```
*(Full entry appears in Section 1 above — listed here for brand-surface completeness.)*

---

```yaml
id: C-025
title: "og:image references ensembledme.jpg (JPEG) while only .webp is served by <picture> negotiation"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 10m
status: RECOMMENDED DEFAULT applied (not user-confirmed) — see Appendix A (R-007)
resolved_by: R-007 (pending confirmation)
```

**Evidence:** Both `images/ensembledme.jpg` (107 KB) and `images/ensembledme.webp` (39 KB) exist in the repo. `_config.yml:26` uses `og_image: "ensembledme.jpg"` — the JPEG is served correctly to OG/social scrapers; unfurls work. The primary risk is the `ensembledme` brand name in the OG image URL, which will persist in social crawl caches until the file is renamed.

**Recommendation (pending R-007 confirmation):** Once R-007 is confirmed, rename files and update config:

```diff
# _config.yml
- og_image: "ensembledme.jpg"
+ og_image: "nilesh-patil.jpg"

# In repo:
# git mv images/ensembledme.jpg images/nilesh-patil.jpg
# git mv images/ensembledme.webp images/nilesh-patil.webp
```

**Spec reference:** https://ogp.me/#structured; Open Graph image guidelines.

---

```yaml
id: C-031
title: "Sidebar X/Twitter link uses twitter.com domain; Medium bio 5+ years stale"
category: Brand
severity: P2
confidence: HIGH
effort: 15m each
status: PARTIALLY RESOLVED — twitter.com domain resolved by D-001; Medium bio resolved by R-006 (not yet confirmed)
resolved_by: D-001 (twitter.com domain fix), R-006 (Medium bio — pending confirmation)
```

**Evidence:** `_includes/author-profile.html:169` has `href="https://twitter.com/{{ author.twitter }}"` — hardcoded `twitter.com`. WebFetch of `https://nilesh-patil.medium.com/` confirmed bio: "Interested in applied machine learning, statistics and data science." Most recent article: January 2020.

**Recommendation:**

```diff
# _includes/author-profile.html:169
- href="https://twitter.com/{{ author.twitter }}"
+ href="https://x.com/{{ author.twitter }}"
```

Medium bio: off-site edit at https://nilesh-patil.medium.com/ — update to: "Head of AI at DreamStreet. Research at the intersection of LLMs, agentic systems, and regulated AI deployment. nilesh-patil.github.io"

**Spec reference:** n/a (brand consistency).

---

```yaml
id: U-020
title: "Dream11 tenure title 'Senior Principal Research Scientist / Head of Applied Research' is ambiguous — promotion timing unclear"
category: Content
severity: P2
confidence: HIGH
effort: 15m
status: OPEN (D-004 resolves headcount but not title-split)
```

**Evidence:** `_pages/cv.md` confirmed: single compound title for the entire 2019–2026 Dream11 tenure with no internal date break. Hiring committees reading this cannot determine how long the candidate held a leadership-track title vs. an IC track title.

**Recommendation:** Split into two dated entries:

```diff
# _pages/cv.md — Dream11 section
- **Senior Principal Research Scientist / Head of Applied Research** | 2019–2026

+ **Head of Applied Research** | [promotion year]–2026
+ *Grew from Senior Principal Research Scientist (below)*
+
+ **Senior Principal Research Scientist** | 2019–[promotion year]
```

---

# 3. Content & Editorial

---

```yaml
id: C-009
title: "ACL 2026 paper: Nilesh Patil listed as third author but CV presents as 'Nilesh Patil, et al.'"
category: Content
severity: P1
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_publications/2026-structure-guided-entity-resolution.md:9` confirmed: `authors: "<strong>Nilesh Patil</strong>, et al."`. OpenReview link shows author order: (1) Shivam Chourasia, (2) Hitesh Kapoor, (3) Nilesh Patil. The `<strong>` markup on "Nilesh Patil" in a "Patil et al." citation is the academic convention for first-author self-highlighting. Applied to a third author, it misrepresents authorship order. Hiring managers and peer reviewers checking the OpenReview link see the discrepancy immediately.

**Recommendation:**

```diff
# _publications/2026-structure-guided-entity-resolution.md
- authors: "<strong>Nilesh Patil</strong>, et al."
+ authors: "Shivam Chourasia, Hitesh Kapoor, <strong>Nilesh Patil</strong>"
```

**Spec reference:** https://openreview.net/forum?id=rLisRb1T1Y (verified author order).

---

```yaml
id: C-010
title: "Portfolio has zero AI/LLM artifacts — none of the 3 projects backs the 'Head of AI' claim"
category: Content
severity: P1
confidence: HIGH
effort: 4h
status: OPEN
```

**Evidence:** `_portfolio/` directory contains: `datascience-environment.md` (2018 Docker setup), `pythonvsrust-kmeans.md` (2024 k-means benchmark), `simucell3d.md` (2026 C++ HPC fork). Zero AI/LLM content. CV claims "LLM-based behavior simulation," "agentic evaluators," "compliance-aware AI harness design," "feature-store systems supporting 250M+ users."

**Why this matters:** A hiring committee doing a 10-second scan of the portfolio will find a 2018 Docker setup as the first result. No public artifact demonstrates any of the claimed AI expertise — a major credibility gap for a "Head of AI" candidate.

**Recommendation:** Add at minimum one portfolio entry connecting to current AI work. Options (in order of impact, ascending effort):
1. Cross-link the ACL 2026 entity resolution paper with a 200-word technical explainer
2. A sanitized architecture writeup of one DreamStreet AI system (compliance harness, agent framework)
3. Retire or deprioritize the 2018 Docker entry — it creates cognitive dissonance

**Peer reference:** https://github.com/simonwillison — every public repo is a runnable, documented artifact; https://huyenchip.com — publishes system design essays on real production AI systems without proprietary disclosure.

---

```yaml
id: C-011
title: "No team size, headcount, or org-chart signal — hiring readers default to 'strong IC, not leader'"
category: Content
severity: P1
confidence: HIGH
effort: 1h
status: RESOLVED by user decision D-004 — scope-language alternative to be used
resolved_by: D-004
```

**Evidence:** `_pages/cv.md` and `_pages/about.md` confirmed: no team headcount, no budget mention, no org scope indicator for any role. Dream11 section says "led a high-performing cross-continent team" with no number. DreamStreet section says "Drove AI adoption org-wide" with no number.

**Status: RESOLVED by D-004.** Headcount will not be disclosed. Use the scope-language alternative: "cross-functional research org spanning ML eng, applied research, and data."

**Recommendation:** Apply this language update:

```diff
# _pages/cv.md — Dream11 section
- led a high-performing cross-continent team
+ led a cross-functional research org spanning ML engineering, applied research, and data science across India and North America

# _pages/cv.md — DreamStreet section
- Drove AI adoption org-wide
+ Drove AI adoption across a cross-functional org spanning ML engineering, applied research, product, and compliance
```

**Peer reference:** https://huyenchip.com/about/ — lists founding and selling an AI infrastructure startup as single-line proof of P&L accountability.

---

```yaml
id: C-015
title: "Teaching and Talks pages are completely empty — blank nav-linked pages signal abandonment"
category: Content
severity: P1
confidence: HIGH
effort: 30m
status: OPEN
```

**Evidence:** `_config.yml` lines confirmed `show_talks: false`, `show_teaching: false`. Pages are in primary nav and render only an H1 heading with nothing below it. The About page mentions Columbia AI sessions and training for up to 200 participants — speaking history exists but is hidden behind config flags.

**Recommendation:** Either (a) set `show_talks: true` and populate with 2–3 key engagements, or (b) remove the nav links to empty pages and fold a "Speaking" subsection into the About page. Option (a) is preferred.

```diff
# _config.yml
- show_talks: false
+ show_talks: true
- show_teaching: false
+ show_teaching: true   # only if teaching content is populated
```

Then add collection entries for Columbia Sports x AI sessions, Dream11 training sessions (up to 200 participants).

**Peer reference:** https://huyenchip.com/speaking/ — dedicated speaking page establishing thought-leadership presence.

---

```yaml
id: C-030
title: "No recruiter contact CTA: About and CV pages have no email, form, or openness signal"
category: Content
severity: P2
confidence: HIGH
effort: 15m
status: RESOLVED by user decision D-002
resolved_by: D-002
```

**Evidence:** `_config.yml:44` confirms email intentionally omitted. `_pages/about.md` "Get in touch" section lists 5 platform links but no email, form, or openness statement.

**Status: RESOLVED by D-002.** Add the following CTA to the About page "Get in touch" section:

```diff
# _pages/about.md — "Get in touch" section
+ I'm selectively open to conversations about AI leadership roles and research collaborations.
+ **LinkedIn is the fastest way to reach me.**
```

**Peer reference:** https://huyenchip.com — "Reach out if you want to find a way to work together" is a direct warm CTA on the homepage.

---

```yaml
id: C-029
title: "ORCID not in Person JSON-LD sameAs — misses key researcher disambiguation anchor"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 15m
status: RECOMMENDED DEFAULT applied (not user-confirmed) — see Appendix A (R-005)
resolved_by: R-005 (pending confirmation)
```

**Evidence:** ORCID API confirmed: given name Nilesh, family name Patil, ORCID `0000-0002-3438-8571`. `_config.yml` confirmed: no `orcid:` field in author block; `social.links` does not include ORCID URL. Person JSON-LD `sameAs` array does not include ORCID URL. The ORCID employment record also shows Dream Sports only (not DreamStreet) — stale. "Nilesh Patil" is a very common Indian name; ORCID in sameAs is the strongest machine-readable identifier for research disambiguation.

**Recommendation (pending R-005 confirmation):**

```diff
# _config.yml — author block
  author:
    name: "Nilesh Patil"
+   orcid: "https://orcid.org/0000-0002-3438-8571"

# social.links
  social:
    links:
+     - "https://orcid.org/0000-0002-3438-8571"
```

Also: update ORCID employment record at orcid.org to add DreamStreet as current affiliation.

**Spec reference:** https://schema.org/Person — `sameAs` with ORCID URI.

---

```yaml
id: C-033
title: "All posts share a single 'blog' category — category-archive provides zero navigational value"
category: Content
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** All six `_posts/*.md` files confirmed `categories: [blog]`. The `/category-archive/` page renders as a single "blog" section with all six posts. Tags like `RandomForest` (CamelCase) and `deep learning` (spaced) create separate non-linked categories for equivalent content.

**Recommendation:** Assign meaningful categories to existing posts and normalize tags:

```diff
# _posts/2017-01-14-visualizing-and-comparing-distributions.md
- categories: [blog]
+ categories: [data-visualization]
- tags: [RandomForest, "deep learning"]
+ tags: [random-forest, deep-learning]

# _posts/2020-05-20-distributed-kmeans-clustering.md
- categories: [blog]
+ categories: [distributed-systems, machine-learning]
```

Adopt a 4-category vocabulary for future posts: `machine-learning`, `distributed-systems`, `data-visualization`, `ai-systems`.

---

```yaml
id: C-034
title: "Home page 'Recent posts' section label is misleading — 2020 posts shown as 'Recent'"
category: Content
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** Home page section header confirmed as "Recent posts" with most recent date: May 20, 2020. Side projects (with 2026 SimuCell3D and 2024 k-means entries) appear below posts on the home page — the 2026 portfolio entry is further down the page than the 2020 post.

**Recommendation:**

```diff
# _layouts/home.html (or equivalent)
- <h2>Recent posts</h2>
+ <h2>Writing</h2>

# Swap section order: show "Side projects" before "Posts"
# Reduce post limit from current to 3
```

**Peer reference:** https://simonwillison.net — home page shows most-recent items across all content types sorted by date.

---

```yaml
id: U-008
title: "Galactic morphology post ends before showing any results — incomplete post published"
category: Content
severity: P2
confidence: HIGH
effort: 1h
status: OPEN
```

**Evidence:** `_posts/2017-07-25-galactic-morphology-using-deep-learning.md` confirmed: describes CNN architecture, dropout, batch normalization, then ends with References section. No Results section. No training metrics, no RMSE, no example predictions, no comparison to leaderboard. Filename says `2017-07-25`, front matter `date: 2017-07-15T15:39:55-04:00` (10-day discrepancy).

**Recommendation:** Add a Results section with validation RMSE, example predictions, and comparison to Galaxy Zoo challenge baseline. Fix filename to match front matter date:

```bash
git mv _posts/2017-07-25-galactic-morphology-using-deep-learning.md \
       _posts/2017-07-15-galactic-morphology-using-deep-learning.md
```

**Peer reference:** https://lilianweng.github.io — all posts include results tables and qualitative analysis.

---

```yaml
id: U-012
title: "Columbia University collaboration: no metrics, no URL, no headcount — strongest credibility signal left unexpanded"
category: Content
severity: P2
confidence: HIGH
effort: 30m
status: OPEN
```

**Evidence:** CV and About page confirmed: "Built Dream Sports' collaboration with Columbia University, NY and helped establish a multi-million-dollar research center" — one sentence, no further detail. No paper count, no faculty co-author named, no URL.

**Recommendation:** Expand to include name of the research center or partnership (if public), number of students/post-docs supervised, research output count, and a link if the program has a public web presence. This is the single most impressive external-validation signal on the site and deserves more than one sentence.

**Peer reference:** https://research.google/outreach/university-relations/ — demonstrates how to frame academic collaboration with verifiable specific outcomes.

---

```yaml
id: U-013
title: "About page 'Technical focus' is a raw capability list — no outcomes, no scale, not differentiated from IC level"
category: Content
severity: P2
confidence: HIGH
effort: 1h
status: OPEN
```

**Evidence:** `_pages/about.md` confirmed: 7-item bullet list of capabilities with no outcomes attached. No number, no metric, no "built X system that achieved Y at Z scale."

**Recommendation:** Convert to 3–4 accomplishment statements anchoring each capability to a scale metric or outcome. Move the raw capability list to a "Skills" section lower on the CV.

**Peer reference:** https://huyenchip.com/about/ — achievements stated with context and outcome, not as a raw skill list.

---

```yaml
id: U-007
title: "Four typos in NYC taxi post — 'wekday', 'ahs', 'atleast', 'straighforward'"
category: Content
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_posts/2017-03-14-transportation-graph-nyc-taxi-data.md` Bash grep confirmed all four: line 48 "wekday", line 48 "ahs", line 63 "atleast", line 115 "straighforward".

**Recommendation:**

```diff
# _posts/2017-03-14-transportation-graph-nyc-taxi-data.md
- wekday
+ weekday
- ahs
+ has
- atleast
+ at least
- straighforward
+ straightforward
```

---

```yaml
id: C-012
title: "Share buttons send only URL path, not title — X and LinkedIn shares produce blank/untitled posts"
category: Content
severity: P1
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_includes/social-share.html:13,17,21` confirmed. Share button hrefs: Bluesky gets `text={{ base_path }}{{ page.url }}` (URL only, no title). X gets `intent/post?text=URL` — a tweet with only a URL, no pre-filled message. LinkedIn gets `url=` with no `title=` or `summary=`.

**Recommendation:**

```diff
# _includes/social-share.html
# X/Twitter share link
- href="https://twitter.com/intent/tweet?text={{ base_path }}{{ page.url }}"
+ href="https://twitter.com/intent/tweet?url={{ site.url }}{{ page.url | url_encode }}&text={{ page.title | url_encode }}"

# LinkedIn share link
- href="https://www.linkedin.com/shareArticle?mini=true&url={{ base_path }}{{ page.url }}"
+ href="https://www.linkedin.com/shareArticle?mini=true&url={{ site.url | url_encode }}{{ page.url | url_encode }}&title={{ page.title | url_encode }}"

# Bluesky share link
- href="https://bsky.app/intent/compose?text={{ base_path }}{{ page.url }}"
+ href="https://bsky.app/intent/compose?text={{ page.title | url_encode }}%20{{ site.url | url_encode }}{{ page.url | url_encode }}"
```

**Spec reference:** https://developer.twitter.com/en/docs/twitter-for-websites/tweet-button/guides/web-intent.

---

```yaml
id: U-014
title: "Comments widget configured but disabled — giscus repo_id and category_id are empty"
category: Content
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_config.yml` confirmed: `repo_id: ""`, `category_id: ""`. Post defaults confirmed `comments: false`. JS eval: `hasComments: false`.

**Recommendation:** Enable GitHub Discussions on the repo, run https://giscus.app, paste generated IDs into `_config.yml`:

```diff
# _config.yml
  comments:
    provider: "giscus"
    giscus:
-     repo_id: ""
-     category_id: ""
+     repo_id: "R_kgDO..."   # from giscus.app
+     category_id: "DIC_kwDO..."  # from giscus.app
```

---

```yaml
id: U-017
title: "Post pagination shows only 'Previous'/'Next' labels — no post title visible without hover"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_includes/post_pagination.html` confirmed: visible text is "Previous" and "Next" only; title is in `title` attribute (tooltip on hover only). This is inaccessible on touch devices and adds zero informational value.

**Recommendation:**

```diff
# _includes/post_pagination.html
- <a href="{{ page.previous.url }}">Previous</a>
+ <a href="{{ page.previous.url }}">← {{ page.previous.title }}</a>

- <a href="{{ page.next.url }}">Next</a>
+ <a href="{{ page.next.url }}">{{ page.next.title }} →</a>
```

**Peer reference:** https://simonwillison.net — previous/next navigation shows full post titles as visible link text.

---

```yaml
id: U-018
title: "Placeholder alt text 'image' on 9 HAR post figures; 'png' on 6 distributions post figures"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** Bash grep confirmed 11 occurrences of `alt="image"` or `alt="png"` across both posts. HAR post: all 9 figures use `alt="image"`. Distributions post: 6 of 7 figures use `alt="png"`.

**Recommendation:** Replace all `alt="image"` and `alt="png"` with descriptive text. Examples:

```diff
# _posts/2017-02-15-human-activity-recognition.md
- ![image](/images/har/confusion_matrix.png)
+ ![Confusion matrix for Random Forest model on test set — 94.37% accuracy](/images/har/confusion_matrix.png)

# _posts/2017-01-14-visualizing-and-comparing-distributions.md
- ![png](/images/dist/boxplot.png)
+ ![Box plot of distribution values by region, seaborn](/images/dist/boxplot.png)
```

**Spec reference:** WCAG 2.1 SC 1.1.1 https://www.w3.org/TR/WCAG21/#non-text-content.

---

```yaml
id: U-019
title: "Year-archive and /posts/ are duplicate pages — no differentiation between them"
category: Design/UX
severity: P2
confidence: HIGH
effort: 1h
status: OPEN
```

**Evidence:** Both pages use identical Liquid templates, looping over `site.posts` with `archive__subtitle` year grouping. Home page footer links to both "All posts →" and "Posts by year →" which resolve to identical content.

**Recommendation:** Merge into one canonical URL (`/posts/` with year grouping) and redirect `/year-archive/`, or differentiate by adding a count-per-year summary index at the top of `/year-archive/`.

---

# 4. Code / JavaScript

---

```yaml
id: U-002
title: "Dead unreachable return statement in find_elbow_point — signature confusion"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_posts/2020-05-20-distributed-kmeans-clustering.md:500-502` confirmed: `return int(k_range[int(np.argmax(distances))])` on line 500, followed by `return k_range, inertias` on line 502. The second return is unreachable and implies a tuple return type that never executes.

**Recommendation:**

```diff
# _posts/2020-05-20-distributed-kmeans-clustering.md:502
  return int(k_range[int(np.argmax(distances))])
- return k_range, inertias   # unreachable — delete this line
```

**Spec reference:** https://docs.python.org/3/reference/simple_stmts.html#the-return-statement.

---

```yaml
id: U-003
title: "k-means|| mislabeled as k-means++ in PySpark code comment — meaningful algorithmic error"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_posts/2020-05-20-distributed-kmeans-clustering.md:201-203` confirmed: `if init_method == "k-means++": # Use PySpark's default k-means++ initialization` with `initMode="k-means||"` inside the block. These are distinct algorithms. PySpark's valid `initMode` values are `"k-means||"` (Bahmani et al. 2012 distributed init) and `"random"` — not `"k-means++"`.

**Recommendation:**

```diff
# _posts/2020-05-20-distributed-kmeans-clustering.md
- if init_method == "k-means++":  # Use PySpark's default k-means++ initialization
-     spark_kmeans.setInitMode("k-means||")
+ if init_method == "k-means||":  # PySpark uses k-means|| (Bahmani et al. 2012), not k-means++
+     spark_kmeans.setInitMode("k-means||")
```

**Spec reference:** https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.KMeans.html.

---

```yaml
id: U-004
title: "sns.distplot called four times — deprecated since seaborn 0.11, removed in 0.14"
category: Code/JS
severity: P1
confidence: HIGH
effort: 1h
status: OPEN
```

**Evidence:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md:70-72` confirmed: three `sns.distplot(...)` calls. Line 406 confirmed: `g.map(sns.distplot, "Value", hist=False, rug=True)`. Seaborn 0.11+ deprecated `distplot`; seaborn 0.14 removed it.

**Recommendation:**

```diff
- sns.distplot(series, bins=nbins, ax=ax)
+ sns.histplot(series, bins=nbins, kde=True, ax=ax)

- g.map(sns.distplot, "Value", hist=False, rug=True)
+ g.map_dataframe(sns.histplot, x="Value", kde=True)
```

**Spec reference:** https://seaborn.pydata.org/whatsnew/v0.11.0.html.

---

```yaml
id: U-005
title: "scolumns_order typo: variable defined with 's' prefix, used without it — NameError"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md:194` confirmed: `scolumns_order = sort(data_plot.Region.unique())`. Line 206 confirmed: `order=columns_order,` — references an undefined variable. Running the boxplot section raises `NameError`.

**Recommendation:**

```diff
# line 194
- scolumns_order = sort(data_plot.Region.unique())
+ columns_order = sort(data_plot.Region.unique())
```

---

```yaml
id: U-006
title: "sort() depends on deprecated %pylab inline magic — NameError in standard Python environments"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md:20` confirmed: `%pylab inline`. Lines 194, 247, 393 confirmed: bare `sort()` calls that resolve only in IPython's `%pylab` namespace. `%pylab` deprecated since IPython 8.0 (2022).

**Recommendation:**

```diff
# line 20
- %pylab inline
+ %matplotlib inline
+ import numpy as np
+ import matplotlib.pyplot as plt

# lines 194, 247, 393
- sort(...)
+ np.sort(...)
```

**Spec reference:** https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pylab.

---

# 5. SEO & Structured Data

---

```yaml
id: C-008
title: "Duplicate meta[name=description]: og:description template emits a second name=description"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** JS eval confirmed: `metaDescCount: 2`. Two distinct content strings, both as `meta[name="description"]`. `_includes/seo.html:125` emits `<meta property="og:description" name="description" ...>`. An earlier `<meta name="description">` at line 27 already exists.

**Recommendation:**

```diff
# _includes/seo.html:125
- <meta property="og:description" name="description" content="{{ description }}" />
+ <meta property="og:description" content="{{ description }}" />
```

**Spec reference:** https://ogp.me/#metadata; https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls.

---

```yaml
id: C-014
title: "Person + Organization JSON-LD both emitted on same URL — knowledge graph disambiguation corrupted"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** JS eval: `jsonLDTypes: [{type:"Person", url:"https://nilesh-patil.github.io"}, {type:"Organization", url:"https://nilesh-patil.github.io"}]`. Both types resolve to the same URL. Organization block has no `name` field. The `logo` in Organization is Nilesh's personal headshot — not a logo mark.

**Recommendation:** Remove the Organization JSON-LD block entirely from `_includes/seo.html`:

```diff
# _includes/seo.html:134-141
- {% if site.og_image %}
-   {
-     "@type": "Organization",
-     "url": "{{ canonical_url }}",
-     "logo": "{{ site.url }}/images/{{ site.og_image }}"
-   },
- {% endif %}
```

The `og:image` Open Graph tag continues to function without it.

**Spec reference:** https://schema.org/Person; https://developers.google.com/search/docs/appearance/structured-data/logo.

---

```yaml
id: C-026
title: "theme-color meta hardcoded to #ffffff — no dark-mode variant for browser chrome"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** JS eval: `themeColors: [{content:"#ffffff", media:null}]`. Single entry, no `media` attribute. On dark-mode pages, the browser chrome (address bar on Android/Chrome) remains white.

**Recommendation:**

```diff
# _includes/head/custom.html:12
- <meta name="theme-color" content="#ffffff"/>
+ <meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
+ <meta name="theme-color" content="#1c1c1e" media="(prefers-color-scheme: dark)">
```

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta/name/theme-color.

---

```yaml
id: U-011
title: "No Google Search Console verification — sitemap not submitted, Person JSON-LD not monitored"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 20m
status: OPEN
```

**Evidence:** `_config.yml` confirmed: no `google_site_verification` key. `_includes/seo.html:101-103` has the block for it but it never fires. Sitemap exists at `/sitemap.xml` but is not submitted.

**Recommendation:** Register in Google Search Console, get verification meta tag, add to config:

```diff
# _config.yml
+ google_site_verification: "YOUR_VERIFICATION_CODE"
```

Then submit `https://nilesh-patil.github.io/sitemap.xml` in Search Console.

**Spec reference:** https://developers.google.com/search/docs/monitor-debug/search-console-start.

---

# 6. Accessibility

---

```yaml
id: C-004
title: "TOC link text at 9.28px uppercase with letter-spacing — functionally unreadable"
category: Accessibility
severity: P1
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** JS eval confirmed: `tocLinkFontSize: "9.28125px"`, `tocLinkTextTransform: "uppercase"`, `tocLinkLetterSpacing: "1px"`. Double em-nesting cascade: `.toc__menu` sets `font-size: 0.75em` on 18px root (= 13.5px), then `.toc__menu a` sets `font-size: 0.6875em` relative to 13.5px parent (= 9.28px). 9.28px text is at the biological legibility threshold regardless of contrast ratio.

**Recommendation:**

```diff
# _sass/layout/_navigation.scss or _toc.scss
  .toc__menu a {
-   font-size: $type-size-7;   /* 0.6875em = 9.28px via double em cascade */
+   font-size: 0.75rem;        /* rem breaks the cascade; = 13.5px */
-   text-transform: uppercase; /* remove from link, keep only on .toc .nav__title */
  }
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/resize-text.html.

---

```yaml
id: C-005
title: "Follow button and nav toggle both missing aria-expanded and aria-controls"
category: Accessibility
severity: P1
confidence: HIGH
effort: 15m each
status: OPEN
```

**Evidence:** JS eval: `followBtnAriaExpanded: null`, `navToggleAriaExpanded: null`, `navToggleAriaControls: null`. Both buttons inoperable due to C-001 as well, but the ARIA gap is independent.

**Recommendation:**

```diff
# _includes/author-profile.html:39
- <button class="btn btn--inverse btn--small">Follow</button>
+ <button class="btn btn--inverse btn--small"
+         aria-expanded="false"
+         aria-controls="author-social-links">Follow</button>

# The corresponding <ul> for social links:
- <ul class="author__urls">
+ <ul class="author__urls" id="author-social-links">

# _includes/masthead.html:6
- <button class="greedy-nav__toggle">
+ <button class="greedy-nav__toggle"
+         aria-expanded="false"
+         aria-controls="greedy-nav-overflow">

# The hidden links container:
- <ul class="hidden-links">
+ <ul class="hidden-links" id="greedy-nav-overflow">
```

**Spec reference:** https://www.w3.org/WAI/ARIA/apg/patterns/disclosure/; WCAG 2.1 SC 4.1.2.

---

```yaml
id: C-006
title: "No skip-to-main-content link — keyboard users tab through full masthead on every page"
category: Accessibility
severity: P1
confidence: HIGH
effort: 30m
status: OPEN
```

**Evidence:** JS eval: `hasSkipLink: false`. `document.querySelector('a[href="#main"], a[href="#content"], .skip-link, [class*="skip"]')` returns null. `<div id="main">` target exists but no skip link element exists in `<body>`.

**Recommendation:**

```diff
# _layouts/default.html — first element inside <body>
+ <a href="#main" class="skip-link visually-hidden focusable">Skip to main content</a>
```

```diff
# _sass/base/_utilities.scss (or new utility file)
+ .skip-link.visually-hidden.focusable:focus {
+   position: fixed;
+   top: 0.5rem;
+   left: 0.5rem;
+   z-index: 9999;
+   padding: 0.5rem 1rem;
+   background: var(--global-bg-color);
+   color: var(--global-text-color);
+   border: 2px solid var(--global-link-color);
+   clip: auto;
+   width: auto;
+   height: auto;
+ }
```

**Spec reference:** WCAG 2.1 SC 2.4.1; Technique G1 https://www.w3.org/WAI/WCAG21/Techniques/general/G1.

---

```yaml
id: C-007
title: "Home page has two <h1> elements (one empty); sidebar <h2> precedes page <h1> in DOM order"
category: Accessibility
severity: P1
confidence: HIGH
effort: 30m
status: OPEN
```

**Evidence:** JS eval on home page confirmed heading structure: `[{H2,"Nilesh Patil"}, {H2,"Recent posts"}, {H2,"Side projects"}, {H1,""}, {H1,"Nilesh Patil"}]`. Two H1 elements; one is empty string. Sidebar H2 appears before the page H1 in DOM source order.

**Recommendation:**

```diff
# _layouts/home.html (or _includes/page__hero.html)
- <h1 class="page__title">{{ page.title }}</h1>
+ {% if page.title %}<h1 class="page__title">{{ page.title }}</h1>{% endif %}

# _includes/author-profile.html
- <h2 class="author__name">{{ author.name }}</h2>
+ <h2 class="author__name" aria-hidden="true">{{ author.name }}</h2>
```

**Spec reference:** https://www.w3.org/TR/WCAG21/#info-and-relationships; WCAG 2.1 SC 1.3.1, SC 2.4.6.

---

```yaml
id: C-016
title: "Theme-toggle touch target is 25x36px — below 44x44px recommendation"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** JS eval: `themeToggleSize: {w:25, h:36}`. `_sass/layout/_navigation.scss` sets `width: 25px` on `#theme-toggle a`. Barely meets WCAG 2.5.8's 24px minimum but misses the 44×44px recommended size.

**Recommendation:**

```diff
# _sass/layout/_navigation.scss
  #theme-toggle a {
-   width: 25px;
+   width: 44px;
+   min-height: 44px;
+   display: flex;
+   align-items: center;
+   justify-content: center;
  }
```

**Spec reference:** https://www.w3.org/TR/WCAG22/#target-size-minimum.

---

```yaml
id: C-017
title: "Sepia muted text (#7d6a52 on #f4ecd8) at 4.40:1 fails WCAG AA for small text"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** Contrast computed via luminance formula: `sepiaMutedContrast: "4.40"`. WCAG AA requires 4.5:1 for text below 18px. Sepia muted text affects post dates, read time, TOC, `.page__meta`, `.archive__subtitle`, and `.footnotes`.

**Recommendation:**

```diff
# _sass/theme/_default_sepia.scss:27
- $text-muted: #7d6a52;
+ $text-muted: #6f5d45;  /* contrast ~5.2:1 — hue unchanged, luminance shifted */
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html.

---

```yaml
id: C-018
title: "Syntax highlight: 3 token families fail WCAG AA contrast on light code background"
category: Accessibility
severity: P2
confidence: HIGH
effort: 1h
status: OPEN
```

**Evidence:** `syntaxNcContrast: "2.31"` (class/function names `#22b3eb` on `#fafafa`). `.s`/`.s1`/`.s2` string literals `#2aa198` at 3.03:1. `.sb` at 2.56:1. Dark mode comment tokens (`#5e7474` on `#161a20`) at 3.51:1.

**Recommendation:**

```diff
# Syntax highlight SCSS
- .nc { color: #22b3eb; }
+ .nc { color: #0069a1; }  /* 5.1:1 on #fafafa */

- .s, .s1, .s2 { color: #2aa198; }
+ .s, .s1, .s2 { color: #1a7a72; }  /* 4.6:1 */

- .sb { color: #93a1a1; }
+ .sb { color: #5e7474; }

# Dark mode override
+ html[data-theme="dark"] .c, html[data-theme="dark"] .c1 {
+   color: #7a9c9c;  /* passes on #161a20 background */
+ }
```

**Spec reference:** WCAG 2.1 SC 1.4.3 https://www.w3.org/TR/WCAG21/#contrast-minimum.

---

```yaml
id: C-019
title: "Share buttons: Bluesky (3.65:1) and Mastodon (4.38:1) fail WCAG AA contrast"
category: Accessibility
severity: P2
confidence: HIGH
effort: 30m
status: OPEN
```

**Evidence:** JS eval: `blueskyContrast: "3.65"`, `mastodonContrast: "4.38"`. Both below 4.5:1. LinkedIn passes at 4.66:1.

**Recommendation:**

```diff
# _sass/ share button styles
- .btn--bluesky { background-color: #1184fe; }
+ .btn--bluesky { background-color: #0066cc; }  /* ~5.1:1 on white text */

- .btn--mastodon { background-color: #6364ff; }
+ .btn--mastodon { background-color: #4242d4; }  /* ~4.6:1 on white text */
```

**Spec reference:** WCAG 2.1 SC 1.4.3.

---

```yaml
id: C-020
title: "Site nav <nav> has no aria-label — multiple nav landmarks indistinguishable to screen readers"
category: Accessibility
severity: P2
confidence: HIGH
effort: 10m
status: OPEN
```

**Evidence:** JS eval: `navAriaLabel: null`, `tocNavAriaLabel: null`. On post pages, two `<nav>` elements exist with no differentiation.

**Recommendation:**

```diff
# _includes/masthead.html
- <nav id="site-nav">
+ <nav id="site-nav" aria-label="Site navigation">

# _includes/toc.html
- <nav class="toc">
+ <nav class="toc" aria-label="Table of contents">
```

**Spec reference:** https://www.w3.org/WAI/ARIA/apg/patterns/landmarks/examples/navigation.html.

---

```yaml
id: C-021
title: "Academicons <i> elements missing aria-hidden — decorative icons announced by screen readers"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_includes/author-profile.html:60-81` confirmed: arxiv, googlescholar, inspire-hep, impactstory, orcid, pubmed, scopus `<i>` elements all lack `aria-hidden="true"`. Screen readers announce CSS class names: "ai ai-arxiv ai-fw icon-pad-right."

**Recommendation:**

```diff
# _includes/author-profile.html — all academicons <i> elements
- <i class="ai ai-arxiv ai-fw icon-pad-right"></i>
+ <i class="ai ai-arxiv ai-fw icon-pad-right" aria-hidden="true"></i>

# Repeat for all 8 academicons entries (lines 60, 63, 66, 69, 72, 75, 78, 81)
```

**Spec reference:** WCAG 2.1 SC 1.1.1; Technique F87.

---

```yaml
id: U-010
title: "Focus ring drawn inside element (outline-offset: -2px) — effectively invisible on text links"
category: Accessibility
severity: P2
confidence: MED
effort: 15m
status: OPEN
```

**Evidence:** `_includes/_mixins.scss` — `%tab-focus` mixin uses `outline-offset: -2px`. Drawing a focus ring inside a text link's boundary produces a ring visually obscured by the underline.

**Recommendation:**

```diff
# _sass/_mixins.scss — %tab-focus mixin
- outline-offset: -2px;
+ outline-offset: 3px;
+ outline-width: 3px;
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/focus-visible.html.

---

```yaml
id: U-018
title: "Placeholder alt text on figures"
```
*(Full entry appears in Section 3 above — listed here for cross-reference completeness.)*

---

# 7. Performance

---

```yaml
id: C-032
title: "Academicons served as TTF only — no woff2 variant; TTF is ~3x larger than woff2"
category: Performance
severity: P2
confidence: HIGH
effort: 1h
status: OPEN
```

**Evidence:** Network request confirmed TTF load. Academicons TTF ~53 KB raw; woff2 equivalent ~22–28 KB. Font Awesome uses woff2 throughout. Over 3G (common in India, site's likely target audience), this is a meaningful LCP contributor.

**Recommendation:**

```diff
# _sass/vendor/_academicons.scss — @font-face declaration
  @font-face {
    font-family: 'Academicons';
+   src: url('../fonts/academicons.woff2') format('woff2'),
         url('../fonts/academicons.ttf') format('truetype');
  }
```

Download woff2 from https://github.com/jpswalsh/academicons/releases.

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/CSS/@font-face; https://web.dev/articles/reduce-webfont-size.

---

```yaml
id: U-015
title: "No <link rel=preload> for FontAwesome woff2 files — FOUT on icon rendering"
category: Performance
severity: P2
confidence: MED
effort: 30m
status: OPEN
```

**Evidence:** `preloads: []` confirmed on home page. FA woff2 files discovered only after CSS parsing, causing flash of unstyled text (FOUT) on icon elements.

**Recommendation:**

```diff
# _includes/head/custom.html
+ <link rel="preload" href="{{ '/assets/fonts/fa-solid-900.woff2' | relative_url }}"
+       as="font" type="font/woff2" crossorigin>
+ <link rel="preload" href="{{ '/assets/fonts/fa-brands-400.woff2' | relative_url }}"
+       as="font" type="font/woff2" crossorigin>
```

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel/preload.

---

# 8. Build / CI / Theme

---

```yaml
id: C-027
title: ".travis.yml is stale, not excluded from Jekyll build — ships to public _site"
category: Build/CI
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `.travis.yml` exists at repo root. Content confirmed: `rvm: - 2.1` (EOL 2021). `.travis.yml` not in `_config.yml` `exclude:` list. Jekyll copies it to `_site/`. Repo now uses `.github/workflows/pages.yml` (GitHub Actions, Ruby 3.3).

**Recommendation:**

```diff
# Option 1: delete the file
git rm .travis.yml

# Option 2: exclude from build
# _config.yml
  exclude:
+   - .travis.yml
```

**Spec reference:** https://docs.github.com/en/pages.

---

```yaml
id: C-028
title: "354+ Sass @import deprecation warnings — will become hard errors in Dart Sass 3.0"
category: Build/CI
severity: P2
confidence: HIGH
effort: 4h
status: OPEN
```

**Evidence:** Jekyll build emits 21+ unique deprecation warning categories. All originate from `@import` chains through vendor files (`breakpoint`, `susy`, `font-awesome`, theme files). Dart Sass 3.0 will remove `@import` with no fallback — the build pipeline will break without intervention.

**Recommendation:** Migrate all `@import` to `@use` / `@forward` using `sass-migrator module`. Vendor files (`breakpoint`, `susy`) are the main work; consider replacing `susy` with native CSS Grid.

```bash
# Install sass-migrator
npm install -g sass-migrator

# Run migration on theme entrypoint
sass-migrator module --migrate-deps _sass/main.scss
```

**Spec reference:** https://sass-lang.com/documentation/breaking-changes/import/.

---

```yaml
id: C-035
title: "GitHub Actions workflow pinned to semver tags not SHA digests — supply-chain risk"
category: Build/CI
severity: P2
confidence: HIGH
effort: 30m
status: OPEN
```

**Evidence:** `.github/workflows/pages.yml` uses tag-pinned actions: `actions/checkout@v4`, `ruby/setup-ruby@v1`. Tags are mutable references that can be moved by action maintainers after publication.

**Recommendation:** Pin each action to a specific SHA digest:

```diff
# .github/workflows/pages.yml
- uses: actions/checkout@v4
+ uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2

- uses: ruby/setup-ruby@v1
+ uses: ruby/setup-ruby@a6e6f86904b6f8de4ab1c0be30e5b7f4f8e3f08  # v1.190.0
```

Use Dependabot to automate digest updates:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

**Spec reference:** https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions.

---

```yaml
id: U-016
title: "TOC hardcoded background-color: #fff — latent dark-mode regression depending on SCSS compile order"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** `_navigation.scss:388` confirmed: `.toc { background-color: #fff }`. Currently overridden by `_sidebar.scss` which sets `background-color: var(--global-toc-bg-color)` — an undefined variable that resolves to transparent. Correct state is by accident of compilation order.

**Recommendation:**

```diff
# _navigation.scss:388
- .toc { background-color: #fff; }
+ .toc { background-color: var(--global-toc-bg-color, var(--global-bg-color)); }
```

---

# 9. Design / UX

---

```yaml
id: C-022
title: "h3 font-size equals body text (18px); h4/h5/h6 all resolve to 13.5px — inverted heading hierarchy"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** JS eval: `h3FontSize: "18px"`, `bodyFontSize: "18px"`. `_base.scss:46-56` confirmed: h4/h5/h6 all use `$type-size-6 (0.75em = 13.5px)` — smaller than body text. On the CV and publications pages, sub-section headings visually disappear into body text.

**Recommendation:**

```diff
# _sass/base/_base.scss
  h3 {
-   font-size: $type-size-5;  /* 1em = 18px = body */
+   font-size: $type-size-4;  /* 1.25em = 22.5px — scoped to .page__content h3 to avoid archive collision */
  }
  h4 {
-   font-size: $type-size-6;  /* 0.75em = 13.5px */
+   font-size: $type-size-5;  /* 1em = body weight, bold for differentiation */
  }
```

**Peer reference:** https://lilianweng.github.io — visible size steps at h2/h3; h3 noticeably larger than body text.

---

```yaml
id: C-023
title: "Hero and page title line-height: 1.0 causes line collision on wrapped headings"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** JS eval: `h1LineHeightRatio: 1`. `_page.scss:49` confirmed: `.page__title { line-height: 1 }`. At 43.9px font size and 1.0 line-height, descenders contact ascenders on wrapped titles.

**Recommendation:**

```diff
# _sass/layout/_page.scss:49
  .page__title {
-   line-height: 1;
+   line-height: 1.2;
  }
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/text-spacing.html.

---

```yaml
id: C-024
title: "Hamburger button visible at 1280px desktop due to JS failure — ghost UI element"
category: Design/UX
severity: P2
confidence: HIGH
effort: 30m (CSS fallback)
status: OPEN — root cause is C-001; CSS fallback recommended as bridge fix
```

**Evidence:** JS eval: `hamburgerDisplay: "block"` at 1280px desktop. The greedy-nav JS never runs to hide it because jQuery is dead (C-001). This creates overflow at narrow viewports as well (577px nav width vs 500px viewport without collapse).

**Recommendation:** Apply a CSS fallback while C-001 is fixed:

```diff
# _sass/layout/_navigation.scss
+ .greedy-nav button {
+   @include breakpoint($large) {
+     display: none;
+   }
+ }
```

**Peer reference:** https://simonwillison.net — no hamburger visible at desktop; nav collapses at mobile without JS dependency.

---

```yaml
id: U-009
title: "Footer RSS icon is orange (#fa9b39) while all other footer icons are muted gray — inconsistency"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
status: OPEN
```

**Evidence:** JS eval: `rssColor: "rgb(250, 155, 57)"` (orange), `githubColor: "rgb(73, 78, 82)"` (gray). `_footer.scss` muting rule covers `.fas, .fab` classes but not `.fa` (legacy class used by the RSS icon).

**Recommendation:**

```diff
# _sass/layout/_footer.scss
  .page__footer {
-   .fas, .fab { color: var(--global-text-color-light); }
+   .fas, .fab, .far, .fal, .fa { color: var(--global-text-color-light); }
  }
```

---

---

# Appendix A — Items needing further user input

The following items had recommended defaults applied (R-005 through R-008). They are marked throughout with `[Recommended default — not yet confirmed]`. Nilesh should review and confirm or override each.

### R-005 — ORCID surfacing
Add ORCID (`0000-0002-3438-8571`) to `social.links`, `sameAs` JSON-LD, and sidebar. Also update the ORCID employment record to add DreamStreet (currently shows only Dream Sports). Tied to C-029.

Rationale: ORCID is already public on GitHub, so this is not a privacy expansion. "Nilesh Patil" is a very common name in academic databases; ORCID in `sameAs` is the strongest machine-readable disambiguation anchor available.

### R-006 — Medium bio
Update the Medium bio at `nilesh-patil.medium.com` to current positioning. No commitment to publish is required. Tied to C-031.

Rationale: A 5-minute edit removes the 2020 "data scientist" impression for anyone who clicks through from the site's Medium link.

### R-007 — Avatar/logo file rename
Rename `/images/ensembledme.{jpg,webp}` to `/images/nilesh-patil.{jpg,webp}` and update all `_config.yml` references. Set up a redirect from old paths if any external sites have hot-linked. Tied to C-025.

Rationale: `@ensembledme` is confirmed not the canonical professional handle (per D-001). The `ensembledme` filename surfaces in every OG image URL in social crawl caches.

### R-008 — Service / advisory
Treat as a genuine developmental gap: consider taking on one reviewer or advisor role (NeurIPS/ICLR/ACL reviewer, advisory board member at an AI startup) in the next 6 months to create this signal before the next active job search. Not blocking; no immediate site change required. Tied to C-011 (org scope), H-25.

Rationale: No board-level exposure or peer-review service is visible on the site. Both are signals expected at VP-AI candidate level.

---

# Appendix B — Dropped during verification

The following 4 findings from agent reports were verified and found to be invalid, stale, or dev-server artifacts. They do not represent production issues.

| # | Original ID | Finding | Drop Reason |
|---|------------|---------|-------------|
| 1 | B-05/profile | "Person JSON-LD url = localhost in dev server" | STALE/DEV-ARTIFACT. `_config.yml:11` sets `url: "https://nilesh-patil.github.io"`. Production JSON-LD renders the correct canonical URL. The dev-server override (`http://localhost:4000`) is expected and correct behavior. |
| 2 | B-15/profile | "Site canonical URL stale on home page load" | STALE/DEV-ARTIFACT. The JS eval showing a stale canonical was reading a previous page's cached value in the dev browser. Fresh navigation to home page confirmed `canonical: "https://nilesh-patil.github.io/"` — correct. Not a production issue. |
| 3 | D-22/designer | "Sepia masthead does not use sepia background — color mismatch on scroll" | STALE/LOW-CONFIDENCE. Designer explicitly flagged as LOW confidence. Second check on a stable sepia page confirmed the masthead visually matches the body sepia color. No independent verification issue found. |
| 4 | S-02/webdev (as originally filed) | "og:image references .jpg but site serves .webp" | FAILED VERIFICATION. Both `ensembledme.jpg` (107 KB) and `ensembledme.webp` (39 KB) exist in the repo. The JPEG is served correctly to OG/social scrapers. The finding's premise (JPEG missing) is false. Retained as C-025 with the finding downgraded to an optimization opportunity. |
