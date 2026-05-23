# Blog Audit — Nilesh Patil
**Posts:** 6 (2017–2020) | **Peer bar:** simonwillison.net, lilianweng.github.io, huyenchip.com

## Scorecard
| Dimension | Grade |
|---|---|
| Archive shape | D — 5-year silence is the loudest signal |
| Post index UX | C+ — read time/excerpts; no tags, no search |
| Per-post UX | B− — sticky TOC good; no copy button; comments not live |
| Content quality 2017 cluster | C — accurate but dated |
| Content quality 2020 post | B — solid structure; dead-code bug + stale API |
| Comments / engagement | F — Giscus wired but IDs blank; broken banner shown |
| RSS | B — auto-discoverable; not actively promoted |
| Distribution loop | C− — share buttons; no newsletter; no LinkedIn |
| Series / related posts | C |
| Head-of-AI signal | D — nothing reflects current seniority/topics |

## [Severity: P0] — Comments widget displays a broken-placeholder banner on every post
**Category:** Engagement
**Location:** `_includes/comments-providers/giscus.html:3-10`; `_config.yml` `repo_id`, `category_id` are `""`
**Evidence:** Template checks `if g.repo_id == "" or g.category_id == ""` and renders yellow "Comments are not yet wired" notice. Visible to every reader on every post.
**Recommendation:** Enable GitHub Discussions on the repo, run https://giscus.app wizard, paste IDs into `_config.yml`. 10 minutes.

## [Severity: P0] — Archive signals abandonment, not selectivity
**Category:** Archive
**Location:** /posts/
**Evidence:** Year headings read "2020 … 2017" with nothing for 2021–2026.
**Recommendation:** (a) Publish 2–3 posts dated 2025/2026 — only permanent fix. Or (b) curated `/writing/` landing page surfacing 2–3 strongest pieces.
**Reference:** https://huyenchip.com

## [Severity: P0] — Head-of-AI in 2026 has zero posts about AI
**Category:** Content
**Recommendation:** One long-form post (2,000–3,000 words): "Building compliance-aware AI for SEBI-regulated workflows" or "What agentic evaluation looks like at scale."
**Reference:** https://lilianweng.github.io/posts/2023-06-23-agent/

## [Severity: P1] — No tags visible on post index; no filter or search
**Category:** Index UX
**Location:** `_pages/posts.html`, `_includes/archive-single.html`
**Evidence:** Tags set in every post's frontmatter but `archive-single.html` doesn't render them. `/tag-archive/` exists but isn't linked.
**Recommendation:** Add `{% include tag-list.html %}` to `archive-single.html` below the excerpt. Link tag archive from posts header.

## [Severity: P1] — No code copy button on code blocks
**Category:** Post UX
**Evidence:** distributed-kmeans has functions exceeding 30 lines. Readers must manually select text.
**Recommendation:** ~25 lines of vanilla JS in `assets/js/copy-code.js` loaded in `_includes/scripts.html`. On DOMContentLoaded, insert `<button>` into each `.highlight`; on click, `navigator.clipboard.writeText(pre.textContent)`.
**Reference:** https://simonwillison.net

## [Severity: P1] — Share buttons pre-fill only the URL; title and author handle absent
**Category:** Distribution
**Location:** `_includes/social-share.html`
**Evidence:** X share URL passes only the bare URL, no title, no `@ensembledme` handle.
**Recommendation:** Change X href to `https://x.com/intent/post?text={{ page.title | url_encode }}%20{{ base_path | append: page.url | url_encode }}&via=ensembledme`. For LinkedIn: append `&title=...&summary=...`.
**Reference:** https://developer.x.com/en/docs/twitter-for-websites/tweet-button/overview

## [Severity: P1] — LinkedIn absent from author profile and distribution chain
**Category:** Distribution
**Recommendation:** Add `linkedin: "<handle>"` to author block in `_config.yml`.

## [Severity: P1] — No "Now" page; no signal of active intellectual work
**Category:** IA
**Recommendation:** Add single Markdown file at `_pages/now.md` with 3–5 bullets updated quarterly.
**Reference:** https://simonwillison.net/now/, https://nownownow.com/about

## [Severity: P1] — No newsletter or email subscription path
**Category:** Engagement
**Recommendation:** Buttondown (buttondown.email) — free tier, GDPR-compliant, native RSS-to-email. One-line embed on posts index and bottom of each post.
**Reference:** https://buttondown.email

## [Severity: P2] — Post index excerpts are generic labels, not hooks
**Category:** Index UX
**Recommendation:** Rewrite each excerpt as 1–2 sentences stating a specific finding or framing a tension.

## [Severity: P2] — No "Back to all posts" link; breadcrumbs disabled
**Category:** Post UX
**Recommendation:** Set `breadcrumbs: true` in `_config.yml` (one line), or add static "← All posts" anchor in `single.html`.

## [Severity: P2] — No analytics
**Category:** Distribution
**Recommendation:** GoatCounter — 2 lines, no cookies, no consent banner.
**Reference:** https://www.goatcounter.com

## [Severity: P2] — Footer advertises AcademicPages template
**Category:** IA
**Recommendation:** Replace with `© 2026 Nilesh Patil` and keep only Jekyll attribution if desired.

## Article-level audit

### `2017-01-14-visualizing-and-comparing-distributions.md`
- Title "Visualizing distributions" — not search-indexable.
- `sns.distplot()` removed in seaborn 0.12 (2022). Code throws `AttributeError` today.
- `%pylab inline` deprecated in IPython 8+.
- `scolumns_order` typo (should be `columns_order`) — NameError.
- SQLite World Development Indicators dataset has no stable URL.

### `2017-02-15-human-activity-recognition.md`
- Adds no unique angle over what search already surfaces.
- Random Forest methodology valid.
- Breiman Random Forests page at stat.berkeley.edu now dead.
- HTTP links should be HTTPS.

### `2017-03-04-working-with-numpy.md`
- Title "Working with numpy" — most unindexable in the archive.
- All NumPy APIs stable and correct.
- 2-minute read. No mention of broadcasting (NumPy's most important confusing feature).

### `2017-03-14-transportation-graph-nyc-taxi-data.md`
- NYC TLC URL changed: correct is `https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page`.
- 2015 yellow-taxi-only is now historically interesting; TLC has since added HVFHV (Uber/Lyft).

### `2017-07-25-galactic-morphology-using-deep-learning.md`
- Most unfinished post in the archive. Ends mid-description of experimental setup. No results section, no accuracy figures, no conclusion.
- Reads like a draft submitted by accident.

### `2020-05-20-distributed-kmeans-clustering.md`
- `dask-ml` in maintenance mode (last major release 2022).
- `KMeans.partial_fit()` called but `dask_ml.cluster.KMeans` doesn't implement it. AttributeError at runtime.
- `find_elbow_point()` has dead `return k_range, inertias` after the actual return.
- `initMode="k-means||"` labeled as "k-means++ initialization" — mislabeled.

## The biggest single move
Publish **one 2,500-word opinionated post on AI deployment in 2025/2026** — something only you can write. Candidates:
- "What I learned building compliance-aware AI for SEBI-regulated investor workflows"
- "Agentic evaluation at 250M-user scale: what actually breaks in production"
- "The AI adoption playbook I wish I had when I started at Dream11"

This single action does six things simultaneously: updates the archive's last-modified signal from 2020 to 2026; signals current seniority; gives share buttons something worth sharing; seeds RSS with a subscriber-worthy entry; provides LinkedIn content for weeks; generates inbound that turns a personal site into a professional asset.

No amount of UX polish substitutes for this. Every other finding is downstream of having something worth reading.
