# Site Migration Spec — So Simple → academicpages

> **Status**: target state for `nilesh-patil.github.io`. This document describes **what "done" looks like**. For the phased execution checklist, see [`ROADMAP.md`](./ROADMAP.md).

---

## 1. Overview

The site at `nilesh-patil.github.io` currently runs a customized fork of **So Simple Theme** (Michael Rose / mmistakes), built around a 2017-era personal blog structure (Articles / Blog / Resume / Search / Tags). This migration moves the site to the **[academicpages](https://github.com/academicpages/academicpages.github.io)** theme — a fork of mmistakes' Minimal Mistakes tuned for academic/research portfolios — while preserving the 6 existing blog posts verbatim, refreshing the personal info from the current resume, and standing up new sections for Publications, Portfolio, and CV. Talks and Teaching sections are wired but hidden behind config flags for later activation. The site stays on `nilesh-patil.github.io` (no custom domain) and ships with **giscus** comments, **Pagefind** search, **KaTeX** math rendering, and a **system-pref-aware dark mode toggle**. No analytics, no public email.

### Decisions table

| Decision | Choice |
|---|---|
| Domain | **`nilesh-patil.github.io`** — no custom domain; GH Pages hosting. |
| Migration mode | **Clean replace + content port** — wipe theme files, pull academicpages skeleton, port content in. |
| Sections enabled | Blog, About, Publications, Portfolio, CV. Talks + Teaching **implemented but hidden** behind config flags. |
| Publications seed source | Manual seed from `resume/nilesh-patil.pdf` (6 concrete entries) + **Google Scholar auto-fetch** via `scholarly` Python lib against profile `IIabY1sAAAAJ`, with **arxiv API** as a supplementary source for Dream11-authored preprints. |
| URL strategy for posts | Move to `/posts/<slug>/`. Add `jekyll-redirect-from` so old `/blog/<slug>/` URLs 301 to the new locations. |
| Post hero image mapping | `image.feature` → `header.overlay_image` with caption + 0.5 dark overlay. |
| Publication schema | Full academic: title, authors, venue, date, paper_url, code_url, citation, abstract/excerpt, tags. |
| Portfolio scope | **Code projects + side projects** — GitHub repos, demos, open-source work. Each entry: image, problem, approach, repo link. Not Medium imports. Not prose case studies. |
| Visibility flag mechanism | `show_talks: false` / `show_teaching: false` in `_config.yml`, honored by `_includes/masthead.html`. |
| Site title | **"Nilesh Patil"** with subtitle "AI systems & applied research". Replaces "Data Curious". |
| About page | Rewritten from resume PROFILE block, expanded with a technical AI/ML skills section. |
| Comments | **giscus** on this site's repo (`nilesh-patil/nilesh-patil.github.io`, Discussions enabled). Disqus is dropped; old threads remain attached to old URLs but do not render under giscus. |
| Search | **Pagefind** (build-time index, browser-side queries). Replaces academicpages' default `lunr`. |
| Math rendering | **KaTeX** (smaller, synchronous render). Replaces academicpages' default MathJax. |
| Dark mode | **Enabled** — respects `prefers-color-scheme` with a manual toggle in the masthead. |
| Analytics | **Removed entirely** — UA-42632518-2 dropped, no GA4 replacement, no privacy-friendly counter, no tracking. |
| CV source of truth | **Web `/cv/` is canonical**; `files/nilesh-patil.pdf` is a manually-maintained snapshot. No auto-PDF generation in this migration. |
| Public email | **Not displayed** — contact via LinkedIn + GitHub only. No `mailto:` in author profile. |
| Author photo | `images/ensembledme.jpg` — kept as-is. |
| Visual customization | **None** — academicpages stock Sass tokens. No accent color override, no font override. |

---

## 2. Source / target theme baseline

| | Current (So Simple) | Target (academicpages) |
|---|---|---|
| Theme | So Simple Theme by mmistakes | academicpages (fork of Minimal Mistakes 4.x by mmistakes) |
| Build | Jekyll + Grunt (JS/image minification) | Jekyll-native Sass; no Grunt |
| Plugins | jekyll-sitemap, jekyll-gist, jekyll-feed | jekyll-feed, jekyll-include-cache, jekyll-paginate, jekyll-sitemap, jekyll-gist, jemoji, **jekyll-redirect-from** |
| Sass | Bourbon / Neat grid | Minimal Mistakes Sass tokens |
| Layouts | `post.html`, `page.html` (2 files) | `single`, `default`, `archive`, `archive-taxonomy`, `splash`, `talk`, `cv-layout`, `compress` (8 files) |
| Includes | 9 includes (head, nav, footer, scripts, disqus, share, feed-footer, open-graph, browser-upgrade) | ~30 includes (masthead, author-profile, sidebar, breadcrumbs, comments, analytics, gallery, paginator, etc.) |

Both themes share the **same author (Michael Rose)** — Sass tokens and Liquid include patterns are sibling-compatible, which reduces migration risk.

---

## 3. Information architecture (target)

Navigation order, defined in `_data/navigation.yml`:

1. About → `/about/`
2. Publications → `/publications/`
3. Portfolio → `/portfolio/` *(replaces current "Articles" section)*
4. Blog → `/posts/` *(academicpages convention)*
5. CV → `/cv/`
6. *(Talks → `/talks/`)* — visible only when `show_talks: true`
7. *(Teaching → `/teaching/`)* — visible only when `show_teaching: true`

Footer links: GitHub, Medium, LinkedIn, Google Scholar. **No email link** (decision in §1).

Homepage (`/`) renders the academicpages default landing — author profile card on the left, recent posts on the right.

---

## 4. File inventory

### Removed (clean-replace targets)
- `_layouts/post.html`, `_layouts/page.html` — So Simple layouts.
- `_includes/*` — all 9 So Simple includes.
- `_sass/` — entire So Simple Sass tree.
- `assets/css/`, `assets/js/` — So Simple built assets.
- `Gruntfile.js`, `package.json`, `.jshintrc` — Grunt pipeline (academicpages doesn't use it).
- Directory-style index pages: `about/`, `articles/`, `blog/`, `resume/`, `search/`, `tags/` → replaced by `_pages/` flat files.
- `search.json`, `feed.xml` — theme provides these.
- `index.html` — academicpages provides root layout.

### Kept verbatim
- `_posts/blog/*.md` — **body content untouched**. Only frontmatter is rewritten (see Section 5).
- `images/**` — all hero and inline images. Paths preserved so post body image refs keep working.
- `resume/nilesh-patil.pdf` — **relocated** to `files/nilesh-patil.pdf` (academicpages convention).
- `favicon.ico`, `favicon.png`.
- `LICENSE`, `.gitignore`, `.editorconfig`.

### Added (from academicpages skeleton)
- `_layouts/{default,single,archive,archive-taxonomy,splash,talk,cv-layout,compress}.html`
- `_includes/` — full set (~30 files).
- `_sass/` — academicpages Sass tree.
- `assets/css/main.scss`, `assets/js/{main.min.js,_main.js,plugins/,vendor/}`
- `_pages/` — `about.md`, `cv.md`, `publications.html`, `portfolio.html`, `talks.html`, `teaching.html`, `year-archive.html`, `category-archive.html`, `tag-archive.html`, `404.md`.
- `_publications/` — seeded entries (6 from resume + Google Scholar + arxiv auto-fetch output).
- `_portfolio/`, `_talks/`, `_teaching/` — collection directories (latter two hidden by flags).
- `_data/{authors.yml,navigation.yml,ui-text.yml}` — rewritten for new schema.
- `_config.yml` — full rewrite (see Section 6).
- `_sass/_dark.scss` — dark-mode token overrides (see §11b).
- `files/` — downloadable assets (resume PDF, future slides).
- `scripts/` — `fetch_publications.py`, `dream11_authors.txt`, `requirements.txt` (see §8).
- `markdown_generator/` — academicpages utility for bulk publication import (kept for future).

### Added at build time (not committed)
- `_site/pagefind/` — Pagefind search index, emitted by `npx pagefind --site _site` after Jekyll builds. Wired into the GH Pages build step (see §12 and ROADMAP Phase 8).

---

## 5. Frontmatter migration mapping (blog posts)

| Current key (So Simple) | New key (academicpages) | Notes |
|---|---|---|
| `layout: post` | `layout: single` | academicpages uses `single` for posts. |
| `title: "..."` | `title: "..."` | unchanged |
| `date: 2020-05-20T...` | `date: 2020-05-20T...` | unchanged |
| `modified: ...` | `last_modified_at: ...` | rename only |
| `categories: blog` | `categories: [blog]` | normalize to array |
| `tags: [a, b]` | `tags: [a, b]` | unchanged |
| `excerpt: '...'` | `excerpt: '...'` | unchanged |
| `image.feature: blog/.../foo.jpg` | `header.overlay_image: /images/blog/.../foo.jpg` + `header.overlay_filter: 0.5` + `header.caption: "<auto>"` | path becomes absolute (`/images/...`); 0.5 dark overlay improves text legibility over photos |
| `image.credit` / `image.creditlink` | `header.caption: "Photo by [credit](creditlink)"` | merged into a single Markdown caption |
| `comments: true` | (drop) | controlled site-wide via `_config.yml: comments.provider: giscus` |
| (none) | `author_profile: true` | sidebar with author info on each post |
| (none) | `toc: true`, `toc_sticky: true` | enables in-post table of contents |
| (none) | `read_time: true` | adds read-time estimate |
| (none) | `share: true` | enables social share buttons |
| (none) | `redirect_from: [/blog/<old-slug>/]` | preserves old URL via 301 |
| (none) | `math: true` | loads KaTeX on the page; set only for posts that use math (e.g., the K-means post) |

**Note on comment continuity**: Migrating to giscus drops Disqus thread continuity. Old Disqus comments remain attached to the old `/blog/<slug>/` URL paths (which still 301 to new ones) but won't render in the new giscus widget. `disqus_identifier` is **not** added to migrated frontmatter — it would have no consumer.

**Filename change required**: `2017-01-14-visualizing-&-comparing-distributions.md` contains `&` — rename to `2017-01-14-visualizing-and-comparing-distributions.md` (Jekyll-safe slug). The `redirect_from` entry preserves the old URL.

---

## 6. `_config.yml` target shape

Full rewrite. Key blocks:

```yaml
# Site
title: "Nilesh Patil"
subtitle: "AI systems & applied research"
name: "Nilesh Patil"
description: >-
  AI leader building deployable AI systems, agentic workflows, and
  organizational adoption in regulated and large-scale environments.
url: "https://nilesh-patil.github.io"
baseurl: ""
repository: "nilesh-patil/nilesh-patil.github.io"

# Author (rendered in sidebar via author_profile)
# NOTE: email omitted intentionally — contact is via LinkedIn / GitHub only.
author:
  name: "Nilesh Patil"
  avatar: "/images/ensembledme.jpg"
  bio: "Head of AI at DreamStreet. Previously Head of Applied Research, Dream11."
  location: "Mumbai"
  employer: "DreamStreet"
  github: "nilesh-patil"
  twitter: "ensembledme"
  medium: "https://medium.com/@ensembledme"
  linkedin: ""              # to be filled
  googlescholar: "https://scholar.google.co.in/citations?user=IIabY1sAAAAJ"
  stackoverflow: "https://stats.stackexchange.com/users/36581/nilesh"

# Build
markdown: kramdown
highlighter: rouge
lsi: false
excerpt_separator: "\n\n"
incremental: false

# Plugins
plugins:
  - jekyll-feed
  - jekyll-include-cache
  - jekyll-paginate
  - jekyll-sitemap
  - jekyll-gist
  - jemoji
  - jekyll-redirect-from

# Collections
collections:
  posts:
    output: true
    permalink: /:collection/:path/     # → /posts/<slug>/
  publications:
    output: true
    permalink: /:collection/:path/
  portfolio:
    output: true
    permalink: /:collection/:path/
  talks:
    output: true
    permalink: /:collection/:path/
  teaching:
    output: true
    permalink: /:collection/:path/

# Defaults (per scope)
defaults:
  - scope: { path: "", type: posts }
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: true
      share: true
      related: true
      toc: true
      toc_sticky: true
  - scope: { path: "", type: publications }
    values: { layout: single, author_profile: true, share: false }
  - scope: { path: "", type: portfolio }
    values: { layout: single, author_profile: true, share: true }
  - scope: { path: "", type: talks }
    values: { layout: talk, author_profile: true }
  - scope: { path: "", type: teaching }
    values: { layout: single, author_profile: true }
  - scope: { path: "_pages" }
    values: { layout: single, author_profile: true }

# Comments — giscus (GitHub Discussions)
# academicpages stock supports disqus/discourse/staticman/utterances/giscus.
comments:
  provider: giscus
  giscus:
    repo: "nilesh-patil/nilesh-patil.github.io"
    repo_id: ""                  # filled from giscus.app config wizard
    category: "General"
    category_id: ""              # filled from giscus.app config wizard
    discussion_term: "pathname"  # threads keyed by URL path
    reactions_enabled: "1"
    theme: "preferred_color_scheme"  # follows site light/dark toggle

# Search — Pagefind (build-time, not a Jekyll plugin)
# Academicpages' built-in `search_provider: lunr` is disabled.
# Pagefind runs as a post-build step (see Section 12) and emits /pagefind/ assets.
search: false

# Math — KaTeX
# Loaded via _includes/scripts.html (added during migration).
# academicpages ships MathJax by default; we replace with KaTeX.
math_engine: katex

# Theme — dark mode
# academicpages exposes a built-in `minimal_mistakes_skin`. We layer on a
# system-preference-aware toggle in the masthead. See Section 11b.
skin: "default"
dark_mode: true                # custom flag, honored by _includes/head.html

# Custom feature flags (not part of academicpages stock)
show_talks: false
show_teaching: false

# Analytics: OMITTED — no tracking
```

**Note**: no `analytics:` block at all. The old UA-42632518-2 is deprecated (stopped collecting data in July 2023) and is dropped without replacement. No GA4, no Plausible, no GoatCounter, no Cloudflare Web Analytics — site ships with zero telemetry.

**giscus identifiers**: `repo_id` and `category_id` are filled by running [giscus.app](https://giscus.app) once after Discussions is enabled on the repo. The config wizard outputs both IDs.

---

## 7. URL / permalink scheme

| Resource | URL pattern |
|---|---|
| Post | `/posts/<slug>/` |
| Publication | `/publications/<slug>/` |
| Portfolio entry | `/portfolio/<slug>/` |
| Talk *(when flag on)* | `/talks/<slug>/` |
| Teaching entry *(when flag on)* | `/teaching/<slug>/` |
| About | `/about/` |
| CV | `/cv/` |
| Publications index | `/publications/` |
| Portfolio index | `/portfolio/` |
| Year archive | `/year-archive/` |
| Category archive | `/category-archive/` |
| Tag archive | `/tag-archive/` |

### Backwards compatibility
- `jekyll-redirect-from` is enabled in `_config.yml` plugins.
- Each migrated post sets `redirect_from: [/blog/<old-slug>/]` in frontmatter.
- Result: old `/blog/<slug>/` URLs return a 301 redirect to the new `/posts/<slug>/` URL.
- Preserves inbound links from Medium, Stack Exchange, search engines, citations.

---

## 8. Publications collection

### Schema (per `_publications/<slug>.md`)

```yaml
---
title: "..."
collection: publications
permalink: /publications/<slug>/
date: YYYY-MM-DD
venue: "..."                  # e.g., "ICMLA 2023", "ACL 2026", "CODS-COMAD '24"
paper_url: "..."              # arxiv / DOI / publisher link
code_url: "..."               # optional, GitHub repo
authors: "A, B, C, ..."       # comma-separated; Nilesh bolded in rendered citation
citation: "<formatted citation string or BibTeX>"
excerpt: "1–2 sentence abstract"
tags: [machine-learning, ...]
---
Body: full abstract, key figures (optional), takeaways.
```

### Seed entries (from `resume/nilesh-patil.pdf`)

**Dream11 era (2019–2026)**:

1. **Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts** — 2026, Association for Computational Linguistics, USA. <https://openreview.net/forum?id=rLisRb1T1Y>
2. **Early Churn Prediction from Large Scale User-Product Interaction Time Series** — 2023 International Conference on Machine Learning and Applications (ICMLA), IEEE, 2023. <https://doi.org/10.1109/ICMLA58977.2023.00314>
3. **Optimizing Fantasy Sports Team Selection with Deep Reinforcement Learning** — Proceedings of CODS-COMAD '24, ACM, 284–291. <https://doi.org/10.1145/3703323.3703743>
4. *(Placeholder — to be resolved by arxiv auto-fetch)* "6+ additional team publications in causal ML, recommender systems, and LLM applications."

**Rochester era (2016–2019)**:

5. **Automated Ultrasound Doppler Angle Estimation Using Deep Learning** — 2019 41st Annual Int'l Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 28–31, IEEE, 2019. <https://pubmed.ncbi.nlm.nih.gov/31945837>
6. **CXCL10+ Perivascular Clusters Nucleate Th1 Cell Tissue Entry and Activation in the Inflamed Skin** — Journal of Immunology. <https://www.jimmunol.org/content/204/1_Supplement/220.9>
7. **CXCL10+ Peripheral Activation Niches Couple Preferred Sites of Th1 Entry with Optimal APC Encounter** — Cell Reports. <https://www.biorxiv.org/content/10.1101/2020.10.04.324525v1.full>

### Auto-fetch script (`scripts/fetch_publications.py`)

Two-source pipeline. **Google Scholar is the primary source** (covers the full publication graph including J Immunol, Cell Reports, EMBC, ACL, ACM, etc.); **arxiv is supplementary** for Dream11 preprints that may not yet appear on Scholar.

**Source 1 — Google Scholar via `scholarly`**:
- Scrapes [scholar.google.co.in/citations?user=IIabY1sAAAAJ](https://scholar.google.co.in/citations?user=IIabY1sAAAAJ&hl=en).
- For each publication: fetches title, authors, venue, year, citation count, abstract (where Scholar exposes it), and the publisher/arxiv URL.
- Output: one `_publications/<slug>.md` per hit with prefilled frontmatter (`paper_url` = best available link, `excerpt` = abstract or first-line teaser, `authors` = full list, `venue` = Scholar's `journal` / `conference` / `book` field).
- Scholar slug rule: lowercased title, dashed, truncated to first ~60 chars.

**Source 2 — arxiv API (supplementary)**:
- Queries arxiv for `au:"Nilesh Patil"`, plus any co-author names in `scripts/dream11_authors.txt` (one per line, optional).
- Used to surface Dream11 preprints not yet indexed by Scholar.
- Deduped against Scholar output by title-similarity (lowercased, punctuation-stripped, ≥ 0.85 Jaccard).

**Behavior**:
- **Idempotent**: skips files that already exist.
- **Re-runnable**: `python scripts/fetch_publications.py --refresh` updates `excerpt`, `authors`, `venue`, `paper_url`, and citation counts for existing files **without overwriting** manual edits to `citation`, `code_url`, body content, or `tags`. Implemented by reading existing frontmatter, merging only the auto-managed keys, and rewriting.
- **Rate-limit caveats**: Scholar throttles. Recommended run cadence: a few times per year, on publish. If blocked, the script logs and exits cleanly without partial writes.
- **DOM-break caveat**: `scholarly` scrapes; Scholar changes its DOM ~every 6 months. Pin `scholarly` to a known-good version in `requirements.txt`; bump on breakage.

**Dependencies** (in `scripts/requirements.txt`):
```
scholarly==1.7.11      # pin — DOM-scraper, breaks otherwise
arxiv==2.1.0
python-frontmatter==1.0.1
```

**Author bolding**: rendered citation bolds "Nilesh Patil" via a Liquid filter in `_layouts/single.html` for the publications layout. Pattern matches case-insensitively to handle "N. Patil" / "Nilesh Patil" / "Patil, N." variants.

---

## 9. Portfolio collection

Replaces the current empty `/articles/` section. Schema:

```yaml
---
title: "..."
collection: portfolio
permalink: /portfolio/<slug>/
date: YYYY-MM-DD
excerpt: "..."
header:
  teaser: "/images/portfolio/<image>.jpg"
tags: [...]
---
Body: project description, screenshots, links.
```

**Scope**: code projects and side projects only. Each entry should have:
- A GitHub repo (or other public code link) — entries without code don't belong here.
- One hero image (`header.teaser`) — screenshot, diagram, or representative output.
- A short problem statement (1–2 sentences).
- The approach in 1–3 paragraphs.
- Links: repo, demo (if any), related blog post (if any).

**Explicitly out of scope** for the portfolio collection:
- Medium article imports (deferred — see §14).
- Prose-only case studies (those belong as blog posts).
- Talks / slide decks (those go to `_talks/` when `show_talks` flips on).

Initial seed: 1–2 entries from existing GitHub repos (e.g. distributed K-means demo if it has a standalone repo) so the layout previews aren't broken.

---

## 10. CV and About pages

### `_pages/cv.md` — source of truth

The **web `/cv/` page is canonical**. The PDF at `files/nilesh-patil.pdf` is a manually-maintained snapshot; **no auto-PDF generation is wired in this migration** (no headless Chromium, no Paged.js, no weasyprint). Drift management is by hand: when you update the web CV, you also export an updated PDF (browser print-to-PDF is acceptable). This trade-off was chosen to keep the migration scope tight; auto-PDF can be added as a follow-up PR.

Uses academicpages' `cv-layout`. Structured Markdown sections:
- **Download**: link to `/files/nilesh-patil.pdf` at top. Caveat note: "Web version above is authoritative; PDF may lag by a few weeks."
- **Education**: M.S. Data Science, University of Rochester (2017); B.Tech, IIT Roorkee (2013).
- **Experience**: DreamStreet (2026–present), Dream11 (2019–2026), University of Rochester / Center for Vaccine Biology (2017–2019), AXA Insurance (2014–2016), AbsolutData (2013–2014). Bullet points ported from PDF.
- **Selected Publications**: inline list of the 7 entries from Section 8 (ungated by `show_talks` / `show_teaching`).
- **Service / Talks**: section present, populated only when `show_talks: true`.
- **Teaching**: section present, populated only when `show_teaching: true`.
- **Contact**: links to GitHub + LinkedIn. **No email** (decision in §1 table).

### `_pages/about.md`
Rewritten from resume PROFILE, expanded to be more technical:

- **Lead paragraph** (verbatim from resume): *"AI leader building deployable AI systems, agentic workflows, and organizational adoption in regulated and large-scale environments..."*
- **Current role**: Head of AI at DreamStreet — compliance-aware AI architecture for SEBI-regulated investor and trader workflows.
- **Technical skills section** (new — explicit AI/ML systems work):
  - LLM-based behavior simulation, persona simulators, agentic evaluators for personalization.
  - Distributed recommendation, content tagging, text similarity search at ~100M-entity scale.
  - Feature-store systems supporting 250M+ users.
  - Real-time forecasting (~50k+ forecasts) under strict latency constraints.
  - Deep-learning churn prediction.
  - Self-hosted SLMs and agent tooling (local, GCP, AWS).
  - Compliance-aware AI harness design (audit-friendly workflows).
- **Closing**: current interests — AI harness design, developer productivity, turning emerging model capabilities into reliable workflows and products.

---

## 11. Visibility flag wiring

Two custom flags in `_config.yml`:
```yaml
show_talks: false
show_teaching: false
```

Honored in `_includes/masthead.html` via a Liquid conditional:

```liquid
{% for link in site.data.navigation.main %}
  {% if link.requires_flag %}
    {% if site[link.requires_flag] %}
      <li><a href="{{ link.url }}">{{ link.title }}</a></li>
    {% endif %}
  {% else %}
    <li><a href="{{ link.url }}">{{ link.title }}</a></li>
  {% endif %}
{% endfor %}
```

`_data/navigation.yml`:
```yaml
main:
  - title: About
    url: /about/
  - title: Publications
    url: /publications/
  - title: Portfolio
    url: /portfolio/
  - title: Blog
    url: /posts/
  - title: CV
    url: /cv/
  - title: Talks
    url: /talks/
    requires_flag: show_talks
  - title: Teaching
    url: /teaching/
    requires_flag: show_teaching
```

**Behavior**: collections still build and pages exist at their URLs (direct access works for previewing), but they're invisible in nav until the flag flips.

---

## 11b. Dark mode

Custom flag in `_config.yml`:
```yaml
dark_mode: true
```

Wiring:

1. **Theme tokens**: academicpages exposes skin Sass variables. We add a `_sass/_dark.scss` partial that redefines `$background-color`, `$text-color`, `$link-color`, `$border-color`, `$code-background-color` for `[data-theme="dark"]`. Imported at the end of `assets/css/main.scss`.

2. **Pre-paint script** in `_includes/head.html` (runs before body renders to prevent FOUC):
   ```html
   <script>
     (function () {
       var stored = localStorage.getItem('theme');
       var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
       var theme = stored || (prefersDark ? 'dark' : 'light');
       document.documentElement.setAttribute('data-theme', theme);
     })();
   </script>
   ```

3. **Toggle button** in `_includes/masthead.html`:
   ```html
   <button id="theme-toggle" aria-label="Toggle dark mode">🌓</button>
   <script>
     document.getElementById('theme-toggle').addEventListener('click', function () {
       var current = document.documentElement.getAttribute('data-theme');
       var next = current === 'dark' ? 'light' : 'dark';
       document.documentElement.setAttribute('data-theme', next);
       localStorage.setItem('theme', next);
     });
   </script>
   ```

4. **giscus theme sync**: giscus `theme: "preferred_color_scheme"` honors the same `prefers-color-scheme` media query the pre-paint script reads. When the user toggles manually, a `postMessage` to the giscus iframe re-themes the embed (added to the toggle handler).

**Behavior**: first visit follows OS preference; toggle persists across visits via `localStorage`; clearing site data resets to OS preference.

---

## 12. Comments, analytics, search, math

| | Setting |
|---|---|
| **Comments** | **giscus** backed by GitHub Discussions on `nilesh-patil/nilesh-patil.github.io`. Threads keyed by `pathname`. Reactions enabled. Theme follows site dark-mode toggle via `preferred_color_scheme`. Commenters must have a GitHub account — acceptable filter for a technical blog. **Old Disqus threads are not migrated** — they remain attached to the old `/blog/<slug>/` URLs (which still 301 to new ones), but won't render under giscus. Disqus shortname `dataCurious` is retired. |
| **Analytics** | None. Block omitted entirely from `_config.yml`. No tracking scripts injected. `_includes/analytics-providers/` left dormant or removed. No GA, no Plausible, no GoatCounter, no Cloudflare Web Analytics. Site has zero outbound telemetry. |
| **Search** | **Pagefind** — build-time index, browser-side queries. academicpages' default `lunr` is disabled. Index covers posts + publications + portfolio. Runs as a post-build step: `npx pagefind --site _site`. Output lives at `_site/pagefind/`. Wired into `_includes/search/search_box.html` as: <br/> `<link rel="stylesheet" href="/pagefind/pagefind-ui.css">` <br/> `<script src="/pagefind/pagefind-ui.js"></script>` <br/> `<div id="search"></div>` <br/> `<script>new PagefindUI({ element: "#search" })</script>` |
| **Math** | **KaTeX** (~280KB, synchronous render). Loaded in `_includes/scripts.html` only on pages where frontmatter sets `math: true`, or in `defaults` for posts/publications. Replaces MathJax. Auto-render delimiters: `$...$`, `$$...$$`, `\\(...\\)`, `\\[...\\]`. |

---

## 13. Verification checklist

When all of these pass, the migration is done:

- [ ] `bundle exec jekyll serve` builds with no errors / warnings.
- [ ] `npx pagefind --site _site` succeeds and emits `_site/pagefind/`.
- [ ] All 6 blog posts render at `/posts/<slug>/` with hero overlay image, TOC, share buttons, giscus widget.
- [ ] Inline body images (under `/images/blog/...`) resolve on every post.
- [ ] Old `/blog/<slug>/` URLs 301-redirect to new locations.
- [ ] giscus widget loads on posts, accepts a test reaction or comment from a logged-in GitHub user, and writes to a GitHub Discussion under `nilesh-patil/nilesh-patil.github.io`.
- [ ] Navigation shows About / Publications / Portfolio / Blog / CV (Talks + Teaching hidden).
- [ ] Flipping `show_talks: true` → Talks appears in nav and page renders. Same for `show_teaching`.
- [ ] `/publications/` lists all seed entries with working `paper_url` links and rendered citations. "Nilesh Patil" is bolded in author lists.
- [ ] `/cv/` renders all sections; resume PDF download link works. No `mailto:` link in author profile or CV contact section.
- [ ] **KaTeX** renders inline + display math on posts with `math: true` (verify on the K-means post).
- [ ] Code blocks render with Rouge syntax highlighting.
- [ ] **Dark mode**: toggle in masthead flips theme; preference persists across reloads; first-visit follows `prefers-color-scheme`; no flash of unstyled content on load.
- [ ] **giscus theme** matches site theme — light when light, dark when dark.
- [ ] **Pagefind search**: search box returns hits from posts + publications + portfolio; no external network call on query.
- [ ] **No analytics / gtag / GA / Plausible / GoatCounter requests** in browser Network tab. Zero outbound telemetry.
- [ ] No public email address rendered anywhere on the site (grep `_site` for `mailto:` → only empty / social-icon stubs).
- [ ] Site builds successfully on GitHub Pages (Action includes Pagefind step).
- [ ] Mobile viewport (375px) renders cleanly in both light + dark.

---

## 14. Out of scope

- Migrating old Disqus comment threads into giscus (not technically possible — different storage backends; threads stay attached to old Disqus URLs and effectively retire).
- Auto-PDF generation for `/cv/` (Paged.js / weasyprint / wkhtmltopdf). Web CV is canonical; PDF is updated manually until this is added as a follow-up PR.
- Adding a Medium-import script.
- Custom theme color / typography overrides (use academicpages stock).
- Multi-language support.
- Newsletter / email subscription forms.
- Privacy-friendly analytics (Plausible / GoatCounter / Cloudflare). Decision is zero telemetry, full stop.
- Custom domain (`nilesh-patil.com` or similar) — staying on `nilesh-patil.github.io`.

---

## 15. Remaining open question

**Curated Dream11 co-author list (`scripts/dream11_authors.txt`)** — optional, non-blocking. The Google Scholar fetch already covers most of the publication graph; the arxiv supplementary fetch with co-author names is a wider net for Dream11 *preprints* not yet on Scholar. This file can be added or edited at any time post-migration without re-doing the migration itself.
