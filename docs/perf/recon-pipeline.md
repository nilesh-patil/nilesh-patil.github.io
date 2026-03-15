# Performance Recon: Asset & Include Pipeline

Captured pre-optimization. Source of truth for what the implementation
worktrees are allowed to touch.

## 1. Render-Blocking Head Assets

`_includes/head.html`:
- Line 11 — inline `<script>` swapping `no-js` → `js` class. ~100 bytes.
  Intentionally synchronous.
- Lines 16–33 — inline pre-paint theme script reading `localStorage`. ~600
  bytes. Intentionally synchronous (FOUC prevention).
- Line 37 — `<link rel="stylesheet" href="…/assets/css/main.css">`. Source
  `assets/css/main.scss` (1 010 B source; bundles all SCSS). Render-blocking,
  no `preload` hint.

`_includes/head/custom.html`:
- Line 6 — `<link rel="stylesheet" href="…/assets/css/academicons.css">`
  (9 207 B unminified; minified sibling exists at 7 768 B but is **not**
  the one referenced). Render-blocking, no `preload`.
- Lines 9–10 — two `<link rel="icon">` tags. Not render-blocking.

No `<link rel="preload">` hints anywhere in the head chain.

## 2. Eager JS Includes

`_includes/scripts.html` (end of `<body>`, every page):

| Tag | File | Size | defer/async |
| --- | --- | --- | --- |
| `<script type="module">` | `assets/js/main.min.js` | **4 623 344 B (4.4 MB)** | module = deferred by default |
| `<script>` | `assets/js/theme-cycle.js` | 4 102 B | none |
| `{% include analytics.html %}` | dispatches to providers | — | `site.analytics.provider` unset → dead branch on every page |
| `{% include comments-providers/scripts.html %}` | giscus/disqus/… | — | giscus case is **missing** from this dispatcher; giscus loads from `comments.html` only when `page.comments == true` |

`_includes/footer/custom.html` (every page):
- KaTeX CSS + two `<script defer>` tags, gated on
  `{% if page.math or layout.math %}`. Not loaded globally.
- KaTeX `auto-render` `onload` inline attribute — loads from
  `cdn.jsdelivr.net`.

`_includes/footer.html`: no script tags; only social-icon `<i>` markup
and copyright.

**Flag:** `main.min.js` at 4.4 MB ships on every page and imports
`theme.js` (13 899 B) which contains full Plotly dark and light theme
JSON blobs — loaded unconditionally even on pages with no Plotly charts.

## 3. Image Surface

- 34 files, ~23 MB total on disk.
- Top 5 largest:
  1. `images/blog/graphs/nycTaxiData/image6.png` — 7.3 MB
  2. `images/blog/graphs/nycTaxiData/image9.png` — 5.0 MB
  3. `images/blog/activityRecognition/image8.png` — 2.5 MB
  4. `images/blog/feature/galaxy.jpg` — 1.2 MB
  5. `images/blog/distributed-kmeans/clusters.jpeg` — 1.0 MB
- Avatar: `images/ensembledme.jpg` (referenced from `_config.yml:26`).

Post header images referenced as `overlay_image` / `teaser`:
- `images/blog/feature/nyc_network.png` (708 KB)
- `images/blog/feature/galaxy.jpg` (1.2 MB)
- `images/blog/distributed-kmeans/clusters.jpeg` (1.0 MB)

`<img>` tags missing one or more of `loading="lazy"`, `width`, `height`,
`decoding`:
- `_includes/author-profile.html:11,13` — has `fetchpriority="high"`, no
  others.
- `_includes/archive-single.html:19`
- `_includes/page__hero.html:55`
- `_includes/sidebar.html:9`
- `_includes/feature_row:23`
- `_includes/gallery:28,37`
- `_includes/comment.html:3` (Gravatar)
- `_includes/archive-single-cv.html:20`,
  `_includes/archive-single-talk.html:19`,
  `_includes/archive-single-talk-cv.html:20`

## 4. Font Surface

`assets/fonts/` (Academicons):

| File | Size |
| --- | --- |
| `academicons.eot` | 68 058 B |
| `academicons.svg` | 387 754 B |
| `academicons.ttf` | 67 872 B |
| `academicons.woff` | 131 616 B |

All four formats shipped; no `woff2` variant. Full icon set (200+ glyphs).

`assets/webfonts/` (FontAwesome):

| File | Size |
| --- | --- |
| `fa-brands-400.ttf` | 209 128 B |
| `fa-brands-400.woff2` | 117 852 B |
| `fa-regular-400.ttf` | 67 860 B |
| `fa-regular-400.woff2` | 25 392 B |
| `fa-solid-900.ttf` | 420 332 B |
| `fa-solid-900.woff2` | 156 400 B |
| `fa-v4compatibility.ttf` | 10 832 B |
| `fa-v4compatibility.woff2` | 4 792 B |

`main.scss` (lines 41–43) imports only `fontawesome`, `solid`, `brands`.
`regular` and `v4-shims` SCSS not imported, but their font binaries are
still shipped — dead weight. No subsetting.

## 5. Dead Code Candidates

**MathJax / Mermaid:** comment-only references in `_config.yml:213`
and `_includes/footer/custom.html:8-9`. No active script tags.

**Analytics providers:** `_config.yml` has no `analytics.provider`;
`_includes/analytics.html` is included from `scripts.html:12` but its
guard emits nothing. All four files under
`_includes/analytics-providers/` are dead.

**Comments providers:** active provider is `giscus`. The `giscus` case
is missing from `_includes/comments-providers/scripts.html` (lines
4–16). Dead alternatives: `disqus.html`, `discourse.html`,
`facebook.html`, `google-plus.html`, `staticman.html`.

**Orphaned `_includes/` (zero references in layouts/includes/pages):**

| Include | Refs |
| --- | --- |
| `archive-single-cv.html` | 0 (verify against `_layouts/cv-layout.html`) |
| `archive-single-talk-cv.html` | 0 (verify against `_layouts/cv-layout.html`) |
| `cv-template.html` | 0 |
| `paginator.html` | 0 |

## 6. Layout / Include Call Graph

`default.html`:
- `head.html` → `seo.html`
- `head/custom.html`, `browser-upgrade.html`, `masthead.html`
- `{{ content }}`
- `footer/custom.html`, `footer.html`
- `scripts.html` → `analytics.html`, `comments-providers/scripts.html`

`single.html` extends `default.html`, plus conditionally:
- `page__hero.html`, `breadcrumbs.html`
- `sidebar.html` → `author-profile.html`
- `read-time.html`, `toc`, `page__taxonomy.html`,
  `social-share.html`, `post_pagination.html`
- `comments.html` → `comments-providers/giscus.html`
- `archive-single.html`

`splash.html` extends `default.html` plus only `page__hero.html`
(no sidebar, no comments, no taxonomy, no toc).
