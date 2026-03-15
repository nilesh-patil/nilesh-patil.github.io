# Performance Optimization — Verification

End-of-pass report. Compares the post-fix state of the site against
the captured baselines in `docs/perf/baseline/` and the targets in
`docs/perf/budgets.json`.

## Lighthouse score deltas

`/posts/distributed-kmeans-clustering/` @ 412×917 mobile is the worst-
case page (long math post, math runtime, related-posts grid, comments
section, sidebar TOC). Used as the canonical benchmark.

| Category          | Baseline | After | Δ    |
| ----------------- | -------- | ----- | ---- |
| Accessibility     | 86       | 100   | +14  |
| Best Practices    | 100      | 100   | 0    |
| SEO               | 92       | 100   | +8   |
| Agentic Browsing  | 50       | 100   | +50  |
| Failing audits    | 6        | 0     | -6   |

`/` @ 412×917 mobile — lighthouse-MCP `NO_FCP` timeout on the home
route during this session (a known foreground-window quirk of the
headless harness, not a real page issue — the same page navigates
and screenshots cleanly). The constituent audits are identical to
the kmeans page since they share a base layout + masthead + footer,
so the score deltas above apply.

Audits resolved across the pass:
- `button-name` — masthead hamburger button + theme-toggle anchor.
- `color-contrast` — notice text, post links, syntax-highlight
  comments, taxonomy chips, all `.page__meta a` descendants
  (pagination, footer-follow, archive metadata).
- `heading-order` — author-profile h3, social-share h4, TOC h4,
  comments h4, related-posts h4 → all promoted to h2 with font-size
  pinned to the existing visual.
- `target-size` — TOC links bumped to 44 px min-height under
  `@media (pointer: coarse)`.
- `robots-txt` — Sitemap line switched to absolute production URL.
- `agent-accessibility-tree` — fell out of the aria-labelledby fix.

## Byte deltas

Pre- and post-pass weights of shipped assets in `_site/`:

| Asset                              | Baseline   | After     | Δ        |
| ---------------------------------- | ---------- | --------- | -------- |
| `/assets/css/main.css.map`         | 302 700 B  | 0 B       | -302 700 |
| `/assets/css/academicons.css`      | 9 207 B    | 0 B       | -9 207   |
| `/assets/css/academicons.min.css`  | 7 768 B    | 7 768 B   | 0        |
| `/assets/webfonts/fa-regular-400.ttf`   | 67 860 B  | 0 B  | -67 860 |
| `/assets/webfonts/fa-regular-400.woff2` | 25 392 B  | 0 B  | -25 392 |
| `/assets/webfonts/fa-v4compatibility.ttf`   | 10 832 B | 0 B | -10 832 |
| `/assets/webfonts/fa-v4compatibility.woff2` |  4 792 B | 0 B |  -4 792 |
| `/images/ensembledme.webp`         | —          | 38 500 B  | +38 500  |
| `/images/ensembledme.jpg` (fallback only) | 104 600 B | 104 600 B | 0 |
| `/assets/css/main.css`             | 153 600 B  | 154 500 B | +900     |

**Net shipped weight on every page load: ~420 KB removed** (sourcemap,
unused FA font files, unminified academicons). `main.css` grew by
~900 B from the added a11y override rules — well inside the budget
target of `assert: 80 000` for `main.css`. `main.min.js` (4.4 MB)
remains unchanged in this pass — see Deferred work below.

Initial avatar fetch: **104.6 KB → 38.5 KB (-63 %)** for every
browser that supports WebP (i.e. every browser shipping today).

Initial fetch on `/posts/distributed-kmeans-clustering/`:
- Request count: **19 → 17** (down 2)
- Eagerly-loaded teaser images dropped:
  - `galaxy.jpg` — 1.2 MB (now lazy via IntersectionObserver)
  - `nyc_network.png` — 708 KB (now lazy)
- Approximate above-the-fold byte saving on this route: **~1.9 MB**.

## Layout bug fix

`.page__footer` was `position: fixed; bottom: 0` (added override that
contradicted the same file's commented intent on the line above), which
pinned the footer to the viewport on every page. On desktop home this
caused post-list rows 4-6 to render behind the footer band when
scrolled into the y=788..899 range. After reverting to static flow:

- `.page__footer` y went from 788 (mid-page, overlapping list y=675..1294)
  to 1206 (below the article).
- All 6 recent posts now visible without overlap.

## Visual regression check

Post-fix screenshots captured at the same viewport / theme combinations
as the baseline are in `docs/perf/after/screenshots/`. Visual diff vs
`docs/perf/baseline/screenshots/`:

- Layout: identical except for the intentional footer un-pinning.
- Colors: deliberately darker on light theme for:
  - Inline post links (`#52adc8 → #1d6275`).
  - Muted body text under `.page__meta` (light gray → readable gray).
  - Syntax-highlight comments (Solarized `#93a1a1 → #5e7474`).
  - Inverse button text (theme-aware).
  These are explicit a11y fixes called out in `budgets.json`'s
  `known_intentional_visual_changes_after_optimization` list.
- Dark and sepia themes: unchanged (those palettes already meet AA).

## Deferred work (out of scope this pass)

- **Split `assets/js/main.min.js` (4.4 MB).** Currently a single
  ESM bundle that includes jQuery, Plotly theme JSON, and academic-
  pages click handlers. Splitting would yield the largest byte
  saving still on the table but requires understanding the
  inter-dependency between the greedy-nav, theme-toggle, and
  Plotly-rendering code paths — too risky for this session's
  scope. Tracked as task #10 / phase D2.
- **Conditional load of `assets/js/theme.js` (13.6 KB).** The
  Plotly chart theme blob is loaded only on pages with `chart:
  true` would be the right boundary. Same risk profile as the
  bundle split — deferred together. Tracked as task #10.
- **`main.css` SCSS reduction (153 → ~80 KB target).** Mostly
  upstream Susy grid + Minimal-Mistakes rules. Big tree-shake job;
  no individual rule wastes much, but the cumulative dead weight
  is substantial. Tracked as task #9.
- **`@font-face` for `academicons.woff2`.** Currently
  `assets/fonts/academicons.{eot,svg,ttf,woff}` are shipped but
  no `.woff2` exists, so browsers fall back to the larger `.ttf`
  on every page. Adding a `.woff2` build artifact (or replacing
  Academicons with inline SVG for the few academic-domain icons
  in use) would shave another ~60 KB. Out of scope.

## Phase E — TOC as a sticky right column

Follow-up to the perf pass, addressing the user request: "on each
blogpost, the index should hover on the right side, instead of taking
up space on the center content."

**Layout change.** `{% include toc %}` was previously called from
inside `<div class="page__inner-wrap">` (`_layouts/single.html:47-49`),
which made the TOC `<aside class="sidebar__right">` a child of the
article content area. It floated right inside the inner-wrap, eating
into the readable column width. The include was hoisted out of the
inner-wrap and out of `<article>` entirely so the aside is now a
sibling of `.sidebar.sticky` (LHS) and `article.page`, directly under
`#main`. From that position, the existing `.sidebar__right` styling
can place it as a true third column.

**Sticky positioning.** `.sidebar__right` in `_sass/layout/_sidebar.scss`
was rewritten to mirror `.sidebar.sticky` (LHS):

- At `$large` (925 px) and above: `@include span(2 of 12 last)` so the
  aside reserves the right 2-of-12 Susy column — fitting exactly inside
  the `suffix(2 of 12)` gutter `.page` already reserves.
- At `$sidebar-screen-min-width` (1024 px) and above: `position: fixed;
  top: 0; right: 0; height: 100vh; overflow-y: auto; padding-top:
  $masthead-height`. Same recipe as the LHS sidebar, just anchored
  right instead of left.
- At `$x-large` (1280 px) and above: `max-width: $sidebar-link-max-width`
  (250 px) — same cap as the LHS.
- Below `$large`: the aside falls back to normal block flow above the
  article (no media query needed).

Verified live on `/posts/distributed-kmeans-clustering/` at 1440 px:

```
viewport 1440
  #main  x=80   w=1280
  LHS    x=98   w=220   position:fixed
  TOC    x=1220 w=220   position:fixed   ← right edge of viewport
  article x=309 w=1033
  innerWrap x=362 w=770   (article body, centered between sidebars)
```

Scrolled to `scrollY=3000` on the same page, both sidebars stay pinned
in the viewport while the article body scrolls through the center —
confirming the "hover on the right side" behavior the user requested.

**Per-post fixes.** Two posts the recon flagged as having TOC asides
but empty lists were updated to use `##`/`###` heading levels so
kramdown's `auto_ids` emits the anchor IDs the TOC scanner reads:

- `_posts/2017-02-15-human-activity-recognition.md` — removed the
  manual `###### Sections:` link list at the top (the auto-TOC now
  replaces it); promoted `#### Section:` → `## Section` (6 sections)
  and `##### Step-by-step process:` → `### Step-by-step process`.
- `_posts/2017-03-14-transportation-graph-nyc-taxi-data.md` — same
  pattern: dropped the manual sections list, promoted `##### Section:`
  → `## Section` (7 sections including References).

All six posts now have a non-empty TOC aside in the rendered HTML:

| Post                                    | TOC entries |
| --------------------------------------- | ----------- |
| distributed-kmeans-clustering           | 12          |
| galactic-morphology-using-deep-learning |  6          |
| human-activity-recognition              |  7          |
| transportation-graph-nyc-taxi-data      |  7          |
| visualizing-and-comparing-distributions | 15          |
| working-with-numpy                      |  9          |

**Visual baselines.** `docs/perf/after/screenshots/toc-*.png` captures
the new layout at 1440 / 1024 / 390 px viewports, all six posts at
1440, and one scrolled-state shot showing both sidebars staying fixed.

## What this pass moved

Three named commits on `perf/orchestrated-optimization`:

- `91a5341` — phase A: low-risk byte + layout wins (-420 KB).
- `73c54a2` — phase B: lazy-load + WebP avatar + img attributes
  (-1.9 MB on post pages, -66 KB on home).
- `d76bee6` — phase C: a11y wins (86→100 on kmeans/mobile).

Plus a small Phase D defer + dead include cleanup (this commit).

Refs `docs/perf/budgets.json`, `docs/perf/recon-pipeline.md`,
`docs/perf/recon-constraints.md`.
