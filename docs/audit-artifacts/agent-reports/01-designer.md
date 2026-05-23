# Visual Design Audit — nilesh-patil.github.io
**Calibrated against:** Stripe Press, Linear, Vercel marketing
**Benchmark peers:** lilianweng.github.io, huyenchip.com, simonwillison.net

## [Severity: P0] — "Follow" button label is stock template copy; acts as a dead end on mobile
**Location:** `_includes/author-profile.html` line 39 — `<button class="btn btn--inverse">Follow</button>`
**Evidence:** The label "Follow" is a Twitter-era affordance from AcademicPages stock. On desktop (≥1024px) the button is `display: none`, but on mobile it's the *first interactive element* in the sidebar. For an AI leadership audience visiting from LinkedIn or a conference referral, tapping "Follow" on a personal site produces cognitive dissonance — it implies a feed subscription they won't get.
**Recommendation:** Change the label to `Connect` or `Links`. In `_includes/author-profile.html` line 39, replace `Follow` with `Connect`. Optionally add a downward chevron icon to signal disclosure.
**Reference:** https://huyenchip.com — sidebar uses plain-text icon links with no trigger button.

## [Severity: P0] — Talks and Teaching pages are completely empty but still route-accessible
**Location:** `http://localhost:4000/talks/` and `http://localhost:4000/teaching/`; `show_talks: false` and `show_teaching: false` hide them from the nav but not from direct URLs or the sitemap
**Evidence:** local-talks.png and local-teaching.png show a full-height page with only a centred h1 heading and nothing else. These pages are findable via direct URL, Google crawl, and `/sitemap/`. An AI leadership audience who lands here sees a half-built site.
**Recommendation:** Add `sitemap: false` and `redirect_to: /cv/` in the page front-matter, or add a visible placeholder sentence ("Forthcoming — check back or see CV"). Never leave a titled empty page accessible from the sitemap.
**Reference:** https://simonwillison.net/speaking/

## [Severity: P1] — No distinctive typeface; system-font stack reads as "developer default"
**Location:** `_sass/_themes.scss` lines 18–19 and 42–43 — `$global-font-family: $sans-serif` and `$header-font-family: $sans-serif` where `$sans-serif` is `-apple-system, "San Francisco", "Roboto", "Segoe UI", "Helvetica Neue"…`
**Evidence:** local-home.png shows body and headings both rendering in SF Pro on macOS — zero typographic differentiation between the h1 "Nilesh Patil," the h2 "Recent posts," and body prose except size and weight. The code block font is pure system default. This site's typography is indistinguishable from a GitHub Pages scaffold from 2017.
**Recommendation:** Introduce one editorial serif for h1/h2 headings — keep body in the system stack. Good candidates: **Source Serif 4**, **DM Serif Display**, or **Libre Baskerville**. In `_themes.scss`: `$header-font-family: 'Source Serif 4', Georgia, serif`. Add preconnect + font link to `_includes/head.html`.
**Reference:** https://simonwillison.net — minimal approach: tuning line-height to 1.65 and measure to ~70ch delivers strong readability without a font import.

## [Severity: P1] — Content column too narrow on 1440px; sidebar proportions leave dead whitespace
**Location:** `_sass/layout/_page.scss` lines 32–34 — `.page` is `span(8 of 12)` with `prefix(0.5 of 12)` at `$large`; `_sass/layout/_sidebar.scss` line 35 — `.sidebar` is `span(2 of 12)`
**Evidence:** In a 1280px container (the `$x-large` max-width), `span(8 of 12)` gives ~756px for the article. The left sidebar at 2/12 (~213px) and right TOC at 2/12 leave visible dead margins. On a 1440px viewport the side gutters expand but the article measure stays fixed — the page looks pinched inside a wider canvas.
**Recommendation:** At `$x-large`, widen article to `span(9 of 12)`. Or convert to CSS Grid with `minmax(0, 68ch)` for the article column and fixed `200px` sidebars.
**Reference:** https://huyenchip.com/blog.html — single-column article at fixed max-width ~700px, no left sidebar on post pages.

## [Severity: P1] — Avatar is a group photo cropped to a circle; identity is unclear at 110px display size
**Location:** `images/ensembledme.webp` rendered at ~110px diameter in the sidebar
**Evidence:** local-home.png shows the avatar: a cropped group photo (appears to include other people) with a warm yellow-toned background. At 110px, individual faces are barely distinguishable.
**Recommendation:** Replace with a clean solo headshot or high-contrast solo portrait. Minimum 400×400px, saved as WebP at q=85. Filename can stay `ensembledme.webp` so no HTML changes are needed.
**Reference:** https://lilianweng.github.io — clean solo headshot at 200px.

## [Severity: P1] — Code block font-size is doubly-scaled
**Location:** `_sass/_syntax.scss` lines 13 and 33 — outer container at 1.25em × inner `.highlight` at 0.75em = ~0.94em effective
**Evidence:** Code blocks in local-post-kmeans.png render at roughly normal body size — not the smaller, clearly-subordinate code that best-practice typography dictates.
**Recommendation:** Remove `font-size: $type-size-4` from the outer container in `_syntax.scss`, and set only `font-size: $type-size-6` directly on `.highlight`. Or: outer at `font-size: 1rem`, inner `.highlight` at `font-size: 0.875em`.
**Reference:** https://simonwillison.net — code renders at 13–14px, clearly subordinate to 17px body text.

## [Severity: P1] — Publications page: "Recommended citation" block destroys scan hierarchy
**Location:** `http://localhost:4000/publications/` — every entry renders: title → venue/year → abstract → full "Recommended citation: Patil, N. et al. …" → Download link
**Evidence:** local-publications.png shows 6 entries where the citation block occupies as much vertical space as the abstract and is typographically identical in weight.
**Recommendation:** Collapse the recommended citation behind a `<details>` disclosure or move it exclusively to the individual publication page. The listing should show: title link, venue + year on one muted line, one-sentence abstract, Download button.
**Reference:** https://lilianweng.github.io/posts/

## [Severity: P1] — Footer attribution ("Powered by Jekyll & AcademicPages, a fork of Minimal Mistakes") undercuts design ambition
**Location:** `_includes/footer.html` line 30
**Evidence:** This attribution appears on every page including the CV — exactly where a recruiter, investor, or conference organiser would be reading.
**Recommendation:** The AcademicPages MIT license does not require footer attribution on the rendered page. Trim to: `&copy; {{ site.time | date: '%Y' }} {{ site.name }}`. If attribution feels important, add a `/colophon/` page.
**Reference:** https://huyenchip.com — footer contains only copyright + contact email.

## [Severity: P1] — Blog list visual grammar is inconsistent between home page and /posts/
**Location:** `_pages/home.md` vs `http://localhost:4000/posts/`
**Evidence:** local-home.png: compact bullet list — title + date · excerpt. local-posts.png: each entry shows a clock icon + "X minute read," then a calendar icon + "Published:" + date, then excerpt. The "Published:" text label is redundant.
**Recommendation:** Align `/posts/` with the home page register: remove the "Published:" prefix text, consolidate read-time into the date line separated by `·`. In `_sass/layout/_archive.scss` hide the `.page__meta` "Published:" label text.
**Reference:** https://simonwillison.net/2025/

## [Severity: P1] — Dark mode active nav underline barely distinguishable from inactive items
**Location:** `_sass/layout/_masthead.scss` lines 77–85
**Evidence:** local-home-dark.png: the active nav item underline is a 2px light-grey-on-dark-grey line with minimal visual salience. The accent cyan `#7fb3d5` is used throughout the dark theme for links — but not for the active nav indicator.
**Recommendation:** In `_sass/layout/_masthead.scss`, change selected state to `border-bottom: 2px solid var(--global-base-color)`.
**Reference:** https://linear.app — nav active state uses a high-contrast accent underline.

## [Severity: P2] — Syntax highlighting is Solarized Light only; tokens broken in dark mode
**Location:** `_sass/_syntax.scss` — all token colors are fixed Solarized Light values; `--global-code-background-color` in dark mode is `#161a20` but token colors don't adapt
**Recommendation:** Add a `html[data-theme="dark"] .highlight { ... }` override block with One Dark or Solarized Dark token colors.
**Reference:** https://github.com/primer/github-markdown-css

## [Severity: P2] — Portfolio and CV are the weakest visual pages
**Location:** `http://localhost:4000/portfolio/`, `http://localhost:4000/cv/`
**Evidence:** Portfolio: two entries with teal h2 link + date icon + one paragraph. CV: job entries are monolithic h3 headings with all metadata concatenated in em-dashes — visually flat wall of text.
**Recommendation:** (Portfolio) Add project thumbnail images and a GitHub/link chip per entry. (CV) Use a two-column date+role layout for experience entries: date at ~80px left column in muted weight, title+company right.
**Reference:** https://huyenchip.com/about/

## [Severity: P2] — Theme toggle is a cryptic 16px icon with no visible label
**Location:** `_includes/masthead.html` line 33
**Evidence:** A first-time visitor has no indication this is a 3-mode cycle. The fa-book-open icon for sepia is not universally recognized.
**Recommendation:** At ≥$large breakpoints, add a visible text label inline next to the icon, or convert to a three-segment control.
**Reference:** https://linear.app/changelog — three-segment "System / Light / Dark" toggle with visible text labels.

## [Severity: P2] — Footer "FOLLOW:" all-caps label is dated AcademicPages 2017 vernacular
**Location:** `_sass/layout/_footer.scss` lines 73–77 (`text-transform: uppercase`); `_includes/footer.html` line 6
**Recommendation:** Remove "FOLLOW:" label or replace with "Find me on". Change `text-transform: uppercase` to `font-weight: 500`. Align footer Google Scholar icon to `ai ai-google-scholar` (already loaded).
**Reference:** https://simonwillison.net

## [Severity: P2] — No OG social card image; site-level fallback is a 712×720 group photo
**Location:** `_config.yml` line 14
**Evidence:** A 712×720 square portrait cropped to 1200×630 OG format will show a badly letterboxed group photo.
**Recommendation:** Create a 1200×630px OG card. Save as `images/og-card.jpg`. Add `og_image: /images/og-card.jpg` to `_config.yml`.
**Reference:** https://huyenchip.com

## Top 3 Single-Change Recommendations
1. **Introduce one editorial serif for headings** (P1, ~45min) — Transforms site from "GitHub Pages template" to "designed artifact."
2. **Remove template attribution from footer** (P1, ~5min) — Disproportionate to effort.
3. **Create 1200×630 OG social card image** (P2, ~30min) — Affects every social share.
