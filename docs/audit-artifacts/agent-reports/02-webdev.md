# Web-Dev Audit — Jekyll personal site
**Scope:** code health, perf, a11y, SEO, build hygiene, local-vs-prod drift

## [Severity: P0] — Giscus comments widget renders a broken-placeholder banner on every post
**Category:** Code
**Location:** `_config.yml:199,201` | `_includes/comments-providers/giscus.html:3-11`
**Evidence:** `_config.yml` has `repo_id: ""` and `category_id: ""`. The giscus template's guard renders a yellow `<p class="notice--warning">Comments are not yet wired.</p>` instead of the comment widget. Visible in production right now.
**Recommendation:** Enable GitHub Discussions on `nilesh-patil/nilesh-patil.github.io`, run the giscus.app wizard to generate IDs, fill into `_config.yml`. Or set `comments: false` in posts defaults to suppress the warning.
**Reference:** https://giscus.app

## [Severity: P0] — `main.min.js` loaded as `type="module"` — wrong for a jQuery bundle; creates race with `theme-cycle.js`
**Category:** Code / Performance
**Location:** `_includes/scripts.html:1`
**Evidence:** `<script type="module" src="{{ base_path }}/assets/js/main.min.js"></script>`. ES modules are implicitly deferred, execute in strict mode, and scope their bindings — none of which is compatible with the academicpages jQuery bundle. `theme-cycle.js` is loaded with `defer` and depends on jQuery's `$(document).ready` ordering. With `type="module"` that ordering guarantee no longer holds.
**Recommendation:** Replace with `<script defer src="{{ base_path }}/assets/js/main.min.js"></script>`. Keep `theme-cycle.js` as-is.
**Reference:** https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script#module

## [Severity: P1] — 354 Dart Sass slash-division deprecations will become hard build errors in Sass 2.0
**Category:** Code
**Location:** `_sass/include/_mixins.scss:18`; `_sass/vendor/susy/...`; `_sass/vendor/breakpoint/...`
**Evidence:** Jekyll build emits 354 deprecation warnings. Susy (last release 2017) and breakpoint (2018) are unmaintained.
**Recommendation:** (1) Fix `_mixins.scss:18` — add `@use "sass:math";` and replace `($target / $context)` with `math.div(...)`. (2) Add `sass: quiet_deps: true` to `_config.yml` to suppress vendored-library warnings while migrating. (3) Plan removal of susy (replace float grid with CSS Grid) and breakpoint (replace with native `@media`).
**Reference:** https://sass-lang.com/documentation/breaking-changes/slash-div/

## [Severity: P1] — Skip-to-content link CSS exists but no HTML is ever rendered
**Category:** A11y (WCAG 2.2 AA 2.4.1)
**Location:** `_layouts/default.html`; `_sass/include/_utilities.scss:68-80`
**Evidence:** `.skip-link` and `.screen-reader-shortcut:focus` are defined but searching all `_includes/` and `_layouts/` for `skip-link` returns zero matches. Keyboard users tab through entire masthead nav before reaching main content.
**Recommendation:** Add immediately after `<body>` in `_layouts/default.html`:
```html
<ul class="skip-link"><li><a href="#main" class="screen-reader-shortcut">Skip to main content</a></li></ul>
```
**Reference:** https://www.w3.org/WAI/WCAG22/Understanding/bypass-blocks.html

## [Severity: P1] — Theme toggle uses `<a role="button">` without Space-key handler
**Category:** A11y (WCAG 2.2 AA 4.1.2)
**Location:** `_includes/masthead.html:33`; `assets/js/theme-cycle.js:48-52,79-86`
**Evidence:** An element with `role="button"` must activate on both click and Space. Native `<a>` fires `click` only on Enter, not Space. `theme-cycle.js:81` binds only `addEventListener("click", ...)`.
**Recommendation:** Replace the `<a role="button">` with a native `<button>` in `masthead.html`. Then simplify `syncIcon()` in `theme-cycle.js` to query `#theme-toggle` directly.
**Reference:** https://www.w3.org/WAI/ARIA/apg/patterns/button/

## [Severity: P1] — Theme change has no `aria-live` announcement for screen-reader users
**Category:** A11y (WCAG 2.2 AA 4.1.3)
**Location:** `assets/js/theme-cycle.js:31-38`
**Evidence:** `setTheme()` updates `aria-label` but no `aria-live` region exists. A screen reader user receives no feedback that the theme changed.
**Recommendation:** Add `<div id="theme-announcement" aria-live="polite" aria-atomic="true" class="visually-hidden"></div>` to default layout. In `theme-cycle.js` `setTheme()`, after `syncIcon()`, set `ann.textContent = theme + " mode active."`.
**Reference:** https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html

## [Severity: P1] — `<button>Follow</button>` in author sidebar has no ARIA disclosure state
**Category:** A11y
**Location:** `_includes/author-profile.html:39`
**Evidence:** Button toggles `.author__urls` but has no `aria-expanded` or `aria-controls`.
**Recommendation:** Either update to proper disclosure pattern (`aria-expanded`/`aria-controls`) or replace with native `<details>`/`<summary>`.
**Reference:** https://www.w3.org/WAI/ARIA/apg/patterns/disclosure/

## [Severity: P1] — Twitter/X Card meta tags are never emitted — `site.twitter.username` is not set
**Category:** SEO
**Location:** `_includes/seo.html:51`; `_config.yml:32`
**Evidence:** `seo.html:51` guards on `site.twitter.username`. `_config.yml` has `twitter: "ensembledme"` nested under `author:`, not at top level. Zero `<meta name="twitter:*">` tags emitted.
**Recommendation:** Add at the top level of `_config.yml`:
```yaml
twitter:
  username: ensembledme
```
**Reference:** https://developer.twitter.com/en/docs/twitter-for-websites/cards/overview/summary

## [Severity: P1] — `Person` JSON-LD is never emitted because `site.social` is absent from `_config.yml`
**Category:** SEO
**Location:** `_includes/seo.html:89-99`
**Recommendation:** Add `social:` block with `type: Person`, `name`, and `links:` array (GitHub, Scholar, Twitter, Medium).
**Reference:** https://schema.org/Person | https://developers.google.com/search/docs/appearance/structured-data/intro-structured-data

## [Severity: P1] — JSON-LD `@context` uses `http://schema.org` instead of `https://schema.org`
**Category:** SEO
**Location:** `_includes/seo.html:92` and `seo.html:133`
**Recommendation:** Change both occurrences to `"@context": "https://schema.org"`.
**Reference:** https://schema.org

## [Severity: P1] — No `<link rel="preload">` for the LCP image (author avatar)
**Category:** Performance
**Location:** `_includes/head.html`
**Evidence:** Avatar is the first image in `<main>` on every page. `fetchpriority="high"` is set but the image is buried inside includes — browser cannot start fetching until parse → CSS → layout.
**Recommendation:** Add to `_includes/head.html` after the CSS `<link>`:
```html
<link rel="preload" href="{{ '/images/ensembledme.webp' | prepend: base_path }}" as="image" type="image/webp" fetchpriority="high">
```
**Reference:** https://web.dev/articles/lcp

## [Severity: P1] — `.travis.yml` targets Ruby 2.1 (EOL 2017) and is stale at project root
**Category:** Build / DX
**Location:** `.travis.yml`
**Evidence:** Project's active CI is `.github/workflows/pages.yml`. Travis file is from circa 2016-2017.
**Recommendation:** Delete `.travis.yml`.
**Reference:** https://docs.github.com/en/actions/about-github-actions/migrating-from-travis-ci-to-github-actions

## [Severity: P2] — `meta[name="theme-color"]` hardcoded `#ffffff` mismatches dark and sepia themes
**Category:** UX
**Location:** `_includes/head/custom.html:12`
**Recommendation:**
```html
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff">
<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#1d2128">
```
**Reference:** https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta/name/theme-color

## [Severity: P2] — `og:type` and default `og:image` are absent on non-article pages
**Category:** SEO
**Location:** `_includes/seo.html:119-146`
**Evidence:** `og:type` emitted only when `page.date` exists. Home, about, publications, CV pages have no `og:type`. `og:image` only set when `page.header.image` is present.
**Recommendation:** Add `{% else %}<meta property="og:type" content="website">{% endif %}` after the article block. Add `og_image: ensembledme.jpg` to `_config.yml` and a fallback block in `seo.html`.
**Reference:** https://ogp.me/#types

## [Severity: P2] — Google Scholar social link emits leading-space accessible name
**Category:** A11y
**Location:** `_includes/author-profile.html:62-64`
**Evidence:** DOM snapshot shows link name `" Google Scholar"` — leading space + missing `aria-hidden` on the `<i>` icon.
**Recommendation:** Add `aria-hidden="true"` to the `<i>` in the `googlescholar` block and the `arxiv` block.
**Reference:** https://www.w3.org/WAI/WCAG22/Techniques/aria/ARIA6

## [Severity: P2] — Local-vs-prod drift: one-day "site last updated" difference only; no content drift
**Category:** Drift
**Location:** `local-home-snapshot.txt:88` vs `prod-home-snapshot.txt:88`
**Evidence:** Only difference across all compared pairs. Date is an expected artifact of `{{ "now" | date: ... }}` in footer.html. **The deploy is current and clean.**
**Recommendation:** No fix required. Optionally replace `"now"` with `site.time`.
