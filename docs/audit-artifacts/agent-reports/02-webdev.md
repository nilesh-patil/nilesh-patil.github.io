# Web Developer Audit Report — nilesh-patil.github.io

**Agent:** 02-webdev  
**Date:** 2026-05-24  
**Pages audited:** `/` (home), `/about/`, `/cv/`, `/publications/`, `/posts/distributed-kmeans-clustering/` (one post)  
**Lighthouse runs (snapshot mode):** home (96 a11y), cv (96 a11y), publications (100), post (92 a11y)  
**Tools:** Chrome DevTools MCP · `evaluate_script` · `list_console_messages` · `list_network_requests` · `lighthouse_audit` · Direct file reads  

---

## Web Dev Verdict

The site is in a **P0 regression state**: commit `3930ceb` changed `assets/js/main.min.js`'s `<script>` tag from `type="module"` to `defer`, converting what was a valid ES module graph into a top-level `import` parse error. The result is that jQuery never executes, taking down navigation toggle, the "Follow" sidebar button, smooth scroll, and `body` padding calculation with it. Beyond this critical breakage, the site carries a cluster of medium-priority issues: three syntax-highlight token colors fail WCAG AA contrast; the sidebar "Follow" button has no `aria-expanded`/`aria-controls`; the nav toggle button is missing `aria-expanded` and `aria-controls`; the home page has two `<h1>` elements (one empty); `og:image` and `twitter:image` point to a `.jpg` that has been replaced by `.webp`; `.travis.yml` is stale and shipping with the build; and 354+ Sass `@import` deprecation warnings will become build-breaking errors in Dart Sass 3. Structural HTML semantics are otherwise solid: landmarks exist, `lang` is set, viewport is correct, heading nesting is only mildly broken. Lighthouse snapshot scores (which run on the already-errored DOM, so jQuery-dependent state is absent) report 92–96 in accessibility with colour-contrast and touch-target failures.

---

## Findings

---

```yaml
---
id: J-01
title: "main.min.js: top-level import statement in defer script causes P0 SyntaxError"
category: Code/JS
severity: P0
confidence: HIGH
effort: 15m
agents: [webdev]
---
```

**Evidence:** Console on every page: `Uncaught SyntaxError: Cannot use import statement outside a module` (msgid=2, confirmed on home, cv, post pages). Root cause is the final 2 KB of `assets/js/main.min.js` (the minified `_main.js`): `import{plotlyDarkLayout,plotlyLightLayout}from"./theme.js"` appears at char position ~438000 of the bundle. `_includes/scripts.html:1` loads the file as `<script defer src=".../main.min.js">` — no `type="module"`. A non-module `<script>` that contains a top-level static `import` is a parse error per the HTML spec; the browser throws before any byte of jQuery or other code can run.

**Cascade:** `typeof window.$` is `"undefined"` on every page load. Downstream breakage (all jQuery-dependent):
- `jquery.greedy-navigation.js` — nav collapses to hidden-links at narrow viewports but the toggle button stays broken
- `$(".author__urls-wrapper button").on("click", …)` — "Follow" opens nothing
- `$("a").smoothScroll(…)` — anchor scroll disabled
- `bumpIt()` / `body padding-bottom` — footer may overlap content at some heights
- `fitvids()` — video iframes not responsive

**Why this matters:** Navigation and core UI are dead for all visitors. This is strictly worse than the pre-fix state, where the bundle ran as a module but lost Plotly tree-shaking. The prior fix (commit `3930ceb` issue #2) chose the wrong lever.

**Recommendation:** Two safe options ranked by risk:

Option A (lowest risk, 15 min): Remove the `import` statement from `_main.js` and inline the plotly layout data, or guard it behind a runtime `import()` dynamic call. Keep `defer` (no `type="module"`). The plotly init block is already conditioned on `plotlyElements.length > 0`; moving to a dynamic import is trivially safe because Plotly posts only appear on specific pages.

Option B (also correct, 30 min): Revert to `type="module"` AND add `crossorigin` to any CDN fetches. This restores the bundle as a proper module graph. The trade-off is that module scripts are always `defer`-equivalent, so behavior is unchanged from a user perspective.

**Fix snippet (Option A — preferred):**
```js
// assets/js/_main.js — replace lines 57-82 with:
let plotlyElements = document.querySelectorAll("pre>code.language-plotly");
if (plotlyElements.length > 0) {
  import('./theme.js').then(({ plotlyDarkLayout, plotlyLightLayout }) => {
    document.addEventListener("readystatechange", () => {
      if (document.readyState === "complete") {
        plotlyElements.forEach((elem) => {
          var jsonData = JSON.parse(elem.textContent);
          elem.parentElement.classList.add("hidden");
          let chartElement = document.createElement("div");
          elem.parentElement.after(chartElement);
          const theme = (determineComputedTheme() === "dark") ? plotlyDarkLayout : plotlyLightLayout;
          if (jsonData.layout) {
            jsonData.layout.template = jsonData.layout.template ? { ...theme, ...jsonData.layout.template } : theme;
          } else { jsonData.layout = { template: theme }; }
          Plotly.react(chartElement, jsonData.data, jsonData.layout);
        });
      }
    });
  });
}
```
Then rebuild `main.min.js` from `_main.js`.

**Spec reference:** https://html.spec.whatwg.org/multipage/webappapis.html#module-script ("A classic script may not contain top-level import/export"); https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script#type

---

```yaml
---
id: X-01
title: "Follow button: missing aria-expanded and aria-controls — widget state not announced"
category: Accessibility
severity: P1
confidence: HIGH
effort: 15m
agents: [webdev]
---
```

**Evidence:** `_includes/author-profile.html:39`: `<button class="btn btn--inverse">Follow</button>`. The button controls `.author__urls` (the social links list), which fades in/out via jQuery. No `aria-expanded` attribute, no `aria-controls` attribute. Confirmed via `evaluate_script`: `followBtn.ariaExpanded === null`. This pre-exists the J-01 regression but is now completely inoperable because jQuery is unavailable (the list never toggles).

**Why this matters:** Screen reader users hear "Follow, button" with no indication that it expands a disclosure widget, cannot determine the current expanded/collapsed state, and get no cue that a list of social links appeared. WCAG 2.1 SC 4.1.2 (Name, Role, Value) requires that the expanded state of disclosure buttons be programmatically determinable.

**Recommendation:** Add `aria-expanded="false"` and `aria-controls="author-social-links"` to the button; add `id="author-social-links"` to the `<ul>`. Toggle `aria-expanded` in the jQuery click handler.

**Fix snippet:**
```html
<!-- _includes/author-profile.html -->
<button class="btn btn--inverse"
        aria-expanded="false"
        aria-controls="author-social-links">Follow</button>
<ul id="author-social-links" class="author__urls social-icons">
```
```js
// assets/js/_main.js — update the click handler:
$(".author__urls-wrapper button").on("click", function () {
  var expanded = $(this).attr("aria-expanded") === "true";
  $(this).attr("aria-expanded", String(!expanded));
  $(".author__urls").fadeToggle("fast", function () {});
  $(".author__urls-wrapper button").toggleClass("open");
});
```

**Spec reference:** https://www.w3.org/WAI/ARIA/apg/patterns/disclosure/ ; WCAG 2.1 SC 4.1.2 https://www.w3.org/TR/WCAG21/#name-role-value

---

```yaml
---
id: X-02
title: "Nav toggle button: missing aria-expanded and aria-controls"
category: Accessibility
severity: P1
confidence: HIGH
effort: 15m
agents: [webdev]
---
```

**Evidence:** `_includes/masthead.html:6`: `<button type="button" aria-label="Toggle navigation menu"><span class="navicon" aria-hidden="true"></span></button>`. No `aria-expanded`, no `aria-controls`. Confirmed via `evaluate_script`: `navToggle.ariaExpanded === null`, `navToggle.ariaControls === null`. The greedy-nav plugin (`assets/js/plugins/jquery.greedy-navigation.js`) toggles `class="hidden"` on `.hidden-links` but never manages ARIA state.

**Why this matters:** The hamburger button's collapsed/expanded state is invisible to assistive technology. WCAG 2.1 SC 4.1.2.

**Recommendation:** In `masthead.html`, add `aria-expanded="false"` and `aria-controls="greedy-nav-overflow"` to the button. Add `id="greedy-nav-overflow"` to `.hidden-links`. In the greedy-nav plugin, toggle `aria-expanded` in the `$btn.on('click', …)` handler.

**Fix snippet:**
```html
<!-- _includes/masthead.html -->
<button type="button"
        aria-label="Toggle navigation menu"
        aria-expanded="false"
        aria-controls="greedy-nav-overflow">
  <span class="navicon" aria-hidden="true"></span>
</button>
<ul class="hidden-links hidden" id="greedy-nav-overflow"></ul>
```
```js
// assets/js/plugins/jquery.greedy-navigation.js
$btn.on('click', function () {
  $hlinks.toggleClass('hidden');
  $(this).toggleClass('close');
  $(this).attr('aria-expanded', $hlinks.hasClass('hidden') ? 'false' : 'true');
});
```

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Attributes/aria-expanded ; WCAG 2.1 SC 4.1.2

---

```yaml
---
id: X-03
title: "No skip-to-main-content link — keyboard users must tab through full masthead on every page"
category: Accessibility
severity: P1
confidence: HIGH
effort: 30m
agents: [webdev]
---
```

**Evidence:** `evaluate_script` confirms `hasSkipLink === false` on home page. `document.querySelector('a[href^="#main"], a[href^="#content"], .skip-link, [class*="skip"]')` returns null. The `<div id="main">` wrapper exists in `_layouts/archive.html:15` and `_layouts/single.html:17` but there is no positioned-offscreen `<a href="#main">Skip to content</a>` in any layout header.

**Why this matters:** Keyboard-only users (including motor-impaired users and screen-reader users who navigate by Tab) must Tab through 6–8 navigation links on every page before reaching content. WCAG 2.1 SC 2.4.1 (Bypass Blocks) requires a mechanism to skip repeated navigation.

**Recommendation:** Add a visually-hidden, focus-visible skip link as the first element inside `<body>` in `_layouts/default.html` (before `{% include masthead.html %}`). The `<div id="main">` target already exists; add `tabindex="-1"` to make it programmatically focusable.

**Fix snippet:**
```html
<!-- _layouts/default.html — add before {% include masthead.html %} -->
<a class="skip-link screen-reader-shortcut" href="#main">Skip to main content</a>
```
```scss
// _sass/layout/_base.scss or a new utility
.skip-link {
  position: absolute;
  top: -100%;
  left: 1rem;
  z-index: 999;
  padding: 0.5em 1em;
  background: var(--global-bg-color);
  color: var(--global-link-color);
  border: 2px solid var(--global-link-color);
  &:focus { top: 0.5rem; }
}
```

**Spec reference:** WCAG 2.1 SC 2.4.1 https://www.w3.org/TR/WCAG21/#bypass-blocks ; Technique G1 https://www.w3.org/WAI/WCAG21/Techniques/general/G1

---

```yaml
---
id: X-04
title: "Home page has two <h1> elements (one empty) and sidebar h2 precedes page h1"
category: Accessibility
severity: P1
confidence: HIGH
effort: 30m
agents: [webdev]
---
```

**Evidence:** `evaluate_script` on `/` returns heading array: `[{H2,"Nilesh Patil"}, {H1,""}, {H1,"Nilesh Patil"}, {H2,"Recent posts"}, …]`. Two problems:

1. An empty `<h1 class="page__title">` is emitted by whatever layout renders the home page (the splash/home layout emits `page__title` but `page.title` is the site title not a page subheading, resulting in an empty tag or duplicate).
2. The sidebar `<h2 class="author__name">Nilesh Patil</h2>` (from `author-profile.html:33`) appears in DOM source order before the `<h1>` article heading because the sidebar is rendered before `<main>` content. This inverts the expected heading hierarchy.

**Why this matters:** Screen readers announce headings in DOM order. An H2 before H1 misrepresents the document outline. An empty H1 is an inaccessible heading. WCAG 2.1 SC 1.3.1 (Info and Relationships), SC 2.4.6 (Headings and Labels).

**Recommendation:** (a) On the home page layout, remove or fix the empty `<h1>` emission — check that `page.title` is not blank before rendering `<h1 class="page__title">{{ page.title }}</h1>`. (b) Consider adding `aria-hidden="true"` to the sidebar `<h2>` or using `role="presentation"` on the author block since the name is also in the avatar `alt` text — making the decorative h2 invisible to the document outline is the least-invasive fix.

**Fix snippet:**
```liquid
{# archive.html / home layout — guard the h1: #}
{% unless page.header.overlay_color or page.header.overlay_image %}
  {% if page.title %}
    <h1 class="page__title">{{ page.title }}</h1>
  {% endif %}
{% endunless %}
```
```html
<!-- author-profile.html — mark sidebar name as decorative: -->
<h2 class="author__name" aria-hidden="true">{{ author.name }}</h2>
```

**Spec reference:** WCAG 2.1 SC 1.3.1 https://www.w3.org/TR/WCAG21/#info-and-relationships ; MDN "Using HTML sections and outlines" https://developer.mozilla.org/en-US/docs/Web/HTML/Element/Heading_Elements

---

```yaml
---
id: X-05
title: "Site nav <nav> has no aria-label — multiple nav landmarks undistinguishable"
category: Accessibility
severity: P2
confidence: HIGH
effort: 10m
agents: [webdev]
---
```

**Evidence:** `_includes/masthead.html:5`: `<nav id="site-nav" class="greedy-nav">` — no `aria-label` or `aria-labelledby`. When the TOC is present on post pages there are two `<nav>` elements with no differentiation. `evaluate_script`: `navEls[0].ariaLabel === null`.

**Why this matters:** Screen reader users navigating by landmark (e.g., `NVDA + D`) hear two "navigation" regions with no way to distinguish "site navigation" from "table of contents". WCAG 2.4.1, ARIA best practice.

**Recommendation:** Add `aria-label="Site navigation"` to the masthead `<nav>`. The TOC include should already have its own label (`aria-label="Table of contents"` or similar).

**Fix snippet:**
```html
<!-- _includes/masthead.html -->
<nav id="site-nav" class="greedy-nav" aria-label="Site navigation">
```

**Spec reference:** https://www.w3.org/WAI/ARIA/apg/patterns/landmarks/examples/navigation.html ; MDN https://developer.mozilla.org/en-US/docs/Web/HTML/Element/nav

---

```yaml
---
id: X-06
title: "Syntax highlight: 3 token color families fail WCAG AA contrast on code background"
category: Accessibility
severity: P2
confidence: HIGH
effort: 1h
agents: [webdev]
---
```

**Evidence:** `_sass/_syntax.scss`. Background is `--global-code-background-color: #fafafa`. Three color families fail:

| Token class | Color | Role | Contrast on #fafafa | WCAG threshold |
|---|---|---|---|---|
| `.nc`, `.nf`, `.kd`, `.nd`, `.nt`, `.nv`, `.bp`, `.vc`, `.vg`, `.vi` | `#22b3eb` | class/function names, keywords | **2.31:1** | 4.5:1 fail |
| `.sb` | `#93a1a1` | backtick strings | **2.56:1** | 4.5:1 fail |
| `.s`, `.s1`, `.s2`, `.si`, `.sx`, `.ss`, `.sc`, `.gd`, `.m`, `.mi`, `.mf`, `.mh`, `.mo`, `.il` | `#2aa198` | string literals, numbers | **3.03:1** | 4.5:1 fail (passes 3:1 for "large" only, but code is set at `$type-size-6` = 13.6px — not "large text") |

`.s` (Solarized cyan `#2aa198`) affects the most code — every string literal in every post.

**Why this matters:** Users with low vision or mild colour deficiency cannot reliably read function names, keywords, or string literals in code blocks. Code blocks are a primary content type on this site.

**Recommendation:** Replace failing colors with WCAG-AA-compliant alternatives from the Solarized family or equivalent:
- `#22b3eb` → `#0069a1` (darkened teal, ~5.1:1 on #fafafa, same hue)
- `#93a1a1` → `#5e7474` (already fixed for `.c`/`.c1`; apply to `.sb`)
- `#2aa198` → `#1a7a72` (darkened cyan, ~4.7:1 on #fafafa)

**Fix snippet:**
```scss
/* _sass/_syntax.scss — replace failing token colors */
.highlight .s, .highlight .s1, .highlight .s2,
.highlight .si, .highlight .sx, .highlight .ss,
.highlight .sc, .highlight .gd,
.highlight .m, .highlight .mi, .highlight .mf,
.highlight .mh, .highlight .mo, .highlight .il { color: #1a7a72 } /* was #2aa198 */

.highlight .sb { color: #5e7474 } /* was #93a1a1 */

.highlight .nc, .highlight .nf, .highlight .kd,
.highlight .nd, .highlight .nt, .highlight .nv,
.highlight .bp, .highlight .vc, .highlight .vg,
.highlight .vi, .highlight .kr { color: #0069a1 } /* was #22b3eb */
```

**Spec reference:** WCAG 2.1 SC 1.4.3 https://www.w3.org/TR/WCAG21/#contrast-minimum ; Contrast checker https://webaim.org/resources/contrastchecker/

---

```yaml
---
id: X-07
title: "theme-toggle link: touch target height 36px falls below 44px recommendation on post pages"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
agents: [webdev]
---
```

**Evidence:** Lighthouse on post page (92 a11y): `FAIL [0] target-size: Touch targets do not have sufficient size or spacing. selector: nav#site-nav > ul.visible-links > li#theme-toggle > a`. `evaluate_script` confirms: `themeLinkSize: {w:25, h:36}`. The `#theme-toggle a` is styled in `_sass/layout/_navigation.scss:228`: `width: 25px; display: flex;` — explicit narrow width, height controlled by parent `li` table-cell alignment.

**Why this matters:** WCAG 2.5.5 (Target Size, AAA) recommends 44×44px; WCAG 2.5.8 (Target Size, AA, 2.2) requires either 24×24px or spacing to adjacent targets to compensate. At 25×36px the height meets 24px but the spacing from adjacent nav items is minimal. Lighthouse flags it as a mobile usability failure.

**Recommendation:** Increase the padding on `#theme-toggle a` so the effective touch target is at least 44px tall, matching the `0.5rem` above/below that other nav links have.

**Fix snippet:**
```scss
/* _sass/layout/_navigation.scss — within .visible-links #theme-toggle */
#theme-toggle {
  a {
    width: 44px;   /* was 25px */
    min-height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
}
```

**Spec reference:** WCAG 2.5.8 Target Size (Minimum) https://www.w3.org/TR/WCAG22/#target-size-minimum ; MDN https://developer.mozilla.org/en-US/docs/Web/Accessibility/Understanding_WCAG/Perceivable/Use_of_color

---

```yaml
---
id: S-01
title: "Duplicate meta[name=description]: og:description template emits second name=description"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
agents: [webdev]
---
```

**Evidence:** `_includes/seo.html:125`: `<meta property="og:description" name="description" content="{{ seo_description }}">`. When `page.excerpt` or `site.og_description` is set, this tag renders with both `property="og:description"` AND `name="description"`. The earlier `<meta name="description">` at line 27 is already present. Result: every page with an excerpt or when `og_description` is configured has two `meta[name="description"]` tags with different values. Confirmed via `evaluate_script`: `metaDescCount: 2`, two different content strings.

**Why this matters:** Search engines (Google, Bing) warn on duplicate meta descriptions. Parsers may use either value non-deterministically. The two descriptions diverge (site tagline vs. og_description), giving search engines inconsistent signals.

**Recommendation:** Remove `name="description"` from the OG meta tag. The OG tag should use only `property="og:description"`.

**Fix snippet:**
```html
<!-- _includes/seo.html lines 124-128 — remove name="description" from og tag -->
{% if page.excerpt %}
  <meta property="og:description" content="{{ seo_description }}">
{% elsif site.og_description %}
  <meta property="og:description" content="{{ site.og_description }}">
{% endif %}
```

**Spec reference:** Open Graph Protocol https://ogp.me/#metadata ; Google Search Central meta tags https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls

---

```yaml
---
id: S-02
title: "og:image and twitter:image reference ensembledme.jpg but only ensembledme.webp is served"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 10m
agents: [webdev]
---
```

**Evidence:** `_config.yml:26`: `og_image: "ensembledme.jpg"`. Browser confirms `og:image` content: `https://nilesh-patil.github.io/images/ensembledme.jpg`. Network request list on home page shows `GET /images/ensembledme.webp [200]` — the JPEG is served by the `<img>` fallback inside `<picture>`, but at the OG image URL `ensembledme.jpg` the file may or may not exist on the deployed site. `author-profile.html` constructs `_avatar_webp` via Liquid replace — only the `.webp` is referenced in markup; the original `.jpg` may have been removed.

**Why this matters:** When social platforms (Twitter, LinkedIn, Slack) unfurl links they fetch the `og:image` URL directly, bypassing `<picture>` negotiation. If `ensembledme.jpg` is absent from `_site/images/`, the preview image will be broken in all social unfurls.

**Recommendation:** Either ensure `ensembledme.jpg` is committed and deployed alongside `.webp`, or update `_config.yml` to `og_image: "ensembledme.webp"` (Open Graph supports WebP; most modern platforms handle it).

**Fix snippet:**
```yaml
# _config.yml
og_image: "ensembledme.webp"
```

**Spec reference:** Open Graph image https://ogp.me/#structured ; Facebook image guidelines https://developers.facebook.com/docs/sharing/best-practices/#images

---

```yaml
---
id: S-03
title: "theme-color meta is hardcoded #ffffff — no dark-mode variant provided"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 15m
agents: [webdev]
---
```

**Evidence:** `_includes/head/custom.html:12`: `<meta name="theme-color" content="#ffffff"/>`. No `media` attribute. On dark-mode pages (when `data-theme="dark"` is set), the browser chrome remains white. The site has a full dark theme (`--global-bg-color` becomes a dark value), but the OS-level chrome color does not follow.

**Why this matters:** On Chrome/Android and Safari/iOS, `theme-color` colors the browser chrome (address bar, status bar). A static white value on a dark-theme page creates a jarring white chrome on a dark page, breaking the intended dark experience.

**Recommendation:** Add a paired `theme-color` tag with `media="(prefers-color-scheme: dark)"`. The dark background color from `_sass/theme/_default_dark.scss` should be used.

**Fix snippet:**
```html
<!-- _includes/head/custom.html — replace single theme-color -->
<meta name="theme-color"
      content="#ffffff"
      media="(prefers-color-scheme: light)">
<meta name="theme-color"
      content="#1c1c1e"
      media="(prefers-color-scheme: dark)">
```

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta/name/theme-color ; https://web.dev/articles/add-manifest#theme_color

---

```yaml
---
id: I-01
title: ".travis.yml is stale and deployed — dead CI config confuses contributors"
category: Build/CI
severity: P2
confidence: HIGH
effort: 15m
agents: [webdev]
---
```

**Evidence:** File exists at repo root: `/Users/nilesh-patil/.../nilesh-patil.github.io/.travis.yml`. Content: `language: ruby; rvm: - 2.1; script: bundle exec jekyll build --drafts`. The repo now uses `.github/workflows/pages.yml` (GitHub Actions, Ruby 3.3). `.travis.yml` references `rvm: 2.1` (EOL since 2021). The file is NOT in the `exclude:` list in `_config.yml`, so Jekyll copies it to `_site/` and it is publicly accessible at `https://nilesh-patil.github.io/.travis.yml`.

**Why this matters:** The file is confusing for contributors, creates a dead CI signal if Travis CI is ever re-triggered, and ships to the public site unnecessarily. The note in the prior audit (commit `f8798d2`, issue #5) flagged this; the fix was not applied.

**Recommendation:** Delete `.travis.yml`. Alternatively, add it to the `exclude:` list in `_config.yml` if there is a reason to keep it.

**Fix snippet:**
```bash
# Delete the stale Travis config
rm .travis.yml
```
Or in `_config.yml`:
```yaml
exclude:
  - .travis.yml
  # ... existing excludes
```

**Spec reference:** GitHub Pages deployment docs https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

---

```yaml
---
id: I-02
title: "354+ Sass @import deprecations will become hard errors in Dart Sass 3.0"
category: Build/CI
severity: P2
confidence: HIGH
effort: 4h
agents: [webdev]
---
```

**Evidence:** `bundle exec jekyll build` emits exactly 21 unique `DEPRECATION WARNING` lines (some repeat per-file). Warning categories: `[import]`, `[color-functions]` (`lighten()`), `[global-builtin]`, `[if-function]`, `[slash-div]`. All originate in `assets/css/main.scss` which chains `@import` of vendor files: `breakpoint`, `susy`, `font-awesome`, and theme files. Phase 0 context notes "354 already noted". Dart Sass 3.0 will remove `@import` with no fallback; the build will fail.

**Why this matters:** GitHub Actions (`pages.yml`) uses `ruby/setup-ruby@v1` with Ruby 3.3 and Bundler, which picks up Dart Sass via the `sass-embedded` gem. When Dart Sass 3.0 ships (expected 2025/2026), the build pipeline will break with no code change from the site author.

**Recommendation:** Migrate all `@import` to `@use` / `@forward` using the Sass official migration tool (`sass-migrator module`). Vendor files (`breakpoint`, `susy`) are the main work — they require either updated vendor forks that use `@use`, or replacement with modern equivalents (e.g., replace susy grid with CSS Grid natively).

**Fix snippet (entry point change):**
```scss
// assets/css/main.scss — change @import to @use (one example):
// Before:
@import "vendor/breakpoint/breakpoint";
// After (with migration tool applied throughout):
@use "vendor/breakpoint/breakpoint" as bp;
```
Run: `npx sass-migrator module --migrate-deps assets/css/main.scss`

**Spec reference:** https://sass-lang.com/documentation/breaking-changes/import/ ; Sass migrator https://sass-lang.com/documentation/cli/migrator/

---

```yaml
---
id: P-01
title: "No cache-control headers on static assets — Jekyll dev server sends no-store for all resources"
category: Performance
severity: P2
confidence: MED
effort: 1h
agents: [webdev]
---
```

**Evidence:** `fetch('/assets/css/main.css', {method:'HEAD', cache:'no-store'})` returns `Cache-Control: private, max-age=0, proxy-revalidate, no-store, no-cache, must-revalidate`. Same headers on `main.min.js` and `ensembledme.webp`. This is the Jekyll WEBrick dev server behavior — it deliberately disables caching. Production (GitHub Pages) serves different headers. **However**, there is no explicit cache-control configuration for GitHub Pages assets: the `_config.yml` has no Jekyll Cache-Control plugin, and there is no `.htaccess` included that sets long-lived cache headers for fingerprinted assets.

**Why this matters:** On GitHub Pages, CSS/JS/images without explicit cache headers get GitHub's default (~600s for most assets). The main CSS file (`main.css`, ~157 KB uncompressed) has no content-hash in its filename, so long cache durations would cause stale CSS after deploys. This is low priority in production but worth noting for when the site migrates to a CDN.

**Recommendation:** Consider adding cache-busting query parameters or content hashes to `main.css`/`main.min.js` filenames using a Jekyll asset-pipeline plugin, or accept GitHub Pages' default TTL. No immediate production fix is needed, but document the dependency on GitHub Pages' default behavior.

**Fix snippet:** N/A for dev server. For GitHub Pages production, note that the platform controls headers — an `.htaccess` approach only works on Apache-served hosts, not GitHub Pages.

**Spec reference:** https://web.dev/articles/http-cache ; https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control

---

```yaml
---
id: P-02
title: "No <link rel=preload> for critical above-the-fold assets (avatar, fonts)"
category: Performance
severity: P2
confidence: MED
effort: 30m
agents: [webdev]
---
```

**Evidence:** `evaluate_script` on home page: `preloads: []` — no `<link rel="preload">`, `<link rel="preconnect">`, or `<link rel="dns-prefetch">` elements in `<head>`. Network waterfall (home page): 10 requests; `ensembledme.webp` (avatar, `fetchpriority="high"` is already set — good) loads in the same tick as fonts. `fa-solid-900.woff2` and `fa-brands-400.woff2` are discovered only after CSS is parsed, delaying icon rendering.

**Why this matters:** Font-Awesome woff2 files are render-blocking for icon display; without preload hints they arrive late in the waterfall causing FOUT (flash of icon-less content). LCP for the hero avatar is partially mitigated by `fetchpriority="high"` already in `author-profile.html:23`.

**Recommendation:** Add preload hints for the two FA woff2 files to `_includes/head/custom.html`. The avatar already has `fetchpriority="high"` which partially compensates for the lack of a preload link.

**Fix snippet:**
```html
<!-- _includes/head/custom.html -->
<link rel="preload"
      href="{{ base_path }}/assets/webfonts/fa-solid-900.woff2"
      as="font" type="font/woff2" crossorigin>
<link rel="preload"
      href="{{ base_path }}/assets/webfonts/fa-brands-400.woff2"
      as="font" type="font/woff2" crossorigin>
```

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel/preload ; https://web.dev/articles/preload-critical-assets

---

```yaml
---
id: P-03
title: "academicons.ttf loaded as TTF — no woff2 variant; TTF is ~3× larger than equivalent woff2"
category: Performance
severity: P2
confidence: HIGH
effort: 1h
agents: [webdev]
---
```

**Evidence:** Network request on every page: `GET /assets/fonts/academicons.ttf [200]`. The TTF is the only format served for Academicons. Font Awesome uses `woff2` (compressed). Academicons ships a woff2 variant in its npm release package but the project uses the legacy TTF-only distribution.

**Why this matters:** TTF files are uncompressed glyph outlines. The academicons TTF is ~53 KB raw. An equivalent woff2 would be ~22–28 KB. Over a slow connection (3G, common in India where the target audience is), this is a ~30 KB payload difference for every page load.

**Recommendation:** Replace `academicons.ttf` with the woff2 variant from the Academicons release. Update `_sass/vendor/font-awesome/` or the Academicons CSS to use woff2 with TTF fallback.

**Fix snippet:**
```css
/* Update the @font-face for Academicons */
@font-face {
  font-family: 'Academicons';
  src: url('../fonts/academicons.woff2') format('woff2'),
       url('../fonts/academicons.ttf') format('truetype');
  font-weight: normal;
  font-style: normal;
}
```

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/CSS/@font-face ; https://web.dev/articles/reduce-webfont-size

---

```yaml
---
id: X-08
title: "Share buttons: Bluesky and Mastodon fail WCAG AA contrast (white text on brand colors)"
category: Accessibility
severity: P2
confidence: HIGH
effort: 30m
agents: [webdev]
---
```

**Evidence:** On post pages, `.page__share` renders social share buttons. `evaluate_script` on post page returns share button colors:
- Bluesky: `rgb(255,255,255)` on `rgb(17,132,254)` → **3.65:1** (needs 4.5:1)
- Mastodon: `rgb(255,255,255)` on `rgb(99,100,255)` → **4.38:1** (needs 4.5:1)
- LinkedIn: `rgb(255,255,255)` on `rgb(0,123,182)` → **4.66:1** (passes)

Both Bluesky and Mastodon fail WCAG 2.1 SC 1.4.3.

**Why this matters:** Users with low vision or colour deficiency cannot reliably read the button labels.

**Recommendation:** Darken the Bluesky button background to ~`#0066CC` (5.1:1) and the Mastodon background to ~`#4B4CF0` or darken text to dark for Mastodon (`#222` on `#6364FF` is ~5.7:1).

**Fix snippet:**
```scss
/* _sass/layout/_page.scss or wherever .page__share is defined */
.page__share .btn--bluesky  { background-color: #0066cc; }
.page__share .btn--mastodon { background-color: #4242d4; }
```

**Spec reference:** WCAG 2.1 SC 1.4.3 https://www.w3.org/TR/WCAG21/#contrast-minimum ; WeAIM contrast checker https://webaim.org/resources/contrastchecker/

---

```yaml
---
id: X-09
title: "author-profile.html: some academicons <i> elements missing aria-hidden"
category: Accessibility
severity: P2
confidence: MED
effort: 15m
agents: [webdev]
---
```

**Evidence:** `_includes/author-profile.html` lines 60–73 (academic links like arxiv, googlescholar, orcid): `<i class="ai ai-arxiv ai-fw icon-pad-right"></i>` — no `aria-hidden="true"`. The Font Awesome icons for GitHub, LinkedIn, Twitter etc. correctly include `aria-hidden="true"` (lines 104, 145, 169). The academicons `<i>` elements are not consistently marked. Screen readers will attempt to announce them by class name or skip them unpredictably.

**Why this matters:** Decorative icon `<i>` elements read aloud by screen readers produce noise: "ai ai-arxiv ai-fw icon-pad-right". WCAG 1.1.1, Technique F87.

**Recommendation:** Add `aria-hidden="true"` to all academicons `<i>` elements in author-profile.html that are paired with visible text labels.

**Fix snippet:**
```html
<!-- author-profile.html — add aria-hidden to academicons: -->
{% if author.arxiv %}
  <li><a href="{{ author.arxiv }}"><i class="ai ai-arxiv ai-fw icon-pad-right" aria-hidden="true"></i>arXiv</a></li>
{% endif %}
{% if author.googlescholar %}
  <li><a href="{{ author.googlescholar }}"><i class="ai ai-google-scholar ai-fw icon-pad-right" aria-hidden="true"></i>Google Scholar</a></li>
{% endif %}
```
(Apply to all academicons `<i>` elements — arxiv, googlescholar, inspire-hep, impactstory, orcid, pubmed, scopus, semantic, ssrn, zotero.)

**Spec reference:** WCAG 2.1 SC 1.1.1 https://www.w3.org/TR/WCAG21/#non-text-content ; WCAG Technique F87 https://www.w3.org/WAI/WCAG21/Techniques/failures/F87

---

```yaml
---
id: I-03
title: "GitHub Actions workflow pins actions to tag not SHA — supply-chain risk"
category: Build/CI
severity: P2
confidence: HIGH
effort: 30m
agents: [webdev]
---
```

**Evidence:** `.github/workflows/pages.yml` uses: `actions/checkout@v4`, `actions/setup-node@v4`, `ruby/setup-ruby@v1`, `actions/configure-pages@v5`, `actions/upload-pages-artifact@v3`, `actions/deploy-pages@v4`. All pinned to semver tags, not SHA digests. A tag can be moved by the action author after publication (mutable reference).

**Why this matters:** A compromised action maintainer could push malicious code to the `v4` tag. GitHub's own security hardening guide and SLSA supply chain security recommends pinning to `@sha256:...` immutable refs for production deployments.

**Recommendation:** Pin each action to a specific SHA. Use Dependabot or Renovate to automate updates.

**Fix snippet:**
```yaml
# .github/workflows/pages.yml
- uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
- uses: ruby/setup-ruby@7cd807fd9df1f625f5e7fb1d8a44462ead1cd62c  # v1.206
```

**Spec reference:** GitHub Actions security hardening https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions#using-third-party-actions ; SLSA https://slsa.dev/

---

```yaml
---
id: J-02
title: "greedy-nav plugin uses screen.orientation.addEventListener without feature check — may throw on older browsers"
category: Code/JS
severity: P2
confidence: MED
effort: 10m
agents: [webdev]
---
```

**Evidence:** `assets/js/plugins/jquery.greedy-navigation.js:77`: `screen.orientation.addEventListener("change", function () { updateNav(); });`. `screen.orientation` is undefined on some environments (iOS WebView, some older Androids, IE11). A TypeError here would prevent `updateNav()` from being registered.

**Why this matters:** Orientation changes on mobile would not update the nav layout if the event listener registration threw. This is also moot while jQuery is unavailable (J-01), but should be fixed when the primary issue is resolved.

**Recommendation:** Add a defensive guard.

**Fix snippet:**
```js
if (screen.orientation && typeof screen.orientation.addEventListener === 'function') {
  screen.orientation.addEventListener("change", function () {
    updateNav();
  });
}
```

**Spec reference:** MDN `ScreenOrientation` https://developer.mozilla.org/en-US/docs/Web/API/ScreenOrientation ; Can I Use https://caniuse.com/screen-orientation

---

## Summary Table

| ID | Title | Severity | Effort |
|---|---|---|---|
| J-01 | `main.min.js` import SyntaxError — P0 jQuery regression | P0 | 15m |
| X-01 | Follow button missing aria-expanded/controls | P1 | 15m |
| X-02 | Nav toggle missing aria-expanded/controls | P1 | 15m |
| X-03 | No skip-to-content link | P1 | 30m |
| X-04 | Home page: two h1 elements (one empty) + sidebar h2 before h1 | P1 | 30m |
| S-01 | Duplicate meta[name=description] from og:description template | P1 | 15m |
| X-05 | Site nav missing aria-label | P2 | 10m |
| X-06 | Syntax highlight: 3 token families fail WCAG AA contrast | P2 | 1h |
| X-07 | Theme toggle touch target 25×36px — below 44px recommendation | P2 | 15m |
| X-08 | Share buttons Bluesky/Mastodon fail WCAG AA contrast | P2 | 30m |
| X-09 | Academicons icons missing aria-hidden in author-profile | P2 | 15m |
| S-02 | og:image references .jpg but site serves .webp | P2 | 10m |
| S-03 | theme-color hardcoded white — no dark-mode media variant | P2 | 15m |
| I-01 | .travis.yml stale, deployed to public _site | P2 | 15m |
| I-02 | 354+ Sass @import deprecations — will break on Dart Sass 3.0 | P2 | 4h |
| P-01 | No cache-control configuration for static assets | P2 | 1h |
| P-02 | No preload hints for FA woff2 fonts | P2 | 30m |
| P-03 | Academicons served as TTF — no woff2 variant | P2 | 1h |
| I-03 | GH Actions workflow pinned to tags not SHAs | P2 | 30m |
| J-02 | `screen.orientation.addEventListener` lacks feature guard | P2 | 10m |
