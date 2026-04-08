# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

Personal Jekyll site — GitHub repo `nilesh-patil/nilesh-patil.github.io`, served at the custom domain **`https://www.nilesh42.science`** (set via `CNAME`; the `*.github.io` URL only redirects). Built on the **academicpages** theme — but the theme is **not vendored**. Every layout, include, Sass partial, and JS file is in-repo and has been customized. Treat the whole tree as project code, not theme code you shouldn't touch.

Client-side search is provided by **Pagefind**, run as a post-build step against `_site/`.

## Environment

**Ruby 4.0+ via Homebrew is required.** The system Ruby at `/usr/bin/ruby` (2.6) cannot run this site — `Gemfile.lock` pins Bundler 4.0.11 which the system Ruby doesn't have. Either prepend the Homebrew Ruby to `PATH` for each command or set it in your shell profile:

```bash
export PATH="/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin:$PATH"
```

Node ≥ 18 is required for the JS bundling pipeline (`package.json` engines).

## Commands

### Local development
```bash
# Dev server. The _config.dev.yml override is REQUIRED — it empties
# site.url so absolute canonical/og:image URLs resolve to localhost
# instead of the production domain (https://www.nilesh42.science).
# Assets and hero backgrounds are root-relative (quirk #8), so they
# render correctly regardless.
bundle exec jekyll serve --config _config.yml,_config.dev.yml --port 4000
```

### Production build (mirrors CI)
```bash
npm run build   # bundle exec jekyll build && pagefind --site _site
```

### Rebuilding the JS bundle
`assets/js/main.min.js` is **hand-concatenated by uglify-js**, not produced by a real module bundler. After editing `assets/js/_main.js` or any of the vendor inputs, you must rebuild:

```bash
npm run uglify        # one-shot rebuild of main.min.js
npm run watch:js      # rebuild on every change under assets/js/
```

The exact concatenation order matters and is defined in `package.json`'s `scripts.uglify`: jquery → fitvids → smooth-scroll → plotly → greedy-nav → `_main.js`.

### Deploy
Pushing to `master` triggers `.github/workflows/pages.yml`: `actions/configure-pages` → `bundle exec jekyll build` → `npx --yes pagefind --site _site` → `actions/upload-pages-artifact` → `actions/deploy-pages` (Ruby via `ruby/setup-ruby@v1`, Node via `actions/setup-node`). **`actions/configure-pages` overrides `site.url`/`site.baseurl` from the live Pages settings** — for the custom domain that resolves to `https://www.nilesh42.science` once "Enforce HTTPS" is on; this override is the root cause of quirk #8.

Set **Settings → Pages → Build and deployment → Source = GitHub Actions** — otherwise the legacy "pages build and deployment" builder also fires on each push and can race-deploy a Pagefind-less (search-broken) build. There is also a legacy `.travis.yml` at the repo root — unused, do not edit.

## Architecture quirks that bite

### 1. `main.min.js` is loaded with `defer`, NOT `type="module"`
Because the script tag is `defer` (see `_includes/scripts.html:1`), the bundle is parsed as a classic script. **Top-level ES6 `import` / `export` statements in `_main.js` will become a `SyntaxError` and kill the entire bundle**, which silently breaks jQuery, navigation, smooth scroll, and the Follow button. If you need to pull in another module from `_main.js`, use dynamic `import('./thing.js').then(...)` instead. (This was the C-001 P0 regression catalogued in `docs/issues.md`.)

### 2. Three-mode theme cycle is a custom add-on
`assets/js/theme-cycle.js` runs AFTER `main.min.js`, unbinds the stock academicpages 2-mode click handler on `#theme-toggle`, and replaces it with a light → dark → sepia → light cycle. The preference lives in `localStorage["theme"]` and is read by an inline pre-paint script in `_includes/head.html` to avoid a flash of wrong theme. Sepia variables are defined in `_sass/_themes.scss`. Don't reintroduce a 2-mode toggle elsewhere — it'll race with this one.

**Note on element structure:** `id="theme-toggle"` is on the parent `<li>`; the actual interactive control is a nested `<button>` (it used to be `<a role="button">` directly with the id, but was converted for accessibility). Click handlers can bind to either since clicks bubble, but anything reading `tagName` of `#theme-toggle` will now see `LI`, not `A`.

### 3. Greedy-nav silently swallows any `<button>` inside the masthead
`assets/js/plugins/jquery.greedy-navigation.js` selects "the hamburger toggle" via `$('#site-nav > button')`. The `>` (direct-child) combinator is load-bearing — earlier the selector was `$('#site-nav button')` which also matched the nested `<button>` inside `#theme-toggle <li>`, causing the theme toggle to be hidden alongside the hamburger on desktop. If you add another `<button>` anywhere in the masthead (e.g. a search trigger, a settings popover), put it as a direct child of `#site-nav` only if you want it treated as a nav-toggle; otherwise nest it inside an `<li>` under `.visible-links` and verify it doesn't get `class="hidden"` and `count="N"` on page load.

### 4. Author identity has two sources of truth
- `_config.yml` `author:` block populates the **sidebar** (and the `site.author` Liquid variable everywhere)
- `_data/authors.yml` is consumed by `_layouts/single.html` for **per-post bylines** (when a post sets `author: <key>` in front matter)

They drift easily. When updating bio, title, handle, or avatar, check both.

### 5. `_config.dev.yml` only overrides `site.url`
It exists so generated **absolute** URLs (canonical, og:image) point at localhost during `jekyll serve` instead of the production domain. **Always pass both configs.** Same-origin assets and hero backgrounds are root-relative now (quirk #8), so they no longer depend on `site.url` — this override only affects the absolute SEO/Open-Graph URLs.

### 6. Sass deprecations are expected (for now)
A clean build prints ~354 Dart Sass slash-division warnings, originating from vendored `_sass/vendor/susy/` and `_sass/vendor/breakpoint/`. These are tracked but not actionable without forking those vendors. Dart Sass 3.0 will turn them into hard errors.

### 7. Search index requires a build, not just `serve`
`/search/` uses Pagefind. `jekyll serve` does not run Pagefind. To test search locally:

```bash
npm run build
bundle exec jekyll serve --skip-initial-build --no-watch --port 4000
```

### 8. Same-origin assets must be root-relative (`site.baseurl`), never `base_path`
`_includes/base_path` sets `base_path = site.url | append: site.baseurl`. The Pages build's `actions/configure-pages` overrides `site.url` with the live custom domain — and before HTTPS was enforced that was `http://www.nilesh42.science`. So any **same-origin asset** referenced as `{{ base_path }}/…` (CSS, JS, favicons, the avatar, hero `overlay_image` backgrounds) was emitted as an insecure `http://` URL and **blocked as mixed content** on the HTTPS page — the site rendered completely unstyled.

Rule: reference same-origin assets with `{{ site.baseurl }}/…` (or `| prepend: site.baseurl`) so they stay **root-relative** and inherit the page's scheme. Reserve `base_path` for URLs that must be absolute with the full domain — `og:image`/`twitter:image` (`_includes/seo.html`) and the social-share links (`_includes/social-share.html`). Fixed across `_includes/head.html`, `head/custom.html`, `scripts.html`, `author-profile.html`, and `page__hero.html`.

## Content surfaces

| Directory | Purpose |
|---|---|
| `_posts/` | Blog posts (year-month-day-slug.md). Front-matter `tags:`/`categories:` feed `/tag-archive/` and `/category-archive/`. |
| `_portfolio/` | Side-project pages, shown on `/portfolio/` (Side Projects). |
| `_publications/` | Papers; rendered on `/publications/`. |
| `_talks/`, `_teaching/` | Currently empty — pages are nav-hidden via `_config.yml` `show_talks: false` / `show_teaching: false`. |
| `_pages/` | All non-collection pages (about, cv, search, archives). |

## Audit & issue tracking

`docs/issues.md` is the **active audit catalogue** — start at **§1 "What to do next"** for current open work (grouped by who can do it: off-site / content commitment / mechanical TODO / deferred). §2 lists the user decisions (D-001…D-004 and R-005…R-008) that should be applied throughout. §3 summarises what's been fixed and the regressions caught during verification — read §3.3 before touching the masthead, theme toggle, or `jquery.greedy-navigation.js`, because the bugs there are non-obvious and have already bitten once. §5 is the full per-finding catalogue. Raw per-agent reports and the overseer's consolidation live under `docs/audit-artifacts/`.

Before proposing changes, search `docs/issues.md` for existing findings. After fixing an item, update its per-finding `status:` line (and the top-section commit log) rather than deleting the entry.

## License & attribution

MIT (see `LICENSE`). The theme is academicpages (itself a fork of Minimal Mistakes); attribution is currently in `_includes/footer.html`.
