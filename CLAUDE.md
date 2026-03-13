# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Personal Jekyll site at `nilesh-patil.github.io`, built on the **academicpages** theme — but the theme is **not vendored**. Every layout, include, Sass partial, and JS file is in-repo and has been customized. Treat the whole tree as project code, not theme code you shouldn't touch.

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
# site.url so canonical/og:image/hero background URLs resolve to localhost
# instead of https://nilesh-patil.github.io.
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
Pushing to `master` triggers `.github/workflows/pages.yml` (Ruby 3.3, `actions/jekyll-build-pages` + `npx --yes pagefind` + `actions/deploy-pages`). There is also a legacy `.travis.yml` at the repo root — unused, do not edit.

## Architecture quirks that bite

### 1. `main.min.js` is loaded with `defer`, NOT `type="module"`
Because the script tag is `defer` (see `_includes/scripts.html:1`), the bundle is parsed as a classic script. **Top-level ES6 `import` / `export` statements in `_main.js` will become a `SyntaxError` and kill the entire bundle**, which silently breaks jQuery, navigation, smooth scroll, and the Follow button. If you need to pull in another module from `_main.js`, use dynamic `import('./thing.js').then(...)` instead. (This was the C-001 P0 regression catalogued in `docs/issues.md`.)

### 2. Three-mode theme cycle is a custom add-on
`assets/js/theme-cycle.js` runs AFTER `main.min.js`, unbinds the stock academicpages 2-mode click handler on `#theme-toggle`, and replaces it with a light → dark → sepia → light cycle. The preference lives in `localStorage["theme"]` and is read by an inline pre-paint script in `_includes/head.html` to avoid a flash of wrong theme. Sepia variables are defined in `_sass/_themes.scss`. Don't reintroduce a 2-mode toggle elsewhere — it'll race with this one.

### 3. Author identity has two sources of truth
- `_config.yml` `author:` block populates the **sidebar** (and the `site.author` Liquid variable everywhere)
- `_data/authors.yml` is consumed by `_layouts/single.html` for **per-post bylines** (when a post sets `author: <key>` in front matter)

They drift easily. When updating bio, title, handle, or avatar, check both.

### 4. `_config.dev.yml` only overrides `site.url`
It exists so generated absolute URLs (canonical, og:image, masthead asset src, hero overlay background-image) point at localhost during `jekyll serve`. **Always pass both configs** — otherwise hero backgrounds load from the live production URL while you're editing.

### 5. Sass deprecations are expected (for now)
A clean build prints ~354 Dart Sass slash-division warnings, originating from vendored `_sass/vendor/susy/` and `_sass/vendor/breakpoint/`. These are tracked but not actionable without forking those vendors. Dart Sass 3.0 will turn them into hard errors.

### 6. Search index requires a build, not just `serve`
`/search/` uses Pagefind. `jekyll serve` does not run Pagefind. To test search locally:

```bash
npm run build
bundle exec jekyll serve --skip-initial-build --no-watch --port 4000
```

## Content surfaces

| Directory | Purpose |
|---|---|
| `_posts/` | Blog posts (year-month-day-slug.md). Front-matter `tags:`/`categories:` feed `/tag-archive/` and `/category-archive/`. |
| `_portfolio/` | Side-project pages, shown on `/portfolio/` (Side Projects). |
| `_publications/` | Papers; rendered on `/publications/`. |
| `_talks/`, `_teaching/` | Currently empty — pages are nav-hidden via `_config.yml` `show_talks: false` / `show_teaching: false`. |
| `_pages/` | All non-collection pages (about, cv, search, archives). |

## Audit & issue tracking

`docs/issues.md` is the **active audit catalogue** — a prioritized list of bugs, accessibility issues, brand consistency findings, and content gaps with concrete fix snippets. Before proposing changes, search it for existing findings. After fixing an item, mark it resolved there rather than deleting. Raw per-agent reports and the overseer's consolidation live under `docs/audit-artifacts/`.

## License & attribution

MIT (see `LICENSE`). The theme is academicpages (itself a fork of Minimal Mistakes); attribution is currently in `_includes/footer.html`.
