# Performance Recon: Invariants & Constraints

Captured from `docs/specs.md`, `docs/ROADMAP.md`, and git history before
any optimization work. The implementation worktrees must not violate
anything below.

## 1. Hard Invariants (MUST hold)

- **Zero outbound telemetry.** No GA / gtag / Plausible / GoatCounter /
  Cloudflare analytics. Verified by network-tab check in spec §9.
- **No public email.** No `email:` key in `_config.yml`; no `mailto:`
  links in `_site`.
- **Domain stays `nilesh-patil.github.io`.**
- **Feature flags `show_talks` / `show_teaching` remain `false`** —
  flipping is Phase 11.
- **Permalink structure fixed.** Posts at `/posts/<slug>/`; legacy
  `/blog/<slug>/` redirect via `jekyll-redirect-from`.
- **Pagefind is build-time, same-origin.** Index lives at
  `_site/pagefind/`. No external network call may fire on query.
- **KaTeX is conditional.** `page.math: true` only. Do not promote to
  site-wide.
- **giscus threads keyed by `pathname`** — changing permalinks orphans
  threads.
- **3-mode theme cycle (light → dark → sepia)** added in `ac6a8c9` is
  intentional. `data-theme`, the pre-paint FOUC script in
  `_includes/head.html`, `assets/js/theme-cycle.js`, and
  `_sass/theme/_default_sepia.scss` are load-bearing.
- **Sass `style: compressed`** in `_config.yml` — do not unset.
- **`jekyll-include-cache` plugin** is a deliberate perf plugin — keep.
- **`remote_theme: false`** — all theme files local; do not enable.
- **`search: false`** — academicpages lunr search disabled in favor of
  Pagefind.
- **No visual customization.** Stock academicpages Sass tokens only.
  The 3-mode theme cycle is the only sanctioned visual change.

## 2. Deferred Work (do NOT implement as side effects)

- Populating `_talks/`, `_teaching/` and flipping their flags.
- `scripts/dream11_authors.txt`.
- Medium article import into `_portfolio/`.
- Auto-PDF for `/cv/`.
- Custom domain.
- Any analytics — explicitly ruled out, not just deferred.
- Newsletter / email subscription.
- Multi-language.
- Migrating old Disqus threads.

## 3. Already-Done Optimizations (do not redo or unwind)

- MathJax + Mermaid removed; KaTeX guarded by `page.math` (`877b170`).
- lunr removed; Pagefind wired (`877b170`).
- Double-`/images/` path bug fixed in `page__hero.html`, `seo.html`,
  `archive-single.html`.
- `sass.style: compressed` enabled.
- `jekyll-include-cache` enabled.
- npm cache step removed from GH Actions (`2d30a6f`).
- 3-mode theme cycle landed in `ac6a8c9` with warm-charcoal dark palette
  (`#1d2128`, WCAG AAA ~11.5:1). Do not revert palette choices.
- Duplicate manual TOC blocks removed from K-means + Numpy posts.
- `math: true` added to K-means post.

## 4. Conventions to Honor

- Commit style: `<scope>: <short description>` + multi-line body
  citing spec section. Scopes seen: `theme`, `config`, `wiring`, `posts`,
  `publications`, `collections`, `pages`, `assets`, `skeleton`, `audit`,
  `verify`, `chore`, `docs`. Add `perf` for this initiative.
- Branch naming: `<category>/<short-description>`. Integration branch
  for this initiative: `perf/orchestrated-optimization`.
- One atomic, verifiable commit per logical change. Each phase rollback-
  able.
- Post body content immutable (frontmatter rewrites only).

## 5. Open Issues / Known Gaps

- giscus `repo_id` / `category_id` empty in `_config.yml` — comment
  widget non-functional until filled.
- `scholarly==1.7.11` pin will need bumping when Google Scholar DOM
  changes (~every 6 mo).
- Upstream Sass deprecation warnings — from academicpages tree, not
  site-level code.
- LinkedIn URL missing — footer omits intentionally.
- `htmlproofer` external-URL 429s on arxiv/DOI/Scholar — acceptable.
- **No prior perf baseline.** Git log has zero `perf|lighthouse|optimize`
  commits. Clean slate but no prior measurement to compare against —
  baseline is being captured now.
