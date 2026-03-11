# Migration Roadmap — So Simple → academicpages

> **Status**: phased execution plan. Reference [`specs.md`](./specs.md) for target state, schemas, and config shapes. This document describes **how + when**.

Each phase is independently verifiable and committed separately, so any phase can be rolled back without losing earlier work.

---

## Phase 0 — Preparation

- [ ] Create a working branch: `git checkout -b migration/academicpages-upgrade`.
- [ ] Tag the current `master` for rollback: `git tag pre-academicpages-migration`.
- [ ] Snapshot the current built site for visual comparison later:
  ```bash
  bundle exec jekyll build && cp -r _site _site.snapshot
  ```
- [ ] *(Optional)* Provide `scripts/dream11_authors.txt` — curated Dream11 co-author names, one per line — to widen the arxiv supplementary search. Google Scholar already covers most publications; this only widens preprint coverage. See [`specs.md` §15](./specs.md). Not blocking.

---

## Phase 1 — Skeleton swap

Delete So Simple files, then layer academicpages skeleton over the repo.

- [ ] Delete So Simple theme files:
  - `_layouts/`, `_includes/`, `_sass/`
  - `assets/css/`, `assets/js/`
  - `Gruntfile.js`, `package.json`, `.jshintrc`
  - `search.json`, `feed.xml`, `index.html`
  - Directory-style pages: `about/`, `articles/`, `blog/`, `resume/`, `search/`, `tags/` *(content is migrated to `_pages/` flat files later — preserve `resume/nilesh-patil.pdf` first by copying it to `files/nilesh-patil.pdf` in Phase 3)*
- [ ] Clone academicpages and rsync the skeleton in:
  ```bash
  git clone https://github.com/academicpages/academicpages.github.io.git /tmp/ap
  rsync -a \
    --exclude='.git' --exclude='_posts' --exclude='_publications' \
    --exclude='_talks' --exclude='_teaching' --exclude='_portfolio' \
    --exclude='images' --exclude='files' \
    /tmp/ap/ ./
  ```
- [ ] `bundle install && bundle exec jekyll serve` — base academicpages site boots locally.
- [ ] **Commit**: `skeleton: pull academicpages baseline`

---

## Phase 2 — Site configuration

Write the target `_config.yml`, navigation, feature flags, and dark-mode wiring.

- [ ] Write `_config.yml` per [`specs.md` §6](./specs.md): site identity, author (**no `email:` field**), plugins (including `jekyll-redirect-from`), collections, defaults, comments (`provider: giscus`, repo `nilesh-patil/nilesh-patil.github.io`, `theme: preferred_color_scheme`), feature flags (`show_talks: false`, `show_teaching: false`, `dark_mode: true`), `math_engine: katex`, `search: false` (Pagefind handles search via build step). No analytics block.
- [ ] Rewrite `_data/authors.yml` with Nilesh's profile (single author entry). Drop `email` key. Add `googlescholar: "https://scholar.google.co.in/citations?user=IIabY1sAAAAJ"`.
- [ ] Rewrite `_data/navigation.yml` per [`specs.md` §11](./specs.md): About → Publications → Portfolio → Blog → CV, plus flagged Talks + Teaching.
- [ ] Patch `_includes/masthead.html` to honor `requires_flag` (Liquid conditional from [`specs.md` §11](./specs.md)) and add the dark-mode toggle button + toggle script ([`specs.md` §11b](./specs.md)).
- [ ] Patch `_includes/head.html` with the pre-paint theme script (sets `data-theme` before body renders to prevent FOUC) ([`specs.md` §11b](./specs.md)).
- [ ] Add `_sass/_dark.scss` with token overrides for `[data-theme="dark"]`; import at end of `assets/css/main.scss`.
- [ ] Enable Discussions on the `nilesh-patil/nilesh-patil.github.io` repo via GitHub settings. Run [giscus.app](https://giscus.app) to get `repo_id` and `category_id`; paste into `_config.yml`.
- [ ] **Verify**:
  - Nav renders only the 5 unflagged items by default.
  - Setting `show_talks: true` and rebuilding makes Talks appear in nav.
  - Dark mode toggle works; preference persists; no FOUC.
  - giscus widget loads on a test post (try reacting from a GitHub-authed browser).
- [ ] **Commit**: `config: site identity, navigation, giscus, dark mode, feature flags`

---

## Phase 3 — Static assets port

Move images and the resume PDF into their new homes.

- [ ] Keep `images/ensembledme.jpg` at the same path. Confirm `_config.yml` author.avatar = `/images/ensembledme.jpg`.
- [ ] Keep `images/blog/**` verbatim — post bodies reference these by relative path and must keep working.
- [ ] Move resume PDF: `git mv resume/nilesh-patil.pdf files/nilesh-patil.pdf`.
- [ ] Remove now-empty `resume/` directory.
- [ ] **Verify** in dev server: `/images/ensembledme.jpg` and `/files/nilesh-patil.pdf` both 200 OK.
- [ ] **Commit**: `assets: relocate resume PDF; preserve images`

---

## Phase 4 — Blog post migration (6 posts, body preserved)

For each post in `_posts/blog/`:

- [ ] Rename `2017-01-14-visualizing-&-comparing-distributions.md` → `2017-01-14-visualizing-and-comparing-distributions.md`.
- [ ] Move all 6 files: `git mv _posts/blog/*.md _posts/`.
- [ ] Rewrite frontmatter per [`specs.md` §5](./specs.md) mapping table. **Body content stays byte-identical** — only the YAML at the top changes (and the K-means manual TOC removal noted below).
  - Add `redirect_from: [/blog/<old-slug>/]`.
  - Map `image.feature: blog/.../foo.jpg` → `header.overlay_image: /images/blog/.../foo.jpg` + `header.overlay_filter: 0.5`.
  - Add `math: true` only on posts that use math (start with the K-means post).
  - **Do NOT add `disqus_identifier`** — we're moving to giscus, which keys threads by pathname. Old Disqus comments are retired (not migrated).
- [ ] For the K-means post: set `toc: true` in frontmatter **and delete the manual TOC** at the top of the body. Auto-TOC supersedes it.
- [ ] **Verify each post** at `/posts/<slug>/`:
  - Hero overlay image renders with caption.
  - TOC, share, giscus widget visible.
  - Inline body images load.
  - Code blocks highlighted; KaTeX math rendered on the K-means post.
  - Old URL `/blog/<slug>/` returns 301 to new URL.
- [ ] **Commit**: `posts: port 6 blog posts to academicpages frontmatter`

---

## Phase 5 — Static pages port

Author the `_pages/` markdown.

- [ ] `_pages/about.md` — rewrite from resume PROFILE block, expanded with technical AI/ML skills section per [`specs.md` §10](./specs.md). **No email** in any contact section.
- [ ] `_pages/cv.md` — academicpages CV layout. **Web is canonical, PDF is a manual snapshot** (see [`specs.md` §10](./specs.md)). Populated from `resume/nilesh-patil.pdf` content (education, experience, selected publications, contact via GitHub + LinkedIn only — no email). Top-of-page link to `/files/nilesh-patil.pdf` with a one-line caveat note that PDF may lag the web version.
- [ ] `_pages/publications.html` — academicpages stock (lists `_publications/`).
- [ ] `_pages/portfolio.html` — academicpages stock (lists `_portfolio/`).
- [ ] `_pages/talks.html`, `_pages/teaching.html` — stock layouts; will render only when their flags are on.
- [ ] `_pages/year-archive.html`, `_pages/category-archive.html`, `_pages/tag-archive.html`, `_pages/404.md`.
- [ ] **Verify** each page renders at its `permalink`. Grep `_site` for `mailto:` — should appear only in social-icon stubs, never as a literal email address.
- [ ] **Commit**: `pages: about, cv, archives, taxonomies`

---

## Phase 6 — Publications seeding

Hand-write the 6 known publications, then auto-fetch more from Google Scholar (primary) + arxiv (supplementary).

- [ ] Create `_publications/` directory.
- [ ] Hand-write 6 seed entries per [`specs.md` §8](./specs.md) — the 3 Dream11 + 3 Rochester papers from the resume PDF. Use the schema from [`specs.md` §8](./specs.md).
- [ ] Write `scripts/fetch_publications.py` with **two sources**:
  - **Source 1 — Google Scholar** via `scholarly` lib, profile `IIabY1sAAAAJ`. Output: `_publications/<title-slug>.md` with prefilled frontmatter (title, authors, venue, year, paper_url, excerpt).
  - **Source 2 — arxiv API** for Dream11 preprints not yet on Scholar. Inputs: base author `"Nilesh Patil"` + optional `scripts/dream11_authors.txt`. Dedupe against Scholar output by title-similarity (≥ 0.85 Jaccard, lowercased, punctuation-stripped).
  - **Idempotent**: skip files that already exist.
  - `--refresh` flag updates `excerpt`, `authors`, `venue`, `paper_url`, citation counts on existing files **without overwriting** manual edits to `citation`, `code_url`, body, or `tags`.
  - **Graceful failure**: if Scholar throttles, log and exit cleanly — no partial writes.
- [ ] Add `scripts/requirements.txt` with **pinned** versions:
  - `scholarly==1.7.11` (pin — DOM-scraper, breaks on Scholar UI changes; bump on breakage)
  - `arxiv==2.1.0`
  - `python-frontmatter==1.0.1`
- [ ] Run the script: `python scripts/fetch_publications.py`.
- [ ] Review generated entries; manually dedupe against the 6 hand-written seeds (especially the ICMLA 2023 churn paper and EMBC 2019 Doppler paper, which both have publisher URLs that Scholar may surface).
- [ ] Add a Liquid filter or `_includes/citation.html` partial that bolds "Nilesh Patil" (case-insensitive, matches "N. Patil" / "Patil, N." variants) in rendered author lists.
- [ ] **Verify**: `/publications/` lists all entries, every `paper_url` clicks through, citations render with "Nilesh Patil" bolded.
- [ ] **Commit**: `publications: seed from resume + scholar + arxiv auto-fetch`

---

## Phase 7 — Portfolio / Talks / Teaching scaffolding

Stand up the collections so flipping the flag later is a one-line change.

- [ ] Create `_portfolio/`, `_talks/`, `_teaching/` directories.
- [ ] Add `.gitkeep` to `_talks/` and `_teaching/`.
- [ ] Seed `_portfolio/` with **1–2 real code-project entries** per [`specs.md` §9](./specs.md) — only entries that have a public GitHub repo. Candidate: a standalone repo of the distributed K-means demo (if one exists). Each entry needs hero image, repo link, problem statement, approach in 1–3 paragraphs. **No Medium imports, no prose-only case studies** (those are out of scope for portfolio).
- [ ] Add one draft entry each in `_talks/` and `_teaching/` (so layout previews aren't broken when flags flip).
- [ ] **Verify**:
  - With `show_talks: false`, Talks is **not** in the nav, but `/talks/` is reachable by direct URL (for preview).
  - Flipping `show_talks: true` exposes the nav link.
  - `/portfolio/` lists seeded entries with hero teasers.
- [ ] **Commit**: `collections: scaffold portfolio/talks/teaching`

---

## Phase 8 — Comments, redirects, search, math wiring (no analytics)

- [ ] **giscus**: confirm GitHub Discussions is enabled on `nilesh-patil/nilesh-patil.github.io`. Run [giscus.app](https://giscus.app) to generate `repo_id` and `category_id`; paste into `_config.yml` per [`specs.md` §6](./specs.md). Load a post locally and verify the widget renders and accepts a test reaction from a GitHub-authed browser.
- [ ] **giscus theme sync**: confirm widget theme follows the dark-mode toggle — toggling site theme re-themes the giscus iframe via `postMessage`. (Add the postMessage call in the toggle handler from [`specs.md` §11b](./specs.md) if not already there.)
- [ ] **Redirects**: confirm `jekyll-redirect-from` emits `_site/blog/<old-slug>/index.html` redirect stubs for all 6 posts. Hit one in a browser — should 301 to `/posts/<slug>/`.
- [ ] **No analytics**: confirm browser Network tab shows zero GA / gtag / Plausible / GoatCounter / Cloudflare analytics requests in `<head>` or `<body>`. Confirm `_includes/analytics-providers/` directory is either gone or its content is not referenced.
- [ ] **Pagefind search**:
  - Add Pagefind as a post-build step. Locally: `bundle exec jekyll build && npx pagefind --site _site`.
  - Add the GH Pages Action step: after Jekyll builds, run `npx pagefind --site _site` and include `_site/pagefind/` in the deploy artifact.
  - Wire the search box per [`specs.md` §12](./specs.md) — drop `<link>`, `<script>`, and `<div id="search">` into `_includes/search/search_box.html` (or a custom search include). Remove the lunr-driven search include.
  - Confirm the search box returns hits across posts + publications + portfolio. Confirm no external network call fires on query (all index assets are same-origin).
- [ ] **KaTeX**: confirm math renders on the K-means post (which sets `math: true`). KaTeX assets load only on math-enabled pages, not site-wide.
- [ ] **No public email**: grep `_site` for `nilesh5760@gmail.com` and `mailto:nilesh5760` — should return zero hits.
- [ ] **Commit**: `wiring: giscus + redirects + pagefind + katex verified`

---

## Phase 9 — Full-site verification

Run the [`specs.md` §13](./specs.md) verification checklist top to bottom.

- [ ] `bundle exec jekyll build && npx pagefind --site _site` — zero warnings, zero errors from either step.
- [ ] Visual diff: spot-check each post against `_site.snapshot` for hero image presence and content body byte-equality (note: the K-means post's body changes by exactly the manual-TOC removal).
- [ ] Mobile viewport check at 375px — verify in **both light and dark**.
- [ ] Dark mode: toggle, reload, confirm preference persists; clear `localStorage`, reload, confirm OS preference is restored.
- [ ] Broken-link scan: `bundle exec htmlproofer ./_site --ignore-status-codes "0,403,429"` (external arxiv / DOI may rate-limit — acceptable). Google Scholar URLs commonly 429 — ignore.
- [ ] Push branch to GitHub; verify the GitHub Actions / GH Pages build succeeds (including the Pagefind step).
- [ ] **Commit** any final fixes.

---

## Phase 10 — Cutover

- [ ] Open PR `migration/academicpages-upgrade` → `master`. Attach screenshots and the Phase 9 checklist output.
- [ ] Merge after manual approval.
- [ ] Monitor the GitHub Pages deploy in the Actions tab.
- [ ] **Verify in production**:
  - New `/posts/<slug>/` URLs load for all 6 posts.
  - Old `/blog/<slug>/` URLs redirect (HTTP 301) to new locations.
  - `/publications/` lists entries.
  - `/cv/` renders; resume PDF download works.
- [ ] Delete the `_site.snapshot` directory locally — no longer needed.

---

## Phase 11 — Post-launch (deferred work)

Flag-gated content and ongoing curation, done as separate PRs:

- [ ] Populate `_talks/` with real entries; flip `show_talks: true`.
- [ ] Populate `_teaching/` with real entries; flip `show_teaching: true`.
- [ ] Fill `scripts/dream11_authors.txt` and re-run `python scripts/fetch_publications.py --refresh` to widen Dream11 preprint coverage.
- [ ] *(Optional)* Import Medium articles into `_portfolio/`.
- [ ] *(Optional)* Add auto-PDF generation for `/cv/` (Paged.js + headless Chromium, or weasyprint) so the PDF stays in sync with the web CV without manual export.
- [ ] *(Optional)* Reconsider custom domain (`nilesh-patil.com` or similar) — keeping `github.io` for now per [`specs.md` §1](./specs.md).

---

## Rollback procedure

If anything goes wrong post-merge:
```bash
git revert <merge-commit>
# or, for full reset:
git checkout pre-academicpages-migration
```
Tag `pre-academicpages-migration` was created in Phase 0 specifically for this.

---

## Critical files touched (cross-reference)

| Path | Action |
|---|---|
| `_config.yml` | Full rewrite (Phase 2) |
| `_data/{navigation,authors,ui-text}.yml` | Rewrite (Phase 2) |
| `_layouts/*`, `_includes/*`, `_sass/*` | Replaced by academicpages (Phase 1) |
| `_pages/{about,cv,publications,portfolio,talks,teaching,*-archive,404}` | New (Phase 5) |
| `_posts/*.md` (6 files) | Frontmatter rewritten; body preserved (Phase 4) |
| `_publications/*.md` | New (Phase 6) |
| `_portfolio/`, `_talks/`, `_teaching/` | Scaffold (Phase 7) |
| `scripts/{fetch_publications.py,dream11_authors.txt,requirements.txt}` | New (Phase 6) |
| `files/nilesh-patil.pdf` | Moved from `resume/` (Phase 3) |
| `assets/css/main.scss`, `_sass/_dark.scss` | academicpages base + dark-mode overrides (Phase 1 + 2) |
| `_includes/{head.html,masthead.html}` | Patched for dark-mode pre-paint + toggle (Phase 2) |
| `_includes/search/search_box.html` | Pagefind wiring, lunr removed (Phase 8) |
| GH Pages Action (or workflow file) | Adds `npx pagefind --site _site` post-build step (Phase 8) |
