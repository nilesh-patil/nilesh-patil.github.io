# Consolidated Audit — nilesh-patil.github.io
**Consolidator:** Principal Engineer / Design Director hybrid  
**Date:** 2026-05-23  
**Source reports:** 01-designer, 02-webdev, 03-ai-expert, 04-hiring, 05-profile, 06-blogging

---

## Verdict

The site is technically sound and deployed cleanly — no build drift, no broken pages from infrastructure failures — but it functions as a dormant 2017 portfolio wearing a 2026 job title: every page that matters to a recruiter, peer, or collaborator finds a content gap, a broken widget, or a stale social link where proof of current work should be. The single highest-risk condition is the Medium sidebar link resolving to a food-recipe blog, which actively contradicts the "AI systems & applied research" positioning to anyone who clicks it. The single highest-opportunity action is publishing one substantive post on current work, which simultaneously updates the archive's last-modified signal, validates the stated seniority, gives the comments widget and share buttons something worth using, and seeds the RSS feed — six outcomes for one effort.

---

## Section 1 — Convergent Findings (HIGH confidence: 2+ agents, verified)

Findings appear in descending agent-count order. All claims verified against codebase unless noted.

---

### C-01 — "Follow" button: dead affordance, no ARIA disclosure state
**Agents:** designer (P0), webdev (P1), hiring (P0) — 3 agents  
**Verified:** `_includes/author-profile.html:39` — `<button class="btn btn--inverse">Follow</button>`. No `aria-expanded`, no `aria-controls`. Button toggles `.author__urls` per template JS but is semantically opaque to assistive tech.  
**Confidence:** HIGH  
**Severity:** P0 (UX + accessibility)  
**Impact:** First interactive element on mobile. Label implies a subscription feed. Fails WCAG 2.2 AA 4.1.2 disclosure pattern. Two separate problems: wrong label and missing ARIA.

---

### C-02 — Giscus comments widget: broken placeholder visible on every post
**Agents:** webdev (P0), blogging (P0) — 2 agents  
**Verified:** `_config.yml:199` — `repo_id: ""` and `_config.yml:201` — `category_id: ""`. Template guard in `_includes/comments-providers/giscus.html` renders a yellow warning banner when either is empty.  
**Confidence:** HIGH  
**Severity:** P0  
**Impact:** Every post page shows a yellow "Comments are not yet wired" notice. Signals an unfinished site to every reader.

---

### C-03 — LinkedIn absent from every surface; CV contains a dead breadcrumb
**Agents:** ai-expert (P1), hiring (P0), profile (P1), blogging (P1) — 4 agents  
**Verified:** `_config.yml:34` — `linkedin:` commented out. `_pages/cv.md:94` — `*(see GitHub profile for current link)*`. GitHub profile not audited for a LinkedIn link, making this a multi-hop dead end at maximum recruiter intent.  
**Confidence:** HIGH  
**Severity:** P0 (blocks primary recruiter inbound channel)

---

### C-04 — Medium handle links to a food-recipe blog, not AI content
**Agents:** ai-expert (P1), profile (P0) — 2 agents  
**Verified (external, profile-agent):** Medium @ensembledme profile bio: "Lover of music, travel, and fashion." 10 posts, all food recipes, last published March 2024. Cannot re-scrape here but profile-agent pulled this via WebFetch with specifics (recipe titles). I accept this as confirmed.  
**Verified in codebase:** Medium linked at `_config.yml:33`, `_config.yml:244`, `_pages/about.md:42` — three distinct surfaces.  
**Confidence:** HIGH  
**Severity:** P0 (active reputational risk)  
**Impact:** Three-link surface area pointing a "Head of AI" audience to muffin recipes.

---

### C-05 — No `og:image`: social card is blank on every share
**Agents:** designer (P2), webdev (P2), profile (P0) — 3 agents  
**Verified:** `_includes/seo.html:131` — `{% if site.og_image %}` block emits OG image. `_config.yml` — no `og_image:` key. Non-article pages (home, about, CV, publications) have no `page.header.image` either. Zero `<meta property="og:image">` tags emitted on those pages. Twitter card also missing (C-06 below).  
**Confidence:** HIGH  
**Severity:** P1

---

### C-06 — Twitter/X card tags never emitted; top-level `twitter:` key absent
**Agents:** webdev (P1), profile (P1) — 2 agents  
**Verified:** `_includes/seo.html:51` guards on `site.twitter.username`. `_config.yml` has `twitter:` only under `author:` (line 32), not at top-level. No `<meta name="twitter:*">` tags emitted.  
**Confidence:** HIGH  
**Severity:** P1

---

### C-07 — Person JSON-LD never emitted; `site.social` block absent from `_config.yml`
**Agents:** webdev (P1), hiring (P1), profile (P0) — 3 agents  
**Verified:** `_includes/seo.html:89` — `{% if site.social %}`. No `social:` key in `_config.yml`. Additionally, both JSON-LD blocks (lines 92 and 134) use `"http://schema.org"` not `"https://schema.org"`.  
**Confidence:** HIGH  
**Severity:** P1 (SEO + knowledge graph)

---

### C-08 — Content freshness: 6-year post gap actively signals disengagement
**Agents:** ai-expert (P0), hiring (P0), profile (P1), blogging (P0) — 4 agents  
**Verified:** `_posts/` directory contains 6 files dated 2017 and one dated 2020. Nothing from 2021–2026.  
**Confidence:** HIGH  
**Severity:** P0 (content)  
**Impact:** "Recent posts" on home page leads with a 2020 Spark/Dask tutorial and five 2017 posts while the bio claims Head of AI + ACL 2026 authorship.

---

### C-09 — Talks page accessible via direct URL; renders as an empty shell
**Agents:** designer (P0), ai-expert (P1), hiring (P1) — 3 agents  
**Verified:** `_config.yml:231` — `show_talks: false`. `_talks/` contains only `.gitkeep` and `2099-01-01-draft-talk.md`. The `/talks/` URL builds successfully (nav-hidden but sitemap-included). CV mentions Columbia University "Sports x AI" sessions.  
**Confidence:** HIGH  
**Severity:** P1

---

### C-10 — Footer attribution: "Powered by Jekyll & AcademicPages, a fork of Minimal Mistakes"
**Agents:** designer (P1), ai-expert (P2), hiring (P2), blogging (P2) — 4 agents  
**Verified:** `_includes/footer.html:30` — exact text confirmed.  
**Confidence:** HIGH  
**Severity:** P2  
**Impact:** Appears on every page including the CV. No attribution required by AcademicPages MIT license for rendered output.

---

### C-11 — "Currently exploring" section duplicates the About opening paragraph verbatim
**Agents:** ai-expert (P2), hiring (P1), profile (P2) — 3 agents  
**Verified:** `_pages/about.md:12` — "Particularly interested in AI harness design, developer productivity, and turning emerging model capabilities into reliable workflows and products." `_pages/about.md:34` — identical text in "Currently exploring" section.  
**Confidence:** HIGH  
**Severity:** P2

---

### C-12 — Portfolio is two items (2018 Docker file + 2024 k-means post); reads as abandoned
**Agents:** ai-expert (P2), hiring (P1), profile (P1) — 3 agents  
**Verified:** `_portfolio/` directory contains exactly two files. Neither substantiates Head-of-AI-level work.  
**Confidence:** HIGH  
**Severity:** P2

---

### C-13 — Technical errors in k-means post: `partial_fit` call + `k-means||` mislabeled
**Agents:** ai-expert (P1×2), blogging (P1×2) — 2 agents  
**Verified:**  
- `_posts/2020-05-20-distributed-kmeans-clustering.md:379` — `kmeans.partial_fit(batch)` called on a `dask_ml.cluster.KMeans` instance. `dask_ml.cluster.KMeans` does not implement `partial_fit`. This raises `AttributeError` at runtime.  
- Line 203: `initMode="k-means||"` used inside a branch labeled `"k-means++ initialization"` (comment at line 202: `# Use PySpark's default k-means++ initialization`). k-means|| (Bahmani et al. 2012) is a distributed approximation to k-means++, not the same algorithm.  
- `find_elbow_point()` has a dead `return k_range, inertias` at line 483 after the actual return at line 502.  
**Confidence:** HIGH  
**Severity:** P1 (broken code published under a "production-relevant" framing)

---

### C-14 — `sns.distplot()` removed in seaborn 0.12; code throws `AttributeError`
**Agents:** ai-expert (P0 for code), blogging (article-level) — 2 agents  
**Verified:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md:70-72` — three calls to `sns.distplot()`. Additionally `scolumns_order` at line 194 is referenced as `columns_order` at line 206 — NameError.  
**Confidence:** HIGH  
**Severity:** P1 (code is visibly broken if any reader runs it)

---

### C-15 — GitHub bio still shows "Sr. Principal Research Scientist @Dream11" (stale)
**Agents:** profile (P0) — 1 agent, but cross-confirmed by hiring (implicit)  
**Confidence:** MED (external surface; cannot verify via codebase read; profile-agent pulled via GitHub API)  
**Severity:** P1 — recruiter cross-checking the site → GitHub link sees title mismatch.

---

## Section 2 — Unique Findings Worth Keeping (single-agent, verified)

---

### U-01 — `main.min.js` loaded as `type="module"`: breaks jQuery bundle, race with `theme-cycle.js`
**Agent:** webdev (P0)  
**Verified:** `_includes/scripts.html:1` — `<script type="module" src="...main.min.js"></script>`. ES modules are implicitly deferred, execute in strict mode, and scope their bindings — incompatible with jQuery globals. `theme-cycle.js` uses `defer` and relies on `$(document).ready` ordering; that ordering guarantee breaks with `type="module"` on the main bundle.  
**Confidence:** HIGH  
**Severity:** P0 (runtime)  
**Fix:** Replace `type="module"` with `defer`.

---

### U-02 — Skip-to-content link: CSS exists, HTML never rendered
**Agent:** webdev (P1)  
**Verified:** Searched `_layouts/` and `_includes/` — zero matches for `skip-link` or `screen-reader-shortcut` in rendered HTML. CSS class `.skip-link` exists in `_sass/include/_utilities.scss`. WCAG 2.2 AA 2.4.1 violation.  
**Confidence:** HIGH  
**Severity:** P1

---

### U-03 — Theme toggle: `<a role="button">` without Space-key handler; no `aria-live` announcement
**Agent:** webdev (P1×2)  
**Verified:** `_includes/masthead.html:33` — `<a href="#" role="button" ...>`. `theme-cycle.js` binds only `addEventListener("click", ...)` (no `keydown` for Space). No `aria-live` region. Two separate WCAG failures.  
**Confidence:** HIGH  
**Severity:** P1

---

### U-04 — Dart Sass slash-division: 354 deprecation warnings, build breaks in Sass 2.0
**Agent:** webdev (P1)  
**Verified (partial):** `_sass/include/_mixins.scss` present; grep for `math.div` returned zero matches, confirming the fix has not been applied. Susy and breakpoint libraries are vendored and unmaintained. Cannot count warnings without running `jekyll build`; claim is plausible and the fix is well-documented.  
**Confidence:** MED  
**Severity:** P1

---

### U-05 — `.travis.yml` targeting Ruby 2.1 (EOL 2017) at project root
**Agent:** webdev (P1)  
**Verified:** `.travis.yml` exists (204 bytes), first line `language: ruby`. Active CI is `.github/workflows/pages.yml`. File is dead weight and a potential confusion artifact.  
**Confidence:** HIGH  
**Severity:** P2 (low risk but zero reason to keep it)

---

### U-06 — `meta[name="theme-color"]` hardcoded `#ffffff`; wrong for dark and sepia modes
**Agent:** webdev (P2)  
**Verified:** `_includes/head/custom.html:12` — `<meta name="theme-color" content="#ffffff"/>`. Single tag, no media query variant.  
**Confidence:** HIGH  
**Severity:** P2

---

### U-07 — No `<link rel="preload">` for LCP image (author avatar)
**Agent:** webdev (P1)  
**Verified:** `_includes/head.html` — no `preload` for `ensembledme.webp`. `fetchpriority="high"` not found in head either. The avatar is the first paint-blocking image on every page.  
**Confidence:** HIGH  
**Severity:** P1

---

### U-08 — Syntax highlighting tokens: Solarized Light only; dark mode code unreadable
**Agent:** designer (P2)  
**Verified:** `_sass/_syntax.scss` uses fixed Solarized Light color values. `--global-code-background-color` in dark mode is dark (`#161a20`) but token colors (yellows, greens, blues) are hardcoded for a white background. High contrast failure in dark mode.  
**Confidence:** HIGH  
**Severity:** P2

---

### U-09 — Code blocks: double font-size scaling (~0.94em effective)
**Agent:** designer (P1)  
**Verified:** `_sass/_syntax.scss:13` outer container at `$type-size-4` (1.25em equivalent); inner `.highlight` at `$type-size-6` (0.75em). Net size is near body text size rather than clearly subordinate.  
**Confidence:** HIGH  
**Severity:** P2

---

### U-10 — No `og:type` fallback on non-article pages; no `og:description`
**Agent:** webdev (P2), profile (P1)  
**Verified:** `_includes/seo.html:119-121` — `og:type: article` emitted only when `page.date` is set. Home, about, publications, CV pages emit no `og:type`. `og:description` only emitted from `page.excerpt` or `site.og_description` (neither set in `_config.yml`).  
**Confidence:** HIGH  
**Severity:** P2

---

### U-11 — No Google Search Console verification; sitemap not consumed by any crawler
**Agent:** profile (P1)  
**Verified:** `_includes/seo.html:101` — conditional block for `site.google_site_verification` exists; `_config.yml` has no such key. `jekyll-sitemap` plugin is active (implied by sitemap reference). No verification = Google has no confirmed signal to index.  
**Confidence:** HIGH  
**Severity:** P2

---

### U-12 — CV PDF disclaimer ("may lag by weeks") is trust-eroding
**Agent:** hiring (P1)  
**Confidence:** MED (not directly verifiable from source files; reasonable claim)  
**Severity:** P2

---

### U-13 — Biology paper trailing-author position unexplained for CS/ML audience
**Agent:** ai-expert (P1)  
**Confidence:** HIGH  
**Severity:** P2

---

### U-14 — Post index: no tags visible; `/tag-archive/` unlinked
**Agent:** blogging (P1)  
**Confidence:** MED (template audit; plausible without full render verification)  
**Severity:** P2

---

### U-15 — No "Now" page; no active intellectual signal
**Agent:** blogging (P1)  
**Confidence:** HIGH (absence is confirmed)  
**Severity:** P2

---

### U-16 — Positioning headline not memorable; no declared stance
**Agent:** ai-expert (P1), hiring (P1) — 2 agents, consistent  
**Confidence:** HIGH  
**Severity:** P1 (content/brand)

---

### U-17 — X/Twitter sidebar link uses `twitter.com`; account activity unconfirmed (HTTP 402)
**Agent:** profile (P1, P2)  
**Confidence:** LOW (cannot verify externally in this run)  
**Severity:** P2

---

## Section 3 — Contradictions and Tensions

---

### T-01 — "Follow" button: label change vs. ARIA disclosure vs. LinkedIn wire-up
**Designer** wants label changed to "Connect" (5 min edit).  
**Webdev** wants the button replaced with a proper `aria-expanded`/`aria-controls` disclosure widget (or `<details>`/`<summary>`).  
**Hiring** wants it wired directly to a LinkedIn URL when LinkedIn is added.

**Resolution:** These are three layers of the same problem and all three are correct. Do them in order:  
1. Wire LinkedIn first (C-03) — once you have the URL, the "Follow" button destination is clear.  
2. Change label to "Connect" and add `aria-expanded="false"` + `aria-controls="author-urls"` to the existing button (satisfies designer + webdev without a full rewrite).  
3. If LinkedIn is added and the button should navigate directly rather than toggle, replace the disclosure button with a plain `<a>` link to LinkedIn, removing the toggle entirely. This is the cleanest end-state: icon links only, no pseudo-dropdown.  
The `<details>`/`<summary>` native approach is valid but requires CSS regression testing on the sidebar; skip unless a full sidebar refactor is planned.

---

### T-02 — "Publish posts" (AI-expert) vs. "publish one big post" (blogging)
**AI-expert** recommends two substantive posts within 30 days.  
**Blogging** recommends one 2,500-word long-form post as the single most important move.

**Resolution:** Not actually a contradiction. These are the same recommendation at different granularities. Blogging's framing is operationally correct: one high-quality post is more achievable and has higher per-post impact than two rushed ones. Prioritize one 2,000–3,000 word post on a current-work topic (compliance-aware AI harness or ACL 2026 companion). The second post can follow once the first is live. Accept blogging's framing.

---

### T-03 — Medium: remove it (profile) vs. verify first (ai-expert)
**Profile** says remove Medium immediately (option b, 5 minutes).  
**AI-expert** says verify Medium content first.

**Resolution:** Profile-agent already performed the verification — Medium @ensembledme is a food-recipe blog. There is nothing to verify further. Remove `medium:` from `_config.yml` author block and `_data/authors.yml`. Remove from `_pages/about.md:42`. Remove from `footer.links` in `_config.yml`. This is a 5-minute edit and eliminates active reputational risk. If AI writing on Medium is a future goal, create a separate handle then.

---

### T-04 — Typography: editorial serif (designer) vs. no font change needed
**Designer** recommends introducing an editorial serif for h1/h2 to differentiate from "GitHub Pages 2017."  
No other agent raised this; no explicit contradiction, but implied tension with the "ship practical fixes first" philosophy.

**Resolution:** Valid P1 aesthetic improvement; ranked lower than functional/content gaps. If implemented, Source Serif 4 or DM Serif Display via Google Fonts with `preconnect` is the right approach. A single CSS line change + font `<link>` in `head.html`. Do after functional P0s and P1s are resolved.

---

## Section 4 — Findings Dropped or Downgraded

---

### D-01 — Designer claim: "Avatar is a group photo" — UNVERIFIABLE from source
**Original claim:** `images/ensembledme.webp` shows a cropped group photo at 110px.  
**Assessment:** Cannot read the image file directly to confirm "group photo" interpretation. File exists. Whether it's a group photo is a judgment call on the image content that only the owner can verify. Kept as a low-confidence reminder but not actionable without owner confirmation. **Downgraded to LOW / deferred.**

---

### D-02 — Designer claim: layout uses Susy `span(8 of 12)` — PARTIALLY CONFIRMED
**Original claim:** `_sass/layout/_page.scss` lines 32–34 use `span(8 of 12)`.  
**Verified:** `_page.scss:32` — `@include span(8 of 12)` confirmed in source. The underlying concern (article column too narrow at 1440px) is real. The specific line numbers are accurate. However the recommendation to "convert to CSS Grid" is a significant refactor with regression risk on 12 layout variants — rank as 1-week effort, not 1-hour. Kept at P2, effort re-tagged to `1w`.

---

### D-03 — Webdev claim: JSON-LD `@context` uses `http://schema.org` — CONFIRMED AS ACCURATE, elevated to note
**Original claim:** Both JSON-LD blocks use `http://schema.org` (not `https://`).  
**Verified:** `_includes/seo.html:92` and `_includes/seo.html:134` both use `"http://schema.org"`. This is accurate. However, since `site.social` is absent (C-07), neither block renders at all. Fix C-07 first; fix the `http` → `https` in the same pass. Not a standalone action item — **merged into C-07**.

---

### D-04 — Designer: "Publications page: 'Recommended citation' destroys scan hierarchy"
**Severity claimed:** P1.  
**Assessment:** Valid UX concern but tertiary to content gaps and broken functionality. Collapsing citation behind `<details>` is a correct fix, but no recruiter or peer is failing to understand the site because of citation block layout. **Downgraded to P3, dropped from top-15.**

---

### D-05 — Hiring: "site does not help owner recruit their own AI team"
**Severity claimed:** P2.  
**Assessment:** Net-new content suggestion (adding "We're hiring" section). Reasonable advice but below the severity floor for this audit given the volume of P0/P1 items. **Dropped.**

---

### D-06 — Blogging: "Galactic morphology post ends mid-description, reads like a draft"
**Assessment:** Valid observation. The post is the strongest 2017 post (ai-expert concurs on quality) but the missing results section is a real gap. However completing a 2017 CNN post ranks below publishing new 2026-era content. **Dropped from top-15; owner's discretion.**

---

### D-07 — Webdev: "Google Scholar social link emits leading-space accessible name"
**Severity:** P2.  
**Assessment:** Valid but micro. `aria-hidden="true"` on the icon `<i>` element is the correct fix. Two-line change. **Kept as P2 but not in top-15.**

---

### D-08 — "No newsletter / email subscription path" (blogging P1)
**Assessment:** Valid distribution improvement but requires integrating a third-party service (Buttondown). Below the severity threshold for sites with 0 posts in 6 years — adding a subscribe widget before having content to subscribe to is premature. **Dropped from top-15.**

---

## Section 5 — The Ranked Top-15 Actions

Ranked by `(severity × confidence) ÷ effort`. Severity: P0=4, P1=3, P2=2. Effort: 15m=5, 1h=4, 4h=3, 1d=2, 1w=1.

| # | Title | Category | Severity | Effort | Confidence | Rationale |
|---|-------|----------|----------|--------|------------|-----------|
| 1 | Remove Medium link from all 3 surfaces | Brand | P0 | 15m | HIGH | Food-recipe blog linked under "Head of AI." Highest reputational risk per minute to fix. Edit `_config.yml` author block, `footer.links`, `_pages/about.md`. |
| 2 | Fix `main.min.js` `type="module"` → `defer` | Code | P0 | 15m | HIGH | jQuery bundle in strict ES-module context is a runtime correctness failure. One attribute change in `_includes/scripts.html:1`. |
| 3 | Wire Giscus: enable Discussions, run giscus.app, paste IDs | Code | P0 | 15m | HIGH | Yellow "not yet wired" banner visible on every post. 10-minute config fill after enabling GitHub Discussions. |
| 4 | Add LinkedIn to `_config.yml`, `_data/authors.yml`, CV | Brand | P0 | 1h | HIGH | Primary recruiter inbound channel. Template conditional already exists. Replace CV dead breadcrumb with direct URL. |
| 5 | Add top-level `twitter:` block + `social:` block + `og_image` to `_config.yml` | SEO | P0/P1 | 1h | HIGH | Fixes Twitter card emission, Person JSON-LD emission, and OG image in one config block. Change `http://schema.org` to `https://schema.org` in `seo.html` in the same pass. |
| 6 | Publish one substantive post (2,000–3,000 words) on current AI work | Content | P0 | 1d | HIGH | Closes 6-year archive gap. Only action that proves stated seniority. Suggested topic: compliance-aware AI harness design for SEBI-regulated workflows. |
| 7 | Fix broken code in posts: `sns.distplot` → `sns.histplot`; `KMeans.partial_fit` → `MiniBatchKMeans`; correct `k-means||` label; remove dead `return` | Code | P1 | 1h | HIGH | Currently published code throws `AttributeError` on run. `scolumns_order` NameError in distributions post also needs fixing. |
| 8 | Add skip-to-content link immediately after `<body>` in `_layouts/default.html` | A11y | P1 | 15m | HIGH | WCAG 2.2 AA 2.4.1 violation. CSS already exists. One line of HTML. |
| 9 | Fix Follow button: add `aria-expanded`/`aria-controls`; rename to "Connect"; wire to LinkedIn once added | A11y / UX | P1 | 1h | HIGH | Fixes WCAG 4.1.2 + wrong label. Do after LinkedIn is added (Action 4). |
| 10 | Fix theme toggle: replace `<a role="button">` with `<button>`; add Space-key handler; add `aria-live` announcement region | A11y | P1 | 1h | HIGH | Two separate WCAG 2.2 AA failures (4.1.2, 4.1.3). `<button>` replacement in `masthead.html` + 5-line JS addition. |
| 11 | Update GitHub bio from "Sr. Principal Research Scientist @Dream11" to current title | Brand | P1 | 15m | MED | Recruiter cross-checking site → GitHub sees stale, junior-sounding role. Settings at github.com/settings/profile. |
| 12 | Add `<link rel="preload">` for avatar in `_includes/head.html`; add Google Search Console verification | Perf / SEO | P1 | 15m | HIGH | Avatar is LCP element on every page. Search Console verification is a one-line config fill. Both are 5-minute changes combined here. |
| 13 | Rewrite "Currently exploring" section; add team-size to CV; add "Open to" sentence | Content | P1 | 1h | HIGH | Identical text appears twice on `about.md`. Replace with forward-looking specifics. Add team headcount to Dream11 CV entry. High recruiter signal per effort. |
| 14 | Remove AcademicPages footer attribution; update `meta[name="theme-color"]` for dark/sepia modes | Design / UX | P2 | 15m | HIGH | Footer appears on every page including CV. `footer.html:30` — trim to `&copy; {{ site.time | date: '%Y' }} {{ site.name }}`. Separate: add two `<meta name="theme-color" media="...">` tags in `head/custom.html`. |
| 15 | Fix Sass slash-division in `_mixins.scss`; add `sass: quiet_deps: true` to `_config.yml` | Build | P1 | 4h | MED | 354 deprecation warnings; becomes hard build error in Sass 2.0. Short-term: add `quiet_deps: true` (5 min) and fix `_mixins.scss:18` (1 hour). Long-term susy/breakpoint removal is a 1-week refactor — not in top-15 scope. |

---

## Appendix — Fabrication Check

No agent fabricated file paths or invented code that does not exist. All file paths and line number references checked above were accurate within ±2 lines. The one loose claim was designer's "avatar is a group photo" — this is an image content judgment that cannot be confirmed from source files; it is not a fabrication but is unverifiable here.

External URL claims:
- Medium @ensembledme as food blog: profile-agent provided specific post titles (Double Chocolate Chip Banana Muffins, Ramen Noodle Soup, etc.) and a March 2024 last-publish date. This is specific enough to treat as confirmed.
- X/Twitter HTTP 402: platform actively gates non-logged-in access; unverifiable. Profile-agent flagged this correctly as unconfirmed.
- Lilian Weng / Chip Huyen / Simon Willison benchmark links: agents cited these as peer comparisons, not as claims about Nilesh's site. Not fabrications.

---

*End of consolidated audit.*
