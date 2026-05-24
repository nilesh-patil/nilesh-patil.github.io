# Overseer Consolidated Audit — nilesh-patil.github.io

**Date:** 2026-05-24  
**Overseer role:** Principal Engineer + Design Director  
**Dev server:** http://localhost:4000/ (Jekyll 4.3, Ruby 4.0.3)

---

## Verdict

The site is in a **P0 regression state on every page load**: commit `3930ceb`'s JS fix introduced a `SyntaxError` that kills jQuery, navigation, the Follow button, smooth scroll, and all JS-dependent UI. That is the single most urgent fix. Beyond the regression, the site's largest strategic risk is a credibility gap: the bio claims active AI leadership at the LLM/agentic-systems level, but the public artifact record — six blog posts ending in 2020, three portfolio projects with zero AI/LLM content, and a GitHub bio still attributing a prior employer — actively contradicts that claim to any technically sophisticated reader. The biggest opportunity is therefore not a code fix but a content commitment: two substantive posts on current AI work would shift the entire perception. The structured-data and brand-consistency issues (three competing Twitter/X handles, Person+Organization JSON-LD collision, ORCID missing from sameAs) are fixable in an afternoon once Nilesh clarifies which X handle is canonical.

---

## Methodology

All six specialist agent reports were read in full; each finding was logged with agent origin, severity, and category. Convergent findings (same underlying issue reported by multiple agents) were merged into single canonical entries. Technical findings were re-verified in the live dev browser via Chrome DevTools MCP `evaluate_script` (contrast ratios, DOM structure, JS state, element dimensions) and via direct file reads. External claims (GitHub bio, ORCID employment, ACL author order) were re-verified via GitHub API and file inspection. The canonical URL/JSON-LD appearing as `localhost` in browser evals was confirmed to be a dev-server override artifact — `_config.yml:11` sets `url: "https://nilesh-patil.github.io"` which produces the correct production output; that finding is dropped.

---

## Numbers

| Metric | Count |
|--------|-------|
| Raw findings (all agents) | 141 (25 + 20 + 21 + 25 + 25 + 25) |
| Post-dedup canonical entries | 58 |
| Dropped during verification | 4 |
| NEEDS_USER items | 8 |
| P0 (Critical) | 4 |
| P1 (High) | 24 |
| P2 (Medium) | 30 |

---

## Section 1 — CONVERGENT Findings (2+ agents, verified)

---

### C-001 · P0 · Code/JS

```yaml
id: C-001
title: "main.min.js: top-level ES6 import in defer script causes SyntaxError — jQuery dead on all pages"
category: Code/JS
severity: P0
confidence: HIGH
effort: 15m
agents: [webdev, designer, hiring, blogging]
agent_count: 4
contributing_ids: [J-01/webdev, D-05/designer, D-06/designer, phase0-prefinding]
```

**Verification:** PASS. Console on page 14 (home): `msgid=41 [error] Uncaught SyntaxError: Cannot use import statement outside a module`. JS eval confirmed `typeof window.$` is `"undefined"`. Script tag confirmed as `defer` with no `type="module"`. Cascade: hamburger button stuck at `display: block` at 1280px desktop, Follow button `aria-expanded === null`, nav collapse broken, smooth scroll dead.

**Evidence:** `assets/js/_main.js:57` contains `import { plotlyDarkLayout, plotlyLightLayout } from './theme.js'` — a static top-level ES6 import. Commit `3930ceb` changed the `<script>` tag from `type="module"` to `defer`. A non-module `<script>` with a top-level static `import` is a parse error per the HTML spec; the browser aborts execution before any byte of jQuery runs.

**Recommendation:** Remove the static `import` from `_main.js` and replace with a dynamic `import('./theme.js').then(...)` gated on `plotlyElements.length > 0`. Rebuild `main.min.js`. This is strictly preferable to reverting to `type="module"` because it allows the plotly feature to remain lazily loaded without making the entire bundle a module.

**Spec reference:** https://html.spec.whatwg.org/multipage/webappapis.html#module-script ("A classic script may not contain top-level import/export"); https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script#type

---

### C-002 · P0 · Brand

```yaml
id: C-002
title: "GitHub bio shows 'Sr. Principal Research Scientist @dream11' — actively contradicts DreamStreet role on site"
category: Brand
severity: P0
confidence: HIGH
effort: 15m
agents: [hiring, profile]
agent_count: 2
contributing_ids: [H-01/hiring, B-01/profile]
```

**Verification:** PASS. GitHub API confirmed: `bio: "Sr. Principal Research Scientist @Dream11"`, `company: "@dream11"`. Site `_config.yml`, `_data/authors.yml`, and `_pages/cv.md` all show "Head of AI at DreamStreet" as current role.

**Evidence:** A recruiter opening three tabs — GitHub, LinkedIn, site — sees a mismatched story on the first tab. The discrepancy signals either role inflation or lack of maintenance.

**Recommendation:** Update GitHub profile bio (github.com/settings/profile) to: "Head of AI at DreamStreet | Previously Head of Applied Research, Dream11 | nilesh-patil.github.io". Set Company field to "DreamStreet".

**Peer reference:** https://github.com/karpathy — bio, employer, and personal site are consistent and current

---

### C-003 · P0 · Content

```yaml
id: C-003
title: "Six-year publishing gap (2021–2026): blog signals disengagement from AI field during the LLM era"
category: Content
severity: P0
confidence: HIGH
effort: 1d (ongoing)
agents: [ai-expert, hiring, blogging]
agent_count: 3
contributing_ids: [C-03/ai-expert, H-02/hiring, C-01/blogging, C-02/blogging]
```

**Verification:** PASS. `_posts/` directory confirmed: six files, newest dated `2020-05-20`. No post contains the words "LLM", "agent", "RAG", "agentic", "compliance", or "DreamStreet".

**Evidence:** Bio claims "compliance-aware AI architecture," "agentic workflows," and "LLM-based behavior simulation." Published content: seaborn histograms (2017), random forests (2017), NumPy reference (2017), NYC taxi graph (2017), galaxy CNN (2017), k-means clustering (2020). The gap covers GPT-3 (2020), ChatGPT (2022), GPT-4 (2023), and the entire agent-framework era. Peers at equivalent seniority publish continuously.

**Recommendation:** Publish at minimum one substantive post per quarter. Two immediate candidates requiring no IP disclosure: (1) architecture patterns for compliance-aware AI in regulated environments (SEBI context), (2) agentic evaluator design at scale. Even 1,500 words on either topic reframes the blog from archive to active practice log.

**Peer reference:** https://lilianweng.github.io/posts/2025-05-01-thinking/ — substantive 2025 post demonstrating current AI thought leadership while in a senior role; https://huyenchip.com/blog/ — continuous content directly mapping to stated expertise

---

### C-004 · P1 · Accessibility

```yaml
id: C-004
title: "TOC link text at 9.28px uppercase with letter-spacing — functionally unreadable"
category: Accessibility
severity: P1
confidence: HIGH
effort: 15m
agents: [designer, webdev]
agent_count: 2
contributing_ids: [D-01/designer, D-13/designer]
```

**Verification:** PASS. JS eval on post page confirmed: `tocLinkFontSize: "9.28125px"`, `tocLinkTextTransform: "uppercase"`, `tocLinkLetterSpacing: "1px"`.

**Evidence:** Double em-nesting cascade: `.toc__menu` sets `font-size: 0.75em` on 18px root (= 13.5px), then `.toc__menu a` sets `font-size: 0.6875em` relative to 13.5px parent (= 9.28px). Additionally `text-transform: uppercase` and `letter-spacing: 1px` apply to the container. 9.28px text is at the biological legibility threshold regardless of contrast ratio.

**Recommendation:** Change `.toc__menu a` font-size from `$type-size-7 (0.6875em)` to `0.75rem` (rem breaks the cascade). Remove `text-transform: uppercase` from `.toc__menu a`; retain only on `.toc .nav__title`.

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/resize-text.html

---

### C-005 · P1 · Accessibility

```yaml
id: C-005
title: "Follow button and nav toggle both missing aria-expanded and aria-controls"
category: Accessibility
severity: P1
confidence: HIGH
effort: 15m each
agents: [webdev, designer]
agent_count: 2
contributing_ids: [X-01/webdev, X-02/webdev, D-05/designer]
```

**Verification:** PASS. JS eval: `followBtnAriaExpanded: null`, `navToggleAriaExpanded: null`, `navToggleAriaControls: null`. Both confirmed via DOM inspection — neither button has ARIA disclosure attributes. Currently also inoperable due to C-001.

**Evidence:** `_includes/author-profile.html:39` — Follow button has no `aria-expanded`. `_includes/masthead.html:6` — nav toggle has no `aria-expanded` or `aria-controls`. WCAG 2.1 SC 4.1.2 (Name, Role, Value) requires expanded state of disclosure buttons to be programmatically determinable.

**Recommendation:** Add `aria-expanded="false"` and `aria-controls="author-social-links"` to Follow button; add `id="author-social-links"` to the social links `<ul>`. Add `aria-expanded="false"` and `aria-controls="greedy-nav-overflow"` to nav toggle button; add `id="greedy-nav-overflow"` to `.hidden-links`. Toggle `aria-expanded` in the corresponding click handlers.

**Spec reference:** https://www.w3.org/WAI/ARIA/apg/patterns/disclosure/; WCAG 2.1 SC 4.1.2 https://www.w3.org/TR/WCAG21/#name-role-value

---

### C-006 · P1 · Accessibility

```yaml
id: C-006
title: "No skip-to-main-content link — keyboard users tab through full masthead on every page"
category: Accessibility
severity: P1
confidence: HIGH
effort: 30m
agents: [webdev, designer]
agent_count: 2
contributing_ids: [X-03/webdev]
```

**Verification:** PASS. JS eval: `hasSkipLink: false`. `document.querySelector('a[href="#main"], a[href="#content"], .skip-link, [class*="skip"]')` returns null. The `<div id="main">` target exists in layouts but no skip link element exists in `<body>`.

**Evidence:** Keyboard-only users must Tab through 6–8 navigation links on every page before reaching content. WCAG 2.1 SC 2.4.1 (Bypass Blocks) requires a mechanism to skip repeated navigation blocks.

**Recommendation:** Add a visually-hidden, focus-visible skip link as the first element in `<body>` in `_layouts/default.html`, targeting the existing `#main` div.

**Spec reference:** WCAG 2.1 SC 2.4.1 https://www.w3.org/TR/WCAG21/#bypass-blocks; Technique G1 https://www.w3.org/WAI/WCAG21/Techniques/general/G1

---

### C-007 · P1 · Accessibility

```yaml
id: C-007
title: "Home page has two <h1> elements (one empty); sidebar <h2> precedes page <h1> in DOM order"
category: Accessibility
severity: P1
confidence: HIGH
effort: 30m
agents: [webdev, hiring]
agent_count: 2
contributing_ids: [X-04/webdev, H-24/hiring]
```

**Verification:** PASS. JS eval on home page confirmed heading structure: `[{H2,"Nilesh Patil"}, {H2,"Recent posts"}, {H2,"Side projects"}, {H1,""}, {H1,"Nilesh Patil"}]`. Two H1 elements present; one is empty string. Sidebar H2 appears before the page H1 in DOM source order.

**Evidence:** The home layout emits `<h1 class="page__title">{{ page.title }}</h1>` even when `page.title` is blank. The sidebar `<h2 class="author__name">` is rendered before `<main>` content in DOM order, inverting the expected outline. WCAG 2.1 SC 1.3.1, SC 2.4.6.

**Recommendation:** Guard the H1 emission with `{% if page.title %}`. Add `aria-hidden="true"` to the sidebar `<h2 class="author__name">` since the name is also conveyed in the avatar `alt` text.

**Spec reference:** https://www.w3.org/TR/WCAG21/#info-and-relationships; MDN heading elements https://developer.mozilla.org/en-US/docs/Web/HTML/Element/Heading_Elements

---

### C-008 · P1 · SEO/Meta

```yaml
id: C-008
title: "Duplicate meta[name=description]: og:description template emits a second name=description"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
agents: [webdev, profile]
agent_count: 2
contributing_ids: [S-01/webdev, B-06/profile]
```

**Verification:** PASS. JS eval: `metaDescCount: 2`. `metaDesc1` = site description; `metaDesc2` = og_description. Two distinct content strings, both as `meta[name="description"]`.

**Evidence:** `_includes/seo.html:125` emits `<meta property="og:description" name="description" ...>`. An earlier `<meta name="description">` at line 27 already exists. Search engines may use either value non-deterministically; Google warns on duplicate meta descriptions.

**Recommendation:** Remove `name="description"` attribute from the `og:description` meta tag in `_includes/seo.html`.

**Spec reference:** https://ogp.me/#metadata; Google meta tags https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls

---

### C-009 · P1 · Content

```yaml
id: C-009
title: "ACL 2026 paper: Nilesh Patil listed as third author but CV presents as 'Nilesh Patil, et al.'"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert, hiring]
agent_count: 2
contributing_ids: [C-01/ai-expert, H-08/hiring]
```

**Verification:** PASS. File `_publications/2026-structure-guided-entity-resolution.md:9` confirmed: `authors: "<strong>Nilesh Patil</strong>, et al."`. OpenReview link cited by ai-expert agent (https://openreview.net/forum?id=rLisRb1T1Y) shows author order: (1) Shivam Chourasia, (2) Hitesh Kapoor, (3) Nilesh Patil.

**Evidence:** The `<strong>` markup on "Nilesh Patil" in a "Patil et al." citation is the academic convention for first-author self-highlighting. Applied to a third author, it misrepresents authorship order. Hiring managers and peer reviewers checking the OpenReview link see the discrepancy immediately.

**Recommendation:** Correct the author string: `"Shivam Chourasia, Hitesh Kapoor, <strong>Nilesh Patil</strong>"`. Update the citation format to acknowledge authorship order accurately.

**Spec reference:** https://openreview.net/forum?id=rLisRb1T1Y (verified author order)

---

### C-010 · P1 · Content

```yaml
id: C-010
title: "Portfolio has zero AI/LLM artifacts — none of the 3 projects backs the 'Head of AI' claim"
category: Content
severity: P1
confidence: HIGH
effort: 4h
agents: [ai-expert, hiring]
agent_count: 2
contributing_ids: [C-02/ai-expert, C-09/ai-expert, H-05/hiring, H-17/hiring, H-18/hiring]
```

**Verification:** PASS. `_portfolio/` directory contains: `datascience-environment.md` (2018 Docker setup), `pythonvsrust-kmeans.md` (2024 k-means benchmark), `simucell3d.md` (2026 C++ HPC fork). Zero AI/LLM content.

**Evidence:** CV claims "LLM-based behavior simulation," "agentic evaluators," "compliance-aware AI harness design," "feature-store systems supporting 250M+ users." The three portfolio entries cover DevOps tooling (2018), algorithmic benchmarking (2024), and computational biology HPC (2026). No public artifact demonstrates any of the claimed AI expertise.

**Recommendation:** Add at minimum one portfolio entry connecting to current AI work. The ACL 2026 entity resolution paper could be cross-linked with a technical explainer. A sanitized architecture writeup of one DreamStreet AI system (compliance harness, agent framework) would be high-impact. The 2018 Docker entry should be retired or deprioritized; it is 8 years old and creates cognitive dissonance.

**Peer reference:** https://github.com/simonwillison — every public repo is a runnable, documented artifact evidencing stated expertise; https://huyenchip.com — publishes system design essays on real production AI systems without proprietary disclosure

---

### C-011 · P1 · Content

```yaml
id: C-011
title: "No team size, headcount, or org-chart signal — hiring readers default to 'strong IC, not leader'"
category: Content
severity: P1
confidence: HIGH
effort: 1h
agents: [hiring]
agent_count: 1
contributing_ids: [H-03/hiring, H-19/hiring, H-04/hiring]
```

**Verification:** PASS. `_pages/cv.md` and `_pages/about.md` confirmed: no team headcount, no budget mention, no org scope indicator for any role. Dream11 section says "led a high-performing cross-continent team" with no number. DreamStreet section says "Drove AI adoption org-wide" with no number.

**Evidence:** A VP-AI role requires evidence of organizational accountability. Without headcount, the recruiter's mental model defaults to minimum: a strong IC with a leadership title. "Led a team of 12 research scientists" vs "led a high-performing team" are materially different calibrations at committee level.

**Recommendation:** Add team size for each role. For Dream11, add the number of direct/indirect reports. For DreamStreet, add the current team size or a range if exact numbers are confidential. Also split the compound "Senior Principal Research Scientist / Head of Applied Research" Dream11 title into two dated entries to show when the promotion occurred.

**Peer reference:** https://huyenchip.com/about/ — lists founding and selling an AI infrastructure startup as single-line proof of P&L accountability

---

### C-012 · P1 · Content

```yaml
id: C-012
title: "Share buttons send only URL path, not title — X and LinkedIn shares produce blank/untitled posts"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [blogging, webdev]
agent_count: 2
contributing_ids: [C-04/blogging]
```

**Verification:** PASS. File `_includes/social-share.html:13,17,21` confirmed. JS eval on post page confirmed share button hrefs: Bluesky gets `text={{ base_path }}{{ page.url }}` (URL only, no title). X gets `text={{ base_path }}{{ page.url }}` (same problem — `intent/post?text=URL` produces a tweet with only a URL, no pre-filled message). LinkedIn gets `url=` with no `title=` or `summary=`.

**Evidence:** X's `intent/tweet` or `intent/post` endpoint expects `url=` as a separate parameter and `text=` for the tweet body. Passing only the URL in `text=` produces a tweet that is just a URL with no context. LinkedIn `shareArticle` supports `title=` parameter. All post-level CTAs for distribution silently fail to pre-populate useful content.

**Recommendation:** Fix each platform's URL schema: X → `intent/tweet?url={{ site.url }}{{ page.url | url_encode }}&text={{ page.title | url_encode }}`; LinkedIn → `shareArticle?mini=true&url=...&title={{ page.title | url_encode }}`; Bluesky → include both title and URL in `text=`.

**Spec reference:** https://developer.twitter.com/en/docs/twitter-for-websites/tweet-button/guides/web-intent

---

### C-013 · P1 · Brand/SEO

```yaml
id: C-013
title: "Three different Twitter/X handles across site surfaces — structured data integrity compromised"
category: Brand
severity: P1
confidence: HIGH
effort: 1h (after user input)
agents: [profile, webdev]
agent_count: 2
contributing_ids: [B-03/profile, B-08/profile, B-09/profile, B-23/profile]
```

**Verification:** PASS. JS eval on home page confirmed: `twitterSite: "@ensembledme"`. `_config.yml` confirmed: `twitter.username: ensembledme`, `author.twitter: "ensembledme"`, `social.links` contains both `x.com/optimistic_flw` AND `twitter.com/nilesh-patil`. Three distinct handles across four surfaces.

**Evidence:** `sameAs` contains `x.com/optimistic_flw` and `twitter.com/nilesh-patil`. The sidebar links to `twitter.com/ensembledme`. The Twitter Card `twitter:site` is `@ensembledme`. Google's knowledge graph will attempt to resolve all three. Note: `medium.com/@ensembledme` is confirmed to be a food/lifestyle blog — if `@ensembledme` on X is similarly not Nilesh's account, every page share on X credits a wrong account.

**Recommendation:** Resolve to one canonical handle. Until confirmed, no on-site fix should be deployed. See Section 4 (NEEDS_USER) for the decision required.

**Spec reference:** https://schema.org/Person — `sameAs` should contain unique, verified canonical URLs

---

### C-014 · P1 · SEO/Meta

```yaml
id: C-014
title: "Person + Organization JSON-LD both emitted on same URL — knowledge graph disambiguation corrupted"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
agents: [profile]
agent_count: 1
contributing_ids: [B-07/profile, B-22/profile]
```

**Verification:** PASS. JS eval: `jsonLDTypes: [{type:"Person", url:"https://nilesh-patil.github.io"}, {type:"Organization", url:"https://nilesh-patil.github.io"}]`. Both types resolve to the same URL. Organization block has no `name` field. The `logo` in Organization is Nilesh's personal headshot — not a logo mark.

**Evidence:** `_includes/seo.html:134-141` emits an Organization block when `site.og_image` is set. A personal portfolio should declare exactly one `@type` per URL. Dual type declarations on the same URL corrupt the knowledge graph signal. Per Google's structured data guidelines, `Organization.logo` must be a logo mark, not a person photo.

**Recommendation:** Remove the Organization JSON-LD block from `_includes/seo.html`. The `og:image` Open Graph tag continues to function without it. If a logo mark is desired, create a separate dedicated SVG and assign it only to the Organization block.

**Spec reference:** https://schema.org/Person; https://developers.google.com/search/docs/appearance/structured-data/logo

---

### C-015 · P1 · Content

```yaml
id: C-015
title: "Teaching and Talks pages are completely empty — blank nav-linked pages signal abandonment"
category: Content
severity: P1
confidence: HIGH
effort: 30m
agents: [designer, hiring]
agent_count: 2
contributing_ids: [D-25/designer, H-13/hiring]
```

**Verification:** PASS. `_config.yml` lines confirmed `show_talks: false`, `show_teaching: false`. Pages are in primary nav and render only an H1 heading with nothing below it. The About page mentions Columbia AI sessions and training for up to 200 participants — speaking history exists but is hidden behind config flags.

**Evidence:** Primary-nav pages that are blank damage credibility more than not having those pages at all. The speaking history claimed in About/CV is substantive (Columbia, organization-wide training) but invisible on the site.

**Recommendation:** Either (a) set `show_talks: true` and populate the collection with 2–3 key engagements (Columbia Sports x AI sessions, Dream11 training sessions), or (b) remove the nav links to empty pages and fold a "Speaking" subsection into the About page. Option (a) is preferred as it demonstrates executive presence.

**Peer reference:** https://huyenchip.com/speaking/ — dedicated speaking page with conference names, dates, and audience establishing thought-leadership presence

---

### C-016 · P2 · Accessibility

```yaml
id: C-016
title: "Theme-toggle touch target is 25×36px — below 44×44px recommendation"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
agents: [designer, webdev]
agent_count: 2
contributing_ids: [D-08/designer, X-07/webdev]
```

**Verification:** PASS. JS eval: `themeToggleSize: {w:25, h:36}`. Confirmed on both home and post pages. Lighthouse flagged same element in post page audit.

**Evidence:** `_sass/layout/_navigation.scss` sets `width: 25px` on `#theme-toggle a`. At 25×36px the element barely meets WCAG 2.5.8's 24px minimum (AA) but misses the 44×44px recommended size. The control is in the mobile masthead's top-right corner where tap precision is most challenged.

**Recommendation:** Set `#theme-toggle a { width: 44px; min-height: 44px; display: flex; align-items: center; justify-content: center; }`.

**Spec reference:** https://www.w3.org/TR/WCAG22/#target-size-minimum

---

### C-017 · P2 · Accessibility

```yaml
id: C-017
title: "Sepia muted text (#7d6a52 on #f4ecd8) at 4.40:1 fails WCAG AA for small text"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
agents: [designer, webdev]
agent_count: 2
contributing_ids: [D-07/designer]
```

**Verification:** PASS. Contrast computed via luminance formula: `sepiaMutedContrast: "4.40"`. WCAG AA requires 4.5:1 for text below 18px. Sepia muted text (`$type-size-6` = 13.5px) affects post dates, read time, TOC, `.page__meta`, `.archive__subtitle`, and `.footnotes`.

**Evidence:** `_sass/theme/_default_sepia.scss:27` — `$text-muted: #7d6a52`. This is the theme designed for long-read sessions, making accessibility failure at this threshold a real regression, not a borderline curiosity.

**Recommendation:** Darken to `#6f5d45` (contrast ≈ 5.2:1). Hue unchanged; luminance shift only.

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html

---

### C-018 · P2 · Accessibility

```yaml
id: C-018
title: "Syntax highlight: 3 token families fail WCAG AA contrast on light code background"
category: Accessibility
severity: P2
confidence: HIGH
effort: 1h
agents: [designer, webdev]
agent_count: 2
contributing_ids: [X-06/webdev, D-23/designer]
```

**Verification:** PASS. Contrast computed: `syntaxNcContrast: "2.31"` (class/function names `#22b3eb` on `#fafafa` — fails, needs 4.5:1). `.s`/`.s1`/`.s2` (string literals `#2aa198`) computed at 3.03:1 by webdev agent — confirmed. `.sb` (`#93a1a1`) at 2.56:1 — confirmed. Code elements are `$type-size-6` (≈13.5px), not large text, so 4.5:1 threshold applies.

**Evidence:** Three token color families fail in light mode. Additionally, designer agent computed dark mode comment tokens (`#5e7474` on `#161a20`) at 3.51:1 — JS eval confirmed `darkCommentContrast: "3.51"`.

**Recommendation:** Replace failing tokens: `#22b3eb` → `#0069a1`; `#2aa198` → `#1a7a72`; `#93a1a1` (as `.sb`) → `#5e7474`. For dark mode, add a `html[data-theme="dark"]` block overriding comment token color to `#7a9c9c`.

**Spec reference:** WCAG 2.1 SC 1.4.3 https://www.w3.org/TR/WCAG21/#contrast-minimum

---

### C-019 · P2 · Accessibility

```yaml
id: C-019
title: "Share buttons: Bluesky (3.65:1) and Mastodon (4.38:1) fail WCAG AA contrast"
category: Accessibility
severity: P2
confidence: HIGH
effort: 30m
agents: [webdev, blogging]
agent_count: 2
contributing_ids: [X-08/webdev]
```

**Verification:** PASS. JS eval: `blueskyBg: "rgb(17, 132, 254)"`, `blueskyColor: "rgb(255, 255, 255)"`. Computed: `blueskyContrast: "3.65"`, `mastodonContrast: "4.38"`. Both below 4.5:1 WCAG AA threshold for normal text.

**Evidence:** Post page share buttons visible on every post. Both Bluesky and Mastodon button labels fail contrast with their brand background colors. LinkedIn passes at 4.66:1.

**Recommendation:** Darken Bluesky background to `#0066cc` (~5.1:1) and Mastodon background to `#4242d4` or equivalent that achieves 4.5:1.

**Spec reference:** WCAG 2.1 SC 1.4.3 https://www.w3.org/TR/WCAG21/#contrast-minimum

---

### C-020 · P2 · Accessibility

```yaml
id: C-020
title: "Site nav <nav> has no aria-label — multiple nav landmarks indistinguishable to screen readers"
category: Accessibility
severity: P2
confidence: HIGH
effort: 10m
agents: [webdev]
agent_count: 1
contributing_ids: [X-05/webdev]
```

**Verification:** PASS. JS eval: `navAriaLabel: null`. On post pages with a TOC, two `<nav>` elements exist with no differentiation. TOC nav also lacks `aria-label` (confirmed: `tocNavAriaLabel: null`).

**Evidence:** Screen reader users navigating by landmark hear two "navigation" regions with no way to distinguish site navigation from table of contents. WCAG 2.4.1; ARIA best practice.

**Recommendation:** Add `aria-label="Site navigation"` to `<nav id="site-nav">` in `_includes/masthead.html`. Add `aria-label="Table of contents"` to the TOC `<nav>` in `_includes/toc.html`.

**Spec reference:** https://www.w3.org/WAI/ARIA/apg/patterns/landmarks/examples/navigation.html

---

### C-021 · P2 · Accessibility

```yaml
id: C-021
title: "Academicons <i> elements missing aria-hidden — decorative icons announced by screen readers"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
agents: [webdev]
agent_count: 1
contributing_ids: [X-09/webdev]
```

**Verification:** PASS. File read of `_includes/author-profile.html:60-81` confirmed: arxiv, googlescholar, inspire-hep, impactstory, orcid, pubmed, scopus `<i>` elements all lack `aria-hidden="true"`. Only the academia entry (line 57) has it. JS eval: `academiconsWithAriaHidden: 0` (on post page where only googlescholar is active).

**Evidence:** Missing `aria-hidden="true"` on decorative icon elements causes screen readers to announce CSS class names: "ai ai-arxiv ai-fw icon-pad-right." WCAG 1.1.1, Technique F87.

**Recommendation:** Add `aria-hidden="true"` to all academicons `<i>` elements in `_includes/author-profile.html` that are paired with visible text labels (lines 60, 63, 66, 69, 72, 75, 81, and equivalent).

**Spec reference:** WCAG 2.1 SC 1.1.1 https://www.w3.org/TR/WCAG21/#non-text-content; Technique F87 https://www.w3.org/WAI/WCAG21/Techniques/failures/F87

---

### C-022 · P2 · Design/UX

```yaml
id: C-022
title: "h3 font-size equals body text (18px); h4/h5/h6 all resolve to 13.5px — inverted heading hierarchy"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
agent_count: 1
contributing_ids: [D-03/designer, D-12/designer]
```

**Verification:** PASS. JS eval: `h3FontSize: "18px"`, `bodyFontSize: "18px"`. The type scale sets `$type-size-5: 1em` at 18px root for both h3 and body. `_base.scss:46-56` confirmed: h4/h5/h6 all use `$type-size-6 (0.75em = 13.5px)` — smaller than body text.

**Evidence:** On the CV page and publications page, sub-section headings (h4: institution) visually disappear into body text. A heading smaller than the body it introduces is a typographic inversion that harms skim-reading — the primary use pattern of academic/professional sites.

**Recommendation:** Bump h3 to `$type-size-4` (1.25em = 22.5px), scoped to `.page__content h3` to avoid archive collision. Set h4 to `$type-size-5` (1em = body weight, but distinguished by bold), h5 to `$type-size-6` with `text-transform: uppercase` for differentiation.

**Peer reference:** https://lilianweng.github.io/posts/2023-06-23-agent/ — visible size steps at h2/h3; h3 noticeably larger than body text

---

### C-023 · P2 · Design/UX

```yaml
id: C-023
title: "Hero and page title line-height: 1.0 causes line collision on wrapped headings"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
agent_count: 1
contributing_ids: [D-04/designer]
```

**Verification:** PASS. JS eval: `h1LineHeight: "43.938px"`, `h1FontSize: "43.938px"`, `h1LineHeightRatio: 1`. Confirmed `_page.scss:49` rule `.page__title { line-height: 1 }`. At 43.9px font size and 1.0 line-height, descenders of one wrapped line contact ascenders of the next.

**Evidence:** Web typography convention (Butterick's Practical Typography; Material Design spec) sets display/heading text line-height at 1.1–1.3. At ratio 1.0, multi-line post titles read as a dense ink block on mobile (390px viewport).

**Recommendation:** Set `.page__title { line-height: 1.2 }`. For overlay hero titles, 1.15 is acceptable.

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/text-spacing.html

---

### C-024 · P2 · Design/UX

```yaml
id: C-024
title: "Hamburger button visible at 1280px desktop due to JS failure — ghost UI element"
category: Design/UX
severity: P2
confidence: HIGH
effort: 30m (CSS fallback)
agents: [designer]
agent_count: 1
contributing_ids: [D-05/designer, D-06/designer]
```

**Verification:** PASS. JS eval: `hamburgerDisplay: "block"`, `hamburgerBounds: {x:1216, y:9, width:46, height:45}`. At current 1280px viewport, hamburger is fully visible at desktop. Root cause is C-001 (JS failure); the greedy-nav JS never runs to hide it.

**Evidence:** A hamburger at desktop is a strong anti-pattern. It also creates overflow at narrow viewports (577px nav width vs 500px viewport without collapse). Both are immediate consequences of the JS failure, but a CSS fallback can partially restore function independently of the JS fix.

**Recommendation:** Add CSS fallback: `.greedy-nav button { @include breakpoint($large) { display: none; } }`. Also add a narrow-viewport rule that hides all `.visible-links li` except the hamburger at `max-width: 599px`. This is a bridge fix; the real fix is C-001.

**Peer reference:** https://simonwillison.net — no hamburger visible at desktop; nav collapses at mobile without JS dependency

---

### C-025 · P2 · SEO/Meta

```yaml
id: C-025
title: "og:image references ensembledme.jpg (JPEG) while only .webp is served by <picture> negotiation"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 10m
agents: [webdev, profile]
agent_count: 2
contributing_ids: [S-02/webdev, B-17/profile]
```

**Verification:** PARTIAL PASS — both files confirmed to exist. `images/ensembledme.jpg` (107 KB) and `images/ensembledme.webp` (39 KB) both present in repo. The S-02 concern (jpg missing) does NOT apply since the JPEG exists. However, `_config.yml:26` still uses `og_image: "ensembledme.jpg"` — the JPEG is served correctly to OG/social scrapers. The S-02 finding is DOWNGRADED to a minor opportunity: switching to `.webp` saves 68 KB on social unfurls (most modern platforms support WebP).

**Evidence:** `og:image` on home page confirmed as `https://nilesh-patil.github.io/images/ensembledme.jpg` — file exists, so unfurls work. The branding risk of the filename containing `ensembledme` (same as the food blogger's Medium handle) is noted as a NEEDS_USER item (B-17).

**Recommendation:** Optionally switch `og_image` to `ensembledme.webp` for payload savings. The JPEG being present means no urgency. Defer renaming the file until the handle question is resolved.

**Spec reference:** https://ogp.me/#structured; Open Graph image guidelines

---

### C-026 · P2 · SEO/Meta

```yaml
id: C-026
title: "theme-color meta hardcoded to #ffffff — no dark-mode variant for browser chrome"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 15m
agents: [webdev]
agent_count: 1
contributing_ids: [S-03/webdev]
```

**Verification:** PASS. JS eval: `themeColors: [{content:"#ffffff", media:null}]`. Single entry, no `media` attribute. On dark-mode pages, the browser chrome (address bar on Android/Chrome) remains white.

**Evidence:** `_includes/head/custom.html:12` — `<meta name="theme-color" content="#ffffff"/>`. The site has a full dark theme but the OS-level browser chrome color does not follow.

**Recommendation:** Add a paired tag: `<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">` and `<meta name="theme-color" content="#1c1c1e" media="(prefers-color-scheme: dark)">`.

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta/name/theme-color

---

### C-027 · P2 · Build/CI

```yaml
id: C-027
title: ".travis.yml is stale, not excluded from Jekyll build — ships to public _site"
category: Build/CI
severity: P2
confidence: HIGH
effort: 15m
agents: [webdev]
agent_count: 1
contributing_ids: [I-01/webdev]
```

**Verification:** PASS. File exists at repo root. Content confirmed: `rvm: - 2.1` (EOL 2021), references `ruby 2.1`. `_config.yml` `exclude:` list confirmed — `.travis.yml` is not in it. Jekyll will copy it to `_site/`. The repository now uses `.github/workflows/pages.yml` (GitHub Actions, Ruby 3.3).

**Evidence:** The file is publicly accessible at `https://nilesh-patil.github.io/.travis.yml`. It confuses contributors and is a dead CI signal.

**Recommendation:** Delete `.travis.yml`. If retaining for history, add it to `exclude:` in `_config.yml`.

**Spec reference:** GitHub Pages deployment docs https://docs.github.com/en/pages

---

### C-028 · P2 · Build/CI

```yaml
id: C-028
title: "354+ Sass @import deprecation warnings — will become hard errors in Dart Sass 3.0"
category: Build/CI
severity: P2
confidence: HIGH
effort: 4h
agents: [webdev]
agent_count: 1
contributing_ids: [I-02/webdev]
```

**Verification:** PASS (reported). Jekyll build emits 21+ unique deprecation warning categories. All originate from `@import` chains through vendor files (`breakpoint`, `susy`, `font-awesome`, theme files). Dart Sass 3.0 (expected 2025/2026) will remove `@import` with no fallback — the build pipeline will break.

**Evidence:** GitHub Actions uses `ruby/setup-ruby@v1` with Ruby 3.3, picking up Dart Sass via `sass-embedded`. When Dart Sass 3.0 ships, no code change from the site author is required to break the build.

**Recommendation:** Migrate all `@import` to `@use` / `@forward` using `sass-migrator module`. Vendor files (`breakpoint`, `susy`) are the main work; consider replacing `susy` with native CSS Grid.

**Spec reference:** https://sass-lang.com/documentation/breaking-changes/import/

---

### C-029 · P2 · Content

```yaml
id: C-029
title: "ORCID not in Person JSON-LD sameAs — misses key researcher disambiguation anchor"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 15m
agents: [profile]
agent_count: 1
contributing_ids: [B-04/profile, B-13/profile]
```

**Verification:** PASS. ORCID API `https://pub.orcid.org/v3.0/0000-0002-3438-8571/person` confirmed: given name Nilesh, family name Patil. `_config.yml` confirmed: no `orcid:` field in author block; `social.links` does not include ORCID URL. Person JSON-LD `sameAs` array (JS eval) does not include ORCID URL. The ORCID employment record also shows Dream Sports only (not DreamStreet) — stale since 2026.

**Evidence:** "Nilesh Patil" is a very common Indian name. ORCID is the strongest machine-readable identifier for research-related disambiguation. Its absence from `sameAs` weakens the knowledge graph entry.

**Recommendation:** Add `orcid: "https://orcid.org/0000-0002-3438-8571"` to the author block and `social.links`. Update ORCID record with DreamStreet employment. See Section 4 (NEEDS_USER) for the visibility decision.

**Spec reference:** https://schema.org/Person — `sameAs` with ORCID URI

---

### C-030 · P2 · Content

```yaml
id: C-030
title: "No recruiter contact CTA: About and CV pages have no email, form, or openness signal"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [hiring, profile]
agent_count: 2
contributing_ids: [H-06/hiring, B-10/profile]
```

**Verification:** PASS. `_config.yml:44` comment confirms email intentionally omitted. `_pages/about.md` "Get in touch" section lists 5 platform links but no email, form, or openness statement. `_pages/cv.md` contact section identical.

**Evidence:** Recruiters work on tight timelines. Without a direct contact method or explicit openness signal, the path from site visit to outreach adds friction. Senior-role recruiting via GitHub Issues or blind LinkedIn InMail is below-average conversion.

**Recommendation:** Add a sentence to the About "Get in touch" section: "I'm selectively open to conversations about AI leadership roles and research collaborations — LinkedIn is the fastest path." Adding a professional email is the higher-impact option; see NEEDS_USER Section 4.

**Peer reference:** https://huyenchip.com — "Reach out if you want to find a way to work together" is a direct warm CTA on the homepage

---

### C-031 · P2 · Brand

```yaml
id: C-031
title: "Sidebar X/Twitter link uses twitter.com domain; Medium bio 5+ years stale"
category: Brand
severity: P2
confidence: HIGH
effort: 15m each
agents: [profile]
agent_count: 1
contributing_ids: [B-12/profile, B-18/profile]
```

**Verification:** PASS (both). File confirmed: `_includes/author-profile.html:169` has `href="https://twitter.com/{{ author.twitter }}"` — hardcoded `twitter.com`. WebFetch of `https://nilesh-patil.medium.com/` confirmed bio: "Interested in applied machine learning, statistics and data science." Most recent article: January 2020.

**Evidence:** X Corp uses `x.com`; `twitter.com` redirects but the rendered anchor label says "X (formerly Twitter)" while pointing to the deprecated domain. Medium bio does not reflect 2026 positioning.

**Recommendation:** Update `_includes/author-profile.html:169` to use `https://x.com/`. Update Medium profile bio (off-site) to current positioning.

**Spec reference:** n/a (brand consistency)

---

### C-032 · P2 · Performance

```yaml
id: C-032
title: "Academicons served as TTF only — no woff2 variant; TTF is ~3x larger than woff2"
category: Performance
severity: P2
confidence: HIGH
effort: 1h
agents: [webdev]
agent_count: 1
contributing_ids: [P-03/webdev]
```

**Verification:** PASS (reported — network request confirmed TTF load). Academicons TTF ~53 KB raw; woff2 equivalent ~22–28 KB. Font Awesome uses woff2 throughout.

**Evidence:** Every page load incurs a ~30 KB payload penalty for Academicons. Over 3G (common in India, site's target audience), this is a meaningful LCP contributor.

**Recommendation:** Replace academicons TTF with the woff2 variant from the Academicons npm release. Update the `@font-face` declaration to use woff2 with TTF as fallback.

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/CSS/@font-face; https://web.dev/articles/reduce-webfont-size

---

### C-033 · P2 · Content

```yaml
id: C-033
title: "All posts share a single 'blog' category — category-archive provides zero navigational value"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
agent_count: 1
contributing_ids: [C-09/blogging, C-22/blogging]
```

**Verification:** PASS. All six `_posts/*.md` files confirmed `categories: [blog]`. Tag taxonomy confirmed inconsistent: `RandomForest`, `deep learning`, `machine-learning`, `pyspark` are separate tags with no canonical normalization.

**Evidence:** The `/category-archive/` page renders as a single "blog" section with all six posts. Tags like `RandomForest` (CamelCase) and `deep learning` (spaced) create separate non-linked categories for equivalent content.

**Recommendation:** Assign meaningful categories to existing posts (`data-visualization`, `machine-learning`, `distributed-systems`). Normalize tags to lowercase-hyphenated form. Adopt a 4-category vocabulary for future posts.

---

### C-034 · P2 · Content

```yaml
id: C-034
title: "Home page 'Recent posts' section label is misleading — 2020 posts shown as 'Recent'"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging, hiring]
agent_count: 2
contributing_ids: [C-25/blogging, H-10/hiring]
```

**Verification:** PASS. Home page section header confirmed as "Recent posts" with most recent date: May 20, 2020. Side projects (with 2026 SimuCell3D and 2024 k-means entries) appear below posts on the home page — the 2026 portfolio entry is further down the page than the 2020 post.

**Evidence:** Showing 2020 content under a "Recent posts" heading is actively misleading to a recruiter doing a 10-second scan. Section order also buries the 2026 portfolio entry below the fold.

**Recommendation:** Change "Recent posts" label to "Writing" or "Posts". Swap section order: show "Side projects" (with 2026 and 2024 entries) before "Posts". Reduce post limit to 3.

**Peer reference:** https://simonwillison.net — home page shows most-recent items across all content types sorted by date

---

### C-035 · P2 · Code/JS

```yaml
id: C-035
title: "GitHub Actions workflow pinned to semver tags not SHA digests — supply-chain risk"
category: Build/CI
severity: P2
confidence: HIGH
effort: 30m
agents: [webdev]
agent_count: 1
contributing_ids: [I-03/webdev]
```

**Verification:** PASS (reported). `.github/workflows/pages.yml` uses tag-pinned actions: `actions/checkout@v4`, `ruby/setup-ruby@v1`, etc. Tags are mutable references that can be moved by action maintainers after publication.

**Evidence:** A compromised action maintainer could push malicious code to the `v4` tag. GitHub's security hardening guide recommends pinning to SHA for production deployments.

**Recommendation:** Pin each action to a specific SHA digest. Use Dependabot or Renovate to automate updates.

**Spec reference:** GitHub Actions security hardening https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions

---

## Section 2 — UNIQUE Findings Worth Keeping (single agent, verified)

---

### U-001 · P0 · Code/JS (AI-expert, verified)

```yaml
id: U-001
title: "dask_ml.cluster.KMeans.partial_fit does not exist — AttributeError on published tutorial code"
category: Code/JS
severity: P0
confidence: HIGH
effort: 1h
agents: [ai-expert]
contributing_ids: [J-01/ai-expert]
```

**Verification:** PASS. File `_posts/2020-05-20-distributed-kmeans-clustering.md:379` confirmed: `kmeans.partial_fit(batch)` called on `dask_ml.cluster.KMeans` instance. The dask-ml API docs list public methods as: `fit`, `fit_transform`, `get_metadata_routing`, `get_params`, `predict`, `set_output`, `set_params`, `transform` — no `partial_fit`. A reader following the tutorial immediately gets `AttributeError`.

**Recommendation:** Replace `dask_ml.cluster.KMeans` with `sklearn.cluster.MiniBatchKMeans` wrapped in `dask_ml.wrappers.Incremental`, which is the documented pattern for streaming k-means. Fix simultaneously with U-002 and U-003 in the same post.

**Spec reference:** https://ml.dask.org/modules/generated/dask_ml.cluster.KMeans.html; https://ml.dask.org/incremental.html

---

### U-002 · P1 · Code/JS (AI-expert, verified)

```yaml
id: U-002
title: "Dead unreachable return statement in find_elbow_point — signature confusion"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
contributing_ids: [J-02/ai-expert]
```

**Verification:** PASS. File `_posts/2020-05-20-distributed-kmeans-clustering.md:500-502` confirmed via Bash read: `return int(k_range[int(np.argmax(distances))])` on line 500, followed by `return k_range, inertias` on line 502. The second return is unreachable and implies a tuple return type that never executes.

**Recommendation:** Delete line 502 (`return k_range, inertias`).

**Spec reference:** https://docs.python.org/3/reference/simple_stmts.html#the-return-statement

---

### U-003 · P1 · Code/JS (AI-expert, verified)

```yaml
id: U-003
title: "k-means|| mislabeled as k-means++ in PySpark code comment — meaningful algorithmic error"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
contributing_ids: [J-03/ai-expert]
```

**Verification:** PASS. File `_posts/2020-05-20-distributed-kmeans-clustering.md:201-203` confirmed via Bash read: `if init_method == "k-means++": # Use PySpark's default k-means++ initialization` with `initMode="k-means||"` inside the block. These are distinct algorithms. PySpark's valid `initMode` values are `"k-means||"` (Bahmani et al. 2012 distributed init) and `"random"` — not `"k-means++"`.

**Recommendation:** Change the function parameter and comment to correctly reference `"k-means||"`. Document the distinction between the two algorithms.

**Spec reference:** https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.KMeans.html (initMode valid values)

---

### U-004 · P1 · Code/JS (AI-expert, verified)

```yaml
id: U-004
title: "sns.distplot called four times — deprecated since seaborn 0.11, removed in 0.14"
category: Code/JS
severity: P1
confidence: HIGH
effort: 1h
agents: [ai-expert]
contributing_ids: [J-04/ai-expert]
```

**Verification:** PASS. File `_posts/2017-01-14-visualizing-and-comparing-distributions.md:70-72` confirmed: three `sns.distplot(...)` calls. Line 406 confirmed: `g.map(sns.distplot, "Value", hist=False, rug=True)`. Seaborn 0.11+ deprecated `distplot`; seaborn 0.14 removed it. Current seaborn release is 0.13.2.

**Recommendation:** Replace with `sns.histplot(series, bins=nbins, kde=True)`. Replace `g.map(sns.distplot, ...)` with `g.map_dataframe(sns.histplot, x="Value", kde=True)`.

**Spec reference:** https://seaborn.pydata.org/whatsnew/v0.11.0.html

---

### U-005 · P1 · Code/JS (AI-expert, verified)

```yaml
id: U-005
title: "scolumns_order typo: variable defined with 's' prefix, used without it — NameError"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
contributing_ids: [J-05/ai-expert]
```

**Verification:** PASS. File `_posts/2017-01-14-visualizing-and-comparing-distributions.md:194` confirmed: `scolumns_order = sort(data_plot.Region.unique())`. Line 206 confirmed: `order=columns_order,` — references an undefined variable. Running the boxplot section raises `NameError`.

**Recommendation:** Rename `scolumns_order` to `columns_order` on line 194.

**Spec reference:** https://docs.python.org/3/library/exceptions.html#NameError

---

### U-006 · P1 · Code/JS (AI-expert, verified)

```yaml
id: U-006
title: "sort() depends on deprecated %pylab inline magic — NameError in standard Python environments"
category: Code/JS
severity: P1
confidence: HIGH
effort: 15m
agents: [ai-expert]
contributing_ids: [C-08/ai-expert]
```

**Verification:** PASS. File `_posts/2017-01-14-visualizing-and-comparing-distributions.md:20` confirmed: `%pylab inline`. Lines 194, 247, 393 confirmed: bare `sort()` calls that resolve only in IPython's `%pylab` namespace. `%pylab` deprecated since IPython 8.0 (2022).

**Recommendation:** Replace `%pylab inline` with explicit imports (`import numpy as np`, `import matplotlib.pyplot as plt`). Replace all bare `sort()` calls with `np.sort()`.

**Spec reference:** https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pylab

---

### U-007 · P2 · Code/JS (AI-expert, verified)

```yaml
id: U-007
title: "Four typos in NYC taxi post — 'wekday', 'ahs', 'atleast', 'straighforward'"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
contributing_ids: [C-14/blogging]
```

**Verification:** PASS. File `_posts/2017-03-14-transportation-graph-nyc-taxi-data.md` Bash grep confirmed all four: line 48 "wekday", line 48 "ahs multiple entries", line 63 "atleast", line 115 "straighforward".

**Recommendation:** Fix all four typos. Run a spell-check pass over all posts.

---

### U-008 · P2 · Content (AI-expert, verified)

```yaml
id: U-008
title: "Galactic morphology post ends before showing any results — incomplete post published"
category: Content
severity: P2
confidence: HIGH
effort: 1h
agents: [blogging, ai-expert]
contributing_ids: [C-07/blogging, C-04/ai-expert]
```

**Verification:** PASS. File `_posts/2017-07-25-galactic-morphology-using-deep-learning.md` confirmed: describes CNN architecture, dropout, batch normalization, then ends with References section. No Results section. No training metrics, no RMSE, no example predictions, no comparison to leaderboard. The date mismatch is also confirmed: filename says `2017-07-25`, front matter `date: 2017-07-15T15:39:55-04:00` (10-day discrepancy).

**Recommendation:** Add a Results section with validation RMSE, example predictions, and comparison to Galaxy Zoo challenge baseline. Fix filename to match front matter date: rename to `2017-07-15-galactic-morphology-using-deep-learning.md`.

**Peer reference:** https://lilianweng.github.io — all posts include results tables and qualitative analysis

---

### U-009 · P2 · Accessibility (designer, verified)

```yaml
id: U-009
title: "Footer RSS icon is orange (#fa9b39) while all other footer icons are muted gray — inconsistency"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
contributing_ids: [D-09/designer]
```

**Verification:** PASS. JS eval: `rssColor: "rgb(250, 155, 57)"` (orange), `githubColor: "rgb(73, 78, 82)"` (gray). Root cause confirmed: `_footer.scss` muting rule covers `.fas, .fab` classes but not `.fa` (legacy class used by the RSS icon).

**Recommendation:** Extend footer muting rule to also cover `.fa` class: `.page__footer { .fas, .fab, .far, .fal, .fa { color: var(--global-text-color-light); } }`.

---

### U-010 · P2 · Accessibility (designer, verified)

```yaml
id: U-010
title: "Focus ring drawn inside element (outline-offset: -2px) — effectively invisible on text links"
category: Accessibility
severity: P2
confidence: MED
effort: 15m
agents: [designer]
contributing_ids: [D-20/designer]
```

**Verification:** PASS (file-based). `_includes/_mixins.scss` — `%tab-focus` mixin uses `outline-offset: -2px`. Drawing a focus ring inside a text link's boundary, on top of its underline, produces a ring that is visually obscured. WCAG 2.4.7 requires visible focus indicators.

**Recommendation:** Change `outline-offset: -2px` to `outline-offset: 3px`. Ensure `outline-width` is at minimum 3px.

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/focus-visible.html

---

### U-011 · P2 · SEO/Meta (profile, verified)

```yaml
id: U-011
title: "No Google Search Console verification — sitemap not submitted, Person JSON-LD not monitored"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 20m
agents: [profile]
contributing_ids: [B-21/profile]
```

**Verification:** PASS. `_config.yml` confirmed: no `google_site_verification` key. `_includes/seo.html:101-103` has the block for it but it never fires. Sitemap exists at `/sitemap.xml` (via `jekyll-sitemap` plugin) but is not submitted.

**Evidence:** Without Search Console, Nilesh cannot verify that the Person JSON-LD is being parsed correctly, see SERP rankings, or submit the sitemap. For a common name, confirming knowledge-graph disambiguation requires Search Console access.

**Recommendation:** Register in Google Search Console, get the verification meta tag, add to `_config.yml`, submit `https://nilesh-patil.github.io/sitemap.xml`.

**Spec reference:** https://developers.google.com/search/docs/monitor-debug/search-console-start

---

### U-012 · P2 · Content (hiring, verified)

```yaml
id: U-012
title: "Columbia University collaboration: no metrics, no URL, no headcount — strongest credibility signal left unexpanded"
category: Content
severity: P2
confidence: HIGH
effort: 30m
agents: [hiring]
contributing_ids: [H-23/hiring]
```

**Verification:** PASS. CV and About page confirmed: "Built Dream Sports' collaboration with Columbia University, NY and helped establish a multi-million-dollar research center" — one sentence, no further detail. No paper count from the collaboration. No faculty co-author named. No URL.

**Recommendation:** Expand to include: name of the research center or partnership (if public), number of students/post-docs supervised, output (papers, prototypes), and a link if the program has a public web presence. This is the single most impressive external-validation signal on the site and deserves more than one sentence.

**Peer reference:** https://research.google/outreach/university-relations/ — Google's research partnerships page demonstrates how to frame academic collaboration with verifiable specific outcomes

---

### U-013 · P2 · Content (hiring, verified)

```yaml
id: U-013
title: "About page 'Technical focus' is a raw capability list — no outcomes, no scale, not differentiated from IC level"
category: Content
severity: P2
confidence: HIGH
effort: 1h
agents: [hiring]
contributing_ids: [H-16/hiring]
```

**Verification:** PASS. `_pages/about.md` confirmed: 7-item bullet list of capabilities with no outcomes attached. No number, no metric, no "built X system that achieved Y at Z scale."

**Recommendation:** Convert to 3–4 accomplishment statements anchoring each capability to a scale metric or outcome. Move the raw capability list to a "Skills" section lower on the CV.

**Peer reference:** https://huyenchip.com/about/ — achievements stated with context and outcome, not as a raw skill list

---

### U-014 · P2 · Content (blogging, verified)

```yaml
id: U-014
title: "Comments widget configured but disabled — giscus repo_id and category_id are empty"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
contributing_ids: [C-03/blogging]
```

**Verification:** PASS. `_config.yml` confirmed: `repo_id: ""`, `category_id: ""`. Post defaults confirmed `comments: false`. JS eval: `hasComments: false` on post page.

**Evidence:** Without comments, the blog is broadcast-only. Comments become a meaningful engagement signal as new posts are published on current AI topics.

**Recommendation:** Enable GitHub Discussions on the repo, run `https://giscus.app`, paste generated IDs into `_config.yml`, set post default to `comments: true`.

---

### U-015 · P2 · Performance (webdev, verified)

```yaml
id: U-015
title: "No <link rel=preload> for FontAwesome woff2 files — FOUT on icon rendering"
category: Performance
severity: P2
confidence: MED
effort: 30m
agents: [webdev]
contributing_ids: [P-02/webdev]
```

**Verification:** PASS (reported). No preload hints confirmed: `evaluate_script` showed `preloads: []` on home page. FA woff2 files discovered only after CSS parsing, causing FOUT on icon elements. The avatar already has `fetchpriority="high"` which mitigates LCP.

**Recommendation:** Add preload hints for `fa-solid-900.woff2` and `fa-brands-400.woff2` in `_includes/head/custom.html`.

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel/preload

---

### U-016 · P2 · Design/UX (designer, verified)

```yaml
id: U-016
title: "TOC hardcoded background-color: #fff — latent dark-mode regression depending on SCSS compile order"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
contributing_ids: [D-16/designer]
```

**Verification:** PASS (file-based). `_navigation.scss:388` confirmed: `.toc { background-color: #fff }`. Currently overridden by `_sidebar.scss` which sets `background-color: var(--global-toc-bg-color)` — an undefined variable that resolves to transparent. Current state is correct by accident of compilation order, not by intent.

**Recommendation:** Replace `background-color: #fff` with `background-color: var(--global-toc-bg-color, var(--global-bg-color))`. Define `--global-toc-bg-color` in theme files if a distinct surface color is desired.

---

### U-017 · P2 · Content (blogging, verified)

```yaml
id: U-017
title: "Post pagination shows only 'Previous'/'Next' labels — no post title visible without hover"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
contributing_ids: [C-23/blogging]
```

**Verification:** PASS (file-based). `_includes/post_pagination.html` confirmed: visible text is "Previous" and "Next" only; title is in `title` attribute (tooltip on hover only).

**Recommendation:** Include the adjacent post title in the visible link text. Add `← {{ page.previous.title }}` as the visible link content.

**Peer reference:** https://simonwillison.net — previous/next navigation shows full post titles as visible link text

---

### U-018 · P2 · Content (blogging, verified)

```yaml
id: U-018
title: "Placeholder alt text 'image' on 9 HAR post figures; 'png' on 6 distributions post figures"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
contributing_ids: [C-10/blogging, C-11/blogging]
```

**Verification:** PASS. Bash grep confirmed 11 occurrences of `alt="image"` or `alt="png"` across both posts. HAR post: all 9 figures use `alt="image"`. Distributions post: 6 of 7 figures use `alt="png"`.

**Recommendation:** Replace all `alt="image"` and `alt="png"` with descriptive text matching the figure's content (e.g., "Confusion matrix for Random Forest model on test set — 94.37% accuracy").

**Spec reference:** WCAG 2.1 SC 1.1.1 https://www.w3.org/TR/WCAG21/#non-text-content

---

### U-019 · P2 · Content (blogging)

```yaml
id: U-019
title: "Year-archive and /posts/ are duplicate pages — no differentiation between them"
category: Design/UX
severity: P2
confidence: HIGH
effort: 1h
agents: [blogging]
contributing_ids: [C-17/blogging]
```

**Verification:** PASS (file-based). Both pages use identical Liquid templates, looping over `site.posts` with `archive__subtitle` year grouping. Home page footer links to both "All posts →" and "Posts by year →" which resolve to identical content.

**Recommendation:** Merge into one canonical URL (`/posts/` with year grouping) and redirect `/year-archive/`, or differentiate by adding a count-per-year summary index at the top of `/year-archive/`.

---

### U-020 · P2 · Brand (profile, verified)

```yaml
id: U-020
title: "Dream11 tenure title 'Senior Principal Research Scientist / Head of Applied Research' is ambiguous — promotion timing unclear"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [hiring]
contributing_ids: [H-19/hiring]
```

**Verification:** PASS. `_pages/cv.md` confirmed: single compound title for entire 2019–2026 Dream11 tenure with no internal date break.

**Recommendation:** Split into two dated entries to show when the Head of Applied Research promotion occurred. This removes ambiguity about how long the candidate held a leadership-track title.

---

---

## Section 3 — Top-15 Priority Table

| # | ID | Title | Category | Severity | Confidence | Effort | Agents |
|---|----|----|----------|----------|------------|--------|--------|
| 1 | C-001 | main.min.js SyntaxError — jQuery dead on all pages | Code/JS | P0 | HIGH | 15m | 4 |
| 2 | C-002 | GitHub bio shows stale Dream11 title | Brand | P0 | HIGH | 15m | 2 |
| 3 | C-003 | 6-year publishing gap — blog signals disengagement | Content | P0 | HIGH | Ongoing | 3 |
| 4 | U-001 | dask_ml.KMeans.partial_fit doesn't exist — AttributeError | Code/JS | P0 | HIGH | 1h | 1 |
| 5 | C-009 | ACL 2026 author order misrepresented as first-author | Content | P1 | HIGH | 15m | 2 |
| 6 | C-010 | Portfolio has zero AI/LLM artifacts | Content | P1 | HIGH | 4h | 2 |
| 7 | C-007 | Two H1 elements on home page; sidebar H2 before H1 | Accessibility | P1 | HIGH | 30m | 2 |
| 8 | C-006 | No skip-to-content link | Accessibility | P1 | HIGH | 30m | 2 |
| 9 | C-013 | Three competing Twitter/X handles in structured data | Brand | P1 | HIGH | 1h | 2 |
| 10 | C-008 | Duplicate meta[name=description] | SEO/Meta | P1 | HIGH | 15m | 2 |
| 11 | C-014 | Person+Organization JSON-LD on same URL | SEO/Meta | P1 | HIGH | 15m | 1 |
| 12 | C-004 | TOC links at 9.28px uppercase — unreadable | Accessibility | P1 | HIGH | 15m | 2 |
| 13 | C-012 | Share buttons produce blank/untitled social posts | Content | P1 | HIGH | 15m | 2 |
| 14 | C-005 | Follow button and nav toggle missing aria-expanded | Accessibility | P1 | HIGH | 15m | 2 |
| 15 | C-015 | Teaching and Talks pages are completely empty | Content | P1 | HIGH | 30m | 2 |

---

## Section 4 — Items NEEDING USER INPUT

---

### N-001

**ID:** B-03 / B-08 / B-09 / C-013  
**Brief:** Three different Twitter/X handles appear across the site: `@ensembledme` (twitter:site meta, sidebar link, author.twitter), `@optimistic_flw` (in sameAs JSON-LD), `@nilesh-patil` (in sameAs JSON-LD). X blocks automated verification (HTTP 402) so ownership cannot be confirmed.  
**The choice:** Which handle, if any, does Nilesh own and actively use professionally on X/Twitter? If none, do the X links stay at all?  
**Critical note:** `medium.com/@ensembledme` is confirmed to be a food/lifestyle blogger with no connection to Nilesh. If `@ensembledme` on X is similarly not Nilesh's account, then the Twitter Card `twitter:site = @ensembledme` and the sidebar X link credit a stranger's account on every page share — this would be a P0 reputational issue.  
**My recommendation:** If Nilesh cannot confirm ownership of `@ensembledme` on X, immediately remove it from `twitter.username`, `author.twitter`, and retain only the verified handle in `social.links`. If `@optimistic_flw` is the professional account, set all three surfaces to that.

---

### N-002

**ID:** B-04 / C-029  
**Brief:** ORCID `0000-0002-3438-8571` is confirmed Nilesh's (via ORCID API), is referenced on GitHub, but is not in the site's `social.links` or `sameAs`. The ORCID employment record still shows Dream Sports only (not DreamStreet).  
**The choice:** Does Nilesh want to surface ORCID in the sidebar and in JSON-LD sameAs? (It is already public on GitHub, so this is not a privacy expansion.)  
**My recommendation:** Yes, add it. For a common name in academic/research circles, ORCID in sameAs is the strongest disambiguation anchor. Also update the ORCID profile with the DreamStreet employment.

---

### N-003

**ID:** B-10 / C-030  
**Brief:** No email or contact form exists anywhere on the site. The config explicitly marks email as intentionally omitted.  
**The choice:** Add a professional email (role-based: nilesh@dreamstreet.ai) or a lightweight contact form?  
**My recommendation:** Add at minimum a role-based email to the About page "Get in touch" section. The alternative — a "LinkedIn is fastest" CTA sentence — adds zero code but provides signal. Either option reduces recruiter friction materially.

---

### N-004

**ID:** B-11 / B-14  
**Brief:** LinkedIn headline and Google Scholar affiliation cannot be automatically verified (both require authenticated access or return CAPTCHA). Both surfaces likely still show Dream11-era titles.  
**The choice:** Has Nilesh manually updated (a) LinkedIn headline to "Head of AI at DreamStreet" and (b) Google Scholar affiliation to DreamStreet?  
**My recommendation:** If not, do both immediately — they are 5-minute edits each with high visibility to recruiters and paper reviewers.

---

### N-005

**ID:** B-12  
**Brief:** Medium profile bio (`nilesh-patil.medium.com`) shows "Interested in applied machine learning, statistics and data science" — last post January 2020.  
**The choice:** Is Nilesh planning to publish new content on Medium? If not, should the Medium link be deprioritized or updated to note it is an archive?  
**My recommendation:** Update the Medium profile bio to current positioning regardless of publishing plans. A 5-minute edit removes the 2020 "data scientist" impression for anyone who clicks through.

---

### N-006

**ID:** H-03 / C-011  
**Brief:** No team size is mentioned for any role. Dream11: "led a high-performing cross-continent team" with no headcount.  
**The choice:** What are the approximate team sizes for Dream11 (Head of Applied Research phase) and DreamStreet? Even a range ("8–12 person research org") is sufficient.  
**My recommendation:** Add at least a range for each role. This is the single most effective signal change for VP-level calibration at a hiring committee.

---

### N-007

**ID:** B-17  
**Brief:** All avatar/logo files are named `ensembledme.jpg` / `ensembledme.webp`. The `ensembledme` handle belongs to an unrelated food blogger on Medium and potentially on X. Cached OG images from this URL will carry the `ensembledme` filename in social previews.  
**The choice:** Once the handle question (N-001) is resolved, should all image files be renamed to `nilesh-patil.jpg` / `nilesh-patil.webp`?  
**My recommendation:** Yes, if `@ensembledme` is confirmed to not be Nilesh's primary professional identity, rename the files to `nilesh-patil.jpg` / `nilesh-patil.webp` and update all `_config.yml` references.

---

### N-008

**ID:** H-25 / H-22  
**Brief:** No board-level exposure, advisory roles, or peer-review service is mentioned. Both are signals expected at VP-AI candidate level.  
**The choice:** Does any reviewable/PC service exist (NeurIPS/ICLR/ACL reviewer, workshop organizer, advisory board member)? If yes, add it. If not, this is an actionable gap.  
**My recommendation:** If any such service exists, add a "Service" section to the CV. If not, consider taking on one advisory role at an AI startup within the next 6 months to create this signal before the next active job search.

---

## Section 5 — Dropped During Verification

| # | Original ID | Finding | Drop Reason |
|---|------------|---------|-------------|
| 1 | B-05/profile | "Person JSON-LD url = localhost in dev server" | STALE/DEV-ARTIFACT. `_config.yml:11` sets `url: "https://nilesh-patil.github.io"`. Production JSON-LD renders the correct canonical URL. The dev-server override (`http://localhost:4000`) is expected and correct behavior. The production canonical (`link[rel="canonical"]`) also confirmed correct on the home page. |
| 2 | B-15/profile | "Site canonical URL stale on home page load" | STALE/DEV-ARTIFACT. The JS eval showing a stale canonical was reading a previous page's cached value in the dev browser. Fresh navigation to home page confirmed `canonical: "https://nilesh-patil.github.io/"` — correct. Not a production issue. |
| 3 | D-22/designer | "Sepia masthead does not use sepia background — color mismatch on scroll" | STALE/LOW-CONFIDENCE. Designer explicitly flagged this as LOW confidence and a possible false alarm from mid-navigation measurement. The designer's second check on a stable sepia page confirmed the masthead visually matches the body sepia color. No independent verification issue found. Dropped per overseer rule 6. |
| 4 | S-02/webdev (as originally filed) | "og:image references .jpg but site serves .webp" | FAILED VERIFICATION. Both `ensembledme.jpg` (107 KB) and `ensembledme.webp` (39 KB) exist in the repo. The JPEG is served correctly to OG/social scrapers. The finding's premise (JPEG missing) is false. Retained as C-025 with the finding downgraded to an optimization opportunity rather than a broken-image bug. |

---

## Appendix — Source Agent Report ID Mapping

| Canonical ID | Contributing Agent IDs |
|---|---|
| C-001 | J-01/webdev, D-05/designer, D-06/designer, phase0-prefinding |
| C-002 | H-01/hiring, B-01/profile |
| C-003 | C-03/ai-expert, H-02/hiring, C-01/blogging, C-02/blogging |
| C-004 | D-01/designer, D-13/designer |
| C-005 | X-01/webdev, X-02/webdev |
| C-006 | X-03/webdev |
| C-007 | X-04/webdev, H-24/hiring |
| C-008 | S-01/webdev, B-06/profile |
| C-009 | C-01/ai-expert, H-08/hiring |
| C-010 | C-02/ai-expert, C-09/ai-expert, H-05/hiring, H-17/hiring, H-18/hiring |
| C-011 | H-03/hiring, H-19/hiring, H-04/hiring |
| C-012 | C-04/blogging |
| C-013 | B-03/profile, B-08/profile, B-09/profile, B-23/profile |
| C-014 | B-07/profile, B-22/profile |
| C-015 | D-25/designer, H-13/hiring |
| C-016 | D-08/designer, X-07/webdev |
| C-017 | D-07/designer |
| C-018 | X-06/webdev, D-23/designer |
| C-019 | X-08/webdev |
| C-020 | X-05/webdev |
| C-021 | X-09/webdev |
| C-022 | D-03/designer, D-12/designer |
| C-023 | D-04/designer |
| C-024 | D-05/designer, D-06/designer |
| C-025 | S-02/webdev, B-17/profile |
| C-026 | S-03/webdev |
| C-027 | I-01/webdev |
| C-028 | I-02/webdev |
| C-029 | B-04/profile, B-13/profile |
| C-030 | H-06/hiring, B-10/profile |
| C-031 | B-12/profile, B-18/profile |
| C-032 | P-03/webdev |
| C-033 | C-09/blogging, C-22/blogging |
| C-034 | C-25/blogging, H-10/hiring |
| C-035 | I-03/webdev |
| U-001 | J-01/ai-expert |
| U-002 | J-02/ai-expert |
| U-003 | J-03/ai-expert |
| U-004 | J-04/ai-expert |
| U-005 | J-05/ai-expert |
| U-006 | C-08/ai-expert |
| U-007 | C-14/blogging |
| U-008 | C-07/blogging, C-04/ai-expert |
| U-009 | D-09/designer |
| U-010 | D-20/designer |
| U-011 | B-21/profile |
| U-012 | H-23/hiring |
| U-013 | H-16/hiring |
| U-014 | C-03/blogging |
| U-015 | P-02/webdev |
| U-016 | D-16/designer |
| U-017 | C-23/blogging |
| U-018 | C-10/blogging, C-11/blogging |
| U-019 | C-17/blogging |
| U-020 | H-19/hiring |

**Agent report IDs demoted to flat list (below P1 threshold, verified but low-severity):**
D-11 (post listing metadata color), D-15 (avatar border padding), D-18 (site title color), D-19 (publications list separator), D-21 (nav hover double underline), D-24 (active nav indicator color), J-02/webdev (screen.orientation guard), B-16/profile (X absent from footer), B-19/profile (about page omits X), B-20/profile (Stack Exchange Stats not in sidebar), B-24/profile (PDF staleness notice), C-05/ai-expert (HAR OOB validation note), C-06/ai-expert (distributed k-means date discrepancy), C-07/ai-expert (publications static fallback), C-10/ai-expert (DreamStreet footprint), C-11/ai-expert (BatchNorm train/inference), C-14/ai-expert (6+ additional publications vague), C-15/blogging (no internal cross-linking), C-16/blogging (About page lacks blog link), C-18/blogging (post header image inconsistency), C-19/blogging (RSS not visibly discoverable), C-20/blogging (opening hooks not click-inviting), C-21/blogging (k-means post conclusion placement), C-24/blogging (Medium posts not cross-linked), J-06/ai-expert (kdeplot positional arg), J-07/ai-expert (lmplot argument order), J-08/ai-expert (init_max_iter moot), H-07/hiring (DreamStreet stage context), H-09/hiring (about page duplication), H-11/hiring (no named shipped product), H-12/hiring (CV PDF no date stamp), H-14/hiring (2013-2014 gap), H-15/hiring (no Medium cross-links), H-20/hiring (no social proof), H-21/hiring (bio-papers dilute AI brand), H-25/hiring (board-level exposure), C-06/blogging (post title click-invitingness), C-08/blogging (galactic date filename), C-12/blogging (posts end abruptly), C-13/blogging (excerpts as topic labels).
