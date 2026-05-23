# Professional Brand Audit
**Surfaces examined:** site (live curl), GitHub API, Medium @ensembledme, X/@ensembledme, repo source

## [Severity: P0] — GitHub bio still says "Sr. Principal Research Scientist @Dream11"
**Surface:** GitHub
**Location:** https://github.com/nilesh-patil
**Evidence:** GitHub API returns `bio: "Sr. Principal Research Scientist @Dream11"` and `company: "@dream11"`. Site sidebar reads "Head of AI at DreamStreet." Recruiter cross-checks GitHub → sees stale, junior-sounding title from prior employer. Site links to this profile from sidebar and footer.
**Recommendation:** Update GitHub bio to "Head of AI @ DreamStreet | AI systems & applied research" and company field to DreamStreet org handle at https://github.com/settings/profile.
**Reference:** https://github.com/karpathy, https://github.com/swyx

## [Severity: P0] — Medium handle @ensembledme resolves to a food-recipe blog with zero AI content
**Surface:** Medium
**Location:** https://medium.com/@ensembledme
**Evidence:** WebFetch returns profile name "Ensembled Me," bio "Lover of music, travel, and fashion," 10 posts — all food recipes (Double Chocolate Chip Banana Muffins, Ramen Noodle Soup, Beef Curry, etc.), last published March 24, 2024. This handle is linked from sidebar, footer, and About page.
**Recommendation:** (a) Create a separate Medium account for AI writing and update site references, or (b) remove Medium link from sidebar/footer by deleting `medium:` from `_config.yml` author block and `_data/authors.yml`. Option (b) is 5 minutes and removes reputational risk today.
**Reference:** https://medium.com/@francois.chollet

## [Severity: P0] — No `og:image` tag: social cards are blank when shared on LinkedIn or X
**Surface:** Site
**Location:** All pages
**Evidence:** Live curl of homepage: OG block contains only `og:locale`, `og:site_name`, `og:title`, `og:url`. No `og:image` or `og:description`. `seo.html` only emits `og:image` when page has `header.image` — none of home/about/CV set it. Twitter card type is `summary` (not `summary_large_image`).
**Recommendation:** Add `og_image: "ensembledme.jpg"` to `_config.yml`. Template already checks `site.og_image` on lines 62–64. One config line. Also add `og_description:`.
**Reference:** https://ogp.me/#structured

## [Severity: P0] — No schema.org Person JSON-LD: zero structured identity for "Nilesh Patil AI" queries
**Surface:** Site
**Location:** `_config.yml` (no `social:` block), `_includes/seo.html:89`
**Evidence:** `seo.html` emits Person `application/ld+json` only when `site.social` is configured. No `social:` key in `_config.yml`. Curl of homepage produces zero JSON-LD output.
**Recommendation:** Add `social:` block as described in webdev/hiring audits.
**Reference:** https://developers.google.com/search/docs/appearance/structured-data/person

## [Severity: P1] — Blog has a 6-year content gap: newest post is May 2020
**Surface:** Site
**Recommendation:** Publish minimum 3 posts in 2026. Or replace homepage "Recent posts" with "Selected writing" surfacing publications/portfolio instead.
**Reference:** https://eugeneyan.com/writing/

## [Severity: P1] — CV contact has dead breadcrumb: "LinkedIn — see GitHub profile for current link"
**Surface:** Site
**Location:** `/cv/` line 94
**Evidence:** CV defers LinkedIn URL to GitHub profile. GitHub profile doesn't visibly link to LinkedIn. Dead end at the moment of maximum intent.
**Recommendation:** Add LinkedIn URL directly to CV and `_config.yml`/`_data/authors.yml`. If uncertain, delete the LinkedIn bullet entirely.
**Reference:** https://simonwillison.net/about/

## [Severity: P1] — X/Twitter link may point to a dormant account; last activity unverifiable
**Surface:** X/Twitter
**Location:** https://twitter.com/ensembledme
**Evidence:** X returned HTTP 402 (payment gate); last post date unconfirmed.
**Recommendation:** Check the X account. If last post predates 2024, remove `twitter:` from `_config.yml` author and `_data/authors.yml`.

## [Severity: P1] — Missing `og:description`: social shares have title only
**Surface:** Site
**Location:** `_config.yml`, `_includes/seo.html:123-125`
**Recommendation:** Add `og_description: "Head of AI at DreamStreet. Building compliance-aware AI systems for regulated financial workflows. AI harness design, agentic evaluators, LLMs at scale."` to `_config.yml`.

## [Severity: P1] — No Search Console verification: Google hasn't been told to index this site
**Surface:** Site
**Location:** `_config.yml` (no `google_site_verification`)
**Evidence:** `seo.html:101` has the conditional verification tag — dormant. `jekyll-sitemap` is enabled but no Search Console consumes it.
**Recommendation:** Verify at https://search.google.com/search-console, paste content value into `_config.yml`.

## [Severity: P1] — Portfolio has only 2 items, both from 2017-era: mismatches Head-of-AI resume
**Surface:** Site
**Recommendation:** Add 3–5 case studies for current-era work. No proprietary disclosure required.

## [Severity: P2] — "Currently exploring" duplicates About opening paragraph verbatim
**Surface:** Site
**Location:** `/about/` lines 32–34
**Recommendation:** Replace with forward-looking specifics: agentic reliability under compliance constraints; structured output guarantees; on-device SLM benchmarks.

## [Severity: P2] — Sidebar X link uses old twitter.com domain; footer is inconsistent
**Surface:** Site
**Recommendation:** Decide whether X is active and make it consistent across sidebar and footer. Update href to https://x.com/ if keeping.

## [Severity: P2] — No site analytics: zero visibility into inbound
**Surface:** Site
**Recommendation:** Consider Plausible or GoatCounter (both privacy-first, GDPR-compliant, no cookies).
**Reference:** https://plausible.io, https://www.goatcounter.com

## [Severity: P2] — Nav label "Blog" implies active commentary; actual content is 2017 technical notebooks
**Surface:** Site
**Recommendation:** Rename in `_data/navigation.yml` from "Blog" to "Writing" or "Notes."
**Reference:** https://eugeneyan.com/

## Proposed bio versions
**12-word:** *Head of AI, DreamStreet. AI systems, agentic workflows, regulated environments.*

**2-sentence:** *Head of AI at DreamStreet, building compliance-aware AI architecture for SEBI-regulated investor and trader workflows. Previously led applied AI research at Dream11, including a Columbia University research collaboration and production ML systems at 250M+ user scale.*

**1-paragraph:** *I lead AI at DreamStreet (Mumbai), building deployable AI systems and agentic workflows for SEBI-regulated investor and trader products — spanning research, brokerage, and advisory domains. Before DreamStreet, I spent seven years at Dream Sports / Dream11 as Head of Applied Research, establishing a multi-million-dollar ML research collaboration with Columbia University, leading cross-continent teams across India and New York, and shipping production systems at 250M+ user scale. I am particularly interested in AI harness design, compliance-aware workflow architecture, and turning emerging model capabilities — LLMs, SLMs, agentic evaluators — into reliable, auditable products.*

## Surfaces that could not be fully audited
- **Google Scholar:** Bot detection blocked live scrape. Recommend manually verifying affiliation reads "DreamStreet" not "Dream11."
- **X/Twitter:** HTTP 402 payment gate; activity unconfirmed.
- **LinkedIn:** Intentionally omitted from site; no handle to audit.
