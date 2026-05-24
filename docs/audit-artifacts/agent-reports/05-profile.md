# Profile Audit — Cross-Surface Brand Consistency

**Agent:** 05 — Professional Profile / Brand Strategist
**Date:** 2026-05-24
**Scope:** Brand/identity consistency across site, GitHub, LinkedIn, Medium, X/Twitter, ORCID, Google Scholar

---

## Profile Verdict

Nilesh Patil's professional identity is **partially coherent but actively inconsistent across two high-traffic entry points**: his GitHub bio still reads "Sr. Principal Research Scientist @dream11" — a role he left in 2026 for Head of AI at DreamStreet — and `medium.com/@ensembledme`, a handle the site previously linked to, still resolves to a food-and-lifestyle blog with zero connection to AI or Nilesh. The site itself (since commit 3930ceb) correctly points to `nilesh-patil.medium.com` for content and `https://www.linkedin.com/in/ensembledme` for LinkedIn (verified live, correct profile). However, the site's structured data carries two Twitter handles simultaneously (`@ensembledme` in meta tags; `x.com/optimistic_flw` and `twitter.com/nilesh-patil` in JSON-LD sameAs), none of which can be independently verified as Nilesh's active professional X/Twitter presence via WebFetch (X returns 402). ORCID (0000-0002-3438-8571, referenced on GitHub but not on site) lists Dream Sports employment only — DreamStreet role is absent. Google Scholar blocks automated access. A recruiter who lands on GitHub first gets the wrong employer; a recruiter who clicks the Twitter/X link in the sidebar lands at `twitter.com/ensembledme` — a handle inconsistent with the `optimistic_flw` handle in the JSON-LD. None of these failures are catastrophic, but the GitHub staleness and the `@ensembledme` vs `@optimistic_flw` handle split are active trust-erosion risks.

---

## Findings

---

```yaml
id: B-01
title: "GitHub bio still shows 'Sr. Principal Research Scientist @dream11' — stale since 2026"
category: Brand
severity: P0
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** WebFetch of `https://github.com/nilesh-patil` returned bio: `"Sr. Principal Research Scientist @dream11"`. Site `_config.yml:49` and `_data/authors.yml:7` both say `"Head of AI at DreamStreet. Previously Head of Applied Research, Dream11."` CV (`_pages/cv.md:32`) shows DreamStreet start date as 2026 – Present.

**Why this matters:** GitHub is typically the first stop for a technical recruiter. The bio actively contradicts the site, CV, and LinkedIn. Someone opening three tabs simultaneously sees: GitHub (Dream11 Research Scientist) | LinkedIn (DreamStreet) | Site (DreamStreet Head of AI). The mismatch signals disorganization or — worse — a profile under someone else's control.

**Recommendation:** Update GitHub bio to: `"Head of AI at DreamStreet | Previously Head of Applied Research, Dream11 | nilesh-patil.github.io"`. This is a 1-minute edit in GitHub profile settings. NEEDS_USER_INPUT: confirm exact preferred wording.

**Fix snippet:** (off-site change — GitHub profile settings > Edit profile > Bio)
```
Head of AI at DreamStreet | Previously Head of Applied Research @Dream11 | nilesh-patil.github.io
```

**Spec reference:** https://schema.org/Person — `jobTitle` consistency across sameAs surfaces

---

```yaml
id: B-02
title: "medium.com/@ensembledme still resolves to a food/lifestyle blog — prior fix unverified live"
category: Brand
severity: P0
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** WebFetch of `https://medium.com/@ensembledme` returned: Name: "Ensembled Me", Bio: "Lover of music, travel, and fashion", with posts including "Double Chocolate Chip Banana Muffins", "Super Easy Ramen Noodle Soup", "Beef Curry", etc. This is emphatically NOT Nilesh Patil's AI blog.

Commit `3930ceb` claimed to fix the Medium link from `@ensembledme` to `nilesh-patil.medium.com`. Current site `_config.yml:55` and `_data/authors.yml:13` now correctly point to `https://nilesh-patil.medium.com/`. The sidebar rendered link (verified via JS eval) points to `https://nilesh-patil.medium.com/`. The footer (`_config.yml:267`) also correctly points to `https://nilesh-patil.medium.com/`.

**However:** The Person JSON-LD `sameAs` array (rendered in `<head>` via `_includes/seo.html`) includes `"https://nilesh-patil.medium.com/"` — CORRECT. The old `@ensembledme` Medium URL is NOT in the sameAs list. The prior fix appears complete on-site.

**Why this matters:** If any cached version of the site, any external blog aggregator, or any search engine still holds the old `@ensembledme` Medium link, clicks go to a food blog. The on-site fix is complete but the `@ensembledme` medium handle itself remains permanently a brand hazard — anyone who remembers or finds the old handle gets the food blog. There is no redirect from `medium.com/@ensembledme` to the correct profile.

**Recommendation:** Verify no remaining references to `medium.com/@ensembledme` exist in the repo, then monitor for any third-party aggregators still citing the old URL. Consider adding a note on the Medium profile at `nilesh-patil.medium.com` redirecting from the old handle, if possible.

**Fix snippet:** (verify clean — run in repo root)
```bash
grep -r "ensembledme" . --include="*.md" --include="*.yml" --include="*.html" | grep -v "_site" | grep -v ".git"
```

**Spec reference:** https://schema.org/Person — `sameAs` should only list canonical, verified profile URLs

---

```yaml
id: B-03
title: "Twitter/X handle split: three different handles across site surfaces"
category: Brand
severity: P1
confidence: HIGH
effort: 1h
agents: [profile]
```

**Evidence:** Verified via JS evaluation of rendered home page HTML:
1. `_config.yml:30`: `twitter.username: ensembledme` → renders as `<meta name="twitter:site" content="@ensembledme">` on every page
2. `_config.yml:38`: `social.links` includes `"https://x.com/optimistic_flw"` → in Person JSON-LD `sameAs`
3. `_config.yml:39`: `social.links` includes `"https://twitter.com/nilesh-patil"` → in Person JSON-LD `sameAs`
4. Sidebar rendered link: `https://twitter.com/ensembledme` (from `_config.yml:54` `author.twitter: ensembledme`)

WebFetch of `https://x.com/optimistic_flw` and `https://x.com/ensembledme` both returned HTTP 402 (X blocks automated access). `https://twitter.com/nilesh-patil` redirected to `https://x.com/nilesh-patil` which also returned 402. Cannot verify which handles are active. NEEDS_USER_INPUT: which of `@ensembledme`, `@optimistic_flw`, `@nilesh-patil` is (a) owned by Nilesh, (b) professionally active?

**Why this matters:** Google's knowledge graph disambiguation for "Nilesh Patil" reads the JSON-LD `sameAs` array. Having three distinct Twitter/X URLs in that array — two of which are contradictory — degrades the disambiguation signal. A recruiter seeing `@ensembledme` in the sidebar but `@optimistic_flw` in the structured data (visible via browser devtools) will be confused about which is the professional account.

**Recommendation:** Consolidate to one canonical X handle. Update `_config.yml` so that `twitter.username`, `author.twitter`, and the `social.links` entry all reference the same handle. Remove the extra entries.

**Fix snippet:** (pending user clarification — template assuming `optimistic_flw` is primary)
```yaml
# _config.yml
twitter:
  username: optimistic_flw   # was: ensembledme

author:
  twitter: "optimistic_flw"  # was: ensembledme

social:
  links:
    - "https://github.com/nilesh-patil"
    - "https://x.com/optimistic_flw"      # keep only one
    # REMOVE: "https://twitter.com/nilesh-patil"
    - "https://scholar.google.co.in/citations?user=IIabY1sAAAAJ"
    - "https://nilesh-patil.medium.com/"
```

**Spec reference:** https://schema.org/Person — `sameAs` should contain unique, verified canonical URLs

---

```yaml
id: B-04
title: "ORCID profile (on GitHub) not linked from site; employment stale (Dream Sports only, no DreamStreet)"
category: Brand
severity: P1
confidence: HIGH
effort: 1h
agents: [profile]
```

**Evidence:** GitHub profile (WebFetch) shows ORCID: `0000-0002-3438-8571`. The ORCID API at `https://pub.orcid.org/v3.0/0000-0002-3438-8571/employments` returns 4 employment records: Dream Sports (Research Scientist, April 2019–ongoing), University of Rochester Medical Center (2018–2019), AXA (2014–2016), AbsolutData (2013–2014). DreamStreet (2026–present) is absent.

`_includes/author-profile.html:71` has an `{% if author.orcid %}` block but `_config.yml` and `_data/authors.yml` contain no `orcid:` field — so no ORCID link renders in the sidebar.

The Person JSON-LD `sameAs` does not include the ORCID URL.

**Why this matters:** ORCID is a primary disambiguation tool for academic/research profiles. "Nilesh Patil" is a very common Indian name. The ORCID record is the most authoritative identity anchor for the research community and is already known (linked from GitHub). Not surfacing it on the site misses a disambiguation signal. The stale Dream Sports employment on ORCID contradicts the DreamStreet role on the site — if a publisher or journal looks up the ORCID, they see a misaligned record.

**Recommendation:** (1) Add `orcid: "https://orcid.org/0000-0002-3438-8571"` to `_config.yml` author block and `_data/authors.yml`. (2) Add the ORCID URL to `social.links` in `_config.yml`. (3) Update the ORCID record itself to add DreamStreet employment. NEEDS_USER_INPUT: confirm ORCID ownership and whether to surface publicly on site.

**Fix snippet:**
```yaml
# _config.yml — author block addition
author:
  orcid: "https://orcid.org/0000-0002-3438-8571"

# _config.yml — social.links addition  
social:
  links:
    - "https://orcid.org/0000-0002-3438-8571"
    # ... existing links

# _data/authors.yml addition
"Nilesh Patil":
  orcid: "https://orcid.org/0000-0002-3438-8571"
```

**Spec reference:** https://schema.org/Person — `sameAs` with ORCID URI for researcher disambiguation

---

```yaml
id: B-05
title: "Person JSON-LD sameAs URL uses localhost in production — kills knowledge graph signal"
category: SEO/Meta
severity: P0
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** JS eval on rendered home page shows the Person JSON-LD:
```json
{
  "@type": "Person",
  "url": "http://localhost:4000",
  "sameAs": [...]
}
```
The `"url"` field is `"http://localhost:4000"` not `"https://nilesh-patil.github.io"`. This is because `seo.html:95` uses `seo_url` which is derived from `site.url`, and the dev server overrides `site.url` to `localhost:4000`. In production builds, `site.url` in `_config.yml:11` is `"https://nilesh-patil.github.io"` — so the production JSON-LD should render the correct URL.

**Why this matters:** This is a dev-server artifact. The live site at `https://nilesh-patil.github.io` must be verified post-deploy to confirm `"url"` renders as the canonical domain. If the site is ever built with `JEKYLL_ENV` not set to production, or if the `--url` flag is not passed, `localhost` could escape into the deployed `_site`. The `twitter:url` meta was also observed pointing to `http://localhost:4000/cv/` on the home page load — same root cause.

**Recommendation:** Verify the deployed site's `<script type="application/ld+json">` on `https://nilesh-patil.github.io` to confirm `"url"` is `"https://nilesh-patil.github.io"`. Add a CI check or post-deploy smoke test that greps `_site` for `localhost`.

**Fix snippet:** (post-deploy verification command)
```bash
grep -r "localhost" _site --include="*.html" | grep -v "<!-- " | head -20
```

**Spec reference:** https://schema.org/Person — `url` must be the canonical production URL

---

```yaml
id: B-06
title: "meta description and og:description contain different text — brand voice fragmentation"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** JS eval on home page confirmed:
- `<meta name="description">`: `"AI leader building deployable AI systems, agentic workflows, and organizational adoption in regulated and large-scale environments."` (from `_config.yml:8-9` site `description:`)
- `<meta property="og:description">`: `"Head of AI at DreamStreet. Building compliance-aware AI systems for SEBI-regulated investor and trader workflows. Previously Head of Applied Research, Dream11."` (from `_config.yml:27` `og_description:`)

These serve different audiences (crawlers vs. social shares) but describe the same person with completely different framing. The `description` leads with role-agnostic positioning; the `og:description` leads with title and employer.

**Why this matters:** When someone shares the site on LinkedIn, Slack, or X, the preview card shows the `og:description`. When Google Search shows the snippet, it uses `meta description`. The two frames are not contradictory but they are not coordinated — a different "voice" shows up depending on where the link surfaces. Peers at competing labs or Google Brain seeing the LinkedIn share see the title-first framing; someone Googling for "Nilesh Patil AI" sees the role-agnostic framing. Minor, but a polished personal brand site should have these aligned.

**Recommendation:** Align both to a single crisp statement. Suggested: `"Head of AI at DreamStreet — building compliance-aware AI systems, agentic workflows, and applied research at scale."` Use this for both `description:` and `og_description:` in `_config.yml`.

**Fix snippet:**
```yaml
# _config.yml
description: >-
  Head of AI at DreamStreet — building compliance-aware AI systems,
  agentic workflows, and applied research at scale. Previously Head of
  Applied Research, Dream Sports / Dream11.
og_description: "Head of AI at DreamStreet — building compliance-aware AI systems, agentic workflows, and applied research at scale."
```

**Spec reference:** https://ogp.me/ — `og:description` should consistently represent the page/person

---

```yaml
id: B-07
title: "Person JSON-LD type inconsistency: both Person and Organization schemas emitted — Organization has no name"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** JS eval on home page confirmed two JSON-LD blocks:
```json
{"@type": "Person", "name": "Nilesh Patil", "url": "http://localhost:4000", "sameAs": [...]}
{"@type": "Organization", "url": "http://localhost:4000", "logo": "http://localhost:4000/images/ensembledme.jpg"}
```
The Organization block (from `_includes/seo.html:134-141`, triggered by `site.og_image`) has no `name` field and uses the same URL as the Person. This creates a conflicting signal: the same URL is simultaneously a Person and an Organization.

**Why this matters:** Google's rich-results parser will attempt to reconcile two `@type` blocks on the same URL. A personal portfolio should be a `Person`, not an `Organization`. Emitting both without differentiated URLs confuses the knowledge graph and dilutes the disambiguation signal for "Nilesh Patil" — a critical concern for a common Indian name.

**Recommendation:** Remove the Organization JSON-LD block or convert it to use the employer's URL instead. The `og:image` Open Graph tag still works without the Organization schema.

**Fix snippet:** (in `_includes/seo.html`, remove or gate the Organization block)
```liquid
{% comment %}
  Organization JSON-LD removed — logo is declared via og:image instead.
  Emitting Person + Organization on same URL confuses knowledge graph.
{% endcomment %}
{% comment %}
{% if site.og_image %}
  <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Organization",
      ...
    }
  </script>
{% endif %}
{% endcomment %}
```

**Spec reference:** https://schema.org/Person — a personal site should declare exactly one `@type` per URL

---

```yaml
id: B-08
title: "Twitter card meta uses @ensembledme site handle but sidebar links to twitter.com/ensembledme — verify handle is Nilesh's"
category: Brand
severity: P1
confidence: MED
effort: 15m
agents: [profile]
```

**Evidence:** Meta tag (verified via JS eval): `<meta name="twitter:site" content="@ensembledme">`. Sidebar rendered link: `https://twitter.com/ensembledme` (from `author.twitter: ensembledme` in `_config.yml:54`). WebFetch of `https://x.com/ensembledme` returned HTTP 402 — cannot confirm profile ownership. Note: `medium.com/@ensembledme` is confirmed to be a food/lifestyle blog with no connection to Nilesh. If `@ensembledme` was Nilesh's Medium handle but belonged to someone else, the same may apply to Twitter. NEEDS_USER_INPUT: does Nilesh own @ensembledme on X/Twitter?

**Why this matters:** If `@ensembledme` on Twitter/X is NOT Nilesh's account (as `@ensembledme` on Medium is not his), then the Twitter Card `twitter:site` and the sidebar link both point to an unrelated third party. Every page share on X would credit the wrong account. This would be a P0 reputational risk.

**Recommendation:** Verify ownership of `@ensembledme` on X/Twitter. If not owned, update `_config.yml:30 twitter.username` and `_config.yml:54 author.twitter` to the correct handle (`optimistic_flw` if that is the primary account).

**Fix snippet:** (contingent on user confirmation)
```yaml
# _config.yml — if optimistic_flw is the correct handle
twitter:
  username: optimistic_flw
author:
  twitter: "optimistic_flw"
```

**Spec reference:** https://developer.twitter.com/en/docs/twitter-for-websites/cards/overview/markup — `twitter:site` should be the site owner's verified handle

---

```yaml
id: B-09
title: "social.links includes twitter.com/nilesh-patil — handle unverified, may not exist"
category: SEO/Meta
severity: P1
confidence: MED
effort: 15m
agents: [profile]
```

**Evidence:** `_config.yml:39`: `social.links` entry `"https://twitter.com/nilesh-patil"`. WebFetch of this URL: redirects to `https://x.com/nilesh-patil`, which returns HTTP 402. Cannot confirm whether this handle exists and belongs to Nilesh. This URL is in the Person JSON-LD `sameAs` array (verified via JS eval). NEEDS_USER_INPUT: does Nilesh own @nilesh-patil on X/Twitter?

**Why this matters:** If `twitter.com/nilesh-patil` does not exist or belongs to another person, Google's knowledge graph links "Nilesh Patil the AI leader" to a potentially wrong/dead account. For a common name, this degrades disambiguation. If the handle belongs to a different Nilesh Patil, this is a P0 reputational issue.

**Recommendation:** Remove `"https://twitter.com/nilesh-patil"` from `social.links` unless Nilesh can confirm ownership. Only include verified, active handles in `sameAs`.

**Fix snippet:**
```yaml
# _config.yml — remove unverified handle
social:
  links:
    - "https://github.com/nilesh-patil"
    - "https://x.com/optimistic_flw"    # retain only verified handle
    # REMOVE: "https://twitter.com/nilesh-patil"
    - "https://scholar.google.co.in/citations?user=IIabY1sAAAAJ"
    - "https://nilesh-patil.medium.com/"
```

**Spec reference:** https://schema.org/Person — `sameAs` should only list verified canonical identity URLs

---

```yaml
id: B-10
title: "No contact email anywhere — recruiter dead-end if LinkedIn wall blocks them"
category: Brand
severity: P1
confidence: HIGH
effort: 1h
agents: [profile]
```

**Evidence:** `_config.yml:44` comment: `# NOTE: email omitted intentionally — contact is via LinkedIn / GitHub only.` `_data/authors.yml:3`: same comment. `_includes/author-profile.html:51-53`: email link is gated on `{% if author.email %}` — never renders. `_pages/about.md` "Get in touch" section has 5 platform links but no email.

LinkedIn at `https://in.linkedin.com/in/ensembledme` exists and shows the profile — but only basic info is available without a login. A recruiter using LinkedIn without a premium account, or a researcher who is not on LinkedIn, has no direct contact path.

**Why this matters:** Peers in academia and open-source do not always have LinkedIn. A researcher from a university who wants to collaborate can reach GitHub but GitHub Issues are not appropriate for professional outreach. The contact friction is high for a site explicitly targeting "AI leader" positioning. Compare: Lilian Weng's profile (lilianweng.github.io) includes institutional email. Andrej Karpathy's site (karpathy.ai) links to a contact form.

**Recommendation:** Add a professional email or contact form. Minimum: add a `mailto:` link on the `/about/` and `/cv/` pages. If privacy is a concern, use a role-based address (e.g., `nilesh@dreamstreet.ai`) or a contact form. NEEDS_USER_INPUT: willing to add professional email or contact form?

**Fix snippet:** (minimal — add to `_pages/about.md` "Get in touch" section)
```markdown
- **Email** — [nilesh@dreamstreet.ai](mailto:nilesh@dreamstreet.ai)
```

**Spec reference:** https://schema.org/Person — `email` or `contactPoint` for professional reachability

---

```yaml
id: B-11
title: "LinkedIn profile shows 'DreamStreet' but no headline/title visible — site says 'Head of AI'"
category: Brand
severity: P2
confidence: MED
effort: 15m
agents: [profile]
```

**Evidence:** WebFetch of `https://in.linkedin.com/in/ensembledme` returned: Name: Nilesh Patil, Employer: DreamStreet, Location: Mumbai — correct. Full headline not visible without auth. The site bio (`_config.yml:49`) says "Head of AI at DreamStreet." Consistency of the LinkedIn headline cannot be confirmed without login. NEEDS_USER_INPUT: does LinkedIn headline say "Head of AI at DreamStreet"?

**Why this matters:** LinkedIn is the primary surface for professional discovery. If the LinkedIn headline says a different title (e.g., still "Sr. Principal Research Scientist" from the Dream11 era), it would create the same inconsistency as B-01. Even partial confirmation that DreamStreet appears as employer is encouraging.

**Recommendation:** Verify and update LinkedIn headline to "Head of AI at DreamStreet" if not already set.

**Fix snippet:** (off-site — LinkedIn profile > Edit intro > Headline)
```
Head of AI at DreamStreet | AI systems, agentic workflows, applied research
```

**Spec reference:** https://schema.org/Person — `jobTitle` consistency

---

```yaml
id: B-12
title: "Medium profile bio is 5+ years stale — shows data science tooling author, not AI leader"
category: Brand
severity: P1
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** WebFetch of `https://nilesh-patil.medium.com/` returned: Bio: `"Interested in applied machine learning, statistics and data science"`. Most recent article: Jan 2020 (Dask parallel computing). The site positions Nilesh as "Head of AI at DreamStreet" building "agentic workflows" and "compliance-aware AI." The Medium bio has not been updated in 5+ years and does not reflect the current positioning.

**Why this matters:** Medium has 318 followers — not massive but the profile appears in searches. A reader clicking through from a 2020 article to the profile sees a "data scientist" bio, not an AI leader. The positioning mismatch reduces the halo effect that the site's strong AI-leader narrative creates.

**Recommendation:** Update Medium profile bio to current positioning. Also consider publishing 1-2 AI-focused articles to signal that the profile is active in the current AI era. NEEDS_USER_INPUT: is Nilesh planning Medium content?

**Fix snippet:** (off-site — Medium profile settings > Bio)
```
Head of AI at DreamStreet. Building compliance-aware AI systems, agentic workflows, 
and applied research at scale. Previously Head of Applied Research, Dream11.
```

**Spec reference:** https://schema.org/Person — `description` consistency across platforms

---

```yaml
id: B-13
title: "ORCID not in Person JSON-LD sameAs — misses key researcher disambiguation anchor"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** ORCID `0000-0002-3438-8571` confirmed for Nilesh Patil via `https://pub.orcid.org/v3.0/0000-0002-3438-8571/person` (returns given name: Nilesh, family name: Patil, keywords: Machine Learning, Computer Vision). Person JSON-LD `sameAs` (JS eval on home page) contains: GitHub, x.com/optimistic_flw, Scholar, twitter.com/nilesh-patil, Medium — no ORCID URL.

**Why this matters:** ORCID is the standard persistent identifier for researchers. Google Scholar profiles, academic publishers, and institutional repositories use it for disambiguation. "Nilesh Patil" is a very common name in India — there are hundreds of Nilesh Patils, including many in data science. The ORCID URL in `sameAs` is the strongest machine-readable signal Google and academic databases can use to uniquely identify this specific Nilesh Patil. Its absence from `sameAs` weakens the knowledge graph entry.

**Recommendation:** Add `"https://orcid.org/0000-0002-3438-8571"` to `social.links` in `_config.yml`.

**Fix snippet:**
```yaml
# _config.yml
social:
  links:
    - "https://github.com/nilesh-patil"
    - "https://x.com/optimistic_flw"
    - "https://scholar.google.co.in/citations?user=IIabY1sAAAAJ"
    - "https://orcid.org/0000-0002-3438-8571"   # ADD
    - "https://nilesh-patil.medium.com/"
```

**Spec reference:** https://schema.org/Person — `sameAs` with ORCID; see also https://orcid.org/organizations/integrators/API

---

```yaml
id: B-14
title: "Google Scholar profile blocked by bot protection — cannot verify name/affiliation/recency"
category: Brand
severity: P2
confidence: HIGH
effort: 1h
agents: [profile]
```

**Evidence:** WebFetch of both `https://scholar.google.co.in/citations?user=IIabY1sAAAAJ` and `https://scholar.google.com/citations?user=IIabY1sAAAAJ&hl=en` both returned 302 redirects to Google's CAPTCHA/sorry page. Content not accessible via automated fetch. NEEDS_USER_INPUT: Nilesh to manually verify that Scholar profile shows correct affiliation (Dream Sports is likely listed — DreamStreet may not be, as it started in 2026).

**Why this matters:** Google Scholar is listed in the sidebar and linked from CV and About pages. If the Scholar profile affiliation still shows Dream11/Dream Sports (a likely scenario given ORCID shows the same), researchers and press who look Nilesh up on Scholar see a stale employer. Scholar affiliation is a signal used by conference organizers and paper reviewers.

**Recommendation:** Log into Google Scholar and update affiliation to "DreamStreet" if not already done. Also confirm the profile photo on Scholar matches the site avatar.

**Fix snippet:** (off-site — scholar.google.com > My profile > Edit)
```
Affiliation: Head of AI, DreamStreet
```

**Spec reference:** https://schema.org/Person — `affiliation` consistency

---

```yaml
id: B-15
title: "Site canonical URL points to /cv/ on home page load — canonical bug, not brand bug, but affects sameAs resolution"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** JS eval on home page (`http://localhost:4000/`) returned `canonical: "http://localhost:4000/posts/distributed-kmeans-clustering/"`. When navigated to `/about/`, canonical returned `"http://localhost:4000/posts/"`. The Person JSON-LD `"url"` field would correctly resolve to the root, but the `<link rel="canonical">` tag on each page appears to be caching/stale from a prior page view, suggesting the page visited was actually a post or cv page that wasn't refreshed. This may be a dev-server artifact or a Jekyll permalink configuration issue.

**Why this matters:** If the canonical tag on the deployed home page points to a sub-page instead of `https://nilesh-patil.github.io/`, Google will index the wrong canonical for the homepage and may not associate the Person JSON-LD's `url` with the home page. This undermines the knowledge graph signal.

**Recommendation:** Verify canonical on the live deployed site at `https://nilesh-patil.github.io/` using a real browser or `curl`. If this reproduces in production, investigate `_includes/seo.html` canonical generation logic.

**Fix snippet:** (verification)
```bash
curl -s https://nilesh-patil.github.io/ | grep 'rel="canonical"'
```

**Spec reference:** https://developers.google.com/search/docs/crawling-indexing/canonicalization

---

```yaml
id: B-16
title: "Twitter/X missing from site footer — inconsistent social surface presence"
category: Brand
severity: P2
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** Footer links (JS eval): GitHub, Medium, LinkedIn, Google Scholar, Feed, Jekyll, AcademicPages, Minimal Mistakes, Sitemap. No Twitter/X link. Sidebar links include X (via `author.twitter`). `_config.yml:261-274` footer links section: GitHub, Medium, LinkedIn, Google Scholar — no Twitter/X.

**Why this matters:** Twitter/X is the primary professional social network for AI practitioners. If Nilesh has an active X presence (`@optimistic_flw` or `@ensembledme`), not linking it in the footer is a missed opportunity. Peers like Andrej Karpathy, Lilian Weng, and Simon Willison are primarily discovered via X posts linking to their sites. The sidebar link IS present but the footer omission is inconsistent.

**Recommendation:** Once the Twitter/X handle consolidation (B-03) is resolved, add X/Twitter to footer links in `_config.yml`. NEEDS_USER_INPUT: confirm active X handle first.

**Fix snippet:** (pending B-03 resolution)
```yaml
# _config.yml — footer section
footer:
  links:
    - label: "GitHub"
      icon: "fab fa-fw fa-github"
      url: "https://github.com/nilesh-patil"
    - label: "X / Twitter"
      icon: "fab fa-fw fa-x-twitter"
      url: "https://x.com/CORRECT_HANDLE"   # fill after B-03
    - label: "Medium"
      ...
```

**Spec reference:** n/a (brand consistency)

---

```yaml
id: B-17
title: "Site logo field points to ensembledme.jpg — avatar filename is the Twitter handle of an unrelated account"
category: Brand
severity: P2
confidence: MED
effort: 1h
agents: [profile]
```

**Evidence:** `_config.yml:14`: `logo: "/images/ensembledme.jpg"`. `_config.yml:48`: `avatar: "ensembledme.jpg"`. `_config.yml:26`: `og_image: "ensembledme.jpg"`. The filename `ensembledme` mirrors the `@ensembledme` handle — which on Medium belongs to a food/lifestyle blogger and on X/Twitter may not belong to Nilesh (B-08). The image file itself is presumably Nilesh's photo, but the filename creates implicit brand association with the `ensembledme` identity across all og:image, Twitter card image, and logo meta tags.

**Why this matters:** If `@ensembledme` is ultimately confirmed to NOT be Nilesh's primary handle, all cached og:image URLs (`/images/ensembledme.jpg`) will permanently carry a filename tied to an identity that does not represent him professionally. The URL appears in Twitter card images and Open Graph embeds cached by Slack, iMessage link previews, etc.

**Recommendation:** Rename the image file to `nilesh-patil.jpg` / `nilesh-patil.webp` after confirming the handle situation. Update all references in `_config.yml`.

**Fix snippet:** (after B-08 confirmed)
```bash
# Rename files
mv images/ensembledme.jpg images/nilesh-patil.jpg
mv images/ensembledme.webp images/nilesh-patil.webp  # if exists
# Update _config.yml references
# logo: "/images/nilesh-patil.jpg"
# avatar: "nilesh-patil.jpg"
# og_image: "nilesh-patil.jpg"
```

**Spec reference:** n/a (brand consistency)

---

```yaml
id: B-18
title: "Sidebar Twitter link points to twitter.com domain (deprecated) not x.com"
category: Brand
severity: P2
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** Sidebar rendered link (JS eval): `"https://twitter.com/ensembledme"`. X Corp has migrated all links to `x.com`. Twitter.com still redirects to x.com for most handles, but the rendered link explicitly says "X (formerly Twitter)" while pointing to the `twitter.com` domain — an inconsistency. The icon already uses `fab fa-fw fa-x-twitter` (correct). `_includes/author-profile.html:169`: `href="https://twitter.com/{{ author.twitter }}"` — hardcoded `twitter.com` host.

**Why this matters:** Minor UX/branding inconsistency but visible to anyone who hovers the link. The label says "X" but the domain says "twitter.com". Should use `https://x.com/` for new links.

**Recommendation:** Update `_includes/author-profile.html:169` to use `https://x.com/` instead of `https://twitter.com/`.

**Fix snippet:**
```html
<!-- _includes/author-profile.html line 169 — change: -->
<li><a href="https://twitter.com/{{ author.twitter }}">
<!-- to: -->
<li><a href="https://x.com/{{ author.twitter }}">
```

**Spec reference:** n/a

---

```yaml
id: B-19
title: "About page does not list X/Twitter in 'Get in touch' — inconsistent with sidebar"
category: Brand
severity: P2
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** `_pages/about.md:38-45` "Get in touch" section lists: GitHub, LinkedIn, Google Scholar, Medium, Stack Exchange (Stats). No X/Twitter link. The sidebar (via `author.twitter`) renders an X link. CV contact section (`_pages/cv.md:92-97`) also omits X/Twitter: GitHub, LinkedIn, Google Scholar, Medium only.

**Why this matters:** The sidebar and the explicit "Get in touch" section are authoritative contact lists. Their inconsistency means a reader scrolling the About page gets a different contact picture than a reader who notices the sidebar. Once the handle situation (B-03) is resolved, X/Twitter should be added to both the About and CV contact sections.

**Recommendation:** After resolving B-03, add X/Twitter to the contact lists in `_pages/about.md` and `_pages/cv.md`.

**Fix snippet:**
```markdown
# _pages/about.md — add after LinkedIn line
- **X / Twitter** — [@HANDLE](https://x.com/HANDLE)
```

**Spec reference:** n/a (brand consistency)

---

```yaml
id: B-20
title: "Stack Exchange Stats profile linked from About/CV but not from sidebar or footer"
category: Brand
severity: P2
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** `_pages/about.md:44`: `Stack Exchange (Stats)` link to `https://stats.stackexchange.com/users/36581/nilesh`. `_pages/cv.md` does not include Stack Exchange. The sidebar `_includes/author-profile.html:109-111` has a `{% if author.stackoverflow %}` block but it prepends `https://www.stackoverflow.com/` — the wrong host for a Stats/CrossValidated profile. `_data/authors.yml:16-18` notes this: "profile lives on stats.stackexchange.com, not stackoverflow.com." The stackoverflow field is intentionally omitted.

**Why this matters:** Stats/CrossValidated is a high-signal platform for data scientists and ML practitioners — it signals depth in statistical thinking. The link is manually added to About but hidden from everywhere else (no sidebar, no footer, no CV). It also appears with username "nilesh" not "Nilesh Patil" — discovery via search is harder.

**Recommendation:** Either add a `crossvalidated:` field to the author template and render it properly in `author-profile.html`, or accept that it lives only in the About text (current state). Also update CV to include it for completeness. Low priority.

**Fix snippet:** (optional — add to `_includes/author-profile.html` after stackoverflow block)
```liquid
{% if author.crossvalidated %}
  <li><a href="https://stats.stackexchange.com/users/{{ author.crossvalidated }}"><i class="fab fa-fw fa-stack-exchange icon-pad-right" aria-hidden="true"></i>Cross Validated</a></li>
{% endif %}
```

**Spec reference:** n/a

---

```yaml
id: B-21
title: "No Google site-verification meta tag — Google Search Console not connected"
category: SEO/Meta
severity: P2
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** `_includes/seo.html:101-103`: `{% if site.google_site_verification %}` block present but `_config.yml` has no `google_site_verification:` key. Same for `bing_site_verification`, `alexa_site_verification`, `yandex_site_verification`. No analytics or search-console integration anywhere in config.

**Why this matters:** Without Search Console, Nilesh cannot see how the site appears in Google SERPs, identify crawl errors, submit the sitemap, or verify that the Person JSON-LD is being parsed correctly. For a common name ("Nilesh Patil"), confirming the knowledge-graph card is attributed correctly requires Search Console access. The sitemap at `/sitemap/` exists (via `jekyll-sitemap` plugin) but is not submitted.

**Recommendation:** Register the site in Google Search Console, get the verification meta tag, add it to `_config.yml`, and submit the sitemap (`https://nilesh-patil.github.io/sitemap.xml`). This is a 20-minute setup.

**Fix snippet:**
```yaml
# _config.yml — after verifying in Search Console
google_site_verification: "PASTE_FROM_SEARCH_CONSOLE"
```

**Spec reference:** https://developers.google.com/search/docs/monitor-debug/search-console-start

---

```yaml
id: B-22
title: "avatar/logo displayed as Organization logo in JSON-LD — personal photo mistyped as Organization logo"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** `_includes/seo.html:134-141`: when `site.og_image` is set, emits:
```json
{"@type": "Organization", "url": "...", "logo": "/images/ensembledme.jpg"}
```
The `og_image` (`ensembledme.jpg`) is Nilesh's personal headshot — the same image used as `author.avatar`. A personal headshot is not an organization logo. Google's structured data guidelines specify that `Organization.logo` should be a logo mark, not a person photo.

**Why this matters:** Google's rich-results test will flag this as an invalid Organization logo (wrong aspect ratio, content type). It also creates a semantic conflict: the headshot is the Person's photo, declared simultaneously as an Organization's logo for the site URL. This corrupts the knowledge graph signal for both the person entity and any organizational association.

**Recommendation:** Remove the Organization JSON-LD (see B-07 for the consolidation rationale). If keeping it, replace `og_image` with an actual logo SVG/PNG and separate it from the avatar.

**Fix snippet:** (see B-07 fix snippet — same root issue)

**Spec reference:** https://developers.google.com/search/docs/appearance/structured-data/logo — logo must be a logo mark, not a person photo

---

```yaml
id: B-23
title: "No Twitter/X link in Person JSON-LD that matches the sidebar @ensembledme handle"
category: SEO/Meta
severity: P1
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** Person JSON-LD `sameAs` (JS eval): `["https://github.com/nilesh-patil", "https://x.com/optimistic_flw", "https://scholar.google.co.in/citations?user=IIabY1sAAAAJ", "https://twitter.com/nilesh-patil", "https://nilesh-patil.medium.com/"]`. The sidebar shows `https://twitter.com/ensembledme`. The `twitter:site` meta is `@ensembledme`. None of these three surfaces agree on the same Twitter/X URL.

Specifically: `sameAs` has `x.com/optimistic_flw` AND `twitter.com/nilesh-patil` but NOT `twitter.com/ensembledme`. The sidebar renders `twitter.com/ensembledme`. The Twitter card meta says `@ensembledme`. Three different identities are spread across three surfaces.

**Why this matters:** This is a concentrated version of B-03 expressed at the structured-data level. Google's Person entity will try to resolve all three URLs. If any of them resolve to different people (which is possible — `nilesh-patil` may be another person, `ensembledme` may be the food blogger), the entity gets polluted.

**Recommendation:** Resolve B-03 first. Then ensure `sameAs`, `twitter:site`, and `author.twitter` all reference exactly one URL.

**Fix snippet:** (see B-03 fix snippet)

**Spec reference:** https://schema.org/Person — `sameAs` integrity

---

```yaml
id: B-24
title: "CV PDF exists (104KB, May 19) but may lag web CV content — no staleness warning beyond inline note"
category: Brand
severity: P2
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** `_pages/cv.md:13`: `[PDF version of this CV]({{ '/files/nilesh-patil.pdf' | relative_url }})`. File confirmed present: `files/nilesh-patil.pdf` — 104KB, last modified May 19 2026. CV page includes notice: "The web version above is authoritative; the PDF may lag the web by a few weeks." Link is live and functional.

**Why this matters:** The PDF was last modified May 19 — 5 days before this audit. The DreamStreet role on the web CV started in 2026 (listed as 2026–Present). A recruiter downloading the PDF gets a snapshot that may or may not include the latest content. This is low severity given the caveat notice, but a dated PDF without a version number creates uncertainty.

**Recommendation:** Consider adding the build date to the CV PDF filename or a visible "Last updated: YYYY-MM-DD" line near the top of the PDF. Low priority given existing caveat.

**Fix snippet:** (optional — add to `_pages/cv.md`)
```markdown
**Download** — [PDF version of this CV]({{ '/files/nilesh-patil.pdf' | relative_url }}) *(last updated: May 2026)*
```

**Spec reference:** n/a (brand/UX)

---

```yaml
id: B-25
title: "NEEDS_USER_INPUT consolidation: 5 brand decisions require Nilesh's direct input before fixes can proceed"
category: Brand
severity: P0
confidence: HIGH
effort: 15m
agents: [profile]
```

**Evidence:** The following findings are blocked on user input:
1. **B-03/B-08/B-09**: Which Twitter/X handle is primary — `@ensembledme`, `@optimistic_flw`, or `@nilesh-patil`? Does Nilesh own `@ensembledme` on X (as opposed to the food blogger on Medium)?
2. **B-04**: Should ORCID be surfaced on the site? (It's already public via GitHub.)
3. **B-10**: Is Nilesh willing to add a contact email or contact form?
4. **B-11/B-14**: Has LinkedIn/Scholar affiliation been updated to DreamStreet manually?
5. **B-12**: Is Nilesh planning Medium content to reactivate that surface?

**Why this matters:** B-03/B-08/B-09 affect live structured data, the Twitter Card attribution on every page share, and the sidebar link. Without knowing which handle is correct, no on-site fix should be deployed — the wrong handle could be worse than three inconsistent handles.

**Recommendation:** Collect these 5 decisions from Nilesh before proceeding with any Twitter/X-related fixes.

**Fix snippet:** (questionnaire for Nilesh)
```
Q1: On X/Twitter, which account is yours and active for professional use?
    (a) @ensembledme  (b) @optimistic_flw  (c) @nilesh-patil  (d) none/not on X

Q2: Do you want to surface your ORCID (0000-0002-3438-8571) on the site sidebar?

Q3: Do you want to add a contact email (professional or personal) to the site?

Q4: Have you updated your LinkedIn headline to "Head of AI at DreamStreet"?

Q5: Have you updated your Google Scholar affiliation to DreamStreet?

Q6: Are you planning to publish new content on Medium?
```

**Spec reference:** n/a — process finding

---

## Summary table

| ID | Title | Severity | Effort | Blocked? |
|----|-------|----------|--------|----------|
| B-01 | GitHub bio stale (Dream11 not DreamStreet) | P0 | 15m | No |
| B-02 | medium.com/@ensembledme = food blog | P0 | 15m | No (on-site fix done; off-site risk remains) |
| B-03 | Three Twitter/X handles across site surfaces | P1 | 1h | NEEDS_USER_INPUT |
| B-04 | ORCID not on site; ORCID employment stale | P1 | 1h | NEEDS_USER_INPUT |
| B-05 | Person JSON-LD url = localhost in dev | P0 | 15m | No (verify in prod) |
| B-06 | meta description ≠ og:description | P1 | 15m | No |
| B-07 | Person + Organization JSON-LD on same URL | P1 | 15m | No |
| B-08 | @ensembledme Twitter sidebar link — ownership unverified | P1 | 15m | NEEDS_USER_INPUT |
| B-09 | twitter.com/nilesh-patil in sameAs — unverified | P1 | 15m | NEEDS_USER_INPUT |
| B-10 | No contact email — recruiter dead-end | P1 | 1h | NEEDS_USER_INPUT |
| B-11 | LinkedIn headline not verified | P2 | 15m | NEEDS_USER_INPUT |
| B-12 | Medium bio 5+ years stale | P1 | 15m | NEEDS_USER_INPUT |
| B-13 | ORCID missing from sameAs | P1 | 15m | No |
| B-14 | Google Scholar affiliation unverified | P2 | 1h | NEEDS_USER_INPUT |
| B-15 | Canonical URL stale in dev (verify in prod) | P1 | 15m | No (verify in prod) |
| B-16 | X/Twitter absent from footer | P2 | 15m | Blocked on B-03 |
| B-17 | Avatar filename = ensembledme (brand risk) | P2 | 1h | Blocked on B-08 |
| B-18 | Sidebar X link uses twitter.com not x.com | P2 | 15m | No |
| B-19 | About/CV contact sections omit X/Twitter | P2 | 15m | Blocked on B-03 |
| B-20 | Stack Exchange Stats not in sidebar/footer | P2 | 15m | No |
| B-21 | No Search Console verification | P2 | 15m | No |
| B-22 | Personal headshot declared as Organization logo | P1 | 15m | No |
| B-23 | sameAs, twitter:site, sidebar handle all differ | P1 | 15m | Blocked on B-03 |
| B-24 | CV PDF exists but may lag web CV; no version date | P2 | 15m | No |
| B-25 | Consolidation of NEEDS_USER_INPUT items | P0 | 15m | Requires Nilesh |
