# Hiring-Lens Audit
**Persona:** Senior technical recruiter / hiring manager / VC talent partner, 90-second cold visit.

## [Severity: P0] — "Follow" button is a dead UI affordance with no action
**Lens:** Inbound recruiter
**Location:** `_includes/author-profile.html` line 39
**Evidence:** Sidebar renders prominent "Follow" pill button. On mobile it's the primary interactive element above the fold. No follow mechanism exists — no newsletter, no RSS subscribe, no LinkedIn deep-link.
**Recommendation:** Rename to "Connect" or "Contact" to match what clicking reveals. When LinkedIn is added, wire button directly to the profile URL.
**Reference:** https://huyenchip.com — sidebar uses clean icon-links with no false-affordance buttons.

## [Severity: P0] — LinkedIn is the recruiter's primary tool; there is no path to it
**Lens:** Inbound recruiter
**Location:** All pages
**Evidence:** LinkedIn absent from sidebar/footer/about. CV mentions LinkedIn but renders dead plain text: *"(see GitHub profile for current link)"* — multi-hop dead end. About page doesn't mention LinkedIn at all.
**Recommendation:** Add `linkedin: <handle>` to `_config.yml` and `_data/authors.yml`. Template already has the conditional. If handle is genuinely withheld, replace CV entry with explicit: "Not on LinkedIn — contact via GitHub DM."
**Reference:** https://lilianweng.github.io

## [Severity: P0] — Blog freshness gap actively signals disengagement
**Lens:** Inbound recruiter
**Location:** `/`, `/posts/`
**Evidence:** "Recent posts" most recent is May 2020. Remaining five are 2017. Six-year silence at a time when the CV claims senior leadership, Columbia collaboration, ACL 2026 paper.
**Recommendation:** Option A: Publish one post from Dream11/DreamStreet era — 600-word "lessons from building LLM-based churn prediction at 250M users" closes the gap. Option B: Replace "Recent posts" with a "Recent work" section that pulls from publications.
**Reference:** https://huyenchip.com — mixes posts and papers in a single feed.

## [Severity: P1] — First-8-second test: WHO clear, WHAT thin, WHY absent
**Lens:** Inbound recruiter
**Location:** `/` above the fold
**Evidence:** "Head of AI at DreamStreet, building compliance-aware AI architecture for SEBI-regulated investor and trader workflows" is fintech-insider language. A non-fintech recruiter does not immediately translate this to impact. No scale/headcount/shipped-product anchor.
**Recommendation:** Rewrite using WHO + scale + angle: *"Head of AI at DreamStreet. Built production AI systems for 250M+ users at Dream11. Particularly interested in making LLMs reliable in regulated, high-stakes environments."*
**Reference:** https://www.lennysnewsletter.com/about

## [Severity: P1] — Team size is never stated; "Head of AI" title is unsubstantiated
**Lens:** Hiring manager / VC talent partner
**Location:** `/about/`, `/cv/`
**Evidence:** CV: "Led a high-performing cross-continent team... across India and New York." Team size never given. No hiring decisions appear anywhere.
**Recommendation:** Add one number to the Dream11 entry: "Led a cross-continent team of N researchers and ML engineers."

## [Severity: P1] — No schema.org/Person markup; site invisible to knowledge-graph differentiation for a common name
**Lens:** Talent partner Googling "Nilesh Patil AI"
**Location:** `_includes/seo.html`, `_config.yml`
**Evidence:** `seo.html` renders Person JSON-LD only if `site.social` is configured. `_config.yml` has no `social:` key. Without `Person` JSON-LD with `sameAs` links, Google cannot associate this page with the knowledge-graph node for "Nilesh Patil."
**Recommendation:** Add `social:` block to `_config.yml` with `type: Person`, `name`, and `links:` array. Also add `og_image: "ensembledme.jpg"`.
**Reference:** https://lilianweng.github.io — has Person schema and OG image; appears with knowledge panel.

## [Severity: P1] — CV PDF download has trust-eroding disclaimer and is buried
**Lens:** Hiring manager (forwards PDF to decision-maker)
**Location:** `/cv/` Download notice
**Evidence:** *"PDF version of this CV — The web version above is authoritative; the PDF may lag the web by a few weeks."* No last-modified date. "May lag by weeks" is trust-eroding for someone forwarding a CV.
**Recommendation:** Replace disclaimer with specific date: *"PDF last updated 2026-05-XX."* Add "Download CV (PDF)" link to sidebar.
**Reference:** https://huyenchip.com/files/huyen-chip-cv.pdf

## [Severity: P1] — Portfolio is two items (one from 2018); reads as a graveyard
**Lens:** Hiring manager
**Location:** `/portfolio/`
**Evidence:** Two items: "Data Science Docker Environment" (July 2018, a Dockerfile setup) and "Python vs Rust: k-means at scale" (August 2024). Neither substantiates Head-of-AI work.
**Recommendation:** Add 2–3 case-study entries. No open-source code required — architecture writeups work: "Compliance-aware AI harness at DreamStreet," "LLM-based persona simulator," "Columbia collaboration model."
**Reference:** https://eugeneyan.com/work/

## [Severity: P1] — "Currently exploring" section duplicates About intro verbatim
**Lens:** Inbound recruiter
**Location:** `/about/`
**Recommendation:** Replace with "How to work with me" or "Open to" — *"Open to advisory conversations on AI architecture in regulated fintech/insurance domains, and talks at AI engineering conferences."* Converts dead repetition into active inbound funnel.
**Reference:** https://www.shreya-shankar.com/about/

## [Severity: P1] — Talks page is empty; referenced in CV but renders as a shell
**Lens:** Hiring manager (assessing thought leadership)
**Location:** `/talks/`
**Evidence:** CV mentions Columbia "Sports x AI" sessions and training sessions ranging 10–200 participants. `/talks/` URL builds to a blank page with only the heading. `show_talks: false` hides nav but the page is still accessible.
**Recommendation:** Either populate `_talks` and flip the flag, or add `redirect_to: /cv/` so the empty URL doesn't exist.
**Reference:** https://lilianweng.github.io/talks/

## [Severity: P2] — "Nilesh Patil" disambiguation relies entirely on headshot; title tags generic
**Lens:** Talent partner search
**Location:** Page titles
**Evidence:** Title is "Nilesh Patil" (home), "About - Nilesh Patil" (about). Subtitle "AI systems & applied research" doesn't appear in any page title. "Nilesh Patil" is a common name.
**Recommendation:** Add specific `description` to home frontmatter.

## [Severity: P2] — Footer attribution signals "unmodified template" to senior reviewers
**Lens:** Hiring manager (care/attention assessment)
**Location:** Footer
**Recommendation:** Edit `_includes/footer.html` to: *"© 2026 Nilesh Patil · Built with Jekyll"* — drop AcademicPages reference. 2-minute edit.
**Reference:** https://eugeneyan.com

## [Severity: P2] — Site does not help the owner recruit for their own AI team (reverse lens)
**Lens:** Reverse (owner-hires-talent)
**Location:** Site-wide
**Evidence:** No "We're hiring" or "AI team" content. "DreamStreet" employer field in sidebar is plain text, not a link.
**Recommendation:** One paragraph on About: *"I'm building a focused AI team at DreamStreet working on [problem]. If you're interested in [role type], reach out via [contact]."*
**Reference:** https://lenny.com/about

## The 5 Changes That Most Increase Inbound-Opportunity Rate
| # | Change | Effort | Impact |
|---|--------|--------|--------|
| 1 | Add LinkedIn to sidebar | 1h | Unblocks primary recruiter inbound channel |
| 2 | Configure `social:` block + `og_image` | 1h | Fixes search differentiation + link-preview cards |
| 3 | Publish one post from the 2020–2026 gap | 1d | Closes freshness gap |
| 4 | Rewrite home hero with WHO + scale + angle | 1h | Maximizes 8-second first impression |
| 5 | Add team size + "Open to" sentence | 1h | Answers the two recruiter questions silently asked after the bio |
