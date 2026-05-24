# Hiring Manager Audit — nilesh-patil.github.io

**Auditor role:** Senior tech recruiter sourcing for Head-of-AI / VP-AI / Director-AI  
**Audit date:** 2026-05-24  
**Site:** http://localhost:4000/ (live) / https://nilesh-patil.github.io (production)

---

## Hiring Verdict

**Conditional forward — requires follow-up to fill in two critical gaps before advancing.**

On a 10-minute walk-through, the candidate clears the basic bar: current title is visible immediately, scale is credible (250M+ users, Columbia collaboration, multi-million-dollar research center), and at least three peer-reviewed venues post-2022 exist. The technical trajectory — IIT Roorkee → U Rochester M.S. → applied ML → 7-year research leadership → AI systems in regulated fintech — is coherent and senior-looking.

What kills the advance: the site provides zero evidence of *organizational leadership at the VP+ tier*. There is no team headcount, no mention of budget ownership, no reported promotion chain, no shipped product users can point to by name, and no post-2020 writing demonstrating current thought leadership in AI. A competing candidate with a public essay on agentic system design from 2025 will be forwarded first. The blog cadence alone — 5 posts in 2017, 1 in 2020, zero for six years — reads as "this person stopped publicly engaging with the field" to any hiring manager who notices. The GitHub bio still says "Sr. Principal Research Scientist @dream11", actively contradicting the "Head of AI at DreamStreet" claim on the site. That single inconsistency triggers a verification instinct that slows the shortlist decision.

---

## Findings

---

```yaml
---
id: H-01
title: "GitHub bio is stale: contradicts site title and current employer"
category: Brand
severity: P0
confidence: HIGH
effort: 15m
agents: [hiring]
---
```

**Evidence:** The site prominently claims "Head of AI at DreamStreet" in the sidebar bio, config, and about page. The live GitHub profile at https://github.com/nilesh-patil shows bio text: "Sr. Principal Research Scientist @dream11" — the previous title at the previous employer. A recruiter who Google-searches the name and lands on GitHub before the personal site sees a mismatched story.

**Why this matters (recruiter view):** Inconsistency between surfaces is the #1 trust-eroding signal in recruiting. A hiring manager who sees "Head of AI" on the site but "Sr. Principal Research Scientist" on GitHub either (a) concludes the self-reported title is inflated, or (b) concludes the candidate is sloppy about personal branding — neither is good. For VP-level roles, seniority calibration happens early and a downgrade sticks.

**Recommendation:** Update the GitHub profile bio to: "Head of AI at DreamStreet. Previously Head of Applied Research, Dream11." and set the company field to "DreamStreet".

**Fix snippet:**
```
GitHub profile bio (edit at github.com/settings/profile):
  Name: Nilesh Patil
  Bio: Head of AI at DreamStreet. Previously Head of Applied Research, Dream11.
  Company: DreamStreet
  Location: Mumbai, India
```

**Peer reference:** https://github.com/karpathy — bio, employer, and personal site title are consistent and current.

---

```yaml
---
id: H-02
title: "Zero posts from 2021–2026: blog cadence signals disengagement from AI field"
category: Content
severity: P0
confidence: HIGH
effort: 4h
agents: [hiring]
---
```

**Evidence:** The blog has 6 posts total: 5 from January–July 2017, 1 from May 2020. The home page "Recent posts" list shows the 2020 post first, then four posts from 2017. A recruiter scanning the dates sees a 6-year gap covering the entire LLM era (GPT-3, ChatGPT, GPT-4, Claude, Llama, agentic systems). The most recent post is about distributed k-means — a topic from 2017-era ML, not 2024-era AI.

**Why this matters (recruiter view):** Blogs are optional, but when they exist and are stale, they actively harm the candidacy. The recruiter's mental model: "This person stopped writing about AI just as the field got interesting. Do they have a perspective on LLMs? On agents? On compliance-aware AI systems?" The CV *claims* the candidate is building agentic workflows at DreamStreet right now — but the blog provides zero evidence of engagement with those topics. Lilian Weng posts 4–6 substantive essays per year while running safety at OpenAI. Chip Huyen published a book in 2025. Simon Willison posts daily. A candidate for Head-of-AI who hasn't written publicly since 2020 will lose to a peer who has.

**Recommendation:** Publish at minimum two posts before any active job search: one on a concrete system built (e.g., the compliance-aware AI harness at DreamStreet — even a sanitized architecture overview), and one opinion piece on a current AI topic (agentic evaluation, SLM self-hosting tradeoffs, regulated AI design). Two posts from 2025–2026 reverse the perception entirely.

**Fix snippet:**
```markdown
Suggested post titles:
1. "Building a Compliance-Aware AI Harness for SEBI-Regulated Workflows" (architecture, not product secrets)
2. "Self-Hosting SLMs in Production: What GCP vs. Local vs. AWS Actually Costs You"
3. "Agentic Evaluators at Scale: How We Simulate 250M User Personas"
```

**Peer reference:** https://lilianweng.github.io/posts/2025-05-01-thinking/ — a substantive 2025 post demonstrating current thought leadership while in a senior role.

---

```yaml
---
id: H-03
title: "No team size, headcount, or org-chart signal anywhere on site"
category: Content
severity: P0
confidence: HIGH
effort: 1h
agents: [hiring]
---
```

**Evidence:** The CV and About page never state how many people Nilesh directly managed or had organizational accountability for. The Dream11 section says "led a high-performing cross-continent team of research scientists, applied scientists, and ML engineers across India and New York" but gives no number. DreamStreet section says "Drove AI adoption org-wide" but not what "org" means (10 people? 500?). There is no mention of a reporting structure, skip-level relationships, or span of control.

**Why this matters (recruiter view):** "Head of AI" and "VP of AI" roles have wildly different organizational expectations. A Head of AI at a 15-person startup (no direct reports) and a Head of AI at a 500-person company (managing 20 engineers) are different jobs. Without a headcount signal, the recruiter defaults to assuming minimum: a strong individual contributor who has a leadership title. Any competing candidate who writes "managed a team of 12 research scientists and 6 ML engineers" immediately calibrates as more senior.

**Recommendation:** Add team size for each leadership role. On CV under Dream11: "Led a cross-continent team of [N] research scientists, applied scientists, and ML engineers." On CV under DreamStreet: "Building and leading [N]-person AI team." If numbers are confidential, use ranges: "5–10 person research org" or "team of 8+".

**Fix snippet:**
```markdown
### Head of AI — DreamStreet — Mumbai — 2026 — Present
- Built and lead a [N]-person AI team covering architecture, agentic systems, and org-wide AI adoption.
...
### Senior Principal Research Scientist / Head of Applied Research — Dream11 — 2019 — 2026
- Led a cross-continent team of [12] research scientists, applied scientists, and ML engineers across Mumbai and New York.
```

**Peer reference:** https://huyenchip.com — bio explicitly references roles at NVIDIA, Netflix, and startup founding, each with implied org scope. LinkedIn-linked job descriptions provide headcount context.

---

```yaml
---
id: H-04
title: "No budget, P&L, or resource-accountability signal for a C-suite-adjacent role"
category: Content
severity: P1
confidence: HIGH
effort: 1h
agents: [hiring]
---
```

**Evidence:** The single financial scale signal is "multi-million-dollar research center" (Columbia collaboration). There is no mention of research budget managed, cloud/infrastructure spend, vendor contracts, or any resource allocation accountability at DreamStreet or Dream11. No mention of business impact (cost savings, revenue influence, user metrics tied to AI systems shipped).

**Why this matters (recruiter view):** VP-AI and Head-of-AI roles at funded companies routinely come with $2M–$15M annual AI budgets, cloud contracts, and board-level reporting. A recruiter screening for those roles looks for evidence that the candidate has operated at that accountability level. "Built X system" is engineer-speak. "Built X system that reduced compliance review time by 40%, eliminating 2 FTE contractor roles" is executive-speak. The site has the former, not the latter.

**Recommendation:** Add one outcome sentence per role that ties a shipped system to a business metric. Can be directional ("significantly reduced", "cut by half", "enabled X new product capability") if exact numbers are confidential.

**Fix snippet:**
```markdown
- Delivered churn prediction system that identified at-risk users [X days] earlier than prior models, enabling targeted retention campaigns across 250M+ user base.
- Led real-time forecasting infrastructure that processes ~50k+ forecasts per [timeframe] under strict latency SLAs, directly powering [product name] real-money game features.
```

**Peer reference:** https://huyenchip.com/about/ — lists "founded and sold an AI infrastructure startup" as a single-line proof of P&L accountability.

---

```yaml
---
id: H-05
title: "Portfolio has zero AI/LLM systems: none of the 3 projects backs the 'Head of AI' claim"
category: Content
severity: P1
confidence: HIGH
effort: 4h
agents: [hiring]
---
```

**Evidence:** The three portfolio (Side projects) entries are: (1) a 2018 Docker environment for data science, (2) a 2024 Python-vs-Rust k-means benchmark, and (3) a 2026 HPC fork for biology simulations (SimuCell3D). None of the three involves LLMs, agentic systems, RAG, fine-tuning, vector search, or any technology the candidate claims to be leading at work. The most recent AI-relevant work is described entirely in CV text — there is no public artifact.

**Why this matters (recruiter view):** Portfolio is where I look for "show, don't tell." The CV claims the candidate is building "LLM-based behavior simulation, persona simulators, agentic evaluators, and compliance-aware AI harness design." The Side Projects page shows none of this. A recruiter who bounces from CV to portfolio and finds no corroboration concludes the work either (a) cannot be shown publicly or (b) does not exist in a form the candidate is comfortable defending. A competitor with even one public LLM project — a GitHub repo, a Hugging Face demo, a sanitized architecture post — instantly wins this comparison.

**Recommendation:** Add at least one portfolio entry covering public AI work: an open-source agent tool, a published LLM evaluation harness, a Hugging Face model card, a sanitized system design write-up. Even the ACL 2026 entity resolution paper (already in Publications) could be cross-linked here as a "portfolio" item with a technical explainer.

**Fix snippet:**
```markdown
---
title: "LLM Entity Resolution: Fine-Tuning for Name Matching at Scale"
collection: portfolio
date: 2026-01-01
excerpt: "Structure-guided fine-tuning approach for LLM-based entity resolution..."
---
This work (published at ACL 2026) addresses name-matching under noisy, linguistically diverse inputs...
[Links to paper, GitHub repo if public, architecture overview]
```

**Peer reference:** https://github.com/simonwillison — every repo is a runnable, documented public artifact that directly evidences stated expertise.

---

```yaml
---
id: H-06
title: "No recruiter CTA: no 'open to conversations', email, or Calendly on any page"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [hiring]
---
```

**Evidence:** The About page ends with a "Get in touch" section listing GitHub, LinkedIn, Google Scholar, Medium, and Stack Exchange. The CV ends with a "Contact" section listing the same. No page has an email address, a contact form, a Calendly link, or any sentence signaling openness to being approached for roles or advisory conversations. The config.yml comment explicitly says "email omitted intentionally."

**Why this matters (recruiter view):** Recruiters work on tight timelines. The path LinkedIn → site → "how do I reach this person quickly?" must have an answer in under 10 seconds. GitHub DMs are uncommon for senior recruiting. LinkedIn InMail works but the site should confirm the LinkedIn handle is correct (it is: ensembledme). The absence of any direct contact option adds one more step and one more opportunity for the recruiter to move to the next candidate. A single sentence — "Open to conversations about AI leadership roles — reach me on LinkedIn" — changes this entirely.

**Recommendation:** Add to the About page "Get in touch" section: "I'm selectively open to conversations about AI leadership roles, advisory engagements, and research collaborations. LinkedIn is the fastest path: linkedin.com/in/ensembledme." Alternatively, add a lightweight contact form or a public email alias.

**Fix snippet:**
```markdown
## Get in touch

I'm selectively open to conversations about AI leadership roles, advisory engagements, and research collaborations.

- **LinkedIn** — [linkedin.com/in/ensembledme](https://www.linkedin.com/in/ensembledme) *(fastest response)*
- **GitHub** — [@nilesh-patil](https://github.com/nilesh-patil)
- **Google Scholar** — [profile](https://scholar.google.co.in/citations?user=IIabY1sAAAAJ)
```

**Peer reference:** https://huyenchip.com — "Reach out if you want to find a way to work together" is a direct, warm CTA on the homepage.

---

```yaml
---
id: H-07
title: "DreamStreet role start date '2026 — Present' with no context of company stage"
category: Content
severity: P1
confidence: HIGH
effort: 30m
agents: [hiring]
---
```

**Evidence:** The CV lists "Head of AI — DreamStreet — Mumbai — 2026 — Present" as the current role. DreamStreet appears nowhere on the web as a well-known brand (unlike Dream11, which is a household name in India). The site gives no context: Is this a startup? A Series B company? A spin-off? How many employees? What is the product?

**Why this matters (recruiter view):** For candidates at senior roles, context about employer stage matters enormously. "Head of AI at a 20-person fintech seed startup" and "Head of AI at a regulated brokerage platform with 200+ employees" are different levels of accountability. Without any context, the recruiter must Google DreamStreet, which may return nothing or return ambiguous results. A competing candidate at a recognizable company (Zerodha, Groww, Upstox) automatically looks more established. One sentence of company context in the CV would close this gap.

**Recommendation:** Add a parenthetical after the company name: "DreamStreet (SEBI-regulated investment platform, [N] employees / Series [X])" or embed context in the first bullet: "Built full-stack AI systems for DreamStreet, a [stage] fintech platform serving Indian retail investors and traders."

**Fix snippet:**
```markdown
### Head of AI — DreamStreet *(SEBI-regulated retail investment platform)* — Mumbai — 2026 — Present
```

**Peer reference:** https://www.linkedin.com/in/chiphuyen — each company entry includes a one-line company descriptor in the LinkedIn summary, providing instant calibration.

---

```yaml
---
id: H-08
title: "Publications page has no citation counts — weakest signal for a research leadership claim"
category: Content
severity: P1
confidence: HIGH
effort: 1h
agents: [hiring]
---
```

**Evidence:** The Publications page lists 6 papers with venue names and recommended citations. No citation count appears anywhere on the page or on individual publication pages. Google Scholar is linked but the site itself shows no h-index, total citations, or per-paper citation count. The CV mentions "6+ additional team publications" without listing them, and does not show citation metrics.

**Why this matters (recruiter view):** For a candidate presenting as a research leader (Columbia collaboration, ACL paper, IEEE papers), citation metrics are the standard academic credibility signal. A recruiter sourcing for AI leadership at a research-forward company will check Scholar. Not embedding the metrics on-site means the recruiter must click away — and may not return. More critically: if Scholar shows strong numbers, that is a trust accelerator that is being left off the table.

**Recommendation:** Add a summary box to the Publications page showing total citations, h-index, and i10-index (pulled from Google Scholar, updated manually twice a year). Alternatively, embed a Google Scholar badge. For individual papers, add "Cited by N" next to the venue line.

**Fix snippet:**
```markdown
## Citation metrics *(as of May 2026)*

| Metric | Value |
|--------|-------|
| Total citations | [N] |
| h-index | [N] |
| i10-index | [N] |

*Source: [Google Scholar profile](https://scholar.google.co.in/citations?user=IIabY1sAAAAJ)*
```

**Peer reference:** https://lilianweng.github.io — the about/bio section links to Google Scholar and her publications page on OpenReview. Hugging Face profile of research leaders typically shows citation count inline.

---

```yaml
---
id: H-09
title: "About page 'Currently exploring' section duplicates the role description verbatim"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [hiring]
---
```

**Evidence:** The About page has a "Currently" section (current role description) and a "Currently exploring" section. The "Currently exploring" text reads: "AI harness design, developer productivity, and turning emerging model capabilities into reliable workflows and products. Self-hosting SLMs and building agent tooling that survives contact with regulated production environments." The "Currently" role description already says "compliance-aware AI architecture" and "agentic workflows and user-facing copilots." The two sections cover nearly identical territory with slightly different phrasing.

**Why this matters (recruiter view):** Duplication reads as filler. It suggests the candidate ran out of substance to add. A recruiter speed-reading the About page skips the second "Currently" block because it adds no new signal. The slot would be better used for: specific technical problems being solved right now, books/papers being read, a community being contributed to, or an open question being investigated.

**Recommendation:** Replace "Currently exploring" with a more specific and differentiated list — concrete technical problems, open questions, or external projects. Examples: specific SLM models being evaluated, an open-source contribution in progress, a conference being prepared for.

**Fix snippet:**
```markdown
## Currently exploring

- Evaluating open-weight SLMs (Phi-3, Qwen2, Mistral) for regulated on-premise deployments under strict data residency constraints.
- Designing audit-trail architectures for agentic AI systems — how to make multi-step agent decisions reproducible and explainable to compliance teams.
- Contributing to [open-source project or community, if any].
```

**Peer reference:** https://simonwillison.net — "Things I'm working on" section on the about page lists live, specific, dated projects.

---

```yaml
---
id: H-10
title: "Home page 'Recent posts' prominently shows 2017 content before 2024/2026 portfolio"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [hiring]
---
```

**Evidence:** The home page layout shows "Recent posts" first, then "Side projects." The "Recent posts" list sorts by post date descending, showing: May 2020, then five posts from 2017. The 2024 portfolio entry (Python vs Rust k-means) and the 2026 entry (SimuCell3D) appear below the fold in "Side projects." A recruiter's first scrolled view therefore shows a list of 2017 data science tutorials as the most recent public intellectual output.

**Why this matters (recruiter view):** The home page is the first 10-second impression. Seeing "January 2017, February 2017, March 2017, March 2017, July 2017, May 2020" as the activity timeline signals "this person was active when I was a grad student and has since gone dark." Reversing the section order (Side projects first, then posts) would surface the 2026 SimuCell3D entry and the 2024 k-means benchmark as the most recent work — a materially better impression.

**Recommendation:** Swap the section order on home.md: show "Side projects" (with 2026 and 2024 entries) before "Recent posts" (which starts at 2020). Or add a "Recent activity" section that interleaves publications, portfolio entries, and posts sorted by date — surfacing the 2026 ACL paper and the 2026 SimuCell3D entry at the top.

**Fix snippet:**
```markdown
## Recent activity

- **[2026]** Published: *Structure-Guided Entity Resolution* at ACL 2026 — [link]
- **[2026]** Project: SimuCell3D HPC fork — parallel thread efficiency 29% → 60%
- **[2024]** Project: Python vs Rust k-means benchmark
- **[2020]** Post: Distributed K-Means Clustering in Python
```

**Peer reference:** https://simonwillison.net — home page shows most-recent items across all content types (posts, links, projects) sorted by date, always surfacing the freshest signal.

---

```yaml
---
id: H-11
title: "No named shipped product: 'copilots' and 'workflows' are unbacked by a product name or screenshot"
category: Content
severity: P1
confidence: MED
effort: 1h
agents: [hiring]
---
```

**Evidence:** The About page says "user-facing copilots" and "agentic workflows." The CV says "Built Hermes-based in-house automation agents across Marketing, Finance, IT, Tech, and HR." No product has a name a recruiter can Google. No screenshot, demo, or case study is linked. The Hermes mention is the closest to a specific artifact, but is opaque without context (Hermes is an internal codename, apparently).

**Why this matters (recruiter view):** "I built AI copilots" without a product name is a claim any candidate can make. The recruiter's mental model: "Show me something I can point to in a committee meeting when I'm advocating for this candidate." A single public-facing artifact — even a Medium post describing the architecture (without IP exposure) — is worth 10 CV bullet points.

**Recommendation:** Add a brief case study or system design sketch to the portfolio section covering one shipped system at DreamStreet (e.g., the compliance-aware AI harness). Frame it as an architecture discussion rather than product disclosure. Alternatively, name the product in the CV if it is public-facing and searchable.

**Fix snippet:**
```markdown
## Portfolio addition: "Hermes: Internal AI Automation at DreamStreet"
Problem: [generic framing]
Architecture: [sanitized stack — which LLM provider, orchestration layer, tool integrations]
Outcome: [org-level adoption metric — teams onboarded, workflows automated, time saved]
```

**Peer reference:** https://huyenchip.com/2024/... — publishes system design essays about real production AI systems without exposing proprietary details.

---

```yaml
---
id: H-12
title: "CV PDF exists but no indication of when it was last updated"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [hiring]
---
```

**Evidence:** The CV page says "Download — PDF version of this CV" and notes "the PDF may lag the web by a few weeks." The PDF exists at `/files/nilesh-patil.pdf` (HTTP 200). There is no date stamp on the download link or adjacent text indicating when the PDF was last refreshed.

**Why this matters (recruiter view):** Recruiters routinely forward a PDF CV to hiring committees. If the PDF is months behind the web version, the committee sees a stale document. The disclaimer "may lag the web by a few weeks" is a yellow flag — it suggests the PDF is an afterthought. A date stamp ("Last updated: May 2026") on the download link removes the ambiguity and signals professionalism.

**Recommendation:** Add the last-updated date adjacent to the PDF download link: "PDF version — last updated May 2026." Update the PDF whenever the web CV changes.

**Fix snippet:**
```markdown
**Download** — [PDF version of this CV]({{ '/files/nilesh-patil.pdf' | relative_url }}) *(last updated May 2026)*
{: .notice--info}
```

**Peer reference:** https://www.cs.cornell.edu/~kilian/papers/KilianWeinberger_CV.pdf — academic CVs typically have a date in the footer on every page.

---

```yaml
---
id: H-13
title: "Talks and Teaching sections hidden behind feature flags — no speaking record visible"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [hiring]
---
```

**Evidence:** `_config.yml` has `show_talks: false` and `show_teaching: false`. The CV has conditional blocks `{% if site.show_talks %}` and `{% if site.show_teaching %}` that render nothing. The About page mentions "Co-led *Sports x AI* sessions at Columbia University" and training sessions for "~10 to 200 participants" but no dedicated page backs this up, and the nav has no Talks link.

**Why this matters (recruiter view):** Speaking experience is a leading indicator of executive presence — the ability to represent the company at conferences, attract talent, and influence industry. A candidate for Head-of-AI who has led sessions at Columbia and trained 200 participants is a meaningfully stronger profile than the site currently signals. Hiding this content is a self-inflicted wound.

**Recommendation:** Either (a) set `show_talks: true` and populate the Talks collection with even 2–3 notable speaking engagements (Columbia, internal training programs), or (b) add a "Speaking & training" subsection directly on the About page listing the most notable engagements with dates and audience size.

**Fix snippet:**
```markdown
## Speaking & training

- **Sports x AI**, Columbia University — co-led [N] sessions for students, post-docs, and faculty on industry AI problems (2020–2025)
- **Dream Sports AI Training** — delivered [N] training sessions on data, ML, and AI for audiences of 10–200 across the organization (2019–2026)
- **[Conference or meetup, if any]**
```

**Peer reference:** https://huyenchip.com/speaking/ — dedicated speaking page with conference names, dates, and talk titles that establishes thought leadership presence.

---

```yaml
---
id: H-14
title: "The 2013–2014 gap between B.Tech and first listed job is unexplained"
category: Content
severity: P2
confidence: MED
effort: 15m
agents: [hiring]
---
```

**Evidence:** The CV lists B.Tech IIT Roorkee 2013, then "Data Analyst — AbsolutData Research & Analytics — Gurgaon — 2013 — 2014." The graduation-to-job timeline appears continuous. However, the AXA Insurance role is listed as "2014 — 2016," and the Rochester M.S. as 2017. This implies the M.S. was started in 2015 or 2016 — not stated explicitly. The "Rochester era" publications section covers "2016–2019," suggesting the M.S. started in 2015–2016. The ambiguity creates a mild gap question: "when exactly did the M.S. start, and was AbsolutData (2013–2014) concurrent with anything?"

**Why this matters (recruiter view):** Non-linear educational paths are common and fine, but unexplained gaps or timeline ambiguities invite scrutiny. A hiring committee member who notices "AXA 2014–2016, M.S. 2017" may ask: "What happened in 2016–2017?" Adding the M.S. enrollment year (e.g., "M.S. in Data Science, University of Rochester — 2015–2017") resolves this immediately.

**Recommendation:** Add enrollment year to the M.S. entry: "2015–2017" instead of just "2017." This makes the chronology unambiguous.

**Fix snippet:**
```markdown
- **M.S. in Data Science**, University of Rochester — *Rochester, NY* — 2015–2017
```

**Peer reference:** https://cs.stanford.edu/~karpathy/cv.pdf — Karpathy's CV lists exact enrollment and graduation years for all degrees, leaving no ambiguity.

---

```yaml
---
id: H-15
title: "No Medium posts cross-linked from the site: the bio says 'Medium' but nothing appears"
category: Content
severity: P2
confidence: HIGH
effort: 30m
agents: [hiring]
---
```

**Evidence:** The sidebar and footer both link to https://nilesh-patil.medium.com/ as an active channel. The About page also lists Medium under "Get in touch." However, no Medium articles appear on the site itself, and no recent Medium post is surfaced on the home page or blog. This creates an expectation gap: a recruiter who sees "Medium" as a contact/content channel expects to find technical writing there.

**Why this matters (recruiter view):** If Medium posts exist and are good, they are invisible to anyone who only visits the personal site. If they do not exist, the Medium link is dead branding real estate. Either way, the site promises intellectual output that it does not deliver. A recruiter who clicks the Medium link and finds no recent articles concludes the candidate is not currently publishing.

**Recommendation:** If Medium articles exist (even older ones), cross-link the top 1–2 on the About page or Blog page. If no articles exist, remove Medium from the "Get in touch" section to avoid the expectation gap — or replace it with a LinkedIn article link.

**Fix snippet:**
```markdown
## Writing elsewhere

- [Article title on Medium](https://nilesh-patil.medium.com/...) — *[date]*
- [Article title on Medium](https://nilesh-patil.medium.com/...) — *[date]*
```

**Peer reference:** https://simonwillison.net — all external writing (newsletters, guest posts) is cross-linked from the main site to create a single discovery surface.

---

```yaml
---
id: H-16
title: "About page 'Technical focus' reads as a capability list, not as evidence of leadership"
category: Content
severity: P1
confidence: HIGH
effort: 1h
agents: [hiring]
---
```

**Evidence:** The About page "Technical focus" section is a 7-item bulleted list: "LLM-based behavior simulation, persona simulators, and agentic evaluators," "Distributed recommendation, content tagging," "Feature-store systems," "Real-time forecasting," "Deep-learning churn prediction," "Self-hosted SLMs," "Compliance-aware AI harness design." Each item is a capability, not an accomplishment. There are no outcomes, no scale numbers attached to most items, and no indication of which were built vs. designed vs. contributed to.

**Why this matters (recruiter view):** A strong individual contributor has a similar capability list. A VP-level candidate differentiates by showing *what was accomplished* with those capabilities and *at what organizational scale*. The list as written could describe a senior engineer, a staff researcher, or a head of AI — it does not distinguish. Competing candidates who write "built X system that achieved Y outcome at Z scale" immediately read as more senior.

**Recommendation:** Convert the bulleted list into 3–4 accomplishment statements that anchor each capability to a scale metric or outcome. Move the pure capability list to a "Skills" or "Tools" section lower on the CV.

**Fix snippet:**
```markdown
## Selected accomplishments

- **LLM behavior simulation at scale** — designed persona simulator and agentic evaluator workflows for 250M+ user personalization at Dream11.
- **Real-time forecasting under SLA** — led system delivering ~50k+ live forecasts per [timeframe] under strict latency constraints for real-money game features.
- **Compliance-aware AI architecture** — built harness and audit-trail design for SEBI-regulated investor and trader workflows at DreamStreet.
- **Feature-store at 250M+ users** — led distributed recommendation, tagging, and feature-store systems supporting Dream11's full user base.
```

**Peer reference:** https://huyenchip.com/about/ — achievements are stated with context and outcome, not as a raw skill list.

---

```yaml
---
id: H-17
title: "DataScience Docker Environment portfolio entry is from 2018 and signals no AI work"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [hiring]
---
```

**Evidence:** The portfolio's oldest entry is "Data Science Docker Environment" dated July 2018 — a Dockerfile/compose setup for Jupyter and scikit-learn. For a candidate presenting as Head-of-AI in 2026, this entry is 8 years old and represents DevOps-adjacent tooling, not AI leadership. It has 18 GitHub stars.

**Why this matters (recruiter view):** Portfolio entries are curated signals. Including a 2018 Docker environment alongside an ACL 2026 paper creates cognitive dissonance about the candidate's level. The implicit question: "Is this the best they can show publicly?" If yes, the portfolio undersells the role. If no, why is an 8-year-old Docker setup competing for real estate with a top-venue AI paper?

**Recommendation:** Either (a) retire the Docker entry from the portfolio (keep it on GitHub, just don't surface it), or (b) add context that reframes it: "Used as the base image for N internal analytics setups at [employer]" — tying it to org-scale impact. Option (a) is preferred; every portfolio slot should advance the Head-of-AI narrative.

**Fix snippet:**
```
Remove from _portfolio/datascience-environment.md or add to frontmatter:
  hidden: true
```

**Peer reference:** https://simonwillison.net/about/ — older projects are in a GitHub archive page, not on the primary portfolio, keeping the featured work current and relevant.

---

```yaml
---
id: H-18
title: "SimuCell3D portfolio entry is a C++ HPC fork — signals engineering hobby, not AI leadership"
category: Content
severity: P2
confidence: MED
effort: 15m
agents: [hiring]
---
```

**Evidence:** The most recent portfolio entry (March 2026) is SimuCell3D — a performance-oriented fork of an ETH C++ framework for simulating 3D tissue mechanics, using OpenMP scheduling optimization. The work is technically impressive (thread efficiency 29% → 60%) but is entirely outside the AI/ML/LLM domain and appears to be a personal hobby project with no connection to DreamStreet or the Head-of-AI positioning.

**Why this matters (recruiter view):** For a generalist engineer, a C++ HPC side project signals depth and curiosity. For a Head-of-AI candidate, it creates a positioning question: "Is this person an AI leader who happens to do HPC as a hobby, or an HPC engineer who has moved into AI leadership?" The framing of the portfolio should leave no ambiguity about the primary expertise. One unrelated hobby project is fine; as one of only three portfolio entries, it takes up disproportionate real estate.

**Recommendation:** Keep SimuCell3D on GitHub but add context in the portfolio entry framing it as a technical depth exercise (e.g., "exploring low-level systems programming outside my primary AI focus"). Better: replace this slot with an AI-relevant portfolio entry and move SimuCell3D to a "Technical experiments" section.

**Fix snippet:**
```markdown
Add a note in the portfolio entry:
> This is a personal systems programming exercise outside my primary AI/ML focus. For work related to AI leadership, see the [Publications](/publications/) page.
```

**Peer reference:** https://github.com/karpathy — side projects (micrograd, nanogpt, llm.c) are tightly aligned with the primary AI researcher identity, even when educational or hobby-grade.

---

```yaml
---
id: H-19
title: "The 7-year Dream11 tenure title is ambiguous: 'Senior Principal' vs 'Head of Applied Research'"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [hiring]
---
```

**Evidence:** The CV role at Dream11 is titled "Senior Principal Research Scientist / Head of Applied Research — Dream11 — Mumbai — 2019 — 2026." The dual title creates ambiguity: Was "Head of Applied Research" a formal promotion within the 7-year tenure, or was it a simultaneous title? If it was a promotion, when did it occur? The LinkedIn equivalent would split this into two entries with dates. On the site, it reads as one continuous role with a compound title.

**Why this matters (recruiter view):** A recruiter trying to calibrate seniority needs to know: "When did this person reach the 'Head of' level — 2019, 2022, 2025?" A 2-year "Head of" tenure signals different maturity than a 5-year one. The compound title also potentially undersells: "Head of Applied Research" is a P6+ / director-equivalent title at many companies, but it is buried behind "Senior Principal Research Scientist" as a slash-prefix.

**Recommendation:** Split the 7-year Dream11 tenure into two entries: one for the IC/staff phase and one for the Head of Applied Research phase, with explicit date ranges. Lead with the most senior title if the roles overlapped.

**Fix snippet:**
```markdown
### Head of Applied Research — Dream11 — Mumbai — [year] — 2026
- [leadership bullets]

### Senior Principal Research Scientist — Dream11 — Mumbai — 2019 — [year]
- [IC/technical bullets]
```

**Peer reference:** https://www.linkedin.com/in/chiphuyen — each promotion at NVIDIA, Netflix, and Snorkel AI is listed as a separate role entry with discrete date ranges, making the career arc unambiguous.

---

```yaml
---
id: H-20
title: "No social proof from external validators: no quote, testimonial, or recognizable recommender"
category: Content
severity: P2
confidence: MED
effort: 4h
agents: [hiring]
---
```

**Evidence:** The entire site is self-reported. There are no quotes from colleagues, no LinkedIn recommendations surface-linked, no press mentions (TechCrunch, Economic Times, VCCircle) cited, no conference program committees listed, no advisory board memberships mentioned.

**Why this matters (recruiter view):** Senior AI leadership roles involve reference checks and board-level scrutiny. A site that has at least one attributed external validator — "Led the collaboration with Columbia that [Columbia professor name] called 'one of the most productive industry partnerships we've had'" — immediately differentiates. For Indian AI leadership specifically, being quoted in an ET or Mint article, or listed as a speaker on an ACM India or NASSCOM event page, is a commonly checked signal.

**Recommendation:** Surface one external validation signal: a press mention link, a conference program committee credit, a LinkedIn recommendation excerpt (with permission), or a named collaborator reference. Even "co-advised [N] Columbia students" is an external anchor.

**Fix snippet:**
```markdown
## External recognition

- Program committee / reviewer: [conference name, year]
- Quoted in: [publication, article title, date] — [link]
- Columbia University collaboration: [faculty collaborator name] and [N] students/post-docs supervised on industry problems
```

**Peer reference:** https://lilianweng.github.io/about/ — the "about" page references her role at OpenAI directly and links to her Wikipedia page, which provides third-party validation.

---

```yaml
---
id: H-21
title: "Publications include biology papers (immunology, ultrasound) that dilute the AI leadership brand"
category: Content
severity: P2
confidence: MED
effort: 30m
agents: [hiring]
---
```

**Evidence:** The Publications page lists 6 papers in reverse-chronological order. The three most recent (ACL 2026, ACM CODS-COMAD 2024, IEEE ICMLA 2023) are directly relevant to AI leadership. The three older papers (Cell Reports 2020, Journal of Immunology 2020, IEEE EMBC 2019) are from the biomedical imaging / immunology phase at the University of Rochester. A recruiter reading the full list sees a non-trivial slice of biomedical research.

**Why this matters (recruiter view):** Breadth is usually a positive, but for a Head-of-AI role, the recruiter wants signal on AI depth. When 50% of the publications list is biomedical, it raises the question: "Is this a medical imaging researcher who pivoted to AI, or an AI systems person who happened to work in a biology lab?" The answer is the latter, but the presentation doesn't make that obvious. Grouping publications by relevance to AI leadership (or reordering with a brief framing note) fixes this.

**Recommendation:** Add a brief framing line at the top of the Publications page separating the research phases: "AI & ML research (2022–present)" and "Prior domain research (biomedical imaging, 2017–2020)." This reframes the bio-papers as depth evidence ("can apply ML across domains") rather than dilution.

**Fix snippet:**
```markdown
## AI & ML research (2022–present)
- [ACL 2026, CODS-COMAD 2024, ICMLA 2023 papers]

## Prior domain research — biomedical imaging & computational biology (2017–2020)
- [EMBC 2019, Cell Reports 2020, Journal of Immunology 2020 papers]
*Work completed during a Staff Data Scientist role at the Center for Vaccine Biology, University of Rochester.*
```

**Peer reference:** https://lilianweng.github.io/publications/ — publications are implicitly in one domain (ML/AI) with no cross-domain dilution; each paper is contextualized by venue and impact.

---

```yaml
---
id: H-22
title: "No indication of conference program committee service, peer review, or professional community role"
category: Content
severity: P2
confidence: MED
effort: 30m
agents: [hiring]
---
```

**Evidence:** Neither the CV nor the About page mentions any service to the research community: no program committee credits, no journal reviewing, no conference organization, no mentorship program participation, no ACM/IEEE/NeurIPS/ICLR/ICML workshop organization.

**Why this matters (recruiter view):** For a candidate bridging industry AI leadership and research (via the Columbia collaboration), community service signals that peers in the research community recognize the candidate's expertise enough to invite them to review work. It also signals that the candidate is embedded in the field's current frontiers, not just consuming it. Its absence is a mild flag for research-forward AI leadership roles.

**Recommendation:** If any peer review or program committee service exists, add a brief "Service" section to the CV. Even "reviewer for [venue], [year]" counts. If none exists, this is an actionable gap to fill before the next active job search.

**Fix snippet:**
```markdown
## Service

- Reviewer: [ACL / EMNLP / ICML / NeurIPS / other], [year(s)]
- [Program committee, workshop organization, mentorship program, if any]
```

**Peer reference:** https://www.cs.cmu.edu/~morency/cv.pdf — CV of research-adjacent industry leader includes reviewing and program committee service as a dedicated section.

---

```yaml
---
id: H-23
title: "No metrics or outcomes for the Columbia University collaboration"
category: Content
severity: P1
confidence: HIGH
effort: 30m
agents: [hiring]
---
```

**Evidence:** The CV states: "Built Dream Sports' collaboration with Columbia University, NY and helped establish a multi-million-dollar research center." The About page echoes this. No outcome is given: How many papers resulted from the collaboration? How many students/post-docs were involved? What is the research center called? Is it still active? The Columbia University collaboration is the single most impressive external-validation signal on the site and it is presented in one sentence.

**Why this matters (recruiter view):** "Multi-million-dollar research center" is a strong claim but it is a black box. A hiring committee for a VP-AI role will ask: "Can we verify this? What came out of it?" If the answer is "6 papers, 3 post-docs, and 2 patent applications over 5 years," that is a dramatically stronger proof point than the current sentence. Even a link to the Columbia website for the center (if public) transforms this from an unverifiable claim to a verifiable fact.

**Recommendation:** Expand the Columbia collaboration bullet to include: (a) the name of the research center or partnership if public, (b) headcount (students, post-docs supervised), (c) output (papers, prototypes, or other artifacts), and (d) a link if the program has a public web presence.

**Fix snippet:**
```markdown
- Established Dream Sports' [N]-year research collaboration with **Columbia University, NY** — a **multi-million-dollar partnership** producing [N] peer-reviewed publications, [N] post-doc/student placements, and applied research across ML, AI, and sports analytics. *(See: [Columbia DSI partnership page / faculty co-author LinkedIn])*
```

**Peer reference:** https://research.google/outreach/university-relations/ — Google's research partnerships page shows how to frame academic collaboration with verifiable, specific outcomes.

---

```yaml
---
id: H-24
title: "Home page lacks a hero summary line that answers 'what do you do' in under 5 words"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [hiring]
---
```

**Evidence:** The home page opens with the site title "Nilesh Patil" (from the masthead) and the sidebar bio "Head of AI at DreamStreet." The main content opens: "AI systems & applied research. Head of AI at DreamStreet, building compliance-aware AI architecture for SEBI-regulated investor and trader workflows." This is good but front-loaded with jargon (SEBI-regulated). There is no rendered `<h1>` visible on the page (confirmed via browser evaluation: `h1` returns empty string). The first bold visual hierarchy element is the sidebar name.

**Why this matters (recruiter view):** The 5-second test: a recruiter landing on this page should be able to answer "what does this person do?" without reading a sentence. The answer should be implicit in the headline hierarchy. Without a rendered H1 on the main content area, the visual anchor is missing. The sidebar bio ("Head of AI at DreamStreet") does this work — but only if the recruiter looks left before center.

**Recommendation:** Restore or add a visible H1 to the home page main content area. The current `# Nilesh Patil` in home.md should render as an H1 but appears to be suppressed by the layout. Verify and ensure the name + title renders visually as the page's primary heading.

**Fix snippet:**
```markdown
# Nilesh Patil — Head of AI

**AI systems & applied research.** Building compliance-aware AI architecture for SEBI-regulated investor and trader workflows at DreamStreet. Previously Head of Applied Research, Dream11 — 250M+ user scale, Columbia University research collaboration.
```

**Peer reference:** https://karpathy.ai — "Andrej Karpathy" renders as an H1 with immediate one-line descriptor that answers "who / what" within 2 seconds.

---

```yaml
---
id: H-25
title: "No mention of funding raised, exits, or board-level AI strategy exposure"
category: Content
severity: P2
confidence: LOW
effort: 1h
agents: [hiring]
---
```

**Evidence:** For a candidate targeting VP-AI or Chief AI Officer roles, board-level exposure is increasingly expected. The site mentions Dream11 (publicly valued at ~$8B, BCCI-affiliated) but does not mention: any board presentations made, any role in fundraising due diligence on AI, any investor relations AI narrative. DreamStreet's funding stage and any AI-related board work are absent.

**Why this matters (recruiter view):** The jump from "Head of Applied Research at a unicorn" to "VP-AI at a public company or late-stage startup" often requires evidence of C-suite communication, board-level AI strategy, or investor-facing AI narrative. This is a lower-confidence finding because confidentiality is a legitimate reason to omit it. But if any public-facing board or investor interaction exists (e.g., a quoted in an earnings call transcript, a PitchBook-linked AI advisory), surfacing it would differentiate the candidacy significantly.

**Recommendation:** If any board or investor interaction is public and attributable, add a single line to the CV: "Presented AI strategy to [company] board / investment committee in [year]." If confidential, skip — but consider adding an advisory board role at a startup to create this signal going forward.

**Fix snippet:**
```markdown
## Advisory & board engagement *(if applicable)*
- AI Advisor, [startup name, stage] — [year–present]
- Presented AI roadmap to Dream11 board / executive leadership, [year]
```

**Peer reference:** https://huyenchip.com/about/ — "I advise Convai, OctoAI, and Photoroom" — explicit advisory roles provide board-level exposure signal without requiring confidential disclosure.
