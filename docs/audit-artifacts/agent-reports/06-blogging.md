# Blogging Audit — Agent 06

**Blogging verdict:** This site reads unambiguously as a dormant archive, not an active technical blog. Six posts exist: five from a five-month burst in early 2017, one isolated post in May 2020, and zero posts in the six years since — a period during which the site owner became Head of AI, built regulated AI systems, and ran a Columbia research collaboration. The stated positioning ("AI systems, agentic workflows, regulated AI") has zero overlap with what is actually published (seaborn histograms, k-means clustering, NumPy tutorials, galactic morphology). A hiring manager, potential collaborator, or media contact who navigates to /posts/ sees a data-science coursework archive frozen in 2017. The infrastructure is well-configured (read time, TOC, share buttons, related posts all render), but there is nothing new to share, and the existing content does not support the current brand claim.

---

## Findings

---

```yaml
id: C-01
title: "Six-year publishing gap (2021–2026): blog is effectively dead"
category: Content
severity: P0
confidence: HIGH
effort: 1d
agents: [blogging]
```

**Evidence:** All six posts are dated 2017-01-14 through 2020-05-20. The most recent post (`2020-05-20-distributed-kmeans-clustering.md`) is now six years old. The site bio reads "Head of AI at DreamStreet, building compliance-aware AI architecture for SEBI-regulated investor and trader workflows." `/posts/` renders a six-entry list whose newest entry predates the GPT-3 paper. At localhost:4000/posts/, the "Recent posts" heading on the home page surfaces `Distributed K-Means Clustering in Python` dated May 20, 2020 as the lead item.

**Why this matters:** A six-year gap signals the site is abandoned to any visitor who checks the dates. Peer sites that command authority in AI publish at radically different cadences: Simon Willison (simonwillison.net) publishes multiple posts per week, Lilian Weng (lilianweng.github.io) publishes meaty technical pieces several times a year and each post becomes a canonical reference — either cadence signals an active practitioner. A six-year silence signals the opposite, and directly undercuts the "AI systems leader" claim made in the bio.

**Recommendation:** Publish a minimum of one post per quarter. Prioritize topics that connect the current role to the stated positioning: lessons from building compliance-aware AI in a regulated brokerage, agentic workflow design patterns, harness design trade-offs. A single 1,500-word post on any of these would immediately reframe the blog from archive to active practice log.

**Fix snippet:**
```
Create _posts/YYYY-MM-DD-{slug}.md with:
  - A concrete lesson learned from current role
  - Code snippet or architecture diagram
  - Honest "what failed before this worked" framing
Target: 800–2000 words, one new post every 6–8 weeks minimum
```

**Peer reference:** https://lilianweng.github.io/posts/ — active posts in 2023, 2024, 2025 on topics like agent systems, alignment, tool use; each post is bookmarked industry-wide.

---

```yaml
id: C-02
title: "Topic mismatch: published content is 2017-era data-viz tutorials, positioning claims 2026 AI leadership"
category: Content
severity: P0
confidence: HIGH
effort: 4h
agents: [blogging]
```

**Evidence:** Site bio (home.md, about.md): "AI systems & applied research … compliance-aware AI architecture … agentic workflows." Published posts: seaborn distribution plots from World Bank data, NumPy vectors, scikit-learn random forests on UCI HAR dataset, k-means clustering, galaxy morphology CNN. Zero posts about LLMs, agents, regulated AI, inference pipelines, or organizational AI adoption. The word "agent" does not appear in any post. The word "LLM" does not appear in any post.

**Why this matters:** The blog is the primary proof-of-work surface for a technical brand. When the bio says "agentic workflows" but the blog shows `sns.distplot`, the disconnect reads as either dishonesty or neglect. Visitors who navigate to /posts/ after reading the home page bio will experience cognitive dissonance. Peer site Chip Huyen (huyenchip.com/blog) maintains direct continuity between stated expertise (ML systems, AI engineering) and published content.

**Recommendation:** Either retire old posts to an "archive" section with a visible note ("Early data science work, 2017") or write a bridging narrative post explaining the trajectory. More impactful: publish two to three posts on current topics to shift the visible content mix within the first six results shown on home.

**Fix snippet:**
```yaml
# In future posts, target these topic clusters:
# - "Building a compliance-aware RAG pipeline for SEBI-regulated advisory"
# - "Agentic workflow design: lessons from 6 months in production"
# - "Evaluation harness design for regulated AI systems"
# None require disclosing proprietary details — pattern-level posts are sufficient
```

**Peer reference:** https://huyenchip.com/blog/ — every post maps directly to the author's stated expertise in AI engineering and ML systems.

---

```yaml
id: C-03
title: "Comments disabled: giscus provider configured but repo_id and category_id are empty"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `_config.yml` lines 224–237:
```yaml
comments:
  provider: giscus
  giscus:
    repo_id: ""                  # paste from giscus.app
    category_id: ""              # paste from giscus.app
```
Defaults also set `comments: false` for all posts (line 177). Verified in browser: visiting `/posts/visualizing-and-comparing-distributions/`, `document.querySelector('#comments, .js-comments, .giscus')` returns `null`. The `{% if site.comments.provider and page.comments %}` gate in `_layouts/single.html:101` fails because `page.comments` is `false`.

**Why this matters:** Without comments, the blog is broadcast-only. Readers who have questions about the NumPy or k-means posts have no on-site channel. More critically, once new posts on agentic workflows appear, comments become the primary signal that the author is responsive and engaged — a meaningful trust signal for hiring managers and collaborators. Simon Willison's blog (simonwillison.net) runs comments; his replies in comment threads have become cited resources themselves.

**Recommendation:** Run `https://giscus.app`, enable GitHub Discussions on the repo, paste the generated `repo_id` and `category_id` into `_config.yml`, then change the posts default to `comments: true`.

**Fix snippet:**
```yaml
# _config.yml
comments:
  provider: giscus
  giscus:
    repo_id: "PASTE_FROM_GISCUS_APP"
    category_id: "PASTE_FROM_GISCUS_APP"

# defaults > posts
comments: true   # was: false
```

**Peer reference:** https://lilianweng.github.io — comments via Disqus on each post, reader engagement visible per article.

---

```yaml
id: C-04
title: "Share buttons send only path (not full URL): links are broken for all social platforms"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `_includes/social-share.html` line 13:
```
https://bsky.app/intent/compose?text={{ base_path }}{{ page.url }}
```
`base_path` resolves to `https://nilesh-patil.github.io` and `page.url` to `/posts/distributed-kmeans-clustering/` — so the combined value is `https://nilesh-patil.github.io/posts/distributed-kmeans-clustering/`. For Bluesky this is correct. However for X (line 21):
```
https://x.com/intent/post?text={{ base_path }}{{ page.url }}
```
The `text` parameter receives only the URL, not a tweet message — X requires `text` to be the composed tweet body and `url` to be a separate parameter. The LinkedIn share URL (line 17) also only passes the article path without a title or summary. Mastodon uses `url_encode` but the others do not URL-encode the path, risking breakage on pages with query strings.

**Why this matters:** A reader who clicks "Share on X" gets a blank tweet with a raw URL. A reader who clicks LinkedIn gets a share with no title. These broken share flows mean every post-level CTA for distributing content silently fails.

**Recommendation:** Fix each platform's URL schema to pass the correct parameters:

**Fix snippet:**
```liquid
{# X / Twitter: use url= separate from text= #}
<a href="https://x.com/intent/tweet?url={{ site.url }}{{ page.url | url_encode }}&text={{ page.title | url_encode }}">

{# LinkedIn: pass title as summary #}
<a href="https://www.linkedin.com/shareArticle?mini=true&url={{ site.url }}{{ page.url | url_encode }}&title={{ page.title | url_encode }}">

{# Bluesky: text should include title + URL #}
<a href="https://bsky.app/intent/compose?text={{ page.title | url_encode }}%20{{ site.url }}{{ page.url | url_encode }}">
```

**Peer reference:** https://simonwillison.net — share buttons pass full URLs with encoded titles on every post.

---

```yaml
id: C-05
title: "No newsletter or follow CTA at post end: readers have no path to stay connected"
category: Content
severity: P1
confidence: HIGH
effort: 1h
agents: [blogging]
```

**Evidence:** All six posts end abruptly: the distributions post ends at a rugplot image, the NumPy post ends mid-code with an array slice output, the k-means post ends at a closing `}` in a code block. Grep for "newsletter", "subscribe", "mailing list", "follow me" across `_posts/` and `_pages/` returns zero results. The `_layouts/single.html` has no end-of-post CTA block. Share buttons are present but they do not invite the reader to return.

**Why this matters:** A reader who finishes a post and wants to hear when the next one drops has no mechanism — not RSS discovery, not email, not even a prominent "follow me on X" line. The author has a Medium presence (`nilesh-patil.medium.com`) referenced in the footer but not mentioned in any post. Karpathy's blog (karpathy.ai) ends posts with a Twitter/X follow prompt; Simon Willison's ends with a link to the TIL feed and newsletter.

**Recommendation:** Add a short closing section to `_layouts/single.html` after the content and before share buttons. At minimum, point to RSS and the author's X/LinkedIn.

**Fix snippet:**
```html
{# _layouts/single.html — insert after </section> .page__content, before page__share #}
{% if page.layout == 'single' and page.collection == 'posts' %}
<div class="page__cta">
  <p>If this was useful, <a href="/feed.xml">subscribe via RSS</a> or follow
  <a href="https://x.com/ensembledme">@ensembledme</a> for new posts.</p>
</div>
{% endif %}
```

**Peer reference:** https://karpathy.ai — every post ends with a follow prompt linking to Twitter/X and GitHub.

---

```yaml
id: C-06
title: "Post titles are descriptive labels, not click invitations"
category: Content
severity: P1
confidence: HIGH
effort: 1h
agents: [blogging]
```

**Evidence:** Current titles:
- "Visualizing distributions" (4 words, generic)
- "Working with numpy" (3 words, tutorial label)
- "Human activity recognition" (3 words, topic label)
- "Galactic Morphology using Deep-Learning" (5 words, academic label)

None state a finding, a surprising result, or a concrete outcome. Compare: "Distributed K-Means Clustering in Python" is slightly better (states the technique and language) but still does not give the reader a reason to click.

**Why this matters:** Titles are the primary discovery surface: they appear in /posts/, year-archive, tag-archive, the home page recent-posts list, RSS readers, and any social share. A title like "Working with numpy" competes with thousands of identical-sounding tutorials. Lilian Weng titles her posts to state a concept or thesis: "Prompt Engineering", "LLM Powered Autonomous Agents", "Extrinsic Hallucinations in LLMs" — descriptive but specific. Chip Huyen uses titles that frame a position: "Real-time machine learning: challenges and solutions".

**Recommendation:** Rename existing posts to state the finding or the reader's gain. For new posts, write the title last — after you know what the most interesting takeaway actually is.

**Fix snippet:**
```yaml
# Current → Proposed
"Visualizing distributions" → "7 distribution plots every data scientist should know (with Python code)"
"Working with numpy" → "NumPy array operations reference: vectors, matrices, linear algebra"
"Human activity recognition" → "94% accuracy on smartphone sensor data with 5 features and 50 trees"
"Galactic Morphology using Deep-Learning" → "Teaching a CNN to classify galaxies: the Galaxy Zoo approach"
```

**Peer reference:** https://huyenchip.com/blog/ — titles state a concrete subject or position, not a generic topic label.

---

```yaml
id: C-07
title: "Galactic morphology post is structurally incomplete: sets up model, never shows results"
category: Content
severity: P1
confidence: HIGH
effort: 1h
agents: [blogging]
```

**Evidence:** `_posts/2017-07-25-galactic-morphology-using-deep-learning.md` describes the Galaxy Zoo dataset (lines 35–48), explains the CNN architecture including dropout and batch normalization (lines 49–78), then ends at "Setup" with two references. No training results, no loss curves, no example predictions, no accuracy table. The post cuts off after describing what the final layers do: "The loss is back-propagated to update the convolutional kernel weights… the network learns features that are optimal for the task at hand." — line 75. The References section follows immediately. There is no Results or Conclusion section.

**Why this matters:** A post that describes a model architecture without showing results is an unfinished draft published as a post. Readers who click expecting to see galaxy classification performance get architectural theory and nothing else. This damages credibility more than not publishing at all.

**Recommendation:** Add a Results section showing at minimum: training/validation loss curves or a RMSE figure, two to three example predictions, and a brief qualitative discussion of failure cases.

**Fix snippet:**
```markdown
## Results

Training for 30 epochs on the 80k-image training split, the model achieved:
- Training RMSE: X.XX
- Validation RMSE: X.XX (Kaggle leaderboard position: XXX/XXX)

Example predictions vs. human volunteer scores:
| Image ID | True score (smooth) | Predicted |
|----------|--------------------:|----------:|
| ...      |                0.87 |      0.82 |

The model performs best on face-on spirals and ellipticals; edge-on galaxies
with ambiguous orientation produce the largest errors.
```

**Peer reference:** https://lilianweng.github.io/posts/2017-06-21-overview-word-embeddings/ — complete posts always include results tables, sample outputs, and qualitative analysis.

---

```yaml
id: C-08
title: "Galactic morphology post has filename/front-matter date mismatch"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** File: `_posts/2017-07-25-galactic-morphology-using-deep-learning.md`. Front matter: `date: 2017-07-15T15:39:55-04:00`. The filename encodes July 25; the front matter date says July 15 — a 10-day discrepancy. Jekyll uses the front matter `date` for rendering, so the post shows "July 16, 2017" in the browser (UTC offset applied to -04:00), while the filename says July 25.

**Why this matters:** When someone verifies dates for citation or audit purposes, the discrepancy raises questions about data integrity on an otherwise careful site. The year-archive groups this post under its rendered date, not the filename date — consistent, but the filename is misleading.

**Recommendation:** Rename the file to match the front matter date.

**Fix snippet:**
```bash
git mv _posts/2017-07-25-galactic-morphology-using-deep-learning.md \
       _posts/2017-07-15-galactic-morphology-using-deep-learning.md
```

**Peer reference:** https://simonwillison.net — filenames and displayed dates are consistent across all posts (verified by RSS feed timestamps).

---

```yaml
id: C-09
title: "All posts share a single category 'blog': category-archive is useless as a discovery tool"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `grep -h "^categories:" _posts/*.md` returns `categories: [blog]` for all six posts. The `/category-archive/` page at localhost:4000/category-archive/ renders one section header — "blog" — with all six posts beneath it. The category taxonomy provides zero navigational value.

**Why this matters:** Category-archive is meant to let readers browse thematically. With a single category used for everything, the page is equivalent to the year-archive minus the year grouping. As post volume grows, this becomes actively misleading — a "machine-learning" vs "systems" vs "data-engineering" split would let readers self-select relevant content.

**Recommendation:** Assign meaningful categories to existing posts and adopt a two-to-four category vocabulary for future posts aligned with stated topics.

**Fix snippet:**
```yaml
# Proposed category vocabulary:
# data-visualization, machine-learning, distributed-systems, ai-systems

# Update each post:
# 2017-01-14: categories: [data-visualization]
# 2017-02-15: categories: [machine-learning]
# 2017-03-04: categories: [machine-learning]
# 2017-03-14: categories: [data-visualization, distributed-systems]
# 2017-07-25: categories: [machine-learning]
# 2020-05-20: categories: [machine-learning, distributed-systems]
```

**Peer reference:** https://simonwillison.net/tags/ — granular tag/category taxonomy allows readers to browse by tool, technique, or topic.

---

```yaml
id: C-10
title: "Human activity recognition post has placeholder alt text 'image' on all 9 figures"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `_posts/2017-02-15-human-activity-recognition.md` lines 47, 64, 99, 108, 109, 136, 157, 158, 169: all HTML `<img>` tags use `alt="image"` — a meaningless, non-descriptive placeholder. Examples:
```html
<img src="/images/blog/activityRecognition/image8.png" alt="image" height="1000px" width="1000px">
<img src="/images/blog/activityRecognition/08.a.ConfusionMatrix-Test_RF.png" alt="image">
```

**Why this matters:** Screen readers announce these as "image" with no context. Search crawlers cannot infer figure content. The confusion matrices, feature importance charts, and distribution plots have meaningful content that would benefit SEO and accessibility if described in alt text.

**Recommendation:** Replace `alt="image"` with descriptive alt text for each figure, matching the existing `<figcaption>` text where available.

**Fix snippet:**
```html
<!-- Before -->
<img src="/images/blog/activityRecognition/image8.png" alt="image">
<img src="/images/blog/activityRecognition/08.a.ConfusionMatrix-Test_RF.png" alt="image">

<!-- After -->
<img src="/images/blog/activityRecognition/image8.png" 
     alt="Correlation matrix heatmap across all 561 sensor features">
<img src="/images/blog/activityRecognition/08.a.ConfusionMatrix-Test_RF.png"
     alt="Confusion matrix for Random Forest model on test set — 94.37% accuracy">
```

**Peer reference:** https://lilianweng.github.io — all figures have descriptive alt text and inline captions matching the figure label in surrounding text.

---

```yaml
id: C-11
title: "Distributions post images use 'Histogram', 'png' as alt text: inconsistent and inadequate"
category: Content
severity: P2
confidence: MED
effort: 15m
agents: [blogging]
```

**Evidence:** `_posts/2017-01-14-visualizing-and-comparing-distributions.md`:
- Line 80: `![Histogram](/images/blog/distributions/01.histogram.png)` — alt text is "Histogram" (acceptable but minimal)
- Lines 124, 170, 218, 274, 367, 410: `![png](...)` — alt text is literally "png", meaningless for screen readers or search

**Why this matters:** Six out of seven images in the first post use "png" as alt text, which describes the file format rather than the content. A screen reader user cannot distinguish the scatter plot from the violin plot from the heatmap.

**Recommendation:** Replace each `![png]` with a brief description of the chart content and the dataset shown.

**Fix snippet:**
```markdown
<!-- Before -->
![png](/images/blog/distributions/02.scatter.png)

<!-- After -->
![Scatter plot of female vs male unemployment rates by country (World Bank data)](/images/blog/distributions/02.scatter.png)
```

**Peer reference:** https://huyenchip.com/blog/ — all figure alt text describes the diagram content, not the file type.

---

```yaml
id: C-12
title: "NumPy and distributions posts end abruptly mid-content: no summary, no conclusion, no CTA"
category: Content
severity: P1
confidence: HIGH
effort: 1h
agents: [blogging]
```

**Evidence:**
- `2017-03-04-working-with-numpy.md` ends at line 347 with a code block showing `array([[2, 3], [2, 3], [2, 3]])` — no concluding sentence, no summary, no "what to read next."
- `2017-01-14-visualizing-and-comparing-distributions.md` ends at line 411 with a rugplot image, no prose after the final `![png]`.
- `2020-05-20-distributed-kmeans-clustering.md` ends at a closing `}` inside a code block (line 592).

**Why this matters:** Posts that trail off into code give readers no landing point. There is no sense of completion, no actionable takeaway, and no natural place to surface "you might also enjoy" before the template's auto-generated related posts appear. Lilian Weng concludes every post with a "Citation" section that explicitly validates the work; Simon Willison ends with a datestamp and contextual note — both signal authorial presence.

**Recommendation:** Add a brief "Takeaways" or "Summary" section to each post, even two to three sentences, before the last code block or image.

**Fix snippet:**
```markdown
## Takeaways

- NumPy vectorized operations replace element-wise loops and stay close
  to mathematical notation.
- `@` (Python 3.5+) is the preferred matrix-multiplication operator over `np.dot`.
- Subsetting with `x[i:j, k:l]` selects row and column ranges simultaneously.

[Next: working with Pandas →](/posts/some-future-post/)
```

**Peer reference:** https://lilianweng.github.io/posts/2018-06-24-attention/ — concludes with a "Summary" heading, then a "Citation" block; clean ending with call to action implicit in the citation format.

---

```yaml
id: C-13
title: "Post excerpts are all one-clause summaries: they do not invite the click"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** All six post excerpts (from front matter):
- "Common visualization examples for distributions"
- "Activity detection from sensor data"
- "Vectors, matrices, and basic linear algebra with NumPy"
- "Analyzing a real world graph : transportation network in NYC"
- "Training a deep neural net to understand galactic structure"
- "Implementing scalable k-means clustering using distributed computing frameworks"

All are noun-phrase descriptions. None state a finding, result, surprising fact, or concrete reader benefit. These render verbatim on the home page recent-posts list and on /category-archive/.

**Why this matters:** Excerpt text is the second copy after the title — it is the deciding text for whether a reader clicks. "Common visualization examples for distributions" and "Vectors, matrices, and basic linear algebra with NumPy" describe the content without differentiation. The home page recent-posts list in `home.md` truncates excerpts at 120 characters, so an excerpt that starts strong is especially valuable.

**Recommendation:** Rewrite excerpts as one sentence that states the most interesting finding or the concrete reader take-away.

**Fix snippet:**
```yaml
# Before:
excerpt: "Common visualization examples for distributions"

# After:
excerpt: "Seven seaborn chart types — histogram to violin — on World Bank life-expectancy and trade data, with full code."

# Before:
excerpt: "Activity detection from sensor data"

# After:
excerpt: "94% accuracy on six activity classes using just 5 of 561 sensor features, with 50 random forest trees."
```

**Peer reference:** https://huyenchip.com/blog/ — each post card shows a two-sentence excerpt that states the central argument, not a topic label.

---

```yaml
id: C-14
title: "NYC taxi post has three uncorrected typos in published prose"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `_posts/2017-03-14-transportation-graph-nyc-taxi-data.md`:
- Line 48: "wekday" (should be "weekday")
- Line 48: "Penn station ahs multiple entries" (should be "has")
- Line 63: "atleast" (should be "at least")
- Line 115: "straighforward" (should be "straightforward")

**Why this matters:** This is the longest and most technically ambitious of the 2017 posts, with 13 academic references and network-analysis methodology. Spelling errors in the body text undermine the impression of careful work. A hiring manager reading this post as a work sample will notice.

**Recommendation:** Fix the four typos. Run a spell-check pass over all posts as part of any content refresh.

**Fix snippet:**
```markdown
# Line 48
We create features for month, day, weekday, period of the day etc.
An issue … Penn station has multiple entries

# Line 63
edges with at least 500 trips in 2015

# Line 115
minor rerouting is usually pretty straightforward
```

**Peer reference:** https://simonwillison.net — published prose is consistently spell-checked; no published post contains "ahs" or "wekday."

---

```yaml
id: C-15
title: "No internal cross-linking between posts: posts are islands"
category: Content
severity: P1
confidence: HIGH
effort: 1h
agents: [blogging]
```

**Evidence:** `grep -rn "/posts/" _posts/` returns zero results (excluding `permalink` and `redirect_from`). No post links to any other post. The NumPy post would naturally link to the distributions post (both use pandas/seaborn); the k-means post would naturally link to the distributions post (both discuss cluster visualization). The "related posts" feature is wired and renders four tiles, but those links are algorithmic — there are no authored cross-references in prose.

**Why this matters:** Internal linking keeps readers on-site, builds navigational context, and signals to search crawlers that posts are part of a coherent body of work rather than isolated pages. Simon Willison's posts habitually link back to earlier TIL notes and blog posts — creating a navigable knowledge graph rather than a list of isolated articles.

**Recommendation:** In the next revision pass, add one to two in-prose links per post to related posts. The "See also" section pattern (a short bulleted list at the end of a post) is the most efficient way to add cross-linking retroactively.

**Fix snippet:**
```markdown
<!-- At the end of working-with-numpy.md -->
## See also

- [Visualizing distributions with seaborn →](/posts/visualizing-and-comparing-distributions/)
- [Distributed K-Means Clustering in Python →](/posts/distributed-kmeans-clustering/)
```

**Peer reference:** https://simonwillison.net — every post contains multiple in-prose links to other posts on the same site; navigation is woven into the writing.

---

```yaml
id: C-16
title: "No link from About page to Blog: the primary positioning page doesn't surface the posts"
category: Content
severity: P1
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `_pages/about.md` contains six sections (bio, Currently, Technical focus, Background, Currently exploring, Get in touch) and links to GitHub, LinkedIn, Google Scholar, Medium, and Stack Exchange — but no link to `/posts/`. The home page surfaces recent posts, but a visitor landing on `/about/` directly (as many do from LinkedIn or a resume link) has no navigation path to the blog unless they use the top nav "Blog" link.

**Why this matters:** The About page is the most frequently visited page for new visitors arriving via LinkedIn or referrals. It's the place where a visitor decides whether to dig deeper. Not linking to the blog from this page means potential readers who form an impression from the bio never discover the evidence (however dated) that this is a practitioner who documents their work.

**Recommendation:** Add one sentence in the "Background" or "Currently exploring" section pointing to the posts.

**Fix snippet:**
```markdown
## Currently exploring

AI harness design, developer productivity, and turning emerging model capabilities
into reliable workflows and products. I write about these topics occasionally on
[the blog →](/posts/).
```

**Peer reference:** https://huyenchip.com/about/ — About page links directly to the author's book, blog, and newsletter; the blog is surfaced as a primary proof-of-work artifact.

---

```yaml
id: C-17
title: "Year-archive and /posts/ listing are duplicate pages with no differentiation"
category: Design/UX
severity: P2
confidence: HIGH
effort: 1h
agents: [blogging]
```

**Evidence:** `/posts/` (`_pages/posts.html`) and `/year-archive/` (`_pages/year-archive.html`) use identical Liquid templates — both loop over `site.posts`, both group by year with `archive__subtitle` headers, both render `archive-single.html` in list mode. The only difference is the page title ("Blog" vs "Blog posts"). The home page footer links to both: "All posts →" and "Posts by year →".

**Why this matters:** Linking two pages with identical content creates navigational confusion and wastes two slots in the home page footer link row. A user who clicks "Posts by year" expects something richer than "Posts" — perhaps a count per year, a visual timeline, or a jump-link index of years. With six posts across two years, both pages render identically.

**Recommendation:** Either merge the two into one canonical URL (`/posts/` with year grouping) and remove `/year-archive/`, or differentiate them: make `/year-archive/` the archive with a count-per-year summary index at top, and keep `/posts/` as a flat chronological list.

**Fix snippet:**
```liquid
{# At top of year-archive.html, before the loop — add a year index #}
{% assign years = site.posts | map: "date" | map: "year" | uniq %}
<nav class="year-index">
  {% for year in years %}
    <a href="#{{ year }}">{{ year }} ({{ site.posts | where_exp: "p", "p.date contains year" | size }})</a>
  {% endfor %}
</nav>
```

**Peer reference:** https://simonwillison.net/archive/ — year-grouped archive with post counts per year at the top, clearly differentiated from the main /blog/ listing.

---

```yaml
id: C-18
title: "Post header images missing on 3 of 6 posts: inconsistent visual treatment in archive listings"
category: Design/UX
severity: P2
confidence: HIGH
effort: 1h
agents: [blogging]
```

**Evidence:** Only three posts have `header.overlay_image` and `header.teaser` set: NYC taxi data, galactic morphology, distributed k-means. The other three (distributions, human activity recognition, NumPy) have no header image. In grid-mode archive (used by the "related posts" widget at the bottom of each post), posts without a teaser image render as text-only tiles — visually inconsistent alongside the image-bearing tiles.

**Why this matters:** When the "You May Also Enjoy" section shows four related posts — some with hero images, some without — the tiles are visually unequal and the text-only tiles look like placeholders. This is visible on every post page. Lilian Weng's blog uses consistent card images across all posts; the visual uniformity signals editorial care.

**Recommendation:** Add a `teaser` image to the three posts that lack one. A simple generated or stock image representing the topic is sufficient — the goal is visual consistency in the related-posts grid.

**Fix snippet:**
```yaml
# Add to front matter of distributions, HAR, and numpy posts:
header:
  teaser: /images/blog/feature/data-viz.jpg   # or any suitable 300x200 image
```

**Peer reference:** https://lilianweng.github.io — every post has a consistent featured image that appears in archive grid views.

---

```yaml
id: C-19
title: "RSS feed is valid and present, but is not discoverable via in-page link"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `/feed.xml` returns a valid Atom 1.0 feed (verified at localhost:4000/feed.xml — generator: Jekyll 4.4.1, 6 entries, all with full content). The `jekyll-feed` plugin is active. However: no post or archive page displays a visible RSS link or icon. The footer links (in `_config.yml` footer.links) include GitHub, Medium, LinkedIn, and Google Scholar, but not the RSS feed. The `<link rel="alternate" type="application/atom+xml">` tag is present in the HTML head for auto-discovery by readers that support it, but no visible CTA exists for users who do not already know to look.

**Why this matters:** RSS remains the primary subscription mechanism for technical blogs in 2026. Developers and power users who read blogs through Feedly, NetNewsWire, or similar expect a visible RSS icon or link. Simon Willison and Lilian Weng both prominently link their RSS feeds in site footers.

**Recommendation:** Add the RSS feed to the footer links array in `_config.yml`.

**Fix snippet:**
```yaml
# _config.yml
footer:
  links:
    - label: "RSS"
      icon: "fas fa-fw fa-rss"
      url: "/feed.xml"
    # ... existing links
```

**Peer reference:** https://simonwillison.net — RSS link is prominent in the site footer with an orange RSS icon, present on every page.

---

```yaml
id: C-20
title: "Opening hooks rely on 'In this post' framing rather than earning reader attention"
category: Content
severity: P2
confidence: HIGH
effort: 4h
agents: [blogging]
```

**Evidence:**
- Visualizing distributions: "Once you have your data, usually you start by building summaries…" — textbook framing, no hook.
- NumPy: "NumPy is a Python library that provides fast computation over arrays…" — dictionary-definition opening.
- Human activity recognition: "A modern smartphone comes equipped with variety of sensors…" — general background statement with no tension or specificity.
- Galactic morphology: "Astronomy has historically been one of the most data intensive fields…" — broad historical claim, no entry point for the reader.
- K-means: "K-means clustering is one of the most widely used unsupervised machine learning algorithms…" — textbook first sentence.

None of the five lead paragraphs start with a surprising finding, a concrete problem, a provocative claim, or a reader-relevant question.

**Why this matters:** Readers decide in the first two sentences whether to continue. An opening like "We got 94% accuracy on six activity classes using just five of 561 features — here's how" is more likely to retain a reader than "A modern smartphone comes equipped with a variety of sensors." Chip Huyen opens "Real-time ML" with the cost of bad latency in revenue terms; this creates immediate stakes.

**Recommendation:** Rewrite the first paragraph of each post to open with the most surprising or concrete result, then build back to the methodology.

**Fix snippet:**
```markdown
<!-- Before (Human activity recognition) -->
A modern smartphone comes equipped with variety of sensors from motion detectors to optical calibrators.

<!-- After -->
We reduced a 561-feature accelerometer dataset to 5 features and still classified walking, sitting,
and standing with 94% accuracy. Here is what we dropped and why it didn't matter.
```

**Peer reference:** https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html — opens with a concrete business cost framing before introducing any methodology.

---

```yaml
id: C-21
title: "Distributed k-means post ends inside a code block: no Conclusion section despite having a prose Conclusion heading mid-post"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `_posts/2020-05-20-distributed-kmeans-clustering.md` line 531: `## Conclusion` section exists and is well-written (two paragraphs + a "Key Takeaways" bullet list ending at line 553). However, the post then continues with a `## Best Practices` section (lines 437–529) and a final code block (`complete_distributed_kmeans_pipeline`, lines 556–592) that appears after the Conclusion — the Conclusion is not the final section. The post ends mid-code:
```python
        finally:
            spark.stop()
```
The structural order is: Introduction → … → Conclusion → (more code). This inverted structure means the prose summary is buried before additional implementation code.

**Why this matters:** Readers who skip to the end see a code block; readers who read the Conclusion expect to be done but then encounter more content. The post is internally coherent but structurally confusing.

**Recommendation:** Move the "Complete pipeline" code block before the Conclusion, or add a two-line note after the final code block pointing back to the key takeaways.

**Fix snippet:**
```markdown
<!-- At the very end of the post, after the pipeline code block -->
---
*The key trade-offs are summarized in the [Conclusion](#conclusion) above.
For questions or corrections, [open an issue on GitHub](https://github.com/nilesh-patil).*
```

**Peer reference:** https://lilianweng.github.io — posts always end with the highest-level summary or citation block; code examples precede the summary, never follow it.

---

```yaml
id: C-22
title: "Tag taxonomy is inconsistent: mixed hyphenation, capitalization, and granularity across posts"
category: Content
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** Tags across all posts:
- `RandomForest` (CamelCase, no space)
- `random forests` — not used but implied by content
- `deep learning` (two words, space)
- `machine-learning` (hyphenated)
- `machine learning` — not used but the k-means post uses `machine-learning`
- `computer-vision` (hyphenated)
- `distributed-computing` (hyphenated)
- `pyspark` (no hyphen, lowercase)
- `nyc` (acronym, lowercase)

The tag-archive page groups each unique string as a separate tag. A reader browsing `/tag-archive/` sees `RandomForest` and `deep learning` and `machine-learning` as distinct non-linked categories. There is no canonical `machine-learning` tag spanning the human activity, galaxy, and k-means posts.

**Why this matters:** Tag archives are only useful when the taxonomy is consistent. A reader looking for "machine learning posts" will miss the HAR post (tagged `RandomForest`, `model`) and the galaxy post (tagged `deep learning`, `neural networks`) unless they know to look for each variant.

**Recommendation:** Normalize tags to lowercase-hyphenated form and align vocabulary across posts.

**Fix snippet:**
```yaml
# Proposed normalized tag set:
# machine-learning, deep-learning, data-visualization, python,
# distributed-computing, graph-analysis, computer-vision, clustering

# Fix:
# 2017-02-15: tags: [sensors, machine-learning, random-forest, python]
# 2017-07-25: tags: [deep-learning, computer-vision, galaxy, neural-networks]
```

**Peer reference:** https://simonwillison.net/tags/ — consistent lowercase-hyphenated tags, no duplicates between "machine learning" and "machine-learning."

---

```yaml
id: C-23
title: "Pagination labels are 'Previous'/'Next' with no titles: navigating between posts is blind"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `_includes/post_pagination.html` lines 6 and 11: the pagination links render as plain "Previous" and "Next" text. The `title` attribute on each `<a>` tag is set correctly (it contains the post title for tooltip on hover), but the visible link text is just the direction label with no preview of what the adjacent post is. Verified in browser on the distributions post: pagination shows "Next" (disabled, first post) and "Previous" → "Human activity recognition" — but only visible on hover.

**Why this matters:** A reader who finishes the distributions post and sees only "Previous" has no reason to click without hovering to discover the title. Including the post title in the link text eliminates friction. This is a common UX pattern on all leading technical blogs.

**Recommendation:** Update `post_pagination.html` to include the adjacent post title in the visible link text.

**Fix snippet:**
```liquid
{# _includes/post_pagination.html — show title in link #}
{% if page.previous %}
  <a href="{{ base_path }}{{ page.previous.url }}" class="pagination--pager">
    ← {{ page.previous.title }}
  </a>
{% endif %}
{% if page.next %}
  <a href="{{ base_path }}{{ page.next.url }}" class="pagination--pager">
    {{ page.next.title }} →
  </a>
{% endif %}
```

**Peer reference:** https://simonwillison.net — previous/next post navigation shows full post titles as visible link text, not just direction arrows.

---

```yaml
id: C-24
title: "Medium presence mentioned in footer but none of the six blog posts links to Medium cross-posts"
category: Content
severity: P2
confidence: MED
effort: 15m
agents: [blogging]
```

**Evidence:** The site footer links to `https://nilesh-patil.medium.com/`. The config `social.links` array includes `https://nilesh-patil.medium.com/`. However, none of the six posts contains a link to or mention of the corresponding Medium post (if any exist), and no post body mentions Medium. The two publishing channels (GitHub Pages and Medium) are siloed.

**Why this matters:** If any posts are cross-posted to Medium, readers on either platform who want the full backlog cannot navigate between them. If they are not cross-posted, the Medium link in the footer is a dead promise — it exists but the account may have no equivalent content. Either way, the relationship between the two channels should be made explicit.

**Recommendation:** If posts exist on Medium, add a canonical note at the bottom of each GitHub Pages post: "Originally published on [Medium](URL)." If Medium is used for different content, link to it prominently from the About page with a description of what to expect there.

**Fix snippet:**
```markdown
<!-- At end of each post, if cross-posted -->
---
*This post is also available on [Medium](https://nilesh-patil.medium.com/your-post-slug).*
```

**Peer reference:** https://huyenchip.com — clearly delineates which content lives on her personal site vs. other platforms; no reader is confused about where the canonical version lives.

---

```yaml
id: C-25
title: "Home page 'Recent posts' section shows all 6 posts: no visible signal that the blog is dormant"
category: Design/UX
severity: P1
confidence: HIGH
effort: 15m
agents: [blogging]
```

**Evidence:** `_pages/home.md` line 21: `{% for post in site.posts limit:6 %}`. With exactly 6 posts, all posts appear in "Recent posts." The most recent post date shown is "May 20, 2020" — six years ago. There is no "Last updated" or "Active since" indicator. A visitor unfamiliar with the author reads "Recent posts" and sees a 2020 entry as the newest — the label "Recent" is actively misleading.

**Why this matters:** "Recent" implies the past few weeks or months. Showing 2020 content under a "Recent posts" heading is a credibility signal that the site is not maintained. The fix is either a new post (the real solution) or an honest label change while new content is being prepared.

**Recommendation:** Change "Recent posts" to "Posts" or "Writing" on the home page, and reduce the limit to 3–4 posts (the most relevant, not the most recent). Add the most recent post's date as a subtitle.

**Fix snippet:**
```liquid
{# _pages/home.md #}
## Writing

<ul class="taxonomy__index">
{% for post in site.posts limit:4 %}
  <li>
    <a href="{{ post.url | relative_url }}">
      <h3>{{ post.title }}</h3>
      <small>{{ post.date | date: "%B %-d, %Y" }}{% if post.excerpt %} &middot; {{ post.excerpt | strip_html | truncate: 100 }}{% endif %}</small>
    </a>
  </li>
{% endfor %}
</ul>
```

**Peer reference:** https://karpathy.ai — home page labels the writing section "Blog" without implying recency; dates are prominently displayed so visitors self-calibrate.
