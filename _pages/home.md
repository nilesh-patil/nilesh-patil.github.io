---
layout: single
title: ""
seo_title: "Nilesh Patil - AI systems & field reports"
permalink: /
author_profile: true
---

# Nilesh Patil
{: .home-title}

**AI systems & applied research.** I'm Head of AI at DreamStreet, building compliance-aware AI for SEBI-regulated (India's securities regulator) investor and trader workflows. Before that I led applied AI research at Dream11 and ran a Columbia University research collaboration.

This site is where I write up the engineering behind that work - mostly field reports on HPC, applied ML, and the measurements behind the results.

[About &rarr;](/about/) &nbsp;&middot;&nbsp; [Publications &rarr;](/publications/) &nbsp;&middot;&nbsp; [Side projects &rarr;](/portfolio/) &nbsp;&middot;&nbsp; [CV &rarr;](/cv/) &nbsp;&middot;&nbsp; [Search &rarr;](/search/)

{% assign latest = site.posts | first %}
<a class="home-feature" href="{{ latest.url | relative_url }}">
  <span class="home-feature__kicker">Latest &middot; {{ latest.date | date: "%d %b %Y" }} &middot; {{ latest.content | strip_html | number_of_words | divided_by: site.words_per_minute | plus: 1 }} min{% if latest.measurement %} &middot; {{ latest.measurement }}{% endif %}</span>
  <span class="home-feature__title">{{ latest.title }}</span>
  <span class="home-feature__lead">{{ latest.excerpt | markdownify | strip_html | strip_newlines | strip }}</span>
  {% if latest.tags.size > 0 %}<span class="post-row__tags">{% for tag in latest.tags limit: 3 %}<span class="post-row__tag">{{ tag }}</span>{% endfor %}</span>{% endif %}
</a>

## More posts

<div class="home-rows">
{% for post in site.posts offset: 1 limit: 5 %}
  {% include archive-single.html summary=true %}
{% endfor %}
</div>

[All {{ site.posts | size }} posts &rarr;](/posts/) &nbsp;&middot;&nbsp; [Posts by year &rarr;](/year-archive/) &nbsp;&middot;&nbsp; [Posts by tag &rarr;](/tag-archive/)

## Side projects

<div class="home-rows">
{% assign side_projects = site.portfolio | sort: 'date' | reverse %}
{% for post in side_projects %}
  {% include archive-single.html %}
{% endfor %}
</div>

[All side projects &rarr;](/portfolio/)
