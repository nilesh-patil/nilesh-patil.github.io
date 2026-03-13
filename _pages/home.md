---
layout: single
title: ""
permalink: /
author_profile: true
---

# Nilesh Patil

**AI systems & applied research.** Head of AI at DreamStreet, building compliance-aware AI architecture for SEBI-regulated investor and trader workflows. Previously led applied AI research at Dream11 and ran a Columbia University research collaboration.

Particularly interested in AI harness design, developer productivity, and turning emerging model capabilities into reliable workflows and products.

[More about me &rarr;](/about/) &nbsp; &middot; &nbsp; [Publications &rarr;](/publications/) &nbsp; &middot; &nbsp; [Portfolio &rarr;](/portfolio/) &nbsp; &middot; &nbsp; [CV &rarr;](/cv/) &nbsp; &middot; &nbsp; [Search &rarr;](/search/)

---

## Recent posts

<ul class="taxonomy__index">
{% for post in site.posts limit:6 %}
  <li>
    <a href="{{ post.url | relative_url }}">
      <h3>{{ post.title }}</h3>
      <small>{{ post.date | date: "%B %-d, %Y" }}{% if post.excerpt %} &middot; {{ post.excerpt | strip_html | strip_newlines | truncate: 120 }}{% endif %}</small>
    </a>
  </li>
{% endfor %}
</ul>

[All posts &rarr;](/posts/) &nbsp; &middot; &nbsp; [Posts by year &rarr;](/year-archive/) &nbsp; &middot; &nbsp; [Posts by tag &rarr;](/tag-archive/)
