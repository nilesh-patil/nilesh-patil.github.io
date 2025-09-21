---
layout: single
title: ""
permalink: /
author_profile: true
---

# Nilesh Patil

**AI systems & applied research.** Head of AI at DreamStreet, building compliance-aware AI architecture for SEBI-regulated investor and trader workflows. Previously led applied AI research at Dream11 and ran a Columbia University research collaboration.

Particularly interested in AI harness design, developer productivity, and turning emerging model capabilities into reliable workflows and products.

[More about me &rarr;](/about/) &nbsp; &middot; &nbsp; [Publications &rarr;](/publications/) &nbsp; &middot; &nbsp; [Side projects &rarr;](/portfolio/) &nbsp; &middot; &nbsp; [CV &rarr;](/cv/) &nbsp; &middot; &nbsp; [Search &rarr;](/search/)

---

## Recent posts

<ul class="taxonomy__index">
{% for post in site.posts limit:6 %}
  <li class="post-row">
    <time class="post-row__date" datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%d %b %Y" }}</time>
    <span class="post-row__title"><a class="post-row__link" href="{{ post.url | relative_url }}">{{ post.title }}</a></span>
  </li>
{% endfor %}
</ul>

[All posts &rarr;](/posts/) &nbsp; &middot; &nbsp; [Posts by year &rarr;](/year-archive/) &nbsp; &middot; &nbsp; [Posts by tag &rarr;](/tag-archive/)

---

## Side projects

<ul class="taxonomy__index">
{% assign side_projects = site.portfolio | sort: 'date' | reverse %}
{% for project in side_projects %}
  <li class="post-row">
    <time class="post-row__date" datetime="{{ project.date | date_to_xmlschema }}">{{ project.date | date: "%d %b %Y" }}</time>
    <span class="post-row__title"><a class="post-row__link" href="{{ project.url | relative_url }}">{{ project.title }}</a></span>
  </li>
{% endfor %}
</ul>

[All side projects &rarr;](/portfolio/)
