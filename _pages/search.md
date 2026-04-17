---
layout: single
title: "Search"
permalink: /search/
author_profile: false
sitemap: false
---

<link rel="stylesheet" href="/pagefind/pagefind-ui.css">
<div id="search" style="margin-top: 2rem;"></div>
<script src="/pagefind/pagefind-ui.js" onload="
  if (typeof PagefindUI === 'function') {
    new PagefindUI({ element: '#search', showSubResults: true });
  }
" onerror="
  document.getElementById('search').innerHTML =
    '<p class=&quot;notice&quot;>Live search is unavailable here (the index is built on deploy). ' +
    'Browse by <a href=&quot;/tag-archive/&quot;>tag</a> or <a href=&quot;/year-archive/&quot;>year</a> instead.</p>';
"></script>
