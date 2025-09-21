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
    '<p class=&quot;notice--warning&quot;><strong>Search index not built.</strong> ' +
    'Run <code>npx pagefind --site _site</code> after <code>bundle exec jekyll build</code> to enable search.</p>';
"></script>
