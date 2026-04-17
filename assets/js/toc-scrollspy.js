/* ==========================================================================
   TOC scrollspy — highlight the current section in the right-rail TOC as the
   reader scrolls. Self-contained, no dependencies, and a no-op when the page
   has no TOC (short posts, listing pages). Loaded `defer` so the DOM is parsed
   before it runs. The "current" section is the last heading whose top has
   crossed a read-line ~130px below the masthead — deterministic, no lag. NN/g:
   a reading-progress bar is intentionally NOT added (redundant with the native
   scrollbar); this just marks "you are here".
   ========================================================================== */
(function () {
  var READ_LINE = 130; // px from viewport top (just below the 70px masthead)

  var entries = Array.prototype.slice
    .call(document.querySelectorAll('.toc__menu a[href^="#"]'))
    .map(function (a) {
      var id = decodeURIComponent(a.getAttribute('href').slice(1));
      return { a: a, el: id && document.getElementById(id) };
    })
    .filter(function (e) { return e.el; });
  if (!entries.length) return;

  var current = null;
  function setActive(a) {
    if (a === current) return;
    if (current) current.classList.remove('is-active');
    if (a) a.classList.add('is-active');
    current = a || null;
  }

  function update() {
    var pick = null;
    for (var i = 0; i < entries.length; i++) {
      if (entries[i].el.getBoundingClientRect().top <= READ_LINE) pick = entries[i].a;
      else break;
    }
    setActive(pick); // null before the first heading is reached
  }

  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () { update(); ticking = false; });
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  update();
})();
