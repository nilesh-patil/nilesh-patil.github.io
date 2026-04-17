/* ==========================================================================
   Post reader affordances, injected client-side so the Markdown stays clean:
     1. a Copy button on every code block (uses the async Clipboard API, which
        is available on https and on localhost);
     2. a hover-reveal permalink anchor on every section heading that carries an
        id (kramdown auto_ids gives h2/h3 ids).
   Self-contained, loaded `defer`, and a no-op when the elements are absent
   (listing pages, short posts). Styles live in _sass/layout/_ollama.scss.
   ========================================================================== */
(function () {
  var content = document.querySelector('.page__content');
  if (!content) return;

  // ---- 1. Copy buttons on code blocks ------------------------------------
  var blocks = content.querySelectorAll('.highlighter-rouge, figure.highlight');
  Array.prototype.forEach.call(blocks, function (block) {
    var code = block.querySelector('pre code') || block.querySelector('code') || block.querySelector('pre');
    if (!code) return;
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');
    block.appendChild(btn);

    btn.addEventListener('click', function () {
      var text = code.innerText.replace(/\n+$/, '');
      var done = function () {
        btn.textContent = 'Copied';
        btn.classList.add('is-copied');
        setTimeout(function () {
          btn.textContent = 'Copy';
          btn.classList.remove('is-copied');
        }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(text, done); });
      } else {
        fallback(text, done);
      }
    });
  });

  function fallback(text, done) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'absolute';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); done(); } catch (e) { /* give up quietly */ }
    document.body.removeChild(ta);
  }

  // ---- 2. Heading permalink anchors --------------------------------------
  var heads = content.querySelectorAll('h2[id], h3[id]');
  Array.prototype.forEach.call(heads, function (h) {
    var label = h.textContent;
    var a = document.createElement('a');
    a.className = 'heading-anchor';
    a.href = '#' + h.id;
    a.setAttribute('aria-label', 'Permalink to “' + label + '”');
    a.innerHTML = '<span aria-hidden="true">#</span>';
    h.appendChild(a);
  });
})();
