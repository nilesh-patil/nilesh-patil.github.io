/* ==========================================================================
   Three-mode theme cycle: light → dark → sepia → light.

   The academicpages bundle (assets/js/main.min.js, source assets/js/_main.js)
   ships a 2-mode toggle that flips between light and dark. This script runs
   AFTER the bundle, unbinds the stock click handler from #theme-toggle, and
   replaces it with a 3-mode cycle. Icon and aria-label update with the
   current mode. Preference persists in localStorage under "theme" -- the
   same key the pre-paint script in _includes/head.html reads.
   ========================================================================== */

(function () {
  var ORDER = ["light", "dark", "sepia"];
  var ICONS = {
    light: "fa-sun",     // showing sun = currently light, click goes to dark
    dark:  "fa-moon",    // showing moon = currently dark, click goes to sepia
    sepia: "fa-book-open" // showing book = currently sepia, click goes to light
  };
  var LABELS = {
    light: "Switch to dark mode",
    dark:  "Switch to sepia (reading) mode",
    sepia: "Switch to light mode"
  };

  function currentTheme() {
    var attr = document.documentElement.getAttribute("data-theme");
    if (attr === "dark" || attr === "sepia") return attr;
    return "light";
  }

  function setTheme(theme) {
    if (theme === "light") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", theme);
    }
    try { localStorage.setItem("theme", theme); } catch (_) {}
    syncIcon();
    syncGiscusTheme(theme);
  }

  function syncIcon() {
    var theme = currentTheme();
    var icon = document.getElementById("theme-icon");
    if (!icon) return;
    icon.classList.remove("fa-sun", "fa-moon", "fa-book-open");
    icon.classList.add(ICONS[theme]);
    var btn = icon.closest("[id='theme-toggle']") || icon.parentElement;
    if (btn && btn.tagName === "A") {
      btn.setAttribute("title", LABELS[theme]);
      btn.setAttribute("aria-label", LABELS[theme]);
    }
  }

  // giscus listens for postMessage to retheme; map our 3 modes to giscus's
  // accepted themes. Giscus has no native "sepia", so use "light" for sepia
  // (cream bg + dark text reads acceptably under the light giscus chrome).
  function syncGiscusTheme(theme) {
    var giscusFrame = document.querySelector("iframe.giscus-frame");
    if (!giscusFrame || !giscusFrame.contentWindow) return;
    var giscusTheme = theme === "dark" ? "dark" : "light";
    giscusFrame.contentWindow.postMessage(
      { giscus: { setConfig: { theme: giscusTheme } } },
      "https://giscus.app"
    );
  }

  function cycle() {
    var i = ORDER.indexOf(currentTheme());
    var next = ORDER[(i + 1) % ORDER.length];
    setTheme(next);
  }

  function attach() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    // Replace the node to drop any stock click handler bound by _main.js
    // ($(document).ready binds toggleTheme on the same #theme-toggle <li>).
    var clone = btn.cloneNode(true);
    btn.parentNode.replaceChild(clone, btn);
    clone.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopImmediatePropagation();
      cycle();
    });
    syncIcon();
  }

  // Use window 'load' rather than DOMContentLoaded -- the academicpages
  // bundle binds its 2-mode toggler on $(document).ready (= DOMContentLoaded),
  // so attaching at the same phase races. window.load fires after all
  // DOMContentLoaded handlers, so by the time we replace the node and bind,
  // the stock handler is already on the old node (which we've discarded).
  if (document.readyState === "complete") {
    attach();
  } else {
    window.addEventListener("load", attach);
  }

  // If OS preference changes and the user hasn't pinned a choice, follow it.
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
      if (!localStorage.getItem("theme")) {
        setTheme(e.matches ? "dark" : "light");
      }
    });
  }
})();
