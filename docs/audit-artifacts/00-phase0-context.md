# Phase 0 Context — Pre-audit observations passed to all agents

**Date:** 2026-05-24
**Dev server:** http://localhost:4000/ (homebrew Ruby 4.0.3, Jekyll 4.3, Pagefind disabled for audit speed)
**Browser:** Chrome DevTools MCP, page id 13.
**Screenshots dir:** `/var/folders/_m/ktffvd9x2fnc_g9spdkkkrr80000gp/T/audit-screenshots/` (MCP-accessible; will be moved to `docs/audit-artifacts/screenshots/` at end).

## Page inventory — audit ALL of these

- `/` (home)
- `/about/`
- `/cv/`
- `/publications/`
- `/talks/`
- `/teaching/`
- `/portfolio/` (Side projects)
- `/year-archive/`, `/tag-archive/`, `/category-archive/`
- `/search/`
- `/404.html`
- Posts (6): `/visualizing-and-comparing-distributions/`, `/human-activity-recognition/`, `/working-with-numpy/`, `/transportation-graph-nyc-taxi-data/`, `/galactic-morphology-using-deep-learning/`, `/distributed-kmeans-clustering/`
- Portfolio entries (3): `/portfolio/datascience-environment/`, `/portfolio/pythonvsrust-kmeans/`, `/portfolio/simucell3d/`

## Critical pre-finding — share with webdev + designer + hiring agents

**`main.min.js` throws `SyntaxError: Cannot use import statement outside a module` on every page load.**

- Root cause: `assets/js/_main.js:57` contains `import { plotlyDarkLayout, plotlyLightLayout } from './theme.js';` — a top-level ES6 import.
- This was concatenated into `assets/js/main.min.js` during bundling.
- Prior audit (commit `3930ceb`) changed the script tag from `type="module"` to `defer` (issue #2 in deleted issues.md). This broke parsing entirely. Under `defer`, top-level `import` is a syntax error.
- **Cascade impact (verified in browser):**
  - `typeof window.$` === `"undefined"` (jQuery never defined → all jQuery-dependent UI dead: masthead, navigation, smooth scroll, search masthead-toggle, masthead-collapse, social URLs reveal)
  - `theme-cycle.js` runs but its `$(document).ready` shim probably also fails (it depends on jQuery if it uses `$`)
  - Follow button still says "Follow", has no `aria-expanded` attribute, and clicking does nothing
  - Skip-to-content link still missing
- Severity: **P0 regression**. Worse than the original problem.

## Baseline browser snapshot

- 0 broken images
- 9 network requests, all 200
- 1 console error (the SyntaxError above)
- No CSP, no service worker

## Prior audit context — for methodology only

- `git show f8782b2:docs/issues.md` and `git show f8782b2:docs/audit-artifacts/overseer-consolidated.md` contain the prior 21-finding audit.
- Commit `3930ceb` claims to have fixed issues 1–5 but issue #2 (the JS module fix) introduced the cascade above.
- Do not assume any prior fix is correct without re-verifying in the browser.

## Rules of engagement for all agents

1. **One Chrome instance, page id 13.** Use `mcp__chrome-devtools__list_pages` before opening new pages. Reuse the existing page when possible. New pages OK for parallel inspection but close them when done.
2. **Save screenshots only to `/var/folders/_m/ktffvd9x2fnc_g9spdkkkrr80000gp/T/audit-screenshots/<agent-name>/`.**
3. **Write your report to `docs/audit-artifacts/agent-reports/0{N}-{name}.md` only.** Do not touch any other file.
4. **Cap at 25 findings.** Force prioritization.
5. **Use the YAML-fronted finding template from the plan** (every finding needs id, title, category, severity, confidence, effort, evidence, recommendation, fix_snippet, spec_reference OR peer_reference).
6. **Do not fix anything.** Even one-line fixes are out of scope.
7. **Cite peer sites for judgment-based findings** — lilianweng.github.io, huyenchip.com, simonwillison.net, karpathy.ai are the comparison set.
