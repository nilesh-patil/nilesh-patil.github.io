# Designer Audit Report — nilesh-patil.github.io

**Agent:** DESIGNER (01)
**Date:** 2026-05-24
**Scope:** Typography, color contrast (light/dark/sepia), spacing, layout integrity at 360/768/1280, visual brand, iconography, first-impression quality

---

## Designer's verdict

The site has a clean, restrained aesthetic that respects academic convention — white space is generous, the avatar is warm and humanizing, and the three-theme system (light/dark/sepia) is architecturally sound. However, several compounding typographic decisions undermine legibility: the HTML root is set to 18px (not 16px), which makes the modular type scale land at unexpected pixel values. The most consequential symptom is TOC link text that bottoms out at 9.28px — rendered in uppercase with letter-spacing, making it functionally unreadable. The heading hierarchy collapses at h3 (same computed size as body text), and the hero title line-height of 1.0 causes line collision on wrapped headings at narrow viewports. The greedy-nav hamburger persists visibly at desktop widths due to JS failure, creating a ghost UI element that confuses the masthead's intent. The masthead itself grows to 80px on narrow screens while body padding-top remains 70px, allowing the hero image top edge to slide 10px under the fixed bar. Dark and sepia themes are well-executed at the color token level; contrast ratios generally meet WCAG AA, with one confirmed failure in sepia muted text (4.40:1 at small size) and a marginal pass in the footer (4.54:1). Peer sites like simonwillison.net and huyenchip.com demonstrate that a lean personal academic site can maintain a coherent typographic ladder all the way to the smallest visible element — this site needs that ladder rebuilt.

---

## Findings

---

```yaml
id: D-01
title: "TOC link text at 9.28px — unreadable due to double em nesting"
category: Accessibility
severity: P0
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** Measured via `window.getComputedStyle`. The TOC link chain: `.toc__menu` (`_sidebar.scss:138`) sets `font-size: $type-size-6` = `0.75em` on the 18px root = **13.5px**. Then `.toc__menu a` (`_navigation.scss:413`) sets `font-size: $type-size-7` = `0.6875em` relative to the 13.5px parent = **9.28px**. The container also applies `text-transform: uppercase` and `letter-spacing: 1px`, compounding illegibility. WCAG 2.1 Success Criterion 1.4.4 (Resize Text) breaks at this size even at 200% zoom on mobile.

**Why this matters:** 9.28px text is at the biological threshold of legibility for most adults regardless of contrast ratio. The TOC is the primary navigation tool inside long posts (the distributed k-means post has 12 TOC entries). At uppercase 9.28px, it reads as decorative noise rather than interactive navigation. The double-em nesting is an accidental cascade collision: the parent font-size was set in `_sidebar.scss` and the child font-size was set in `_navigation.scss` without coordinating the base context.

**Recommendation:** Change `.toc__menu a` font-size from `$type-size-7` (0.6875em) to a `rem`-based value (e.g., `0.75rem` = 13.5px at 18px root). Do not use `em` for the link font-size since the parent already uses an em-relative size.

**Fix snippet:**
```scss
// _navigation.scss line 413
.toc__menu {
  // ...existing...
  a {
    font-size: 0.75rem; // was $type-size-7 (0.6875em → 9.28px); rem breaks the cascade
  }
}
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/resize-text.html
**Peer reference:** https://simonwillison.net — TOC links use 13–14px with normal letter-spacing

---

```yaml
id: D-02
title: "Masthead 10px taller than body padding-top at narrow viewport — content slides under nav"
category: Design/UX
severity: P1
confidence: HIGH
effort: 1h
agents: [designer]
```

**Evidence:** Screenshot `/var/folders/_m/ktffvd9x2fnc_g9spdkkkrr80000gp/T/audit-screenshots/designer/post-390-mobile.png`. At 500px viewport width, the "Nilesh Patil" site title wraps to two lines (measured: `siteTitle.height = 64px`) making the masthead grow to **80px** actual height, while `body { padding-top: 70px }` remains fixed. The hero image top edge (at `y=70`) sits 10px below the viewport top but 10px inside the masthead footer (`y=80`). Verified with JavaScript: `mastheadActualHeight=80, bodyPaddingTop=70px, gap=10`.

**Why this matters:** The masthead uses `height: fit-content` (not the static `$masthead-height: 70px`). When the site-title wraps, the masthead grows but the body offset does not follow. The top 10px of page content — or the top 10px of a post's hero image — is invisible under the nav bar. On pages without a hero (About, Teaching), the first visible content begins 10px below where readers expect.

**Recommendation:** Either (a) apply `height: $masthead-height` and `overflow: hidden` to the masthead to prevent growth, or (b) set `padding-top` dynamically via `position: sticky + scroll` JS, or (c) add `white-space: nowrap; overflow: hidden; text-overflow: ellipsis` to the site-title link so it never wraps and the masthead stays 70px.

**Fix snippet:**
```scss
// _masthead.scss — option C (simplest, no JS required)
.masthead__menu-item--lg {
  a {
    white-space: nowrap;
    overflow: hidden;
    max-width: 140px;          // Prevents wrapping at 390px; tune as needed
    text-overflow: ellipsis;
    display: inline-block;
  }
}
```

**Spec reference:** https://developer.mozilla.org/en-US/docs/Web/CSS/white-space
**Peer reference:** https://huyenchip.com — site title is constrained, masthead height is stable across all viewports

---

```yaml
id: D-03
title: "h3 font-size equals body text — no visual heading hierarchy below h2"
category: Design/UX
severity: P1
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** Computed: `h3.fontSize = 18px` == `body.fontSize = 18px` (both `$type-size-5 = 1em` on an 18px base). The type scale in `_themes.scss` sets `$type-size-5: 1em`. `_base.scss:42` uses `$type-size-5` for `h3`. At the 18px root, h3 = body text = 18px. Only `font-weight: bold` distinguishes a heading from a paragraph. Verified: `h2_equals_h3 = true` (both 18px in the archive/sidebar context, h2 content is 22.5px but h3 content/post listing is 18px).

**Why this matters:** Weight alone (bold vs normal) is insufficient heading differentiation for skim-reading — the primary use pattern of academic/professional sites. A reader scanning a post or CV for subheadings must read each bold line to determine if it is a heading or an emphasized paragraph. The WCAG principle of "Adaptable" content (SC 1.3.1) requires heading structure to be programmatically determinable, but visual hierarchy reinforces cognitive navigation.

**Recommendation:** Bump h3 to `$type-size-4` (1.25em = 22.5px). The scale already has that slot occupied by h2 on archive pages; use `$type-size-4` only in `.page__content h3` to avoid colliding with the archive item title.

**Fix snippet:**
```scss
// _base.scss
h3 {
  font-size: $type-size-4;   // was $type-size-5 (same as body). 1.25em = 22.5px.
}
// Or, scope to page content only to avoid archive collision:
// _page.scss
.page__content {
  h3 { font-size: $type-size-4; }
}
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/info-and-relationships.html
**Peer reference:** https://lilianweng.github.io/posts/2023-06-23-agent/ — h2/h3 headings have visible size steps, with h3 noticeably larger than body text

---

```yaml
id: D-04
title: "Hero and page title line-height: 1.0 causes line collision on wrapped headings"
category: Design/UX
severity: P1
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** Computed at all viewports: `h1.lineHeight = 28.134px = h1.fontSize = 28.134px` (ratio 1.0). Hero title at `$type-size-2` (31.25px mobile, 43.94px desktop): `lineHeight = 31.248px` (ratio 1.0). The rule is in `_page.scss:49`: `.page__title { line-height: 1 }`. Screenshot `post-390-mobile.png` shows the "Distributed K-Means Clustering in Python" title wrapping across three lines with zero breathing room between them — ascenders and descenders touch.

**Why this matters:** Web typography convention (supported by readability research cited by Butterick's Practical Typography and Google's Material Design spec) sets display/heading text line-height at 1.1–1.3. At ratio 1.0, descenders of one line visually contact ascenders of the next. On the hero image with a dark overlay, this causes the multi-line title to read as a dense ink block rather than as individual words. The problem is most acute for post titles that wrap at 390px.

**Recommendation:** Set `.page__title { line-height: 1.2 }`. For the hero overlay title, use 1.15 (slightly tighter is fine since contrast is not an issue against the dark overlay).

**Fix snippet:**
```scss
// _page.scss
.page__title {
  margin-top: 0;
  line-height: 1.2;   // was 1. Gives 5-6px breathing room between wrapped lines.

  &--overlay {
    line-height: 1.15; // slightly tighter for hero display text, still readable
  }
}
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/text-spacing.html
**Peer reference:** https://karpathy.ai — hero text uses ~1.2 line-height; wrapping is always comfortable

---

```yaml
id: D-05
title: "Ghost hamburger button visible at 1280px desktop due to JS failure"
category: Design/UX
severity: P1
confidence: HIGH
effort: 1h
agents: [designer]
```

**Evidence:** At 1280px viewport, the hamburger button (`.greedy-nav button`) is `display: block`, `position: absolute`, measured at `{x: 1216, y: 9, width: 46, height: 45}`. The full nav link set is simultaneously visible in `.visible-links` (687px wide). The greedy-nav JS never runs (because `main.min.js` throws a SyntaxError — see Phase 0 context). Without JS, the button is never hidden at desktop. Screenshot: `home-1280-light-v2.png` — the hamburger icon is visible at top-right of masthead at 1280px.

**Why this matters:** A hamburger icon at desktop is a strong anti-pattern. Users interpret it as a mobile navigation element that should not exist. The icon physically overlaps with the theme-toggle icon (which ends at `x=705`) at 768px, and at 1280px it sits in an empty rightmost position creating visual asymmetry. It signals a broken interface to any visitor who notices it.

**Recommendation:** As a CSS-only fallback while JS is broken, hide the button at desktop widths: add a media query to `.greedy-nav button { display: none }` above the `$large` breakpoint. The full nav is readable at 1280 without the button.

**Fix snippet:**
```scss
// _navigation.scss — CSS fallback, valid even without JS
.greedy-nav {
  button {
    // existing position: absolute; right: 0; etc.
    @include breakpoint($large) {
      display: none; // JS-managed at this breakpoint; hide via CSS as fallback
    }
  }
}
```

**Peer reference:** https://simonwillison.net — no hamburger visible at desktop; collapses cleanly at mobile

---

```yaml
id: D-06
title: "Nav overflow at 390–600px: all links visible-links, no collapse, items hidden behind hamburger"
category: Design/UX
severity: P1
confidence: HIGH
effort: 1h
agents: [designer]
```

**Evidence:** At 500px viewport (narrow mobile equivalent): `totalNavWidth = 577px` vs `viewportWidth = 500px`. `lastNavItemRight = 591px` exceeds the 500px masthead boundary. `isNavOverflowing = true`. The hamburger button at `x=689` overlaps the theme-toggle icon which ends at `x=705` at 768px. Without JS, the greedy-nav cannot move overflow items to `.hidden-links`, so the "CV" and theme-toggle links are visually clipped or overlapped. Screenshot: `home-390-light.png` shows nav items running off-screen.

**Why this matters:** The mobile nav is the site's primary wayfinding at the most common real-world viewport width (390px = iPhone 14/15). Clipped nav items are invisible and untappable. The greedy-nav pattern depends entirely on JS to function at mobile widths. The P0 JS failure (pre-finding) is the root cause, but a CSS fallback (overflow:hidden on the nav + always-visible hamburger at mobile) would partially restore function.

**Recommendation:** At narrower than `$small` (600px), hide `.visible-links li` (except the site title and hamburger), rely entirely on `.hidden-links` populated server-side, or use a `<details>`/`<summary>` progressive-enhancement approach that works without JS.

**Fix snippet:**
```scss
// _navigation.scss — CSS-only fallback for mobile nav
@media screen and (max-width: 599px) {
  .visible-links li:not(.persist) {
    display: none; // hide all non-pinned items at narrow viewport
  }
  .greedy-nav button {
    display: block; // always show hamburger at mobile
  }
}
```

**Peer reference:** https://huyenchip.com — nav collapses gracefully to hamburger at mobile without JS dependency

---

```yaml
id: D-07
title: "Sepia muted text (#7d6a52 on #f4ecd8) at 4.40:1 — fails WCAG AA for small text"
category: Accessibility
severity: P1
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** Computed contrast ratio: `contrastRatio(125,106,82, 244,236,216) = 4.40:1`. WCAG AA requires **4.5:1** for text below 18px (or bold below 14px). `--global-text-color-light` in sepia is `$text-muted = #7d6a52`. This color is used for: `.page__meta` (post dates, read time), `.archive__subtitle`, `.footnotes`, `.page__footer-follow` links in sepia mode, and TOC text. All render at `$type-size-6` (13.5px) — well below the 18px threshold. Measured in `_default_sepia.scss:27`.

**Why this matters:** Sepia mode is explicitly designed for "long-read paper" sessions (per the file header), meaning readers use it when accessibility matters most: low-light environments, extended reading. The failure at 4.40:1 (0.10 below threshold) on the very elements that orient the reader (dates, TOC text, captions) is a real accessibility regression, not a borderline curiosity. Darkening `$text-muted` by ~10% of luminance would close the gap without disturbing the warm palette.

**Recommendation:** Darken `$text-muted` in sepia from `#7d6a52` to `#6f5d45` (contrast ≈ 5.2:1). The hue is unchanged; only luminance shifts.

**Fix snippet:**
```scss
// _sass/theme/_default_sepia.scss
$text-muted: #6f5d45;   // was #7d6a52 (4.40:1 on sepia bg). 5.2:1 at #6f5d45.
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
**Peer reference:** https://lilianweng.github.io — warm-tinted sepia-like blog uses #5a4a3a-level text on cream backgrounds

---

```yaml
id: D-08
title: "Theme-toggle touch target is 25×36px — below WCAG 2.5.5 minimum of 44×44px"
category: Accessibility
severity: P1
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** `#theme-toggle a` measured: `{width: 25, height: 36}`. The WCAG 2.5.5 (AAA, but referenced as 2.5.8 AA in WCAG 2.2) target size minimum is 24×24px minimum, with 44×44px recommended. At 25px wide the minimum is barely met, but 36px tall falls 8px short of the recommended size. The link uses `display: flex; justify-content: center; width: 25px` (set in `_navigation.scss:230–234`). The `width: 25px` is hardcoded.

**Why this matters:** The theme toggle is one of the most-used controls on a site with three reading modes. It is positioned in the top navigation bar where tap precision is tested (near the browser chrome edge). Mobile users have approximately a 10mm finger contact area (~38px at 96dpi). A 25px wide, 36px tall target produces missed taps and frustration. This is a design issue, not just accessibility.

**Recommendation:** Increase the touch target to at minimum 44×44px. Add `padding` rather than changing visual size, using a negative-margin or `::before`/`::after` pseudo-element to extend the clickable area.

**Fix snippet:**
```scss
// _navigation.scss
.visible-links #theme-toggle a {
  width: 44px;          // was 25px
  min-height: 44px;     // extend touch target
  display: flex;
  align-items: center;
  justify-content: center;
}
```

**Spec reference:** https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html
**Peer reference:** https://simonwillison.net — theme toggle uses icon-button pattern with adequate padding

---

```yaml
id: D-09
title: "RSS icon is brand-colored orange in footer while all other icons are muted gray — visual inconsistency"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** Footer icons measured: GitHub, Medium, LinkedIn, Google Scholar all render at `rgb(108,111,115)` (muted gray) because they use `.fab` or `.fas` class, which is caught by `_footer.scss:50` rule `.page__footer { .fas, .fab { color: var(--global-text-color-light) } }`. The RSS icon uses `.fa` (legacy class), which is NOT caught by that rule, so `.social-icons .fa-rss-square { color: $rss-color = #fa9b39 }` fires instead. Result: RSS = `rgb(250,155,57)` (orange), all others = gray. Screenshot: `portfolio-1280.png` shows the orange FEED icon standing out prominently against the gray row.

**Why this matters:** Inconsistent icon coloring without semantic justification creates visual noise. The RSS icon draws disproportionate attention relative to its actual importance — most readers will not use the feed. The inconsistency signals unintentional implementation rather than deliberate design.

**Recommendation:** Either (a) extend the footer muting rule to also cover `.fa` class, or (b) apply brand colors to all footer icons consistently. Option (a) is simpler.

**Fix snippet:**
```scss
// _footer.scss
.page__footer {
  .fas, .fab, .far, .fal, .fa {   // add .fa to cover legacy RSS icon class
    color: var(--global-text-color-light);
  }
}
```

**Peer reference:** https://simonwillison.net/about/ — footer social icons are uniformly styled (same weight, same color treatment)

---

```yaml
id: D-10
title: "author__content line-height: 1 collapses bio text at mobile sidebar strip"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** `_sidebar.scss:219`: `.author__content { line-height: 1 }`. At viewports below `$large` (925px), the sidebar renders as a horizontal strip with `display: table-cell`. The bio text "Head of AI at DreamStreet. Previously Head of Applied Research, Dream11." wraps to 2 lines at 768px. With `line-height: 1` (= 13.5px on 13.5px font-size), lines are flush — no leading between them. Measured: `authorContent_lineHeight = 18px` (it gets 18px from the outer element since the 13.5px text inside inherits from the cell). The visual cramping is visible in `home-768-correct.png`: the bio text has zero inter-line breathing room inside the sidebar strip.

**Why this matters:** `line-height: 1` is appropriate only for single-line headings. Multi-line descriptive text needs at minimum 1.4–1.5 (matching WCAG 1.4.8 guidance). The author bio is the first thing a new visitor reads about the site owner — it deserves readable presentation at all viewports.

**Recommendation:** Set `.author__content { line-height: 1.4 }`.

**Fix snippet:**
```scss
// _sidebar.scss
.author__content {
  display: table-cell;
  vertical-align: top;
  padding-left: 15px;
  padding-right: 25px;
  line-height: 1.4;   // was 1; allows bio text to breathe at mobile
  // ...
}
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/text-spacing.html
**Peer reference:** https://huyenchip.com/about/ — author bio text at narrow viewports uses comfortable line-height

---

```yaml
id: D-11
title: "Post listing date/excerpt uses link color instead of muted meta color"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** On the home page, each post entry has a `<small>` element inside an `<a>` tag containing the date and excerpt (e.g., "May 20, 2020 · Implementing scalable k-means..."). The `<small>` inherits the link color `rgb(29,98,117)` (#1d6275), same as the post title link above it. Result: the title and its metadata are the same teal-blue, creating a large homogeneous blue block per post item. Measured: `small.color = rgb(29,98,117)` on all post entries.

**Why this matters:** Date and excerpt text is metadata, not primary navigation. Peer sites universally use a lighter/muted treatment (gray, italic, smaller) to visually separate primary content (title) from contextual metadata (date, description). The current treatment provides no visual hierarchy within each post card — readers cannot quickly distinguish "what to click" from "when it was written."

**Recommendation:** Apply `--global-text-color-light` to the `<small>` date/excerpt text specifically. In the `_home.html` include or the archive layout, style the small metadata tag separately.

**Fix snippet:**
```css
/* Add to _archive.scss or _page.scss */
.list__item a small,
.archive a small {
  color: var(--global-text-color-light);
}
```

**Peer reference:** https://simonwillison.net — post listings use gray/muted date text distinctly separate from blue title links

---

```yaml
id: D-12
title: "h4/h5/h6 all resolve to 13.5px — no hierarchy below h3"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** `_base.scss:46–56`: `h4 { font-size: $type-size-6 }`, `h5 { font-size: $type-size-6 }`, `h6 { font-size: $type-size-6 }`. `$type-size-6 = 0.75em` on 18px root = **13.5px**. All three levels render identically, and at 13.5px they are smaller than body text (18px). A heading smaller than the body text it introduces is a typographic inversion.

**Why this matters:** The CV page and publications page use multi-level content hierarchies. When a user exports or reads the CV, sub-section headings (h4: institution, h5: details) visually disappear into the text body. The 13.5px inverted heading is smaller than every `<p>` on the page — it communicates subordination rather than structure. CV pages on peer sites like huyenchip.com maintain a visible step at h3, h4 by using size AND color or weight variation.

**Recommendation:** Set h4 to `$type-size-5` (1em = 18px, matching body, differentiated by weight), h5 to `$type-size-6` (13.5px, but only for fine-grained sub-structure), and add `text-transform: uppercase` and `letter-spacing` to h5/h6 to compensate for the small size.

**Fix snippet:**
```scss
// _base.scss
h4 {
  font-size: $type-size-5; // was $type-size-6 (13.5px < body). Now 18px = body, bold.
}
h5 {
  font-size: $type-size-6; // keep small but add textual differentiation
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
```

**Peer reference:** https://huyenchip.com/about/ — h3/h4 headings maintain visible size reduction ladder without inverting below body size

---

```yaml
id: D-13
title: "TOC text-transform: uppercase + letter-spacing: 1px at 9.28px compounds unreadability"
category: Accessibility
severity: P1
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** `.toc` in `_navigation.scss:385–388` applies `text-transform: uppercase; letter-spacing: 1px` to the entire TOC container. The TOC links inherit this and render at **9.28px uppercase with 1px letter-spacing**. Uppercase text at this size requires significantly higher contrast than the WCAG standard (which already requires 4.5:1 for small text). The combination makes headings like "CHALLENGES WITH LARGE-SCALE DATA" effectively invisible in peripheral vision. (See also D-01 which is the font-size root cause.)

**Why this matters:** Uppercase + letter-spacing is a stylistic pattern for labels and badges, not for navigational text. It was designed for 11–13px usage, but when the underlying font-size shrinks via cascade (D-01), the uppercase forces all text to cap-height only with no descenders, reducing the shape recognition that aids reading speed. Fixing D-01 (font-size) is the primary fix; this finding documents the compounding stylistic layer.

**Recommendation:** After fixing D-01 font-size: remove `text-transform: uppercase` from `.toc` and from `.toc__menu a`. The "CONTENTS" header label can keep uppercase; individual TOC entries should not.

**Fix snippet:**
```scss
// _navigation.scss
.toc {
  // remove: text-transform: uppercase;
  // remove: letter-spacing: 1px;
  font-family: $sans-serif-narrow;
  color: var(--global-text-color-light);
}
.toc__menu a {
  text-transform: none;   // override inherited uppercase from .toc
  letter-spacing: normal;
}
.toc .nav__title {
  text-transform: uppercase; // keep only on the "CONTENTS" label
  letter-spacing: 1px;
}
```

**Peer reference:** https://lilianweng.github.io/posts/2023-06-23-agent/ — TOC uses sentence case, normal letter-spacing, readable at any size

---

```yaml
id: D-14
title: "Footer text at marginally-passing contrast (4.54:1) with small 13.5px text in light mode"
category: Accessibility
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** Footer `--global-text-color-light = #6c6f73` on `--global-footer-bg-color = #f2f3f3`. Computed ratio: `contrastRatio(108,111,115, 242,243,243) = 4.54:1`. WCAG AA threshold is 4.5:1 for small text. Margin: **0.04:1** above the threshold. Footer text is `$type-size-6 = 13.5px`, uppercase, bold. At this margin, any subpixel rendering variation, CSS rounding, or OS font-smoothing difference can push it below threshold.

**Why this matters:** A 0.04:1 margin is engineering tolerance, not a design decision. The footer contains the "FOLLOW:" links — functionally important for the site's social presence. The `#6c6f73` value was set as a "darkened" version (per comments in `_default_light.scss`) to fix the previous failure of `#bdc1c4` (1.8:1). It was darkened enough to pass, but barely. Darkening to `#5e6166` achieves 5.5:1 with negligible visual difference.

**Recommendation:** Change `--global-text-color-light` from `#6c6f73` to `#5e6166` in `_default_light.scss`.

**Fix snippet:**
```scss
// _sass/theme/_default_light.scss
--global-text-color-light: #5e6166;   // was #6c6f73 (4.54:1 → margin). 5.5:1.
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
**Peer reference:** https://simonwillison.net/about/ — footer text maintained well above 5:1 on light background

---

```yaml
id: D-15
title: "Avatar image uses non-circular border in sidebar desktop view — padding inconsistency"
category: Design/UX
severity: P2
confidence: MED
effort: 15m
agents: [designer]
```

**Evidence:** `_sidebar.scss:208–211`: at `$large` breakpoint, the avatar `img` gets `padding: 5px; border: 1px solid var(--global-border-color)`. With `border-radius: 50%` on the img, the visible circle is reduced by the 5px padding from all sides, creating a 10px gap between the circular photo and the 1px border ring. At desktop sidebar width (175px rendered), the photo is 165px diameter with a loose circular border ring. This is inconsistent with mobile where the avatar shows edge-to-edge circular crop with no border.

**Why this matters:** The avatar is the primary brand element — the human face that anchors the sidebar. The 5px gap + circular border creates an old-fashioned photo-frame effect that is inconsistent with contemporary academic site design. Peer sites (simonwillison.net, karpathy.ai) use clean edge-to-edge circular avatars with minimal or no border ring.

**Recommendation:** Remove the `padding: 5px` or reduce to 2px. If a ring is desired for definition against light backgrounds, use `box-shadow: 0 0 0 1px var(--global-border-color)` without padding, which preserves the full-bleed circular crop.

**Fix snippet:**
```scss
// _sidebar.scss
.author__avatar {
  img {
    max-width: 175px;
    border-radius: 50%;

    @include breakpoint($large) {
      // was: padding: 5px; border: 1px solid var(--global-border-color);
      box-shadow: 0 0 0 1px var(--global-border-color); // ring without gap
    }
  }
}
```

**Peer reference:** https://karpathy.ai — avatar is full-bleed circular crop with no internal padding gap

---

```yaml
id: D-16
title: "Hardcoded background-color: #fff on .toc in _navigation.scss bleeds into dark/sepia override"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** `_navigation.scss:388`: `.toc { background-color: #fff }`. This is currently overridden by `_sidebar.scss` which applies `.toc { background-color: var(--global-toc-bg-color) }`. The `--global-toc-bg-color` CSS custom property is not defined in any theme file, resolving to empty (transparent). So in practice the TOC is transparent in dark/sepia mode and the body background shows through — which looks acceptable. However, the residual hardcoded `#fff` is a latent bug: if `_navigation.scss` compiles AFTER `_sidebar.scss`, the white background returns in dark mode. The current state is correct by accident of compilation order, not by intent.

**Why this matters:** The `_navigation.scss` rule is a maintenance trap. Any recompilation order change, SCSS import reorder, or Sass upgrade could activate the `#fff` override and produce a white TOC box on a dark background. The fix requires either removing the hardcoded color or defining `--global-toc-bg-color` in the theme files.

**Recommendation:** Replace `background-color: #fff` in `_navigation.scss` with `background-color: var(--global-toc-bg-color, var(--global-bg-color))`. Then optionally define `--global-toc-bg-color` in each theme if a distinct surface color is desired (e.g., `$bg-elevated` in dark mode).

**Fix snippet:**
```scss
// _navigation.scss
.toc {
  background-color: var(--global-toc-bg-color, var(--global-bg-color)); // was #fff
}
// Optionally in _default_dark.scss:
// --global-toc-bg-color: #{$bg-elevated}; // #262b33 — slightly elevated surface
```

**Peer reference:** https://lilianweng.github.io — TOC background adapts to dark mode using CSS custom properties

---

```yaml
id: D-17
title: "Post hero title at $type-size-1 (43.9px) with line-height: 1 and dark overlay is not WCAG tested on all images"
category: Accessibility
severity: P2
confidence: MED
effort: 1h
agents: [designer]
```

**Evidence:** The post hero overlay applies `background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5))` over the header image. The `rgba(0,0,0,0.5)` overlay on an arbitrary image background is not guaranteed to produce sufficient contrast. Computed for the distributed k-means post: `text(#fff) on overlay(rgb(0,0,0) at 50% alpha) = depends on image luminance beneath`. The overlay is 50% opacity black — if the image has a light area (e.g., sky, whiteboard, snow), the effective background luminance may be too high for white text. Measured hero title color: `rgb(255,255,255)` on `rgba(0,0,0,0.5)` overlay. At 50% opacity black on a mid-gray image region, the blended color is approximately `rgba(0,0,0,0.5)` on the image pixel.

**Why this matters:** Any post with a bright header image (light colors in the image) will produce sub-4.5:1 contrast for the white hero title. The current distributed k-means post (dark background image of data visualization) is fine. Future posts with lighter images will fail. The shadow text-shadow `1px 1px 4px rgba(0,0,0,0.5)` is insufficient as a substitute for adequate background contrast.

**Recommendation:** Increase overlay opacity to `0.65` (consistent with WCAG guidance on overlay coverage). Alternatively, add a fixed dark gradient band across the bottom 60% of the hero where text appears.

**Fix snippet:**
```scss
// _page.scss
.page__hero--overlay {
  // ...existing...
  background-image: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65));  // was 0.5
  // Or for bottom-weighted gradient:
  // linear-gradient(to bottom, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.75) 100%)
}
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
**Peer reference:** https://lilianweng.github.io — hero overlays use darker gradient bands to ensure text contrast

---

```yaml
id: D-18
title: "Site-brand anchor in masthead uses dark-gray color, not a distinct brand color — weak visual identity"
category: Design/UX
severity: P2
confidence: MED
effort: 15m
agents: [designer]
```

**Evidence:** `.masthead__menu-item--lg a { color: var(--global-masthead-link-color) = rgb(73,78,82) }` — identical to all other nav link colors. The site title "Nilesh Patil" has `font-weight: 700` (bold) which provides minor distinction but no color differentiation. The `$primary-color: #2f7f93` (teal) is the defined brand color but is not applied to the site title. Measured: `siteTitle.color = rgb(73,78,82)`, `siteTitle.fontWeight = 700`.

**Why this matters:** The site title/logo anchor is the brand touchpoint. Academic peer sites use either a distinct color (huyenchip.com uses the accent color), bold typographic scale, or a logo mark to create a distinct brand anchor in the masthead. When the site title is the same color as "About" and "Publications", there is no visual signal that it is the "home" identifier. This is a judgment-based finding relative to peer site comparisons.

**Recommendation:** Apply `var(--global-base-color)` (the teal `$primary-color`) to the site title link, making it the only masthead element in the brand accent color. This echoes the `--global-link-color` used throughout and creates coherent identity.

**Fix snippet:**
```scss
// _masthead.scss
.masthead__menu-item--lg a {
  color: var(--global-base-color);   // was var(--global-masthead-link-color)
  &:hover {
    color: var(--global-masthead-link-color-hover);
  }
}
```

**Peer reference:** https://huyenchip.com — the site name "Chip Huyen" renders in the site's accent color, clearly distinguished from navigation links

---

```yaml
id: D-19
title: "Publications archive items have no visual separator — 45px gap is insufficient for dense list"
category: Design/UX
severity: P2
confidence: MED
effort: 15m
agents: [designer]
```

**Evidence:** Publications page: `archiveItemGap = 45px` between bottom of one publication and top of the next. Each publication has: h2 title (~22.5px bold), "Published in..." line (18px), citation block (~2 lines at 18px), "Download Paper" link (18px). Total block height per entry: ~100–130px. The 45px gap is 35–45% of entry height — visually the items bleed into each other on scroll. There is no horizontal rule, border, or alternating background to delineate entries. Screenshot: `publications-1280.png`.

**Why this matters:** Publication lists are scanned, not read linearly. The cognitive task is "find this paper" — which requires clear entry boundaries. A 45px gap without a visual separator relies entirely on the reader parsing indentation and size changes to detect entry boundaries. Adding a border-bottom or background-tinted alternate rows reduces reading load. Peer academic sites with long publication lists (lilianweng.github.io) use top-border separators between entries.

**Recommendation:** Add a `border-top: 1px solid var(--global-border-color)` and `padding-top: 1.5em` to `.archive__item + .archive__item` (sibling selector) for the publications layout.

**Fix snippet:**
```scss
// _archive.scss or publications layout
.archive__item + .archive__item {
  border-top: 1px solid var(--global-border-color);
  padding-top: 1.5em;
  margin-top: 0.5em;
}
```

**Peer reference:** https://lilianweng.github.io/publications/ — paper entries separated by horizontal rules

---

```yaml
id: D-20
title: "No focus-ring visible for keyboard navigation — %tab-focus mixin uses outline that may be hidden"
category: Accessibility
severity: P2
confidence: MED
effort: 1h
agents: [designer]
```

**Evidence:** `_mixins.scss:5–11`: `%tab-focus { outline: thin dotted $warning-color; outline: 5px auto $warning-color; outline-offset: -2px }`. `$warning-color = #f89406` in light mode. The `outline-offset: -2px` draws the outline inside the element boundary — for inline text links, this is often obscured by the element's own background. Additionally, `:focus` rings are suppressed for mouse users via `:focus:not(:focus-visible)` if such a rule exists, but modern Chrome shows focus rings on keyboard tab only. The `%tab-focus` extend is applied to `a:focus` — but only if `a` has `@extend %tab-focus`. The `_base.scss:117–126` shows `a { &:focus { @extend %tab-focus } }` which is correct, but the `outline-offset: -2px` (drawing inside) combined with the link underline makes the ring nearly invisible on link text.

**Why this matters:** WCAG 2.4.7 (Focus Visible) requires that keyboard focus indicators be visible. A focus ring drawn inside (-2px offset) on underlined text with a 5px orange auto outline may meet the technical requirement but fails the "perceivable" intent. For a site where the primary audience includes technical readers using keyboard navigation (researchers, engineers), visible focus rings are essential.

**Recommendation:** Change `outline-offset: -2px` to `outline-offset: 3px` and ensure `outline-width` is at minimum 3px.

**Fix snippet:**
```scss
// _include/_mixins.scss
%tab-focus {
  outline: 3px solid $warning-color;
  outline-offset: 3px;  // was -2px (inside = hidden on text links)
}
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/focus-visible.html
**Peer reference:** https://simonwillison.net — focus rings are clearly visible at 3px with positive offset

---

```yaml
id: D-21
title: "Greedy-nav nav link hover indicator (scaleX underline) invisible without JS context"
category: Design/UX
severity: P2
confidence: HIGH
effort: 30m
agents: [designer]
```

**Evidence:** `_navigation.scss:240–259`: nav links have a `::before` pseudo-element with `transform: scaleX(0)` (hidden) that reveals on `hover` with `scaleX(1)`. This creates a sliding underline hover effect. Without the greedy-nav JS, the selected item uses the static CSS rule from `_masthead.scss:76–85`: `.masthead__menu-item.selected a { border-bottom: 2px solid }`. Both the hover underline (pseudo-element) and the active indicator (border-bottom) are visible simultaneously on the selected item when hovering — the 2px border-bottom + 4px pseudo-element underline (positioned at `bottom: 0`) may stack or collide. Verified: selected item has both `border-bottom: 2px solid` AND the `::before` hover overlay active on hover.

**Why this matters:** Two simultaneous underline signals on the selected nav item create visual redundancy and potential misalignment. At desktop, the 4px `::before` element (pseudo, `background: var(--global-border-color)`) and the 2px `border-bottom` (text color) overlap at the bottom of the nav link, producing a double-line artifact on hover.

**Recommendation:** On `.masthead__menu-item.selected a`, suppress the `::before` hover animation since the selected state is already indicated by the border-bottom.

**Fix snippet:**
```scss
// _masthead.scss
.masthead__menu-item.selected a::before {
  display: none;  // suppress hover pseudo-underline on already-selected item
}
```

**Peer reference:** https://simonwillison.net — selected nav item has a single consistent indicator, no doubling on hover

---

```yaml
id: D-22
title: "Sepia theme masthead does not use sepia background — creates color mismatch on scroll"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** In sepia mode: `mastheadBg = rgb(255,255,255)` (white) while `bodyBg = rgb(244,236,216)` (#f4ecd8, warm cream). The masthead uses `background: var(--global-bg-color)` per `_masthead.scss:7`. In sepia, `--global-bg-color = #f4ecd8`. So by CSS variable the masthead SHOULD be sepia-tinted. However, measured `mastheadBg` was white (`rgb(255,255,255)`). This was measured on the home page during sepia mode evaluation where the page had just navigated. This suggests the theme attribute had not fully propagated, or the masthead is reading from a cached value.

**Evidence clarification:** The computed value was taken mid-navigation. A second check on a stable sepia page (`home-1280-sepia.png`) shows the masthead background visually matching the body sepia color. This finding is LOW confidence — the visual may be correct.

**Recommendation:** Verify by loading a fresh page in sepia mode and measuring `getComputedStyle(masthead).backgroundColor`. If it returns `rgb(244,236,216)`, this finding is a false alarm and can be closed.

**Fix snippet:** N/A — if confirmed as a false alarm, no fix needed.

**Peer reference:** N/A (contingent finding)

---

```yaml
id: D-23
title: "Code block syntax font-size at 12.66px (below 14px) with Solarized comment tokens at 3.51:1 in dark mode"
category: Accessibility
severity: P2
confidence: HIGH
effort: 1h
agents: [designer]
```

**Evidence:** Code block: `div.highlighter-rouge { font-size: $type-size-4 }` (`_syntax.scss:13`) = 1.25em on 18px = **22.5px** for the container. Inside: `.highlight { font-size: $type-size-6 }` (`_syntax.scss:32`) = 0.75em on 22.5px = **16.875px**. Then `code` inside `.highlight` is the monospace text — measured at **12.66px** (likely a browser-level em cascade rounding from the `pre code` structure). In dark mode, `.highlight .c` (comment tokens) are `#5e7474`. Measured: `contrastRatio(94,116,116, 22,26,32) = 3.51:1` — below the 4.5:1 threshold for small text at 12.66px.

**Why this matters:** Code comments are arguably the most important tokens to read — they explain intent. Comment contrast at 3.51:1 on the `#161a20` dark code background at 12.66px is an accessibility failure. WCAG 2.1 SC 1.4.3 applies to all text, including code. The `#5e7474` value was already darkened (per code comment in `_syntax.scss:57`) from `#93a1a1` for light mode, but that adjustment was not validated against the dark mode code background.

**Recommendation:** For dark mode, dark-mode-specific syntax highlighting is needed. Either (a) add a dark Solarized palette override under `html[data-theme="dark"]`, or (b) darken comment tokens specifically: `#5e7474` → `#7a9c9c` for better contrast on `#161a20`.

**Fix snippet:**
```scss
// _syntax.scss — add dark mode override block
html[data-theme="dark"] {
  .highlight .c,
  .highlight .c1,
  .highlight .cm { color: #7a9c9c; }  // contrast ~4.7:1 on #161a20
}
```

**Spec reference:** https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
**Peer reference:** https://lilianweng.github.io — dark mode uses a full dark-specific syntax theme, not a light-mode palette with darkened tokens

---

```yaml
id: D-24
title: "Active nav indicator (border-bottom: 2px) is identical color to surrounding text — low perceptibility"
category: Design/UX
severity: P2
confidence: HIGH
effort: 15m
agents: [designer]
```

**Evidence:** `_masthead.scss:77–80`: `.masthead__menu-item.selected a { color: var(--global-text-color); font-weight: 700; border-bottom: 2px solid var(--global-text-color) }`. Both the selected link text and its indicator underline are `var(--global-text-color) = rgb(73,78,82)`. At light mode: dark gray text + dark gray 2px underline. The indicator relies solely on the text weight (bold) and the 2px bottom line, both in the same color as adjacent non-selected nav items (which are also dark gray, just not bold). Screenshot: `publications-1280.png` — "Publications" appears bold with an underline, but the distinction from "About" (not bold, no underline) is subtle given both are `rgb(73,78,82)`.

**Why this matters:** Navigation active-state indicators serve critical orientation — "where am I?" — especially for a multi-section academic site. Using the same gray as nav text for the underline, against a white masthead, produces a ~4.5px visual artifact. Peer sites use the brand accent color for the active indicator (the teal `$primary-color` would provide high contrast against the nav text and the white masthead background).

**Recommendation:** Change the active indicator border color to `var(--global-base-color)` (the teal primary color). This creates a clear, color-coded "you are here" signal that uses the established brand palette.

**Fix snippet:**
```scss
// _masthead.scss
.masthead__menu-item.selected a {
  color: var(--global-text-color);
  font-weight: 700;
  border-bottom: 2px solid var(--global-base-color);  // was global-text-color; use brand accent
}
.masthead__menu-item.selected a:hover {
  border-bottom: 2px solid var(--global-base-color);  // consistent
}
```

**Peer reference:** https://huyenchip.com — active nav item underline uses the site accent color, clearly distinct from the text color

---

```yaml
id: D-25
title: "Teaching and Talks pages are visually empty — no placeholder or stub content; layout feels broken"
category: Design/UX
severity: P2
confidence: HIGH
effort: 1h
agents: [designer]
```

**Evidence:** Screenshots `teaching-1280.png` and `talks-1280.png` show: masthead, sidebar with avatar, one `<h1>` ("Teaching" / "Talks and presentations"), then nothing except the footer. Main area height: 526px with essentially one element. The pages are in the primary navigation and listed in the `_config.yml` site nav. Visitors who click them see a blank page that suggests a broken site, not a planned empty section.

**Why this matters:** Blank stub pages that are linked from the primary navigation damage first impressions and site credibility. This is especially relevant for an academic/professional personal site where a hiring manager or collaborator clicking "Teaching" expects course history. An empty page without even a placeholder ("Coming soon" or "No listed courses yet") signals abandonment or neglect. The visual design has no mechanism for graceful empty states.

**Recommendation:** Add a `.page__content` stub for empty-state pages. Either: (a) remove the nav links to empty pages, or (b) add a styled empty-state notice in the page markdown files.

**Fix snippet:**
```markdown
<!-- _pages/teaching.md — add stub content -->
---
layout: single
title: "Teaching"
---

No listed courses yet. Check back later.
```

**Peer reference:** https://huyenchip.com — all nav-linked sections have at minimum a paragraph of content or a clear "in progress" notice; no blank stubs are linked from primary nav
```
