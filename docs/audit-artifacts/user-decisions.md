# User Decisions — Phase 3

Date: 2026-05-24
Captured via AskUserQuestion in batches during the audit.

## Confirmed by user (Batch 1)

### D-001 — X/Twitter handle
**Decision:** `@optimistic_flw` is the canonical handle. **Remove `@ensembledme` from every X/Twitter surface.**
- `_config.yml` `twitter.username` → `optimistic_flw`
- `_config.yml` `author.twitter` → `optimistic_flw`
- `_data/authors.yml` Twitter entry → `optimistic_flw`
- Footer sidebar X link → `https://x.com/optimistic_flw`
- `_config.yml` `social.links` → drop both `https://twitter.com/nilesh-patil` and the duplicate `https://x.com/optimistic_flw` already present; retain only one canonical X URL: `https://x.com/optimistic_flw`
- Resolves N-001, B-03, B-08, B-09, C-013.

### D-002 — Contact method
**Decision:** No email surfaced. Add a single CTA line on the About page (and consider sidebar): **"LinkedIn is the fastest way to reach me."**
- Do NOT add `author.email` or any mailto link.
- Resolves N-003, B-10, C-030.

### D-003 — LinkedIn + Google Scholar headlines
**Decision:** Both already updated to "Head of AI at DreamStreet." Mark the corresponding audit items as verified-by-user.
- Resolves N-004, B-11, B-14.

### D-004 — Team sizes
**Decision:** **Do not disclose team headcount** for Dream11 or DreamStreet on the public site.
- Use scope-language alternative where the audit recommended headcount: e.g. "cross-functional research org spanning ML eng, applied research, and data."
- Resolves N-006, H-03, C-011.

---

## Recommended defaults (Batch 2 — user opted to skip; my recommendations applied with NOT-CONFIRMED flag)

These are recorded in `docs/issues.md` with a `confirmed: false` marker so they can be revisited.

### R-005 — ORCID surfacing
**Default:** Add ORCID (`0000-0002-3438-8571`) to both `social.links` / `sameAs` (JSON-LD) and the sidebar.
- Rationale: ORCID is already public on GitHub, so this is not a privacy expansion. It is the strongest disambiguation anchor for a common name in academic databases.
- Side-action: update the ORCID employment record to add DreamStreet (currently only shows Dream Sports).
- Tied to N-002, B-04, C-029.

### R-006 — Medium bio
**Default:** Update the Medium bio at `nilesh-patil.medium.com` to current positioning. **No commitment to publish.**
- Rationale: 5-minute edit removes the 2020 "data scientist" impression for anyone who clicks through. Avoids the harder content-commitment trade-off.
- Tied to N-005, B-12.

### R-007 — Avatar/logo file rename
**Default:** Rename `/images/ensembledme.{jpg,webp}` → `/images/nilesh-patil.{jpg,webp}` and update all `_config.yml` references.
- Rationale: `@ensembledme` confirmed NOT yours (per D-001). The filename surfaces in every OG image URL. Renaming aligns asset names with confirmed brand identity.
- Optional: set up a redirect from the old paths if any external sites may have hot-linked the file (unlikely but cheap to add).
- Tied to N-007, B-17.

### R-008 — Service / advisory
**Default:** Treat as a genuine gap. Audit adds a developmental recommendation: "Consider taking on one reviewer/advisor role in the next 6 months to create this signal before the next active search." Not blocking.
- Rationale: No service surfaced means either it's absent or undisclosed; either way, the public artifact gap is real.
- Tied to N-008, H-25, H-22.
