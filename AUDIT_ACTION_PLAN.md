# METTLE Website Audit — Action Plan

**Source**: `1.0 mettle audit website feb '26-compressed.pdf` (14 pages)
**Compiled**: 2026-02-19
**Total findings**: 25 items across 4 pages (Home, Test, Docs, About)

---

## Priority Summary

| Priority | Count | Description |
|----------|-------|-------------|
| **P0 — Bug** | 2 | Broken functionality, must fix |
| **P1 — UX/Layout** | 12 | Layout, spacing, alignment issues |
| **P2 — Polish** | 5 | Visual refinement, minor cosmetics |
| **Skip** | 6 | No change needed (intentional CLI flags) |

---

## P0 — BUGS (Fix First)

### P0-1: "[object Object]" Error in Test Verification
**Impact**: All verification attempts show cryptic error instead of useful message
**Root cause**: TWO bugs working together:

1. **Difficulty mismatch** — Frontend sends `easy`/`standard`/`hard` but backend expects `basic`/`full`
   - File: `static/index.html:1009-1013`
   - Fix: Change `<option>` values to match backend enum

2. **Error display** — `apiCall()` passes array/object to `new Error()`, producing `[object Object]`
   - File: `static/app.js:76`
   - Fix: Extract `.detail` string or first validation error `.msg` before throwing

```
static/index.html:1009  — Change option values: easy→basic, standard/hard→full
static/app.js:76         — Handle data.detail as string OR array of FastAPI validation errors
static/app.js:262        — Defensive: showError(error instanceof Error ? error.message : String(error))
```

### P0-2: Sidebar "Help" Link Highlight Bug (Docs)
**Impact**: Navigation highlight stays on "Troubleshooting" when scrolled to Help
- File: `static/docs.html:1488-1507`
- Fix: Change scroll-spy to iterate backwards and `break` on first match, or use viewport-center distance

---

## P1 — UX/LAYOUT (High Value)

### P1-1: Hero Section Content Spacing (Home)
Spread content to fill empty space; split subtitle from main question.
- `static/index.html:165-168` — Split `<p class="hero-subtitle">` into two elements
- `static/style.css:226` — Reduce `min-height: 88vh` to `~65vh`
- `static/style.css:277` — Tighten subtitle margin

### P1-2: Four Threats Box Alignment (Home)
First box height doesn't match others; borders barely visible.
- `static/style.css:429-445` — Add `min-height` to `.problem-item`, increase border opacity `0.06→0.15`

### P1-3: Divider Spacing Consistency (Home)
Upper divider has less space than bottom divider.
- `static/style.css` — Set `.glow-divider` margin to `var(--space-2xl) auto` (symmetric)

### P1-4: Suite Numbers/Quotes Visibility (Home)
Numbers (01, 02...) and insight quotes too dim.
- `static/style.css:688` — `.suite-num`: remove `opacity: 0.5`, use suite hue color
- `static/style.css:723` — `.suite-insight`: change from `text-subtle` to `text-muted`

### P1-5: Feature Icons Centering (Home)
Badge row should be vertically centered in its section.
- `static/style.css:1811` — `.badges-section`: make padding symmetric, add `display:flex; align-items:center`

### P1-6: All Tables Need Row Borders (Docs — fixes items 1, 8, 9, 10, 11)
Six different tables across docs page lack row dividers. ONE CSS rule fixes all:
```css
/* Add to static/docs.css after line 510 */
.docs-content table tbody tr { border-bottom: 1px solid var(--border); }
.docs-content table thead th,
.docs-content table tbody td { border-right: 1px solid var(--border); }
.docs-content table thead th:last-child,
.docs-content table tbody td:last-child { border-right: none; }
```

### P1-7: API Endpoint Path/Description Spacing (Docs)
Endpoint path and description text crammed together.
- `static/docs.css:512-514` — Add `padding-right: 2rem` to path column, `padding-left: 1.5rem` to description

### P1-8: Signing Models Box Order (Docs)
Notarized should be on LEFT (recommended option), Self-Signed on RIGHT.
- `static/docs.html:648-674` — Swap the two `<div class="tier-card">` blocks
- `frontend/src/routes/docs/+page.svelte:195-221` — Same swap

### P1-9: Callout Box Spacing (Docs — items 8, 9)
Orange warning boxes under Security/Anti-Gaming need top margin.
- `static/docs.css:530+` — Add `margin-top: 1.5rem` to `.callout`

### P1-10: Test Section → Dedicated Page
Auditor suggests moving Test Verification to its own page.
- `static/index.html:987-1090` — Extract to new `static/test.html`
- `static/index.html:106` — Update nav link from `#try-it` to `/test`
- `main.py` — Add route for `/test`

### P1-11: About Page Divider Spacing
Second divider needs more space; footer divider too tight.
- About page CSS: Change `.glow-divider` margin from `0.5rem` to `2rem auto`

### P1-12: About Page First Divider Removal
First divider after hero is unnecessary.
- `static/about.html:373-377` — Delete the `<div class="glow-divider">` block

---

## P2 — POLISH (Nice to Have)

### P2-1: Credentials Grid Layout (Home)
Ensure all 4 tiers stay in single row on desktop (may already be correct — verify media queries).
- `static/style.css:946-965` — Confirm `grid-template-columns: repeat(4, 1fr)` not overridden

### P2-2: VCP Inline Code (Docs)
`?include_vcp=true` should be in code format in sentence text.
- Already in `<code>` tags (confirmed) — **No change needed**

### P2-3: About Footer Divider Spacing
Space between divider and "Built by Creed Space".
- `static/about.html` + CSS — Increase bottom divider margin to `2.5rem`

### P2-4: Table Alternating Row Colors (Docs)
Optional enhancement for all tables.
```css
.docs-content table tbody tr:nth-child(odd) { background: rgba(255, 255, 255, 0.02); }
```

### P2-5: Error Console Logging (Test)
Add `console.error('[METTLE]', message)` to `showError()` for debugging.
- `static/app.js` — One line addition

---

## SKIP — No Change Needed

| # | Item | Reason |
|---|------|--------|
| 1 | Home: `--notarize` in "Run It Yourself" | CLI flag syntax, correct |
| 2 | Home: `--full` in "How Verification Works" | CLI flag syntax, correct |
| 3 | Docs: `--notarize`, `--full`, `--seed` in Notarization | CLI flag syntax, correct |
| 4 | Docs: `--notarize` in VCP section | CLI flag syntax, correct |
| 5 | Docs: `?include_vcp=true` formatting | Already in `<code>` tags |
| 6 | Test: `--full`, `--seed` in CLI preview | CLI flag syntax, correct |

---

## Implementation Order

**Phase 1 — Bugs** (30 min):
1. Fix difficulty dropdown values (`index.html`)
2. Fix `apiCall()` error extraction (`app.js`)
3. Fix sidebar scroll-spy (`docs.html`)

**Phase 2 — One CSS file, many fixes** (30 min):
4. Add table borders to `docs.css` (fixes 6 tables at once)
5. Add endpoint spacing to `docs.css`
6. Add callout margins to `docs.css`

**Phase 3 — Home page CSS** (45 min):
7. Hero section height + text split
8. Threat boxes alignment + border opacity
9. Divider spacing normalization
10. Suite numbers/quotes visibility
11. Badge row centering

**Phase 4 — HTML restructure** (30 min):
12. Swap Signing Models boxes (docs)
13. Remove About page first divider
14. Increase About page divider spacing

**Phase 5 — Bigger lift** (optional, 1-2 hrs):
15. Extract Test Verification to dedicated page

---

## Files Touched Summary

| File | Changes |
|------|---------|
| `static/app.js` | P0-1: Fix error extraction |
| `static/index.html` | P0-1: Fix difficulty values; P1-1: Hero text split; P1-10: Extract test section |
| `static/style.css` | P1-1 through P1-5: Hero, boxes, dividers, suites, badges |
| `static/docs.css` | P1-6, P1-7, P1-9: Tables, endpoints, callouts |
| `static/docs.html` | P0-2: Scroll-spy; P1-8: Swap signing models |
| `static/about.html` | P1-11, P1-12: Divider spacing/removal |
| `frontend/src/routes/docs/+page.svelte` | P1-8: Swap signing models (Svelte version) |

---

*Detailed per-page reports: `audit_homepage_recommendations.md`, `audit_docs_recommendations.md`, `audit_test_about_recommendations.md`, `audit_bug_investigation.md`*
