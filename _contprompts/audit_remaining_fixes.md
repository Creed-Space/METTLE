# METTLE Audit — Remaining Fixes

**Source**: `1.0 mettle audit website feb '26-compressed.pdf` (14 pages)
**Previous session**: 2026-02-19 — Implemented 19 of 25 audit items
**Status**: 6 items remaining (1 medium, 2 small, 3 trivial)

## Context

An external UX audit was performed on the METTLE website (mettle.sh). The audit PDF lives at the project root. A team of agents mapped all 25 findings to exact source files. The action plan is at `AUDIT_ACTION_PLAN.md`. Per-page reports are in `audit_*_recommendations.md` files.

The site is a FastAPI backend (`main.py`) serving static HTML/CSS/JS from `static/`. There's also a SvelteKit frontend in `frontend/` that mirrors some pages. The dev server runs with:
```bash
cd /Users/nellwatson/Documents/GitHub/METTLE
source .venv/bin/activate
METTLE_SECRET_KEY=dev-secret-key-for-local uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Port 8000 may be in use** — use 8001 or another free port.

## Already Completed (19 items)

- P0-1: `[object Object]` error — fixed difficulty dropdown values (`basic`/`full`) + error extraction in `apiCall()` (`app.js:75-87`)
- P0-2: Sidebar "Help" highlight — fixed scroll-spy to iterate backwards (`docs.html:1488-1507`)
- P1-1: Hero section — reduced `min-height: 88vh→65vh`, split subtitle into `.hero-subtitle` + `.hero-question`
- P1-2: Threat boxes — added `min-height: 140px`, border opacity `0.06→0.15`
- P1-4: Suite numbers/quotes — removed `opacity: 0.5`, used suite-hue color, changed `text-subtle→text-muted`
- P1-5: Badge row — symmetric padding `var(--space-2xl) 0`
- P1-6: Table borders — added column dividers to all docs tables (`docs.css`)
- P1-7: Endpoint spacing — `padding-right: 2rem` on path column
- P1-8: Signing models swap — Notarized now on LEFT in both `docs.html` and Svelte
- P1-9: Callout margins — `margin-top: 1.5rem` on `.callout`
- P1-11: About divider spacing — increased from `0.5rem` to `2.5rem/2rem`
- P1-12: About first divider — deleted unnecessary divider after hero
- 6x Skip: CLI double-dash flags confirmed intentional

## Remaining Fixes

### 1. P1-3: Home Page Divider Spacing (Small)

**Audit finding**: Space between the upper glow-divider and "Design Philosophy" section should match the bottom divider spacing. Currently asymmetric.

**File**: `static/style.css`

**What to do**: Find the `.glow-divider` rule and ensure it has symmetric vertical margins. The dividers on the home page (`index.html` lines 235-239, 298-302, etc.) use the `.glow-divider` class. Set:
```css
.glow-divider {
    margin: var(--space-2xl) auto;  /* symmetric top/bottom */
}
```
Make sure this doesn't conflict with the about page overrides (which were already changed in `about.html`'s inline `<style>`).

**Verify**: Visually compare spacing above and below each glow-divider on the home page.

---

### 2. P1-10: Move Test Verification to Dedicated Page (Medium)

**Audit finding**: The "Test Verification" section is at the bottom of the Home page (`index.html:987-1090`). The auditor suggests it should be on its own page since it already has a "Test" button in the header nav.

**Files to modify**:
1. Create `static/test.html` — new standalone page containing the verification UI
2. `static/index.html` — remove lines 987-1090 (the `<section class="verification-section" id="try-it">` block)
3. `static/index.html:106` — change `<a href="#try-it"` to `<a href="/test"`
4. `static/index.html:137` — same for mobile nav
5. `static/index.html:1114` — update footer link from `/#try-it` to `/test`
6. `main.py` — add route to serve `test.html`:
```python
@app.get("/test", include_in_schema=False)
async def serve_test():
    """Serve the test verification page."""
    if _static_dir.exists():
        return FileResponse(str(_static_dir / "test.html"))
    return RedirectResponse(url="/")
```

**Important**: The new `test.html` needs:
- Full HTML boilerplate (same head as index.html — meta tags, CSS, Font Awesome)
- The header nav (copy from index.html)
- The verification section HTML
- The footer
- Script tags: `app.js` and `webmcp.js`
- The scroll-aware header and mobile menu JS (from bottom of index.html)

**Alternative approach**: Instead of duplicating the header/footer, consider using a shared template approach. But since the site is static HTML, the simplest path is to copy the boilerplate.

---

### 3. P2-1: Credentials Grid Layout Verification (Trivial)

**Audit finding**: All 4 credential tiers (BASIC, AUTONOMOUS, GENUINE, SAFE) should stay in a single row on desktop.

**File**: `static/style.css`

**What to do**: Check if any media query overrides `.credentials-grid { grid-template-columns: repeat(4, 1fr); }` on desktop widths. Search for `credentials-grid` in the CSS and ensure it only collapses to fewer columns on mobile. If it's already correct (likely), just verify visually and mark as done.

---

### 4. P2-4: Alternating Table Row Colors (Trivial)

**File**: `static/docs.css`

**What to add** (after the existing table border rules around line 512):
```css
.docs-content table tbody tr:nth-child(odd) {
    background: rgba(255, 255, 255, 0.02);
}
```

This is a subtle readability enhancement for all tables on the docs page.

---

### 5. P2-5: Console Error Logging (Trivial)

**File**: `static/app.js`

**Current** (line 234-238):
```javascript
function showError(message) {
    stopTimer();
    elements.errorMessage.textContent = message;
    showScreen('error');
}
```

**Add one line**:
```javascript
function showError(message) {
    stopTimer();
    console.error('[METTLE]', message);
    elements.errorMessage.textContent = message;
    showScreen('error');
}
```

---

## Verification Checklist

After implementing all fixes:
- [ ] Home page dividers have equal spacing above and below
- [ ] Test page loads at `/test` with full header/footer
- [ ] Home page "Test" nav link goes to `/test`
- [ ] Mobile nav "Test" link works
- [ ] Footer "Test" link works
- [ ] Credentials grid shows 4 columns on desktop
- [ ] Docs tables have subtle alternating row colors
- [ ] Browser console shows `[METTLE]` prefix on errors
- [ ] All existing functionality still works (start verification, submit answers, view results)
- [ ] Visual spot-check on mobile breakpoints (768px, 480px)

## Audit Report Files (Reference)

- `AUDIT_ACTION_PLAN.md` — master action plan with all 25 items
- `audit_homepage_recommendations.md` — Home page details
- `audit_docs_recommendations.md` — Docs page details
- `audit_test_about_recommendations.md` — Test + About page details
- `audit_bug_investigation.md` — [object Object] bug root cause analysis
- `1.0 mettle audit website feb '26-compressed.pdf` — original audit PDF
