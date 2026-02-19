# METTLE Audit: Test Verification & About Page Recommendations

## Overview

This document maps audit findings from the Test Verification section (on Home page) and About page to specific source code locations with concrete fix recommendations.

---

## TEST VERIFICATION PAGE ITEMS

### Issue 1: [object Object] Error Display

**Severity**: High (affects user experience)

**Symptom**: Clicking "Start Verification" shows "Something went wrong" with "[object Object]" displayed in the error message.

**Root Cause Analysis**:

The error handling in `/Users/nellwatson/Documents/GitHub/METTLE/static/app.js` (lines 262-263) catches exceptions but doesn't properly stringify error objects:

```javascript
// Current code - PROBLEMATIC
catch (error) {
    showError(error.message);
}
```

When the API returns an error response with a `detail` field (like `{"detail": "Some error message"}`), the code passes `error.message` which may be `undefined`, causing the error object itself to be converted to a string as `[object Object]`.

**File**: `/Users/nellwatson/Documents/GitHub/METTLE/static/app.js`

**Line**: 262-263 (in `handleStart` function) and similar code around 309-310 (in `handleSubmit` function)

**Fix Recommendation**:

Replace the error display logic to properly extract the error message from API responses:

```javascript
// In apiCall function (around line 60-80):
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        },
    };

    if (body) {
        options.body = JSON.stringify(body);
    }

    const response = await fetch(`${API_BASE}${endpoint}`, options);
    const data = await response.json();

    if (!response.ok) {
        // FIX: Properly extract error message
        const errorMessage = typeof data.detail === 'string'
            ? data.detail
            : (data.message || 'API request failed');
        throw new Error(errorMessage);
    }

    return data;
}

// In handleStart and handleSubmit - update error handling:
catch (error) {
    // Use error.message which is now properly set
    showError(error instanceof Error ? error.message : String(error));
}
```

**Implementation Steps**:

1. Update the `apiCall` function to ensure errors thrown have `.message` as a plain string (not an object)
2. Update both `handleStart` (line 262) and `handleSubmit` (line 309) to safely extract error messages
3. Test with invalid API responses to verify error display

---

### Issue 2: Test Verification Section Location

**Severity**: Medium (UX/Navigation issue)

**Audit Finding**: The "Test Verification" section (lines 987-1090 in `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html`) is currently at the bottom of the Home page. However, the header menu has a "Test" button that links to `#try-it` (line 106), which targets this same section.

**Current State**:
- Section has `id="try-it"` (line 987) ✓
- Navigation links to this ID properly ✓
- Visual placement: bottom of home page (after all other content)

**Audit Recommendation**: Create a separate dedicated page for Test Verification instead of embedding it on the home page.

**Files Involved**:
1. `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` - Main home page
2. `/Users/nellwatson/Documents/GitHub/METTLE/frontend/src/routes/` - Svelte routes (if using SvelteKit)

**Fix Recommendation**:

If this is a static site architecture:
1. Create a new file: `/Users/nellwatson/Documents/GitHub/METTLE/static/test.html`
2. Move lines 987-1090 from `index.html` to the new `test.html`
3. Update the header navigation (line 106) to point to `/static/test.html` instead of `/#try-it`
4. Include proper header/footer in the new page for consistency

If using SvelteKit:
1. Create a new route: `/Users/nellwatson/Documents/GitHub/METTLE/frontend/src/routes/test/+page.svelte`
2. Move the Test Verification component there
3. Update navigation accordingly

**Current Implementation** (lines 103-116 in index.html):
```html
<a href="#try-it" class="nav-link">Test</a>  <!-- Currently anchors to same page -->
```

**Proposed Change**:
```html
<a href="/test" class="nav-link">Test</a>  <!-- Link to dedicated page -->
```

---

### Issue 3: CLI Flags (--full, --seed)

**Severity**: None - Works as intended

**Audit Finding**: The `--full` and `--seed` flags shown in the CLI preview (lines 763-807) are intentional CLI options and do not require changes.

**Current Code Location**: `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` lines 780, 783

These are valid CLI flags for the METTLE verifier and are functioning correctly.

---

## ABOUT PAGE ITEMS

### Issue 1: First Divider Removal (After Hero)

**Severity**: Low (visual design)

**Finding**: The first decorative divider (lines 373-377) appears immediately after the hero section heading and is visually unnecessary.

**File**: `/Users/nellwatson/Documents/GitHub/METTLE/static/about.html`

**Current Code** (lines 372-377):
```html
</section>

<div class="glow-divider" style="--glow-color: rgba(245, 158, 11, 0.5);" aria-hidden="true">
    <div class="glow-line"></div>
    <div class="glow-dot"></div>
    <div class="glow-line"></div>
</div>
```

**Fix Recommendation**: Delete lines 373-377. The hero section provides sufficient visual separation without an additional divider.

---

### Issue 2: Second Divider Spacing Enhancement

**Severity**: Low (visual refinement)

**Finding**: The divider between "METTLE inverts the question: can you prove you're not human?" (line 392) and "Design Principles" (line 403) needs increased vertical spacing for better visual breathing.

**File**: `/Users/nellwatson/Documents/GitHub/METTLE/static/about.html`

**Current Code** (lines 390-399):
```html
<!-- Pullquote -->
<div class="about-pullquote">
    <p>METTLE inverts the question:<br>can you prove you're <em>not human</em>?</p>
</div>

<div class="glow-divider" style="--glow-color: rgba(245, 158, 11, 0.4);" aria-hidden="true">
    <div class="glow-line"></div>
    <div class="glow-dot"></div>
    <div class="glow-line"></div>
</div>
```

**Current CSS** (lines 104-109):
```css
.about-hero + .glow-divider,
.about-content + .about-pullquote + .glow-divider,
.about-principles + .glow-divider {
    margin: 0.5rem auto;
}
```

**Fix Recommendation**: Increase margin on the divider between pullquote and principles section:

```css
.about-hero + .glow-divider,
.about-content + .about-pullquote + .glow-divider,
.about-principles + .glow-divider {
    margin: 2rem auto;  /* Increased from 0.5rem */
}
```

Alternative approach: Add specific rule for this divider:
```css
.about-pullquote + .glow-divider {
    margin: 2.5rem auto 2rem;  /* More space above than below */
}
```

---

### Issue 3: Footer Divider Spacing

**Severity**: Low (visual refinement)

**Finding**: The divider above "Built by Creed Space" section needs larger vertical spacing for better visual separation.

**File**: `/Users/nellwatson/Documents/GitHub/METTLE/static/about.html`

**Current Code** (lines 429-433):
```html
</div>

<div class="glow-divider" style="--glow-color: rgba(245, 158, 11, 0.4);" aria-hidden="true">
    <div class="glow-line"></div>
    <div class="glow-dot"></div>
    <div class="glow-line"></div>
</div>

<!-- Builder -->
<div class="about-builder">
```

**Current CSS** (lines 104-109): Already covered above

**Fix Recommendation**: Update CSS rule or add specific styling:

```css
/* Option 1: Update existing rule */
.about-principles + .glow-divider {
    margin: 2.5rem auto 2rem;  /* Increased from 0.5rem */
}

/* Option 2: Inline style adjustment */
<div class="glow-divider" style="--glow-color: rgba(245, 158, 11, 0.4); margin: 2.5rem auto 2rem;" aria-hidden="true">
```

---

## SUMMARY TABLE

| Page | Item | Type | File | Line(s) | Severity | Action |
|------|------|------|------|---------|----------|--------|
| Test | [object Object] error | Bug | app.js | 60-80, 262, 309 | High | Fix error message extraction in apiCall() |
| Test | Test section location | UX | index.html | 106, 987-1090 | Medium | Move to dedicated /test page |
| Test | CLI flags | Info | index.html | 780, 783 | None | No change needed |
| About | First divider | Design | about.html | 373-377 | Low | Delete lines |
| About | Second divider spacing | Design | about.html + CSS | 395-399, 106-109 | Low | Increase margin: 0.5rem → 2rem |
| About | Footer divider spacing | Design | about.html + CSS | 429-433, 106-109 | Low | Increase margin: 0.5rem → 2.5rem |

---

## IMPLEMENTATION PRIORITY

1. **High Priority**: Fix [object Object] error display (Issue Test #1)
   - Quick fix, high user impact
   - ~15 minutes

2. **Medium Priority**: Reorganize Test Verification to dedicated page (Issue Test #2)
   - Navigation improvement
   - Depends on site architecture (static vs SvelteKit)
   - ~30-45 minutes

3. **Low Priority**: Visual refinements (About page issues #1-3)
   - Polish and spacing improvements
   - Can be done incrementally
   - ~10 minutes per fix

---

## VALIDATION CHECKLIST

- [ ] Error messages display plain text instead of [object Object]
- [ ] Test page accessible from both home anchor link and dedicated URL
- [ ] First divider on About page removed (visual inspection)
- [ ] Second divider has increased spacing (visual inspection)
- [ ] Footer divider has increased spacing (visual inspection)
- [ ] All responsive breakpoints still work correctly
- [ ] Accessibility attributes preserved (aria-hidden, etc.)

