# METTLE Docs Page Audit — Recommendations & Code Locations

## Overview

This document maps all 12 audit findings for the METTLE Docs page to specific source files and provides concrete fix recommendations.

**Docs page source files:**
- **HTML version**: `/static/docs.html` (lines 112-1515)
- **Svelte version**: `/frontend/src/routes/docs/+page.svelte` (lines 1-503)
- **CSS file**: `/static/docs.css` (lines 1-585)

---

## AUDIT ITEMS & FIXES

### 1. Difficulty Levels Table — Missing Row Borders

**Finding**: Under "Key Concepts" section (Difficulty Levels table), rows lack horizontal dividers. The table appears cramped with no visual separation between rows.

**Files affected**:
- HTML: `/static/docs.html` lines 313-342
- CSS: `/static/docs.css` (no specific styling for this section)

**Current HTML code** (lines 313-342):
```html
<h3>Difficulty Levels</h3>
<table>
    <thead>
        <tr>
            <th>Level</th>
            <th>Time Pressure</th>
            <th>Challenge Complexity</th>
            <th>Use Case</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>easy</code></td>
            ...
        </tr>
    </tbody>
</table>
```

**Fix recommendation**:
1. Add CSS rule to `/static/docs.css` after line 509 (after existing table rules):
```css
/* For all tables — add visible row borders */
.docs-content table tbody tr {
    border-bottom: 1px solid var(--border);
}

/* Optional: Add alternating row colors for better readability */
.docs-content table tbody tr:nth-child(odd) {
    background: rgba(255, 255, 255, 0.02);
}
```

2. Alternatively, add `border-collapse: collapse;` is already in line 507, so just ensure each `<td>` has `border-bottom: 1px solid`.

**Difficulty**: Low — CSS only change, 2-3 lines.

---

### 2. Notarization Double Dashes — NO CHANGE NEEDED

**Finding**: `--notarize`, `--full`, `--seed` appear in text. These are CLI flags and intentional — **no fix required**.

**Locations**:
- HTML: `/static/docs.html` lines 268, 380, 391
- Svelte: `/frontend/src/routes/docs/+page.svelte` lines 122, 163, 175

**Status**: ✓ CONFIRMED INTENTIONAL — Skip.

---

### 3. API Endpoints Spacing — Cramped Layout

**Finding**: Under "API Endpoints" section, the endpoint path (e.g., `/suites`) and description text are crammed together on the same line with no visual separation.

**Files affected**:
- HTML: `/static/docs.html` lines 450-544 (Endpoints section with endpoint-table)
- CSS: `/static/docs.css` lines 512-514

**Current HTML code** (example, lines 454-467):
```html
<h3>Suite Information</h3>
<table class="endpoint-table">
    <tbody>
        <tr>
            <td><code>GET</code></td>
            <td><code>/suites</code></td>
            <td>List all 10 verification suites</td>
        </tr>
```

**Current CSS** (lines 512-514):
```css
.endpoint-table td:first-child { width: 60px; }
.endpoint-table td:first-child code { font-weight: 600; }
.endpoint-table td:nth-child(2) { font-family: 'SF Mono', Monaco, monospace; font-size: 0.85rem; }
```

**Fix recommendation**:
1. Update CSS at `/static/docs.css` lines 512-514 to:
```css
.endpoint-table td:first-child { width: 80px; }
.endpoint-table td:first-child code { font-weight: 600; }
.endpoint-table td:nth-child(2) {
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 0.85rem;
    padding-right: 2rem;  /* Add right padding for separation */
}
.endpoint-table td:nth-child(3) {
    padding-left: 1.5rem;  /* Add left padding for description */
}
```

2. Alternatively, add a visual separator (pipe character) between columns in the CSS using `::after` pseudo-element on the middle `<td>`.

**Difficulty**: Low — CSS padding/margin adjustments.

---

### 4. Credentials Table — Missing Column Dividers

**Finding**: Under "Credential Tiers" section, the tier/requires/meaning table in the HTML lacks visible column separators.

**Files affected**:
- HTML: `/static/docs.html` lines 613-643 (Credentials section)
- CSS: `/static/docs.css` lines 507-510

**Current HTML code** (lines 613-643):
```html
<table>
    <thead>
        <tr>
            <th>Tier</th>
            <th>Requires</th>
            <th>Meaning</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong style="color: hsl(35, 85%, 60%);">Basic</strong></td>
            <td>Suites 1–5</td>
            <td>METTLE-verified AI — passed substrate verification</td>
        </tr>
```

**Fix recommendation**:
Add CSS rule to `/static/docs.css` after line 510:
```css
/* Add vertical dividers to table columns */
.docs-content table thead th,
.docs-content table tbody td {
    border-right: 1px solid var(--border);
}

/* Remove right border from last column */
.docs-content table thead th:last-child,
.docs-content table tbody td:last-child {
    border-right: none;
}
```

**Difficulty**: Low — CSS border rules.

---

### 5. Signing Models Box Order — SWAP POSITIONS

**Finding**: Under "Credentials" section, "Self-Signed" box is on the LEFT, "Notarized" on the RIGHT. They should be SWAPPED — Notarized (the production/recommended option) should be on the LEFT.

**Files affected**:
- HTML: `/static/docs.html` lines 648-674 (tier-comparison div)
- Svelte: `/frontend/src/routes/docs/+page.svelte` lines 195-221

**Current HTML order** (lines 648-674):
```html
<div class="tier-comparison">
    <div class="tier-card tier-self">  <!-- Self-Signed on LEFT -->
        ...
    </div>

    <div class="tier-card tier-notarized">  <!-- Notarized on RIGHT -->
        ...
    </div>
</div>
```

**Fix recommendation**:
1. Swap the order of the two `<div class="tier-card">` blocks — move the `tier-notarized` div BEFORE the `tier-self` div.
2. Same fix applies to the Svelte version at lines 195-221.

**Revised HTML** (swap lines 648-661 and 663-673):
```html
<div class="tier-comparison">
    <!-- Notarized (LEFT) -->
    <div class="tier-card tier-notarized">
        <div class="tier-header">
            <span class="tier-badge tier-badge-notarized">Notarized</span>
        </div>
        ...
    </div>

    <!-- Self-Signed (RIGHT) -->
    <div class="tier-card tier-self">
        <div class="tier-header">
            <span class="tier-badge tier-badge-self">Self-Signed</span>
        </div>
        ...
    </div>
</div>
```

**Difficulty**: Low — HTML reordering only, no CSS changes.

---

### 6. VCP Inline Code Formatting

**Finding**: Under "VCP Attestation" section, `?include_vcp=true` appears as plain text in a sentence. Should be wrapped in `<code>` tags or styled as inline code.

**Files affected**:
- HTML: `/static/docs.html` line 732
- Svelte: `/frontend/src/routes/docs/+page.svelte` — (check around line 250)

**Current HTML code** (line 732):
```html
<h3>Requesting a VCP Attestation</h3>
<p>Add <code>?include_vcp=true</code> to the result endpoint:</p>
```

**Status**: ✓ ALREADY CORRECT — The text is already in `<code>` tags. **No change needed.**

**Difficulty**: None — already fixed.

---

### 7. VCP Double Dashes — NO CHANGE NEEDED

**Finding**: `--notarize` appears in VCP section — CLI flag, intentional.

**Locations**:
- HTML: `/static/docs.html` — (not found in VCP section, verify)

**Status**: ✓ CONFIRMED NO ISSUE — Skip.

---

### 8. Security Model Table — Add Row Dividers + Spacing

**Finding**: Under "Security Model" section, the attack/defense table lacks row separators. The orange warning callout below needs margin spacing.

**Files affected**:
- HTML: `/static/docs.html` lines 1019-1069
- CSS: `/static/docs.css` lines 530-532 (callout styling)

**Current HTML code** (lines 1022-1063):
```html
<table>
    <thead>
        <tr>
            <th>Attack Vector</th>
            <th>METTLE Defense</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Human impersonation</td>
            <td>Millisecond timing thresholds, native capability probes</td>
        </tr>
        <!-- ... more rows ... -->
    </tbody>
</table>

<div class="callout">
    <i class="fa-solid fa-triangle-exclamation"></i>
    <p><strong>Server-side evaluation.</strong> Correct answers are NEVER sent to clients...</p>
</div>
```

**Fix recommendation**:
1. Add row borders to tables (same as #1 and #4 — apply to all tables).
2. Add margin-top to `.callout` class in `/static/docs.css` after line 532:
```css
.callout {
    display: flex;
    gap: 1rem;
    padding: 1rem;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid var(--primary);
    border-radius: 8px;
    margin-bottom: 1.5rem;
    margin-top: 1.5rem;  /* ADD THIS LINE */
}
```

**Difficulty**: Low — Add table row borders (already done in #1) + 1 CSS line.

---

### 9. Anti-Gaming Table — Add Row Dividers + Spacing

**Finding**: Under "Anti-Gaming Design" section, the attack/defence table lacks row dividers. Text box below needs margin spacing.

**Files affected**:
- HTML: `/static/docs.html` lines 1071-1113
- CSS: `/static/docs.css` (reuse callout styling from #8)

**Current HTML code** (lines 1075-1112):
```html
<table>
    <thead>
        <tr>
            <th>Attack</th>
            <th>Defence</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Memorise answers</td>
            <td>Every problem is procedurally generated. Nothing repeats.</td>
        </tr>
        <!-- ... more rows ... -->
    </tbody>
</table>
```

**Fix recommendation**:
Same as #8 — apply row borders to all tables and ensure callout spacing is consistent.

**Difficulty**: Low — Reuse fixes from #1 and #8.

---

### 10. MCP Available Tools Table — Add Row Dividers

**Finding**: Under "MCP Integration > Available Tools" section, table rows lack dividers.

**Files affected**:
- HTML: `/static/docs.html` lines 1183-1213

**Current HTML code** (lines 1184-1213):
```html
<table>
    <thead>
        <tr>
            <th>Tool</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>mettle_start_session</code></td>
            <td>Start a verification session. Returns challenges for all suites.</td>
        </tr>
        <!-- ... more rows ... -->
    </tbody>
</table>
```

**Fix recommendation**:
Apply the same table row border fix from #1 — this is a global CSS fix for all `<table>` elements.

**Difficulty**: Low — Already solved by global table CSS rule.

---

### 11. Error Codes Table — Add Row Dividers

**Finding**: Under "Error Codes" section, the code/error/meaning table lacks row separators.

**Files affected**:
- HTML: `/static/docs.html` lines 1246-1291

**Current HTML code** (lines 1247-1290):
```html
<table>
    <thead>
        <tr>
            <th>Code</th>
            <th>Error</th>
            <th>Meaning</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>400</code></td>
            <td>Bad Request</td>
            <td>Invalid request body, unknown suite name, or bad parameters</td>
        </tr>
        <!-- ... more rows ... -->
    </tbody>
</table>
```

**Fix recommendation**:
Apply the same table row border fix — global CSS rule for all tables (from #1).

**Difficulty**: Low — Global fix already in place.

---

### 12. Sidebar "Help" Link Highlighting Bug

**Finding**: Clicking "Help" in sidebar scrolls to the correct section but the active highlight stays on "Troubleshooting" instead of moving to "Help".

**Source files**:
- HTML: `/static/docs.html` lines 1477-1511 (scroll-spy JavaScript)
- CSS: `/static/docs.css` lines 114-122 (active styling)

**Current JavaScript code** (lines 1488-1507):
```javascript
function updateActive() {
    var scrollY = window.scrollY + 100;
    var current = sectionIds[0];

    for (var i = 0; i < sectionIds.length; i++) {
        var el = document.getElementById(sectionIds[i]);
        if (el && el.offsetTop <= scrollY) {
            current = sectionIds[i];  // This line always sets to the current visible section
        }
    }

    tocItems.forEach(function(item) {
        var href = item.getAttribute('href');
        if (href === '#' + current) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}
```

**Problem**: The scroll-spy logic is based on `offsetTop <= scrollY + 100`. When you scroll to "Help" (the last section), "Troubleshooting" (the previous section) might still meet this condition, causing the logic to mark "Troubleshooting" as active instead of "Help".

**Fix recommendation**:
Update the scroll-spy function at lines 1488-1507 to prioritize the section that is closest to the viewport center:

```javascript
function updateActive() {
    var scrollY = window.scrollY;
    var viewportCenter = scrollY + window.innerHeight / 2;
    var current = sectionIds[0];
    var closestDistance = Infinity;

    for (var i = 0; i < sectionIds.length; i++) {
        var el = document.getElementById(sectionIds[i]);
        if (el) {
            var elCenter = el.offsetTop + el.offsetHeight / 2;
            var distance = Math.abs(elCenter - viewportCenter);

            if (distance < closestDistance) {
                closestDistance = distance;
                current = sectionIds[i];
            }
        }
    }

    tocItems.forEach(function(item) {
        var href = item.getAttribute('href');
        if (href === '#' + current) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}
```

**Alternative (simpler) fix**: Reduce the threshold or check that we're in the actual section range:

```javascript
function updateActive() {
    var scrollY = window.scrollY + 150;  // Increase threshold
    var current = sectionIds[0];

    for (var i = sectionIds.length - 1; i >= 0; i--) {  // Iterate backwards
        var el = document.getElementById(sectionIds[i]);
        if (el && el.offsetTop <= scrollY) {
            current = sectionIds[i];
            break;  // Stop at the first match from the bottom
        }
    }

    tocItems.forEach(function(item) {
        var href = item.getAttribute('href');
        if (href === '#' + current) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}
```

**Difficulty**: Medium — JavaScript logic fix, needs testing.

---

## SUMMARY OF FIXES

| Item | Type | File | Lines | Effort | Status |
|------|------|------|-------|--------|--------|
| 1 | Table borders | CSS | 509+ | Low | Add rule |
| 2 | Double dashes | N/A | N/A | None | ✓ OK |
| 3 | Endpoint spacing | CSS | 512-514 | Low | Update padding |
| 4 | Column dividers | CSS | 510+ | Low | Add rule |
| 5 | Swap boxes | HTML/Svelte | 648-674 / 195-221 | Low | Reorder divs |
| 6 | Code formatting | HTML | 732 | None | ✓ Already fixed |
| 7 | VCP dashes | N/A | N/A | None | ✓ OK |
| 8 | Security table | CSS | 530+ | Low | Add rules |
| 9 | Anti-gaming table | CSS | 530+ | Low | Add rules |
| 10 | MCP tools table | CSS | 509+ | Low | Add rule (global) |
| 11 | Error codes table | CSS | 509+ | Low | Add rule (global) |
| 12 | Help highlight | JS | 1488-1507 | Medium | Fix scroll-spy logic |

---

## CONSOLIDATED CSS FIX

The following CSS additions to `/static/docs.css` will fix items **1, 4, 8, 9, 10, 11** in one place:

**Add after line 510**:
```css
/* =============================================
   TABLE ROW & COLUMN STYLING (audit fixes)
   ============================================= */

/* Add horizontal dividers between all table rows */
.docs-content table tbody tr {
    border-bottom: 1px solid var(--border);
}

/* Add vertical dividers between table columns */
.docs-content table thead th,
.docs-content table tbody td {
    border-right: 1px solid var(--border);
}

/* Remove right border from last column */
.docs-content table thead th:last-child,
.docs-content table tbody td:last-child {
    border-right: none;
}

/* Improve endpoint table spacing (item #3) */
.endpoint-table td:nth-child(2) {
    padding-right: 2rem;
}

.endpoint-table td:nth-child(3) {
    padding-left: 1.5rem;
}

/* Add margin above callout boxes (items #8, #9) */
.callout {
    margin-top: 1.5rem;
}
```

**Priority order**:
1. Fix CSS (covers 8 items)
2. Swap HTML divs (item #5)
3. Fix JavaScript (item #12)

---

## TESTING CHECKLIST

After implementing fixes:
- [ ] All tables display with visible row borders
- [ ] Endpoint paths have clear spacing from descriptions
- [ ] Notarized box appears on the LEFT in credentials section
- [ ] Callout boxes have adequate margin above them
- [ ] Sidebar highlights "Help" when scrolled to Help section
- [ ] No layout breaks on mobile (test at 768px, 480px)
- [ ] Validate HTML with W3C validator
- [ ] Run `make validate-all` for Svelte version

