# METTLE Home Page Audit - Code Location & Fix Recommendations

## Audit Item 1: Hero Section Spacing

**Files:** `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` (lines 156-180) & `/Users/nellwatson/Documents/GitHub/METTLE/static/style.css` (lines 224-285)

### Current Problem
- Hero section has excessive empty space at top and bottom
- Subtitle text ("The Turing test asks...") should be directly under "Prove Your Metal" heading
- "METTLE inverts the question..." should be a separate element

### Current Code Structure (HTML)
```html
<section class="mettle-hero">
  <div class="container hero-content">
    <p class="hero-eyebrow">...</p>
    <h1 class="hero-heading">Prove Your <span class="highlight">Metal.</span></h1>
    <p class="hero-subtitle">
      The Turing test asks if machines can pass for human.
      <br>METTLE inverts the question: <strong>"Can you prove you're NOT human?"</strong>
    </p>
    <p class="hero-typewriter" id="hero-typewriter">...</p>
```

### Current CSS (style.css, lines 224-238, 273-294)
```css
.mettle-hero {
    min-height: 88vh;  /* 88% of viewport = excessive empty space */
    display: flex;
    align-items: center;
    justify-content: center;
}

.hero-content {
    padding: var(--space-2xl) var(--space-lg);  /* 4rem vertical = 64px */
}

.hero-subtitle {
    font-size: var(--text-xl);
    color: var(--color-text-muted);
    max-width: 800px;
    margin: 0 auto var(--space-md);
}
```

### Recommendation
**Split the subtitle into two elements:**
1. Change HTML structure to separate the subtitle from the main question
2. Reduce hero section min-height from 88vh to 70vh or use dynamic height based on content
3. Adjust `.hero-subtitle` margin-bottom to connect it tighter to heading
4. Create new `.hero-question` class for the inverted question part with distinct styling

**HTML Fix (lines 165-168):**
```html
<p class="hero-subtitle">
  The Turing test asks if machines can pass for human.
</p>
<p class="hero-question">
  <strong>METTLE inverts the question: "Can you prove you're NOT human?"</strong>
</p>
```

**CSS Fixes (style.css):**
- Line 226: Change `min-height: 88vh;` to `min-height: 70vh;` or `min-height: 60vh;`
- Line 277: Change `margin: 0 auto var(--space-md);` to `margin: 0 auto var(--space-sm);` (tighter spacing)
- Add new rule after line 285:
```css
.hero-question {
    font-size: var(--text-lg);
    color: var(--color-text);
    max-width: 800px;
    margin: var(--space-md) auto var(--space-xl);
    animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.9s both;
}
```

---

## Audit Item 2: Four Threats Boxes Alignment

**Files:** `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` (lines 194-231) & `/Users/nellwatson/Documents/GitHub/METTLE/static/style.css` (lines 429-471)

### Current Problem
- "Humanslop" box (first box) bottom line is not aligned with other 3 boxes
- Boxes need consistent minimum height and equal padding
- Box borders are barely visible (opacity too low)

### Current Code (HTML)
```html
<div class="problems-row">
  <div class="scroll-reveal">
    <div class="problem-item">
      <span class="problem-icon"><i class="fa-solid fa-user-secret"></i></span>
      <div>
        <strong>Humanslop</strong>
        <p>Humans infiltrating...</p>
      </div>
    </div>
  </div>
  <!-- 3 more similar items -->
</div>
```

### Current CSS (style.css, lines 422-471)
```css
.problems-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-md);
    margin-top: var(--space-xl);
}

.problem-item {
    display: flex;
    align-items: flex-start;
    gap: var(--space-md);
    padding: var(--space-md) var(--space-lg);  /* Not enough vertical padding */
    background: rgba(18, 18, 32, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.06);  /* Border too faint: 0.06 opacity */
    border-radius: var(--radius-lg);
}
```

### Recommendation
**CSS Fixes (style.css, lines 429-445):**

Replace the `.problem-item` rule:
```css
.problem-item {
    display: flex;
    flex-direction: column;  /* Change from row flex to column to equalize height */
    padding: var(--space-lg) var(--space-lg);  /* Increase from space-md to space-lg (1.5rem) */
    min-height: 200px;  /* Add explicit minimum height */
    background: rgba(18, 18, 32, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.15);  /* Increase opacity from 0.06 to 0.15 */
    border-radius: var(--radius-lg);
    -webkit-backdrop-filter: blur(8px);
    backdrop-filter: blur(8px);
    transition: all var(--transition-normal);
}

.problem-item:hover {
    border-color: rgba(245, 158, 11, 0.3);  /* Increase hover border visibility */
    transform: translateY(-2px);
}
```

**Note on structure:** If you want the icon and text side-by-side as currently displayed, keep flex-direction as row but add `align-items: stretch` and ensure `.problem-item > div` has `flex: 1` to expand and equalize heights.

---

## Audit Item 3: Divider Spacing

**Files:** `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` (lines 235-239, 298-302) & `/Users/nellwatson/Documents/GitHub/METTLE/static/style.css` (lines 162-218)

### Current Problem
- Space between upper divider and "Design Philosophy" section differs from bottom divider spacing
- Need consistent spacing on both sides

### Current Code (HTML)
```html
<!-- Line 235-239: First divider (after problem section) -->
<div class="glow-divider" style="--glow-color: rgba(245, 158, 11, 0.5);" aria-hidden="true">
    <div class="glow-line"></div>
    <div class="glow-dot"></div>
    <div class="glow-line"></div>
</div>

<!-- Line 242: Design Philosophy section starts immediately -->
<section class="philosophy-section">
```

### Current CSS (style.css, lines 545-548)
```css
.philosophy-section {
    position: relative;
    padding: var(--space-2xl) 0;  /* No top margin specified; depends on default */
}
```

### Recommendation
**CSS Fix (style.css):**
- Line 162-174: `.glow-divider` needs explicit margin
- Line 545-548: `.philosophy-section` needs explicit top margin

Add these rules:
```css
.glow-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin: var(--space-2xl) auto;  /* Add explicit top/bottom margin (4rem = 64px) */
    padding: 0;
    width: 100%;
    max-width: 500px;
}

.philosophy-section {
    position: relative;
    padding: var(--space-2xl) 0;  /* Keep consistent padding */
    margin-top: 0;  /* Prevent margin collapse */
}
```

**Rationale:** Use the same `var(--space-2xl)` (4rem) margin above and below all dividers for visual consistency.

---

## Audit Item 4: Verification Suites Numbers/Quotes Visibility

**Files:** `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` (lines 319-587) & `/Users/nellwatson/Documents/GitHub/METTLE/static/style.css` (lines 688-738)

### Current Problem
- Suite numbers (01, 02, etc.) are too dark/dim
- Quoted text in `.suite-insight` is too dim and hard to read
- Both need higher contrast/visibility

### Current Code (HTML - example from Suite 1, lines 327-335)
```html
<span class="suite-num">01</span>  <!-- Color issue -->
...
<div class="suite-insight">
    <i class="fa-solid fa-quote-left" aria-hidden="true"></i>
    If you need to think about the answer, you already failed the time limit.
</div>
```

### Current CSS (style.css, lines 688-738)
```css
.suite-num {
    font-size: var(--text-xs);
    font-weight: 700;
    font-family: var(--font-mono);
    color: var(--color-text-subtle);  /* Too subtle: #6b6b80 */
    opacity: 0.5;  /* Reduces visibility even further */
    padding: 2px 8px;
    background: rgba(255, 255, 255, 0.04);
}

.suite-insight {
    font-size: var(--text-xs);
    color: var(--color-text-subtle);  /* Same dim color */
    font-style: italic;
    padding: var(--space-sm) var(--space-md);
    background: rgba(255, 255, 255, 0.02);
}
```

### Recommendation
**CSS Fixes (style.css):**

**Line 688-697 (.suite-num):**
```css
.suite-num {
    font-size: var(--text-xs);
    font-weight: 700;
    font-family: var(--font-mono);
    color: hsla(var(--suite-hue, 35), 80%, 70%, 1);  /* Match suite-question color */
    opacity: 1;  /* Remove opacity: 0.5 */
    padding: 2px 8px;
    background: hsla(var(--suite-hue, 35), 70%, 55%, 0.2);  /* Lighter background */
    border-radius: var(--radius-sm);
}
```

**Line 723-738 (.suite-insight):**
```css
.suite-insight {
    font-size: var(--text-xs);
    color: var(--color-text-muted);  /* Lighter than text-subtle: #a0a0b0 */
    font-style: italic;
    padding: var(--space-sm) var(--space-md);
    background: rgba(255, 255, 255, 0.04);  /* Increase from 0.02 */
    border-left: 2px solid hsla(var(--suite-hue, 35), 70%, 55%, 0.5);  /* Increase opacity */
}

.suite-insight i {
    color: hsla(var(--suite-hue, 35), 70%, 55%, 0.8);  /* Increase from 0.4 */
    margin-right: var(--space-xs);
    font-size: 0.625rem;
}
```

---

## Audit Item 5: Verifiable Credentials Layout - Single Row

**Files:** `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` (lines 724-753) & `/Users/nellwatson/Documents/GitHub/METTLE/static/style.css` (lines 946-1010)

### Current Problem
- Credential tiers (BASIC, AUTONOMOUS, GENUINE, SAFE) are displayed in a 4-column grid
- Should be in a single row (as shown in AUTONOMOUS display)
- Need to adjust grid layout

### Current Code (HTML)
```html
<section class="credentials-section">
    <div class="credentials-grid">
        <div class="scroll-reveal">
            <div class="credential-card credential-basic">
                <span class="credential-level">Basic</span>
                ...
            </div>
        </div>
        <!-- 3 more cards (Autonomous, Genuine, Safe) -->
    </div>
</section>
```

### Current CSS (style.css, lines 946-965)
```css
.credentials-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);  /* 4 columns - this is correct for single row */
    gap: var(--space-md);
    margin-top: var(--space-xl);
}
```

### Analysis
**The CSS already shows `repeat(4, 1fr)` which IS a single row.** This suggests the HTML structure is correct but responsive breakpoints may be collapsing it. Check media queries.

### Recommendation
**CSS Fix (style.css):**

Verify media query doesn't override this. Look for responsive overrides around line 2100-2250. Ensure:

```css
/* Ensure single row on desktop - line 946 */
.credentials-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-md);
    margin-top: var(--space-xl);
}

/* Mobile breakpoint (if exists, around line 2100+) - ONLY reduce for small screens */
@media (max-width: 768px) {
    .credentials-grid {
        grid-template-columns: repeat(2, 1fr);  /* 2 cols on tablets */
    }
}

@media (max-width: 480px) {
    .credentials-grid {
        grid-template-columns: 1fr;  /* 1 col on mobile */
    }
}
```

**HTML: No changes needed** - The HTML structure is correct as-is.

---

## Audit Item 6: "Run It Yourself" Double Dashes

**Files:** `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` (lines 780, 783)

### Finding
**✅ NO CHANGE NEEDED** - This is intentional and correct.

The double dashes (`--full`, `--notarize`) are CLI flag syntax in the terminal code block (lines 780, 783). This is the proper convention for command-line arguments and should NOT be changed.

```html
<span class="tok-cmd">$</span> mettle verify <span class="tok-flag">--full</span>
<span class="tok-cmd">$</span> mettle verify <span class="tok-flag">--full</span> <span class="tok-flag">--notarize</span>
```

**Status:** Correct as-is. No modifications required.

---

## Audit Item 7: "How Verification Works" Double Dashes

**Files:** `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` (lines 838, 849)

### Finding
**✅ NO CHANGE NEEDED** - This is intentional and correct.

The double dashes in the "How Verification Works" section (lines 838, 849) are also CLI flag syntax:

```html
<code class="step-endpoint">mettle verify --full</code>
<code class="step-endpoint">--notarize</code>
```

**Status:** Correct as-is. No modifications required.

---

## Audit Item 8: Feature Icons Row Positioning

**Files:** `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html` (lines 917-949) & `/Users/nellwatson/Documents/GitHub/METTLE/static/style.css` (lines 1811-1843)

### Current Problem
- The badges row (JWT Signed, Fresh Badges, Revocable, Open Source, Self-Hostable, 117 Tests) is not centered
- Should be positioned in the middle of the section after the divider
- Current positioning may be too high or misaligned

### Current Code (HTML)
```html
<!-- Line 917-949 -->
<section class="badges-section">
    <div class="container">
        <div class="scroll-reveal">
            <div class="badges-row">
                <div class="badge-item">
                    <i class="fa-solid fa-key" aria-hidden="true"></i>
                    <span>JWT Signed</span>
                </div>
                <!-- 5 more badge items -->
            </div>
        </div>
    </div>
</section>
```

### Current CSS (style.css, lines 1811-1843)
```css
.badges-section {
    padding: var(--space-xl) 0 var(--space-2xl);  /* Asymmetric padding */
}

.badges-row {
    display: flex;
    justify-content: center;  /* Horizontally centered */
    flex-wrap: wrap;
    gap: var(--space-lg);
    /* No explicit vertical centering or positioning */
}
```

### Recommendation
**CSS Fix (style.css, lines 1811-1843):**

```css
.badges-section {
    padding: var(--space-2xl) 0;  /* Make padding symmetric (4rem top/bottom) */
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 300px;  /* Ensure minimum height for vertical centering */
}

.badges-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: var(--space-lg);
    width: 100%;  /* Full width for proper centering */
}
```

**Alternative if you prefer keeping badges lower:**
```css
.badges-section {
    padding: var(--space-xl) 0 var(--space-2xl);  /* Keep asymmetric if intentional */
    text-align: center;
}

.badges-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: var(--space-lg);
    margin-top: var(--space-xl);  /* Add explicit top margin for spacing */
}
```

---

## Summary Table

| Item | File | Lines | Issue | Fix Type |
|------|------|-------|-------|----------|
| 1 | index.html / style.css | 156-180 / 224-285 | Hero spacing, text hierarchy | HTML restructure + CSS height reduction |
| 2 | index.html / style.css | 194-231 / 429-471 | Box alignment, border visibility | CSS padding, min-height, border opacity |
| 3 | style.css | 162-548 | Divider spacing inconsistency | CSS margin normalization |
| 4 | style.css | 688-738 | Suite numbers & quotes too dim | CSS color opacity increase |
| 5 | style.css | 946-965 | Credential grid layout | Already correct; verify media queries |
| 6 | index.html | 780, 783 | Double dashes in CLI | ✅ No change needed (intentional) |
| 7 | index.html | 838, 849 | Double dashes in steps | ✅ No change needed (intentional) |
| 8 | index.html / style.css | 917-949 / 1811-1843 | Badge row positioning | CSS padding/min-height adjustment |

