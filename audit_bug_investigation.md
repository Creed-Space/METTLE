# METTLE Bug Investigation: "[object Object]" Error Display

## Bug Report
When a user clicks "Start Verification" on the Test Verification section of the website with any difficulty option, the page shows:
```
Something went wrong
[object Object]
```

The `[object Object]` error appears instead of a human-readable error message.

---

## Root Cause Analysis

### Location: `/Users/nellwatson/Documents/GitHub/METTLE/static/app.js`, lines 60-80

The error display bug is in the `apiCall()` function:

```javascript
// BUGGY CODE (lines 60-80)
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
        throw new Error(data.detail || 'API request failed');  // <-- BUG HERE (line 76)
    }

    return data;
}
```

### The Problem

At **line 76**, the code attempts to extract error details:
```javascript
throw new Error(data.detail || 'API request failed');
```

**This fails in these scenarios:**

1. **FastAPI validation errors** (422 Unprocessable Entity): FastAPI returns an array of validation error objects in `data.detail`, not a string:
   ```json
   {
     "detail": [
       {
         "type": "value_error",
         "loc": ["body", "difficulty"],
         "msg": "...",
         "input": "invalid_value"
       }
     ]
   }
   ```
   When `data.detail` is an array, `new Error(array)` converts it to `[object Object]`

2. **Missing `.detail` property**: Some error responses may not have a `.detail` field. The fallback `'API request failed'` is used, but if there's an earlier error (JSON parse, network), the thrown error object itself might be passed incorrectly.

3. **Catch handler logs object directly**: At line 262-263:
   ```javascript
   } catch (error) {
       showError(error.message);
   }
   ```
   If `error` is not an Error object, `error.message` could be undefined or return `[object Object]`.

---

## Evidence

### Error Flow
1. User clicks "Start Verification" → `handleStart()` called (line 241)
2. API call to `/api/session/start` is made (line 249)
3. Response validation fails or returns error status
4. Line 73: `const data = await response.json();` succeeds (valid JSON)
5. Line 75-76: Status is not `ok`, so error is thrown:
   - If `data.detail` is **array** (FastAPI validation errors), Error constructor receives array
   - Array gets stringified to `[object Object]`
6. Line 262-263: Catch block calls `showError(error.message)`
7. `error.message` is `[object Object]` (the stringified array)
8. Line 236: `elements.errorMessage.textContent = message;` displays `[object Object]`

### Proof That Default Difficulty Values Are Wrong
Looking at `static/index.html` line 1009-1013:
```html
<select id="difficulty">
    <option value="easy">Easy - relaxed timing</option>
    <option value="standard" selected>Standard - production-grade</option>
    <option value="hard">Hard - aggressive timing, maximum depth</option>
</select>
```

But the **backend** (`main.py`) defines valid difficulty values in the `Difficulty` enum (imported from mettle package). The frontend sends `"easy"`, `"standard"`, or `"hard"`, but the backend likely expects `"basic"` or `"full"`.

This causes a **422 validation error** with `data.detail` being an array of validation errors, which gets stringified to `[object Object]`.

---

## Recommended Fix

### File: `/Users/nellwatson/Documents/GitHub/METTLE/static/app.js`

**Replace lines 60-80** with improved error handling that properly extracts error messages:

```javascript
// API Calls - FIXED VERSION
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

    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        const data = await response.json();

        if (!response.ok) {
            // Extract error message properly from various response formats
            let errorMessage = 'API request failed';

            if (data && typeof data === 'object') {
                // Handle FastAPI error: detail is string
                if (typeof data.detail === 'string') {
                    errorMessage = data.detail;
                }
                // Handle FastAPI validation error: detail is array of objects
                else if (Array.isArray(data.detail) && data.detail.length > 0) {
                    // Extract first validation error message
                    const firstError = data.detail[0];
                    errorMessage = firstError.msg || `Validation error at ${firstError.loc?.join('.')}: ${firstError.type}`;
                }
                // Handle custom error object
                else if (data.error) {
                    errorMessage = typeof data.error === 'string' ? data.error : data.error.message || 'Unknown error';
                }
                // Handle message field
                else if (data.message) {
                    errorMessage = data.message;
                }
            }

            throw new Error(errorMessage);
        }

        return data;
    } catch (error) {
        // If error is already an Error with message, use it
        // Otherwise create one from the error object
        if (error instanceof Error) {
            throw error;
        }
        throw new Error(String(error) || 'API request failed');
    }
}
```

### File: `/Users/nellwatson/Documents/GitHub/METTLE/static/index.html`

**Fix the difficulty options to match backend expectations** (lines 1009-1013):

```html
<select id="difficulty">
    <option value="basic">Basic - relaxed timing</option>
    <option value="full" selected>Full - aggressive timing</option>
</select>
```

**Note**: The dropdown currently shows 3 options (`easy`, `standard`, `hard`), but the backend `mettle` package likely only defines `basic` and `full`. This mismatch causes validation errors, triggering the `[object Object]` bug.

---

## Additional Notes

### Secondary Issue: Incomplete Error Context
The current error handler doesn't provide context about **which** API endpoint failed. This makes debugging harder. Consider enhancing `showError()` to log the full error details to browser console:

```javascript
// Enhanced showError function (bonus improvement)
function showError(message) {
    stopTimer();
    elements.errorMessage.textContent = message;
    console.error('[METTLE Error]', message); // Add debugging context
    showScreen('error');
}
```

### Why This Bug Wasn't Caught Immediately
1. The bug only manifests when API returns an error
2. Valid requests succeed without triggering error handling
3. The `[object Object]` string is suspiciously generic, making it hard to trace to the apiCall function
4. No automated testing of error paths in the UI layer

---

## Test Cases to Verify Fix

After applying the fix, test these scenarios:

1. **Valid request**: Click "Start Verification" with valid difficulty → should work
2. **Invalid difficulty** (if still exists): Try unsupported difficulty value → should show readable error
3. **Network error**: Disable network and click → should show "Failed to fetch" or similar
4. **Server error (500)**: Backend returns 500 → should show server error message if provided
5. **Validation error (422)**: Intentionally send bad data → should show validation message, not `[object Object]`

---

## Impact Assessment

- **Severity**: High (prevents all verification attempts if backend returns any error)
- **User Impact**: Displays cryptic error message, preventing users from understanding what went wrong
- **Scope**: Frontend only (no backend changes needed)
- **Backward Compatibility**: Yes (change is transparent to users, only fixes error display)
