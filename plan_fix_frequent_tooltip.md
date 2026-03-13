# Plan: Fix Tooltip Coordinate Space Mismatch

## Context

`dcc.Tooltip` positions itself using `position: fixed` in the viewport — meaning it needs the `bbox` coordinates in **viewport space** (pixels from the top-left of the browser window). Plotly's `hoverData.points[i].bbox` provides the bounding box of the hovered bar in **SVG-pixel space** (pixels from the top-left of the Plotly SVG element). These spaces are identical only if the graph SVG origin coincides with the viewport origin — which is never the case once there's any chrome (navbar, tabs, etc.) above the graph.

The compounding problem: the `dcc.Tooltip` is currently rendered *inside* `dcc.Loading`, which applies `position: relative` to its wrapper div. Since Dash's Tooltip does **not** use a React portal, it gets caught by the nearest `position: relative` ancestor and is positioned relative to the `dcc.Loading` div rather than the viewport. This creates a double offset: the `dcc.Loading` offset within the page, plus the viewport-vs-SVG mismatch.

## Root Cause Summary

```
dcc.Loading (position: relative)         ← traps Tooltip's absolute position
  └─ dcc.Graph  (SVG at offset X,Y)      ← bbox origin != viewport origin
       hoverData.bbox = SVG-relative coords
  └─ dcc.Tooltip (position: fixed / absolute, but resolved relative to Loading div)
       → renders at wrong location
```

## Fix: two coordinated changes

### 1. Move `dcc.Tooltip` out of `frequent.layout` to page level

**File:** `src/fstg_toolkit/app/views/frequent.py`

Remove the `dcc.Tooltip` declaration from the `layout` list. The `dcc.Store` and all other elements stay. Only remove:
```python
dcc.Tooltip(id='frequent-pattern-tooltip', style={'max-width': '320px', 'padding': '0'}),
```

The `show_pattern_tooltip` callback does NOT change — it still writes to `frequent-pattern-tooltip`.

### 2. Place `dcc.Tooltip` at the dashboard container level

**File:** `src/fstg_toolkit/app/pages/dashboard.py`

In `dashboard_layout()`, add `dcc.Tooltip` as a direct child of `dbc.Container`, **outside and after** the `dbc.Tabs` block:
```python
dbc.Container(
    children=[
        dbc.Tabs([...], id='tabs'),
        dcc.Store(...),
        dcc.Tooltip(id='frequent-pattern-tooltip', style={'max-width': '320px', 'padding': '0'}),
    ],
    fluid='xxl')
```

`dbc.Container` does **not** apply `position: relative`, so the tooltip's absolute/fixed positioning resolves against the viewport as intended.

### 3. (Optional but recommended) Correct the bbox to viewport coordinates

If the mismatch between SVG-relative and viewport-relative still causes offset after step 1+2, add a clientside callback in `views/frequent.py` that adjusts `bbox` before the Python callback fires, or split the current callback: one clientside callback handles `show` + `bbox` (using `getBoundingClientRect` on the graph div to convert), one server callback handles `children`.

This is only needed if the tooltip is still offset after the relocation; try step 1+2 first.

## Critical Files

| File | Change |
|---|---|
| `src/fstg_toolkit/app/views/frequent.py` | Remove `dcc.Tooltip` from `layout` list |
| `src/fstg_toolkit/app/pages/dashboard.py` | Add `dcc.Tooltip` as sibling of `dbc.Tabs` in `dbc.Container` children |

## Verification

1. Start the dashboard: `python -m fstg_toolkit show my_graph.zip`
2. Open the "Frequent Patterns" tab with a dataset that has patterns
3. Hover over bars in the frequency histogram
4. Confirm the tooltip appears directly adjacent to (or above) the hovered bar
5. Scroll the page before hovering — tooltip should still align correctly
6. Test with faceted plots (multiple factors selected) to confirm subplots work too