---
target_identity: "file:C:\\Users\\Admin\\Documents\\anaconda_projects\\FYPClaude\\homeshield\\templates\\index.html"
target_fingerprint: "sha256:4962688cea12ff2903cc9ade0879343bfb4ac46ff2307c6b5b44cd0c62ec5376"
target_path: "C:\\Users\\Admin\\Documents\\anaconda_projects\\FYPClaude\\homeshield\\templates\\index.html"
timestamp: 2026-10-08T14-14-51Z
slug: homeshield-templates-index-html
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Face/intruder detector absent from topbar; no stream loading indicator |
| 2 | Match System / Real World | 2 | LYING_MOTIONLESS, bbox coords, Camera: 0, SYSTEM NOMINAL -- developer language throughout |
| 3 | User Control and Freedom | 2 | Clear wipes all alerts with no confirmation and no undo |
| 4 | Consistency and Standards | 3 | ALL CAPS in banner vs sentence case elsewhere; enum names in filter labels |
| 5 | Error Prevention | 1 | No confirmation on destructive clear; no in-app face-detection enrollment warning |
| 6 | Recognition Rather Than Recall | 2 | Camera IDs not mapped to room names; no tooltips; bbox adds noise |
| 7 | Flexibility and Efficiency of Use | 1 | No keyboard shortcuts; no multi-camera switcher; no time-range filter or search |
| 8 | Aesthetic and Minimalist Design | 2 | Dev telemetry in stats card; bbox on every alert row; persistent banner reduces urgency signal |
| 9 | Error Recovery | 1 | Raw server error strings; camera offline has no action path; frontend errors console-only |
| 10 | Help and Documentation | 0 | Zero in-app help, tooltips, onboarding, or glossary |
| Total | | 17/40 | Poor |

## Design Specificity Verdict
Category-interchangeable. Raw enum labels, numeric camera IDs, bbox coords, and frame counts are engineering artifacts surfaced verbatim to non-technical users.

Detector: 4 findings -- 1 low-contrast (loading overlay #999999 on #faf9f5, 2.7:1), 3 side-tab on alert border-left (app.css:167,168,170).

## Priority Issues
[P1] Raw technical labels throughout alert feed -- /impeccable clarify
[P1] No onboarding, empty states, or first-run guidance -- /impeccable onboard
[P1] Clear button destroys all alerts with no confirmation/undo -- /impeccable harden
[P1] Face/intruder detector has no topbar status indicator -- /impeccable clarify
[P2] Stats card exposes developer telemetry -- /impeccable clarify

## Persona Red Flags
Jordan: empty state gives no guidance; LYING_MOTIONLESS uninterpretable; accidental clear; no tooltips.
Casey: video blocks alert list on mobile scroll; sub-44pt touch targets; filter checkboxes too small.
Priya (family member): looks like developer tool; no response guidance; Camera: 0 unmapped.

## Minor Observations
- Icon/LOGO.svg unused; brand mark is a CSS circle
- Persistent SYSTEM NOMINAL banner creates cry-wolf desensitization
- Modal Escape key not handled in JS
- LYING_MOTIONLESS gets generic warning triangle emoji
- Alert list max-height 60vh shows only 5-6 alerts on mobile
