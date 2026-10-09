---
target: HomeShield dashboard
total_score: 17
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 4
target_identity: "file:C:\\Users\\Admin\\Documents\\anaconda_projects\\FYPClaude\\homeshield\\templates\\index.html"
target_fingerprint: "sha256:1b458b801aed9f407ead5678c2e3c01ababcc89ff94701fa26588e7c495d45fc"
target_path: "C:\\Users\\Admin\\Documents\\anaconda_projects\\FYPClaude\\homeshield\\templates\\index.html"
timestamp: 2026-10-08T20-11-21Z
slug: homeshield-templates-index-html
closed: true
---
Method: dual-agent (A: design-review sub-agent · B: detector + browser sub-agent)

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | SSE/LIVE/detector bar good; alerting camera has no alarm state; "Loading events…" can hang forever |
| 2 | Match System / Real World | 2 | Confidence words excellent; "Fire detected (FIRE)", file paths in rows, Settings jargon |
| 3 | User Control and Freedom | 2 | No acknowledge/false-alarm; only reset is destructive "Clear all" |
| 4 | Consistency and Standards | 2 | Native alert/prompt beside styled toast; red = LIVE and critical |
| 5 | Error Prevention | 1 | Stop system unconfirmed; zones drawn on cropped image; invisible radio state |
| 6 | Recognition Rather Than Recall | 2 | Intruder register via typed prompt(); zone list shows camera ids |
| 7 | Flexibility and Efficiency | 1 | No shortcuts, no deep links, no camera/Fire filter, mouse-only divider |
| 8 | Aesthetic and Minimalist Design | 2 | Disciplined type/colour; duplicate chips, add-camera on Live, 71 sub-11px labels |
| 9 | Error Recovery | 2 | Good face-capture hints; raw errors elsewhere |
| 10 | Help and Documentation | 1 | No legend, no "what to do when FIRE" |
| **Total** | | **17/40** | **Poor** |

## Design Specificity Verdict
Look authored (navy, Chakra Petch, hairlines, split-flap, plain confidence words); composition generic (sidebar, KPI cards, video wall, log). Board buried (~670px desktop / 845px phone); no status column or verdict; palette slips on admin panes (violet/blue tiles, pink names, extra reds/oranges, canvas glow, emoji).
Detector: 113 findings (undersized-ui-text 71, tiny-text 20, all-caps-body 12 [real cascade leak .form-group label -> radio small], cramped-padding 5 [false positive], pulsing-dot 3, skipped-heading 1, em-dash-overuse 1). Browser-only: low-contrast (#m-alerts 2.9:1, red inline strong 2.9-3.4:1, placeholders 3.0-3.4:1), gray-on-color, repeated-container-text, clipped-overflow (hidden radios 1172px wide). Config "*" ignoreValues for low-contrast/broken-image are broader than their stated reasons.

## Priority Issues
- [P0] Mobile drops what matters: board hides Location (<=900) and Confidence/Snapshot (<=480), rows untappable; .set-nav display:none <=900 makes 6/7 Settings panes unreachable. Fix: stacked tappable cards, board under a status line, tab-strip settings nav. /impeccable adapt
- [P1] Red doesn't mean danger; critical text #e8002a 3.4:1 fails AA; fall amber vs motionless red; no alert state on camera tile. Fix: red only for active critical, LIVE neutral/green, red row band, camera alert frame, narrow config wildcards. /impeccable colorize
- [P1] Alerts are a log not incidents: 63 fire rows in ~20 min, per-frame confidence 0.35-0.9 flips labels, no acknowledge, destructive Clear all, no Fire filter. Fix: group into incidents with peak confidence, Acknowledge/False alarm, View live, unacknowledged count. /impeccable shape
- [P1] Events/Zones hang: MJPEG streams kept open off-Live + duplicate hero/thumb stream + SSE exhaust ~6 connections/host; no timeout/error. Fix: pause streams off-Live, one stream per camera, timeouts + retry state. /impeccable optimize, /impeccable harden
- [P1] Admin controls mislead: zone image object-fit:cover crops while clicks map to full 640x480; radio marks collapse (label cascade); Stop system unconfirmed next to Sign out; empty quick-add saves "Camera"/"0". /impeccable harden

## Persona Red Flags
Nadia (parent on phone): no events in first view; rows lack room/photo/tap; red "Alerts today" not actionable; red LIVE on healthy cameras; fire rows say "Possible"; login advertises admin/admin.
Sam (screen reader/keyboard): no aria-live; div grid not table; modal no role/focus; radio focus on invisible input; no aria-current/aria-expanded; 3.4:1 red text; 8-9px labels; blinking dots ignore reduced motion; split-flap leaves per-char spans.
Alex (power user): no shortcuts/deep links; tiles and snapshots not keyboard-focusable; mouse-only divider; no camera/Fire filter; two chained prompt() for intruders; no bulk actions.

## Minor Observations
Undefined classes for person/intruder cards (unstyled Dismiss); empty-state run-on text; tap targets <44px; loading-state contradictions; raw login error; mobile drawer stays open after sign-out; add-camera not admin-gated; Settings says model changes need restart (they're live); WhatsApp/Twilio cloud dependency vs positioning; user-requested larger hero conflicts with brief's uniform-tile rule (update the brief).

## Questions to Consider
Verdict line first? What does "done" mean for an alert? Should red ever show when nothing is wrong? Diagnostic view for the examiner vs family view for the parent?
