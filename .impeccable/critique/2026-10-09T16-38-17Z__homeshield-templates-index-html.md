---
target: the login page (sign-in screen)
total_score: 27
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
target_identity: "file:C:\\Users\\Admin\\Documents\\anaconda_projects\\FYPClaude\\homeshield\\templates\\index.html"
target_fingerprint: "sha256:fb5dc052e023e8fd1fed8ef6f1065b861ea15f71e6a94db4959d3b2ae1a700c2"
target_path: "C:\\Users\\Admin\\Documents\\anaconda_projects\\FYPClaude\\homeshield\\templates\\index.html"
timestamp: 2026-10-09T16-38-17Z
slug: homeshield-templates-index-html
closed: true
---
Method: dual-agent (A: design-review agent · B: detector + browser agent)
Surface: the sign-in screen only (auth shell in homeshield/templates/index.html). Earlier snapshots for this slug scored the whole dashboard, so trend lines are not like-for-like.

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | "ONLINE" lamp checked once; form stays active when unreachable |
| 2 | Match System / Real World | 3 | "127.0.0.1:5055", "venv", raw IP in toast are system words |
| 3 | User Control and Freedom | 3 | Lockout escape is plain text, not a link |
| 4 | Consistency and Standards | 2 | .btn overrides .auth-submit; native bubbles beside custom banners; critical red for a wrong password |
| 5 | Error Prevention | 2 | No warning before 5-try lockout; can re-pick "admin" as new password |
| 6 | Recognition Rather Than Recall | 3 | Change-password card doesn't name the account |
| 7 | Flexibility and Efficiency | 3 | Good autocomplete, Enter submits, remembered-user focus |
| 8 | Aesthetic and Minimalist Design | 3 | Forgot card crowds two recovery paths plus a command |
| 9 | Error Recovery | 2 | No next step after wrong password; "a few minutes" vs 4:47; focus lost after "Request sent" |
| 10 | Help and Documentation | 3 | Conditional first-run hint, honest no-email copy |
| **Total** | | **27/40** | **Acceptable** |

## Design Specificity Verdict
Authored on the left, template on the right. The board (split-flap clock with seam, ONLINE lamp, ruled watch rows, privacy line) is HomeShield's own; the card (Sign in / Welcome back / two fields / remember+forgot / full-width button) and the brand-left-form-right layout are category-interchangeable. The departure-board idea stops at the clock: no status column, lockout countdown is plain text, no board character in the card.
Detector: CLI 19 findings, 0 on the sign-in screen (10 all-caps label false positives, 6 cramped-padding, 2 pulsing-dot, 1 icon-tile — all dashboard). Sign-in-only scan: 0. Browser overlay injected in the agent's tab: 0 inside #authShell at desktop, phone and forgot views. Detector missed the button cascade override, low-contrast input edges and suppressed focus; config ignores low-contrast for this file.

## Priority Issues
- [P1] Sign-in button loses its styling: .auth-submit (line 139) overridden by later .btn (line 300) → 32px / 12px button beside 39px inputs; disabled lockout state still looks live. Fix: `.auth-card .auth-submit{padding:12px 16px;font-size:var(--fs-ui);min-height:44px}`, grey disabled state. /impeccable polish
- [P1] Lockout without warning or exit: no attempts-left warning, critical red, "a few minutes" vs 4:47, "use Forgot password" not a link, countdown not announced. Fix: attempts_left from /api/login, warn from 3rd failure, amber "SIGN-IN PAUSED 04:47" flaps, real link, polite announcements. /impeccable harden
- [P1] Sole-admin recovery assumes a developer: terminal + venv shown to everyone. Fix: default to "Ask an admin"; "I'm the admin" disclosure with 3 numbered steps; double-clickable reset-password.bat; keep command as fallback. /impeccable clarify
- [P2] Input fields nearly invisible: border 1.36:1, fill 1.16:1 vs card (WCAG 1.4.11 needs 3:1); focus only a 1px border (outline:none). Fix: border >= rgba(255,255,255,.40), focus outline 2px amber offset 1px. /impeccable audit
- [P2] Board claims unchecked; phone card pushed down: "Watching every camera" with zero cameras, ONLINE never re-checked, 206px before the card on phones, button below fold at 320x568. Fix: poll setup_state every 15s with non-sensitive monitoring flag, honest heading, inline small clock on phones. /impeccable adapt

## Persona Red Flags
Jordan: "Welcome back" vs "First time here?"; admin/admin hint below the button; "temporary" password confusion; native bubble for short password; no confirmation after saving.
Sam: no landmark around the card, ~15 board items before the H1; Caps Lock warning not announced; focus not moved to the forgot heading; focus lost after "Request sent"; no aria-invalid; silent countdown.
Casey: Sign in below the fold at 568px tall; 16x16 remember checkbox; 3s toast; terminal section irrelevant on phone.

## Minor Observations
text-2 ≈ text-3; card jumps 28px on error; credits spill past the divider at 1280x620, board top-heavy at 1920x1080; tab title not "Sign in · HomeShield"; last-sign-in IP in a 3s toast; unreachable empty-field message (required blocks submit); admin/admin hint broadcast on the LAN until first sign-in.

## Questions to Consider
- Should the board show an open incident before sign-in, or is that a leak to a Wi-Fi guest?
- Why is a clock the largest thing on a safety product's front door; could the flaps carry HomeShield-only facts?
- Should the first-run password be printed on the PC console instead of the LAN-visible page?
