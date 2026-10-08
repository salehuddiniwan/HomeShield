"""Build the complete Concourse-redesigned decoded_template.html."""
import re

with open('decoded_template.html', encoding='utf-8') as f:
    old = f.read()

# ── Extract original script blocks ───────────────────────────────────────────
scripts = []
pos = 0
while True:
    idx = old.find('<script', pos)
    if idx == -1:
        break
    end = old.find('</script>', idx) + len('</script>')
    scripts.append(old[idx:end])
    pos = end

script_img_error = scripts[0]   # img 404 handler
script_main      = scripts[1]   # main app JS (will be modified)
script_shader    = scripts[2]   # loginShader WebGL
script_settings  = scripts[3]   # settings toggle

# ── Patch main JS: replace eventRow and loadRecentEvents ────────────────────
NEW_EVENT_ROW = r'''function eventRow(e) {
  const typeStyles = {
    fall_detected:    {color:'amber',  label:'Fall detected'},
    lying_motionless: {color:'danger', label:'Lying motionless'},
    inactivity:       {color:'amber',  label:'Inactivity'},
    zone_entry:       {color:'amber',  label:'Child in danger zone'},
    intruder_detected:{color:'amber',  label:'Intruder detected'},
    fire_detected:    {color:'danger', label:'Fire detected'},
    normal:           {color:'success',label:'Normal'},
    system:           {color:'muted',  label:'System'},
  };
  const {color, label} = typeStyles[e.event_type] || {color:'muted', label:e.event_type};
  const dt   = e.created_at ? new Date(e.created_at) : null;
  const time = dt ? dt.toLocaleTimeString('en-GB', {hour12:false}) : '';
  const date = dt ? dt.toLocaleDateString('en-GB', {day:'2-digit',month:'short'}) : '';
  const loc  = escapeHtml(e.camera_name || 'Unknown');
  const personPart = (e.person_category && e.person_category !== 'unknown')
    ? escapeHtml(e.person_category) + ' · ' : '';
  const details = e.details ? ' (' + escapeHtml(e.details) + ')' : '';
  const conf = e.confidence || 0;
  const [confCls, confLabel] = conf >= 0.75
    ? ['conf-high','Confirmed'] : conf >= 0.5
    ? ['conf-mid','Likely'] : ['conf-low','Possible'];
  const snap = e.snapshot_path
    ? `<img class="log-snap" src="/snapshots/${encodeURIComponent(e.snapshot_path)}"` +
      ` data-snap="${escapeHtml(e.snapshot_path)}"` +
      ` data-title="${escapeHtml(label)}"` +
      ` data-time="${escapeHtml(time)}">`
    : '';
  return `<div class="board-row row-${color}" data-type="${e.event_type || ''}">` +
    `<span class="board-cell cell-time">` +
      `<span class="cell-date">${escapeHtml(date)}</span>` +
      `<span class="cell-clock">${escapeHtml(time)}</span>` +
    `</span>` +
    `<span class="board-cell cell-loc">${loc}</span>` +
    `<span class="board-cell cell-event">${personPart}${escapeHtml(label)}${details}</span>` +
    `<span class="board-cell cell-conf ${confCls}">${confLabel}</span>` +
    `<span class="board-cell cell-snap">${snap}</span>` +
  `</div>`;
}'''

NEW_LOAD_RECENT = r'''function loadRecentEvents() {
  fetch(API + '/api/events?limit=10').then(r => r.json()).then(events => {
    const container = document.getElementById('logRows');
    container.innerHTML = events.map(e => eventRow(e)).join('');
    const rows = container.querySelectorAll('.board-row');
    let stagger = 0;
    rows.forEach((row, i) => {
      const id = events[i]?.event_id;
      if (id != null && !seenEventIds.has(id) && stagger < 6) {
        row.classList.add('board-row--enter');
        splitFlip(row, stagger * 35);
        row.addEventListener('animationend', () => {
          row.classList.remove('board-row--enter');
        }, {once: true});
        stagger++;
      }
    });
    seenEventIds = new Set(events.map(e => e.event_id).filter(id => id != null));
  });
}'''

FLIP_HELPER = r'''function splitFlip(row, baseDelay) {
  row.querySelectorAll('.cell-clock,.cell-loc,.cell-event').forEach((cell, ci) => {
    const txt = cell.textContent;
    cell.innerHTML = [...txt].map((ch, i) => {
      const delay = baseDelay + ci * 60 + i * 16;
      const safe = ch === '<' ? '&lt;' : ch === '&' ? '&amp;' : ch === ' ' ? ' ' : ch;
      return `<span class="flip-char" style="animation-delay:${delay}ms">${safe}</span>`;
    }).join('');
  });
}'''

# Patch script_main: replace eventRow function
def replace_fn(src, fn_name, new_fn):
    m = re.search(r'function ' + fn_name + r'\s*\(', src)
    if not m:
        print(f'  WARN: {fn_name} not found')
        return src
    start = m.start()
    depth = 0
    i = start
    while i < len(src):
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return src[:start] + new_fn + src[i+1:]
        i += 1
    return src

script_main_patched = replace_fn(script_main, 'eventRow',         NEW_EVENT_ROW)
script_main_patched = replace_fn(script_main_patched, 'loadRecentEvents', NEW_LOAD_RECENT)
# Insert splitFlip before loadRecentEvents
script_main_patched = script_main_patched.replace(
    'function loadRecentEvents()',
    FLIP_HELPER + '\nfunction loadRecentEvents()'
)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = '''<style>
/* Concourse world — HomeShield */
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#16213e;--surface:#1e2d4a;--surface-2:#243356;
  --border:rgba(255,255,255,0.10);--border-mid:rgba(255,255,255,0.16);
  --border-bright:rgba(255,255,255,0.24);
  --text:#f0f4ff;--text-2:#8896b0;--text-3:#4a5870;
  --amber:#f5a200;--danger:#e8002a;--success:#00d68f;
  --primary:#f5a200;
  --primary-dim:rgba(245,162,0,0.12);
  --danger-dim:rgba(232,0,42,0.12);
  --success-dim:rgba(0,214,143,0.10);
  --amber-dim:rgba(245,162,0,0.12);
  --sidebar-w:220px;--topbar-h:52px;--r:3px;
}
body{
  font-family:'Rubik',system-ui,sans-serif;
  background:var(--bg);color:var(--text);
  min-height:100vh;overflow-x:hidden;font-size:14px;line-height:1.5;
}
/* AUTH */
.auth-screen{
  position:fixed;inset:0;z-index:1000;
  display:flex;align-items:center;justify-content:center;background:var(--bg);
}
.auth-shader{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0.3}
.auth-card{
  position:relative;z-index:1;width:340px;max-width:calc(100vw - 40px);
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:36px 32px;
}
.auth-brand{text-align:center;margin-bottom:28px}
.auth-brand img{width:36px;height:36px;margin-bottom:10px;opacity:0.9}
.auth-brand h1{
  font-family:'Chakra Petch',sans-serif;font-size:20px;font-weight:700;
  letter-spacing:0.5px;color:var(--text);margin-bottom:5px;
}
.auth-brand p{font-size:12px;color:var(--text-2)}
.auth-error{
  background:var(--danger-dim);border:1px solid rgba(232,0,42,0.3);
  color:var(--danger);border-radius:var(--r);padding:9px 12px;
  font-size:12px;margin-bottom:14px;
}
.auth-card label{
  display:block;font-size:9px;color:var(--text-2);
  font-family:'Chakra Petch',sans-serif;text-transform:uppercase;
  letter-spacing:1px;font-weight:600;margin-bottom:12px;
}
.auth-card label input{
  display:block;width:100%;margin-top:6px;padding:9px 11px;
  background:var(--bg);border:1px solid var(--border);
  border-radius:var(--r);color:var(--text);font-size:13px;
  font-family:'Rubik',sans-serif;outline:none;transition:border-color .15s;
}
.auth-card label input:focus{border-color:var(--amber)}
.auth-hint{font-size:11px;color:var(--text-3);text-align:center;margin-top:14px;line-height:1.6}
.auth-hint b{color:var(--text-2)}
/* ROLE GATES */
body:not(.is-admin):not(.is-guest) .role-admin-only,
body:not(.is-admin):not(.is-guest) .role-any{display:none !important}
body.is-guest .role-admin-only{display:none !important}
body:not(.is-admin):not(.is-guest) .sidebar,
body:not(.is-admin):not(.is-guest) .main-content,
body:not(.is-admin):not(.is-guest) .topbar,
body:not(.is-admin):not(.is-guest) .sidebar-backdrop{visibility:hidden !important}
/* SIDEBAR */
.sidebar{
  position:fixed;top:0;left:0;bottom:0;width:var(--sidebar-w);
  background:var(--bg);border-right:1px solid var(--border);
  display:flex;flex-direction:column;z-index:200;
}
.sidebar-logo{
  display:flex;align-items:center;gap:10px;
  padding:14px 16px;border-bottom:1px solid var(--border);
}
.logo-mark{width:30px;height:30px;flex-shrink:0}
.logo-mark img{width:30px;height:30px;object-fit:contain}
.logo-text{line-height:1.2}
.logo-name{
  font-family:'Chakra Petch',sans-serif;font-size:14px;font-weight:700;
  letter-spacing:0.5px;color:var(--text);
}
.logo-ver{font-size:9px;color:var(--text-3);letter-spacing:0.5px;margin-top:1px}
.sidebar-section{
  font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1.8px;
  color:var(--text-3);font-family:'Chakra Petch',sans-serif;
  padding:16px 16px 5px;display:block;
}
.nav-item{
  display:flex;align-items:center;gap:9px;padding:9px 14px 9px 16px;
  border:none;background:none;color:var(--text-2);
  font-family:'Rubik',sans-serif;font-size:13px;font-weight:500;
  cursor:pointer;width:100%;text-align:left;position:relative;
  transition:color .15s,background .15s;
}
.nav-item:hover{color:var(--text);background:rgba(255,255,255,0.04)}
.nav-item.active{color:var(--text);background:rgba(245,162,0,0.08)}
.nav-item.active::before{
  content:'';position:absolute;left:0;top:5px;bottom:5px;
  width:2px;background:var(--amber);
}
.nav-icon{width:24px;height:24px;flex-shrink:0;display:flex;align-items:center;justify-content:center}
.nav-icon svg{width:14px;height:14px}
.nav-icon img{width:15px;height:15px;opacity:0.5;filter:brightness(0) invert(1)}
.nav-item.active .nav-icon img{opacity:0.85}
.sidebar-bottom{margin-top:auto;padding:10px 12px;border-top:1px solid var(--border)}
.sys-status{
  display:flex;align-items:center;gap:8px;padding:7px 9px;
  border:1px solid var(--border);border-radius:var(--r);margin-bottom:7px;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.status-dot{
  width:6px;height:6px;border-radius:50%;flex-shrink:0;
  background:var(--success);animation:blink 2s infinite;
}
.status-dot.off{background:var(--text-3);animation:none}
.s-label{font-size:9px;color:var(--text-3);text-transform:uppercase;letter-spacing:1px;font-family:'Chakra Petch',sans-serif}
.s-val{font-size:11px;font-weight:600;color:var(--text);margin-top:1px}
/* SIDEBAR USER */
.sidebar-user{
  margin:0 0 8px;padding:7px 9px;
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);display:flex;align-items:center;gap:8px;
}
.sidebar-user .user-avatar{
  width:24px;height:24px;border-radius:50%;flex-shrink:0;
  background:var(--surface-2);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  font-size:10px;font-weight:700;color:var(--amber);
  font-family:'Chakra Petch',sans-serif;
}
.sidebar-user .user-meta{flex:1;min-width:0}
.sidebar-user .user-name{font-size:11px;color:var(--text);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sidebar-user .user-role{font-size:9px;color:var(--text-3);text-transform:uppercase;letter-spacing:1px;font-family:'Chakra Petch',sans-serif;margin-top:1px}
.sidebar-user .user-role.admin{color:var(--amber)}
.sidebar-user .logout-btn{
  background:transparent;border:1px solid var(--border);
  color:var(--text-2);padding:3px 7px;border-radius:var(--r);
  font-size:9px;font-family:'Chakra Petch',sans-serif;
  letter-spacing:0.5px;cursor:pointer;transition:all 0.15s;
}
.sidebar-user .logout-btn:hover{color:var(--danger);border-color:rgba(232,0,42,0.4)}
/* TOPBAR */
.topbar{
  position:fixed;top:0;left:var(--sidebar-w);right:0;height:var(--topbar-h);
  background:var(--bg);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;z-index:100;gap:14px;
}
.topbar-title{
  font-size:11px;font-weight:600;letter-spacing:1.2px;flex:1;
  min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  font-family:'Chakra Petch',sans-serif;text-transform:uppercase;color:var(--text-2);
}
.topbar-chips{display:flex;align-items:center;gap:5px}
.topbar-chip{
  display:flex;align-items:center;gap:7px;padding:5px 11px;
  border:1px solid var(--border);border-radius:var(--r);transition:border-color .2s;
}
.chip-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.chip-dot.green{background:var(--success)}
.chip-dot.red{background:var(--danger)}
.chip-dot.blue{background:var(--text-2)}
.chip-content{display:flex;flex-direction:column;align-items:flex-end}
.chip-label{font-size:8px;color:var(--text-3);font-family:'Chakra Petch',sans-serif;text-transform:uppercase;letter-spacing:1px}
.chip-val{font-size:13px;font-weight:700;letter-spacing:-0.5px;line-height:1.1;font-family:'Chakra Petch',sans-serif}
.burger-btn{
  display:none;background:transparent;border:1px solid var(--border);
  color:var(--text);padding:7px 9px;border-radius:var(--r);
  cursor:pointer;align-items:center;justify-content:center;transition:background .15s;flex-shrink:0;
}
.burger-btn:hover{background:var(--surface)}
.sidebar-backdrop{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);
  z-index:199;opacity:0;transition:opacity .2s;
}
.sidebar-backdrop.show{display:block;opacity:1}
/* MAIN */
.main{margin-left:var(--sidebar-w);padding-top:var(--topbar-h);min-height:100vh}
.content{padding:18px 22px}
@keyframes pageEnter{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.page{display:none}
.page.active{display:block;animation:pageEnter 160ms cubic-bezier(0.16,1,0.3,1) both}
/* BUTTONS */
.btn{
  padding:7px 13px;border-radius:var(--r);
  border:1px solid var(--border);background:var(--surface);
  cursor:pointer;font-size:11px;font-weight:600;
  font-family:'Chakra Petch',sans-serif;letter-spacing:0.5px;
  color:var(--text);transition:all .15s;
  display:inline-flex;align-items:center;gap:6px;
}
.btn:hover{background:var(--surface-2);border-color:var(--border-mid)}
.btn-primary{background:var(--amber);color:#000;border-color:var(--amber);font-weight:700}
.btn-primary:hover{background:#ffb300;border-color:#ffb300}
.btn-danger{background:var(--danger-dim);color:var(--danger);border-color:rgba(232,0,42,0.28)}
.btn-danger:hover{background:rgba(232,0,42,0.2)}
.btn-sm{font-size:10px;padding:4px 9px}
.btn-full{width:100%;justify-content:center}
/* METRICS */
.metrics{
  display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
  border:1px solid var(--border);border-radius:var(--r) var(--r) 0 0;
  background:var(--border);overflow:hidden;
}
.metric{background:var(--surface);padding:13px 16px;display:flex;flex-direction:column;gap:2px}
.metric-label{
  font-size:9px;color:var(--text-3);text-transform:uppercase;
  letter-spacing:1.2px;font-family:'Chakra Petch',sans-serif;font-weight:600;
}
.metric-value{
  font-size:22px;font-weight:700;font-family:'Chakra Petch',sans-serif;
  letter-spacing:-0.5px;line-height:1;color:var(--text);transition:color 0.2s;
}
.metric-value.success{color:var(--success)}
.metric-value.danger{color:var(--danger)}
@keyframes metricFlash{0%,40%{color:var(--amber)}100%{color:inherit}}
.metric-value--changed{animation:metricFlash 500ms ease both}
/* DET BAR */
.det-bar{
  display:flex;gap:1px;background:var(--border);
  border:1px solid var(--border);border-top:none;
  border-radius:0 0 var(--r) var(--r);overflow:hidden;margin-bottom:10px;
}
.det-pill{
  flex:1;display:flex;align-items:center;gap:7px;
  padding:7px 12px;background:var(--surface);
  font-family:'Chakra Petch',sans-serif;font-size:10px;cursor:default;
}
.det-pill.on .det-state{color:var(--success)}
.det-pill.off{opacity:0.55}
.det-pill.off .det-state{color:var(--text-3)}
.det-icon{font-size:12px}
.det-name{font-weight:600;letter-spacing:0.3px;color:var(--text-2)}
.det-state{margin-left:auto;font-weight:700;font-size:9px;letter-spacing:1px}
/* BOARD */
.board-wrap{
  border:1px solid var(--border);border-radius:var(--r);
  overflow:hidden;margin-bottom:14px;
}
.board-head{
  display:grid;grid-template-columns:130px 150px 1fr 90px 50px;
  background:var(--surface);border-bottom:1px solid var(--border);
}
.bhcell{
  padding:8px 12px;font-size:9px;font-weight:600;
  font-family:'Chakra Petch',sans-serif;text-transform:uppercase;
  letter-spacing:1.5px;color:var(--text-3);
  border-right:1px solid var(--border);
}
.bhcell:last-child{border-right:none}
.board-row{
  display:grid;grid-template-columns:130px 150px 1fr 90px 50px;
  border-bottom:1px solid var(--border);
  background:var(--bg);transition:background .12s;
}
.board-row:last-child{border-bottom:none}
.board-row:hover{background:var(--surface)}
.board-row.row-danger{border-left:2px solid var(--danger)}
.board-row.row-amber{border-left:2px solid var(--amber)}
.board-row.row-success{border-left:2px solid var(--success)}
.board-row.row-muted{border-left:2px solid transparent}
.board-cell{
  padding:10px 12px;font-size:12px;color:var(--text-2);
  border-right:1px solid var(--border);
  display:flex;align-items:center;overflow:hidden;
}
.board-cell:last-child{border-right:none;justify-content:center}
.cell-time{flex-direction:column;gap:1px;align-items:flex-start}
.cell-date{font-size:9px;color:var(--text-3);font-family:'Chakra Petch',sans-serif;letter-spacing:0.5px}
.cell-clock{font-family:'Chakra Petch',sans-serif;font-size:12px;color:var(--text);letter-spacing:0.5px}
.cell-loc{font-weight:500;color:var(--text)}
.cell-event{color:var(--text)}
.board-row.row-danger .cell-event{color:var(--danger)}
.board-row.row-amber .cell-event{color:var(--amber)}
.cell-conf{justify-content:flex-start}
.conf-high{color:var(--success);font-family:'Chakra Petch',sans-serif;font-size:9px;letter-spacing:0.5px;font-weight:700}
.conf-mid{color:var(--amber);font-family:'Chakra Petch',sans-serif;font-size:9px;letter-spacing:0.5px;font-weight:700}
.conf-low{color:var(--text-3);font-family:'Chakra Petch',sans-serif;font-size:9px;letter-spacing:0.5px}
.board-empty{
  padding:28px;text-align:center;font-size:11px;color:var(--text-3);
  font-family:'Chakra Petch',sans-serif;letter-spacing:1px;text-transform:uppercase;
}
/* SPLIT-FLAP */
@keyframes flipChar{
  0%  {opacity:0;transform:rotateX(-90deg) scaleY(0.5)}
  60% {opacity:1;transform:rotateX(8deg) scaleY(1.02)}
  100%{opacity:1;transform:rotateX(0) scaleY(1)}
}
.flip-char{display:inline-block;transform-origin:center 60%;backface-visibility:hidden}
.board-row--enter .flip-char{animation:flipChar 180ms cubic-bezier(0.16,1,0.3,1) both}
@media(prefers-reduced-motion:reduce){.board-row--enter .flip-char{animation:none}}
/* LOG SNAP */
.log-snap{
  width:40px;height:30px;object-fit:cover;
  border-radius:2px;border:1px solid var(--border);
  cursor:pointer;display:block;transition:opacity .15s;
}
.log-snap:hover{opacity:0.75}
/* LOG HEADER */
.log-header{
  display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;
}
.log-header h3{
  font-size:12px;font-weight:700;font-family:'Chakra Petch',sans-serif;
  letter-spacing:0.5px;color:var(--text);text-transform:uppercase;
}
.log-filters{display:flex;gap:6px}
.log-filters select{padding:5px 8px;font-size:11px}
/* CAMERA SECTION */
.cam-section{margin-bottom:14px}
.cam-section-head{
  display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;
}
.cam-section-title{
  font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;
  color:var(--text-3);font-family:'Chakra Petch',sans-serif;
}
.live-layout{
  display:flex;gap:1px;border:1px solid var(--border);border-radius:var(--r);
  background:var(--border);overflow:hidden;height:190px;
}
.hero-col{flex:0 0 280px}
.hero-feed{display:flex;flex-direction:column;height:100%}
.hero-screen{
  flex:1;overflow:hidden;position:relative;background:var(--surface);
  display:flex;align-items:center;justify-content:center;
}
.hero-overlay{
  position:absolute;inset:0;display:flex;
  align-items:flex-start;padding:8px;z-index:1;pointer-events:none;
}
.live-pill{
  display:flex;align-items:center;gap:5px;background:rgba(0,0,0,0.72);
  padding:3px 7px;border-radius:2px;
  font-size:9px;font-family:'Chakra Petch',sans-serif;
  font-weight:700;letter-spacing:1.5px;color:var(--text);
}
.live-dot{width:5px;height:5px;border-radius:50%;background:var(--danger);animation:blink 1s infinite;flex-shrink:0}
.live-dot.offline{background:var(--text-3);animation:none}
.hero-bar{
  background:var(--surface);border-top:1px solid var(--border);
  padding:5px 9px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
}
.hero-cam-name{font-size:10px;font-weight:700;color:var(--text);font-family:'Chakra Petch',sans-serif}
.hero-cam-loc{font-size:9px;color:var(--text-3);margin-top:1px}
.thumb-col{flex:1;overflow:hidden;display:flex;flex-direction:column}
.thumb-panel{
  flex:1;display:flex;gap:1px;background:var(--border);
  overflow-x:auto;overflow-y:hidden;
}
.thumb-card{
  flex:0 0 140px;cursor:pointer;overflow:hidden;
  display:flex;flex-direction:column;transition:opacity .12s;
}
.thumb-card.selected{outline:2px solid var(--amber);outline-offset:-2px}
.thumb-card.alert-cam{opacity:0.5}
.thumb-screen{flex:1;overflow:hidden;background:var(--surface)}
.thumb-screen img{width:100%;height:100%;object-fit:cover;display:block}
.thumb-offline{
  display:flex;align-items:center;justify-content:center;height:100%;
  font-size:9px;color:var(--text-3);font-family:'Chakra Petch',sans-serif;
  letter-spacing:1px;text-transform:uppercase;
}
.thumb-bar{
  background:var(--surface);border-top:1px solid var(--border);
  padding:4px 7px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
}
.thumb-name{font-size:9px;font-family:'Chakra Petch',sans-serif;color:var(--text-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.live-divider{width:3px;background:var(--bg);cursor:col-resize;flex-shrink:0;transition:background .12s}
.live-divider:hover{background:var(--amber)}
.feed-badge{
  font-size:8px;font-weight:700;letter-spacing:1.5px;padding:2px 5px;
  border-radius:1px;font-family:'Chakra Petch',sans-serif;flex-shrink:0;
}
.feed-badge.live{background:var(--danger);color:#fff}
.feed-badge.alert{background:var(--danger-dim);color:var(--danger)}
.add-cam-panel{padding:8px;background:var(--surface);border-top:1px solid var(--border);flex-shrink:0}
.add-cam-label{font-size:9px;color:var(--text-3);font-family:'Chakra Petch',sans-serif;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px}
#addCamForm input{
  display:block;width:100%;margin-bottom:4px;padding:5px 8px;
  background:var(--bg);border:1px solid var(--border);border-radius:var(--r);
  color:var(--text);font-size:11px;outline:none;
}
#addCamForm input:focus{border-color:var(--amber)}
/* FORMS */
input,select,textarea{
  font-family:'Rubik',sans-serif;background:var(--surface);
  border:1px solid var(--border);border-radius:var(--r);
  color:var(--text);outline:none;transition:border-color .15s;
}
input:focus,select:focus,textarea:focus{border-color:var(--amber)}
.form-group{margin-bottom:13px}
.form-group label,.form-label{
  display:block;font-size:9px;color:var(--text-2);
  font-family:'Chakra Petch',sans-serif;text-transform:uppercase;
  letter-spacing:1px;font-weight:600;margin-bottom:5px;
}
.form-input,.form-group input,.form-group select,.form-group textarea{
  width:100%;padding:8px 10px;background:var(--surface);
  border:1px solid var(--border);border-radius:var(--r);
  color:var(--text);font-size:12px;outline:none;transition:border-color .15s;
}
.form-input:focus,.form-group input:focus,.form-group select:focus,.form-group textarea:focus{
  border-color:var(--amber);
}
/* SETTINGS */
.settings-grid{display:grid;grid-template-columns:1fr 360px;gap:14px;align-items:start}
.zone-canvas-wrap{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:16px}
.zone-list{display:flex;flex-direction:column;gap:6px;margin-top:8px}
.zone-item{
  display:flex;align-items:center;gap:8px;padding:9px 11px;
  background:var(--bg);border:1px solid var(--border);border-radius:var(--r);
}
.zone-name{flex:1;font-size:12px;color:var(--text)}
.zone-type{
  font-size:9px;font-family:'Chakra Petch',sans-serif;text-transform:uppercase;
  letter-spacing:1px;padding:2px 7px;border-radius:var(--r);
}
.zone-type.danger{background:var(--danger-dim);color:var(--danger)}
.zone-type.safe{background:var(--success-dim);color:var(--success)}
.persons-layout{display:grid;grid-template-columns:1fr 300px;gap:18px;align-items:start}
.persons-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.person-card{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
  padding:10px;text-align:center;
}
.person-card img{width:54px;height:54px;border-radius:50%;object-fit:cover;margin-bottom:7px;border:2px solid var(--border)}
.person-name{font-size:11px;font-weight:700;color:var(--text);font-family:'Chakra Petch',sans-serif}
.person-cat{font-size:9px;color:var(--text-3);text-transform:uppercase;letter-spacing:0.8px;margin-top:2px}
.person-actions{display:flex;gap:4px;justify-content:center;margin-top:7px}
.persons-empty{color:var(--text-3);font-size:12px;padding:22px;text-align:center;grid-column:1/-1}
.settings-shell{
  display:grid;grid-template-columns:175px 1fr;gap:1px;
  background:var(--border);border:1px solid var(--border);border-radius:var(--r);
  overflow:hidden;min-height:calc(100vh - var(--topbar-h) - 80px);
}
.set-nav{background:var(--surface);padding:7px 0}
.set-nav-section{
  font-size:8px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;
  color:var(--text-3);font-family:'Chakra Petch',sans-serif;padding:12px 13px 4px;display:block;
}
.set-nav-item{
  display:flex;align-items:center;gap:7px;width:100%;
  padding:8px 13px;border:none;background:none;
  color:var(--text-2);font-family:'Rubik',sans-serif;
  font-size:12px;font-weight:500;cursor:pointer;text-align:left;
  transition:color .15s,background .15s;position:relative;
}
.set-nav-item:hover{color:var(--text);background:rgba(255,255,255,0.03)}
.set-nav-item.active{color:var(--text);background:var(--primary-dim)}
.set-nav-item.active::before{
  content:'';position:absolute;left:0;top:4px;bottom:4px;width:2px;background:var(--amber);
}
.set-nav-item img{width:13px;height:13px;opacity:0.45;filter:brightness(0) invert(1)}
.set-nav-item.active img{opacity:0.8}
.set-content{background:var(--bg);padding:18px;overflow-y:auto}
.set-pane{display:none}
.set-pane.active{display:block}
.set-pane-head{margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)}
.set-pane-head h2{
  font-size:15px;font-weight:700;font-family:'Chakra Petch',sans-serif;
  color:var(--text);margin-bottom:3px;
}
.set-pane-head p{font-size:12px;color:var(--text-2)}
.settings-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:16px;margin-bottom:12px;
}
.s-card-head{
  display:flex;gap:11px;align-items:flex-start;
  margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--border);
}
.s-icon{
  width:30px;height:30px;border-radius:var(--r);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.s-icon img{width:16px;height:16px;filter:brightness(0) invert(1);opacity:0.65}
.s-icon svg{width:15px;height:15px;color:var(--text-2)}
.icon-violet{background:rgba(139,92,246,0.14)}
.icon-blue{background:rgba(99,179,237,0.11)}
.icon-red{background:var(--danger-dim)}
.icon-orange{background:rgba(245,158,11,0.11)}
.icon-green{background:var(--success-dim)}
.icon-amber{background:var(--amber-dim)}
.icon-info{background:rgba(255,255,255,0.05)}
.s-head-text h3{font-size:12px;font-weight:700;color:var(--text);font-family:'Chakra Petch',sans-serif;margin-bottom:2px}
.s-head-text p{font-size:11px;color:var(--text-2)}
.slider-group{margin-bottom:13px}
.slider-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}
.slider-head label{
  font-size:9px;color:var(--text-2);font-family:'Chakra Petch',sans-serif;
  text-transform:uppercase;letter-spacing:1px;font-weight:600;
}
.slider-val{font-size:12px;font-weight:700;color:var(--amber);font-family:'Chakra Petch',sans-serif}
.slider{
  -webkit-appearance:none;appearance:none;width:100%;height:3px;
  background:var(--surface-2);border-radius:0;outline:none;cursor:pointer;
}
.slider::-webkit-slider-thumb{
  -webkit-appearance:none;width:13px;height:13px;border-radius:50%;
  background:var(--amber);cursor:pointer;border:2px solid var(--bg);
}
.slider::-moz-range-thumb{
  width:13px;height:13px;border-radius:50%;
  background:var(--amber);cursor:pointer;border:2px solid var(--bg);
}
.radio-group{display:flex;flex-direction:column;gap:7px}
.radio-opt{
  display:flex;align-items:flex-start;gap:9px;cursor:pointer;
  padding:9px 11px;border:1px solid var(--border);border-radius:var(--r);transition:border-color .15s;
}
.radio-opt:hover{border-color:var(--border-mid)}
.radio-opt input:checked ~ .radio-mark{background:var(--amber);border-color:var(--amber)}
.radio-mark{
  width:13px;height:13px;border-radius:50%;flex-shrink:0;
  border:2px solid var(--border);background:transparent;transition:all .15s;margin-top:1px;
}
.radio-text{font-size:12px;color:var(--text)}
.radio-text small{display:block;font-size:10px;color:var(--text-2);margin-top:2px}
input[type=radio]{position:absolute;opacity:0;pointer-events:none}
.save-bar{
  position:fixed;bottom:0;left:var(--sidebar-w);right:0;padding:10px 22px;
  background:var(--surface);border-top:1px solid var(--border);
  display:none;align-items:center;justify-content:space-between;gap:12px;z-index:90;
}
.save-bar.dirty{display:flex}
.save-status{display:flex;align-items:center;gap:7px}
.save-dot{width:5px;height:5px;border-radius:50%;background:var(--amber);animation:blink 1.5s infinite;flex-shrink:0}
.save-actions{display:flex;gap:7px}
.cam-list{display:flex;flex-direction:column;gap:5px;margin-bottom:9px}
.cam-item{
  display:flex;align-items:center;gap:8px;padding:9px 11px;
  background:var(--bg);border:1px solid var(--border);border-radius:var(--r);
}
.cam-item-info{flex:1;min-width:0}
.cam-name{font-size:12px;font-weight:600;color:var(--text);font-family:'Chakra Petch',sans-serif}
.cam-url{font-size:10px;color:var(--text-3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
.modal-overlay{
  position:fixed;inset:0;z-index:500;background:rgba(0,0,0,0.75);
  display:none;align-items:center;justify-content:center;
}
.modal-overlay.show{display:flex}
@keyframes modalCardIn{from{opacity:0;transform:scale(0.97) translateY(5px)}to{opacity:1;transform:none}}
.modal{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
  padding:20px;max-width:min(600px,95vw);width:100%;position:relative;
  animation:modalCardIn 180ms cubic-bezier(0.16,1,0.3,1) both;
}
.modal-close{
  position:absolute;top:10px;right:12px;
  background:transparent;border:1px solid var(--border);
  color:var(--text-2);width:24px;height:24px;border-radius:var(--r);
  cursor:pointer;font-size:15px;display:flex;align-items:center;justify-content:center;
  transition:all .15s;
}
.modal-close:hover{color:var(--text);border-color:var(--border-mid)}
.modal h3{font-size:13px;font-weight:700;font-family:'Chakra Petch',sans-serif;color:var(--text);margin-bottom:12px}
.modal img{width:100%;border-radius:2px;margin-bottom:10px;border:1px solid var(--border)}
.modal p{font-size:12px;color:var(--text-2);margin-top:6px}
#hs-toast{
  position:fixed;bottom:22px;right:22px;z-index:600;
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
  padding:9px 14px;font-size:11px;font-family:'Chakra Petch',sans-serif;
  letter-spacing:0.3px;color:var(--text);display:none;
}
.users-table{width:100%;border-collapse:collapse;font-size:12px}
.users-table th{
  text-align:left;font-size:9px;color:var(--text-3);text-transform:uppercase;
  letter-spacing:1px;font-family:'Chakra Petch',sans-serif;
  padding:7px 9px;border-bottom:1px solid var(--border);
}
.users-table td{padding:9px;border-bottom:1px solid var(--border);color:var(--text)}
.users-table tr:last-child td{border-bottom:none}
.role-tag{
  display:inline-block;font-size:8px;font-weight:700;letter-spacing:1px;
  padding:2px 6px;border-radius:var(--r);font-family:'Chakra Petch',sans-serif;
}
.role-tag.admin{background:var(--amber-dim);color:var(--amber)}
.role-tag.guest{background:rgba(120,136,168,0.12);color:var(--text-2)}
.user-actions{display:flex;gap:4px;flex-wrap:wrap}
.set-nav-item .det-dot{
  margin-left:auto;width:6px;height:6px;border-radius:50%;
  background:var(--text-3);flex-shrink:0;transition:background .15s;
}
.set-nav-item .det-dot.on{background:var(--success);animation:blink 2.2s infinite}
.set-nav-item .det-dot.off{background:var(--danger)}
.mustchange-label{
  display:inline-flex !important;align-items:center;gap:8px;
  margin-bottom:0 !important;padding:0;font-size:12px !important;
  color:var(--text-2) !important;font-family:'Rubik',sans-serif !important;
  font-weight:500 !important;text-transform:none !important;
  letter-spacing:.02em !important;white-space:nowrap;cursor:pointer;
  text-align:left !important;justify-content:flex-start !important;width:auto !important;
}
.mustchange-label input[type=checkbox]{margin:0;width:14px;height:14px;cursor:pointer;flex-shrink:0}
select option,.form-group select option,.form-input option{background:#16213e !important;color:#f0f4ff !important}
select option:checked,.form-group select option:checked,.form-input option:checked{background:var(--amber) !important;color:#000 !important}
select option:disabled,.form-group select option:disabled{color:var(--text-3) !important}
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none;appearance:none}
input[type=number]{-moz-appearance:textfield}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border-mid);border-radius:2px}
::selection{background:rgba(245,162,0,0.22);color:var(--text)}
:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
@media(max-width:900px){
  .sidebar{transform:translateX(-100%);transition:transform .25s cubic-bezier(0.16,1,0.3,1)}
  .sidebar.open{transform:translateX(0)}
  .burger-btn{display:flex}
  .main{margin-left:0}
  .topbar{left:0}
  .save-bar{left:0}
  .metrics{grid-template-columns:repeat(2,1fr)}
  .settings-grid{grid-template-columns:1fr}
  .persons-layout{grid-template-columns:1fr}
  .settings-shell{grid-template-columns:1fr}
  .set-nav{display:none}
  .board-head,.board-row{grid-template-columns:90px 1fr 80px 42px}
  .bhcell:nth-child(2),.board-cell.cell-loc{display:none}
  .live-layout{height:150px}
  .hero-col{flex:0 0 200px}
}
@media(max-width:480px){
  .content{padding:12px 14px}
  .topbar{padding:0 12px}
  .topbar-chips{display:none}
  .metrics{grid-template-columns:1fr 1fr}
  .board-head,.board-row{grid-template-columns:78px 1fr 40px}
  .bhcell:nth-child(n+4),.board-cell:nth-child(n+4){display:none}
  .live-layout{flex-direction:column;height:280px}
  .hero-col{flex:0 0 160px}
}
</style>'''

# ── HEAD ─────────────────────────────────────────────────────────────────────
HEAD = '''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HomeShield — Smart Surveillance Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=Rubik:wght@400;500;600;700&display=swap" rel="stylesheet">
''' + CSS

# ── BODY ─────────────────────────────────────────────────────────────────────
BODY = '''
''' + script_img_error + '''
</head>
<body>

<!-- ══ LOGIN OVERLAY ══ -->
<div id="loginScreen" class="auth-screen">
  <canvas id="loginShader" class="auth-shader" aria-hidden="true"></canvas>
  <div class="auth-card">
    <div class="auth-brand">
      <img src="/icons/LOGO.svg" alt="HomeShield" onerror="this.style.display='none'">
      <h1>HomeShield</h1>
      <p>Sign in to continue</p>
    </div>
    <div id="loginError" class="auth-error" hidden></div>
    <form id="loginForm" autocomplete="on" onsubmit="event.preventDefault(); doLogin();">
      <label>Username
        <input type="text" id="loginUsername" autocomplete="username" autocapitalize="off" autocorrect="off" spellcheck="false" required>
      </label>
      <label>Password
        <input type="password" id="loginPassword" autocomplete="current-password" required>
      </label>
      <button type="submit" class="btn btn-primary btn-full">Sign in</button>
    </form>
    <div class="auth-hint">First-run default: <b>admin / admin</b><br>You\'ll be asked to pick a new password on first login.</div>
  </div>
</div>

<!-- ══ CHANGE-PASSWORD OVERLAY ══ -->
<div id="changePwScreen" class="auth-screen" hidden>
  <div class="auth-card">
    <div class="auth-brand">
      <h1>Set a new password</h1>
      <p>Required before you can use the dashboard</p>
    </div>
    <div id="changePwError" class="auth-error" hidden></div>
    <form id="changePwForm" autocomplete="off" onsubmit="event.preventDefault(); doChangePassword();">
      <label id="currentPwLabel" hidden>Current password
        <input type="password" id="currentPw" autocomplete="current-password">
      </label>
      <label>New password
        <input type="password" id="newPw1" autocomplete="new-password" minlength="4" required>
      </label>
      <label>Confirm new password
        <input type="password" id="newPw2" autocomplete="new-password" minlength="4" required>
      </label>
      <button type="submit" class="btn btn-primary btn-full">Update password</button>
    </form>
  </div>
</div>

<!-- ══ SIDEBAR ══ -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="logo-mark">
      <img src="/icons/LOGO.svg" alt="HomeShield">
    </div>
    <div class="logo-text">
      <div class="logo-name">HomeShield</div>
      <div class="logo-ver">v2.0.0</div>
    </div>
  </div>

  <span class="sidebar-section">Monitor</span>
  <button class="nav-item active" onclick="showPage(\'live\',this)">
    <div class="nav-icon">
      <svg viewBox="0 0 16 16" fill="none"><rect x="1" y="1" width="14" height="10" rx="2" stroke="currentColor" stroke-width="1.4"></rect><path d="M6 14h4M8 11v3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"></path></svg>
    </div>
    Live feeds
  </button>
  <button class="nav-item" onclick="showPage(\'events\',this)">
    <div class="nav-icon">
      <svg viewBox="0 0 16 16" fill="none"><path d="M2 4h12M2 8h8M2 12h5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"></path></svg>
    </div>
    Events
  </button>

  <span class="sidebar-section role-admin-only">Configure</span>
  <button class="nav-item role-admin-only" onclick="showPage(\'zones\',this)">
    <div class="nav-icon">
      <svg viewBox="0 0 16 16" fill="none"><rect x="2" y="2" width="12" height="12" rx="1" stroke="currentColor" stroke-width="1.4"></rect><rect x="5" y="5" width="6" height="6" rx="0.5" stroke="currentColor" stroke-width="1.2" stroke-dasharray="2 1"></rect></svg>
    </div>
    Zones
  </button>
  <button class="nav-item role-admin-only" onclick="showPage(\'persons\',this)">
    <div class="nav-icon">
      <svg viewBox="0 0 16 16" fill="none"><circle cx="8" cy="5.5" r="2.8" stroke="currentColor" stroke-width="1.4"></circle><path d="M2.5 14c0-3 2.5-5 5.5-5s5.5 2 5.5 5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"></path></svg>
    </div>
    Persons
  </button>
  <button class="nav-item role-admin-only" onclick="showPage(\'settings\',this)">
    <div class="nav-icon">
      <svg viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="2.5" stroke="currentColor" stroke-width="1.4"></circle><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.22 3.22l1.42 1.42M11.36 11.36l1.42 1.42M11.36 4.64l-1.42 1.42M4.64 11.36l-1.42 1.42" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"></path></svg>
    </div>
    Settings
  </button>

  <div class="sidebar-bottom">
    <div class="sys-status">
      <div class="status-dot" id="statusDot"></div>
      <div style="flex:1">
        <div class="s-label">System</div>
        <div class="s-val" id="statusText">Starting…</div>
      </div>
    </div>
    <div class="sys-status">
      <div class="status-dot" id="fpsDot" style="background:var(--amber);animation:none"></div>
      <div style="flex:1">
        <div class="s-label">Avg FPS</div>
        <div class="s-val" id="fpsText" style="font-family:\'Chakra Petch\',sans-serif">--</div>
      </div>
    </div>
    <div class="sidebar-user">
      <div class="user-avatar" id="userAvatar">?</div>
      <div class="user-meta">
        <div class="user-name" id="userName">—</div>
        <div class="user-role" id="userRole">—</div>
      </div>
      <button class="logout-btn" onclick="doLogout()">Sign out</button>
    </div>
    <button class="btn btn-danger btn-full role-admin-only" id="toggleBtn" onclick="toggleSystem()">Stop system</button>
  </div>
</aside>

<!-- ══ TOPBAR ══ -->
<header class="topbar">
  <button class="burger-btn" id="burgerBtn" aria-label="Toggle menu" onclick="toggleSidebar()">
    <svg viewBox="0 0 24 24" fill="none" width="18" height="18"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>
  </button>
  <div class="topbar-title" id="topbarTitle">Live feeds</div>
  <div class="topbar-chips">
    <div class="topbar-chip" title="Cameras connected">
      <div class="chip-dot green"></div>
      <div class="chip-content">
        <span class="chip-label">Cameras</span>
        <span class="chip-val" style="color:var(--success)" id="tb-cameras">–</span>
      </div>
    </div>
    <div class="topbar-chip" title="Alerts today">
      <div class="chip-dot red"></div>
      <div class="chip-content">
        <span class="chip-label">Alerts</span>
        <span class="chip-val" style="color:var(--danger)" id="tb-alerts">–</span>
      </div>
    </div>
    <div class="topbar-chip" title="People detected">
      <div class="chip-dot blue"></div>
      <div class="chip-content">
        <span class="chip-label">People</span>
        <span class="chip-val" id="tb-people">–</span>
      </div>
    </div>
  </div>
</header>

<div class="sidebar-backdrop" id="sidebarBackdrop" onclick="closeSidebar()"></div>

<!-- ══ MAIN ══ -->
<main class="main">
<div class="content">

  <!-- LIVE PAGE -->
  <div class="page active" id="page-live">

    <div class="metrics">
      <div class="metric"><div class="metric-label">Cameras online</div><div class="metric-value success" id="m-cameras">–</div></div>
      <div class="metric"><div class="metric-label">Alerts today</div><div class="metric-value danger" id="m-alerts">0</div></div>
      <div class="metric"><div class="metric-label">People detected</div><div class="metric-value" id="m-people">0</div></div>
      <div class="metric"><div class="metric-label">System status</div><div class="metric-value success" id="m-status">Active</div></div>
    </div>

    <div class="det-bar" id="detBar">
      <div class="det-pill" id="det-fall" title="Fall detection">
        <span class="det-icon">&#x1F9D8;</span><span class="det-name">Fall</span><span class="det-state">–</span>
      </div>
      <div class="det-pill" id="det-fire" title="Fire &amp; smoke detection">
        <span class="det-icon">&#x1F525;</span><span class="det-name">Fire</span><span class="det-state">–</span>
      </div>
      <div class="det-pill" id="det-face" title="Face &amp; intruder detection">
        <span class="det-icon">&#x1F464;</span><span class="det-name">Face</span><span class="det-state">–</span>
      </div>
    </div>

    <!-- Cameras -->
    <div class="cam-section">
      <div class="cam-section-head">
        <span class="cam-section-title">Camera feeds</span>
      </div>
      <div class="live-layout" id="liveLayout">
        <div class="hero-col">
          <div class="hero-feed">
            <div class="hero-screen" id="heroScreen">
              <div class="hero-overlay" id="heroOverlay">
                <div class="live-pill"><div class="live-dot"></div><span id="heroPillText">LIVE</span></div>
              </div>
              <span style="color:var(--text-3);font-size:11px;font-family:\'Chakra Petch\',sans-serif;letter-spacing:1px">NO CAMERAS</span>
            </div>
            <div class="hero-bar">
              <div>
                <div class="hero-cam-name" id="heroCamName">–</div>
                <div class="hero-cam-loc" id="heroCamLoc">–</div>
              </div>
              <span class="feed-badge live" id="heroBadge">LIVE</span>
            </div>
          </div>
        </div>
        <div class="live-divider" id="liveDivider" title="Drag to resize \xb7 Double-click to reset"></div>
        <div class="thumb-col" id="thumbCol">
          <div class="thumb-panel" id="thumbPanel"></div>
          <div class="add-cam-panel" id="addCamPanel">
            <div class="add-cam-label">+ Add camera</div>
            <div id="addCamForm">
              <input id="qcName" placeholder="Name (e.g. Front Door)">
              <input id="qcUrl" placeholder="RTSP URL or 0 for webcam">
              <input id="qcLoc" placeholder="Location (e.g. Living Room)">
              <button class="btn btn-primary btn-full" style="font-size:10px;padding:6px" onclick="quickAddCamera()">Add camera</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Board -->
    <div class="log-header">
      <h3>Event board</h3>
      <div style="display:flex;gap:6px">
        <button class="btn btn-sm" onclick="refreshEvents()">Refresh</button>
        <button class="btn btn-sm btn-danger" onclick="clearEvents()">Clear all</button>
      </div>
    </div>
    <div class="board-wrap">
      <div class="board-head">
        <span class="bhcell">Time</span>
        <span class="bhcell">Location</span>
        <span class="bhcell">Event</span>
        <span class="bhcell">Status</span>
        <span class="bhcell"></span>
      </div>
      <div id="logRows"><div class="board-empty">No events yet</div></div>
    </div>
  </div>

  <!-- EVENTS PAGE -->
  <div class="page" id="page-events">
    <div class="log-header">
      <h3>Event history</h3>
      <div class="log-filters">
        <select id="filterType" onchange="loadEvents()">
          <option value="">All types</option>
          <option value="fall_detected">Falls</option>
          <option value="lying_motionless">Lying motionless</option>
          <option value="inactivity">Inactivity</option>
          <option value="zone_entry">Zone entry</option>
          <option value="intruder_detected">Intruder</option>
        </select>
        <select id="filterLimit" onchange="loadEvents()">
          <option value="25">Last 25</option>
          <option value="50" selected>Last 50</option>
          <option value="100">Last 100</option>
        </select>
      </div>
    </div>
    <div class="board-wrap">
      <div class="board-head">
        <span class="bhcell">Time</span>
        <span class="bhcell">Location</span>
        <span class="bhcell">Event</span>
        <span class="bhcell">Status</span>
        <span class="bhcell"></span>
      </div>
      <div id="eventRows"><div class="board-empty">Loading events…</div></div>
    </div>
  </div>

  <!-- ZONES PAGE -->
  <div class="page" id="page-zones">
    <div class="settings-grid">
      <div class="zone-canvas-wrap">
        <h3 style="margin-bottom:9px;font-size:13px;font-weight:700;font-family:\'Chakra Petch\',sans-serif">Define zones</h3>
        <p style="font-size:12px;color:var(--text-2);margin-bottom:13px;line-height:1.6">Select a camera and zone type, then click points on the image to draw a polygon. <strong style="color:var(--danger)">Danger zones</strong> trigger an alert when a child enters. <strong style="color:var(--success)">Safe zones</strong> suppress lying/inactivity alerts.</p>
        <div class="form-group">
          <label>Camera</label>
          <select id="zoneCamSelect" onchange="startZoneRefresh()" style="width:100%"></select>
        </div>
        <div class="form-group"><label>Zone name</label><input id="zoneName" placeholder="e.g. Kitchen stove area"></div>
        <div class="form-group">
          <label>Zone type</label>
          <select id="zoneType" style="width:100%" onchange="drawZonePoints()">
            <option value="danger">Danger — alert when child enters</option>
            <option value="safe">Safe — no alerts for lying/inactivity here</option>
          </select>
        </div>
        <div style="position:relative;width:100%;border-radius:var(--r);overflow:hidden;background:var(--surface);aspect-ratio:4/3;border:1px solid var(--border)">
          <img id="zoneImg" src="" alt="" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;display:block">
          <canvas id="zoneCanvas" width="640" height="480" style="position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair;background:transparent"></canvas>
        </div>
        <div style="margin-top:10px;display:flex;gap:7px">
          <button class="btn btn-primary" onclick="saveZone()">Save zone</button>
          <button class="btn" onclick="clearZonePoints()">Clear points</button>
        </div>
      </div>
      <div>
        <div class="settings-card">
          <h3 style="font-size:13px;font-weight:700;font-family:\'Chakra Petch\',sans-serif;margin-bottom:10px">Active zones</h3>
          <div id="zoneList" class="zone-list"><p style="color:var(--text-2);font-size:12px">No zones defined yet.</p></div>
        </div>
      </div>
    </div>
  </div>

  <!-- PERSONS PAGE -->
  <div class="page" id="page-persons">
    <div class="persons-layout">
      <div>
        <div id="faceRecStatusWrap" style="margin-bottom:6px"></div>
        <h3 style="font-size:13px;font-weight:700;font-family:\'Chakra Petch\',sans-serif;margin-bottom:4px">Registered people</h3>
        <p style="font-size:12px;color:var(--text-2);margin-bottom:16px;line-height:1.6">
          People on this list can be identified by name on the live feed. Anyone detected but <strong>not on this list</strong> will be flagged as an <strong style="color:var(--danger)">intruder</strong>.
        </p>
        <div id="personGrid" class="persons-grid">
          <div class="persons-empty"><strong>No people registered yet</strong><br>Use the panel on the right to register your first person.</div>
        </div>
      </div>
      <div>
        <div class="settings-card">
          <h3 style="font-size:13px;font-weight:700;font-family:\'Chakra Petch\',sans-serif;margin-bottom:12px">Register a person</h3>
          <p style="font-size:11px;color:var(--text-2);margin-bottom:14px;line-height:1.7">Ask the person to stand in front of the selected camera, looking straight at it. Click <em>Capture</em> when they are clearly visible.</p>
          <div class="form-group"><label>Name</label><input id="newPersonName" placeholder="e.g. Grandma Siti" maxlength="40"></div>
          <div class="form-group">
            <label>Category</label>
            <select id="newPersonCategory" style="width:100%">
              <option value="elderly">Elderly</option>
              <option value="adult" selected>Adult</option>
              <option value="child">Child</option>
            </select>
          </div>
          <div class="form-group">
            <label>Capture from camera</label>
            <select id="newPersonCamera" style="width:100%" onchange="startFacePreview()"></select>
          </div>
          <div style="margin-bottom:13px">
            <label style="font-size:9px;color:var(--text-2);display:block;margin-bottom:7px;letter-spacing:1px;text-transform:uppercase;font-family:\'Chakra Petch\',sans-serif;font-weight:600">Live preview</label>
            <div id="facePreviewWrap" style="position:relative;width:100%;aspect-ratio:4/3;background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden">
              <img id="facePreviewImg" src="" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block">
              <canvas id="facePreviewCanvas" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none"></canvas>
              <div id="facePreviewHint" style="position:absolute;left:8px;bottom:8px;font-size:9px;font-family:\'Chakra Petch\',sans-serif;padding:3px 8px;border-radius:var(--r);letter-spacing:0.5px;background:rgba(0,0,0,0.7);color:var(--text-2)">SELECT A CAMERA</div>
            </div>
          </div>
          <button class="btn btn-primary btn-full" onclick="registerPerson()" id="captureBtn" disabled style="margin-top:4px;opacity:0.5">
            <svg viewBox="0 0 16 16" fill="none" style="width:13px;height:13px"><circle cx="8" cy="8" r="3.2" stroke="currentColor" stroke-width="1.5"></circle><path d="M2.5 4h2l1-1.5h5l1 1.5h2A1.5 1.5 0 0 1 15 5.5v7A1.5 1.5 0 0 1 13.5 14h-11A1.5 1.5 0 0 1 1 12.5v-7A1.5 1.5 0 0 1 2.5 4z" stroke="currentColor" stroke-width="1.3"></path></svg>
            Capture from camera
          </button>
          <p style="font-size:10px;color:var(--text-3);margin-top:12px;font-family:\'Chakra Petch\',sans-serif;line-height:1.6">A face thumbnail is saved locally. The 512-dim embedding used for recognition never leaves this device.</p>
        </div>
      </div>
    </div>

    <div style="margin-top:24px;padding-top:18px;border-top:1px solid var(--border)">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;gap:12px;flex-wrap:wrap">
        <div>
          <h3 style="font-size:13px;font-weight:700;font-family:\'Chakra Petch\',sans-serif;margin-bottom:3px;display:flex;align-items:center;gap:8px">
            <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--danger)"></span>
            Intruders seen
          </h3>
          <p style="font-size:12px;color:var(--text-2);line-height:1.6;max-width:600px">People detected whose face did not match anyone registered. Click <strong>Register</strong> to add them, or <strong>Dismiss</strong> to remove.</p>
        </div>
        <label style="font-size:11px;color:var(--text-2);display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none;font-family:\'Chakra Petch\',sans-serif">
          <input type="checkbox" id="intruderShowDismissed" onchange="loadIntruders()" style="accent-color:var(--amber)">
          Show dismissed
        </label>
      </div>
      <div id="intruderGrid" class="persons-grid">
        <div class="persons-empty"><strong>No intruders recorded</strong><br>When the system sees someone it does not recognise, they appear here.</div>
      </div>
    </div>
  </div>

  <!-- SETTINGS PAGE -->
  <div class="page" id="page-settings">
    <div class="settings-shell">
      <aside class="set-nav" id="setNav">
        <div class="set-nav-section">Detection</div>
        <button class="set-nav-item active" data-subject="detection" onclick="showSubject(\'detection\',this)" data-det="fall">
          <img src="/icons/Fall.svg" alt=""> Fall Detection
          <span class="det-dot" aria-hidden="true"></span>
        </button>
        <button class="set-nav-item" data-subject="fire" onclick="showSubject(\'fire\',this)" data-det="fire">
          <img src="/icons/Fire.svg" alt=""> Fire Detection
          <span class="det-dot" aria-hidden="true"></span>
        </button>
        <button class="set-nav-item" data-subject="face" onclick="showSubject(\'face\',this)" data-det="face">
          <img src="/icons/Face.svg" alt=""> Face Detection
          <span class="det-dot" aria-hidden="true"></span>
        </button>
        <div class="set-nav-section">System</div>
        <button class="set-nav-item" data-subject="notify" onclick="showSubject(\'notify\',this)">
          <img src="/icons/Notifcation.svg" alt=""> Notifications
        </button>
        <button class="set-nav-item" data-subject="cameras" onclick="showSubject(\'cameras\',this)">
          <img src="/icons/Camera.svg" alt=""> Cameras
        </button>
        <button class="set-nav-item" data-subject="users" onclick="showSubject(\'users\',this)">
          <img src="/icons/Register.svg" alt=""> User Management
        </button>
        <button class="set-nav-item" data-subject="about" onclick="showSubject(\'about\',this)">
          <img src="/icons/LOGO.svg" alt=""> About
        </button>
      </aside>

      <main class="set-content">
      <!-- pane: Fall Detection -->
      <section class="set-pane active" id="pane-detection">
        <div class="set-pane-head"><h2>Fall Detection</h2><p>Sensitivity, pose YOLO model, and shared engine knobs (image size, FPS, FP16).</p></div>
      <div class="settings-card">
        <div class="s-card-head">
          <div class="s-icon icon-violet"><img src="/icons/Fall.svg" alt=""></div>
          <div class="s-head-text"><h3>Fall sensitivity</h3><p>Fall confidence threshold, inactivity timeout, and alert cooldown.</p></div>
        </div>
        <div class="form-group">
          <label>Enable fall detection</label>
          <div class="radio-group">
            <label class="radio-opt"><input type="radio" name="fallEnabled" value="true" checked><span class="radio-mark"></span><span class="radio-text">Enabled <small>Emit fall, lying-motionless and inactivity alerts</small></span></label>
            <label class="radio-opt"><input type="radio" name="fallEnabled" value="false"><span class="radio-mark"></span><span class="radio-text">Disabled <small>Suppress fall alerts (pose model still runs for face/zones)</small></span></label>
          </div>
        </div>
        <div class="slider-group">
          <div class="slider-head"><label>Fall confidence threshold</label><span class="slider-val" id="setThresholdVal">80</span></div>
          <input type="range" class="slider" id="setThreshold" min="50" max="99" value="80" oninput="updateSlider(this,\'setThresholdVal\')">
        </div>
        <div class="form-group"><label>Inactivity timeout (seconds)</label><input type="number" id="setInactivity" min="60" max="3600" value="300"></div>
        <div class="form-group"><label>Alert cooldown (seconds)</label><input type="number" id="setCooldown" min="10" max="600" value="60"></div>
      </div>
      <div class="settings-card">
        <div class="s-card-head">
          <div class="s-icon icon-blue"><img src="/icons/Detection.svg" alt=""></div>
          <div class="s-head-text"><h3>Pose Model &amp; Performance</h3><p>Choose the YOLO pose weights, image size, FPS cap and FP16.</p></div>
        </div>
        <div class="form-group">
          <label>YOLO pose model</label>
          <select id="setModel"><option value="">— scanning weights folder —</option></select>
          <p style="font-size:10px;color:var(--text-3);margin-top:5px;font-family:\'Chakra Petch\',sans-serif">Lists every <code>*.pt</code> file under <code>Fall_Detection/weights/</code>.</p>
        </div>
        <div class="slider-group">
          <div class="slider-head"><label>Detection confidence</label><span class="slider-val" id="setYoloConfVal">50</span></div>
          <input type="range" class="slider" id="setYoloConf" min="10" max="95" value="50" oninput="updateSlider(this,\'setYoloConfVal\')">
        </div>
        <div class="form-group">
          <label>Input image size <small style="color:var(--text-3)">(multiple of 32)</small></label>
          <select id="setImgsz">
            <option value="320">320 — fastest</option><option value="480">480</option>
            <option value="640" selected>640 — default</option>
            <option value="960">960</option><option value="1280">1280 — most accurate</option>
          </select>
        </div>
        <div class="form-group"><label>Processing FPS</label><input type="number" id="setFps" min="1" max="60" value="15"></div>
        <div class="form-group">
          <label>FP16 half-precision (GPU only)</label>
          <div class="radio-group">
            <label class="radio-opt"><input type="radio" name="fp16" value="true" checked><span class="radio-mark"></span><span class="radio-text">Enabled <small>~2\xd7 faster on supported GPUs</small></span></label>
            <label class="radio-opt"><input type="radio" name="fp16" value="false"><span class="radio-mark"></span><span class="radio-text">Disabled <small>Safer fallback for older hardware</small></span></label>
          </div>
        </div>
        <p style="font-size:10px;color:var(--text-2);margin-bottom:10px;font-family:\'Chakra Petch\',sans-serif;line-height:1.6">⚠ Model &amp; image size changes take effect after restarting the system.</p>
      </div>
      </section>

      <!-- pane: Fire -->
      <section class="set-pane" id="pane-fire">
        <div class="set-pane-head"><h2>Fire detection</h2><p>Run a YOLO model dedicated to fire and smoke classes.</p></div>
      <div class="settings-card">
        <div class="s-card-head">
          <div class="s-icon icon-red"><img src="/icons/Fire.svg" alt=""></div>
          <div class="s-head-text"><h3>Fire detection</h3><p>Run fire/smoke YOLO model and tune its confidence + classes.</p></div>
        </div>
        <div class="form-group">
          <label>Enable fire detection</label>
          <div class="radio-group">
            <label class="radio-opt"><input type="radio" name="fireEnabled" value="true" checked><span class="radio-mark"></span><span class="radio-text">Enabled <small>Run fire/smoke YOLO model on every frame</small></span></label>
            <label class="radio-opt"><input type="radio" name="fireEnabled" value="false"><span class="radio-mark"></span><span class="radio-text">Disabled <small>Skip fire inference (saves ~30–40% GPU)</small></span></label>
          </div>
        </div>
        <div class="form-group">
          <label>Fire YOLO model</label>
          <select id="setFireModel"><option value="">— scanning Fire_Detection folder —</option></select>
          <p style="font-size:10px;color:var(--text-3);margin-top:5px;font-family:\'Chakra Petch\',sans-serif">Lists every <code>*.pt</code> under <code>Fire_Detection/</code>.</p>
        </div>
        <div class="slider-group">
          <div class="slider-head"><label>Fire confidence threshold</label><span class="slider-val" id="setFireConfVal">35</span></div>
          <input type="range" class="slider" id="setFireConf" min="10" max="95" value="35" oninput="updateSlider(this,\'setFireConfVal\')">
        </div>
        <div class="form-group">
          <label>Alert classes (comma-separated)</label>
          <input id="setFireClasses" placeholder="fire,smoke">
          <p style="font-size:10px;color:var(--text-3);margin-top:5px;font-family:\'Chakra Petch\',sans-serif">Class names from the YOLO model that should trigger alerts.</p>
        </div>
        <div class="form-group"><label>Alert cooldown per class (seconds)</label><input type="number" id="setFireCooldown" min="1" max="120" value="5"></div>
      </div>
      </section>

      <!-- pane: Face -->
      <section class="set-pane" id="pane-face">
        <div class="set-pane-head"><h2>Face recognition</h2><p>InsightFace ArcFace matching and intruder alerts.</p></div>
      <div class="settings-card">
        <div class="s-card-head">
          <div class="s-icon icon-orange"><img src="/icons/Face.svg" alt=""></div>
          <div class="s-head-text"><h3>Face recognition</h3><p>InsightFace ArcFace matching and intruder detection.</p></div>
        </div>
        <div class="form-group">
          <label>Enable face recognition</label>
          <div class="radio-group">
            <label class="radio-opt"><input type="radio" name="faceEnabled" value="true" checked><span class="radio-mark"></span><span class="radio-text">Enabled <small>InsightFace ArcFace + intruder detection</small></span></label>
            <label class="radio-opt"><input type="radio" name="faceEnabled" value="false"><span class="radio-mark"></span><span class="radio-text">Disabled <small>Skip face inference (saves CPU/GPU)</small></span></label>
          </div>
        </div>
        <div class="slider-group">
          <div class="slider-head"><label>Match threshold</label><span class="slider-val" id="setFaceMatchVal">42</span></div>
          <input type="range" class="slider" id="setFaceMatch" min="20" max="80" value="42" oninput="updateSlider(this,\'setFaceMatchVal\')">
          <p style="font-size:10px;color:var(--text-3);margin-top:5px;font-family:\'Chakra Petch\',sans-serif">Cosine similarity. Lower = more permissive. Higher = stricter.</p>
        </div>
        <div class="form-group">
          <label>Run face every N frames</label>
          <input type="number" id="setFaceEveryN" min="1" max="60" value="5">
          <p style="font-size:10px;color:var(--text-3);margin-top:5px;font-family:\'Chakra Petch\',sans-serif">Higher = faster pipeline but slower face refresh.</p>
        </div>
        <div class="form-group"><label>Intruder cooldown (seconds)</label><input type="number" id="setIntruderCooldown" min="5" max="600" value="30"></div>
        <p id="faceStatusHint" style="font-size:10px;color:var(--text-2);margin-bottom:10px;font-family:\'Chakra Petch\',sans-serif;line-height:1.6">Status will appear here after saving.</p>
      </div>
      </section>

      <!-- pane: Notifications -->
      <section class="set-pane" id="pane-notify">
        <div class="set-pane-head"><h2>Notifications</h2><p>Send critical alerts to one or more WhatsApp numbers via Twilio.</p></div>
      <div class="settings-card">
        <div class="s-card-head">
          <div class="s-icon icon-green"><img src="/icons/Notifcation.svg" alt=""></div>
          <div class="s-head-text"><h3>WhatsApp notifications</h3><p>Send critical alerts via Twilio.</p></div>
        </div>
        <div class="form-group">
          <label>Phone numbers (comma-separated, with country code)</label>
          <textarea id="setPhones" rows="3" placeholder="+60123456789, +60198765432"></textarea>
        </div>
        <p style="font-size:10px;color:var(--text-2);margin-bottom:10px;font-family:\'Chakra Petch\',sans-serif;line-height:1.6">Configure Twilio credentials in the .env file. Numbers must be registered in your Twilio WhatsApp sandbox.</p>
      </div>
      </section>

      <!-- pane: Cameras -->
      <section class="set-pane" id="pane-cameras">
        <div class="set-pane-head"><h2>Cameras</h2><p>Add, list, and remove RTSP / USB / file-based camera sources.</p></div>
      <div class="settings-card">
        <div class="s-card-head">
          <div class="s-icon icon-blue"><img src="/icons/Camera.svg" alt=""></div>
          <div class="s-head-text"><h3>Camera management</h3><p>Add, list, and remove camera sources.</p></div>
        </div>
        <div id="camList" class="cam-list"></div>
        <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border)">
          <h4 style="font-size:9px;margin-bottom:9px;font-weight:600;color:var(--text-2);text-transform:uppercase;letter-spacing:1px;font-family:\'Chakra Petch\',sans-serif">Add camera</h4>
          <div class="form-group"><label>Name</label><input id="newCamName" placeholder="Living Room"></div>
          <div class="form-group"><label>RTSP URL (or 0 for webcam)</label><input id="newCamUrl" placeholder="rtsp://admin:pass@192.168.1.100:554/stream1"></div>
          <div class="form-group"><label>Location</label><input id="newCamLoc" placeholder="Living Room"></div>
          <button class="btn btn-primary" onclick="addCamera()">Add camera</button>
        </div>
      </div>
      </section>

      <!-- pane: User Management -->
      <section class="set-pane" id="pane-users">
        <div class="set-pane-head"><h2>User Management</h2><p>Create dashboard accounts and assign roles. <strong>Admins</strong> have full access; <strong>guests</strong> see only live feeds and events.</p></div>
        <div class="settings-card">
          <div class="s-card-head">
            <div class="s-icon icon-amber"><img src="/icons/Register.svg" alt=""></div>
            <div class="s-head-text"><h3>Add a new user</h3><p>Username + initial password + role.</p></div>
          </div>
          <div class="cam-form" style="margin-top:7px">
            <div class="form-group"><label class="form-label">Username</label><input class="form-input" id="newUserName" placeholder="e.g. salehuddin"></div>
            <div class="form-group"><label class="form-label">Initial password</label><input class="form-input" id="newUserPass" type="password" placeholder="min. 4 chars"></div>
            <div class="form-group">
              <label class="form-label">Role</label>
              <div class="radio-group" id="newUserRoleGroup">
                <label class="radio-opt"><input type="radio" name="newUserRole" value="guest" checked><span class="radio-mark"></span><span class="radio-text">Guest<small>View live feeds and event log only</small></span></label>
                <label class="radio-opt"><input type="radio" name="newUserRole" value="admin"><span class="radio-mark"></span><span class="radio-text">Admin<small>Full access</small></span></label>
              </div>
            </div>
            <div class="form-group" style="display:block">
              <label class="mustchange-label"><input type="checkbox" id="newUserMustChange" checked><span>Force password change on first sign-in</span></label>
            </div>
            <div class="form-group"><button class="btn btn-primary" onclick="addUser()">Add user</button></div>
          </div>
        </div>
        <div class="settings-card">
          <div class="s-card-head">
            <div class="s-icon icon-info"><svg viewBox="0 0 24 24" fill="none"><path d="M4 6h16M4 12h16M4 18h10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path></svg></div>
            <div class="s-head-text"><h3>Existing accounts</h3><p>Change a user\'s role, reset their password, or remove their account.</p></div>
          </div>
          <div id="usersTableWrap" style="overflow-x:auto"><p style="color:var(--text-2);font-size:12px;padding:12px">Loading users…</p></div>
        </div>
      </section>

      <section class="set-pane" id="pane-about">
        <div class="set-pane-head"><h2>About</h2><p>Project information and credits.</p></div>
      <div class="settings-card">
        <div class="s-card-head">
          <div class="s-icon icon-amber"><img src="/icons/LOGO.svg" alt=""></div>
          <div class="s-head-text"><h3>About HomeShield</h3><p>Project, detection stack and architecture credits.</p></div>
        </div>
        <p style="font-size:12px;color:var(--text-2);line-height:1.9">
          <strong style="color:var(--text)">HomeShield</strong> is a centralized dashboard for real-time CCTV monitoring and machine-learning-powered anomaly detection. One unified pipeline runs three GPU-accelerated detectors on every camera feed, draws bounding-box overlays on the live MJPEG stream, persists every alert to SQLite with an annotated snapshot, and pushes updates to this dashboard via Server-Sent Events.<br><br>

          <span style="color:var(--text)">Detection stack</span><br>
          &nbsp;&nbsp;&#x1F9D8; <strong>Fall</strong> — YOLO11 / YOLO26 pose estimation feeds a 7-state finite-state machine (Standing → Walking → Sitting → Fall_Detected → Lying_After_Fall → Lying_Motionless → Inactivity). Two-stage decision rejects controlled sit-downs.<br>
          &nbsp;&nbsp;&#x1F525; <strong>Fire</strong> — custom YOLO weights covering fire and smoke classes, with per-class cooldown to debounce alerts.<br>
          &nbsp;&nbsp;&#x1F464; <strong>Face</strong> — InsightFace ArcFace (buffalo_l) producing 512-dim embeddings, cosine-matched against the registered Persons gallery; unknown faces are auto-logged as intruders.<br>
          &nbsp;&nbsp;&#x1F6A7; <strong>Zones</strong> — polygon zones per camera. Safe zones suppress lying/inactivity alerts; danger zones trigger child-entry alerts.<br><br>

          <span style="color:var(--text)">Version:</span> 2.0.0 (Combined Fall + Fire + Face)<br>
          <span style="color:var(--text)">Developer:</span> Mohammad Salehuddin bin Iwan<br>
          <span style="color:var(--text)">Supervisor:</span> Andi Fitriah binti Abdul Kadir<br>
          <span style="color:var(--text)">Institution:</span> IIUM, Kulliyyah of Information and Communication Technology<br>
          <span style="color:var(--text)">Project:</span> Final Year Project — HomeShield: Centralized Dashboard for Real-Time CCTV Monitoring and Anomaly Detection
        </p>
      </div>
      </section>
      </main>
    </div>

    <div class="save-bar" id="saveBar">
      <div class="save-status">
        <span class="save-dot"></span>
        <span id="saveBarMsg">You have unsaved changes</span>
      </div>
      <div class="save-actions">
        <button class="btn" onclick="discardChanges()">Discard</button>
        <button class="btn btn-primary" onclick="saveAllSettings()">Save changes</button>
      </div>
    </div>
  </div>

</div>
</main>

<!-- Snapshot modal -->
<div class="modal-overlay" id="snapshotModal" onclick="this.classList.remove(\'show\')">
  <div class="modal" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="document.getElementById(\'snapshotModal\').classList.remove(\'show\')" aria-label="Close">\xd7</button>
    <h3 id="modalTitle">Event snapshot</h3>
    <img id="modalImg" src="" alt="snapshot">
    <p id="modalInfo"></p>
  </div>
</div>

<div id="hs-toast"></div>
'''

# ── Compose full file ─────────────────────────────────────────────────────────
new_html = (
    HEAD
    + BODY
    + script_main_patched + '\n'
    + script_shader + '\n'
    + script_settings + '\n'
    + '</body>\n</html>\n'
)

with open('decoded_template.html', 'w', encoding='utf-8') as f:
    f.write(new_html)

print(f'Written: {len(new_html):,} chars')
print('Done.')
