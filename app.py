import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import time
import ruptures as rpt

# --- OPTIONAL: SHAP (install with: pip install shap) ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ─────────────────────────────────────────────
# 1. PAGE CONFIG & CYBER-INDUSTRIAL STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PreSense | SME Autonomous Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');


/* === BASE === */
.stApp {
    background: #060a0f;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.03) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0,255,136,0.02) 0%, transparent 50%);
}


/* === METRICS === */
[data-testid="stMetricValue"] {
    font-size: 2.0rem !important;
    font-family: 'Orbitron', monospace !important;
    color: #00D4FF !important;
    text-shadow: 0 0 20px rgba(0,212,255,0.8), 0 0 40px rgba(0,212,255,0.3);
    letter-spacing: 0.05em;
}
[data-testid="stMetricLabel"] {
    color: #8899aa !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.2em;
}
[data-testid="stMetricDelta"] {
    color: #00FF88 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}


/* === METRIC CONTAINERS — add card borders === */
[data-testid="stMetric"] {
    background: rgba(0,212,255,0.03);
    border: 1px solid rgba(0,212,255,0.12);
    border-top: 2px solid rgba(0,212,255,0.4);
    border-radius: 4px;
    padding: 16px 20px !important;
}


/* === AGENT LOG === */
.agent-log {
    background: #000507;
    color: #00C896;
    font-family: 'Share Tech Mono', monospace;
    padding: 16px;
    border-radius: 4px;
    border: 1px solid #00C89622;
    border-left: 3px solid #00C896;
    font-size: 0.8rem;
    line-height: 1.7;
    height: 240px;
    overflow-y: auto;
    box-shadow: inset 0 0 40px rgba(0,200,150,0.02);
}


/* === WORK ORDER CARD === */
.wo-card {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.2);
    border-left: 3px solid #00D4FF;
    border-radius: 4px;
    padding: 20px 24px;
    margin: 12px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #64748b;
}
.wo-card-critical {
    background: linear-gradient(135deg, rgba(255,50,50,0.06) 0%, rgba(255,100,0,0.03) 100%);
    border: 1px solid rgba(255,60,60,0.35);
    border-left: 3px solid #FF3C3C;
    box-shadow: 0 0 30px rgba(255,60,60,0.08), inset 0 0 30px rgba(255,60,60,0.02);
}


/* === KPI BANNER === */
.kpi-banner {
    background: rgba(0,212,255,0.03);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 4px;
    padding: 10px 24px;
    margin-bottom: 24px;
    display: flex;
    gap: 40px;
    font-size: 0.75rem;
    color: #8899aa;
    font-family: 'Rajdhani', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}


/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background: #040810 !important;
    border-right: 1px solid #1a2a3a !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p {
    color: #8899aa !important;
    font-family: 'Rajdhani', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem !important;
}


/* === BUTTONS === */
.stButton > button {
    border-radius: 2px;
    border: 1px solid #1a3a4a;
    background: rgba(0,212,255,0.04);
    color: #00A8CC;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    transition: all 0.15s;
    width: 100%;
    padding: 10px;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.1);
    border-color: #00D4FF55;
    color: #00D4FF;
    box-shadow: 0 0 20px rgba(0,212,255,0.15);
}


/* === DIVIDERS === */
hr { border-color: #1a2a3a !important; margin: 8px 0 !important; }


/* === STATUS BADGES === */
.badge-critical {
    color: #FF3C3C; font-weight: 900;
    font-family: 'Orbitron', monospace;
    animation: blink 1s step-end infinite;
    font-size: 1rem;
}
.badge-warning  {
    color: #FFB800; font-weight: 700;
    font-family: 'Orbitron', monospace;
}
.badge-healthy  {
    color: #00FF88; font-weight: 700;
    font-family: 'Orbitron', monospace;
}


@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }


/* === SECTION LABELS === */
.section-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.25em;
    color: #6b7f8f;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2a3a;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 12px;
    height: 1px;
    background: #00D4FF44;
}


h3 {
    font-family: 'Rajdhani', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    color: #6b7f8f !important;
    font-size: 0.68rem !important;
    margin-bottom: 8px !important;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2a3a;
}
</style>
""", unsafe_allow_html=True)




# ─────────────────────────────────────────────
# 2. DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    columns = (
        ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3']
        + [f'sensor_{i}' for i in range(1, 22)]
    )
    df_raw = pd.read_csv(
        'archive/train_FD001.txt',
        sep=r'\s+', header=None, names=columns
    )
    df_raw['unit_number'] = df_raw['unit_number'].astype(int)
    return df_raw




@st.cache_resource
def load_artifacts():
    model = joblib.load('rul_model.pkl')
    scaler = joblib.load('rul_scaler.pkl') if os.path.exists('rul_scaler.pkl') else None
    return model, scaler




@st.cache_resource
def load_explainer(_model, _background_data):
    if not SHAP_AVAILABLE:
        return None
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception:
        return None




df       = load_data()
model, scaler = load_artifacts()
# Load full feature list (includes _roll3 columns)
if os.path.exists('feature_names.pkl'):
    FEATURES = joblib.load('feature_names.pkl')
else:
    FEATURES = [f'sensor_{i}' for i in range(1, 22)]


# Base sensors only (no roll3) — what the raw dataframe has
BASE_SENSORS = [f for f in FEATURES if '_roll3' not in f]


# Add rolling features to df for prediction use
for col in BASE_SENSORS:
    df[f'{col}_roll3'] = (
        df.groupby('unit_number')[col]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )


background_sample = df.sample(min(200, len(df)), random_state=42)[FEATURES]
explainer = load_explainer(model, background_sample)


# ─────────────────────────────────────────────
# 3. SIDEBAR — CONTROL CENTER
# ─────────────────────────────────────────────
st.sidebar.markdown("## ⬡ PRESENSE\n### Control Center")
st.sidebar.divider()


machine_id  = st.sidebar.selectbox("Machine Unit", df['unit_number'].unique())
full_data   = df[df['unit_number'] == machine_id]
max_cycles  = int(full_data['time_in_cycles'].max())


st.sidebar.divider()


if 'current_cycle' not in st.session_state:
    st.session_state.current_cycle = 1


selected_cycle = st.sidebar.slider(
    "Timeline Control",
    min_value=1,
    max_value=max_cycles,
    value=st.session_state.current_cycle
)
st.session_state.current_cycle = selected_cycle


st.sidebar.markdown(f"""
<div style='font-family:Share Tech Mono,monospace; font-size:0.75rem; color:#8899aa;
     padding: 8px; background: rgba(0,212,255,0.03); border-radius:3px; margin:8px 0;'>
  UNIT: #{machine_id:03d}<br>
  MAX CYCLES: {max_cycles}<br>
  PROGRESS: {int(selected_cycle/max_cycles*100)}%
</div>
""", unsafe_allow_html=True)


if st.sidebar.button("🔄 Reset Timeline"):
    st.session_state.current_cycle = 1
    st.rerun()


st.sidebar.divider()
st.sidebar.markdown("""
<div style='font-family:Rajdhani,sans-serif; font-size:0.7rem; color:#8899aa;
     text-transform:uppercase; letter-spacing:0.1em;'>
  FLEET STATUS<br>
</div>
<div style='font-family:Share Tech Mono,monospace; font-size:0.72rem; color:#2d3748; margin-top:6px;'>
  UNIT #001 — <span style='color:#00FF88'>HEALTHY</span><br>
  UNIT #002 — <span style='color:#FFB800'>WARNING</span><br>
  UNIT #003 — <span style='color:#00D4FF'>ACTIVE</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("""
<div style='background:rgba(0,255,136,0.05); border:1px solid #00FF8822;
     border-radius:4px; padding:12px; margin-top:8px;'>
  <div style='font-family:Share Tech Mono,monospace; color:#00FF88;
       font-size:0.75rem; margin-bottom:8px; letter-spacing:0.05em;'>
    🌍 UN SDG ALIGNMENT
  </div>
  <div style='color:#00FF88; font-size:0.72rem; font-family:Rajdhani,sans-serif;
       font-weight:700; letter-spacing:0.08em; margin-bottom:4px;'>
    GOAL 9: INDUSTRY, INNOVATION &amp; INFRASTRUCTURE
  </div>
  <div style='color:#8899aa; font-size:0.68rem; font-family:Rajdhani,sans-serif;
       line-height:1.5; letter-spacing:0.04em;'>
    Target 9.4: Upgrade infrastructure and retrofit industries
    to make them sustainable, with greater adoption of clean
    and environmentally sound technologies.
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. DATA SYNC & PREDICTION
# ─────────────────────────────────────────────
@st.cache_data
def detect_degradation(sensor_data):
    """Uses PELT algorithm to find mathematical regime shift in sensor data."""
    signal = np.array(sensor_data).reshape(-1, 1)
    if len(signal) < 15: return None
    algo = rpt.Pelt(model="rbf").fit(signal)
    change_points = algo.predict(pen=10)
    valid_points = [cp for cp in change_points if cp < len(signal)]
    return valid_points[0] if valid_points else None

display_data = full_data[full_data['time_in_cycles'] <= selected_cycle]
current_row  = display_data[display_data['time_in_cycles'] == selected_cycle]


pred_rul  = 0
rul_std   = 0
shap_vals = None
top_factors = pd.Series(dtype=float)
X = None


try:
    if not current_row.empty:
        # ── Build rolling features on display_data first ──
        for col in BASE_SENSORS:
            display_data[f'{col}_roll3'] = (
                display_data[col]
                .rolling(3, min_periods=1)
                .mean()
                .values
            )
        # Re-slice current_row AFTER rolling features are added
        current_row = display_data[display_data['time_in_cycles'] == selected_cycle]


        X = current_row[FEATURES].copy()
        X = X.fillna(0)  # handles NaN at early cycles
        if scaler:
            X = pd.DataFrame(scaler.transform(X), columns=FEATURES)


        # Confidence interval via individual trees
        tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
        pred_rul   = float(np.mean(tree_preds))
        rul_std    = float(np.std(tree_preds))
        pred_rul   = max(0.0, min(pred_rul, 200.0))


        # SHAP feature importance
        if explainer is not None:
            shap_vals   = explainer.shap_values(X)
            feature_imp = pd.Series(np.abs(shap_vals[0]), index=FEATURES)
            top_factors = feature_imp.sort_values(ascending=False).head(5)


except Exception as e:
    st.sidebar.error(f"Prediction error: {e}")
    pred_rul = 0




# ─────────────────────────────────────────────
# 5. MCP AGENT — TOOL EXECUTION
# ─────────────────────────────────────────────
# Import tools directly from mcp_server (same logic, no async needed for demo)
import importlib.util, sys

def load_mcp_tools():
    """Load MCP tools directly from mcp_server.py"""
    try:
        spec = importlib.util.spec_from_file_location("mcp_server", "mcp_server.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None

mcp_mod = load_mcp_tools()

def run_mcp_agent(machine_id, selected_cycle, pred_rul, top_factors, rul_std):
    log   = []
    wo_id = f"WO-{machine_id:03d}-{selected_cycle:04d}"

    # ── Multi-factor Decision Intelligence ──
    failure_risk = max(0, min(100, (1 - pred_rul / 125) * 100))
    confidence   = max(0, 100 - rul_std * 2)
    risk_score   = failure_risk * (0.5 + confidence / 200)
    risk_score   = max(0, min(100, risk_score))

    if risk_score > 60:
        decision = "AUTO_DISPATCH"
    elif risk_score > 30:
        decision = "SCHEDULE"
    else:
        decision = "MONITOR"

    log.append(f"> [AGENT] PreSense — Unit #{machine_id:03d} heartbeat... OK")
    log.append(f"> [ML]    RUL: {int(pred_rul)} ± {int(rul_std)} cycles")
    log.append(f"> [RISK]  Failure Risk: {failure_risk:.0f}%  |  Confidence: {confidence:.0f}%")
    log.append(f"> [RISK]  Risk Score: {risk_score:.1f}/100  →  Decision: {decision}")

    if pred_rul < 80:
        log.append(f"> [WARN]  Degradation pattern detected at Cycle {selected_cycle}")

    if decision == "AUTO_DISPATCH":
        log.append(f"> [ALERT] RISK SCORE {risk_score:.0f}/100 — AUTONOMOUS DISPATCH INITIATED")
    elif decision == "SCHEDULE":
        log.append(f"> [WARN]  RISK SCORE {risk_score:.0f}/100 — SCHEDULING PREVENTIVE MAINTENANCE")

    # MCP tools — separate block, fires for BOTH decisions
    if decision in ["AUTO_DISPATCH", "SCHEDULE"]:

        # ── TOOL 1: check_inventory() via MCP server ──
        log.append(f"> [MCP:CALL] check_inventory(part_name='HPT-BLD')")
        try:
            if mcp_mod:
                result = mcp_mod.check_inventory("HPT-BLD")
            else:
                inv = pd.read_csv('inventory.csv')
                available = inv[inv['stock'] > 0]
                part = available.iloc[0]
                result = (f"PARTS LOCATED: {part['stock']} units of "
                         f"{part['part_name']} @ {part['location']}. "
                         f"Cost: RM {part['unit_cost_myr']}")
            log.append(f"> [MCP:RESPONSE] {result}")
        except Exception as ex:
            log.append(f"> [MCP:ERROR] {ex}")

        # ── TOOL 2: schedule_technician() via MCP server ──
        log.append(f"> [MCP:CALL] schedule_technician(certification='Turbine')")
        try:
            if mcp_mod:
                result = mcp_mod.schedule_technician("Turbine")
            else:
                sch = pd.read_csv('schedule.csv')
                available = sch[sch['status'] == 'Available']
                tech = available.iloc[0]
                result = f"ASSIGNED: {tech['name']} ({tech['contact']}). ETA: 2 hours."
            log.append(f"> [MCP:RESPONSE] {result}")
        except Exception as ex:
            log.append(f"> [MCP:ERROR] {ex}")

        # ── TOOL 3: create_work_order() via MCP server ──
        log.append(f"> [MCP:CALL] create_work_order(unit={machine_id}, cycle={selected_cycle})")
        try:
            if mcp_mod:
                result = mcp_mod.create_work_order(
                    int(machine_id), int(selected_cycle), float(pred_rul), decision
                )
            else:
                result = f"WORK ORDER {wo_id} CREATED. STATUS: DISPATCHED."
            log.append(f"> [MCP:RESPONSE] {result}")
        except Exception as ex:
            log.append(f"> [MCP:ERROR] {ex}")

    return log, decision, risk_score

agent_log, decision, risk_score_global = run_mcp_agent(
    machine_id, selected_cycle, pred_rul, top_factors, rul_std
)
show_roi = decision in ["AUTO_DISPATCH", "SCHEDULE"] and not current_row.empty

# ─────────────────────────────────────────────
# 6. DASHBOARD HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:baseline; gap:16px; margin-bottom:4px;'>
  <span style='font-family:Share Tech Mono,monospace; font-size:1.8rem; 
               color:#00D4FF; text-shadow:0 0 20px rgba(0,212,255,0.5);'>
    ⬡ PRESENSE
  </span>
  <span style='font-family:Rajdhani,sans-serif; font-size:0.9rem; color:#6b7f8f;
               text-transform:uppercase; letter-spacing:0.2em;'>
    SME Autonomous Maintenance Agent
  </span>
</div>
<div style='font-family:Rajdhani,sans-serif; font-size:0.8rem; color:#8899aa;
     text-transform:uppercase; letter-spacing:0.15em; margin-bottom:16px;'>
  ASEAN Predictive Maintenance Platform &nbsp;·&nbsp;  
  <span style='color:#00FF88'>● SYSTEM NOMINAL</span>
</div>
""", unsafe_allow_html=True)

# KPI Banner
st.markdown("""
<div class='kpi-banner'>
  <span>🏭 <b style='color:#00D4FF88'>Fleet:</b> 3 Units Active</span>
  <span>⏱ <b style='color:#00D4FF88'>Uptime:</b> 99.2%</span>
  <span>💰 <b style='color:#00D4FF88'>Savings MTD:</b> RM 24,800</span>
  <span>🔧 <b style='color:#00D4FF88'>Open WOs:</b> 2</span>
  <span>📡 <b style='color:#00D4FF88'>Data Points:</b> 21 Sensors</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 7. METRICS ROW
# ─────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

m1.metric("Current Cycle", f"{selected_cycle}", delta=f"of {max_cycles} max")

if not current_row.empty:
    m2.metric(
        "AI Predicted RUL",
        f"{int(pred_rul)} cyc",
        delta=f"±{int(rul_std)} uncertainty"
    )
    # Health %
    health_pct = max(0, min(100, int((pred_rul / max_cycles) * 100)))
    m3.metric("Machine Health", f"{health_pct}%")
    
    with m4:
        if decision == "AUTO_DISPATCH":
            st.markdown("""
            <div style='background:rgba(255,60,60,0.08); border:1px solid #FF3C3C66;
                border-radius:4px; padding:12px; text-align:center;'>
            <div class='badge-critical'>🚨 CRITICAL</div>
            <div style='color:#FF3C3C88; font-size:0.72rem; font-family:Rajdhani,sans-serif;
                text-transform:uppercase; letter-spacing:0.1em; margin-top:4px;'>
                Auto-Dispatch Active
            </div>
            </div>""", unsafe_allow_html=True)
        elif decision == "SCHEDULE":
            st.markdown("""
            <div style='background:rgba(255,184,0,0.06); border:1px solid #FFB80055;
                border-radius:4px; padding:12px; text-align:center;'>
            <div class='badge-warning'>⚠ WARNING</div>
            <div style='color:#FFB80088; font-size:0.72rem; font-family:Rajdhani,sans-serif;
                text-transform:uppercase; letter-spacing:0.1em; margin-top:4px;'>
                Schedule Maintenance
            </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(0,255,136,0.05); border:1px solid #00FF8833;
                border-radius:4px; padding:12px; text-align:center;'>
            <div class='badge-healthy'>✅ HEALTHY</div>
            <div style='color:#00FF8888; font-size:0.72rem; font-family:Rajdhani,sans-serif;
                text-transform:uppercase; letter-spacing:0.1em; margin-top:4px;'>
                System Optimal
            </div>
            </div>""", unsafe_allow_html=True)
else:
    m2.metric("AI Predicted RUL", "N/A")
    m3.metric("Machine Health", "N/A")
    m4.info("Synchronising...")

# ─────────────────────────────────────────────
# 8. HEALTH SCORE CHART
# ─────────────────────────────────────────────
st.markdown("<div class='section-label'>📈 &nbsp; Machine Health Index — Multi-Sensor Fusion</div>", unsafe_allow_html=True)

HEALTH_SENSORS = [f'sensor_{i}' for i in [7, 11, 12, 15, 20]]
full_sensor_data = full_data[HEALTH_SENSORS]
disp_sensor_data = display_data[HEALTH_SENSORS]

sensor_range = full_sensor_data.max() - full_sensor_data.min() + 1e-9
health_score = (
    1 - (disp_sensor_data - full_sensor_data.min()) / sensor_range
).mean(axis=1) * 100
health_score = health_score.rolling(5, min_periods=1).mean()

BG = '#060a0f'
fig, ax = plt.subplots(figsize=(14, 3.5), facecolor=BG)
ax.set_facecolor(BG)

cycles = display_data['time_in_cycles'].values

# Zone backgrounds
ax.axhspan(75, 100, alpha=0.04, color='#00FF88', zorder=0)
ax.axhspan(40, 75,  alpha=0.03, color='#FFB800', zorder=0)
ax.axhspan(0,  40,  alpha=0.05, color='#FF3C3C', zorder=0)

# Zone labels on right
ax.text(max_cycles * 0.995, 87, 'HEALTHY',  color='#00FF8833', fontsize=6.5, ha='right', fontfamily='monospace', va='center')
ax.text(max_cycles * 0.995, 57, 'WARNING',  color='#FFB80033', fontsize=6.5, ha='right', fontfamily='monospace', va='center')
ax.text(max_cycles * 0.995, 20, 'CRITICAL', color='#FF3C3C44', fontsize=6.5, ha='right', fontfamily='monospace', va='center')

# Fill under curve — colour by zone
ax.fill_between(cycles, health_score, 0, where=(health_score >= 75), color='#00FF88', alpha=0.07, zorder=1)
ax.fill_between(cycles, health_score, 0, where=((health_score >= 40) & (health_score < 75)), color='#00D4FF', alpha=0.07, zorder=1)
ax.fill_between(cycles, health_score, 0, where=(health_score < 40), color='#FF3C3C', alpha=0.1, zorder=1)

# Main line — colour segments
healthy_mask = health_score >= 75
warning_mask = (health_score >= 40) & (health_score < 75)
critical_mask = health_score < 40

for mask, colour in [(healthy_mask, '#00FF88'), (warning_mask, '#00D4FF'), (critical_mask, '#FF3C3C')]:
    if mask.any():
        masked_cycles = np.where(mask, cycles, np.nan)
        masked_health = np.where(mask, health_score, np.nan)
        ax.plot(masked_cycles, masked_health, color=colour, linewidth=2.0, zorder=5, solid_capstyle='round')

# Threshold line
ax.axhline(y=40, color='#FF3C3C', linestyle='--', linewidth=1.0, alpha=0.5, zorder=4)

# Draw Change-Point Detection Line
shift_cycle = detect_degradation(full_data['sensor_11'].values)
if shift_cycle and shift_cycle <= selected_cycle:
    ax.axvline(x=shift_cycle, color='#FFB800', linestyle=':', linewidth=2.0, zorder=4)
    ax.text(shift_cycle + 2, 90, 'AI ANOMALY DETECTED', color='#FFB800', fontsize=7, fontfamily='monospace', rotation=90)


# Current position dot
if not current_row.empty and len(health_score) > 0:
    ch = float(health_score.iloc[-1])
    dot_color = '#FF3C3C' if ch < 40 else '#FFB800' if ch < 75 else '#00FF88'
    ax.plot(selected_cycle, ch, 'o', color=dot_color, markersize=8,
            markerfacecolor=BG, markeredgewidth=2.5, zorder=7)
    ax.annotate(f'  {ch:.0f}%', xy=(selected_cycle, ch),
                color=dot_color, fontsize=8, fontfamily='monospace',
                va='center', fontweight='bold')

# Styling
ax.set_xlim(0, max_cycles)
ax.set_ylim(0, 105)
ax.set_xlabel("Operational Cycles", color='#6b7f8f', fontsize=8, labelpad=8)
ax.set_ylabel("Health %", color='#6b7f8f', fontsize=8)
ax.grid(color='#1a2a3a', linestyle='-', alpha=1.0, linewidth=0.5)
ax.tick_params(colors='#6b7f8f', labelsize=7.5)
for spine in ax.spines.values():
    spine.set_edgecolor('#1a2a3a')
fig.tight_layout(pad=0.5)

st.pyplot(fig)
plt.close(fig)

# ─────────────────────────────────────────────
# 9. AGENT LOG + SHAP (two columns)
# ─────────────────────────────────────────────
st.markdown("---")
log_col, shap_col = st.columns([3, 2])

with log_col:
    st.markdown("<div class='section-label'>🤖 &nbsp; Agent Neural Activity & MCP Context</div>", unsafe_allow_html=True)
    log_html = "<br>".join(
        f"<span style='color:#FF4444'>{line}</span>"
        if "ALERT" in line or "ERROR" in line
        else f"<span style='color:#FFB800'>{line}</span>"
        if "WARN" in line
        else f"<span style='color:#00D4FF'>{line}</span>"
        if "MCP:TOOL" in line or "MCP:ACTION" in line or "MCP:CALL" in line or "MCP:RESPONSE" in line
        else line
        for line in agent_log
    )
    st.markdown(f"<div class='agent-log'>{log_html}</div>", unsafe_allow_html=True)

with shap_col:
    st.markdown("<div class='section-label'>🧠 &nbsp; ML Explainability — SHAP Impact</div>", unsafe_allow_html=True)
    if not top_factors.empty and SHAP_AVAILABLE:
        fig2, ax2 = plt.subplots(figsize=(5, 2.8), facecolor='#070b0f')
        ax2.set_facecolor('#070b0f')

        sensors = top_factors.index.tolist()[::-1]
        values  = top_factors.values[::-1]
        colors  = ['#00D4FF' if v == values.max() else '#1a4a5a' for v in values]

        bars = ax2.barh(sensors, values, color=colors, height=0.5, edgecolor='none')
        ax2.set_xlabel("SHAP Impact (|value|)", color='#2d3748', fontsize=8)
        ax2.tick_params(colors='#4a5568', labelsize=8)
        ax2.set_facecolor('#070b0f')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#1a2332')
        fig2.patch.set_facecolor('#070b0f')
        st.pyplot(fig2)
        plt.close(fig2)
        st.markdown(
            "<div style='font-family:Share Tech Mono,monospace; font-size:0.72rem;"
            "color:#1a2332; text-align:center;'>Higher = stronger degradation driver</div>",
            unsafe_allow_html=True
        )
    elif not SHAP_AVAILABLE:
        st.markdown("""
        <div style='background:rgba(0,212,255,0.03); border:1px solid #1a2332;
             border-radius:4px; padding:20px; text-align:center; height:180px;
             display:flex; flex-direction:column; justify-content:center;'>
          <div style='font-family:Share Tech Mono,monospace; color:#2d3748; font-size:0.8rem;'>
            SHAP UNAVAILABLE<br><br>
            <span style='color:#1a2332; font-size:0.7rem;'>
              pip install shap<br>to enable explainability
            </span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(0,212,255,0.03); border:1px solid #1a2332;
             border-radius:4px; padding:20px; text-align:center;'>
          <div style='font-family:Share Tech Mono,monospace; color:#2d3748; font-size:0.8rem;'>
            AWAITING DATA<br>
            <span style='color:#1a2332; font-size:0.7rem;'>Select a valid cycle</span>
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 10. CRITICAL — ROI METRICS + WORK ORDER
# ─────────────────────────────────────────────
if decision in ["AUTO_DISPATCH", "SCHEDULE"] and not current_row.empty:
    st.markdown("---")
    st.markdown("<div class='section-label'>💰 &nbsp; Business Impact — Autonomous Intervention</div>",
                unsafe_allow_html=True)

    # Dynamic ROI calculation
    if pred_rul < 40:
        downtime_hrs = max(1, int((40 - pred_rul) / 8))
    else:
        downtime_hrs = 1  # Preventive maintenance — minimal downtime
    prevented_loss   = downtime_hrs * 4200
    maintenance_cost = 1650
    net_saving       = prevented_loss - maintenance_cost

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("⏱ Downtime Prevented", f"{downtime_hrs} hrs")
    r2.metric("💸 Loss Prevented",    f"RM {prevented_loss:,}")
    r3.metric("🔧 Maintenance Cost",  f"RM {maintenance_cost:,}")
    r4.metric("📈 Net Saving",        f"RM {net_saving:,}", delta="vs reactive model")

    wo_id        = f"WO-{machine_id:03d}-{selected_cycle:04d}"
    is_critical  = decision == "AUTO_DISPATCH"
    header_color = "#FF3C3C" if is_critical else "#FFB800"
    header_text  = "AUTO-DISPATCHED" if is_critical else "SCHEDULED"
    priority     = "CRITICAL" if is_critical else "PREVENTIVE"
    status_color = "#00FF88" if is_critical else "#FFB800"
    status_text  = "DISPATCHED ✓" if is_critical else "RESERVED ✓"
    card_class   = "wo-card wo-card-critical" if is_critical else "wo-card"

    st.markdown(f"""
    <div class='{card_class}'>
      <div style='color:{header_color}; font-size:0.88rem; margin-bottom:10px;
           font-family:Share Tech Mono,monospace;'>
        ████ {header_text} WORK ORDER: {wo_id} ████
      </div>
      <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;
           font-family:Share Tech Mono,monospace; font-size:0.8rem; color:#64748b;'>
        <span>MACHINE UNIT &nbsp;: #{machine_id:03d}</span>
        <span>PRIORITY &nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:{header_color}'>{priority}</span></span>
        <span>PREDICTED RUL : {int(pred_rul)} ± {int(rul_std)} cycles</span>
        <span>RISK SCORE &nbsp;&nbsp;: {risk_score_global:.1f}/100</span>
        <span>TECHNICIAN &nbsp;&nbsp;: Assigned via MCP</span>
        <span>PARTS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Sourced via MCP</span>
        <span>ETA &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 2 hours</span>
        <span>STATUS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:{status_color}'>{status_text}</span></span>
      </div>
      <div style='margin-top:12px; color:#4a5568; font-size:0.68rem;
           font-family:Rajdhani,sans-serif; text-transform:uppercase; letter-spacing:0.08em;'>
        Generated autonomously by PreSense MCP Agent ·
        Random Forest RMSE 19.00 · SHAP Explainability · PELT Change-Point Detection
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 11. MANUAL COMMAND BUTTONS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-label'>⚡ &nbsp; Manual Override Commands</div>", unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)

with b1:
    if st.button("🛠️ Dispatch Maintenance"):
        st.toast("Work order created — Ahmad Razif notified!", icon="👷")
with b2:
    if st.button("📦 Reserve Spare Parts"):
        st.toast("HPT-BLD-007 reserved at Warehouse A", icon="📦")
with b3:
    if st.button("📞 Alert Plant Manager"):
        st.toast("Emergency SMS dispatched to manager!", icon="📱")
        st.session_state['show_sms'] = True

with b4:
    if st.button("📊 Export Report"):
        report_lines = [
            f"PreSense Maintenance Report",
            f"Unit: #{machine_id:03d}",
            f"Cycle: {selected_cycle}",
            f"Predicted RUL: {int(pred_rul)} ± {int(rul_std)} cycles",
            f"Status: {'CRITICAL' if pred_rul < 40 else 'WARNING' if pred_rul < 80 else 'HEALTHY'}",
            "---",
        ] + agent_log
        report_text = "\n".join(report_lines)
        st.download_button(
            "⬇ Download Report",
            data=report_text,
            file_name=f"presense_report_unit{machine_id}_cycle{selected_cycle}.txt",
            mime="text/plain"
        )

# SMS panel — AFTER all 4 button columns
if st.session_state.get('show_sms') and show_roi:
    col_sms, col_dismiss = st.columns([9, 1])
    with col_sms:
        st.markdown(f"""
        <div style='background:rgba(0,212,255,0.04); border:1px solid #00D4FF33;
             border-radius:4px; padding:14px 20px; margin-top:8px;
             font-family:Share Tech Mono,monospace; font-size:0.78rem; color:#8899aa;'>
          <div style='color:#00D4FF; margin-bottom:6px; font-size:0.7rem;
               letter-spacing:0.15em;'>📱 SMS ALERT SENT</div>
          <b style='color:#a0aec0;'>TO:</b> Plant Manager (+60171234567)<br>
          <b style='color:#a0aec0;'>MSG:</b> [PRESENSE ALERT] Unit #{machine_id:03d} — 
          RUL {int(pred_rul)} cycles remaining. Risk Score {risk_score_global:.0f}/100. 
          Work Order WO-{machine_id:03d}-{selected_cycle:04d} {decision.replace("_", "-")}. 
          Technician assigned. Immediate review required.<br>
          <span style='color:#4a5568; font-size:0.68rem;'>Delivered · Cycle {selected_cycle}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_dismiss:
        if st.button("✕", key="dismiss_sms"):
            st.session_state['show_sms'] = False
            st.rerun()

# ─────────────────────────────────────────────
# 12. FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-top:40px; padding:16px; border-top:1px solid #1a2332;
     font-family:Share Tech Mono,monospace; font-size:0.7rem; color:#6b7f8f;
     text-align:center; letter-spacing:0.15em;'>
  PRESENSE SME AGENT &nbsp;·&nbsp; 
  RANDOM FOREST · SHAP · MCP · PELT ·
  V HACK 2026 &nbsp;·&nbsp; ROTI CANAI
</div>
""", unsafe_allow_html=True)