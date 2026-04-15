"""
dashboard.py
Streamlit dashboard for the AI Energy Trading Agent.

Run with:
    streamlit run dashboard.py

Features:
  - Manual input mode: enter live market values → get instant decision
  - Simulation mode: step through historical data hour by hour
  - Decision history log with profit tracking
  - Model performance metrics
  - Alert system for high-confidence opportunities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from models.predictor import EnergyPredictor, build_features
from agent.planning_agent import PlanningAgent

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Energy Trading AI Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS — clean, minimal styling
# ──────────────────────────────────────────────

st.markdown("""
<style>
  .metric-card {
      background: #f8f9fa;
      border-radius: 10px;
      padding: 16px 20px;
      border-left: 4px solid #1f77b4;
      margin-bottom: 10px;
  }
  .decision-SELL { color: #d62728; font-size: 28px; font-weight: 700; }
  .decision-BUY  { color: #2ca02c; font-size: 28px; font-weight: 700; }
  .decision-HOLD { color: #ff7f0e; font-size: 28px; font-weight: 700; }
  .confidence-bar { height: 12px; border-radius: 6px; background: #e0e0e0; }
  .explanation-box {
      background: #f0f4ff;
      border-radius: 8px;
      padding: 14px 18px;
      border-left: 3px solid #4a6cf7;
      font-size: 14px;
      line-height: 1.6;
  }
  .alert-high {
      background: #fff3cd;
      border-radius: 8px;
      padding: 10px 14px;
      border-left: 3px solid #ffc107;
  }
  .stButton > button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load models (cached so they only load once)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    predictor = EnergyPredictor()
    agent     = PlanningAgent(use_learned_model=True)
    try:
        predictor.load()
        agent.load()
        return predictor, agent, True
    except Exception as e:
        st.warning(f"Models not found — run `python train.py` first. Error: {e}")
        return None, None, False


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    path = "data/energy_data.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return build_features(df)
    return None


# ──────────────────────────────────────────────
# Session state — decision history
# ──────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if "sim_idx" not in st.session_state:
    st.session_state.sim_idx = 100   # start mid-dataset (past lags available)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_confidence_html(confidence: float) -> str:
    pct    = round(confidence * 100)
    color  = "#2ca02c" if pct >= 70 else ("#ff7f0e" if pct >= 40 else "#d62728")
    return f"""
    <div style="margin:6px 0;">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:13px;color:#666;">Confidence</span>
        <span style="font-size:13px;font-weight:600;color:{color};">{pct}%</span>
      </div>
      <div class="confidence-bar">
        <div style="width:{pct}%;height:100%;background:{color};border-radius:6px;transition:width 0.3s;"></div>
      </div>
    </div>
    """

def action_color(action):
    return {"SELL": "#d62728", "BUY": "#2ca02c", "HOLD": "#ff7f0e"}.get(action, "#999")

def action_emoji(action):
    return {"SELL": "⚡ SELL", "BUY": "🔋 BUY", "HOLD": "⏳ HOLD"}.get(action, action)


def profit_chart(profits: dict):
    actions = list(profits.keys())
    values  = list(profits.values())
    colors  = [action_color(a) for a in actions]
    fig = go.Figure(go.Bar(
        x=actions, y=values,
        marker_color=colors,
        text=[f"₹{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Expected profit (₹/unit)",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def history_chart(history: list):
    if not history:
        return None
    df_h = pd.DataFrame(history)
    fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Price & predictions", "Cumulative profit"),
                         vertical_spacing=0.12)

    fig.add_trace(go.Scatter(
        x=df_h.index, y=df_h["current_price"],
        name="Current price", line=dict(color="#1f77b4", width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_h.index, y=df_h["price_t1"],
        name="Forecast t+1", line=dict(color="#aec7e8", dash="dash")
    ), row=1, col=1)

    # Action markers
    for action, color in [("SELL","#d62728"), ("BUY","#2ca02c"), ("HOLD","#ff7f0e")]:
        mask = df_h["action"] == action
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df_h[mask].index, y=df_h[mask]["current_price"],
                mode="markers", name=action,
                marker=dict(color=color, size=10, symbol="circle"),
            ), row=1, col=1)

    df_h["cum_profit"] = df_h["best_profit"].cumsum()
    fig.add_trace(go.Scatter(
        x=df_h.index, y=df_h["cum_profit"],
        fill="tozeroy", name="Cumulative profit",
        line=dict(color="#2ca02c"),
    ), row=2, col=1)

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ──────────────────────────────────────────────
# MAIN DASHBOARD
# ──────────────────────────────────────────────

predictor, agent, models_ready = load_models()
df_feat = load_data()

# ── Header ─────────────────────────────────────
st.title("⚡ Autonomous AI Energy Trading Agent")
st.caption("Perceives market conditions · Predicts prices · Plans optimal actions · Explains every decision")

if not models_ready:
    st.error("❌ Models not trained yet. Run `python train.py` in your terminal first, then refresh.")
    st.stop()

# ── Sidebar ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Control Panel")
    mode = st.radio("Input mode", ["Manual input", "Simulate from data"], index=0)
    st.divider()

    if mode == "Manual input":
        st.subheader("Current market state")
        current_price = st.slider("Current price (₹/unit)", 15.0, 140.0, 55.0, 0.5)
        temperature   = st.slider("Temperature (°C)", 10.0, 45.0, 30.0, 0.5)
        demand        = st.slider("Demand (MW)", 80.0, 450.0, 270.0, 5.0)
        sunlight      = st.slider("Sunlight index (0–1)", 0.0, 1.0, 0.6, 0.05)
        wind_speed    = st.slider("Wind speed (m/s)", 0.0, 20.0, 8.0, 0.5)
        price_lag_1   = st.number_input("Price 1h ago (₹/unit)", value=float(current_price - 2), step=0.5)
        demand_lag_1  = st.number_input("Demand 1h ago (MW)", value=float(demand - 10), step=5.0)
        run_manual    = st.button("🤖 Get AI decision", use_container_width=True, type="primary")
    else:
        if df_feat is not None:
            max_idx = len(df_feat) - 1
            st.subheader("Simulation controls")
            sim_idx = st.slider("Hour index", 100, max_idx, st.session_state.sim_idx)
            st.session_state.sim_idx = sim_idx
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("◀ Prev", use_container_width=True):
                    st.session_state.sim_idx = max(100, st.session_state.sim_idx - 1)
                    st.rerun()
            with col_b:
                if st.button("Next ▶", use_container_width=True):
                    st.session_state.sim_idx = min(max_idx, st.session_state.sim_idx + 1)
                    st.rerun()
            auto_run = st.button("▶▶ Auto-step 10 hours", use_container_width=True)
        else:
            st.warning("Dataset not found. Run `python train.py` first.")

    st.divider()
    if st.button("🗑 Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ──────────────────────────────────────────────
# Get decision
# ──────────────────────────────────────────────

result      = None
sim_row     = None

if mode == "Manual input" and run_manual:
    # Build a dummy feature row from slider inputs
    dummy_row = pd.DataFrame([{
        "temperature": temperature, "sunlight": sunlight,
        "wind_speed": wind_speed, "demand": demand,
        "price": current_price, "is_weekend": 0, "hour": 14, "month": 6,
        "price_lag_1": price_lag_1, "price_lag_2": price_lag_1 - 1,
        "price_lag_3": price_lag_1 - 2,
        "demand_lag_1": demand_lag_1, "demand_lag_2": demand_lag_1 - 5,
        "demand_lag_3": demand_lag_1 - 8,
        "temperature_lag_1": temperature - 0.5,
        "price_rolling_mean_3h": (current_price + price_lag_1 * 2) / 3,
        "demand_rolling_mean_3h": (demand + demand_lag_1 * 2) / 3,
        "price_rolling_std_3h": abs(current_price - price_lag_1),
        "hour_sin":  np.sin(2 * np.pi * 14 / 24),
        "hour_cos":  np.cos(2 * np.pi * 14 / 24),
        "month_sin": np.sin(2 * np.pi * 6  / 12),
        "month_cos": np.cos(2 * np.pi * 6  / 12),
    }])
    preds        = predictor.predict(dummy_row)
    current_price_use = current_price
    result       = agent.decide(
        current_price=current_price_use,
        price_t1=preds["price_t1"],
        price_t2=preds["price_t2"],
        demand=preds["demand"],
        temperature=temperature,
        price_lag_1=price_lag_1,
        demand_lag_1=demand_lag_1,
    )
    st.session_state.history.append({
        "timestamp":     datetime.now().strftime("%H:%M:%S"),
        "current_price": current_price_use,
        "price_t1":      preds["price_t1"],
        "price_t2":      preds["price_t2"],
        "demand":        preds["demand"],
        "action":        result["action"],
        "confidence":    result["confidence"],
        "best_profit":   result["profits"][result["action"]],
    })

elif mode == "Simulate from data" and df_feat is not None:
    sim_row       = df_feat.iloc[st.session_state.sim_idx]
    current_price_use = float(sim_row["price"])
    preds         = predictor.predict(pd.DataFrame([sim_row]))
    result        = agent.decide(
        current_price=current_price_use,
        price_t1=preds["price_t1"],
        price_t2=preds["price_t2"],
        demand=float(sim_row["demand"]),
        temperature=float(sim_row["temperature"]),
        price_lag_1=float(sim_row.get("price_lag_1", current_price_use)),
        demand_lag_1=float(sim_row.get("demand_lag_1", sim_row["demand"])),
    )
    st.session_state.history.append({
        "timestamp":     str(sim_row.get("timestamp", st.session_state.sim_idx)),
        "current_price": current_price_use,
        "price_t1":      preds["price_t1"],
        "price_t2":      preds["price_t2"],
        "demand":        float(sim_row["demand"]),
        "action":        result["action"],
        "confidence":    result["confidence"],
        "best_profit":   result["profits"][result["action"]],
    })


# ──────────────────────────────────────────────
# Main display
# ──────────────────────────────────────────────

if result:
    # ── KPI row ────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Current price",    f"₹{result.get('current_price', current_price_use if 'current_price_use' in dir() else 0):.1f}")
    k2.metric("Forecast t+1",     f"₹{result['price_t1']:.1f}",
              delta=f"{result['price_t1'] - (current_price_use if 'current_price_use' in dir() else result['price_t1']):.1f}")
    k3.metric("Forecast t+2",     f"₹{result['price_t2']:.1f}",
              delta=f"{result['price_t2'] - result['price_t1']:.1f}")
    k4.metric("Demand",           f"{result['demand']:.0f} MW")
    k5.metric("Decisions logged", len(st.session_state.history))

    st.divider()

    # ── Decision + explanation ──────────────────
    col_dec, col_chart = st.columns([1, 1])

    with col_dec:
        st.subheader("Agent decision")
        color = action_color(result["action"])
        st.markdown(
            f'<div style="font-size:36px;font-weight:700;color:{color};margin:8px 0;">'
            f'{action_emoji(result["action"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(make_confidence_html(result["confidence"]), unsafe_allow_html=True)

        # Alert
        if result["confidence"] >= 0.8:
            st.markdown(
                f'<div class="alert-high">🔔 <strong>High-confidence signal</strong> — '
                f'strong alignment across all market indicators.</div>',
                unsafe_allow_html=True,
            )
        elif result["confidence"] < 0.35:
            st.warning("⚠️ Low confidence — market signals are mixed. Consider manual review.")

        st.markdown("**Agent reasoning:**")
        st.markdown(
            f'<div class="explanation-box">{result["explanation"]}</div>',
            unsafe_allow_html=True,
        )

    with col_chart:
        st.subheader("Simulated profit comparison")
        st.caption("Agent picks the highest expected profit action")
        st.plotly_chart(profit_chart(result["profits"]), use_container_width=True)

        # Profit breakdown table
        df_profits = pd.DataFrame([
            {"Action": a, "Expected profit (₹/unit)": v,
             "Selected": "✓" if a == result["action"] else ""}
            for a, v in result["profits"].items()
        ])
        st.dataframe(df_profits, hide_index=True, use_container_width=True)

else:
    # Placeholder when no decision yet
    st.info("👈 Use the sidebar to enter market data and get an AI decision.", icon="ℹ️")

    # Show last result from history if available
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.caption(f"Last decision at {last['timestamp']}: "
                   f"**{last['action']}** (confidence {round(last['confidence']*100)}%)")


# ──────────────────────────────────────────────
# Decision history
# ──────────────────────────────────────────────

st.divider()
st.subheader("📊 Decision history & performance")

if st.session_state.history:
    fig_hist = history_chart(st.session_state.history)
    if fig_hist:
        st.plotly_chart(fig_hist, use_container_width=True)

    df_hist = pd.DataFrame(st.session_state.history)

    col_h1, col_h2, col_h3 = st.columns(3)
    action_counts = df_hist["action"].value_counts()
    col_h1.metric("Total decisions", len(df_hist))
    col_h2.metric("Avg confidence",  f"{round(df_hist['confidence'].mean() * 100)}%")
    col_h3.metric("Cumulative profit", f"₹{df_hist['best_profit'].sum():.1f}")

    st.dataframe(
        df_hist[["timestamp","current_price","price_t1","demand","action","confidence","best_profit"]]
        .rename(columns={
            "current_price": "Price (₹)",
            "price_t1": "Forecast t+1 (₹)",
            "demand": "Demand (MW)",
            "best_profit": "Profit (₹/unit)",
        })
        .style.format({
            "Price (₹)": "{:.1f}",
            "Forecast t+1 (₹)": "{:.1f}",
            "Demand (MW)": "{:.0f}",
            "confidence": "{:.0%}",
            "Profit (₹/unit)": "{:.1f}",
        }),
        use_container_width=True,
        height=300,
    )
else:
    st.caption("No decisions logged yet — make a decision above to start tracking.")


# ──────────────────────────────────────────────
# Model performance section
# ──────────────────────────────────────────────

with st.expander("🔬 Model performance metrics"):
    if hasattr(predictor, "metrics") and predictor.metrics:
        m = predictor.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price t+1 MAE",  f"₹{m.get('price_t1_mae','—')}/unit")
        c2.metric("Price t+1 R²",   m.get('price_t1_r2','—'))
        c3.metric("Demand MAE",     f"{m.get('demand_mae','—')} MW")
        c4.metric("Demand R²",      m.get('demand_r2','—'))
    else:
        st.info("Train models to see performance metrics.")

    if hasattr(agent, "cv_accuracy") and agent.cv_accuracy:
        st.metric("Agent Decision Tree CV accuracy",
                  f"{round(agent.cv_accuracy * 100, 1)}%")


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.divider()
st.caption(
    "Autonomous AI Energy Trading Agent · "
    "Stack: Python · Pandas · scikit-learn · Streamlit · Plotly · "
    "Models: Linear Regression (price) · Random Forest (demand) · "
    "Agent: Decision Tree (planning) · Future scope: RL (Q-learning)"
)
