"""Business-facing simulation research dashboard for collections policies."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from inkasso.data import load_profiles, source_counts
from inkasso.env import ACTIONS, COSTS, UPLIFTS

st.set_page_config(page_title="INKASSO | Decision Intelligence", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1320px; padding-top: 2rem;}
.hero {background:linear-gradient(115deg,#172C46,#0D172B 65%,#17392F);
 padding:28px 30px; border:1px solid #294A50; border-radius:20px; margin-bottom:18px;}
.eyebrow {color:#8ADCC5;letter-spacing:3px;font-weight:700;font-size:.75rem;}
.hero h1 {font-size:2.6rem;letter-spacing:-1.3px;margin:8px 0 0;}
.hero p {color:#C8D7E7;font-size:1.06rem;max-width:780px;margin:10px 0 0;}
div[data-testid="stMetric"] {border:1px solid #25354E;background:#101B2F;
 border-radius:14px;padding:16px;}
.pill {display:inline-block;background:#223D39;color:#9DF2D0;padding:6px 10px;
 border-radius:20px;font-size:.76rem;font-weight:700;}
</style>""", unsafe_allow_html=True)

st.sidebar.markdown("### ◈ INKASSO LAB")
st.sidebar.caption("OFFLINE POLICY INTELLIGENCE / RESEARCH DEMO")
data_dir = st.sidebar.text_input("OpenAR data folder", "data/openar")
size = st.sidebar.slider("Accounts to inspect", 200, 3000, 1200, 100)
seed = st.sidebar.number_input("Reproducible seed", min_value=0, value=42, step=1)
profiles, source = load_profiles(data_dir, size, int(seed))

st.markdown("""<div class='hero'><div class='eyebrow'>DECISION INTELLIGENCE / DEBT COLLECTION</div>
<h1>Smarter recovery. Fewer unnecessary contacts.</h1>
<p>Explore a transparent, consumer-aware approach to collections decisions using offline
reinforcement learning and explicitly hypothetical response simulation.</p></div>""", unsafe_allow_html=True)
st.info("**SIMULATION / EDUCATIONAL USE ONLY.** OpenAR is synthetic, and its log does not identify the causal effectiveness of collection actions. All RL policy improvements shown here (if trained) come from a separate assumed simulator.")
st.caption(f"DATA SOURCE: **{source}** · {len(profiles):,} sampled synthetic accounts · Seed {seed}")

num = len(profiles)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accounts in view", f"{num:,}")
m2.metric("Placed balance", f"${profiles.balance.sum()/1e6:,.2f}M")
m3.metric("No-hold profiles", f"{100*(~profiles.hold).mean():,.1f}%")
m4.metric("Illustrative 12-wk propensity", f"{100*profiles.p12.mean():,.1f}%")

t_overview, t_policy, t_methods = st.tabs(["Portfolio intelligence", "Policy experiments", "How the model works"])
with t_overview:
    a, b = st.columns([1.3, 1])
    with a:
        st.subheader("Debt distribution")
        fig = px.histogram(profiles, x="balance", nbins=45, log_y=False,
                           labels={"balance": "Initial placement balance ($)"})
        fig.update_traces(marker_color="#8ADCC5")
        fig.update_layout(height=365, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=15,t=15,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Contact channel reachability")
        channels = pd.DataFrame({"channel": ["Phone", "Email", "Postal"],
                                 "accounts": [int(profiles.phone.sum()), int(profiles.email.sum()),
                                              int(profiles.address.sum())]})
        fig = px.bar(channels, x="channel", y="accounts", color="channel",
                     color_discrete_sequence=["#8ADCC5", "#5E91D7", "#E1C88C"])
        fig.update_layout(showlegend=False, height=365, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=15,t=15,b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Data quality and portfolio restrictions")
    cols = st.columns(3)
    cols[0].metric("Cases requiring review / no outreach", int(profiles.hold.sum()))
    cols[1].metric("Do-not-call flag", int(profiles.do_not_call.sum()))
    cols[2].metric("Absent email channel", int((~profiles.email).sum()))
    counts = source_counts(data_dir)
    if counts:
        st.caption("Actual source-file row counts: " + " · ".join(f"{k}: {v:,}" for k,v in counts.items()))
    else:
        st.caption("OpenAR source CSVs were not found; visualized data are fabricated demo profiles.")

with t_policy:
    st.subheader("Offline DRL benchmark")
    path = Path("outputs/metrics.csv")
    if not path.exists():
        st.warning("No trained policies yet. Run the training command below; no performance figures are fabricated on this page.")
        st.code("python -m inkasso.experiment --data-dir data/openar --episodes 700 --steps 1500 --seed 42", language="bash")
    else:
        metric = pd.read_csv(path)
        meta_file = Path("outputs/metadata.json")
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf8"))
            st.markdown(f"<span class='pill'>SIMULATOR EVALUATION • {meta.get('test_accounts', '?')} HOLDOUT ACCOUNTS</span>", unsafe_allow_html=True)
            st.caption(f"Source: {meta.get('source')} · Fit steps: {meta.get('steps')} · Seed: {meta.get('seed')}")
        left, right = st.columns(2)
        with left:
            fig = px.bar(metric, x="policy", y="simulated_net_usd", color="policy",
                         title="Simulated net collections (USD)", text_auto=".2s")
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.scatter(metric, x="outreach_per_account", y="simulated_recovery_pct_of_balance",
                             text="policy", size="simulated_collection_usd",
                             title="Illustrative tradeoff: recovery vs contacts")
            fig.update_traces(textposition="top center")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metric, hide_index=True, use_container_width=True)
        st.caption("Not an empirical lift estimate; only valid conditional on the assumed state-transition and action-response rules.")
    st.markdown("#### Re-run an experiment")
    c1, c2 = st.columns(2)
    episodes = c1.slider("Training accounts", 150, 1600, 500, 50)
    steps = c2.slider("CQL gradient updates", 100, 4000, 600, 100)
    if st.button("Train discrete CQL and compare policies", type="primary"):
        with st.spinner("Training on simulator-generated episodes..."):
            try:
                from inkasso.experiment import run_experiment
                run_experiment(data_dir=data_dir, episodes=episodes, steps=steps, seed=int(seed))
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.caption("Check installed dependencies, source files and available memory.")

with t_methods:
    st.subheader("Why conservative Q-learning?")
    st.markdown("""**Discrete CQL** learns from prior episodes while deliberately lowering the value of actions not well represented in the logged dataset. The project uses *model-generated* logged episodes so that counterfactual policy evaluations are possible against the same synthetic world. It does **not** claim that observed OpenAR actions caused higher repayment rates.""")
    st.markdown("**State:** original balance, remaining balance, debt age, available contact channels, pre-placement creditor-payment indicator, weeks elapsed, contact count, and whether right-party contact was confirmed in the simulator.")
    st.markdown("**Action mask:** stop/review cases cannot receive outreach; blocked phone numbers are never called; plans are only proposed after simulated contact; total outreach is capped. Real-world legal compliance requires a significantly fuller rule engine.")
    st.markdown("**Reward:** simulated collected dollars minus outreach cost, divided by 100 for numeric stability.")
    assumptions = pd.DataFrame({"Action": ACTIONS, "Hypothetical response multiplier": UPLIFTS,
                                "Assumed cost ($)": COSTS})
    st.dataframe(assumptions, hide_index=True, use_container_width=True)
    st.caption("All multipliers and costs above are authored assumptions and can be edited in inkasso/env.py.")
    st.markdown("**Do not mistake:** repayment propensity prediction, retrospective association, and causal improvement from action optimization.")
