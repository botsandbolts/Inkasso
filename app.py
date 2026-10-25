"""Inkasso: a guided business story connecting collection data to deep RL decisions.

All accounts and outcomes are synthetic. No simulated policy gain constitutes
evidence of a real-world causal uplift.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from inkasso.data import load_profiles, source_counts
from inkasso.env import ACTIONS, COSTS, UPLIFTS, CollectionEnv

st.set_page_config(
    page_title="INKASSO | Smarter collections",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# A purpose-built presentation language: business concepts first, models last.
# -------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {max-width: 1240px; padding-top: 1.6rem; padding-bottom: 5rem;}
[data-testid="stSidebar"] {background: #0A1626; border-right:1px solid #25435A;}
[data-testid="stSidebar"] .stRadio label {line-height:1.7;}
h1,h2,h3 {letter-spacing: -.035em;}
h1 {font-weight:800;}
p, li {line-height:1.65;}
.hero {border:1px solid #355968; border-radius:22px; padding:30px 34px;
background: radial-gradient(circle at 95% 4%,#235C57 0%,transparent 33%),
linear-gradient(117deg,#172E45,#0B1829 70%); margin-bottom:20px;}
.hero-tag {letter-spacing:2.8px;font-size: .75rem;color:#91EBCB;font-weight:750;}
.hero h1 {font-size: clamp(2.0rem,3.5vw,3.2rem); line-height:1.13; margin:10px 0;}
.hero p {color:#CEDDEB;font-size:1.1rem;max-width:820px;margin-bottom:0;}
.story-card {background:#122237;border:1px solid #2B4961;border-radius:17px;
padding:20px 20px 16px; height:100%; min-height:150px;}
.story-n {font-size:.73rem;color:#83DCC0;letter-spacing:2px;font-weight:800;}
.story-card h3 {margin:8px 0;font-size:1.13rem;color:#F4F8FE;}
.story-card p {margin:0;color:#C1D1DF;font-size:.93rem;line-height:1.55;}
.annotation {border-left:4px solid #81DBC1;padding:13px 16px;margin:14px 0;
background:#142A38;border-radius:0 12px 12px 0;color:#D8E8ED;}
.small-print {color:#B9CAD8;font-size:.88rem;line-height:1.6;}
.stepbar {background:#1B2A3B;border-radius:14px;padding:12px 16px;
border:1px solid #314A5A;color:#CFE3EB;margin-bottom:14px;}
div[data-testid="stMetric"] {border:1px solid #2A435B;border-radius:15px;
padding:15px 18px;background:#122137;}
div[data-testid="stMetricLabel"] p {color:#C2D1DE; font-size:.95rem}
div[data-testid="stMetricValue"] {color:#F4F9FE;}
body {background:#081422;}
hr {border-color:#294155;}
</style>""", unsafe_allow_html=True)

MENU = [
    "1 · Start here",
    "2 · Understand the portfolio",
    "3 · Try a collection decision",
    "4 · How the AI learns",
    "5 · Compare strategies",
    "6 · Trust, data & definitions",
]
ACTION_NAMES = {
    0: "Wait and review later",
    1: "Send an email reminder",
    2: "Send a postal letter",
    3: "Make a permitted phone call",
    4: "Offer a repayment arrangement",
}
ACTION_SHORT = {0: "Wait", 1: "Email", 2: "Letter", 3: "Phone", 4: "Payment plan"}
ACTION_WHY = {
    0: "No immediate contact. Preserves the option to act later and avoids contact costs.",
    1: "Low-cost written reminder, when an email channel is available.",
    2: "Formal postal outreach where a postal channel exists.",
    3: "A permitted call can sometimes establish contact, but costs more.",
    4: "An arrangement can be proposed after confirmed contact in this demonstration.",
}


def safe(value: object) -> str:
    return html.escape(str(value), quote=True)


def money(value: float) -> str:
    return f"${value:,.0f}"


def heading(tag: str, title: str, subtitle: str) -> None:
    st.markdown(
        "<div class='hero'><div class='hero-tag'>" + safe(tag) + "</div>"
        "<h1>" + safe(title) + "</h1><p>" + safe(subtitle) + "</p></div>",
        unsafe_allow_html=True,
    )


def card(tag: str, title: str, text: str) -> None:
    st.markdown(
        "<div class='story-card'><div class='story-n'>" + safe(tag) + "</div>"
        "<h3>" + safe(title) + "</h3><p>" + safe(text) + "</p></div>",
        unsafe_allow_html=True,
    )


def take_away(text: str) -> None:
    st.markdown("<div class='annotation'>" + safe(text) + "</div>", unsafe_allow_html=True)


def clean_fig(fig, height: int = 360):
    fig.update_layout(
        height=height,
        margin=dict(l=5, r=6, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CDE0E9",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor="#41576B")
    fig.update_yaxes(gridcolor="#233B50", zeroline=False)
    return fig


st.sidebar.markdown("## ◈ INKASSO")
st.sidebar.caption("Smarter collection decisions · Business demo")
st.sidebar.divider()
st.sidebar.markdown("**Your guided tour**")
if "page" not in st.session_state:
    st.session_state.page = MENU[0]
page = st.sidebar.radio("Choose a screen", MENU, key="page", label_visibility="collapsed")
st.sidebar.divider()
with st.sidebar.expander("Portfolio settings", expanded=False):
    data_dir = st.text_input("OpenAR folder (advanced)", "data/openar")
    size = st.slider("Accounts in the sample", 200, 3000, 1200, 100)
    seed = st.number_input("Random seed for repeatable examples", min_value=0, value=42, step=1)
    st.caption("These settings select fictional accounts. They do not retrain the AI.")
st.sidebar.caption("No real customers · No real collection outreach")

try:
    profiles, source = load_profiles(data_dir, int(size), int(seed))
except Exception as exc:
    st.error(f"The portfolio could not be loaded: {exc}")
    st.stop()

is_openar = source.startswith("OPENAR")
source_label = "OpenAR-generated fictional accounts" if is_openar else "Illustrative demo accounts"
st.sidebar.markdown(f"**Data:** {source_label}")
st.sidebar.caption(f"{len(profiles):,} accounts in view")

# An explicit link between portfolio description, chosen actions, simulated
# outcomes, learning and policy evaluation appears on every screen.
position = MENU.index(page) + 1
st.markdown(
    f"<div class='stepbar'>GUIDED STORY &nbsp; {position} OF {len(MENU)} "
    f"&nbsp; · &nbsp; {safe(source_label)} &nbsp; · &nbsp; "
    "All outcomes in this app are hypothetical</div>",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------------
# SCREEN 1. What is the actual business question?
# -------------------------------------------------------------------------
if page == MENU[0]:
    heading(
        "THE COLLECTIONS DECISION · FIRST, THE BIG PICTURE",
        "Who should we contact, how, and when?",
        "Inkasso is not a model that simply predicts whether someone will pay. "
        "It explores a harder management question: which permitted next step "
        "could create the best long-term outcome with fewer unnecessary contacts?",
    )

    st.markdown("### One story, four connected parts")
    cols = st.columns(4)
    items = [
        ("01 · INPUT", "Understand the cases",
         "We begin with a fictional portfolio: balances, age of debt, available channels and cases on hold."),
        ("02 · DECISION", "Choose the next step",
         "For each case, choose whether to wait, send a reminder, make a permitted call or propose a plan."),
        ("03 · LEARNING", "See what happens next",
         "A simulated customer might repay, partly repay or not respond. The next decision depends on what happened."),
        ("04 · BUSINESS RESULT", "Compare whole strategies",
         "We compare total recovery, contact workload and cost across different decision-making approaches."),
    ]
    for col, item in zip(cols, items):
        with col:
            card(*item)

    st.markdown("### Why this is different from traditional analytics")
    left, right = st.columns(2, gap="large")
    with left:
        card("TRADITIONAL PREDICTION", "Will this account pay?",
             "A prediction model estimates the chance of repayment. It does not, by itself, decide when to contact someone or whether waiting is better.")
    with right:
        card("DEEP REINFORCEMENT LEARNING", "What is the best next permitted action?",
             "A decision model considers today's choice, its possible outcome and the effect on future choices. It learns a strategy across multiple weeks.")

    take_away(
        "The portfolio screen tells us WHAT cases exist. The AI-learning screen tells us "
        "HOW a strategy can use that information. The results screen asks WHETHER the "
        "learned strategy beats simpler approaches in the simulated world."
    )
    st.markdown("### What you can do")
    st.markdown(
        "Use the six screens in the left menu. First inspect the portfolio, then "
        "personally try a collection decision, see how the AI learns, and finally "
        "compare the strategies in a simulated test."
    )
    st.info(
        "This is an educational demonstration. The account data are synthetic. "
        "The effect of each collection action is an assumption in the simulator—not a proven real-world effect."
    )


# -------------------------------------------------------------------------
# SCREEN 2. Read the portfolio in plain language.
# -------------------------------------------------------------------------
elif page == MENU[1]:
    heading(
        "STEP 1 · UNDERSTAND THE ACCOUNTS",
        "What kind of cases are we managing?",
        "Before an AI chooses an action, it needs to understand the account. "
        "These figures describe the fictional portfolio; they are not AI results.",
    )

    cols = st.columns(4)
    cols[0].metric("Accounts", f"{len(profiles):,}", help="The number of fictional debts in this sample.")
    cols[1].metric("Amount owed", money(float(profiles.balance.sum())),
                   help="Original amount assigned for collection, not money recovered.")
    cols[2].metric("Typical account balance", money(float(profiles.balance.median())),
                   help="The median amount owed on one fictional account.")
    cols[3].metric("On hold / human review", f"{int(profiles.hold.sum()):,}",
                   help="Cases the demonstration blocks from collection contact.")

    st.caption(
        "**Accounts** = separate fictional debts. **Amount owed** = the opening balance, not revenue. "
        "**On hold** = cases that need review instead of automated outreach."
    )
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("How large are the balances?")
        bands = pd.cut(
            profiles.balance, bins=[0, 250, 1000, 3000, 10000, np.inf],
            labels=["Up to $250", "$251–$1,000", "$1,001–$3,000", "$3,001–$10,000", "Above $10,000"],
            include_lowest=True,
        )
        tab = bands.value_counts().reindex(bands.cat.categories, fill_value=0)
        fig = px.bar(x=tab.index.astype(str), y=tab.to_numpy(), color_discrete_sequence=["#7FD9C2"],
                     labels={"x": "Original balance", "y": "Number of accounts"})
        st.plotly_chart(clean_fig(fig), use_container_width=True)
        st.caption("Why the AI cares: action costs matter more for a small balance than for a large one.")
    with col2:
        st.subheader("Which contact channels exist?")
        channels = pd.DataFrame({
            "Channel": ["Email", "Phone (before restrictions)", "Postal address"],
            "Accounts": [int(profiles.email.sum()), int(profiles.phone.sum()), int(profiles.address.sum())],
        })
        fig = px.bar(channels, x="Accounts", y="Channel", orientation="h",
                     color_discrete_sequence=["#75B6DC"])
        st.plotly_chart(clean_fig(fig), use_container_width=True)
        st.caption("Why the AI cares: unavailable or restricted channels must never be selected.")

    st.subheader("Why this portfolio matters for the decision engine")
    c1, c2, c3 = st.columns(3)
    with c1:
        card("SIGNAL 1", "Outstanding amount",
             "A larger balance may justify more effort—but only if the expected benefit outweighs contact costs.")
    with c2:
        card("SIGNAL 2", "Age of the debt",
             "Older cases may behave differently. This is an input, not proof of ability or willingness to pay.")
    with c3:
        card("SIGNAL 3", "Eligibility and history",
             "The next action also depends on permitted channels, previous outreach and whether a case is on hold.")

    take_away(
        "These portfolio characteristics become the starting situation for an AI decision. "
        "Now open 'Try a collection decision' to see exactly how one case changes over time."
    )


# -------------------------------------------------------------------------
# SCREEN 3. Real interactive step-through of the simulator.
# -------------------------------------------------------------------------
elif page == MENU[2]:
    heading(
        "STEP 2 · THE COLLECTOR'S CHOICES",
        "Try one customer journey",
        "You are the decision-maker in this exercise. Pick an allowed next step "
        "and see a hypothetical response. The learning algorithm later evaluates the very same choices.",
    )

    eligible = profiles.loc[~profiles.hold]
    held = profiles.loc[profiles.hold]
    options = pd.concat([eligible.head(7), held.head(2)], ignore_index=True)
    if options.empty:
        options = profiles.head(7).reset_index(drop=True)
    id_list = options.account_id.astype(str).tolist()
    selection = st.selectbox("Choose a fictional account", options=id_list,
                             format_func=lambda account: f"Account {account}")
    chosen = options.loc[options.account_id.astype(str).eq(selection)].iloc[0]
    identity = (str(selection), int(seed), int(size), str(source))
    if st.session_state.get("case_identity") != identity:
        st.session_state.case_identity = identity
        st.session_state.case_environment = CollectionEnv(chosen.to_dict(), seed=int(seed) + 701)
        st.session_state.case_history = []
        st.session_state.pop("chosen_action", None)
    env = st.session_state.case_environment

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original amount owed", money(env.original))
    c2.metric("Still outstanding", money(env.balance))
    c3.metric("Week of the journey", f"{env.week + 1 if not env.done else env.week} / {env.horizon}")
    c4.metric("Contacts used", f"{env.outreach} / 4")
    st.caption("Each choice moves the story forward by one simulated week. Outcomes vary with the repeatable random seed.")
    st.progress(min(1., env.week / env.horizon), text=f"Journey progress: {env.week} of {env.horizon} weeks")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Choose what to do next")
        permitted = env.permitted() if not env.done else []
        if env.done:
            st.success("This fictional journey is complete. Choose another account or reset the journey.")
        else:
            pick = st.radio(
                "Permitted actions for this week",
                options=permitted,
                format_func=lambda action: ACTION_NAMES[action],
                key="chosen_action",
            )
            st.caption(ACTION_WHY[pick] + f" Assumed action cost: ${COSTS[pick]:.2f}.")
            if st.button("Apply this action and simulate the next week", type="primary"):
                week = env.week + 1
                outcome = env.step(int(pick))
                st.session_state.case_history.append({
                    "Week": week,
                    "Decision": ACTION_SHORT[pick],
                    "Hypothetical payment": round(outcome.payment, 2),
                    "Contact cost": round(outcome.cost, 2),
                    "Balance after": round(env.balance, 2),
                })
                st.rerun()
        if st.button("Start this fictional case again", type="secondary"):
            st.session_state.case_environment = CollectionEnv(chosen.to_dict(), seed=int(seed) + 701)
            st.session_state.case_history = []
            st.rerun()
    with right:
        st.subheader("Why some actions are unavailable")
        if bool(chosen.hold):
            st.warning("This account requires human review. The simulator only allows waiting.")
        else:
            reasons = []
            if not bool(chosen.email):
                reasons.append("Email is unavailable.")
            if not bool(chosen.address):
                reasons.append("No usable postal channel.")
            if not bool(chosen.phone) or bool(chosen.do_not_call):
                reasons.append("Phone contact is unavailable or restricted.")
            if not env.rpc:
                reasons.append("A payment plan is only offered after confirmed contact in this demo.")
            if env.outreach >= 4:
                reasons.append("The illustrative contact limit has been reached.")
            if not reasons:
                reasons.append("All applicable demo checks currently pass.")
            for reason in reasons:
                st.markdown("- " + reason)
        st.info("A real collection operation needs fuller legal, consent, vulnerability and human-review safeguards.")

    st.subheader("What happened so far?")
    history = st.session_state.case_history
    if history:
        frame = pd.DataFrame(history)
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.caption(
            "Payments above were sampled from an invented response model. "
            "They do not show whether the chosen contact caused a payment."
        )
    else:
        st.info("No decisions yet. Choose an action above to begin the fictional journey.")

    take_away(
        "You just chose actions one week at a time. A deep reinforcement learning model "
        "tries to learn a repeatable strategy for choosing among permitted actions—"
        "while considering how today's choice affects the later weeks."
    )


# -------------------------------------------------------------------------
# SCREEN 4. Explain the mechanics without leading with acronyms.
# -------------------------------------------------------------------------
elif page == MENU[3]:
    heading(
        "STEP 3 · THE DECISION ENGINE",
        "How can an AI learn a better strategy?",
        "Instead of asking only 'who will repay?', the AI practices choosing "
        "the next permitted action, observes a simulated response, and updates "
        "its view of what works over the entire customer journey.",
    )
    st.subheader("The learning loop")
    cols = st.columns(4)
    for col, item in zip(cols, [
        ("1 · UNDERSTAND", "Read the situation",
         "Current balance, how long overdue, available channels, prior contacts and restrictions."),
        ("2 · DECIDE", "Choose an action",
         "Wait, email, letter, permitted call or propose a plan—only from the actions allowed."),
        ("3 · OBSERVE", "See the response",
         "The fictional customer may pay some money, make no payment, or become contactable."),
        ("4 · LEARN", "Judge the full outcome",
         "The model weighs money collected, action cost and what choices remain in future weeks."),
    ]):
        with col:
            card(*item)
    take_away(
        "Think of it as a decision coach that learns from many fictional customer journeys, "
        "not a crystal ball predicting one isolated payment."
    )

    st.subheader("What is the difference from an ordinary prediction model?")
    comp = pd.DataFrame([
        {"Question": "What does it answer?", "Ordinary prediction": "How likely is repayment?",
         "Decision-learning AI": "Which permitted action should happen next?"},
        {"Question": "Does timing matter?", "Ordinary prediction": "Not necessarily",
         "Decision-learning AI": "Yes—waiting can be a sensible choice."},
        {"Question": "Can a choice affect future options?", "Ordinary prediction": "Not modeled directly",
         "Decision-learning AI": "Yes—it considers several weeks."},
        {"Question": "How is it judged?", "Ordinary prediction": "Prediction accuracy",
         "Decision-learning AI": "Total recoveries, costs and contacts."},
    ])
    st.dataframe(comp, hide_index=True, use_container_width=True)

    st.subheader("How this particular research model learns")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### It studies earlier journeys instead of experimenting with real people")
        st.write(
            "A simulator creates thousands of fictional week-by-week examples. "
            "Each example records the situation, permitted action, simulated payment, "
            "and next situation. The AI studies those examples offline."
        )
        st.write(
            "This method is called **offline reinforcement learning**. "
            "The word 'offline' matters: no customers are contacted during training."
        )
    with b:
        st.markdown("#### It is careful about actions it has rarely seen")
        st.write(
            "**Conservative Q-Learning (CQL)** is the technique used here. "
            "In plain language: it avoids being overconfident about an action "
            "when the training examples give little reason to trust it."
        )
        st.write(
            "It is one possible strategy learner, not automatically the best one. "
            "The next screen compares it with simpler approaches."
        )

    with st.expander("What do the technical terms mean?"):
        st.markdown(
            "**Deep learning:** neural networks learn patterns from many examples.\n\n"
            "**Reinforcement learning:** learning a sequence of choices by rewarding good longer-term outcomes.\n\n"
            "**Offline:** learning from a recorded set of simulated journeys rather than live trials.\n\n"
            "**Policy:** a practical decision rule for what to do next.\n\n"
            "**CQL:** Conservative Q-Learning, a method intended to be cautious about poorly supported actions."
        )
    st.warning(
        "Important distinction: OpenAR provides fictional case profiles. The claimed "
        "effectiveness of calls, emails or payment plans is not measured by OpenAR. "
        "Those effects are assumptions in our simulator."
    )


# -------------------------------------------------------------------------
# SCREEN 5. Benchmarks only if a run really exists.
# -------------------------------------------------------------------------
elif page == MENU[4]:
    heading(
        "STEP 4 · MEASURE THE BUSINESS OUTCOME",
        "Does the AI outperform simpler approaches?",
        "The key test is not whether the model is sophisticated. "
        "It is whether a complete decision strategy produces better outcomes "
        "under the same hypothetical customer-response assumptions.",
    )
    st.markdown("### The four approaches we compare")
    cols = st.columns(4)
    for col, item in zip(cols, [
        ("BASELINE", "Wait and observe",
         "No outreach. Shows what the simulated portfolio might recover without contact."),
        ("CURRENT-STYLE PROCESS", "Fixed calendar rules",
         "Send basic reminders on a schedule. Everyone follows the same transparent rules."),
        ("CHANCE", "Random permitted actions",
         "A weak benchmark: pick from allowed options without learning a strategy."),
        ("LEARNED STRATEGY", "AI chooses the next step",
         "Conservative Q-Learning learns a policy from fictional customer journeys."),
    ]):
        with col:
            card(*item)

    path = Path("outputs/metrics.csv")
    meta_file = Path("outputs/metadata.json")
    if path.exists() and meta_file.exists():
        try:
            results = pd.read_csv(path)
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            required = {
                "policy", "simulated_net_usd", "simulated_paid_accounts_pct",
                "simulated_recovery_pct_of_balance", "outreach_per_account",
                "simulated_collection_usd", "simulated_cost_usd",
            }
            if not required.issubset(results.columns):
                raise ValueError("Results file uses an unsupported format")
        except (ValueError, OSError, json.JSONDecodeError) as err:
            st.error(f"Experiment results could not be read: {err}")
        else:
            st.success(
                f"Simulation completed: {meta.get('test_accounts','?')} held-out fictional accounts. "
                f"Source: {meta.get('source', 'unknown')}."
            )
            st.caption(
                "Numbers below are **simulated** and depend on assumed response rates, "
                "action effects and random variation—not measured business lift."
            )
            label = {
                "Wait only": "Wait and observe",
                "Transparent rules": "Fixed calendar rules",
                "Random permitted": "Random permitted actions",
                "Discrete CQL": "AI learned strategy (CQL)",
            }
            results["Approach"] = results["policy"].map(label).fillna(results["policy"])
            top = results.loc[results.simulated_net_usd.idxmax()]
            st.markdown(
                f"**In this one simulation**, the highest net collection total came from "
                f"**{safe(top['Approach'])}**. This ranking might change with different assumptions."
            )
            l, r = st.columns(2, gap="large")
            with l:
                st.subheader("Money recovered after contact costs")
                viz = results.sort_values("simulated_net_usd")
                fig = px.bar(
                    viz, x="simulated_net_usd", y="Approach", orientation="h",
                    labels={"simulated_net_usd": "Simulated net collections ($)"},
                    color_discrete_sequence=["#80DBBF"],
                )
                st.plotly_chart(clean_fig(fig, height=330), use_container_width=True)
                st.caption("Net collections = fictional repayments minus simulated outreach expenses.")
            with r:
                st.subheader("How many contacts did each strategy make?")
                viz = results.sort_values("outreach_per_account")
                fig = px.bar(
                    viz, x="outreach_per_account", y="Approach", orientation="h",
                    labels={"outreach_per_account": "Average outreach attempts per account"},
                    color_discrete_sequence=["#8CAFE3"],
                )
                st.plotly_chart(clean_fig(fig, height=330), use_container_width=True)
                st.caption("Lower contacts can be better when collection outcomes are otherwise similar.")
            st.subheader("Business scorecard")
            table = results[[
                "Approach", "simulated_collection_usd", "simulated_cost_usd",
                "simulated_net_usd", "simulated_paid_accounts_pct",
                "outreach_per_account",
            ]].rename(columns={
                "simulated_collection_usd": "Gross collections ($)",
                "simulated_cost_usd": "Contact costs ($)",
                "simulated_net_usd": "Net collections ($)",
                "simulated_paid_accounts_pct": "Accounts paying something (%)",
                "outreach_per_account": "Contacts per account",
            })
            st.dataframe(
                table.style.format({
                    "Gross collections ($)": "${:,.0f}",
                    "Contact costs ($)": "${:,.0f}",
                    "Net collections ($)": "${:,.0f}",
                    "Accounts paying something (%)": "{:.1f}%",
                    "Contacts per account": "{:.2f}",
                }),
                hide_index=True, use_container_width=True,
            )
            take_away(
                "A good policy should balance recovery, workload and customer protection. "
                "This run compares simulated alternatives only; it cannot demonstrate real-world gains."
            )
    else:
        st.info(
            "**No model comparison has been run yet.** The dashboard will not invent "
            "AI performance numbers. You can run a test using the controls below."
        )
        st.markdown(
            "Once trained, this screen will display simulated net collections, "
            "the number of accounts that paid something, and contact workload "
            "for each approach."
        )

    st.markdown("### Run a simulated strategy comparison")
    st.caption("This trains the decision engine using fictional journeys. It does not send messages or make phone calls.")
    with st.expander("Test settings", expanded=False):
        train_accounts = st.slider("Fictional accounts used for learning", 150, 1600, 500, 50)
        training_steps = st.slider("AI learning iterations (more may take longer)", 100, 4000, 600, 100)
        st.caption(
            "More examples or iterations increase computing work, "
            "but do not guarantee that the learned strategy will win."
        )
    if st.button("Run the simulated strategy test", type="primary"):
        with st.spinner("Training on fictional customer journeys and testing all four approaches..."):
            try:
                from inkasso.experiment import run_experiment
                run_experiment(
                    data_dir=data_dir, episodes=int(train_accounts),
                    steps=int(training_steps), seed=int(seed),
                )
            except Exception as exc:
                st.error(f"Unable to complete the training run: {exc}")
                st.caption("The compute environment may need model dependencies or more resources.")
            else:
                st.rerun()


# -------------------------------------------------------------------------
# SCREEN 6. Trust and approachable dictionary.
# -------------------------------------------------------------------------
else:
    heading(
        "DATA, TRUST & PLAIN-ENGLISH DEFINITIONS",
        "What should—and shouldn't—we conclude?",
        "A responsible demo makes its assumptions visible. "
        "This prototype is for explaining strategy learning, not making "
        "decisions about real people.",
    )
    st.markdown("### Where does the information come from?")
    cols = st.columns(2)
    with cols[0]:
        card("DATA USED NOW", source_label,
             "Every account is fictional. Original amounts, contact availability and debt ages describe the simulated starting portfolio.")
    with cols[1]:
        card("WHERE RESULTS COME FROM", "A separate response simulator",
             "The app invents how fictional debtors react to different actions. An AI learns from those assumed responses.")
    counts = source_counts(data_dir)
    if counts:
        st.caption("Available OpenAR-generated files: " + ", ".join(f"{k}: {v:,} rows" for k, v in counts.items()))
    else:
        st.caption("OpenAR files were not found in the selected folder, so the app uses clearly marked fallback profiles.")

    st.markdown("### The five decisions in our demonstration")
    action_table = pd.DataFrame([
        {
            "Action in plain English": ACTION_NAMES[i],
            "Example cost": f"${float(COSTS[i]):.2f}",
            "What it represents": ACTION_WHY[i],
        }
        for i in range(len(ACTIONS))
    ])
    st.dataframe(action_table, hide_index=True, use_container_width=True)
    st.caption(
        "The example costs and action-effect multipliers are design assumptions, "
        "not verified statistics for Germany or any debt-collection agency."
    )

    st.markdown("### The essential terms")
    glossary = [
        ("Portfolio", "The collection of debt accounts the organization is managing."),
        ("Account balance", "Money originally assigned for collection, before simulated repayments."),
        ("Recovery rate", "The share of the original amount collected over the simulation period."),
        ("Net collections", "Collected money minus the modeled cost of communication actions."),
        ("Contact workload", "How often the collection team reaches out per account."),
        ("On hold", "A case blocked from outreach by the demonstration's review rules."),
        ("Strategy / policy", "A repeatable rule for choosing the next permitted action."),
        ("Deep reinforcement learning", "A neural-network-based method learning sequences of decisions from outcomes."),
        ("Conservative Q-Learning (CQL)", "The particular offline-learning algorithm used here; it is cautious about unsupported actions."),
        ("Synthetic data", "Artificial records invented to resemble real operations, not actual consumers."),
        ("Simulation", "A constructed model of how customers might respond. Results depend on its assumptions."),
        ("Causal effect", "A change proven to be caused by an action—not simply associated with it."),
    ]
    for term, meaning in glossary:
        with st.expander(term):
            st.write(meaning)

    st.markdown("### Safeguards and scientific limits")
    st.markdown(
        "- **Never infer causation from OpenAR records alone.** The provided data do not "
        "demonstrate how different actions change repayment outcomes.\n"
        "- **No real-world outreach.** Decisions affect only fictional cases inside the app.\n"
        "- **Human review and contact restrictions.** The prototype blocks some cases and "
        "unavailable channels, but this is not a complete legal compliance system.\n"
        "- **No automated deployment.** Real use needs jurisdiction-specific review, "
        "fairness testing, robust causal evaluation and prospective validation.\n"
        "- **No guaranteed AI advantage.** A simple policy may outperform the learned one, "
        "depending on assumptions and training."
    )
    take_away(
        "The value of this demo is to show HOW decisions can improve across a journey, "
        "what a rigorous comparison requires, and why apparently impressive simulated "
        "results must not be sold as measured real-world impact."
    )

st.divider()
st.caption(
    "INKASSO DECISION LAB  ·  Business demonstration  ·  Synthetic cases only  ·  "
    "Simulated action effects  ·  Not a live collection tool"
)
