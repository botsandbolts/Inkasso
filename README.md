# Inkasso · Offline Deep Reinforcement Learning Lab

A business-friendly **Streamlit dashboard** and an educational, reproducible **discrete Conservative Q-Learning (CQL)** demonstration for debt-collection action policies.

> **Important:** This is simulation research, **not** a validated debt-collection decision system. [OpenAR](https://github.com/OpenAR-Collective/simple-mock-collections-data-generator) generates synthetic debt accounts and payment records, but **does not establish causal action-response effects**. Therefore the prototype trains CQL on **explicitly hypothetical interactions** sampled from a separate simulator, using OpenAR only as the starting profile distribution.

## GitHub Codespaces: automatic launch

When you **create or start an Inkasso Codespace**, the development-container configuration installs dependencies during creation, runs `bash scripts/start_dashboard.sh` on startup, forwards port **8501**, and requests an automatic **Preview in Editor** for the Streamlit dashboard.

**Create your Codespace:** [Inkasso Codespaces creation](https://github.com/codespaces/new?repo=botsandbolts/Inkasso).

If your Codespace was created *before* this setting was added, pull the latest `main` and run **Codespaces: Rebuild Container** from the VS Code Command Palette to apply the updated development-container configuration. GitHub's client may require you to click the preview if the automatic preview is blocked. The log is `/tmp/inkasso-streamlit.log`.

**Note:** Starting or creating a Codespace still requires opening it in GitHub; this configuration automates the dashboard *inside* the running Codespace, not creation or start of a stopped Codespace.

## New guided business presentation (six screens)

The Streamlit homepage is now designed for managers, clients and recruiters rather than primarily data scientists:

1. **Start here** — The business problem, from portfolio to action to learning to outcomes.
2. **Understand the portfolio** — Plain-language KPIs, balance distribution, contact reachability and why they matter for decisions.
3. **Try a collection decision** — Follow one fictional account week by week; choose an allowed contact action and see its *simulated* consequences.
4. **How the AI learns** — A visual four-stage loop, prediction vs sequential decision-making and an accessible explanation of Conservative Q-Learning.
5. **Compare strategies** — Contrast waiting, fixed contact rules, random permitted actions and an AI-learned strategy; clearly label any outputs as **simulation-only**. Training can be started here.
6. **Trust, data & definitions** — What the numbers mean, what the simulator assumes, restrictions and a plain-English glossary.

Use the **left sidebar** to move between the screens. Click **Portfolio settings** to inspect a different sample. By default, the app works with entirely fictional fallback profiles even if OpenAR has not been downloaded.

This is **not a live collections tool**. No direct consumer outreach exists, no improvement is fabricated before running a training experiment, and the contact-response simulator is not fitted to real causal outcomes.

## Publish Inkasso as a standalone website (no Codespace required to view)

Use **[Streamlit Community Cloud](https://share.streamlit.io/)** to publish this public GitHub repository with a stable shareable `*.streamlit.app` URL.

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using your GitHub account `botsandbolts` and authorize it to read the Inkasso repository.
2. Click **Create app → Yup, I have an app**.
3. Set **Repository:** `botsandbolts/Inkasso`, **Branch:** `main`, **Main file path:** `app.py`.
4. Optionally choose a memorable **App URL** subdomain (subject to availability). Under **Advanced settings**, choose **Python 3.11**.
5. Click **Deploy**. The service installs `requirements.txt` and presents the six-screen business app at its own public URL. Share that actual URL after deployment; do not infer it from the repository name.

**Hosting behavior:** Unlike a GitHub Codespace, the URL stays the same across visits and you do not need to run a terminal to open the app. On the free Community Cloud tier, apps with no visitors for 12 hours can hibernate, requiring visitors to wake them. GitHub commits update the deployed application automatically. All account data shown are synthetic.

**Keeping the site lightweight:** `requirements.txt` installs only the website's presentation dependencies. For a reproducible Discrete CQL experiment in Codespaces, install `requirements-train.txt` instead (the devcontainer is already configured to do so). The hosted website intentionally disables expensive model training. The live interactive customer journey still works.

### Publish real *simulated* comparison results

In a Codespace with the training dependencies installed:

```bash
python -m inkasso.experiment --data-dir data/openar --episodes 700 --steps 1500 --seed 42
bash scripts/publish_report.sh
git diff --cached -- reports/latest
git commit -m "Publish simulated policy comparison"
git push origin main
```

The publication helper stages **only aggregate simulation scorecards and simulator metadata** in `reports/latest/`. The hosted app automatically displays that scorecard once it is committed. It never presents assumed simulation gains as validated, real-world causal improvements. These reports do not exist until someone has successfully trained and published them.

## Start the dashboard

Python 3.11 recommended.

```bash
python -m pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```

The dashboard starts immediately in explicitly labeled **SIMULATED FALLBACK** mode. To load the actual OpenAR-generated **synthetic** data:

```bash
git clone https://github.com/OpenAR-Collective/simple-mock-collections-data-generator.git vendor/openar
python vendor/openar/generate.py --seed "Inkasso-2026" --accounts 10000 --out "$(pwd)/data/openar"
streamlit run app.py --server.address 0.0.0.0
```

The standard `generate.py` output is *clean* (without intentionally planted data-quality defects). Run the upstream validation script from its own directory if you want to audit the generation.

## Train and compare offline RL

```bash
python -m inkasso.experiment --data-dir data/openar --episodes 700 --steps 1500 --seed 42
```

This generates behavior-policy trajectories from the synthetic simulator, fits [d3rlpy DiscreteCQL](https://d3rlpy.readthedocs.io/en/latest/references/algos.html), evaluates policies on held-out accounts using the same simulator assumptions, then writes `outputs/metrics.csv`, `outputs/metadata.json` and `models/discrete_cql.d3`. These files are intentionally excluded from Git. Reload the Streamlit dashboard to display results.

**Compared policies:** Wait only, transparent rule-based outreach, random permitted outreach, Discrete CQL.

## Why CQL?

- Collections decisions are sequential and actions are discrete; historical event streams motivate *offline* RL.
- CQL penalizes high values assigned to unseen/uncommon actions, combating out-of-distribution overestimation.
- [CQL paper: Kumar et al., NeurIPS 2020](https://arxiv.org/abs/2006.04779)
- For model-free offline RL, next comparisons should include **Discrete BCQ / BC**, and **IQL** only if using a suitable compatible action-space implementation. Standard `d3rlpy` IQL targets continuous actions, so it is **not** a drop-in equivalent for this discrete action setup.

### Simulator assumptions and ethical guardrails

Allowed actions: WAIT, EMAIL, LETTER, CALL, OFFER_PLAN. `inkasso/env.py` contains **author-chosen**, inspectable action-response uplift coefficients; the dataset did **not** estimate them. Actions require channel availability, confirmed simulated right-party contact before a plan offer, and a small weekly outreach budget. Stop/review, dispute, represented, cease-and-desist, or closed accounts are never contacted in this demo. These toy rules **do not replace** real collections compliance screening or legal advice.

Reward = (hypothetical payment − hypothetical action cost) / 100. A different simulator could rank policies differently; no simulated gain should be represented as a validated business lift. The upstream data represent fictional US-style accounts, not German Inkasso processes.

### Leakage and evaluation

The loader uses placement balance, placement-time debt age, and basic reachability. It excludes current balances, future payments and later notes from initial predictor features. Current account status is used only as a *conservative outreach safety flag*, not as a propensity predictor. This is not a historical offline causal benchmark. Real-world work needs (i) representative behavior-policy propensities, (ii) support checks, (iii) doubly robust/off-policy evaluation with confounding controls, (iv) prospective randomized validation, (v) jurisdiction-specific legal review and fairness assessment.

### Project map

```
app.py                      Streamlit dashboard
inkasso/data.py             Synthetic OpenAR profile loader and fallback
inkasso/env.py              Hypothetical, action-masked weekly simulator
inkasso/experiment.py       Offline dataset, Discrete CQL, policy evaluation
tests/test_env.py           Behavioral restrictions and reproducibility
```

Run tests: `pytest -q`.

OpenAR generator: Apache 2.0. This repository does not redistribute their generated synthetic CSVs.
