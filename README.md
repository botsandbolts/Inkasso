# Inkasso · Offline Deep Reinforcement Learning Lab

A business-friendly **Streamlit dashboard** and an educational, reproducible **discrete Conservative Q-Learning (CQL)** demonstration for debt-collection action policies.

> **Important:** This is simulation research, **not** a validated debt-collection decision system. [OpenAR](https://github.com/OpenAR-Collective/simple-mock-collections-data-generator) generates synthetic debt accounts and payment records, but **does not establish causal action-response effects**. Therefore the prototype trains CQL on **explicitly hypothetical interactions** sampled from a separate simulator, using OpenAR only as the starting profile distribution.

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
