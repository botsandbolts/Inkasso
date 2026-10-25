"""Offline discrete CQL trained and evaluated *only* on a synthetic simulator.

OpenAR determines placement profiles, not treatment effects. Every comparative
reward is contingent on the specified simulator assumptions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from .data import load_profiles
from .env import ACTIONS, CollectionEnv


def choose_rule(env: CollectionEnv, rng: np.random.Generator) -> int:
    valid = env.permitted()
    if env.week in (0, 4, 8):
        for a in (1, 2, 3):
            if a in valid:
                return a
    return 0


def choose_random(env: CollectionEnv, rng: np.random.Generator) -> int:
    valid = env.permitted()
    weights = np.array([.45 if a == 0 else .55 / (len(valid) - 1)
                        for a in valid], dtype=float) if len(valid) > 1 else np.array([1.])
    weights /= weights.sum()
    return int(rng.choice(valid, p=weights))


def simulate_logged(profiles: pd.DataFrame, seed: int = 42) -> tuple[Any, int]:
    """Generate a logged behavior-policy dataset; NEVER label it as observed OpenAR actions."""
    try:
        from d3rlpy.constants import ActionSpace
        from d3rlpy.dataset import MDPDataset
    except ImportError as ex:
        raise RuntimeError("Install d3rlpy with pip install -r requirements.txt") from ex

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminals: list[int] = []
    rng = np.random.default_rng(seed)
    for i, (_, profile) in enumerate(profiles.iterrows()):
        env = CollectionEnv(profile, seed=seed + i * 1291)
        while not env.done:
            state = env.observe()
            action = choose_random(env, rng)
            result = env.step(action)
            observations.append(state)
            actions.append(action)
            rewards.append(result.reward)
            terminals.append(int(result.done))
    dataset = MDPDataset(
        observations=np.asarray(observations, np.float32),
        actions=np.asarray(actions, np.int64),
        rewards=np.asarray(rewards, np.float32),
        terminals=np.asarray(terminals, np.float32),
        action_space=ActionSpace.DISCRETE,
        action_size=len(ACTIONS),
    )
    return dataset, len(actions)


def train_cql(dataset: Any, n_steps: int = 1500):
    from d3rlpy.algos import DiscreteCQLConfig
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    model = DiscreteCQLConfig(
        batch_size=64,
        gamma=.98,
        alpha=1.0,
        learning_rate=6.25e-5,
        target_update_interval=250,
    ).create(device=False)
    model.fit(dataset, n_steps=n_steps, n_steps_per_epoch=min(500, n_steps),
              show_progress=False, experiment_name="inkasso_cql_simulated")
    return model


def eval_policy(profiles: pd.DataFrame, name: str, policy, seed: int = 1001) -> dict:
    collected = costs = contacts = attempted = 0.
    any_recovery = 0
    for i, (_, profile) in enumerate(profiles.iterrows()):
        env = CollectionEnv(profile, seed=seed + i * 1291)
        rng = np.random.default_rng(seed + i * 61 + 37)
        account_payments = 0.
        while not env.done:
            selected = int(policy(env, rng))
            attempted += int(selected != 0)
            action = selected if selected in env.permitted() else 0  # Safe deployment projection.
            outcome = env.step(action)
            collected += outcome.payment
            account_payments += outcome.payment
            costs += outcome.cost
            contacts += int(action != 0)
        any_recovery += int(account_payments > 0)
    n = len(profiles)
    return {
        "policy": name,
        "simulated_collection_usd": round(collected, 2),
        "simulated_cost_usd": round(costs, 2),
        "simulated_net_usd": round(collected - costs, 2),
        "simulated_paid_accounts_pct": round(100 * any_recovery / n, 2),
        "simulated_recovery_pct_of_balance": round(100 * collected / profiles.balance.sum(), 2),
        "outreach_per_account": round(contacts / n, 3),
        "ineligible_actions_replaced": int(attempted - contacts),
        "evaluated_accounts": n,
    }


def run_experiment(data_dir: str = "data/openar", episodes: int = 700,
                   steps: int = 1500, seed: int = 42, out_dir: str = "outputs") -> pd.DataFrame:
    profiles, source = load_profiles(data_dir, n=episodes + max(150, episodes // 4), seed=seed)
    if len(profiles) < 50:
        raise ValueError("Need at least 50 profiles")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(profiles))
    cut = min(episodes, max(20, int(.8 * len(profiles))))
    train = profiles.iloc[indices[:cut]].reset_index(drop=True)
    heldout = profiles.iloc[indices[cut:]].reset_index(drop=True)
    if heldout.empty:
        raise ValueError("No held-out accounts")
    dataset, n_transitions = simulate_logged(train, seed)
    algo = train_cql(dataset, steps)
    policies = {
        "Wait only": lambda e, r: 0,
        "Transparent rules": choose_rule,
        "Random permitted": choose_random,
        "Discrete CQL": lambda e, r: int(algo.predict(e.observe()[None, :])[0]),
    }
    rows = [eval_policy(heldout, name, fn, seed=seed+7) for name, fn in policies.items()]
    result = pd.DataFrame(rows)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "metrics.csv", index=False)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    algo.save_model(str(model_dir / "discrete_cql.d3"))
    meta = {
        "source": source,
        "evaluation_type": "SIMULATOR-ONLY; NOT CAUSAL EVIDENCE",
        "train_accounts": len(train), "test_accounts": len(heldout),
        "logged_simulator_transitions": n_transitions,
        "seed": seed, "steps": steps,
        "horizon_weeks": 12,
        "hypothesis": "Assumed uplift multipliers are specified in inkasso/env.py",
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inkasso — clearly hypothetical offline RL experiment")
    parser.add_argument("--data-dir", default="data/openar")
    parser.add_argument("--episodes", type=int, default=700)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(run_experiment(args.data_dir, args.episodes, args.steps, args.seed).to_string(index=False))


if __name__ == "__main__":
    main()
