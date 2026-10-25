"""Clearly hypothetical action-response model for *synthetic* debt portfolios."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np

ACTIONS = ("WAIT", "EMAIL", "LETTER", "CALL", "OFFER_PLAN")
COSTS = np.array([0., 0.20, 1.30, 3.80, 2.00], dtype=float)
# Hypothetical *multipliers* — intentionally NOT learned from OpenAR history.
UPLIFTS = np.array([1.00, 1.23, 1.38, 1.70, 2.10], dtype=float)
HORIZON = 12
MAX_OUTREACH = 4


@dataclass
class Outcome:
    reward: float
    payment: float
    cost: float
    done: bool


class CollectionEnv:
    def __init__(self, profile: Mapping, seed: int = 42, horizon: int = HORIZON):
        self.p = profile
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.original = float(profile["balance"])
        self.balance = self.original
        self.week = 0
        self.outreach = 0
        self.rpc = False
        self.done = False

    def observe(self) -> np.ndarray:
        return np.array([
            np.log1p(self.balance) / 10,
            np.log1p(self.original) / 10,
            min(2500., float(self.p["debt_age_days"])) / 2500,
            float(self.p["phone"] and not self.p["do_not_call"]),
            float(self.p["email"]),
            float(self.p["address"]),
            float(self.p["paid_creditor"]),
            self.week / self.horizon,
            self.outreach / MAX_OUTREACH,
            float(self.rpc),
            float(self.p["hold"]),
        ], dtype=np.float32)

    def permitted(self) -> list[int]:
        if self.done or bool(self.p["hold"]):
            return [0]
        out = [0]
        if self.outreach < MAX_OUTREACH:
            if bool(self.p["email"]):
                out.append(1)
            if bool(self.p["address"]):
                out.append(2)
            if bool(self.p["phone"]) and not bool(self.p["do_not_call"]):
                out.append(3)
            if self.rpc:  # A proposal requires preceding confirmed contact in this demo.
                out.append(4)
        return out

    def step(self, action: int) -> Outcome:
        if self.done:
            raise RuntimeError("Episode is complete")
        if action not in self.permitted():
            raise ValueError(f"Disallowed action {action}: {ACTIONS[action] if 0 <= action < len(ACTIONS) else '?'}")
        cost = float(COSTS[action])
        if action != 0:
            self.outreach += 1
        if action == 3 and self.rng.random() < 0.32:
            self.rpc = True
        # Calibrated neither to true treatment effects nor to real-world policies.
        p12 = float(self.p["p12"])
        hazard = 1 - (1 - p12) ** (1 / self.horizon)
        if bool(self.p["hold"]):
            hazard = 0.0   # Excluded pending human review; not optimized for outreach.
        effective_hazard = min(.90, hazard * UPLIFTS[action] * max(.65, 1 - .09 * self.outreach))
        payment = 0.
        if self.rng.random() < effective_hazard:
            # Non-guaranteed and partial repayment; no added fees or interest.
            fraction = self.rng.uniform(.15, .65)
            payment = min(self.balance, max(5., fraction * self.original))
            self.balance = max(0., self.balance - payment)
        self.week += 1
        self.done = self.balance < 1. or self.week >= self.horizon
        return Outcome(reward=(payment - cost) / 100.0, payment=payment, cost=cost, done=self.done)
