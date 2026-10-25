"""Placement-time OpenAR features. No current balance or future payments in states."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _yes(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])


def demo_profiles(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Purely fabricated fallback profiles, NOT OpenAR records."""
    rng = np.random.default_rng(seed)
    balance = np.exp(rng.uniform(np.log(90), np.log(10000), n))
    age = rng.integers(20, 1500, n)
    phone = rng.random(n) < .75
    email = rng.random(n) < .56
    address = rng.random(n) < .86
    paid_cred = rng.random(n) < .35
    hold = rng.random(n) < .08
    dnc = rng.random(n) < .10
    return _finish(pd.DataFrame({
        "account_id": [f"DEMO-{x:06d}" for x in range(n)],
        "balance": balance, "debt_age_days": age,
        "phone": phone, "email": email, "address": address,
        "paid_creditor": paid_cred, "hold": hold, "do_not_call": dnc,
        "source": "SIMULATED FALLBACK",
    }))


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    # Illustrative, NOT fitted causal response or calibrated OpenAR propensity.
    logit = (-1.40 - .48 * np.log10(np.maximum(1, df.balance) / 500)
             - .0004 * df.debt_age_days + .55 * df.phone.astype(float)
             + .38 * df.address.astype(float)
             + .45 * df.paid_creditor.astype(float))
    df = df.copy()
    df["p12"] = (1. / (1. + np.exp(-np.clip(logit, -20, 20)))).clip(.03, .90)
    return df


def load_profiles(data_dir: str = "data/openar", n: int = 1200, seed: int = 42) -> tuple[pd.DataFrame, str]:
    """OpenAR snapshot -> placement-time profiles; ignores end-of-life outcomes.

    The upstream data are SYNTHETIC. Current statuses may encode future information:
    they are intentionally excluded from observations; used only to mark exceptions
    in a demonstration safety mask, not as payment predictors.
    """
    p = Path(data_dir) / "accounts.csv"
    if not p.exists():
        return demo_profiles(n=n, seed=seed), "SIMULATED FALLBACK (OpenAR accounts.csv not found)"
    df = pd.read_csv(p, dtype=str, low_memory=False)
    required = {"account_id", "placement_balance", "placement_date", "charge_off_date"}
    absent = required - set(df)
    if absent:
        raise ValueError(f"OpenAR accounts.csv missing columns: {sorted(absent)}")
    # Consistent sampling independent of CSV row order.
    df = df.drop_duplicates("account_id").sample(n=min(n, len(df)), random_state=seed)
    def field(name: str) -> pd.Series:
        return df[name] if name in df else pd.Series("", index=df.index)
    bal = pd.to_numeric(field("placement_balance").str.replace(r"[$,]", "", regex=True), errors="coerce")
    age = (pd.to_datetime(field("placement_date"), errors="coerce") -
           pd.to_datetime(field("charge_off_date"), errors="coerce")).dt.days
    phone_status = field("phone_status").fillna("").str.upper()
    address_status = field("address_status").fillna("").str.upper()
    phone_fields = ["phone_cell", "phone_home", "phone_work"]
    any_phone = pd.Series(False, index=df.index)
    for col in phone_fields:
        any_phone |= field(col).fillna("").str.strip().ne("")
    status = field("account_status").fillna("").str.upper()
    # Conservative demo-only gating. Not a replacement for legal review.
    hold = (_yes(field("cease_desist_flag")) | _yes(field("dispute_flag")) |
            _yes(field("attorney_represented_flag")) |
            status.isin(["DISPUTED", "LEGAL", "MILITARY_SCRA", "HARDSHIP_REVIEW",
                           "CLIENT_HOLD", "PENDING_CLIENT_REVIEW", "BANKRUPTCY",
                           "DECEASED", "STATUTE_EXPIRED", "RETURNED", "RECALLED",
                           "PAID_IN_FULL", "SETTLED_IN_FULL", "UNCOLLECTIBLE"]))
    ret = pd.DataFrame({
        "account_id": field("account_id").astype(str),
        "balance": bal.clip(lower=0),
        "debt_age_days": age.clip(lower=0).fillna(365),
        "phone": phone_status.isin(["VERIFIED", "UNVERIFIED"]) & any_phone,
        "email": field("email").fillna("").str.contains("@", regex=False),
        "address": address_status.isin(["VERIFIED", "UNVERIFIED"]),
        "paid_creditor": field("client_last_payment_date").fillna("").str.strip().ne(""),
        "hold": hold,
        "do_not_call": _yes(field("do_not_call_flag")),
        "source": "OPENAR SYNTHETIC CSV",
    }).dropna(subset=["balance"])
    ret = ret[ret.balance.gt(0)].reset_index(drop=True)
    if ret.empty:
        raise ValueError("No positive-balance accounts remain after OpenAR validation")
    return _finish(ret), "OPENAR SYNTHETIC CSV (not real consumers)"


def source_counts(data_dir: str = "data/openar") -> dict[str, int]:
    path = Path(data_dir)
    result = {}
    for table in ("accounts", "notes", "payments", "payment_arrangements"):
        p = path / f"{table}.csv"
        if p.exists():
            # Chunked for memory safety; no row contents displayed.
            result[table] = sum(len(chunk) for chunk in pd.read_csv(p, chunksize=30000))
    return result
