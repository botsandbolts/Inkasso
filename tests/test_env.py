import numpy as np
import pytest
from inkasso.data import demo_profiles, load_profiles
from inkasso.env import CollectionEnv


def profile(**kw):
    p = demo_profiles(1).iloc[0].to_dict()
    p.update(balance=500., p12=.5, phone=True, email=True, address=True,
             hold=False, do_not_call=False)
    p.update(kw)
    return p


def test_restricted_profile_never_contacted():
    env = CollectionEnv(profile(hold=True))
    assert env.permitted() == [0]
    with pytest.raises(ValueError):
        env.step(3)
    assert env.step(0).payment == 0.


def test_channel_mask_and_plan_requires_contact():
    env = CollectionEnv(profile(phone=False, email=False, address=True, do_not_call=True))
    assert env.permitted() == [0, 2]
    with pytest.raises(ValueError):
        env.step(4)
    with pytest.raises(ValueError):
        env.step(3)


def test_episodes_are_reproducible_and_end():
    p = profile(p12=.9)
    def play():
        e = CollectionEnv(p, seed=181, horizon=4)
        results = []
        while not e.done:
            r = e.step(0)
            results.append((r.payment, r.reward, r.done))
        return results
    assert play() == play()
    assert play()[-1][2]


def test_fallback_clear_source_and_no_payment_leakage():
    profiles, source = load_profiles("directory-that-does-not-exist", n=80)
    assert len(profiles) == 80
    assert "SIMULATED FALLBACK" in source
    assert not {"total_paid", "last_payment_date", "status_class"}.intersection(profiles.columns)


def test_outreach_cap():
    e = CollectionEnv(profile(p12=.01), seed=4)
    for _ in range(4):
        if e.done:
            break
        e.step(1)
    assert e.permitted() == [0] or e.done


def test_openar_csv_loader_uses_placement_fields(tmp_path):
    import pandas as pd
    pd.DataFrame([{
        "account_id": "A-100", "placement_balance": "780.00",
        "placement_date": "2025-04-12", "charge_off_date": "2025-01-01",
        "phone_status": "VERIFIED", "phone_cell": "555-0199",
        "address_status": "VERIFIED", "email": "fake@example.test",
        "cease_desist_flag": "Y", "dispute_flag": "N",
        "attorney_represented_flag": "N", "do_not_call_flag": "N",
        "account_status": "ACTIVE", "client_last_payment_date": "2024-12-10",
        "total_paid": "1000000", "current_balance": "0",  # deliberate future leakage traps
    }]).to_csv(tmp_path / "accounts.csv", index=False)
    profiles, source = load_profiles(str(tmp_path), 1, 42)
    assert source.startswith("OPENAR")
    assert len(profiles) == 1
    assert profiles.iloc[0].balance == 780
    assert bool(profiles.iloc[0].hold)
    assert "current_balance" not in profiles.columns
    assert "total_paid" not in profiles.columns
