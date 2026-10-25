"""Presentation smoke checks: every management screen should render without training."""
from pathlib import Path
import ast


def test_streamlit_source_compiles():
    ast.parse(Path("app.py").read_text(encoding="utf-8"))


def test_business_story_has_six_linked_screens():
    source = Path("app.py").read_text(encoding="utf-8")
    for phrase in [
        "Start here", "Understand the portfolio", "Try a collection decision",
        "How the AI learns", "Compare strategies", "Trust, data & definitions",
        "The portfolio screen tells us", "Customer", "Simulated", "CQL",
    ]:
        # Case insensitive to avoid coupling to headline capitalization.
        assert phrase.lower() in source.lower(), phrase


def test_streamlit_pages_render_without_experiment_results():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=40).run()
    assert not at.exception, str(at.exception)

    # The portfolio, learning and transparency screens should not need a
    # trained model or data downloads to explain the experience.
    for label in [
        "2 · Understand the portfolio", "3 · Try a collection decision",
        "4 · How the AI learns", "5 · Compare strategies",
        "6 · Trust, data & definitions",
    ]:
        at.sidebar.radio[0].set_value(label).run()
        assert not at.exception, f"{label}: {at.exception}"
