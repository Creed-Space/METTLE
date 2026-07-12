"""Regression checks for public claims that must match the implementation."""

from pathlib import Path

from mettle.challenge_adapter import SUITE_REGISTRY


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_public_homepages_match_registered_suite_count() -> None:
    """Both homepage sources must advertise the authoritative registry count."""
    assert len(SUITE_REGISTRY) == 12

    static_homepage = _read("static/index.html")
    svelte_homepage = _read("frontend/src/routes/+page.svelte")

    assert "12 Verification Suites" in static_homepage
    assert "12 Verification Suites" in svelte_homepage
    assert "Twelve suites. Seven questions." in static_homepage
    assert "Twelve suites. Seven questions." in svelte_homepage

    for stale_claim in ("11 verification suites", "Eleven suites.", "117 Tests"):
        assert stale_claim not in static_homepage
        assert stale_claim not in svelte_homepage


def test_frontend_cli_examples_use_supported_flags() -> None:
    """Public CLI examples must stay within the parser's supported surface."""
    homepage = _read("frontend/src/routes/+page.svelte")
    docs_page = _read("frontend/src/routes/docs/+page.svelte")

    assert "--difficulty" not in homepage
    assert "--difficulty" not in docs_page
    assert "five-challenge" in homepage
    assert "five-challenge" in docs_page
