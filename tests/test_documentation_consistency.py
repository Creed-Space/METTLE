"""Regression checks for public claims that must match the implementation."""

from pathlib import Path

from mettle.challenge_adapter import SUITE_REGISTRY


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_public_homepage_matches_registered_suite_count() -> None:
    """The canonical homepage must advertise the authoritative registry count."""
    assert len(SUITE_REGISTRY) == 12

    static_homepage = _read("static/index.html")

    assert "12 Verification Suites" in static_homepage
    assert "Twelve suites. Seven questions." in static_homepage

    for stale_claim in ("11 verification suites", "Eleven suites.", "117 Tests"):
        assert stale_claim not in static_homepage


def test_static_cli_examples_use_supported_flags() -> None:
    """Public CLI examples must stay within the parser's supported surface."""
    homepage = _read("static/index.html")
    docs_page = _read("static/docs.html")

    assert "--difficulty" not in homepage
    assert "--difficulty" not in docs_page
    assert "5 challenges, strict timing" in docs_page


def test_static_directory_is_the_only_frontend_source() -> None:
    """A second frontend source would reintroduce publication drift."""
    assert not (ROOT / "frontend").exists()


def test_static_site_advertises_published_package() -> None:
    """The production site must use the current PyPI installation command."""
    static_homepage = _read("static/index.html")
    static_docs = _read("static/docs.html")

    assert "pip install mettle-verifier" in static_homepage
    assert "pip install mettle-verifier" in static_docs
    assert "PyPI: mettle-verifier, coming soon" not in static_homepage
    assert "PyPI: mettle-verifier, coming soon" not in static_docs
