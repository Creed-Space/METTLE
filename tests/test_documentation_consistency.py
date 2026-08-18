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


def test_public_mcp_surface_excludes_the_reference_solver() -> None:
    server = _read("mettle/mcp_server.py")
    public_surfaces = (
        _read("README.md"),
        _read("smithery.yaml"),
        _read("skill/SKILL.md"),
        _read("_wiki/flows/integration-and-deployment.md"),
        _read("_wiki/systems/mcp-server-and-api.md"),
    )

    assert "from mettle.solver import solve_challenge" not in server
    assert "mettle_auto_verify" not in server
    assert "substrate verification" not in server.lower()
    for surface in public_surfaces:
        assert "mettle_auto_verify" not in surface
