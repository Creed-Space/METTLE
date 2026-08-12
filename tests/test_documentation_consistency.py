"""Regression checks for public claims that must match the implementation."""

import re
from pathlib import Path

from mettle.challenge_adapter import SUITE_REGISTRY
from mettle.vcp import TIER_RANGES

ROOT = Path(__file__).resolve().parents[1]

# The explainer video is a published surface (it is embedded in index.html), but
# it lived outside this net for its whole life, which is exactly how it came to
# claim "Bronze confirms AI substrate" on a page that says passing proves no such
# thing, and a 100ms budget for an 800ms challenge. It is covered now.
VIDEO_SURFACES = ("video/build.py", "static/mettle-explainer.vtt")


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _video_narration() -> str:
    """The narration the video build would speak, straight from the scene list."""
    import sys

    sys.path.insert(0, str(ROOT / "video"))
    import build  # type: ignore[import-not-found]

    return "\n".join(scene["narration"] for scene in build.SCENES)


def test_public_homepage_matches_registered_suite_count() -> None:
    """The canonical homepage must advertise the authoritative registry count."""
    assert len(SUITE_REGISTRY) == 12

    static_homepage = _read("static/index.html")

    assert "12 Experimental Suites" in static_homepage
    assert "Passing does not prove the named property" in static_homepage

    for stale_claim in ("11 verification suites", "Eleven suites.", "117 Tests"):
        assert stale_claim not in static_homepage


def test_static_cli_examples_use_supported_flags() -> None:
    """Public CLI examples must stay within the parser's supported surface."""
    homepage = _read("static/index.html")
    docs_page = _read("static/docs.html")

    assert "--difficulty" not in homepage
    assert "--difficulty" not in docs_page
    assert "--full" in docs_page
    assert "--auto" not in homepage
    assert "--auto" not in docs_page
    assert "--notarize" not in homepage
    assert "--notarize" not in docs_page


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


def test_mcp_surface_does_not_expose_an_automatic_solver() -> None:
    """Public claims and the credential boundary must agree on solver removal."""
    server = _read("mettle/mcp_server.py")
    public_surfaces = (
        _read("README.md"),
        _read("static/docs.html"),
        _read("smithery.yaml"),
        _read("_wiki/systems/mcp-server-and-api.md"),
    )

    assert "from mettle.solver import solve_challenge" not in server
    assert "mettle_auto_verify" not in server
    assert "substrate verification" not in server.lower()
    for surface in public_surfaces:
        assert "mettle_auto_verify" not in surface


# ---------------------------------------------------------------------------
# Explainer video: a published surface, held to the same standard as the pages.
# ---------------------------------------------------------------------------


def test_shipped_captions_match_the_current_narration_script() -> None:
    """The rendered VTT must match `video/build.py`.

    Editing the script without re-running the build ships a video whose audio and
    captions state the old claims. This is the check that makes a stale render a
    test failure rather than a silent regression.
    """
    caption_text = _read("static/mettle-explainer.vtt")
    spoken = " ".join(caption_text.split("\n"))

    for scene_narration in _video_narration().split("\n"):
        # Captions are re-flowed into cues, so compare on a distinctive fragment
        # of each scene rather than the whole paragraph.
        opening = " ".join(scene_narration.split()[:6])
        assert opening in " ".join(spoken.split()), (
            f"Caption track is stale: {opening!r} is in video/build.py but not in "
            f"the rendered VTT. Re-run `python3 video/build.py`."
        )


def test_video_states_no_absolute_certainty_claims() -> None:
    """The video must not out-claim the site it is embedded in."""
    for surface in VIDEO_SURFACES:
        text = _read(surface).lower()
        for banned in (
            "impossible to fake",
            "nothing ever repeats",
            "confirms ai substrate",
            "reveal the substrate",
            "only a machine mind can do",
            "rule out a human relaying",
            "proves beyond doubt",
            "cryptographically verifiable proof of machine agency",
        ):
            assert banned not in text, f"{surface} makes an absolute claim: {banned!r}"


def test_video_time_budget_claim_matches_the_generators() -> None:
    """No surface may advertise a time budget the challenges do not enforce."""
    from mettle.challenger import (
        generate_chained_reasoning_challenge,
        generate_speed_math_challenge,
    )
    from mettle.models import Difficulty

    limits = [
        generate_speed_math_challenge(Difficulty.FULL).time_limit_ms,
        generate_chained_reasoning_challenge(Difficulty.FULL).time_limit_ms,
    ]
    # Every enforced limit is sub-second but none is as tight as 100ms, so no
    # public surface may promise "<100ms" for these challenges.
    assert all(100 < limit < 1000 for limit in limits), limits

    for surface in (*VIDEO_SURFACES, "static/index.html", "static/docs.html"):
        text = _read(surface)
        assert not re.search(r"100\s*-?\s*ms", text, re.IGNORECASE), (
            f"{surface} promises a 100ms budget; the enforced limits are {limits}"
        )
        assert "hundred-millisecond" not in text.lower()


def test_acronym_expansion_is_consistent_across_surfaces() -> None:
    """METTLE must expand the same way everywhere.

    The repo shipped two expansions at once: the site, docs, API and video said
    "Machine Evaluation Through ...", while the README, package docstring and PWA
    manifest said "Machine Entity Trustbuilding through ...".
    """
    canonical = "Machine Evaluation Through Turing-inverse Logic Examination"
    surfaces = (
        "README.md",
        "main.py",
        "mettle/__init__.py",
        "mettle/router.py",
        "docs/SECURITY_WHITEPAPER.md",
        "docs/METTLE_VERIFICATION_SYSTEM.md",
        "static/index.html",
        "static/about.html",
        "static/docs.html",
        "static/site.webmanifest",
    )
    for surface in surfaces:
        text = _read(surface)
        assert canonical.lower() in text.lower(), f"{surface} lost the expansion"
        assert "machine entity trustbuilding" not in text.lower(), (
            f"{surface} uses the abandoned second expansion"
        )

    # The video's copy is wrapped across source lines, so check what it speaks.
    spoken = " ".join(_video_narration().split()).lower()
    assert canonical.lower() in spoken
    assert "machine entity trustbuilding" not in spoken


def test_video_credential_tiers_match_the_tier_registry() -> None:
    """Tier copy in the video must match `vcp.TIER_RANGES`, not invent semantics."""
    script = _read("video/build.py")

    for tier, (low, high) in TIER_RANGES.items():
        assert f"suites {low}&ndash;{high}" in script, (
            f"{tier} slide must state its real suite range {low}-{high}"
        )

    # Constitutional/governance binding is suite 11, so it belongs to platinum.
    # The video used to sell it as gold (suites 1-9).
    assert TIER_RANGES["gold"] == (1, 9)
    assert "Gold</b><span>suites 1&ndash;9" in script
    assert "constitutionally bound" not in script


def test_governance_and_operations_contracts_are_present() -> None:
    """The release must retain every named governance and response contract."""
    documents = (
        "SECURITY.md",
        "docs/ASSURANCE_CASE.md",
        "docs/COMPATIBILITY.md",
        "docs/CREDENTIAL_TRANSPARENCY.md",
        "docs/DEPRECATION_POLICY.md",
        "docs/ERROR_TAXONOMY.md",
        "docs/IDEMPOTENCY.md",
        "docs/INDEPENDENT_REVIEW_PLAN.md",
        "docs/PRIVACY_RETENTION.md",
        "docs/PROTOCOL_GOVERNANCE.md",
        "docs/RELEASE_CHECKLIST.md",
        "docs/REVIEW_DISPOSITIONS.md",
    )
    for document in documents:
        assert (ROOT / document).is_file(), f"missing governance contract: {document}"

    runbooks = {
        "ABUSIVE_TRAFFIC.md",
        "BACKUP_RESTORE_AND_KEY_LOSS.md",
        "DATABASE_LOSS.md",
        "DEPLOYMENT_ROLLBACK.md",
        "FALSE_DECISION_SPIKE.md",
        "PUBLIC_KEY_PUBLICATION.md",
        "REDIS_LOSS.md",
        "SIGNING_KEY_COMPROMISE.md",
    }
    runbook_root = ROOT / "docs/runbooks"
    assert runbooks.issubset({path.name for path in runbook_root.glob("*.md")})
    for filename in runbooks:
        content = (runbook_root / filename).read_text()
        assert "Owner" in content
        assert "## Trigger" in content or "## Scheduled restore drill" in content
        assert "evidence" in content.lower()


def test_independent_review_is_never_claimed_by_empty_scaffolding() -> None:
    """Planning and an empty ledger must stay visibly distinct from review."""
    plan = " ".join(_read("docs/INDEPENDENT_REVIEW_PLAN.md").split())
    dispositions = " ".join(_read("docs/REVIEW_DISPOSITIONS.md").split())
    assert "No independent review is claimed" in plan
    assert "pending gate" in dispositions
    assert "Not publishable as reviewed" in dispositions
