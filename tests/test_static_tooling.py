"""Regression tests for static-site maintenance tooling and cache contracts."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_static_site
from scripts.update_footer_year import copyright_label, update_file


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2026, "&copy; 2026 Creed Space."),
        (2027, "&copy; 2026-27 Creed Space."),
        (2034, "&copy; 2026-34 Creed Space."),
    ],
)
def test_copyright_label(year: int, expected: str) -> None:
    assert copyright_label(year) == expected


def test_copyright_label_rejects_prelaunch_year() -> None:
    with pytest.raises(ValueError, match="2026 or later"):
        copyright_label(2025)


def test_footer_update_is_scoped_and_idempotent(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text("<footer>&copy; 2026 Creed Space.</footer>", encoding="utf-8")

    assert update_file(page, 2027, check=False) is True
    assert page.read_text(encoding="utf-8") == (
        "<footer>&copy; 2026-27 Creed Space.</footer>"
    )
    assert update_file(page, 2027, check=False) is False
    assert update_file(page, 2028, check=True) is True
    assert "2026-27" in page.read_text(encoding="utf-8")


def test_footer_update_rejects_ambiguous_page(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text("<footer>missing canonical footer</footer>", encoding="utf-8")
    with pytest.raises(ValueError, match="expected one"):
        update_file(page, 2027, check=False)


def test_checked_in_asset_fingerprints_are_current() -> None:
    subprocess.run(
        [sys.executable, "scripts/update_asset_fingerprints.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_static_site_contract() -> None:
    subprocess.run(
        [sys.executable, "scripts/check_static_site.py"],
        cwd=ROOT,
        check=True,
    )


def test_sitemap_checker_rejects_dtds_before_xml_parsing(
    tmp_path: Path, monkeypatch
) -> None:
    """A malicious pull request cannot make CI expand XML entities."""
    (tmp_path / "sitemap.xml").write_text(
        '<!DOCTYPE urlset [<!ENTITY x "expanded">]><urlset>&x;</urlset>',
        encoding="utf-8",
    )
    monkeypatch.setattr(check_static_site, "STATIC", tmp_path)
    errors: list[str] = []

    check_static_site._check_sitemap_and_robots(errors)

    assert errors == [
        "sitemap.xml: cannot parse: sitemap DTDs and entities are forbidden"
    ]


def test_public_guide_and_sitemap_use_canonical_committed_routes(client) -> None:
    guide = client.get("/guide")
    assert guide.status_code == 200
    assert 'rel="canonical" href="https://mettle.sh/guide"' in guide.text
    assert guide.headers["cache-control"] == "public, max-age=0, must-revalidate"

    legacy = client.get("/static/docs.html", follow_redirects=False)
    assert legacy.status_code == 308
    assert legacy.headers["location"] == "/guide"

    sitemap = client.get("/sitemap.xml")
    assert sitemap.status_code == 200
    assert sitemap.content == (ROOT / "static/sitemap.xml").read_bytes()


def test_static_cache_requires_matching_content_fingerprint(client) -> None:
    style = ROOT / "static/style.css"
    version = hashlib.sha256(style.read_bytes()).hexdigest()[:12]

    versioned = client.get(f"/static/style.css?v={version}")
    assert versioned.status_code == 200
    assert versioned.headers["cache-control"] == ("public, max-age=31536000, immutable")

    stale = client.get("/static/style.css?v=000000000000")
    assert stale.status_code == 200
    assert stale.headers["cache-control"] == "public, max-age=0, must-revalidate"


def test_dynamic_api_responses_are_not_cacheable(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
