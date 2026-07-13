"""Structural checks for the canonical static production frontend."""

from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
HTML_FILES = tuple(sorted(STATIC.glob("*.html")))


class FrontendParser(HTMLParser):
    """Collect structural accessibility and asset information from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.local_assets: list[str] = []
        self.images_without_alt: list[int] = []
        self.unsafe_blank_links: list[int] = []
        self.html_languages: list[str] = []
        self.h1_count = 0

    def handle_starttag(
        self, tag: str, attrs_list: list[tuple[str, str | None]]
    ) -> None:
        attrs = dict(attrs_list)
        element_id = attrs.get("id")
        if element_id is not None:
            self.ids.append(element_id)
        if tag == "html":
            self.html_languages.append(attrs.get("lang") or "")
        elif tag == "h1":
            self.h1_count += 1
        elif tag == "img" and "alt" not in attrs:
            self.images_without_alt.append(self.getpos()[0])
        elif tag == "a" and attrs.get("target") == "_blank":
            rel = set((attrs.get("rel") or "").split())
            if not {"noopener", "noreferrer"}.issubset(rel):
                self.unsafe_blank_links.append(self.getpos()[0])

        for attribute in ("href", "poster", "src"):
            if value := attrs.get(attribute):
                self._record_local_asset(value)

    def _record_local_asset(self, value: str) -> None:
        parsed = urlsplit(value)
        if (
            parsed.scheme
            or parsed.netloc
            or value.startswith(("#", "data:", "mailto:"))
        ):
            return
        path = parsed.path
        if path.startswith("/static/"):
            self.local_assets.append(path.removeprefix("/static/"))
        elif not path.startswith("/") and Path(path).suffix:
            self.local_assets.append(path)


@pytest.mark.parametrize("html_file", HTML_FILES, ids=lambda path: path.name)
def test_static_html_structure(html_file: Path) -> None:
    """Every deployed HTML page has valid identity and accessibility basics."""
    parser = FrontendParser()
    parser.feed(html_file.read_text(encoding="utf-8"))

    duplicate_ids = sorted(
        element_id for element_id, count in Counter(parser.ids).items() if count > 1
    )
    assert parser.html_languages == ["en"]
    assert parser.h1_count == 1
    assert not duplicate_ids
    assert not parser.images_without_alt
    assert not parser.unsafe_blank_links


@pytest.mark.parametrize("html_file", HTML_FILES, ids=lambda path: path.name)
def test_static_html_local_assets_exist(html_file: Path) -> None:
    """Every local asset referenced by a deployed page must be committed."""
    parser = FrontendParser()
    parser.feed(html_file.read_text(encoding="utf-8"))

    missing = sorted(
        asset for asset in set(parser.local_assets) if not (STATIC / asset).is_file()
    )
    assert not missing
