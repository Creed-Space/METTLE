#!/usr/bin/env python3
"""Validate METTLE's static routes, assets, metadata, and fingerprints."""

from __future__ import annotations

import json
import sys

# Repository XML is bounded to 64 KiB and rejects DTDs before this parser runs.
import xml.etree.ElementTree as ET  # nosec B405
from html.parser import HTMLParser
from importlib import import_module
from urllib.parse import urlparse

_fingerprints = import_module(
    f"{__package__}.update_asset_fingerprints"
    if __package__
    else "update_asset_fingerprints"
)
REFERENCE = _fingerprints.REFERENCE
STATIC = _fingerprints.STATIC
asset_version = _fingerprints.asset_version
is_fingerprintable = _fingerprints.is_fingerprintable


ORIGIN = "https://mettle.sh"
PAGES = {
    "index.html": "/",
    "guide.html": "/guide",
    "docs.html": "/guide",
    "test.html": "/test",
    "about.html": "/about",
}
PUBLIC_PAGES = {"index.html", "docs.html", "test.html", "about.html"}
KNOWN_ROUTES = {
    "/",
    "/about",
    "/api",
    "/docs",
    "/guide",
    "/redoc",
    "/sitemap.xml",
    "/robots.txt",
    "/test",
}


class PageParser(HTMLParser):
    """Collect references, identifiers, metadata, and JSON-LD blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str]] = []
        self.ids: set[str] = set()
        self.canonical: list[str] = []
        self.og_url: list[str] = []
        self._json_ld = False
        self._json_parts: list[str] = []
        self.json_ld: list[object] = []

    def handle_starttag(
        self, tag: str, attrs_list: list[tuple[str, str | None]]
    ) -> None:
        attrs = dict(attrs_list)
        if identifier := attrs.get("id"):
            self.ids.add(identifier)
        for attribute in ("href", "src", "poster"):
            if value := attrs.get(attribute):
                self.references.append((attribute, value))
        if tag == "link" and attrs.get("rel") == "canonical" and attrs.get("href"):
            self.canonical.append(str(attrs["href"]))
        if tag == "meta" and attrs.get("property") == "og:url" and attrs.get("content"):
            self.og_url.append(str(attrs["content"]))
        if tag == "script" and attrs.get("type") == "application/ld+json":
            self._json_ld = True
            self._json_parts = []

    def handle_data(self, data: str) -> None:
        if self._json_ld:
            self._json_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._json_ld:
            self._json_ld = False
            self.json_ld.append(json.loads("".join(self._json_parts)))


def _local_path(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme or parsed.netloc:
        return None
    return parsed.path


def _check_page(name: str, errors: list[str]) -> PageParser:
    path = STATIC / name
    parser = PageParser()
    try:
        parser.feed(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{name}: cannot parse page: {exc}")
        return parser

    route = PAGES[name]
    expected_url = f"{ORIGIN}{route}" if route != "/" else f"{ORIGIN}/"
    if parser.canonical != [expected_url]:
        errors.append(f"{name}: canonical must be exactly {expected_url!r}")
    if parser.og_url != [expected_url]:
        errors.append(f"{name}: og:url must be exactly {expected_url!r}")

    for attribute, value in parser.references:
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https", "mailto"} or value.startswith("//"):
            continue
        local = _local_path(value)
        if not local:
            if parsed.fragment and parsed.fragment not in parser.ids:
                errors.append(f"{name}: missing fragment target #{parsed.fragment}")
            continue
        if local.startswith("/static/"):
            target = STATIC / local.removeprefix("/static/")
            if not target.is_file():
                errors.append(f"{name}: missing {attribute} target {value!r}")
            continue
        if local not in KNOWN_ROUTES:
            errors.append(f"{name}: undocumented route in {attribute}: {value!r}")
    return parser


def _check_fingerprints(errors: list[str]) -> None:
    for source in [*STATIC.rglob("*.css"), *STATIC.glob("*.html")]:
        text = source.read_text(encoding="utf-8")
        for match in REFERENCE.finditer(text):
            relative = match.group("path")
            if not is_fingerprintable(relative):
                continue
            target = STATIC / relative
            if not target.is_file():
                continue
            expected = asset_version(target)
            if match.group("version") != expected:
                errors.append(
                    f"{source.relative_to(STATIC)}: {relative} needs fingerprint v={expected}"
                )


def _check_manifest(errors: list[str]) -> None:
    try:
        manifest = json.loads((STATIC / "site.webmanifest").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"site.webmanifest: cannot parse: {exc}")
        return
    for icon in manifest.get("icons", []):
        src = str(icon.get("src", ""))
        local = _local_path(src)
        if not local or not local.startswith("/static/"):
            errors.append(f"site.webmanifest: invalid icon path {src!r}")
            continue
        if not (STATIC / local.removeprefix("/static/")).is_file():
            errors.append(f"site.webmanifest: missing icon {src!r}")


def _check_sitemap_and_robots(errors: list[str]) -> None:
    try:
        sitemap = (STATIC / "sitemap.xml").read_bytes()
        if len(sitemap) > 65536:
            raise ValueError("sitemap exceeds the 64 KiB static-site limit")
        if b"<!DOCTYPE" in sitemap or b"<!ENTITY" in sitemap:
            raise ValueError("sitemap DTDs and entities are forbidden")
        # The byte limit and DTD rejection above make entity expansion impossible.
        root = ET.fromstring(sitemap)  # nosec B314
        namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locations = {node.text for node in root.findall("s:url/s:loc", namespace)}
    except (OSError, ValueError, ET.ParseError) as exc:
        errors.append(f"sitemap.xml: cannot parse: {exc}")
        return
    expected = {
        f"{ORIGIN}/",
        *{f"{ORIGIN}{PAGES[p]}" for p in PUBLIC_PAGES - {"index.html"}},
    }
    if locations != expected:
        errors.append(
            f"sitemap.xml: locations differ, missing={expected - locations}, extra={locations - expected}"
        )
    robots = (STATIC / "robots.txt").read_text(encoding="utf-8")
    if f"Sitemap: {ORIGIN}/sitemap.xml" not in robots:
        errors.append("robots.txt: canonical sitemap directive is missing")


def main() -> int:
    errors: list[str] = []
    for name in sorted(PUBLIC_PAGES):
        _check_page(name, errors)
    _check_fingerprints(errors)
    _check_manifest(errors)
    _check_sitemap_and_robots(errors)
    if errors:
        print("Static site validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Static site validation passed: routes, metadata, assets, and fingerprints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
