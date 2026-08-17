#!/usr/bin/env python3
"""Vendor and subset the pinned Font Awesome Free browser assets."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil

# The only child is the current interpreter's fixed fontTools module.
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "node_modules" / "@fortawesome" / "fontawesome-free"
DESTINATION = ROOT / "static" / "vendor" / "fontawesome"
FONT_FILES = {
    "brands": "fa-brands-400",
    "solid": "fa-solid-900",
}
PASSTHROUGH_FONTS = ("fa-regular-400.woff2", "fa-v4compatibility.woff2")
# Font Awesome ships the subsettable outlines as TrueType up to 6.x and as
# WOFF2 only from 7.x. TrueType stays first so a release carrying both keeps
# producing the bytes already committed here rather than moving every
# fingerprint for no reason.
FONT_SOURCE_SUFFIXES = (".ttf", ".woff2")
ICON_CLASS = re.compile(r"fa-(solid|brands)\s+fa-([a-z0-9-]+)")
# One CSS rule: everything up to the opening brace, then the declarations.
CSS_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
# 6.x declares the glyph as `content:"\f09b"`; 7.x as the `--fa` custom
# property. Both spell the codepoint the same way, and `--fa--fa` (the duotone
# pair) is excluded by requiring the closing quote right after one codepoint.
CSS_CODEPOINT = re.compile(
    r'(?:^|;)\s*(?:content|--fa)\s*:\s*"\\([0-9a-f]+)"\s*(?:;|$)'
)
# A whole selector in the rule's comma-separated list: `.fa-github` in 7.x,
# `.fa-github:before` in 6.x. Anything compound (`.fa-stack .fa-github`) is a
# layout rule and does not match. A family or utility class such as `.fa-solid`
# does match this shape, but only rules carrying a glyph declaration are read
# at all, and Font Awesome never declares a glyph on one of those.
CSS_ICON_SELECTOR = re.compile(r"\.fa-([a-z0-9-]+?)(?::{1,2}before)?")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _used_icons() -> dict[str, set[str]]:
    used: dict[str, set[str]] = {family: set() for family in FONT_FILES}
    for page in sorted((ROOT / "static").glob("*.html")):
        for family, name in ICON_CLASS.findall(page.read_text(encoding="utf-8")):
            used[family].add(name)
    if not all(used.values()):
        raise RuntimeError("expected at least one solid and one brands icon")
    return used


def _glyph_table(css: str) -> dict[str, int]:
    """Map every icon name the stylesheet defines to its codepoint.

    Building the whole table once keeps the two stylesheet dialects in one
    place, and it resolves aliases (`fa-shield-alt` beside `fa-shield-halved`)
    without a second pass, because a rule names every alias it defines.
    """
    table: dict[str, int] = {}
    for selectors, declarations in CSS_RULE.findall(css):
        codepoint = CSS_CODEPOINT.search(declarations)
        if codepoint is None:
            continue
        value = int(codepoint.group(1), 16)
        for selector in selectors.split(","):
            name = CSS_ICON_SELECTOR.fullmatch(selector.strip())
            if name is not None:
                table[name.group(1)] = value
    return table


def _codepoints(css: str, icons: dict[str, set[str]]) -> dict[str, set[int]]:
    table = _glyph_table(css)
    result: dict[str, set[int]] = {family: set() for family in FONT_FILES}
    for family, names in icons.items():
        for name in sorted(names):
            if name not in table:
                raise RuntimeError(f"cannot map Font Awesome icon fa-{name}")
            result[family].add(table[name])
    return result


def _font_source(basename: str) -> Path:
    for suffix in FONT_SOURCE_SUFFIXES:
        candidate = PACKAGE / "webfonts" / f"{basename}{suffix}"
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"Font Awesome package has no outline font for {basename}")


def _subset(source: Path, destination: Path, codepoints: set[int]) -> None:
    unicodes = ",".join(f"U+{value:X}" for value in sorted(codepoints))
    # Every argument is locally derived and subprocess.run keeps shell=False.
    subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "fontTools.subset",
            str(source),
            f"--unicodes={unicodes}",
            "--flavor=woff2",
            "--layout-features=*",
            "--glyph-names",
            "--symbol-cmap",
            "--legacy-cmap",
            "--notdef-glyph",
            "--notdef-outline",
            "--recommended-glyphs",
            "--name-IDs=*",
            "--name-legacy",
            "--name-languages=*",
            f"--output-file={destination}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _build(destination: Path) -> None:
    css_source = PACKAGE / "css" / "all.min.css"
    if not css_source.is_file():
        raise RuntimeError(
            "Font Awesome package is missing; run npm ci before vendoring"
        )
    css = css_source.read_text(encoding="utf-8")
    codepoints = _codepoints(css, _used_icons())
    webfonts = destination / "webfonts"
    css_dir = destination / "css"
    webfonts.mkdir(parents=True)
    css_dir.mkdir(parents=True)

    for family, basename in FONT_FILES.items():
        _subset(
            _font_source(basename),
            webfonts / f"{basename}.woff2",
            codepoints[family],
        )
    for filename in PASSTHROUGH_FONTS:
        shutil.copyfile(PACKAGE / "webfonts" / filename, webfonts / filename)
    shutil.copyfile(PACKAGE / "LICENSE.txt", destination / "LICENSE.txt")

    for font in sorted(webfonts.glob("*.woff2")):
        old = f"../webfonts/{font.name}"
        new = f"/static/vendor/fontawesome/webfonts/{font.name}?v={_digest(font)}"
        if old not in css:
            raise RuntimeError(f"Font Awesome CSS does not reference {font.name}")
        css = css.replace(old, new)
    (css_dir / "all.min.css").write_text(css, encoding="utf-8")


def update(*, check: bool) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="mettle-fontawesome-") as temp:
        expected_root = Path(temp) / "fontawesome"
        _build(expected_root)
        expected_files = {
            path.relative_to(expected_root): path
            for path in expected_root.rglob("*")
            if path.is_file()
        }
        actual_files = (
            {
                path.relative_to(DESTINATION): path
                for path in DESTINATION.rglob("*")
                if path.is_file()
            }
            if DESTINATION.is_dir()
            else {}
        )
        stale = sorted(
            relative
            for relative in expected_files.keys() | actual_files.keys()
            if relative not in expected_files
            or relative not in actual_files
            or expected_files[relative].read_bytes()
            != actual_files[relative].read_bytes()
        )
        if stale and not check:
            if DESTINATION.exists():
                # All files are owned by this deterministic vendor directory.
                shutil.rmtree(DESTINATION)
            shutil.copytree(expected_root, DESTINATION)
        return [DESTINATION / relative for relative in stale]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = update(check=args.check)
    if stale and args.check:
        for path in stale:
            print(f"stale vendored asset: {path.relative_to(ROOT)}")
        return 1
    for path in stale:
        print(f"updated: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
