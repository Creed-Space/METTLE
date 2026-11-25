"""Export captions from measured per-scene ASR word timings.

The remaster's narration is assembled from one voice file per beat.  ASR
timestamps are local to each of those files, so this module only translates
them by the start recorded in ``narration-mapping.json``.  It deliberately
does not resample, stretch, invent, or split timings.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence


class CaptionError(ValueError):
    """A caption export cannot be produced from the supplied evidence."""


_LEXICAL_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[0-9]+")
_DISPLAY_RE = re.compile(
    r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?(?:[-‐‑‒–—][A-Za-z0-9]+)*(?:[.!?,;:]+)?"
)
_HYPHENS_RE = re.compile(r"[-‐‑‒–—]")
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}


def sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 fingerprint of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lexical_tokens(text: str) -> list[str]:
    """Tokenise speech for equality checks, ignoring punctuation."""

    # A hyphen is a word boundary for comparison.  This makes ``eight-digit``
    # and ``eight digit`` equivalent without weakening any other mismatch.
    return [
        part.lower().replace("’", "'")
        for part in _LEXICAL_RE.findall(_HYPHENS_RE.sub(" ", text))
    ]


def _canonical_tokens(text: str) -> list[str]:
    result: list[str] = []
    for token in _lexical_tokens(text):
        if token.isdigit():
            result.append(str(int(token)))
        else:
            result.append(_NUMBER_WORDS.get(token, token))
    return result


def normalized(text: str) -> list[str]:
    """Compatibility helper used by the earlier trailer caption exporter."""

    return _canonical_tokens(text)


def _display_tokens(text: str) -> list[str]:
    """Return script words with their authored punctuation retained."""

    return _DISPLAY_RE.findall(text)


def _units_for_display(token: str) -> list[str]:
    return _canonical_tokens(_HYPHENS_RE.sub(" ", token))


def _equivalent(expected: str, observed: str) -> bool:
    # METTLE is pronounced "metal" in the approved voice direction.  This is
    # the sole ASR proper-name exception and is recorded in the receipt.
    return expected == observed or (expected == "mettle" and observed == "metal")


def _alias_tokens(value: Any, label: str) -> list[str]:
    """Canonicalise one adjudicated alias side from text or token values."""

    if isinstance(value, str):
        tokens = _canonical_tokens(value)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        tokens = []
        for item in value:
            tokens.extend(_canonical_tokens(item))
    else:
        raise CaptionError(f"Adjudication {label} must be text or token list")
    if not tokens:
        raise CaptionError(f"Adjudication {label} must not be empty")
    return tokens


def _load_adjudication(
    path: str | os.PathLike[str] | None,
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    """Load reviewed ASR aliases and verify their independent review proof."""

    if path is None:
        return {}, None
    adjudication_path = Path(path)
    data = _load_json(adjudication_path, "caption adjudication")
    entries_value = data.get(
        "reviewed_segments", data.get("adjudications", data.get("entries"))
    )
    if entries_value is None:
        entries_value = data.get("segments")
    entries: list[dict[str, Any]] = []
    if isinstance(entries_value, list):
        entries = [entry for entry in entries_value if isinstance(entry, dict)]
        if len(entries) != len(entries_value):
            raise CaptionError("Caption adjudication segments must be objects")
    elif isinstance(entries_value, dict):
        for segment_id, entry in entries_value.items():
            if not isinstance(entry, dict):
                raise CaptionError("Caption adjudication segments must be objects")
            entries.append({"segment_id": segment_id, **entry})
    else:
        raise CaptionError(
            "Caption adjudication requires reviewed_segments or adjudications"
        )

    by_segment: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for entry in entries:
        segment_id = entry.get("segment_id", entry.get("id"))
        if not isinstance(segment_id, str) or not segment_id:
            raise CaptionError("Caption adjudication segment_id is required")
        if segment_id in seen:
            raise CaptionError(f"Duplicate caption adjudication {segment_id}")
        seen.add(segment_id)
        source_hash = entry.get("source_sha256", entry.get("source_hash"))
        asr_hash = entry.get("asr_sha256", entry.get("asr_hash"))
        if not isinstance(source_hash, str) or not source_hash:
            raise CaptionError(
                f"Adjudication source_sha256 is required for {segment_id}"
            )
        if not isinstance(asr_hash, str) or not asr_hash:
            raise CaptionError(f"Adjudication asr_sha256 is required for {segment_id}")
        review_value = entry.get("review")
        review = review_value if isinstance(review_value, dict) else {}
        review_path_value = entry.get(
            "review_path",
            entry.get(
                "proof_path",
                review_value if isinstance(review_value, str) else review.get("path"),
            ),
        )
        review_hash = entry.get(
            "review_sha256", entry.get("proof_sha256", review.get("sha256"))
        )
        if not isinstance(review_path_value, str) or not review_path_value:
            raise CaptionError(f"Adjudication review_path is required for {segment_id}")
        if not isinstance(review_hash, str) or not review_hash:
            raise CaptionError(
                f"Adjudication review_sha256 is required for {segment_id}"
            )
        review_path = Path(review_path_value)
        if not review_path.is_absolute():
            review_path = adjudication_path.parent / review_path
        if not review_path.is_file() or sha256(review_path) != review_hash:
            raise CaptionError(f"Adjudication review hash mismatch for {segment_id}")
        aliases_value = entry.get("aliases", entry.get("token_aliases", []))
        if not isinstance(aliases_value, list):
            raise CaptionError(f"Adjudication aliases must be a list for {segment_id}")
        aliases: list[dict[str, Any]] = []
        for alias in aliases_value:
            if not isinstance(alias, dict):
                raise CaptionError(
                    f"Adjudication alias must be an object for {segment_id}"
                )
            expected_value = alias.get(
                "expected_tokens",
                alias.get(
                    "expected_sequence",
                    alias.get("canonical_tokens", alias.get("expected")),
                ),
            )
            observed_value = alias.get(
                "observed_tokens",
                alias.get(
                    "observed_sequence", alias.get("asr_tokens", alias.get("observed"))
                ),
            )
            expected = _alias_tokens(
                expected_value, f"expected tokens for {segment_id}"
            )
            observed = _alias_tokens(
                observed_value, f"observed tokens for {segment_id}"
            )
            reason = alias.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise CaptionError(
                    f"Adjudication alias reason is required for {segment_id}"
                )
            aliases.append(
                {
                    "expected": expected,
                    "observed": observed,
                    "expected_tokens": expected_value,
                    "observed_tokens": observed_value,
                    "reason": reason,
                    "review_path": str(review_path),
                    "review_sha256": review_hash,
                }
            )
        by_segment[segment_id] = [
            {
                "segment_id": segment_id,
                "source_sha256": source_hash,
                "asr_sha256": asr_hash,
                "review_path": str(review_path),
                "review_sha256": review_hash,
                "aliases": aliases,
            }
        ]
    return by_segment, sha256(adjudication_path)


def _alias_pairs(
    expected_index: int,
    observed_index: int,
    expected_length: int,
    observed_length: int,
) -> list[tuple[int, int]]:
    """Map an alias's complete authored sequence to measured token indices."""

    pairs: list[tuple[int, int]] = []
    if expected_length >= observed_length:
        for index in range(expected_length):
            mapped = min(
                observed_length - 1, index * observed_length // expected_length
            )
            pairs.append((expected_index + index, observed_index + mapped))
    else:
        for index in range(observed_length):
            mapped = min(
                expected_length - 1, index * expected_length // observed_length
            )
            pairs.append((expected_index + mapped, observed_index + index))
    return pairs


def _align_tokens(
    segment_id: str,
    expected: Sequence[str],
    observed: Sequence[str],
    aliases: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[tuple[int, int]], list[dict[str, Any]], set[int]]:
    """Align expected and measured canonical units without unreviewed drift."""

    pairs: list[tuple[int, int]] = []
    events: list[dict[str, Any]] = []
    used_aliases: set[int] = set()
    expected_index = observed_index = 0
    while expected_index < len(expected) and observed_index < len(observed):
        matching: list[tuple[int, Mapping[str, Any]]] = []
        for alias_index, alias in enumerate(aliases):
            if list(
                expected[expected_index : expected_index + len(alias["expected"])]
            ) == list(alias["expected"]) and list(
                observed[observed_index : observed_index + len(alias["observed"])]
            ) == list(alias["observed"]):
                matching.append((alias_index, alias))
        if len(matching) > 1:
            raise CaptionError(f"Ambiguous adjudication aliases for {segment_id}")
        if matching:
            alias_index, alias = matching[0]
            pairs.extend(
                _alias_pairs(
                    expected_index,
                    observed_index,
                    len(alias["expected"]),
                    len(alias["observed"]),
                )
            )
            events.append({"kind": "adjudicated_alias", "alias": alias})
            used_aliases.add(alias_index)
            expected_index += len(alias["expected"])
            observed_index += len(alias["observed"])
            continue
        if _equivalent(expected[expected_index], observed[observed_index]):
            pairs.append((expected_index, observed_index))
            if expected[expected_index] != observed[observed_index]:
                events.append(
                    {
                        "kind": "mettle",
                        "expected": "METTLE",
                        "observed": "metal",
                        "token_index": expected_index,
                        "reason": "approved METTLE pronunciation variant",
                    }
                )
            expected_index += 1
            observed_index += 1
            continue
        raise _mismatch(segment_id, expected, observed)
    if expected_index != len(expected) or observed_index != len(observed):
        raise _mismatch(segment_id, expected, observed)
    return pairs, events, used_aliases


def _mismatch(
    segment_id: str,
    expected: Sequence[str],
    observed: Sequence[str],
) -> CaptionError:
    matcher = difflib.SequenceMatcher(
        a=list(expected), b=list(observed), autojunk=False
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        expected_text = " ".join(expected[i1:i2]) or "<missing>"
        observed_text = " ".join(observed[j1:j2]) or "<missing>"
        return CaptionError(
            f"Transcript mismatch for {segment_id}: expected {expected_text!r}, "
            f"observed {observed_text!r}"
        )
    return CaptionError(f"Transcript mismatch for {segment_id}")


def _compare_tokens(
    segment_id: str,
    expected_text: str,
    observed_text: str,
    aliases: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    return _compare_token_values(
        segment_id,
        _canonical_tokens(expected_text),
        _canonical_tokens(observed_text),
        aliases,
    )


def _compare_token_values(
    segment_id: str,
    expected: Sequence[str],
    observed: Sequence[str],
    aliases: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    _, events, _ = _align_tokens(segment_id, expected, observed, aliases)
    result: list[dict[str, Any]] = []
    for event in events:
        if event["kind"] == "mettle":
            result.append({key: value for key, value in event.items() if key != "kind"})
        else:
            alias = event["alias"]
            result.append(
                {
                    "kind": "adjudicated_alias",
                    "expected_tokens": alias["expected_tokens"],
                    "observed_tokens": alias["observed_tokens"],
                    "reason": alias["reason"],
                    "review_path": alias["review_path"],
                    "review_sha256": alias["review_sha256"],
                }
            )
    return result


def _display_for_observed(
    segment_id: str,
    expected_text: str,
    observed_words: Sequence[Mapping[str, Any]],
    aliases: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[str], list[dict[str, Any]], set[int]]:
    """Align authored display words to measured ASR words.

    Hyphenated compounds can be split or joined by ASR.  A joined measured
    word receives the complete authored compound; a split measurement keeps
    the display on its first measured word and leaves the second display
    value empty.  No timestamp is subdivided.
    """

    expected_display = _display_tokens(expected_text)
    expected_units: list[tuple[str, int]] = []
    for parent, token in enumerate(expected_display):
        expected_units.extend((unit, parent) for unit in _units_for_display(token))
    observed_units: list[tuple[str, int]] = []
    for parent, word in enumerate(observed_words):
        observed_units.extend(
            (unit, parent) for unit in _canonical_tokens(str(word["word"]))
        )
    pairs, events, used_aliases = _align_tokens(
        segment_id,
        [unit for unit, _ in expected_units],
        [unit for unit, _ in observed_units],
        aliases,
    )

    by_observed: dict[int, list[int]] = {}
    for expected_index, observed_index in pairs:
        expected_parent = expected_units[expected_index][1]
        observed_parent = observed_units[observed_index][1]
        by_observed.setdefault(observed_parent, []).append(expected_parent)
    display: list[str] = []
    emitted_parents: set[int] = set()
    for index in range(len(observed_words)):
        parents = list(dict.fromkeys(by_observed.get(index, [])))
        if not parents:
            raise CaptionError(
                f"Transcript mismatch for {segment_id}: unworded ASR token"
            )
        fresh_parents = [parent for parent in parents if parent not in emitted_parents]
        display.append(" ".join(expected_display[parent] for parent in fresh_parents))
        emitted_parents.update(fresh_parents)
    if _canonical_tokens(" ".join(display)) != _canonical_tokens(expected_text):
        raise CaptionError(f"Authored caption display drift for {segment_id}")
    ambiguities: list[dict[str, Any]] = []
    for event in events:
        if event["kind"] == "mettle":
            item = {key: value for key, value in event.items() if key != "kind"}
            observed_index = next(
                observed_index
                for expected_index, observed_index in pairs
                if expected_index == event["token_index"]
            )
            item["observed"] = str(
                observed_words[observed_units[observed_index][1]]["word"]
            )
            ambiguities.append(item)
        else:
            alias = event["alias"]
            ambiguities.append(
                {
                    "kind": "adjudicated_alias",
                    "expected_tokens": alias["expected_tokens"],
                    "observed_tokens": alias["observed_tokens"],
                    "reason": alias["reason"],
                    "review_path": alias["review_path"],
                    "review_sha256": alias["review_sha256"],
                }
            )
    return display, ambiguities, used_aliases


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CaptionError(f"{label} must be finite")
    return result


def _load_json(path: str | os.PathLike[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptionError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptionError(f"{label} must be an object")
    return value


def _resolve_source(source: str, mapping_path: Path) -> Path:
    candidate = Path(source)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        relative = mapping_path.parent / candidate
        if relative.is_file():
            return relative
    raise CaptionError(f"Narration source is missing: {source}")


def _validate_words(
    segment_id: str,
    words: Any,
    local_duration: float,
    final_duration: float,
    offset: float,
) -> list[dict[str, Any]]:
    if not isinstance(words, list) or not words:
        raise CaptionError(f"True ASR word timestamps required for {segment_id}")
    checked: list[dict[str, Any]] = []
    previous_start = -math.inf
    for index, raw in enumerate(words):
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("word"), str)
            or not raw["word"].strip()
        ):
            raise CaptionError(f"Invalid ASR word {index} for {segment_id}")
        start = _number(raw.get("start"), f"ASR start {segment_id}[{index}]")
        end = _number(raw.get("end"), f"ASR end {segment_id}[{index}]")
        if start < 0 or end < 0 or end < start:
            raise CaptionError(f"ASR timing out of range for {segment_id}[{index}]")
        if start > local_duration or end > local_duration:
            raise CaptionError(f"ASR timing out of range for {segment_id}[{index}]")
        if start < previous_start:
            raise CaptionError(f"Unordered ASR words for {segment_id}")
        previous_start = start
        mapped_start = start + offset
        mapped_end = end + offset
        if mapped_start < 0 or mapped_end > final_duration:
            raise CaptionError(
                f"Mapped ASR timing out of range for {segment_id}[{index}]"
            )
        checked.append(
            {
                "word": raw["word"],
                "raw": copy.deepcopy(raw),
                "raw_start": raw["start"],
                "raw_end": raw["end"],
                "start": mapped_start,
                "end": mapped_end,
            }
        )
    return checked


def _wrap(text: str) -> list[str]:
    if not text:
        return []
    lines = textwrap.wrap(
        text,
        width=44,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=True,
        drop_whitespace=True,
    )
    if not lines or len(lines) > 2 or any(len(line) > 44 for line in lines):
        raise CaptionError("Caption phrase exceeds two lines of 44 characters")
    return lines


def _phrase_cues(
    segment_id: str,
    words: Sequence[Mapping[str, Any]],
    display: Sequence[str],
) -> list[dict[str, Any]]:
    if len(words) != len(display):
        raise CaptionError(f"Caption phrase map does not cover {segment_id}")
    cues: list[dict[str, Any]] = []
    group: list[int] = []

    def visible(indices: Sequence[int]) -> str:
        return " ".join(display[index] for index in indices if display[index]).strip()

    def emit(indices: Sequence[int]) -> None:
        if not indices:
            return
        text = visible(indices)
        if not text:
            raise CaptionError(f"Caption phrase has no authored text for {segment_id}")
        lines = _wrap(text)
        start = float(words[indices[0]]["start"])
        end = float(words[indices[-1]]["end"])
        if end <= start:
            raise CaptionError(f"Caption phrase has no real duration for {segment_id}")
        cues.append(
            {
                "start": start,
                "end": end,
                "text": "\n".join(lines),
                "word_indices": list(indices),
            }
        )

    for index, word in enumerate(words):
        # Empty display values are the second half of a split compound. Keep
        # them attached to the preceding authored phrase where possible.
        if not group and not display[index] and cues:
            previous = cues[-1]
            previous["word_indices"].append(index)
            previous["end"] = float(word["end"])
            if previous["end"] <= previous["start"]:
                raise CaptionError(
                    f"Caption phrase has no real duration for {segment_id}"
                )
            continue
        candidate = group + [index]
        prior = words[group[-1]] if group else None
        candidate_text = visible(candidate)
        too_long = False
        try:
            _wrap(candidate_text)
        except CaptionError:
            too_long = True
        pause_break = bool(prior and float(word["start"]) - float(prior["end"]) > 0.65)
        punctuation_break = bool(
            prior and re.search(r"[.!?][\"')\]]*$", display[group[-1]])
        )
        group_has_duration = bool(
            group and float(words[group[-1]]["end"]) > float(words[group[0]]["start"])
        )
        word_has_duration = float(word["end"]) > float(word["start"])
        # Whisper can quantise a short word to a point. Keep its raw evidence,
        # but join it to a real phrase, never invent a word duration.
        if (
            group
            and group_has_duration
            and (word_has_duration or too_long)
            and (too_long or len(group) >= 7 or pause_break or punctuation_break)
        ):
            emit(group)
            group = [index]
        else:
            group = candidate
    emit(group)
    for index in range(1, len(cues)):
        if cues[index]["start"] < cues[index - 1]["end"]:
            raise CaptionError(f"Caption cues overlap for {segment_id}")
    return cues


def _stamp(seconds: float, comma: bool = False) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{whole:02}{',' if comma else '.'}{milliseconds:03}"


def _render(cues: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    vtt = "WEBVTT\n\n" + "".join(
        f"{_stamp(float(cue['start']))} --> {_stamp(float(cue['end']))}\n{cue['text']}\n\n"
        for cue in cues
    )
    vtt = vtt.rstrip() + "\n"
    srt = "".join(
        f"{index}\n{_stamp(float(cue['start']), True)} --> {_stamp(float(cue['end']), True)}\n{cue['text']}\n\n"
        for index, cue in enumerate(cues, 1)
    )
    return vtt, srt


def _write_outputs(out: Path, files: Mapping[str, bytes]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".captions-", dir=out))
    try:
        for name, content in files.items():
            (temporary / name).write_bytes(content)
        for name in files:
            os.replace(temporary / name, out / name)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def export(
    brief: str | os.PathLike[str],
    mapping: str | os.PathLike[str],
    asr_dir: str | os.PathLike[str],
    video: str | os.PathLike[str],
    audio: str | os.PathLike[str],
    out: str | os.PathLike[str],
    adjudication: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build VTT/SRT plus measured-word and comparison receipts.

    All validation happens before any final output is replaced.  The returned
    dictionary is the receipt written to ``caption-receipt.json``.
    """

    brief_path = Path(brief)
    mapping_path = Path(mapping)
    asr_root = Path(asr_dir)
    video_path = Path(video)
    audio_path = Path(audio)
    out_path = Path(out)
    brief_data = _load_json(brief_path, "brief")
    mapping_data = _load_json(mapping_path, "narration mapping")
    adjudications, adjudication_hash = _load_adjudication(adjudication)
    beats = brief_data.get("beats")
    segments = mapping_data.get("segments")
    if not isinstance(beats, list) or not isinstance(segments, list):
        raise CaptionError("Brief beats and narration mapping segments are required")
    total_seconds = _number(mapping_data.get("total_seconds"), "mapping total_seconds")
    if total_seconds <= 0:
        raise CaptionError("mapping total_seconds must be positive")
    beat_by_id: dict[str, Mapping[str, Any]] = {}
    for beat in beats:
        if (
            not isinstance(beat, dict)
            or not isinstance(beat.get("id"), str)
            or not isinstance(beat.get("text"), str)
        ):
            raise CaptionError("Each brief beat needs id and text")
        if beat["id"] in beat_by_id:
            raise CaptionError(f"Duplicate brief beat {beat['id']}")
        beat_by_id[beat["id"]] = beat
    if not audio_path.is_file() or not video_path.is_file():
        raise CaptionError("Final audio and video files are required")

    audio_hash = sha256(audio_path)
    video_hash = sha256(video_path)
    if (
        mapping_data.get("narration_sha256")
        and mapping_data["narration_sha256"] != audio_hash
    ):
        raise CaptionError("Final audio fingerprint does not match narration mapping")
    if mapping_data.get("media_sha256") and mapping_data["media_sha256"] != video_hash:
        raise CaptionError("Final video fingerprint does not match narration mapping")

    seen: set[str] = set()
    all_words: list[dict[str, Any]] = []
    raw_words: list[dict[str, Any]] = []
    all_cues: list[dict[str, Any]] = []
    comparison_segments: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    asr_hashes: dict[str, str] = {}
    segment_receipts: list[dict[str, Any]] = []
    raw_words_by_segment: dict[str, list[dict[str, Any]]] = {}
    adjudication_receipts: list[dict[str, Any]] = []

    unknown_adjudications = set(adjudications) - set(beat_by_id)
    if unknown_adjudications:
        unknown = sorted(unknown_adjudications)[0]
        raise CaptionError(f"Adjudication has no mapped segment {unknown}")

    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("id"), str):
            raise CaptionError("Each narration segment needs an id")
        segment_id = segment["id"]
        if segment_id in seen:
            raise CaptionError(f"Duplicate narration segment {segment_id}")
        seen.add(segment_id)
        beat = beat_by_id.get(segment_id)
        if beat is None:
            raise CaptionError(f"Narration segment {segment_id} has no brief beat")
        source_name = segment.get("source")
        if not isinstance(source_name, str) or not isinstance(
            segment.get("sha256"), str
        ):
            raise CaptionError(
                f"Narration source fingerprint required for {segment_id}"
            )
        source_path = _resolve_source(source_name, mapping_path)
        source_hash = sha256(source_path)
        if source_hash != segment["sha256"]:
            raise CaptionError(f"Narration source hash mismatch for {segment_id}")
        asr_path = asr_root / f"{segment_id}-asr.json"
        if not asr_path.is_file():
            raise CaptionError(f"ASR receipt is missing for {segment_id}")
        asr_data = _load_json(asr_path, f"ASR {segment_id}")
        asr_hashes[segment_id] = sha256(asr_path)
        if asr_data.get("source_sha256") != source_hash:
            raise CaptionError(f"ASR source hash mismatch for {segment_id}")
        segment_adjudications = adjudications.get(segment_id, [])
        aliases: list[dict[str, Any]] = []
        for proof in segment_adjudications:
            if proof["source_sha256"] != source_hash:
                raise CaptionError(
                    f"Adjudication source hash mismatch for {segment_id}"
                )
            if proof["asr_sha256"] != asr_hashes[segment_id]:
                raise CaptionError(f"Adjudication ASR hash mismatch for {segment_id}")
            aliases.extend(proof["aliases"])
            adjudication_receipts.append(
                {
                    "segment_id": segment_id,
                    "source_sha256": proof["source_sha256"],
                    "asr_sha256": proof["asr_sha256"],
                    "review_path": proof["review_path"],
                    "review_sha256": proof["review_sha256"],
                    "aliases": [
                        {
                            "expected_tokens": alias["expected_tokens"],
                            "observed_tokens": alias["observed_tokens"],
                            "reason": alias["reason"],
                        }
                        for alias in proof["aliases"]
                    ],
                }
            )
        offset = _number(segment.get("start"), f"segment {segment_id} start")
        if offset < 0:
            raise CaptionError(f"segment {segment_id} start must not be negative")
        scene_start = _number(
            segment.get("scene_start", 0), f"segment {segment_id} scene_start"
        )
        scene_end = _number(segment.get("scene_end"), f"segment {segment_id} scene_end")
        if (
            scene_end < scene_start
            or offset < scene_start
            or offset > scene_end
            or scene_end > total_seconds + 1e-9
        ):
            raise CaptionError(f"Invalid narration bounds for {segment_id}")
        local_duration = scene_end - scene_start
        words = _validate_words(
            segment_id, asr_data.get("words"), local_duration, total_seconds, offset
        )
        asr_text = asr_data.get("text")
        if not isinstance(asr_text, str):
            raise CaptionError(f"ASR transcript missing for {segment_id}")
        text_ambiguities = _compare_tokens(
            segment_id, str(beat["text"]), asr_text, aliases
        )
        _, _, text_aliases_used = _align_tokens(
            segment_id,
            _canonical_tokens(str(beat["text"])),
            _canonical_tokens(asr_text),
            aliases,
        )
        display, word_ambiguities, word_aliases_used = _display_for_observed(
            segment_id, str(beat["text"]), words, aliases
        )
        comparison_ambiguities = text_ambiguities + word_ambiguities
        for ambiguity in comparison_ambiguities:
            item = {"segment_id": segment_id, **ambiguity}
            if item not in ambiguities:
                ambiguities.append(item)
        measured_text = " ".join(str(word["word"]) for word in words)
        _compare_tokens(segment_id, str(beat["text"]), measured_text, aliases)
        if text_aliases_used != word_aliases_used or len(text_aliases_used) != len(
            aliases
        ):
            raise CaptionError(
                f"Adjudication alias was not observed consistently for {segment_id}"
            )
        cues = _phrase_cues(segment_id, words, display)
        for cue in cues:
            cue["segment_id"] = segment_id
        all_cues.extend(cues)
        for index, word in enumerate(words):
            mapped = {
                "segment_id": segment_id,
                "word": word["word"],
                "raw_start": word["raw_start"],
                "raw_end": word["raw_end"],
                "start": word["start"],
                "end": word["end"],
                "mapped_start": word["start"],
                "mapped_end": word["end"],
                "display": display[index],
                "raw_word": word["raw"],
                "raw": word["raw"],
            }
            all_words.append(mapped)
            raw_words.append({"segment_id": segment_id, **copy.deepcopy(word["raw"])})
        comparison_segments.append(
            {
                "id": segment_id,
                "status": "matched",
                "expected_text": beat["text"],
                "observed_text": asr_text,
                "expected_word_count": len(_canonical_tokens(str(beat["text"]))),
                "observed_word_count": len(_canonical_tokens(asr_text)),
            }
        )
        raw_words_by_segment[segment_id] = [
            copy.deepcopy(word["raw"]) for word in words
        ]
        segment_receipts.append(
            {
                "id": segment_id,
                "source_sha256": source_hash,
                "asr_sha256": asr_hashes[segment_id],
                "model": asr_data.get("model"),
                "start": offset,
                "raw_word_count": len(words),
                "raw_words": raw_words_by_segment[segment_id],
            }
        )

    if seen != set(beat_by_id):
        missing = sorted(set(beat_by_id) - seen)
        raise CaptionError(f"Narration mapping is missing beat {missing[0]}")
    for index, cue in enumerate(all_cues):
        if cue["end"] <= cue["start"]:
            raise CaptionError("Caption phrase has no real duration")
        if index and cue["start"] < all_cues[index - 1]["end"]:
            raise CaptionError("Caption cues overlap")
        if _stamp(float(cue["start"])) == _stamp(float(cue["end"])):
            raise CaptionError("Caption phrase rounds to zero duration")

    vtt, srt = _render(all_cues)
    words_data = {
        "version": 1,
        "audio_sha256": audio_hash,
        "video_sha256": video_hash,
        "total_seconds": total_seconds,
        "segments": segment_receipts,
        "raw_words": raw_words,
        "raw_words_by_segment": raw_words_by_segment,
        "words": all_words,
        "timing_policy": "Raw ASR timestamps are retained; mapped timestamps equal raw timestamps plus segment start, with no stretching or subdivision.",
    }
    receipt: dict[str, Any] = {
        "version": 1,
        "audio_sha256": audio_hash,
        "video_sha256": video_hash,
        "source_audio_sha256": audio_hash,
        "media_sha256": video_hash,
        "mapping_sha256": sha256(mapping_path),
        "asr_sha256": asr_hashes,
        "adjudication_sha256": adjudication_hash,
        "adjudications": adjudication_receipts,
        "hashes": {
            "audio_sha256": audio_hash,
            "video_sha256": video_hash,
            "asr_sha256": asr_hashes,
        },
        "comparison": {
            "status": "matched",
            "segments": comparison_segments,
        },
        "ambiguity_mappings": ambiguities,
        "ambiguities": ambiguities,
        "cues": len(all_cues),
        "words": len(all_words),
        "point_timestamps": sum(
            word["raw_start"] == word["raw_end"] for word in all_words
        ),
        "timing_policy": "Measured ASR word timings only; segment offsets are added without time stretching.",
    }
    words_bytes = (json.dumps(words_data, indent=2, ensure_ascii=False) + "\n").encode()
    receipt["caption_sha256"] = {
        "captions.vtt": hashlib.sha256(vtt.encode()).hexdigest(),
        "captions.srt": hashlib.sha256(srt.encode()).hexdigest(),
        "words.json": hashlib.sha256(words_bytes).hexdigest(),
    }
    receipt_bytes = (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode()
    _write_outputs(
        out_path,
        {
            "captions.vtt": vtt.encode(),
            "captions.srt": srt.encode(),
            "words.json": words_bytes,
            "caption-receipt.json": receipt_bytes,
        },
    )
    return receipt


def build(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for callers that used the prior exporter naming convention."""

    return export(*args, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--asr-dir", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--adjudication",
        type=Path,
        help="Reviewed ASR divergence JSON with source/ASR/review hashes",
    )
    args = parser.parse_args(argv)
    try:
        receipt = export(
            args.brief,
            args.mapping,
            args.asr_dir,
            args.video,
            args.audio,
            args.out,
            args.adjudication,
        )
    except CaptionError as exc:
        parser.error(str(exc))
    print(
        f"Exported {receipt['cues']} cues and {receipt['words']} measured words to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
