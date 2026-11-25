from __future__ import annotations

import hashlib
import json
import struct
import wave
from pathlib import Path

import pytest

from video.remaster.captions import CaptionError, export


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wav(path: Path, seconds: float = 2.0) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        stream.writeframes(struct.pack("<h", 0) * int(8_000 * seconds))


def _fixture(
    tmp_path: Path, text: str = "METTLE makes one bounded test."
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "voice.wav"
    audio = tmp_path / "narration.wav"
    video = tmp_path / "video.mp4"
    _wav(source)
    _wav(audio, 5)
    video.write_bytes(b"small synthetic media")
    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"beats": [{"id": "01", "text": text}]}))
    mapping = tmp_path / "narration-mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "total_seconds": 5,
                "segments": [
                    {
                        "id": "01",
                        "scene_start": 0,
                        "start": 1.25,
                        "scene_end": 3,
                        "source": str(source),
                        "sha256": _sha(source),
                    }
                ],
            }
        )
    )
    asr = tmp_path / "asr"
    asr.mkdir()
    words = [
        {"word": "metal", "start": 0.1, "end": 0.3, "confidence": 0.9},
        {"word": "makes", "start": 0.4, "end": 0.6, "confidence": 0.8},
        {"word": "1", "start": 0.7, "end": 0.9, "confidence": 0.8},
        {"word": "bounded", "start": 1.0, "end": 1.2, "confidence": 0.8},
        {"word": "test", "start": 1.3, "end": 1.6, "confidence": 0.8},
    ]
    (asr / "01-asr.json").write_text(
        json.dumps(
            {
                "text": "metal makes 1 bounded test",
                "words": words,
                "source_sha256": _sha(source),
                "model": "test-asr",
            }
        )
    )
    return {
        "brief": brief,
        "mapping": mapping,
        "asr": asr,
        "audio": audio,
        "video": video,
        "source": source,
    }


def test_export_preserves_raw_words_maps_offset_and_emits_readable_cues(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    out = tmp_path / "out"
    receipt = export(
        paths["brief"],
        paths["mapping"],
        paths["asr"],
        paths["video"],
        paths["audio"],
        out,
    )

    assert receipt["comparison"]["status"] == "matched"
    assert receipt["ambiguity_mappings"][0]["expected"] == "METTLE"
    assert {
        name
        for name in (
            "captions.vtt",
            "captions.srt",
            "words.json",
            "caption-receipt.json",
        )
        if (out / name).is_file()
    } == {
        "captions.vtt",
        "captions.srt",
        "words.json",
        "caption-receipt.json",
    }
    words = json.loads((out / "words.json").read_text())
    assert words["raw_words"][0]["start"] == 0.1
    assert words["words"][0]["raw_start"] == 0.1
    assert words["words"][0]["start"] == pytest.approx(1.35)
    assert words["words"][4]["end"] == pytest.approx(2.85)
    assert (out / "captions.vtt").read_text().count("METTLE") == 1
    assert (out / "captions.vtt").read_text().endswith("\n")
    assert not (out / "captions.vtt").read_text().endswith("\n\n")

    for cue in (out / "captions.vtt").read_text().split("\n\n")[1:]:
        lines = cue.splitlines()[1:]
        assert len(lines) <= 2
        assert all(len(line) <= 44 for line in lines)


def _reviewed_alias_fixture(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    paths = _fixture(tmp_path, text="Visit METTLE dot S H today.")
    asr_path = paths["asr"] / "01-asr.json"
    source_hash = _sha(paths["source"])
    asr_data = {
        "text": "visit metal sh today",
        "words": [
            {"word": "visit", "start": 0.1, "end": 0.3},
            {"word": "metal", "start": 0.4, "end": 0.6},
            {"word": "sh", "start": 0.7, "end": 0.9},
            {"word": "today", "start": 1.0, "end": 1.3},
        ],
        "source_sha256": source_hash,
        "model": "test-asr",
    }
    asr_path.write_text(json.dumps(asr_data))
    review = tmp_path / "audio-review.md"
    review.write_text(
        "Human audio review confirms dot S H is one measured ASR token.\n"
    )
    adjudication = tmp_path / "adjudication.json"
    adjudication.write_text(
        json.dumps(
            {
                "version": 1,
                "reviewed_segments": [
                    {
                        "segment_id": "01",
                        "source_sha256": source_hash,
                        "asr_sha256": _sha(asr_path),
                        "review_path": review.name,
                        "review_sha256": _sha(review),
                        "aliases": [
                            {
                                "expected_tokens": ["dot", "S", "H"],
                                "observed_tokens": ["sh"],
                                "reason": "Audio review confirms one ASR token for the spoken letters.",
                            }
                        ],
                    }
                ],
            }
        )
    )
    return paths, adjudication, review


def test_reviewed_alias_maps_complete_authored_text_without_splitting_timing(
    tmp_path: Path,
) -> None:
    paths, adjudication, _ = _reviewed_alias_fixture(tmp_path)
    out = tmp_path / "out"
    receipt = export(
        paths["brief"],
        paths["mapping"],
        paths["asr"],
        paths["video"],
        paths["audio"],
        out,
        adjudication,
    )
    assert receipt["adjudications"][0]["aliases"][0]["observed_tokens"] == ["sh"]
    assert any(
        item.get("kind") == "adjudicated_alias" for item in receipt["ambiguities"]
    )
    words = json.loads((out / "words.json").read_text())["words"]
    assert [word["word"] for word in words] == ["visit", "metal", "sh", "today"]
    assert words[2]["raw_start"] == 0.7
    assert words[2]["start"] == pytest.approx(1.95)
    assert "dot S H" in (out / "captions.vtt").read_text()


def test_reviewed_alias_requires_exact_source_asr_and_review_hashes(
    tmp_path: Path,
) -> None:
    paths, adjudication, _ = _reviewed_alias_fixture(tmp_path)
    data = json.loads(adjudication.read_text())
    data["reviewed_segments"][0]["source_sha256"] = "0" * 64
    adjudication.write_text(json.dumps(data))
    with pytest.raises(CaptionError, match="Adjudication source hash mismatch"):
        export(
            paths["brief"],
            paths["mapping"],
            paths["asr"],
            paths["video"],
            paths["audio"],
            tmp_path / "out",
            adjudication,
        )

    paths, adjudication, _ = _reviewed_alias_fixture(tmp_path / "review-mismatch")
    data = json.loads(adjudication.read_text())
    data["reviewed_segments"][0]["review_sha256"] = "f" * 64
    adjudication.write_text(json.dumps(data))
    with pytest.raises(CaptionError, match="Adjudication review hash mismatch"):
        export(
            paths["brief"],
            paths["mapping"],
            paths["asr"],
            paths["video"],
            paths["audio"],
            tmp_path / "out-review",
            adjudication,
        )


def test_long_transcript_splits_into_ordered_two_line_cues(tmp_path: Path) -> None:
    text = "One two three four five six seven eight nine ten eleven twelve."
    paths = _fixture(tmp_path, text=text)
    asr_path = paths["asr"] / "01-asr.json"
    data = json.loads(asr_path.read_text())
    tokens = text[:-1].lower().split()
    data["text"] = " ".join(tokens)
    data["words"] = [
        {"word": token, "start": 0.1 + index * 0.2, "end": 0.2 + index * 0.2}
        for index, token in enumerate(tokens)
    ]
    asr_path.write_text(json.dumps(data))
    out = tmp_path / "out"
    export(
        paths["brief"],
        paths["mapping"],
        paths["asr"],
        paths["video"],
        paths["audio"],
        out,
    )
    cues = [cue for cue in (out / "captions.vtt").read_text().split("\n\n")[1:] if cue]
    starts = [cue.splitlines()[0].split(" --> ")[0] for cue in cues]
    assert starts == sorted(starts)
    assert len(cues) >= 2
    for cue in cues:
        lines = cue.splitlines()[1:]
        assert 1 <= len(lines) <= 2
        assert all(len(line) <= 44 for line in lines)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.update(text="metal makes two bounded test"), "mismatch"),
        (
            lambda data: data["words"].__setitem__(
                0, {"word": "metal", "start": -0.1, "end": 0.1}
            ),
            "out of range",
        ),
        (
            lambda data: data["words"].__setitem__(
                1, {"word": "makes", "start": 0.05, "end": 0.6}
            ),
            "Unordered",
        ),
        (
            lambda data: (
                data["words"].__setitem__(
                    0, {"word": "metal", "start": 0.1, "end": 0.1}
                )
                or [word.update(start=0.1, end=0.1) for word in data["words"][1:]]
            ),
            "no real duration",
        ),
    ],
)
def test_invalid_evidence_fails_without_replacing_existing_outputs(
    tmp_path: Path, change, message: str
) -> None:
    paths = _fixture(tmp_path)
    asr_path = paths["asr"] / "01-asr.json"
    data = json.loads(asr_path.read_text())
    change(data)
    asr_path.write_text(json.dumps(data))
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "captions.vtt"
    sentinel.write_text("keep me")
    with pytest.raises(CaptionError, match=message):
        export(
            paths["brief"],
            paths["mapping"],
            paths["asr"],
            paths["video"],
            paths["audio"],
            out,
        )
    assert sentinel.read_text() == "keep me"


def test_mapping_and_asr_source_fingerprints_are_checked(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    mapping = json.loads(paths["mapping"].read_text())
    mapping["segments"][0]["sha256"] = "0" * 64
    paths["mapping"].write_text(json.dumps(mapping))
    with pytest.raises(CaptionError, match="Narration source hash mismatch"):
        export(
            paths["brief"],
            paths["mapping"],
            paths["asr"],
            paths["video"],
            paths["audio"],
            tmp_path / "out",
        )

    paths = _fixture(tmp_path / "asr-mismatch")
    asr_path = paths["asr"] / "01-asr.json"
    asr_data = json.loads(asr_path.read_text())
    asr_data["source_sha256"] = "f" * 64
    asr_path.write_text(json.dumps(asr_data))
    with pytest.raises(CaptionError, match="ASR source hash mismatch"):
        export(
            paths["brief"],
            paths["mapping"],
            paths["asr"],
            paths["video"],
            paths["audio"],
            tmp_path / "out-asr",
        )


def test_split_compounds_display_once_and_keep_all_measured_timestamps(
    tmp_path: Path,
) -> None:
    from video.remaster.captions import _display_for_observed, _phrase_cues

    words = [
        {"word": "Turing", "start": 0.1, "end": 0.4},
        {"word": "inverse", "start": 0.4, "end": 0.8},
        {"word": "eight", "start": 0.8, "end": 1.1},
        {"word": "digit", "start": 1.1, "end": 1.4},
        {"word": "tasks", "start": 1.4, "end": 1.7},
    ]
    display, _, _ = _display_for_observed(
        "01", "Turing-inverse eight-digit tasks.", words
    )
    assert display == ["Turing-inverse", "", "eight-digit", "", "tasks."]
    cues = _phrase_cues("01", words, display)
    assert cues[0]["text"] == "Turing-inverse eight-digit tasks."
    assert cues[0]["start"] == 0.1 and cues[0]["end"] == 1.7
    assert cues[0]["word_indices"] == [0, 1, 2, 3, 4]
    assert len(words) == 5


def test_quantised_point_word_joins_measured_phrase_without_new_timing() -> None:
    from video.remaster.captions import _phrase_cues

    words = [
        {"word": "people", "start": 0.1, "end": 0.5},
        {"word": "models", "start": 1.3, "end": 1.3},
        {"word": "and", "start": 2.0, "end": 2.2},
        {"word": "solvers", "start": 2.2, "end": 2.6},
    ]
    cues = _phrase_cues("03", words, ["people,", "models,", "and", "solvers."])
    assert cues[0] == {
        "start": 0.1,
        "end": 1.3,
        "text": "people, models,",
        "word_indices": [0, 1],
    }
    assert cues[1]["start"] == 2.0 and cues[1]["end"] == 2.6
    assert words[1]["start"] == words[1]["end"] == 1.3
