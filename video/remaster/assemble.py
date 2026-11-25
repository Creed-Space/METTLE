"""Build the local animatic or final edit from explicit, reusable media.

Run with the existing remaster Python runtime containing OpenTimelineIO.
Never publishes or replaces existing website media.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import shutil
import inspect
from pathlib import Path
import opentimelineio as ot
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "video/build/remaster"
ASSETS = WORK / "assets"
FPS = 30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)


def duration(path):
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        )
    )


def tr(seconds):
    return ot.opentime.TimeRange(
        ot.opentime.RationalTime(0, FPS),
        ot.opentime.RationalTime(round(seconds * FPS), FPS),
    )


def scratch_sources():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "legacy_video", ROOT / "video/build.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load video/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {s["id"]: module.tts_path(s) for s in module.SCENES}


def inserted_shots(beat, seconds):
    """Finite authored inserts. Source timing is unchanged, transitions dissolve."""
    ident = beat["id"]
    choices = {
        "01-hook": [(0.4, ASSETS / "instrument-motion.mp4", 8, "What can we measure?")],
        "03-challenges": [
            (
                2.7,
                WORK / "manim/videos/math_scene/1080p30/FreshChallenge.mp4",
                8.6,
                None,
            )
        ],
        "04-suites": [(4.0, WORK / "graphics/constellation.mp4", 8, None)],
        "07-controls": [(4.0, WORK / "graphics/packet.mp4", 8, None)],
        "08-credentials": [
            (1.0, ASSETS / "ledger-tablet-motion.mp4", 8, "A portable result")
        ],
        "09-uses": [
            (
                0.4,
                ASSETS / "observatory-stable-motion.mp4",
                8,
                "One signal in a\nwider assessment",
            )
        ],
        "10-close": [
            (0.2, WORK / "graphics/hyperframes-endcard.mp4", 1.4, None),
            (1.3, ASSETS / "instrument-motion.mp4", 8, "Measure your mettle."),
        ],
    }
    return choices.get(ident, [])


def render_composite(beat, rendered, clip, seconds, directory):
    shots = inserted_shots(beat, seconds)
    cache = clip.with_suffix(".render.json")
    inputs = {
        "rendered_sha256": sha(rendered),
        "seconds": seconds,
        "algorithm_sha256": hashlib.sha256(
            inspect.getsource(render_composite).encode()
        ).hexdigest(),
        "shots": [
            (start, str(source), sha(source), length, title)
            for start, source, length, title in shots
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode()
    ).hexdigest()
    if cache.is_file() and clip.is_file():
        prior = json.loads(cache.read_text())
        if prior["request_sha256"] == fingerprint and prior["sha256"] == sha(clip):
            print("Reusing verified composite", beat["id"], flush=True)
            return prior["inserts"]
    args = ["ffmpeg", "-y", "-v", "error", "-i", rendered]
    filters = ["[0:v]scale=1920:1080,setsar=1,fps=30,format=yuv420p[base]"]
    previous = "base"
    receipt = []
    input_index = 1
    for index, (start, source, length, title) in enumerate(shots, 1):
        if not source.is_file():
            raise RuntimeError("Missing accepted insert " + str(source))
        if duration(source) + 1 / FPS < length:
            raise RuntimeError("Insert shorter than edit " + str(source))
        args += ["-i", source]
        treatment = f"[{input_index}:v]trim=duration={length},setpts=PTS-STARTPTS,scale=1920:1080,setsar=1,fps=30"
        input_index += 1
        if title:
            # This machine's FFmpeg has no drawtext/libass. Author transparent
            # typography only, then composite it with the untouched video.
            overlay = directory / (beat["id"] + f"-title-{index}.png")
            img = Image.new("RGBA", (1920, 1080))
            draw = ImageDraw.Draw(img)
            # A soft veil keeps the photograph continuous, with no hard panel edge.
            for x in range(1400):
                alpha = round(107 * max(0, 1 - max(0, x - 700) / 700) ** 1.5)
                draw.line((x, 0, x, 1080), fill=(7, 27, 27, alpha))
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", 66
            )
            draw.multiline_text(
                (110, 170), title, font=font, fill="#f3f0e5", spacing=18
            )
            img.save(overlay)
            args += ["-loop", "1", "-framerate", "30", "-i", overlay]
            filters.append(treatment + f"[shot{index}]")
            treatment = f"[shot{index}][{input_index}:v]overlay=shortest=1"
            input_index += 1
        # Preserve each shot's actual motion, with no invented frames or time stretch.
        fade_out = max(0, length - 0.4)
        treatment += f",format=yuva420p,fade=t=in:st=0:d=0.4:alpha=1,fade=t=out:st={fade_out}:d=0.4:alpha=1,setpts=PTS+{start}/TB[insert{index}]"
        filters.append(treatment)
        output = f"layer{index}"
        filters.append(
            f"[{previous}][insert{index}]overlay=eof_action=pass:enable='between(t,{start},{start + length})'[{output}]"
        )
        previous = output
        receipt.append(
            {
                "source": str(source),
                "sha256": sha(source),
                "start": start,
                "duration": length,
                "fade_seconds": 0.4,
                "title": title,
            }
        )
    filters.append(f"[{previous}]format=yuv420p[v]")
    args += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-t",
        str(seconds),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        clip,
    ]
    run(args)
    cache.write_text(
        json.dumps(
            {
                "request_sha256": fingerprint,
                "inputs": inputs,
                "sha256": sha(clip),
                "inserts": receipt,
            },
            indent=2,
        )
        + "\n"
    )
    return receipt


def build(animatic=False, repair_animatic=None):
    brief = json.loads((HERE / "brief.json").read_text())
    d = WORK / ("animatic-edit" if animatic else "final-edit")
    d.mkdir(parents=True, exist_ok=True)
    scratch = scratch_sources() if animatic else {}
    using_scratch = animatic and not all(
        (ASSETS / (b["id"] + ".wav")).is_file() for b in brief["beats"]
    )
    previous = None
    old_movie = WORK / "animatic-before-repair.mp4"
    if repair_animatic:
        if not animatic:
            raise RuntimeError("Repair selection applies only to the animatic")
        previous = json.loads((WORK / "animatic-mapping.json").read_text())
        if sha(WORK / "animatic.mp4") != previous["media_sha256"]:
            raise RuntimeError("Prior animatic hash changed")
        shutil.copy2(WORK / "animatic.mp4", old_movie)
        old_timeline = ot.adapters.read_from_file(str(WORK / "animatic.otio"))
        old_pictures = {c.name: c.metadata["sha256"] for c in old_timeline.tracks[0]}
        previous_sources = {x["id"]: x for x in previous["segments"]}
    segments = []
    audio_paths = []
    time = 0.0
    sources = []
    for beat in brief["beats"]:
        source = (
            scratch[beat["id"]] if using_scratch else ASSETS / (beat["id"] + ".wav")
        )
        if not source.is_file():
            raise RuntimeError(f"Missing narration: {source}")
        sec = round((duration(source) + 1.3) * FPS) / FPS
        if previous:
            original = previous_sources[beat["id"]]
            if sha(source) != original["sha256"]:
                raise RuntimeError("Repair cannot change narration placement")
            sec = round((original["scene_end"] - original["scene_start"]) * FPS) / FPS
        audio = d / (beat["id"] + ".wav")
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                source,
                "-af",
                "adelay=450:all=1,apad",
                "-t",
                str(sec),
                "-ar",
                "48000",
                "-ac",
                "1",
                audio,
            ]
        )
        audio_paths.append(audio)
        sources.append(
            {
                "id": beat["id"],
                "source": str(source),
                "sha256": sha(source),
                "start": time + 0.45,
                "scene_start": time,
                "scene_end": time + sec,
                "mapping": "No trim or speed change; 450ms lead and silence pad to frame boundary.",
            }
        )
        time += sec
        stills = WORK / "graphics/stills"
        first = stills / (beat["id"] + "-a.png")
        second = stills / (beat["id"] + "-b.png")
        for p in [first, second]:
            if not p.exists():
                raise RuntimeError("Missing reviewed layout " + str(p))
        # The animatic uses held layouts, with a clean dissolve across the midpoint.
        # Final uses the rendered full-duration composition and validated shot inserts.
        clip = d / (beat["id"] + ".mp4")
        if animatic and repair_animatic and beat["id"] not in repair_animatic:
            if not clip.is_file() or sha(clip) != old_pictures[beat["id"]]:
                raise RuntimeError("Unchanged reviewed animatic segment hash mismatch")
        elif animatic:
            overlap = 0.45
            half = sec / 2
            through_teal = beat["id"] in ["05-questions", "10-close"]
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-loop",
                    "1",
                    "-t",
                    str(half if through_teal else half + overlap),
                    "-i",
                    first,
                    "-loop",
                    "1",
                    "-t",
                    str(half),
                    "-i",
                    second,
                    "-filter_complex",
                    (
                        f"[0:v]scale=1280:720,format=yuv420p,fade=t=out:st={half - 0.2}:d=0.2:color=0x071b1b,setpts=PTS-STARTPTS[a];[1:v]scale=1280:720,format=yuv420p,fade=t=in:st=0:d=0.2:color=0x071b1b,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[v]"
                        if through_teal
                        else f"[0:v]scale=1280:720,format=yuv420p,setpts=PTS-STARTPTS[a];[1:v]scale=1280:720,format=yuv420p,setpts=PTS-STARTPTS[b];[a][b]xfade=transition=fade:duration={overlap}:offset={half}[v]"
                    ),
                    "-map",
                    "[v]",
                    "-t",
                    str(sec),
                    "-r",
                    "30",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "21",
                    clip,
                ]
            )
        else:
            rendered = WORK / "graphics/final" / (beat["id"] + ".mp4")
            if not rendered.is_file():
                raise RuntimeError("Missing rendered final " + str(rendered))
            shot_receipt = render_composite(beat, rendered, clip, sec, d)
        if animatic:
            shot_receipt = []
        segments.append(
            {
                "id": beat["id"],
                "source": str(clip),
                "duration": sec,
                "narration": str(audio),
                "inserts": shot_receipt,
            }
        )
    (d / "audio.ffconcat").write_text(
        "ffconcat version 1.0\n" + "".join(f"file '{p}'\n" for p in audio_paths)
    )
    narration = WORK / (
        "animatic-narration.wav" if animatic else "narration-timeline.wav"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            d / "audio.ffconcat",
            "-c:a",
            "pcm_s16le",
            narration,
        ]
    )
    (d / "picture.ffconcat").write_text(
        "ffconcat version 1.0\n" + "".join(f"file '{s['source']}'\n" for s in segments)
    )
    out = WORK / ("animatic.mp4" if animatic else "mettle-remastered.mp4")
    args = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        d / "picture.ffconcat",
        "-i",
        narration,
    ]
    if repair_animatic:
        args = args[:-2] + ["-i", old_movie, "-map", "0:v", "-map", "1:a"]
    elif not animatic and (ASSETS / "score.mp3").is_file():
        args += [
            "-i",
            ASSETS / "score.mp3",
            "-filter_complex",
            "[1:a]loudnorm=I=-16:TP=-1.5:LRA=7,asplit=2[voice][side];[2:a]volume=0.28,apad[bed];[bed][side]sidechaincompress=threshold=0.07:ratio=4:attack=20:release=350[duck];[voice][duck]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.83:level=0[mix]",
            "-map",
            "0:v",
            "-map",
            "[mix]",
        ]
    else:
        args += ["-map", "0:v", "-map", "1:a", "-af", "loudnorm=I=-16:TP=-1.5:LRA=7"]
    args += [
        "-t",
        str(time),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        out,
    ]
    if repair_animatic:
        # The existing AAC is unchanged, already reviewed and exactly placed.
        args = args[:-1] + ["-c:a", "copy", out]
    run(args)
    mapping = {
        "mode": "existing narration scratch, NOT the rewritten final script"
        if using_scratch
        else "final generated narration",
        "total_seconds": time,
        "segments": sources,
        "narration_sha256": sha(narration),
        "media_sha256": sha(out),
    }
    (
        WORK / ("animatic-mapping.json" if animatic else "narration-mapping.json")
    ).write_text(json.dumps(mapping, indent=2) + "\n")
    timeline = ot.schema.Timeline(
        name="METTLE " + ("animatic" if animatic else "remaster")
    )
    picture = ot.schema.Track(name="Picture", kind=ot.schema.TrackKind.Video)
    for s in segments:
        r = tr(s["duration"])
        media = ot.schema.ExternalReference(
            target_url=str(Path(s["source"]).relative_to(WORK)), available_range=r
        )
        clip = ot.schema.Clip(name=s["id"], media_reference=media, source_range=r)
        clip.metadata["sha256"] = sha(s["source"])
        clip.metadata["inserts"] = s["inserts"]
        picture.append(clip)
    timeline.tracks.append(picture)
    sound = ot.schema.Track(name="Narration", kind=ot.schema.TrackKind.Audio)
    sound.append(
        ot.schema.Clip(
            name="Scratch narration" if using_scratch else "Final narration",
            media_reference=ot.schema.ExternalReference(
                target_url=narration.name, available_range=tr(time)
            ),
            source_range=tr(time),
        )
    )
    timeline.tracks.append(sound)
    if not animatic and (ASSETS / "score.mp3").is_file():
        music = ot.schema.Track(
            name="Music, sidechain ducked below narration",
            kind=ot.schema.TrackKind.Audio,
        )
        music.append(
            ot.schema.Clip(
                name="Original Lyria instrumental",
                media_reference=ot.schema.ExternalReference(
                    target_url="assets/score.mp3", available_range=tr(time)
                ),
                source_range=tr(time),
            )
        )
        music.metadata.update(
            volume=0.28,
            sidechain_ratio=4,
            sidechain_threshold=0.07,
            source_sha256=sha(ASSETS / "score.mp3"),
        )
        timeline.tracks.append(music)
    timeline.metadata.update(
        aspect_ratio="16:9",
        fps=FPS,
        local_only=True,
        media_sha256=sha(out),
        narration_sha256=sha(narration),
        draft=animatic,
    )
    otio = WORK / ("animatic.otio" if animatic else "METTLE.otio")
    ot.adapters.write_to_file(timeline, str(otio))
    readback = ot.adapters.read_from_file(str(otio))
    assert abs(readback.duration().to_seconds() - time) < 1 / FPS
    for clip in readback.find_clips():
        assert (WORK / clip.media_reference.target_url).is_file()
    print(out, f"{time:.2f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--animatic", action="store_true")
    ap.add_argument("--repair-animatic", nargs="+")
    args = ap.parse_args()
    build(args.animatic, args.repair_animatic)
