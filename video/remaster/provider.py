"""Resumable, budget-checked media generation for the METTLE remaster.

Uses the OpenAI SDK against OpenRouter. Secrets are transient Keychain reads.
A submitted receipt without a completed asset is an explicit recovery hold.
"""

from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import time
import wave
import os
import threading
from typing import Any
from pathlib import Path
from openai import OpenAI, APIStatusError

# Eager imports avoid SDK lazy-loader deadlocks in independent threaded calls.
from openai.resources.images import Images  # noqa: F401
from openai.resources.chat.completions import Completions  # noqa: F401
import httpx

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "build" / "remaster"
ASSETS = WORK / "assets"
BRIEF = HERE / "brief.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(
        path.suffix + f".{os.getpid()}.{threading.get_ident()}.pending"
    )
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def secret(which="openrouter"):
    spec = json.loads(BRIEF.read_text())["keychain"][which]
    result = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-s",
            spec["service"],
            "-a",
            spec["account"],
            "-w",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(
            f"Keychain access unavailable for {spec['service']}/{spec['account']}"
        )
    return result.stdout.rstrip("\r\n")


def client():
    return OpenAI(
        api_key=secret(),
        base_url="https://openrouter.ai/api/v1",
        max_retries=0,
        timeout=600,
    )


def budget(c, reserve):
    data = c.get("/key", cast_to=dict[str, Any])["data"]
    before = json.loads((WORK / "provider-preflight.json").read_text())[
        "budget_before"
    ]["usage"]
    delta = data["usage"] - before
    remaining = data.get("limit_remaining")
    credits = c.get("/credits", cast_to=dict[str, Any])["data"]
    account_remaining = credits["total_credits"] - credits["total_usage"]
    costs = {}
    unresolved = []
    unresolved_reserve = 0.0
    for folder in [ASSETS, WORK / "review"]:
        for path in folder.rglob("*.json"):
            receipt = json.loads(path.read_text())
            if not isinstance(receipt, dict):
                continue
            usage = receipt.get("usage") or {}
            if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
                costs[str(path.relative_to(WORK))] = usage["cost"]
            elif receipt.get("status") == "failed" and receipt.get("id", "").startswith(
                "gen-vid-"
            ):
                unresolved.append(str(path.relative_to(WORK)))
                unresolved_reserve += receipt.get("duration", 8) * 0.25
    billing = WORK / "tts-billing.json"
    if billing.is_file():
        for name, receipt in json.loads(billing.read_text()).items():
            if isinstance(receipt.get("total_cost"), (int, float)):
                costs[name] = receipt["total_cost"]
            else:
                unresolved.append(name)
    confirmed = sum(costs.values())
    write(
        WORK / "spend-ledger.json",
        {
            "account_usage_before_usd": before,
            "account_usage_now_usd": data["usage"],
            "account_delta_usd": delta,
            "confirmed_task_spend_usd": confirmed,
            "task_costs_usd": costs,
            "unresolved_billing": unresolved,
            "unresolved_failed_video_reserve_usd": unresolved_reserve,
            "key_remaining_usd": remaining,
            "account_credit_remaining_usd": account_remaining,
            "user_ceiling_usd": 100,
            "effective_ceiling_usd": 40,
            "account_delta_caveat": "Concurrent use of this key contributes to account delta. Receipt costs isolate this task; in-flight calls reserve their maximum checked allocation.",
        },
    )
    if (
        confirmed + unresolved_reserve + reserve > 40
        or account_remaining < reserve
        or (remaining is not None and remaining < reserve)
    ):
        raise RuntimeError(
            "Effective production budget or available credit gate reached"
        )


def begin(name, body):
    if not json.loads(BRIEF.read_text())["authorization"].get(
        "paid_generation_authorized"
    ):
        raise RuntimeError(
            "Paid generation held pending provider-key rotation confirmation"
        )
    path = ASSETS / name
    receipt = path.with_suffix(path.suffix + ".json")
    fp = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        data = json.loads(receipt.read_text())
        if data["request_sha256"] != fp:
            raise RuntimeError(f"Request changed for {name}; use a new asset identity")
        if (
            data.get("state") == "completed"
            and path.exists()
            and data.get("sha256") == sha(path)
        ):
            print("reuse", name, flush=True)
            return path, receipt, None
        raise RuntimeError(
            f"Unresolved prior submission for {name}; reconcile before retry"
        )
    data = {
        "state": "submitting",
        "request_sha256": fp,
        "model": body.get("model", body.get("model_id")),
        "request_settings": {
            k: v
            for k, v in body.items()
            if k not in ["input_references", "frame_images", "input_audio"]
        },
    }
    return path, receipt, data


def complete(path, receipt, record, content, **metadata):
    path.write_bytes(content)
    record.update(state="completed", sha256=sha(path), bytes=len(content), **metadata)
    write(receipt, record)
    print("completed", path.name, len(content), "bytes", flush=True)
    return path


def ref(path):
    path = Path(path)
    mime = "image/webp" if path.suffix == ".webp" else "image/png"
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:"
            + mime
            + ";base64,"
            + base64.b64encode(path.read_bytes()).decode()
        },
    }


def voice(index):
    brief = json.loads(BRIEF.read_text())
    beat = brief["beats"][index]
    body = {
        "model": brief["speech_model"],
        "input": beat["text"],
        "voice": brief["voice"],
        "response_format": "pcm",
        "provider": {
            "options": {
                "google-ai-studio": {
                    "speech_metadata": {"style": brief["voice_direction"]}
                }
            }
        },
    }
    path, receipt, record = begin(beat["id"] + ".wav", body)
    if record is None:
        return path
    c = client()
    budget(c, 0.35)
    write(receipt, record)
    raw = c.audio.speech.with_raw_response.create(
        model=body["model"],
        input=body["input"],
        voice=body["voice"],
        response_format="pcm",
        extra_body={"provider": body["provider"]},
    )
    data = raw.http_response.content
    headers = raw.http_response.headers
    mime = headers.get("content-type", "")
    (ASSETS / (beat["id"] + ".pcm")).write_bytes(data)
    rate_match = re.search(r"rate=(\d+)", mime)
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        content = data
    elif "audio/pcm" in mime and rate_match is not None:
        rate = int(rate_match[1])
        chan_match = re.search(r"channels=(\d+)", mime)
        channels = int(chan_match[1]) if chan_match is not None else 1
        if len(data) % (2 * channels):
            raise RuntimeError("Incomplete signed 16-bit PCM sample")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(data)
        content = buf.getvalue()
    else:
        raise RuntimeError(f"Unexpected audio format: {mime}")
    return complete(
        path,
        receipt,
        record,
        content,
        generation_id=headers.get("x-generation-id"),
        content_type=mime,
        raw_sha256=hashlib.sha256(data).hexdigest(),
    )


COMMON = (
    "Use case: stylized-concept. A breathtaking 16:9 cinematic keyframe for a university-professor science explainer about METTLE. "
    "Editorial architectural and macro photography, physically coherent, midnight teal (#071b1b), clear smoked glass, black anodised metal, extremely fine warm brass edges and soft ivory light. "
    "A real-feeling optical measurement instrument, sculptural and exquisitely made. Creative visual metaphor for a bounded experiment, no invented product interface. "
    "Restrained museum-grade composition and materials, tactile, extraordinary light and depth, true optical glass, clean silhouettes. "
    "No text, letters, numbers, logos, watermarks, robot faces, brains, people, holographic UI, busy circuit patterns, neon cyan, lens flares or generic sci-fi embellishment. "
    "Leave generous negative space on the LEFT half and bottom for later typography and captions. Subject should sit mainly in the RIGHT half. "
)
PROMPTS = {
    "instrument": COMMON
    + "Close view of a single luminous ivory glass prism, resting inside a precision black circular optical cradle in a dark teal research gallery. Fine brass registration marks without numbers. A narrow vertical beam passes through the prism and emerges as a thin teal trace. Low oblique camera, foreground glass edge, very refined caustics, quiet volumetric light, deep background. Large subject right of centre, memorable simple geometry.",
    "observatory": COMMON
    + "Preserve the reference instrument and its prism exactly in a broader architectural view. The same dark teal research gallery, a row of twelve slim glass optical lenses arranged in a graceful arc, the original luminous prism on a central table at the right. Beautiful spatial depth with low stone platforms, brass edges, side-lit atmosphere. The twelve lenses are a visual metaphor for experimental perspectives, no individual screens or written information.",
    "ledger": COMMON
    + "Preserve the reference palette, materials and luminous prism motif. An elegant transparent glass specimen tablet sits in a finely machined black and brass cradle, three fine luminous traces embedded within the glass, a thin brass circular seal shape with NO letters. Macro product photograph from a gentle oblique angle, lit as a museum object. A traceable record is the visual metaphor. The tablet is abstract, no UI, no icons, no readable glyphs. Subtle beautiful reflection, luminous ivory edges.",
}


PROMPTS["ledger-tablet"] = (
    COMMON
    + "Use the reference ONLY for materials, palette and lighting. Create an entirely distinct subject: a thin horizontal rectangular transparent glass tablet, with three embedded luminous line traces and one small abstract circular brass seal. The tablet has flat parallel surfaces and four rounded corners, lying at a low oblique angle in a dark machined cradle. The pyramid in the reference must NOT appear. No triangular shapes. Subject on the right, left empty. A refined museum specimen record, entirely physical, no UI or text."
)


def image(name):
    brief = json.loads(BRIEF.read_text())
    refs = [] if name == "instrument" else [ref(ASSETS / "instrument-keyframe.png")]
    body = {
        "model": brief["image_model"],
        "prompt": PROMPTS[name],
        "aspect_ratio": "16:9",
        "quality": "high",
        "n": 1,
        "input_references": refs,
    }
    path, receipt, record = begin(name + "-keyframe.png", body)
    if record is None:
        return path
    c = client()
    budget(c, 1.2)
    write(receipt, record)
    raw = c.images.with_raw_response.generate(
        model=body["model"],
        prompt=body["prompt"],
        quality="high",
        n=1,
        extra_body={"aspect_ratio": "16:9", "input_references": refs},
    )
    result = raw.parse()
    data = result.model_dump()
    content = base64.b64decode(data["data"][0]["b64_json"])
    return complete(
        path,
        receipt,
        record,
        content,
        usage=data.get("usage"),
        generation_id=raw.http_response.headers.get("x-generation-id"),
        reference_sha256=[] if not refs else [sha(ASSETS / "instrument-keyframe.png")],
    )


def music_router(phase):
    brief = json.loads(BRIEF.read_text())
    prompt = (
        brief["music_prompt"]
        + " Instrumental only, absolutely no vocals or lyrics. D minor, approximately 72 bpm, a delicate recurring D-A-C-F four-note piano motif. Produce an approximately three-minute continuous cue. "
    )
    prompt += (
        "Opening movement: spare felt piano and glass harmonics, patient discovery, gradually introduce warm low strings, finish in an open sustained texture suitable for a crossfade."
        if phase == "opening"
        else "Concluding movement: continue the same restrained felt-piano and low-string palette with the recurring four-note motif, gentle development and an elegant quiet resolution at the end."
    )
    body = {
        "model": "google/lyria-3-pro-preview",
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"format": "mp3"},
        "stream": True,
    }
    path, receipt, record = begin("score-" + phase + ".mp3", body)
    if record is None:
        return path
    c = client()
    budget(c, 0.2)
    write(receipt, record)
    parts = []
    generation_id = None
    usage = None
    with c.chat.completions.create(**body) as stream:
        for chunk in stream:
            data = chunk.model_dump()
            generation_id = data.get("id") or generation_id
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                audio = choice.get("delta", {}).get("audio") or {}
                if audio.get("data"):
                    parts.append(base64.b64decode(audio["data"]))
    content = b"".join(parts)
    if len(content) < 10000:
        raise RuntimeError(
            "No complete music audio returned; retain receipt before any retry"
        )
    return complete(
        path,
        receipt,
        record,
        content,
        generation_id=generation_id,
        usage=usage,
        audio_chunk_count=len(parts),
        music_fallback_authorized=True,
    )


def music():
    brief = json.loads(BRIEF.read_text())
    body = {
        "prompt": brief["music_prompt"],
        "music_length_ms": 300000,
        "force_instrumental": True,
        "model_id": "music_v2_5",
    }
    path, receipt, record = begin("score.mp3", body)
    if record is None:
        return path
    # No automatic purchase, top-up or repeated request on account rejection.
    write(receipt, record)
    with httpx.Client(timeout=600, follow_redirects=False) as c:
        response = c.post(
            "https://api.elevenlabs.io/v1/music?output_format=mp3_48000_192",
            headers={"xi-api-key": secret("elevenlabs")},
            json=body,
        )
    if response.status_code != 200:
        record.update(state="rejected", http_status=response.status_code)
        write(receipt, record)
        raise RuntimeError(
            f"Music API HTTP {response.status_code}; no retry or purchase performed"
        )
    if "audio" not in response.headers.get("content-type", ""):
        raise RuntimeError("Music response is not audio")
    return complete(
        path,
        receipt,
        record,
        response.content,
        song_id=response.headers.get("song-id"),
        content_type=response.headers.get("content-type"),
        separate_music_spend_ceiling_usd=1.2,
    )


def transcribe(path, model="openai/whisper-1"):
    if not json.loads(BRIEF.read_text())["authorization"].get(
        "paid_generation_authorized"
    ):
        raise RuntimeError(
            "Paid generation held pending provider-key rotation confirmation"
        )
    path = Path(path)
    receipt = (
        ASSETS
        if model == "openai/whisper-1"
        else ASSETS / ("asr-" + model.split("/")[-1])
    ) / (path.stem + "-asr.json")
    if receipt.exists():
        data = json.loads(receipt.read_text())
        if data.get("source_sha256") != sha(path):
            raise RuntimeError("ASR source changed")
        return data
    body = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(path.read_bytes()).decode(),
            "format": path.suffix[1:],
        },
        "language": "en",
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
    }
    c = client()
    budget(c, 0.1)
    data = c.post("/audio/transcriptions", cast_to=dict[str, Any], body=body)
    if not data.get("words"):
        raise RuntimeError("No measured word timestamps returned")
    data.update(source_sha256=sha(path), model=body["model"])
    write(receipt, data)
    print("transcribed", path.name, len(data["words"]), flush=True)
    return data


def video(name, model, duration, retry_failed=False):
    if not json.loads(BRIEF.read_text())["authorization"].get(
        "paid_generation_authorized"
    ):
        raise RuntimeError(
            "Paid generation held pending provider-key rotation confirmation"
        )
    gate = json.loads((WORK / "animatic-gate.json").read_text())
    if gate.get("state") != "reviewed_for_full_generation" or gate["sha256"] != sha(
        WORK / "animatic.mp4"
    ):
        raise RuntimeError("Full video requires the reviewed current animatic")
    source = ASSETS / (name + "-keyframe.png")
    job = ASSETS / (name + "-job.json")
    out = ASSETS / (name + "-motion.mp4")
    prompt = f"A {duration}-second 16:9 continuous cinematic camera shot. Preserve the supplied first frame exactly: glass and brass measurement instrument, shapes, architecture, palette and light. "
    directions = {
        "instrument": "A very slow, beautiful forward-and-right macro dolly around the optical cradle and luminous prism. Gentle physically coherent parallax and caustics. The instrument remains rigid and stable.",
        "observatory": "A graceful slow sideways tracking move revealing the arc of glass lenses. Architectural precision and stable lens spacing, delicate reflections shifting naturally. No objects appear or disappear.",
        "ledger": "A delicate close dolly along the transparent glass specimen tablet, fine brass seal and luminous edges. Elegant physical reflections; no text or new objects.",
    }
    directions["observatory-stable"] = (
        "A tiny, constant-speed sideways camera track, parallel to the table, keeping exactly the same focal length and distance from the optical cradle. The instrument and every lens stem stay perfectly rigid and attached to the table. Preserve the original geometry and ray direction, with only a slight physically coherent change in reflection. No forward push, zoom, rack focus, lens breathing or shape changes."
    )
    prompt += directions.get(name, directions[name.split("-")[0]])
    prompt += " No cuts, dialogue, music, people, text, interface, morphed geometry, flicker, pulsing energy or exaggerated motion. Premium restrained scientific editorial film."
    body = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": "16:9",
        "resolution": "720p" if "seedance" in model else "1080p",
        "generate_audio": False,
        "frame_images": [{**ref(source), "frame_type": "first_frame"}],
    }
    fp = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    c = client()
    if retry_failed and job.exists():
        prior = json.loads(job.read_text())
        if not prior.get("id") or out.exists():
            raise RuntimeError("Retry requires a known failed job with no output")
        fresh = c.get("/videos/" + prior["id"], cast_to=dict[str, Any])
        if fresh.get("status") != "failed":
            raise RuntimeError("Retry held: existing job is not terminally failed")
        prior.update(
            fresh,
            state="failed",
            retry_reason="Verified terminal provider failure; explicit bounded retry",
        )
        archive = job.with_name(name + "-failed-" + prior["id"] + ".json")
        if archive.exists():
            raise RuntimeError("Failed job archive already exists")
        write(archive, prior)
        job.unlink()
    if job.exists():
        record = json.loads(job.read_text())
        if record["request_sha256"] != fp:
            raise RuntimeError("Video fingerprint changed")
        if out.exists() and record.get("sha256") == sha(out):
            return out
        if not record.get("id"):
            raise RuntimeError("Prior submit outcome unknown; do not resubmit")
    else:
        budget(c, 5 if "seedance" in model else duration * 0.25)
        capabilities = next(
            m
            for m in c.get("/videos/models", cast_to=dict[str, Any])["data"]
            if m["id"] == model
        )
        for key, field in [
            ("duration", "supported_durations"),
            ("resolution", "supported_resolutions"),
            ("aspect_ratio", "supported_aspect_ratios"),
        ]:
            if body[key] not in capabilities[field]:
                raise RuntimeError(f"Unsupported {key}")
        record = {
            "state": "submitting",
            "request_sha256": fp,
            "model": model,
            "reference_sha256": sha(source),
            "prompt": prompt,
            "duration": duration,
        }
        write(job, record)
        try:
            data = c.post("/videos", cast_to=dict[str, Any], body=body)
        except APIStatusError as error:
            record.update(
                state="rejected", http_status=error.status_code, error=str(error)[:1500]
            )
            write(job, record)
            raise
        record.update(data)
        write(job, record)
        print("submitted", name, record["id"], flush=True)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        status = c.get("/videos/" + record["id"], cast_to=dict[str, Any])
        record.update(status)
        write(job, record)
        if record.get("status") == "completed":
            response = c.get(
                "/videos/" + record["id"] + "/content?index=0", cast_to=httpx.Response
            )
            if "video" not in response.headers.get("content-type", ""):
                raise RuntimeError(
                    "Completed video content response is not video; preserve existing job"
                )
            pending = out.with_suffix(".pending.mp4")
            pending.write_bytes(response.content)
            subprocess.run(["ffprobe", "-v", "error", str(pending)], check=True)
            pending.replace(out)
            record.update(state="completed", sha256=sha(out), bytes=out.stat().st_size)
            write(job, record)
            print(
                "completed", name, "usage", json.dumps(record.get("usage")), flush=True
            )
            return out
        if record.get("status") in ["failed", "canceled", "cancelled"]:
            record["state"] = record["status"]
            write(job, record)
            raise RuntimeError(
                "Video failed, retained job: " + str(record.get("error"))[:250]
            )
        time.sleep(30)
    raise RuntimeError("Existing video job still pending; resume without resubmission")


def review(name, prompt, paths):
    if not json.loads(BRIEF.read_text())["authorization"].get(
        "paid_generation_authorized"
    ):
        raise RuntimeError(
            "Paid generation held pending provider-key rotation confirmation"
        )
    content = [{"type": "text", "text": prompt}]
    hashes = {}
    if sum(Path(value).stat().st_size * 4 / 3 for value in paths) > 28_000_000:
        raise RuntimeError(
            "Review media exceeds safe router payload; use smaller frame-sequence-preserving review proxies"
        )
    for value in paths:
        path = Path(value)
        hashes[str(path)] = sha(path)
        data = base64.b64encode(path.read_bytes()).decode()
        if path.suffix in [".wav", ".mp3"]:
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": data, "format": path.suffix[1:]},
                }
            )
        elif path.suffix == ".mp4":
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64," + data},
                }
            )
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/"
                        + (
                            "png"
                            if path.suffix == ".png"
                            else "webp"
                            if path.suffix == ".webp"
                            else "jpeg"
                        )
                        + ";base64,"
                        + data
                    },
                }
            )
    body = {
        "model": "google/gemini-3.8-flash",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.15,
        "max_tokens": 3500,
        "extra_body": {"reasoning": {"effort": "low"}},
    }
    out = WORK / "review" / (name + ".json")
    fp = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("request_sha256") == fp and old.get("state") == "completed":
            return old
        raise RuntimeError("Review identity changed or prior outcome unresolved")
    c = client()
    budget(c, 0.8)
    write(out, {"state": "submitted", "request_sha256": fp, "media_sha256": hashes})
    try:
        data = c.chat.completions.create(**body).model_dump()
    except APIStatusError as error:
        write(
            out,
            {
                "state": "rejected",
                "http_status": error.status_code,
                "request_sha256": fp,
                "media_sha256": hashes,
                "error": str(error)[:1500],
            },
        )
        raise
    finished = data["choices"][0].get("finish_reason") == "stop"
    data.update(
        state="completed" if finished else "incomplete",
        request_sha256=fp,
        media_sha256=hashes,
    )
    write(out, data)
    if not finished:
        raise RuntimeError(
            "Review response incomplete; preserve it and narrow a new review identity"
        )
    print(data["choices"][0]["message"]["content"], flush=True)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "kind",
        choices=[
            "voice",
            "image",
            "music",
            "asr",
            "video",
            "review",
            "budget",
            "music-router",
        ],
    )
    ap.add_argument("value", nargs="?")
    ap.add_argument("--model")
    ap.add_argument("--duration", type=int, default=8)
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="Verify and retain a terminal failed job before one new submission",
    )
    ap.add_argument("--prompt")
    ap.add_argument("--media", nargs="*")
    args = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    if args.kind == "voice":
        voice(int(args.value))
    elif args.kind == "image":
        image(args.value)
    elif args.kind == "music":
        music()
    elif args.kind == "music-router":
        music_router(args.value)
    elif args.kind == "asr":
        transcribe(args.value, args.model or "openai/whisper-1")
    elif args.kind == "video":
        video(
            args.value, args.model or "google/veo-3.1", args.duration, args.retry_failed
        )
    elif args.kind == "review":
        review(args.value, Path(args.prompt).read_text(), args.media)
    else:
        budget(client(), 0)
