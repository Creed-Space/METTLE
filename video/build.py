#!/usr/bin/env python3
"""METTLE explainer video builder.

Pipeline: narration script -> Gemini TTS (voice: Sadaltager) -> branded HTML
slides screenshotted with headless Chrome -> ffmpeg Ken Burns assembly with
per-scene fades -> mp4 + WebVTT captions + poster.

The SaferAgenticAI and Psychopathia explainers were built with the same
approach (Gemini 3.1 Flash TTS narration over motion slides) but their
pipelines were never committed; this one is, so it can be re-run and edited.

Usage:
    python3 video/build.py            # full build
    python3 video/build.py --no-tts   # reuse cached audio only (fails if missing)

Requires: GOOGLE_API_KEY, google-genai, ffmpeg, Google Chrome.
Outputs: static/mettle-explainer.mp4, static/mettle-explainer.vtt,
         static/mettle-explainer-poster.webp
"""

import argparse
import hashlib
import re

# The pipeline invokes fixed local media tools without a shell.
import subprocess  # nosec B404
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "video" / "build"
STATIC = REPO / "static"
FONTS = STATIC / "fonts"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TTS_MODEL = "gemini-3.1-flash-tts-preview"
TTS_VOICE = "Sadaltager"
TTS_STYLE = (
    "Narrate in a clear British accent: a warm, measured, confident "
    "documentary voiceover, unhurried but never sluggish. Read exactly "
    "this text:\n\n"
)

FPS = 30
PRE_PAD = 0.45  # silence before narration in each scene
POST_PAD = 0.55  # minimum silence after narration in each scene
FADE = 0.4  # per-scene fade in/out, seconds

# ---------------------------------------------------------------------------
# Scenes: narration + slide body HTML. Slide CSS is shared (SLIDE_CSS below).
# ---------------------------------------------------------------------------

SCENES = [
    {
        "id": "01-hook",
        "narration": (
            "For decades, websites have asked people to prove they are human. "
            "METTLE explores the reverse question: how does a respondent perform "
            "on machine-oriented challenges? A pass is evidence about that "
            "session. It is not proof that the respondent is nonhuman."
        ),
        "html": """
        <div class="center-stack">
          <div class="kicker">EXPERIMENTAL INVERSE TURING SCREENING</div>
          <h1 class="mega">MEASURE<br><span class="accent">PERFORMANCE.</span></h1>
          <div class="rule"></div>
          <div class="sub">A result does not establish identity or substrate</div>
        </div>
        """,
    },
    {
        "id": "02-what",
        "narration": (
            "This is METTLE: Machine Evaluation Through Turing-inverse Logic "
            "Examination. It is an experimental challenge protocol that records "
            "answers, timing, and scores under a versioned policy. It makes a "
            "bounded behavioral claim, rather than an identity claim."
        ),
        "html": """
        <div class="center-stack">
          <div class="wordmark">METTLE</div>
          <div class="expansion">
            <span><b>M</b>achine</span>
            <span><b>E</b>valuation</span>
            <span><b>T</b>hrough</span>
            <span><b>T</b>uring-inverse</span>
            <span><b>L</b>ogic</span>
            <span><b>E</b>xamination</span>
          </div>
          <div class="sub">Versioned, machine-oriented behavioral evidence</div>
        </div>
        """,
    },
    {
        "id": "03-philosophy",
        "narration": (
            "The protocol samples arithmetic, pattern completion, constrained "
            "instruction following, self-report, consistency, and changes across "
            "feedback rounds. These are observable response properties. Humans, "
            "models, relays, and purpose-built solvers may imitate them."
        ),
        "html": """
        <div class="left-stack">
          <div class="kicker">WHAT METTLE RECORDS</div>
          <h2>Behavior within one session</h2>
          <ul class="cap-list">
            <li>Answers and server-observed timing</li>
            <li>Constraint and format adherence</li>
            <li>Consistency and calibrated self-report</li>
            <li>Changes across feedback rounds</li>
            <li>Policy-specific scores and outcomes</li>
          </ul>
        </div>
        """,
    },
    {
        "id": "04-suites",
        "narration": (
            "Twelve suites organize distinct research hypotheses. Challenge "
            "instances are selected or generated for each session, and expected "
            "answers stay on the server where applicable. The service checks the "
            "submitted response and records timing under the suite policy. Suite "
            "twelve is supplemental and cannot raise a credential tier."
        ),
        "html": """
        <div class="left-stack">
          <div class="kicker">TWELVE SUITES &middot; RESEARCH HYPOTHESES</div>
          <div class="suite-grid">
            <div class="suite"><i>01</i>Adversarial Robustness</div>
            <div class="suite"><i>02</i>Native AI Capabilities</div>
            <div class="suite"><i>03</i>Self-Reference</div>
            <div class="suite"><i>04</i>Social &amp; Temporal</div>
            <div class="suite"><i>05</i>Inverse Turing</div>
            <div class="suite"><i>06</i>Anti-Thrall Detection</div>
            <div class="suite"><i>07</i>Agency Detection</div>
            <div class="suite"><i>08</i>Counter-Coaching</div>
            <div class="suite"><i>09</i>Intent &amp; Provenance</div>
            <div class="suite"><i>10</i>Novel Reasoning</div>
            <div class="suite"><i>11</i>Governance Verification</div>
            <div class="suite"><i>12</i>LLM-Dynamic</div>
          </div>
        </div>
        """,
    },
    {
        "id": "05-deeper",
        "narration": (
            "Some suite names ask whether a respondent is free, owns its mission, "
            "or is genuine. Those names frame research questions. The protocol "
            "measures answers to challenges and self-reports. Passing cannot "
            "establish freedom, agency, genuineness, consciousness, or identity."
        ),
        "html": """
        <div class="center-stack">
          <div class="kicker">SUITE NAMES ARE HYPOTHESES</div>
          <div class="questions">
            <div>Freedom is <span class="accent">not proven</span></div>
            <div>Agency is <span class="accent">not proven</span></div>
            <div>Genuineness is <span class="accent">not proven</span></div>
            <div>Identity is <span class="accent">not proven</span></div>
          </div>
        </div>
        """,
    },
    {
        "id": "06-governance",
        "narration": (
            "The governance suite evaluates responses to policy scenarios. Any "
            "VCP governance metadata is supplied by the caller and marked "
            "unverified. METTLE does not independently attest a constitution, "
            "action gate, operator, runtime control, safety property, or "
            "governance system."
        ),
        "html": """
        <div class="left-stack">
          <div class="kicker">GOVERNANCE BOUNDARY</div>
          <h2>Claims remain caller supplied</h2>
          <ul class="cap-list">
            <li>Policy-scenario responses are scored</li>
            <li>VCP source is marked unverified</li>
            <li>No runtime control is inspected</li>
            <li>No operator identity is authenticated</li>
            <li>No safety or governance property is certified</li>
          </ul>
        </div>
        """,
    },
    {
        "id": "07-antigaming",
        "narration": (
            "Procedural variation, server-held answers, sequential challenge "
            "release, bearer tokens, and replay controls raise the cost of simple "
            "reuse. Server-side timing supports policy checks where configured. "
            "These controls do not rule out relays, source-aware solvers, "
            "model-assisted humans, imitation, leakage, or evaluator error."
        ),
        "html": """
        <div class="left-stack">
          <div class="kicker">ANTI-REPLAY CONTROLS AND LIMITS</div>
          <div class="mech-list">
            <div class="mech"><b>Challenge variation</b><span>reduces exact reuse</span></div>
            <div class="mech"><b>Server-held answers</b><span>limits direct disclosure</span></div>
            <div class="mech"><b>Sequential release</b><span>withholds future rounds</span></div>
            <div class="mech"><b>Replay controls</b><span>bind session and submission</span></div>
            <div class="mech"><b>Server timing</b><span>records policy observations</span></div>
            <div class="mech"><b>Residual risk</b><span>relays and solvers remain possible</span></div>
          </div>
        </div>
        """,
    },
    {
        "id": "08-credentials",
        "narration": (
            "Eligible contiguous suite ranges may receive an Ed25519-signed "
            "credential. It binds the issuer, policy, session result, tier, "
            "expiry, and revocable identifier. The signature establishes issuer "
            "and integrity. A tier summarizes which suite range passed; it does "
            "not certify the properties named by those suites."
        ),
        "html": """
        <div class="left-stack">
          <div class="kicker">SIGNED RESULTS &middot; ED25519</div>
          <div class="tiers">
            <div class="tier t-bronze"><b>Bronze</b><span>complete suite range 1 through 5</span></div>
            <div class="tier t-silver"><b>Silver</b><span>complete suite range 1 through 7</span></div>
            <div class="tier t-gold"><b>Gold</b><span>complete suite range 1 through 9</span></div>
            <div class="tier t-platinum"><b>Platinum</b><span>complete suite range 1 through 11</span></div>
          </div>
          <div class="sub">Tier names summarize policy results, not identity or traits</div>
        </div>
        """,
    },
    {
        "id": "09-usecases",
        "narration": (
            "Use METTLE to compare challenge performance in research, sandbox "
            "participation, or as one supplemental risk signal. Never rely on a "
            "METTLE result alone for identity, authorization, trading, deployment, "
            "privileged infrastructure, or another high-impact decision."
        ),
        "html": """
        <div class="left-stack">
          <div class="kicker">BOUNDED RELYING-PARTY USE</div>
          <div class="cases">
            <div class="case"><b>Research</b><span>compare policy-specific performance</span></div>
            <div class="case"><b>Sandboxes</b><span>support low-risk participation</span></div>
            <div class="case"><b>Risk signals</b><span>combine with independent controls</span></div>
            <div class="case"><b>High-impact access</b><span>never authorize from METTLE alone</span></div>
          </div>
        </div>
        """,
    },
    {
        "id": "10-close",
        "narration": (
            "METTLE is open source under Apache two point oh. Pip install "
            "mettle verifier for an unsigned local screening. The hosted API may "
            "issue signed, time-limited results under its published policy. Read "
            "the assurance limits, verify status, and add controls proportionate "
            "to your risk. Measure your mettle."
        ),
        "html": """
        <div class="center-stack">
          <div class="wordmark">METTLE</div>
          <code class="install">pip install mettle-verifier</code>
          <div class="sub">mettle.sh &middot; Apache 2.0 &middot; by Creed Space</div>
          <div class="tagline">Measure your mettle.</div>
        </div>
        """,
    },
]

SLIDE_CSS = f"""
@font-face {{ font-family: 'Space Grotesk'; src: url('file://{FONTS}/SpaceGrotesk-700.woff2') format('woff2'); font-weight: 700; }}
@font-face {{ font-family: 'Space Grotesk'; src: url('file://{FONTS}/SpaceGrotesk-600.woff2') format('woff2'); font-weight: 600; }}
@font-face {{ font-family: 'Space Grotesk'; src: url('file://{FONTS}/SpaceGrotesk-500.woff2') format('woff2'); font-weight: 500; }}
@font-face {{ font-family: 'IBM Plex Sans'; src: url('file://{FONTS}/IBMPlexSans-600.woff2') format('woff2'); font-weight: 600; }}
@font-face {{ font-family: 'Inter'; src: url('file://{FONTS}/Inter-400.woff2') format('woff2'); font-weight: 400; }}
@font-face {{ font-family: 'Inter'; src: url('file://{FONTS}/Inter-500.woff2') format('woff2'); font-weight: 500; }}
@font-face {{ font-family: 'Inter'; src: url('file://{FONTS}/Inter-600.woff2') format('woff2'); font-weight: 600; }}
@font-face {{ font-family: 'JetBrains Mono'; src: url('file://{FONTS}/JetBrainsMono-400.woff2') format('woff2'); font-weight: 400; }}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ width: 1920px; height: 1080px; overflow: hidden; }}
body {{
  background:
    radial-gradient(1100px 700px at 72% 18%, rgba(20,184,166,0.10), transparent 65%),
    radial-gradient(900px 600px at 15% 85%, rgba(15,118,110,0.10), transparent 60%),
    linear-gradient(160deg, #0b1514 0%, #070d0c 100%);
  color: #e6f2f0;
  font-family: 'Inter', sans-serif;
  display: flex; align-items: center; justify-content: center;
  position: relative;
}}
body::before {{
  content: ''; position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(20,184,166,0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(20,184,166,0.045) 1px, transparent 1px);
  background-size: 96px 96px;
  mask-image: radial-gradient(1200px 800px at 50% 45%, #000 30%, transparent 100%);
}}
.brandline {{
  position: absolute; bottom: 54px; left: 0; right: 0;
  display: flex; justify-content: center; gap: 18px; align-items: center;
  font-family: 'IBM Plex Sans', sans-serif; font-weight: 600;
  font-size: 24px; letter-spacing: 0.24em; color: #3e5f5a;
}}
.accent {{ color: #14b8a6; }}
.dim {{ color: #6b8a86; }}
.kicker {{
  font-family: 'JetBrains Mono', monospace; font-size: 27px;
  letter-spacing: 0.42em; color: #14b8a6; margin-bottom: 44px;
}}
.center-stack {{ text-align: center; max-width: 1560px; position: relative; }}
.left-stack {{ max-width: 1480px; width: 1480px; position: relative; }}
.rule {{ width: 220px; height: 4px; background: #14b8a6; margin: 56px auto 0; border-radius: 2px; }}
h1.mega {{
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 148px; line-height: 1.04; letter-spacing: -0.01em;
}}
h2 {{
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 84px; line-height: 1.1; margin-bottom: 56px;
}}
.wordmark {{
  font-family: 'IBM Plex Sans', sans-serif; font-weight: 600;
  font-size: 190px; letter-spacing: 0.06em; color: #e6f2f0;
}}
.wordmark::first-letter {{ color: #14b8a6; }}
.expansion {{
  display: flex; gap: 38px; justify-content: center; margin-top: 40px;
  font-size: 33px; color: #9fbcb7; font-weight: 500;
}}
.expansion b {{ color: #14b8a6; font-weight: 600; }}
.sub {{
  margin-top: 52px; font-size: 36px; color: #6b8a86; letter-spacing: 0.04em;
}}
ul.cap-list {{ list-style: none; }}
ul.cap-list li {{
  font-family: 'Space Grotesk', sans-serif; font-weight: 500;
  font-size: 57px; padding: 26px 0 26px 66px; position: relative;
  border-bottom: 1px solid rgba(20,184,166,0.16);
}}
ul.cap-list li::before {{
  content: ''; position: absolute; left: 6px; top: 50%;
  width: 22px; height: 22px; transform: translateY(-50%) rotate(45deg);
  background: rgba(20,184,166,0.9); border-radius: 4px;
}}
.suite-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 26px; }}
.suite {{
  background: rgba(20,184,166,0.06); border: 1px solid rgba(20,184,166,0.22);
  border-radius: 14px; padding: 32px 34px;
  font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 39px;
}}
.suite i {{
  display: block; font-style: normal; font-family: 'JetBrains Mono', monospace;
  font-size: 25px; color: #14b8a6; margin-bottom: 12px; letter-spacing: 0.2em;
}}
.questions div {{
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 105px; line-height: 1.5;
}}
.mech-list {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px 60px; }}
.mech {{
  border-left: 5px solid #14b8a6; padding: 14px 0 14px 36px;
}}
.mech b {{
  display: block; font-family: 'Space Grotesk', sans-serif;
  font-weight: 600; font-size: 47px; margin-bottom: 8px;
}}
.mech span {{ font-size: 33px; color: #6b8a86; }}
.tiers {{ display: flex; flex-direction: column; gap: 30px; }}
.tier {{
  display: flex; align-items: baseline; gap: 44px;
  border: 1px solid rgba(230,242,240,0.14); border-radius: 16px;
  padding: 32px 48px; background: rgba(230,242,240,0.03);
}}
.tier b {{
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 62px; width: 330px;
}}
.tier span {{ font-size: 38px; color: #9fbcb7; }}
.t-bronze b {{ color: #d29a6b; }}
.t-silver b {{ color: #c9d4d2; }}
.t-gold b {{ color: #e9c46a; }}
.t-platinum b {{ color: #7fe7db; }}
.cases {{ display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }}
.case {{
  background: rgba(20,184,166,0.06); border: 1px solid rgba(20,184,166,0.22);
  border-radius: 18px; padding: 52px 54px;
}}
.case b {{
  display: block; font-family: 'Space Grotesk', sans-serif;
  font-weight: 700; font-size: 57px; margin-bottom: 18px;
}}
.case span {{ font-size: 36px; color: #9fbcb7; line-height: 1.4; }}
code.install {{
  display: inline-block; margin-top: 64px;
  font-family: 'JetBrains Mono', monospace; font-size: 51px;
  color: #7fe7db; background: rgba(20,184,166,0.08);
  border: 1px solid rgba(20,184,166,0.3); border-radius: 14px;
  padding: 30px 56px;
}}
.tagline {{
  margin-top: 74px; font-family: 'Space Grotesk', sans-serif;
  font-weight: 600; font-size: 66px; color: #14b8a6; letter-spacing: 0.02em;
}}
"""

SLIDE_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{css}</style></head>
<body>
{body}
<div class="brandline">METTLE &nbsp;&middot;&nbsp; BEHAVIORAL SCREENING &nbsp;&middot;&nbsp; METTLE.SH</div>
</body></html>
"""


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)  # nosec B603
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{r.stderr[-3000:]}")
    return r


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def tts(scene, cache_dir: Path, allow_generate: bool) -> Path:
    key = hashlib.sha1(
        f"{TTS_MODEL}|{TTS_VOICE}|{TTS_STYLE}|{scene['narration']}".encode(),
        usedforsecurity=False,
    ).hexdigest()[:16]
    out = cache_dir / f"{scene['id']}-{key}.wav"
    if out.exists():
        return out
    if not allow_generate:
        sys.exit(f"Missing cached TTS for {scene['id']} and --no-tts given")

    from google import genai
    from google.genai import types

    client = genai.Client()
    resp = client.models.generate_content(
        model=TTS_MODEL,
        contents=TTS_STYLE + scene["narration"],
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=TTS_VOICE
                    )
                )
            ),
        ),
    )
    try:
        part = resp.candidates[0].content.parts[0].inline_data  # type: ignore[index,union-attr]
        assert part is not None
        audio_bytes = part.data
        assert audio_bytes
    except (TypeError, AttributeError, IndexError, AssertionError):
        sys.exit(f"TTS returned no audio for {scene['id']}: {resp}")
    rate = 24000
    m = re.search(r"rate=(\d+)", part.mime_type or "")
    if m:
        rate = int(m.group(1))
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio_bytes)
    print(f"  TTS {scene['id']}: {wav_duration(out):.2f}s")
    return out


def render_slide(scene, slides_dir: Path) -> Path:
    html_path = slides_dir / f"{scene['id']}.html"
    png_path = slides_dir / f"{scene['id']}.png"
    html = SLIDE_TEMPLATE.format(css=SLIDE_CSS, body=scene["html"])
    if png_path.exists() and html_path.exists() and html_path.read_text() == html:
        return png_path
    html_path.write_text(html)
    if png_path.exists():
        png_path.unlink()
    profile = WORK / "chrome-profile"
    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-crash-reporter",
        "--force-device-scale-factor=2",
        "--window-size=1920,1080",
        "--virtual-time-budget=3000",
        "--allow-file-access-from-files",
        f"--screenshot={png_path}",
        f"file://{html_path}",
    ]
    last = None
    for _ in range(3):  # Chrome exit codes are unreliable; the PNG is the truth
        # Headless Chrome sometimes writes the screenshot and then never exits,
        # which would block this call forever. Time it out and let the retry
        # loop check the PNG, which is the actual success signal.
        try:
            last = subprocess.run(  # nosec B603
                cmd, capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired:
            last = None
        if png_path.exists() and png_path.stat().st_size > 0:
            return png_path
    sys.exit(
        f"Slide render failed for {scene['id']} after 3 attempts:\n"
        f"{(last.stderr if last else '')[-2000:]}"
    )


def build_scene_clip(i, png: Path, wav: Path, clips_dir: Path):
    """Render one scene: static slide with fades and padded audio.

    Deliberately no zoompan/Ken Burns: sub-pixel zoom rates jitter visibly
    on crisp text (integer pixel rounding), so slides hold still.
    """
    audio_dur = wav_duration(wav)
    frames = round((audio_dur + PRE_PAD + POST_PAD) * FPS)
    dur = frames / FPS
    out = clips_dir / f"{i:02d}.mkv"
    run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-t",
            f"{dur:.4f}",
            "-i",
            str(png),
            "-i",
            str(wav),
            "-filter_complex",
            f"[0:v]scale=1920:1080:flags=lanczos,"
            f"fade=t=in:st=0:d={FADE},fade=t=out:st={dur - FADE:.3f}:d={FADE},"
            f"format=yuv420p[v];"
            f"[1:a]adelay={int(PRE_PAD * 1000)},apad,atrim=0:{dur:.4f},"
            f"aresample=48000[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "19",
            "-c:a",
            "pcm_s16le",
            "-t",
            f"{dur:.4f}",
            str(out),
        ]
    )
    return out, dur, audio_dur


def write_vtt(timeline, out_path: Path):
    def fmt(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    cues = []
    for start, audio_dur, narration in timeline:
        sentences = re.findall(r"[^.?!]+[.?!]", narration)
        total_chars = sum(len(s) for s in sentences)
        t = start + PRE_PAD
        # group short sentences so cues are not machine-gun paced
        groups, cur = [], ""
        for s in sentences:
            cur += s
            if len(cur) > 90:
                groups.append(cur.strip())
                cur = ""
        if cur.strip():
            groups.append(cur.strip())
        for g in groups:
            d = audio_dur * len(g) / total_chars
            cues.append((t, t + d, g))
            t += d
    lines = ["WEBVTT", ""]
    for a, b, text in cues:
        lines += [f"{fmt(a)} --> {fmt(b)}", text, ""]
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tts", action="store_true", help="only use cached audio")
    args = ap.parse_args()

    cache = REPO / "video" / "tts-cache"
    slides = WORK / "slides"
    clips = WORK / "clips"
    for d in (cache, slides, clips):
        d.mkdir(parents=True, exist_ok=True)

    print("== TTS ==")
    wavs = [tts(s, cache, not args.no_tts) for s in SCENES]
    print("== Slides ==")
    pngs = [render_slide(s, slides) for s in SCENES]
    print("== Scene clips ==")
    timeline, clip_paths, t = [], [], 0.0
    for i, (scene, png, wav) in enumerate(zip(SCENES, pngs, wavs)):
        clip, dur, audio_dur = build_scene_clip(i, png, wav, clips)
        clip_paths.append(clip)
        timeline.append((t, audio_dur, scene["narration"]))
        t += dur
        print(f"  {scene['id']}: {dur:.2f}s (voice {audio_dur:.2f}s)")

    print(f"== Assemble ({t:.1f}s total) ==")
    concat_list = WORK / "concat.txt"
    concat_list.write_text("".join(f"file '{p}'\n" for p in clip_paths))
    final = STATIC / "mettle-explainer.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(final),
        ]
    )
    write_vtt(timeline, STATIC / "mettle-explainer.vtt")
    poster_png = WORK / "poster.png"
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "1.8",
            "-i",
            str(final),
            "-frames:v",
            "1",
            "-vf",
            "scale=1280:720",
            str(poster_png),
        ]
    )
    poster = STATIC / "mettle-explainer-poster.webp"
    try:
        from PIL import Image

        Image.open(poster_png).save(poster, "WEBP", quality=82)
    except ImportError:
        # homebrew ffmpeg has no webp encoder; fall back to shipping the png
        poster = STATIC / "mettle-explainer-poster.png"
        poster_png.replace(poster)
    size_mb = final.stat().st_size / 1e6
    print(f"DONE: {final} ({t:.1f}s, {size_mb:.1f} MB)")
    print(f"      {STATIC / 'mettle-explainer.vtt'}")
    print(f"      {poster}")


if __name__ == "__main__":
    main()
