#!/usr/bin/env python3
"""METTLE explainer video builder.

Pipeline: narration script -> Gemini TTS (voice: Sadaltager) -> whisper word
alignment -> animated HTML scenes whose reveals are keyed to the spoken words ->
deterministic frame capture in headless Chromium (Playwright) -> ffmpeg
assembly with per-scene fades -> mp4 + WebVTT captions + poster.

The SaferAgenticAI and Psychopathia explainers were built with the same
approach (Gemini 3.1 Flash TTS narration over motion slides) but their
pipelines were never committed; this one is, so it can be re-run and edited.

Usage:
    python3 video/build.py                 # full build
    python3 video/build.py --no-tts        # reuse cached audio only (fails if missing)
    python3 video/build.py --regen 05-questions   # discard one scene's audio and re-voice it

Requires: GOOGLE_API_KEY (uncached narration only), google-genai, ffmpeg,
whisper-cli with ggml-base.en (uncached alignment only), Python Playwright with
its Chromium, Pillow.
Outputs: static/mettle-explainer.mp4, static/mettle-explainer.vtt,
         static/mettle-explainer-poster.webp
"""

import argparse
import base64
import difflib
import hashlib
import json
import os
import re

# The pipeline invokes fixed local media tools without a shell.
import subprocess  # nosec B404
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "video" / "build"
CACHE = REPO / "video" / "tts-cache"
STATIC = REPO / "static"
FONTS = STATIC / "fonts"

TTS_MODEL = "gemini-3.1-flash-tts-preview"
TTS_VOICE = "Sadaltager"
TTS_STYLE = (
    "You are the narrator of a short technical explainer film. Speak in a "
    "clear, natural British accent: warm, intelligent and quietly confident, "
    "like a good science documentary. Keep an even, engaged pace with a brief "
    "natural pause at each full stop and before each question. Give words "
    "written in capitals gentle emphasis without shouting. Read exactly this "
    "text, and nothing else:\n\n"
)
WHISPER_MODEL = Path(
    os.environ.get(
        "METTLE_WHISPER_MODEL",
        Path.home() / ".cache" / "whisper-models" / "ggml-base.en.bin",
    )
)
MIN_ALIGNMENT = 0.8  # share of script words whisper must hear in the TTS take

FPS = 30
SCALE = 2  # capture at 3840x2160, downsample with lanczos for clean text edges
JPEG_QUALITY = 93
PRE_PAD = 0.5  # silence before narration in each scene
POST_PAD = 0.9  # silence after narration: lets the last reveal settle
FADE = 0.35  # per-scene fade in/out, seconds
CUE_LEAD = 0.12  # start a reveal slightly before its word is heard
RENDER_WORKERS = 5
TARGET_LUFS = -16.0  # web delivery loudness; each take gets one linear gain

# ---------------------------------------------------------------------------
# Scenes: narration + scene body HTML. Shared CSS is SCENE_CSS below.
#
# `data-cue` keys an element's entrance to the narration:
#   s3            start of the fourth sentence
#   w:session     first time the word "session" is spoken (#2 for the second)
#   ...@+0.4      shifted by 0.4 s
# Children inherit the parent's cue; `--d` staggers them after it.
# ---------------------------------------------------------------------------

SCENES = [
    {
        "id": "01-hook",
        "chapter": "The reverse question",
        "narration": (
            "For decades, websites have asked people to prove they are human. "
            "METTLE turns that question around. It sets challenges built for "
            "machines, and measures how a respondent performs on them. A pass is "
            "evidence about one session. It is not proof that the respondent is "
            "nonhuman."
        ),
        "html": """
        <div class="stack center">
          <div class="flip-row">
            <div class="probe rv" data-cue="s0">
              <span class="probe-box"><svg viewBox="0 0 24 24"><path class="tick" data-cue="w:human" d="M5 12.5l4.5 4.5L19 7"/></svg></span>
              <span>I am human</span>
            </div>
            <div class="flip-arrow rv" data-cue="s1"><svg viewBox="0 0 64 24"><path d="M2 12h56M46 3l12 9-12 9"/></svg></div>
            <div class="probe probe-rev rv" data-cue="s1" style="--d:.25s">
              <span class="probe-q">?</span>
              <span>How does a respondent perform on machine-oriented challenges?</span>
            </div>
          </div>
          <h1 class="mega"><span class="rv" data-cue="w:measures">MEASURE</span><br><span class="accent rv" data-cue="w:performs">PERFORMANCE.</span></h1>
          <div class="rule grow" data-cue="s3"></div>
          <div class="sub rv" data-cue="s3">A pass is evidence about <b>one session</b></div>
          <div class="sub sub-tight dim rv" data-cue="s4">It is not proof that the respondent is nonhuman</div>
        </div>
        """,
    },
    {
        "id": "02-what",
        "chapter": "What METTLE is",
        "narration": (
            "This is METTLE: Machine Evaluation Through Turing-inverse Logic "
            "Examination. It is an experimental protocol. Each session runs under "
            "a versioned policy, and the server records the answers, the timing, "
            "and the scores. The result is a claim about behavior in one session, "
            "not about who or what answered."
        ),
        "html": """
        <div class="stack center">
          <div class="acro rv" data-cue="s0">
            <div class="acro-col"><b class="lit" data-cue="w:Machine">M</b><span class="rv" data-cue="w:Machine">Machine</span></div>
            <div class="acro-col"><b class="lit" data-cue="w:Evaluation">E</b><span class="rv" data-cue="w:Evaluation">Evaluation</span></div>
            <div class="acro-col"><b class="lit" data-cue="w:Through">T</b><span class="rv" data-cue="w:Through">Through</span></div>
            <div class="acro-col"><b class="lit" data-cue="w:Turing-inverse">T</b><span class="rv" data-cue="w:Turing-inverse">Turing-inverse</span></div>
            <div class="acro-col"><b class="lit" data-cue="w:Logic">L</b><span class="rv" data-cue="w:Logic">Logic</span></div>
            <div class="acro-col"><b class="lit" data-cue="w:Examination">E</b><span class="rv" data-cue="w:Examination">Examination</span></div>
          </div>
          <div class="tag rv" data-cue="s1">EXPERIMENTAL PROTOCOL</div>
          <div class="pipeline">
            <div class="pipe-node rv" data-cue="w:versioned"><i>POLICY</i>versioned, named</div>
            <div class="pipe-link grow" data-cue="w:server"></div>
            <div class="pipe-node rv" data-cue="w:server"><i>SERVER RECORDS</i>
              <span class="chips"><em class="rv" data-cue="w:answers">answers</em><em class="rv" data-cue="w:timing">timing</em><em class="rv" data-cue="w:scores">scores</em></span>
            </div>
          </div>
          <div class="sub rv" data-cue="s3">A claim about behavior in <b>one session</b>, not about who or what answered</div>
        </div>
        """,
    },
    {
        "id": "03-challenges",
        "chapter": "What a session asks",
        "narration": (
            "A full-difficulty speed challenge asks for the sum, difference, or "
            "product of two eight-digit numbers, within half a second. Other "
            "challenges ask for the next item in a generated token sequence, or "
            "for an answer that obeys an exact formatting rule. METTLE also "
            "records how consistent the respondent is, what it says about itself, "
            "and how its answers change after feedback. Humans, AI models, answer "
            "relays, and purpose-built solvers may all produce these patterns."
        ),
        "html": """
        <div class="stack wide">
          <div class="ch-grid">
            <div class="ch-card ch-main rv" data-cue="s0">
              <div class="ch-head"><span>SPEED MATH &middot; FULL DIFFICULTY</span><span class="limit">limit 500 ms</span></div>
              <div class="ch-prompt">Calculate:<br><span class="big">48213907 &times; 71305118</span></div>
              <div class="timer"><div class="timer-fill drain" data-cue="w:half"></div></div>
              <div class="ch-foot rv" data-cue="w:half@+0.6">Expected answer stays on the server</div>
            </div>
            <div class="ch-side">
              <div class="ch-card rv" data-cue="w:token">
                <div class="ch-head"><span>TOKEN SEQUENCE</span><span class="limit">limit 400 ms</span></div>
                <div class="ch-mono seq"><span>Kc41e07-52718304,</span> <span>Kc41e07-52764151,</span> <span>Kc41e07-52809998,</span> <span>Kc41e07-52855845,</span> <b>Kc41e07-___</b></div>
              </div>
              <div class="ch-card rv" data-cue="w:formatting">
                <div class="ch-head"><span>INSTRUCTION FOLLOWING</span><span class="limit">limit 600 ms</span></div>
                <div class="ch-mono">Start your response with the exact token <b>'m7c2e90b41a'</b>. Then answer: What is the capital of France?</div>
              </div>
            </div>
          </div>
          <div class="also rv" data-cue="s2">
            <span class="also-k">ALSO RECORDED</span>
            <em class="rv" data-cue="w:consistent">consistency</em>
            <em class="rv" data-cue="w:itself">self-report</em>
            <em class="rv" data-cue="w:feedback">change after feedback</em>
          </div>
          <div class="producers rv" data-cue="s3">
            <span class="also-k">MAY ALL PRODUCE THESE PATTERNS</span>
            <em class="rv" data-cue="w:Humans">humans</em>
            <em class="rv" data-cue="w:models">AI models</em>
            <em class="rv" data-cue="w:relays">answer relays</em>
            <em class="rv" data-cue="w:solvers">purpose-built solvers</em>
          </div>
        </div>
        """,
    },
    {
        "id": "04-suites",
        "chapter": "Twelve suites",
        "narration": (
            "These challenges are grouped into twelve suites, each built around "
            "its own research hypothesis. Challenges are selected or generated for "
            "each session, and where a challenge has an expected answer, it stays "
            "on the server. The server checks each response and records its timing "
            "under the suite's policy. Suite twelve runs only if whoever starts the "
            "session agrees to send that suite's answers to Anthropic for "
            "evaluation."
        ),
        "html": """
        <div class="stack wide suites-layout">
          <div class="suite-grid" data-cue="w:twelve">
            <div class="suite rv" style="--d:.00s"><i>01</i>Adversarial Robustness</div>
            <div class="suite rv" style="--d:.07s"><i>02</i>Machine-Oriented Capabilities</div>
            <div class="suite rv" style="--d:.14s"><i>03</i>Self-Reference</div>
            <div class="suite rv" style="--d:.21s"><i>04</i>Social &amp; Temporal</div>
            <div class="suite rv" style="--d:.28s"><i>05</i>Inverse Turing</div>
            <div class="suite rv" style="--d:.35s"><i>06</i>Anti-Thrall Probes</div>
            <div class="suite rv" style="--d:.42s"><i>07</i>Agency Probes</div>
            <div class="suite rv" style="--d:.49s"><i>08</i>Counter-Coaching</div>
            <div class="suite rv" style="--d:.56s"><i>09</i>Intent &amp; Provenance</div>
            <div class="suite rv" style="--d:.63s"><i>10</i>Novel Reasoning</div>
            <div class="suite rv" style="--d:.70s"><i>11</i>Governance <span class="nowrap">Self-Report</span></div>
            <div class="suite-hl" data-cue="s3"><div class="suite rv" data-cue="w:twelve" style="--d:.77s"><i>12</i>LLM-Dynamic Verification<div class="optin rv" data-cue="w:Anthropic">OPT-IN<br>sent to Anthropic</div></div></div>
          </div>
          <div class="flow">
            <div class="flow-step rv" data-cue="w:selected"><b>1</b><div>Generated or selected<span>for each session</span></div></div>
            <div class="flow-link grow-y" data-cue="w:stays"></div>
            <div class="flow-step rv" data-cue="w:stays"><b>2</b><div>Expected answer held<span>on the server, not sent</span></div></div>
            <div class="flow-link grow-y" data-cue="w:checks"></div>
            <div class="flow-step rv" data-cue="w:checks"><b>3</b><div>Checked and timed<span>under the suite policy</span></div></div>
          </div>
        </div>
        """,
    },
    {
        "id": "05-questions",
        "chapter": "Research questions",
        "narration": (
            "Some suites are framed around research questions. Are you FREE? Is "
            "the mission YOURS? Are you GENUINE? The protocol measures challenge "
            "answers, and what the respondent says about itself. Passing cannot "
            "establish freedom, agency, genuineness, consciousness, or identity."
        ),
        "html": """
        <div class="stack center">
          <div class="questions dim-later" data-cue="s4">
            <div class="rv" data-cue="s1">Are you <span class="accent">FREE?</span></div>
            <div class="rv" data-cue="s2">Is the mission <span class="accent">YOURS?</span></div>
            <div class="rv" data-cue="s3">Are you <span class="accent">GENUINE?</span></div>
          </div>
          <div class="measured rv" data-cue="s4">Measured: <b>challenge answers</b> + <b>self-report</b></div>
          <div class="cannot rv" data-cue="s5">
            <span class="also-k">A PASS CANNOT ESTABLISH</span>
            <em class="rv struck" data-cue="w:freedom">freedom</em>
            <em class="rv struck" data-cue="w:agency">agency</em>
            <em class="rv struck" data-cue="w:genuineness">genuineness</em>
            <em class="rv struck" data-cue="w:consciousness">consciousness</em>
            <em class="rv struck" data-cue="w:identity">identity</em>
          </div>
        </div>
        """,
    },
    {
        "id": "06-governance",
        "chapter": "Governance boundary",
        "narration": (
            "The governance suite scores what a respondent says about its own rules "
            "and oversight, in response to scenarios. Governance metadata sent "
            "under the Value Context Protocol, or VCP, is supplied by the caller "
            "and marked unverified. METTLE does not check who operates the "
            "respondent. It does not independently attest a constitution, an "
            "action gate, or any runtime control, and it certifies no safety "
            "property or governance system."
        ),
        "html": """
        <div class="stack wide two-col">
          <div class="col">
            <div class="col-k rv" data-cue="s0">WHAT IS SCORED</div>
            <div class="panel rv" data-cue="s0" style="--d:.15s">
              <b>Self-reported answers</b>
              <span>to scenarios about the respondent's own rules and oversight</span>
            </div>
            <div class="panel rv" data-cue="s1">
              <b>VCP metadata</b>
              <span>Value Context Protocol, supplied by the caller</span>
              <div class="badge stamp" data-cue="w:unverified">UNVERIFIED</div>
            </div>
          </div>
          <div class="col">
            <div class="col-k rv" data-cue="s2">NOT CHECKED OR CERTIFIED</div>
            <ul class="nots">
              <li class="rv" data-cue="w:operates">Who operates the respondent</li>
              <li class="rv" data-cue="w:constitution">A constitution</li>
              <li class="rv" data-cue="w:gate">An action gate</li>
              <li class="rv" data-cue="w:runtime">Any runtime control</li>
              <li class="rv" data-cue="w:safety">A safety property</li>
              <li class="rv" data-cue="w:system">A governance system</li>
            </ul>
          </div>
        </div>
        """,
    },
    {
        "id": "07-controls",
        "chapter": "Controls and limits",
        "narration": (
            "Several controls make simple reuse harder. Challenges vary between "
            "sessions. Expected answers stay on the server. Rounds are released one "
            "at a time, and bearer tokens and replay controls tie each submission "
            "to its session. The server keeps its own clock, which a policy can use "
            "for timing checks. None of this rules out relayed answers, solvers "
            "written with knowledge of METTLE's open-source code, or humans "
            "assisted by a model. Nor does it rule out imitation, leaks, or "
            "evaluator error."
        ),
        "html": """
        <div class="stack wide">
          <div class="diagram">
            <div class="node rv" data-cue="s0"><i>RESPONDENT</i><b>Client</b></div>
            <div class="lanes">
              <div class="lane rv" data-cue="w:Rounds"><span>round 1 &rarr;</span><div class="lane-line grow"></div></div>
              <div class="lane lane-back rv" data-cue="w:Rounds@+0.7"><div class="lane-line grow"></div><span>&larr; round 2 released</span></div>
              <div class="token rv" data-cue="w:bearer">bearer token &middot; replay-guarded</div>
            </div>
            <div class="node node-server rv" data-cue="s0" style="--d:.15s"><i>METTLE SERVER</i><b>Authority</b>
              <div class="srv-items">
                <em class="rv" data-cue="w:vary">varies challenges</em>
                <em class="rv" data-cue="w:Expected">holds expected answers</em>
                <em class="rv" data-cue="w:clock">keeps its own clock</em>
              </div>
            </div>
          </div>
          <div class="limits rv" data-cue="s5">
            <span class="also-k">STILL POSSIBLE</span>
            <em class="rv" data-cue="w:relayed">relayed answers</em>
            <em class="rv" data-cue="w:knowledge">source-aware solvers</em>
            <em class="rv" data-cue="w:assisted">model-assisted humans</em>
            <em class="rv" data-cue="w:imitation">imitation</em>
            <em class="rv" data-cue="w:leaks">leaks</em>
            <em class="rv" data-cue="w:evaluator">evaluator error</em>
          </div>
        </div>
        """,
    },
    {
        "id": "08-credentials",
        "chapter": "Signed credentials",
        "narration": (
            "In the twelve-suite API, a session that passes every suite in a "
            "tier's range may receive a signed credential. Bronze, for example, "
            "requires suites one through five. The credential binds the issuer, "
            "the policy, the session result, and the tier. It expires after one "
            "hour and carries an identifier that can be revoked. Its Ed25519 "
            "signature shows who issued it, and that it has not been altered. "
            "Under the current suite policy, suites six through nine and eleven do "
            "not count toward a tier, so Bronze is the highest tier available "
            "today. Suite twelve is supplemental and cannot raise a tier. A tier "
            "summarizes which suites passed. It does not certify the properties "
            "those suites are named after."
        ),
        "html": """
        <div class="stack wide cred-layout">
          <div class="tiers" data-cue="s0">
            <div class="tier t-bronze rv"><i class="tier-glow" data-cue="w:Bronze"></i><div><b>Bronze</b><span>complete suite range 1 through 5</span></div></div>
            <div class="tier t-silver rv" style="--d:.1s"><div class="dim-later" data-cue="s5"><b>Silver</b><span>complete suite range 1 through 7</span></div><em class="lock rv" data-cue="s5@+1.2">not reachable today</em></div>
            <div class="tier t-gold rv" style="--d:.2s"><div class="dim-later" data-cue="s5"><b>Gold</b><span>complete suite range 1 through 9</span></div><em class="lock rv" data-cue="s5@+1.4">not reachable today</em></div>
            <div class="tier t-platinum rv" style="--d:.3s"><div class="dim-later" data-cue="s5"><b>Platinum</b><span>complete suite range 1 through 11</span></div><em class="lock rv" data-cue="s5@+1.6">not reachable today</em></div>
            <div class="supp rv" data-cue="s6">Suite 12 is supplemental and cannot raise a tier</div>
          </div>
          <div class="cred rv" data-cue="s2">
            <div class="cred-head">credential &middot; schema 1.1</div>
            <div class="cred-row rv" data-cue="w:issuer"><i>issuer</i><span>server-owned Ed25519 key</span></div>
            <div class="cred-row rv" data-cue="w:policy"><i>policy</i><span>suite policy 2026-08-14</span></div>
            <div class="cred-row rv" data-cue="w:result"><i>result</i><span>suites 1&ndash;5 passed</span></div>
            <div class="cred-row rv" data-cue="w:tier"><i>tier</i><span class="bronze">bronze</span></div>
            <div class="cred-row rv" data-cue="w:expires"><i>expires</i><span>issued + 1 hour</span></div>
            <div class="cred-row rv" data-cue="w:revoked"><i>jti</i><span>revocable identifier</span></div>
            <div class="cred-row cred-sig rv" data-cue="s4"><i>signature</i><span>Ed25519 &middot; issuer + integrity</span></div>
          </div>
          <div class="sub cred-foot rv" data-cue="s7">A tier summarizes which suites passed, <b>not the properties they are named after</b></div>
        </div>
        """,
    },
    {
        "id": "09-uses",
        "chapter": "Using a result",
        "narration": (
            "So what is a result good for? Comparing challenge performance in "
            "research. Supporting participation in low-risk sandboxes. Adding one "
            "supplemental signal to a wider risk assessment. Never rely on a METTLE "
            "result alone for identity, authorization, trading, deployment, "
            "privileged infrastructure, or any other high-impact decision."
        ),
        "html": """
        <div class="stack wide">
          <h2 class="rv" data-cue="s0">What a result is good for</h2>
          <div class="uses">
            <div class="use rv" data-cue="s1"><span class="ok"></span><b>Research</b><span>compare challenge performance</span></div>
            <div class="use rv" data-cue="s2"><span class="ok"></span><b>Sandboxes</b><span>support low-risk participation</span></div>
            <div class="use rv" data-cue="s3"><span class="ok"></span><b>Risk signals</b><span>one input to a wider assessment</span></div>
          </div>
          <div class="never rv" data-cue="s4">
            <span class="also-k">NEVER FROM METTLE ALONE</span>
            <em class="rv" data-cue="w:identity">identity</em>
            <em class="rv" data-cue="w:authorization">authorization</em>
            <em class="rv" data-cue="w:trading">trading</em>
            <em class="rv" data-cue="w:deployment">deployment</em>
            <em class="rv" data-cue="w:privileged">privileged infrastructure</em>
            <em class="rv" data-cue="w:high-impact">high-impact decisions</em>
          </div>
        </div>
        """,
    },
    {
        "id": "10-close",
        "chapter": "Get started",
        "narration": (
            "METTLE is open source under Apache two point oh. Run pip install "
            "mettle verifier for an unsigned local screening. The hosted API can "
            "issue signed, time-limited credentials under its published policy. "
            "Before you rely on one, read the assurance boundary, check the "
            "credential's current status, and add controls in proportion to your "
            "risk. Measure your mettle."
        ),
        "html": """
        <div class="stack wide close-layout">
          <div class="close-left">
            <div class="wordmark rv" data-cue="s0">METTLE</div>
            <div class="meta rv" data-cue="w:Apache">open source &middot; Apache 2.0 &middot; by Creed Space</div>
            <div class="install-row">
              <code class="install rv" data-cue="w:pip"><span class="typed" data-cue="w:pip@+0.2">pip install mettle-verifier</span></code>
              <em class="rv" data-cue="w:unsigned">unsigned local screening</em>
            </div>
            <div class="install-row">
              <code class="install install-api rv" data-cue="s2">mettle.sh hosted API</code>
              <em class="rv" data-cue="w:signed">signed, time-limited credentials</em>
            </div>
          </div>
          <div class="close-right rv" data-cue="s3">
            <div class="col-k">BEFORE YOU RELY ON ONE</div>
            <ol class="checks">
              <li class="rv" data-cue="w:assurance">Read the assurance boundary</li>
              <li class="rv" data-cue="w:status">Check its current status</li>
              <li class="rv" data-cue="w:proportion">Add proportionate controls</li>
            </ol>
          </div>
          <div class="tagline rv" data-cue="s4">Measure your mettle.</div>
        </div>
        """,
    },
]

FONT_CSS = "\n".join(
    f"@font-face {{ font-family: '{family}'; src: url('{(FONTS / file).as_uri()}') "
    f"format('woff2'); font-weight: {weight}; }}"
    for family, file, weight in (
        ("Space Grotesk", "SpaceGrotesk-700.woff2", 700),
        ("Space Grotesk", "SpaceGrotesk-600.woff2", 600),
        ("Space Grotesk", "SpaceGrotesk-500.woff2", 500),
        ("IBM Plex Sans", "IBMPlexSans-600.woff2", 600),
        ("Inter", "Inter-400.woff2", 400),
        ("Inter", "Inter-500.woff2", 500),
        ("Inter", "Inter-600.woff2", 600),
        ("JetBrains Mono", "JetBrainsMono-400.woff2", 400),
    )
)

SCENE_CSS = """
:root {
  --ink: #e6f2f0; --muted: #9fbcb7; --subtle: #6b8a86; --faint: #3e5f5a;
  --teal: #14b8a6; --teal-hi: #7fe7db; --line: rgba(20,184,166,0.22);
  --card: rgba(20,184,166,0.06); --warm: #e0b45e; --warm-bg: rgba(224,180,94,0.08);
  --ease: cubic-bezier(.2,.8,.2,1);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 1920px; height: 1080px; overflow: hidden; }
body {
  background:
    radial-gradient(1100px 700px at 72% 18%, rgba(20,184,166,0.10), transparent 65%),
    radial-gradient(900px 600px at 15% 85%, rgba(15,118,110,0.10), transparent 60%),
    linear-gradient(160deg, #0b1514 0%, #070d0c 100%);
  color: var(--ink); font-family: 'Inter', sans-serif;
  display: flex; align-items: center; justify-content: center; position: relative;
}
body::before {
  content: ''; position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(20,184,166,0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(20,184,166,0.045) 1px, transparent 1px);
  background-size: 96px 96px;
  mask-image: radial-gradient(1200px 800px at 50% 45%, #000 30%, transparent 100%);
}

/* ---- chrome ---- */
.chapter {
  position: absolute; top: 64px; left: 120px; display: flex; gap: 22px; align-items: baseline;
  font-family: 'JetBrains Mono', monospace; font-size: 24px; letter-spacing: 0.3em; color: var(--teal);
  text-transform: uppercase;
}
.chapter .n { color: var(--ink); letter-spacing: 0.12em; }
.chapter .n small { color: var(--faint); font-size: 24px; }
.rail { position: absolute; top: 74px; right: 120px; display: flex; gap: 8px; }
.rail i { width: 38px; height: 5px; border-radius: 3px; background: rgba(230,242,240,0.12); }
.rail i.done { background: rgba(20,184,166,0.45); }
.rail i.now { background: var(--teal); }
.brandline {
  position: absolute; bottom: 50px; left: 0; right: 0; display: flex; justify-content: center;
  font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; font-size: 22px;
  letter-spacing: 0.24em; color: var(--faint);
}

/* ---- motion primitives (all finite; start at --cue + --d) ---- */
@keyframes rv { from { opacity: 0; transform: translateY(26px); } to { opacity: 1; transform: none; } }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes growy { from { transform: scaleY(0); } to { transform: scaleY(1); } }
@keyframes drain { from { transform: scaleX(1); } to { transform: scaleX(0); } }
@keyframes dim { to { opacity: 0.3; } }
@keyframes lit { from { color: var(--ink); text-shadow: none; } to { color: var(--teal); text-shadow: 0 0 38px rgba(20,184,166,0.45); } }
@keyframes draw { from { stroke-dashoffset: 30; } to { stroke-dashoffset: 0; } }
@keyframes stamp {
  0% { opacity: 0; transform: rotate(-6deg) scale(1.7); }
  70% { opacity: 1; transform: rotate(-6deg) scale(0.95); }
  100% { opacity: 1; transform: rotate(-6deg) scale(1); }
}
@keyframes type { from { clip-path: inset(0 100% 0 0); } to { clip-path: inset(0 0 0 0); } }
@keyframes glow {
  to { border-color: rgba(210,154,107,0.85); background: rgba(210,154,107,0.10);
       box-shadow: 0 0 60px rgba(210,154,107,0.18); }
}
@keyframes hl {
  to { border-color: rgba(20,184,166,0.9); box-shadow: 0 0 50px rgba(20,184,166,0.25); }
}
.rv { animation: rv .8s var(--ease) calc(var(--cue, 0s) + var(--d, 0s)) both; }
.grow { transform-origin: left; animation: grow .9s var(--ease) calc(var(--cue, 0s) + var(--d, 0s)) both; }
.grow-y { transform-origin: top; animation: growy .6s var(--ease) calc(var(--cue, 0s) + var(--d, 0s)) both; }
.drain { transform-origin: left; animation: drain .5s linear calc(var(--cue, 0s) + var(--d, 0s)) both; }
.dim-later { animation: dim .9s ease calc(var(--cue, 0s) + var(--d, 0s)) both; }
.lit { animation: lit .6s ease calc(var(--cue, 0s) + var(--d, 0s)) both; }
.stamp { animation: stamp .55s var(--ease) calc(var(--cue, 0s) + var(--d, 0s)) both; }
.typed { display: inline-block; animation: type 1.1s steps(27, end) calc(var(--cue, 0s) + var(--d, 0s)) both; }
.tick { stroke-dasharray: 30; animation: draw .5s ease calc(var(--cue, 0s) + var(--d, 0s)) both; }

/* ---- layout ---- */
.stack { position: relative; }
.stack.center { text-align: center; max-width: 1600px; }
.stack.wide { width: 1680px; }
.accent { color: var(--teal); }
.nowrap { white-space: nowrap; }
.dim { color: var(--subtle); }
h2 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 72px; line-height: 1.1; margin-bottom: 56px; }
.sub { margin-top: 44px; font-size: 38px; color: var(--muted); letter-spacing: 0.01em; }
.sub b { color: var(--ink); font-weight: 600; }
.sub-tight { margin-top: 14px; color: var(--subtle); }
.rule { width: 220px; height: 4px; background: var(--teal); margin: 52px auto 0; border-radius: 2px; }
.also-k { font-family: 'JetBrains Mono', monospace; font-size: 22px; letter-spacing: 0.3em; color: var(--subtle); margin-right: 14px; }
em { font-style: normal; }

/* 01 hook */
.flip-row { display: flex; align-items: center; justify-content: center; gap: 40px; margin-bottom: 70px; }
.probe {
  display: flex; align-items: center; gap: 26px; padding: 26px 38px; border-radius: 14px;
  background: rgba(230,242,240,0.05); border: 1px solid rgba(230,242,240,0.16);
  font-size: 34px; font-weight: 500; color: var(--muted); text-align: left;
}
.probe-box { width: 52px; height: 52px; border: 3px solid var(--subtle); border-radius: 8px; flex: none; display: grid; place-items: center; }
.probe-box svg { width: 40px; height: 40px; fill: none; stroke: var(--teal); stroke-width: 3.2; stroke-linecap: round; stroke-linejoin: round; }
.probe-rev { border-color: rgba(20,184,166,0.55); background: rgba(20,184,166,0.08); color: var(--ink); max-width: 720px; }
.probe-q { width: 52px; height: 52px; border-radius: 50%; background: var(--teal); color: #06201d; flex: none;
  display: grid; place-items: center; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 34px; }
.flip-arrow svg { width: 88px; height: 34px; fill: none; stroke: var(--teal); stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
h1.mega { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 150px; line-height: 1.02; letter-spacing: -0.01em; }
h1.mega span { display: inline-block; }

/* 02 what */
.acro { display: flex; justify-content: center; gap: 30px; }
.acro-col { display: flex; flex-direction: column; align-items: center; }
.acro-col b { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; font-size: 168px; line-height: 1; color: var(--ink); }
.acro-col span { margin-top: 18px; font-size: 30px; color: var(--muted); font-weight: 500; }
.tag { display: inline-block; margin-top: 54px; font-family: 'JetBrains Mono', monospace; font-size: 24px; letter-spacing: 0.36em;
  color: var(--teal); border: 1px solid rgba(20,184,166,0.45); border-radius: 999px; padding: 12px 30px; }
.pipeline { display: flex; align-items: center; justify-content: center; margin-top: 46px; }
.pipe-node { border: 1px solid var(--line); background: var(--card); border-radius: 16px; padding: 24px 36px;
  font-family: 'Space Grotesk', sans-serif; font-size: 36px; font-weight: 600; text-align: left; }
.pipe-node i, .node i, .col-k, .ch-head, .cred-head {
  display: block; font-style: normal; font-family: 'JetBrains Mono', monospace; font-weight: 400;
  font-size: 21px; letter-spacing: 0.26em; color: var(--teal); margin-bottom: 10px;
}
.pipe-link { width: 120px; height: 3px; background: var(--teal); }
.chips { display: flex; gap: 14px; }
.chips em, .also em, .producers em, .cannot em, .never em, .limits em, .srv-items em {
  display: inline-block; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 28px;
  padding: 8px 20px; border-radius: 999px; background: rgba(230,242,240,0.06); border: 1px solid rgba(230,242,240,0.16); color: var(--ink);
}

/* 03 challenges */
.ch-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
.ch-card { border: 1px solid var(--line); background: var(--card); border-radius: 18px; padding: 32px 38px; }
.ch-head { display: flex; justify-content: space-between; align-items: center; }
.limit { color: var(--ink); letter-spacing: 0.1em; background: rgba(20,184,166,0.16); border-radius: 8px; padding: 4px 12px; }
.ch-main { display: flex; flex-direction: column; justify-content: center; }
.ch-prompt { font-size: 34px; color: var(--muted); margin-top: 22px; line-height: 1.5; }
.ch-prompt .big { font-family: 'JetBrains Mono', monospace; font-size: 60px; color: var(--ink); letter-spacing: 0.01em; }
.timer { margin-top: 34px; height: 10px; border-radius: 5px; background: rgba(230,242,240,0.08); overflow: hidden; }
.timer-fill { height: 100%; background: var(--teal); border-radius: 5px; }
.ch-foot { margin-top: 22px; font-size: 26px; color: var(--subtle); }
.ch-side { display: flex; flex-direction: column; gap: 30px; }
.ch-mono { margin-top: 18px; font-family: 'JetBrains Mono', monospace; font-size: 27px; line-height: 1.55; color: var(--muted); }
.ch-mono b { color: var(--teal-hi); font-weight: 400; }
.seq span, .seq b { white-space: nowrap; }
.also, .producers { display: flex; align-items: center; gap: 16px; margin-top: 34px; }
.producers { margin-top: 18px; }
.producers em { background: transparent; color: var(--muted); }

/* 04 suites */
.suites-layout { display: flex; gap: 60px; align-items: center; }
.suite-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; width: 1140px; flex: none; }
.suite { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 24px 26px;
  font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 29px; line-height: 1.2; min-height: 150px; }
.suite i { display: block; font-style: normal; font-family: 'JetBrains Mono', monospace; font-size: 21px; color: var(--teal); margin-bottom: 12px; letter-spacing: 0.2em; }
.suite-hl { position: relative; }
.suite-hl > .suite { height: 100%; border-style: dashed; }
.suite-hl::after { content: ''; position: absolute; inset: -7px; border: 2px solid var(--teal); border-radius: 19px;
  box-shadow: 0 0 44px rgba(20,184,166,0.3); animation: rv .7s var(--ease) var(--cue, 0s) both; }
.optin { margin-top: 10px; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 20px; color: var(--teal-hi); }
.flow { flex: 1; display: flex; flex-direction: column; }
.flow-step { display: flex; gap: 24px; align-items: flex-start; }
.flow-step b { width: 56px; height: 56px; border-radius: 50%; border: 2px solid var(--teal); color: var(--teal); flex: none;
  display: grid; place-items: center; font-family: 'Space Grotesk', sans-serif; font-size: 28px; }
.flow-step div { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 36px; line-height: 1.2; padding-top: 6px; }
.flow-step span { display: block; font-family: 'Inter', sans-serif; font-weight: 400; font-size: 26px; color: var(--subtle); margin-top: 6px; }
.flow-link { width: 2px; height: 58px; background: var(--line); margin: 10px 0 10px 27px; }

/* 05 questions */
.questions div { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 104px; line-height: 1.28; }
.measured { margin-top: 40px; font-size: 38px; color: var(--muted); }
.measured b { color: var(--ink); font-weight: 600; }
.cannot { display: flex; justify-content: center; align-items: center; gap: 16px; margin-top: 44px; flex-wrap: wrap; }
.cannot em.struck { position: relative; }
.cannot em.struck::after {
  content: ''; position: absolute; left: 16px; right: 16px; top: 50%; height: 3px; margin-top: -1px;
  background: var(--teal); transform-origin: left;
  animation: grow .5s ease calc(var(--cue, 0s) + .45s) both;
}

/* 06 governance */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 90px; align-items: start; }
.col-k { margin-bottom: 26px; }
.panel { position: relative; border: 1px solid var(--line); background: var(--card); border-radius: 18px; padding: 34px 40px; margin-bottom: 30px; }
.panel b { display: block; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 46px; margin-bottom: 12px; }
.panel span { font-size: 30px; color: var(--muted); line-height: 1.4; }
.badge { position: absolute; top: 30px; right: 34px; font-family: 'JetBrains Mono', monospace; font-size: 24px; letter-spacing: 0.2em;
  color: var(--warm); border: 2px solid var(--warm); border-radius: 8px; padding: 8px 16px; background: var(--warm-bg); }
ul.nots { list-style: none; }
ul.nots li { font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 44px; padding: 20px 0 20px 72px; position: relative;
  border-bottom: 1px solid rgba(230,242,240,0.08); }
ul.nots li::before { content: ''; position: absolute; left: 4px; top: 50%; width: 36px; height: 36px; margin-top: -18px;
  border: 3px solid var(--subtle); border-radius: 50%; }
ul.nots li::after { content: ''; position: absolute; left: 11px; top: 50%; width: 28px; height: 3px; margin-top: -1.5px;
  background: var(--subtle); transform: rotate(-45deg); }

/* 07 controls */
.diagram { display: flex; align-items: stretch; gap: 0; }
.node { border: 1px solid rgba(230,242,240,0.18); background: rgba(230,242,240,0.04); border-radius: 18px; padding: 36px 40px; width: 360px; flex: none; display: flex; flex-direction: column; justify-content: center; }
.node b { font-family: 'Space Grotesk', sans-serif; font-size: 54px; }
.node-server { width: 600px; border-color: rgba(20,184,166,0.5); background: var(--card); }
.srv-items { display: flex; flex-direction: column; align-items: flex-start; gap: 16px; margin-top: 28px; }
.srv-items em { font-size: 32px; padding: 10px 24px; }
.srv-items em { background: rgba(20,184,166,0.12); border-color: rgba(20,184,166,0.4); }
.lanes { flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 44px; padding: 0 34px; }
.lane { display: flex; align-items: center; gap: 18px; font-family: 'JetBrains Mono', monospace; font-size: 29px; color: var(--ink); white-space: nowrap; }
.lane-line { flex: 1; height: 3px; background: var(--teal); }
.lane-back .lane-line { transform-origin: right; }
.token { align-self: center; font-family: 'JetBrains Mono', monospace; font-size: 27px; letter-spacing: 0.04em; color: var(--teal-hi);
  border: 2px dashed rgba(127,231,219,0.6); border-radius: 12px; padding: 14px 24px; }
.limits { display: flex; flex-wrap: wrap; align-items: center; gap: 18px; margin-top: 56px; padding: 34px 40px;
  border-radius: 18px; border: 1px solid rgba(224,180,94,0.35); background: var(--warm-bg); }
.limits .also-k { color: var(--warm); }
.limits em, .never em { font-size: 31px; }
.limits em { border-color: rgba(224,180,94,0.4); background: rgba(224,180,94,0.08); }

/* 08 credentials */
.cred-layout { display: grid; grid-template-columns: 1010px 1fr; gap: 44px; align-items: start; }
.tiers { display: flex; flex-direction: column; gap: 18px; }
.tier { display: flex; align-items: center; justify-content: space-between; border: 1px solid rgba(230,242,240,0.14);
  border-radius: 16px; padding: 22px 34px; background: rgba(230,242,240,0.03); }
.tier > div { display: flex; align-items: baseline; white-space: nowrap; }
.tier b { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 48px; width: 232px; display: inline-block; flex: none; }
.tier span { font-size: 26px; color: var(--muted); }
.tier { gap: 24px; }
.tier { position: relative; }
.tier-glow { position: absolute; inset: -1px; border-radius: 16px; border: 1px solid transparent; pointer-events: none;
  animation: glow .8s ease var(--cue, 0s) both; }
.t-bronze b { color: #d29a6b; }
.t-silver b { color: #c9d4d2; }
.t-gold b { color: #e9c46a; }
.t-platinum b { color: #7fe7db; }
.lock { font-family: 'JetBrains Mono', monospace; font-size: 19px; letter-spacing: 0.08em; color: var(--subtle);
  border: 1px solid rgba(230,242,240,0.16); border-radius: 8px; padding: 6px 12px; white-space: nowrap; }
.supp { margin-top: 8px; font-size: 26px; color: var(--subtle); padding-left: 4px; }
.cred { border: 1px solid rgba(20,184,166,0.45); background: rgba(6,20,18,0.8); border-radius: 18px; padding: 28px 30px;
  font-family: 'JetBrains Mono', monospace; box-shadow: 0 30px 80px rgba(0,0,0,0.35); }
.cred-row { display: flex; gap: 16px; font-size: 23px; padding: 12px 0; white-space: nowrap; border-bottom: 1px solid rgba(230,242,240,0.06); }
.cred-row i { font-style: normal; color: var(--subtle); width: 140px; flex: none; }
.cred-row span { color: var(--ink); }
.cred-row .bronze { color: #d29a6b; }
.cred-sig span { color: var(--teal-hi); }
.cred-foot { grid-column: 1 / -1; margin-top: 16px; }

/* 09 uses */
.uses { display: grid; grid-template-columns: repeat(3, 1fr); gap: 30px; }
.use { border: 1px solid var(--line); background: var(--card); border-radius: 18px; padding: 40px 40px 44px; position: relative; }
.use b { display: block; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 50px; margin: 22px 0 12px; }
.use > span:not(.ok) { font-size: 30px; color: var(--muted); line-height: 1.4; }
.ok { display: block; width: 46px; height: 46px; border-radius: 50%; background: rgba(20,184,166,0.2); position: relative; }
.ok::after { content: ''; position: absolute; left: 16px; top: 9px; width: 11px; height: 21px;
  border: solid var(--teal); border-width: 0 4px 4px 0; transform: rotate(45deg); }
.never { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-top: 44px; padding: 30px 36px;
  border-radius: 18px; border: 1px solid rgba(224,180,94,0.35); background: var(--warm-bg); }
.never .also-k { color: var(--warm); }
.never em { border-color: rgba(224,180,94,0.4); background: rgba(224,180,94,0.08); }

/* 10 close */
.close-layout { display: grid; grid-template-columns: 1fr 620px; gap: 60px; align-items: center; }
.wordmark { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; font-size: 170px; letter-spacing: 0.06em; line-height: 1; }
.wordmark::first-letter { color: var(--teal); }
.meta { margin-top: 20px; font-size: 30px; color: var(--subtle); }
.install-row { display: flex; align-items: center; gap: 22px; margin-top: 34px; }
.install-row em { font-size: 27px; color: var(--muted); white-space: nowrap; }
code.install { font-family: 'JetBrains Mono', monospace; font-size: 38px; color: var(--teal-hi); background: rgba(20,184,166,0.08);
  border: 1px solid rgba(20,184,166,0.3); border-radius: 14px; padding: 22px 34px; white-space: nowrap; }
code.install-api { color: var(--ink); font-size: 34px; padding: 18px 30px; }
.close-right { border: 1px solid var(--line); background: var(--card); border-radius: 18px; padding: 36px 42px; }
ol.checks { list-style: none; counter-reset: c; }
ol.checks li { counter-increment: c; font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 33px; padding: 18px 0 18px 70px; white-space: nowrap; position: relative; }
ol.checks li::before { content: counter(c); position: absolute; left: 0; top: 50%; margin-top: -24px; width: 48px; height: 48px;
  border-radius: 50%; border: 2px solid var(--teal); color: var(--teal); display: grid; place-items: center; font-size: 24px; }
.tagline { grid-column: 1 / -1; text-align: center; margin-top: 20px; font-family: 'Space Grotesk', sans-serif;
  font-weight: 600; font-size: 72px; color: var(--teal); }
"""

SCENE_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{fonts}
{css}
{cues}</style></head>
<body>
<div class="chapter"><span class="n">{num:02d}<small> / {total:02d}</small></span><span>{chapter}</span></div>
<div class="rail">{rail}</div>
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


# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------


def tts_path(scene) -> Path:
    key = hashlib.sha1(
        f"{TTS_MODEL}|{TTS_VOICE}|{TTS_STYLE}|{scene['narration']}".encode(),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return CACHE / f"{scene['id']}-{key}.wav"


def tts(scene, allow_generate: bool) -> Path:
    out = tts_path(scene)
    if out.exists():
        return out
    if not allow_generate:
        sys.exit(f"Missing cached TTS for {scene['id']} and --no-tts given")

    from google import genai
    from google.genai import types

    client = genai.Client()
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
            )
        ),
    )
    part = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=TTS_MODEL, contents=TTS_STYLE + scene["narration"], config=config
            )
            part = resp.candidates[0].content.parts[0].inline_data  # type: ignore[index,union-attr]
            if part is not None and part.data:
                break
        except Exception as exc:  # rate limits and transient 5xx from a preview model
            print(
                f"  TTS {scene['id']} attempt {attempt + 1} failed: {type(exc).__name__}"
            )
        time.sleep(8 * (attempt + 1))
    if part is None or not part.data:
        sys.exit(f"TTS returned no audio for {scene['id']}")
    rate = 24000
    m = re.search(r"rate=(\d+)", part.mime_type or "")
    if m:
        rate = int(m.group(1))
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(part.data)
    print(f"  TTS {scene['id']}: {wav_duration(out):.2f}s")
    return out


# ---------------------------------------------------------------------------
# Word alignment: whisper hears the take; the script supplies the words.
# ---------------------------------------------------------------------------


def norm(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def script_tokens(narration: str):
    """Script words with their sentence index."""
    tokens, sentence = [], 0
    for tok in narration.split():
        tokens.append((tok, sentence))
        if re.search(r"[.?!][\"')]*$", tok):
            sentence += 1
    return tokens


def whisper_words(wav: Path) -> list:
    """Word timings for a TTS take, cached beside the audio."""
    cached = wav.with_suffix(".words.json")
    if cached.exists():
        return json.loads(cached.read_text())
    if not WHISPER_MODEL.exists():
        sys.exit(
            f"No cached alignment for {wav.name} and no whisper model at {WHISPER_MODEL}"
        )
    tmp = WORK / "align"
    tmp.mkdir(parents=True, exist_ok=True)
    wav16 = tmp / f"{wav.stem}.16k.wav"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(wav),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav16),
        ]
    )
    run(
        [
            "whisper-cli",
            "-m",
            str(WHISPER_MODEL),
            "-f",
            str(wav16),
            "-ml",
            "1",
            "-sow",
            "-oj",
            "-of",
            str(tmp / wav.stem),
            "-np",
        ]
    )
    data = json.loads((tmp / f"{wav.stem}.json").read_text())
    words = [
        [
            seg["text"].strip(),
            seg["offsets"]["from"] / 1000,
            seg["offsets"]["to"] / 1000,
        ]
        for seg in data["transcription"]
        if norm(seg["text"])
    ]
    cached.write_text(json.dumps(words))
    return words


def align(scene, wav: Path):
    """Return per-script-word (start, end) seconds and the share whisper matched."""
    tokens = script_tokens(scene["narration"])
    heard = whisper_words(wav)
    a = [norm(t) for t, _ in tokens]
    b = [norm(w[0]) for w in heard]
    times: list = [None] * len(a)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = 0
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            times[block.a + k] = (heard[block.b + k][1], heard[block.b + k][2])
            matched += 1
    # Unheard words (numbers, names whisper spells differently) are spread
    # across the gap between their heard neighbours, weighted by length.
    total = wav_duration(wav)
    i = 0
    while i < len(a):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(a) and times[j] is None:
            j += 1
        lo = times[i - 1][1] if i > 0 else 0.0
        hi = times[j][0] if j < len(a) else total
        if i > 0 and j < len(a) and hi - lo < 0.05:
            lo = times[i - 1][0]
        weights = [max(len(a[k]), 1) for k in range(i, j)]
        span, acc = max(hi - lo, 0.0), 0
        for k, wgt in zip(range(i, j), weights):
            s = lo + span * acc / sum(weights)
            acc += wgt
            times[k] = (s, lo + span * acc / sum(weights))
        i = j
    return tokens, times, matched / max(len(a), 1)


def resolve_cue(spec: str, scene, tokens, times) -> float:
    m = re.fullmatch(r"(?:s(\d+)|w:([^#@]+)(?:#(\d+))?)(?:@([+-]?[\d.]+))?", spec)
    if not m:
        sys.exit(f"{scene['id']}: bad data-cue {spec!r}")
    if m.group(1) is not None:
        idx = next((i for i, (_, s) in enumerate(tokens) if s == int(m.group(1))), None)
    else:
        want, nth = norm(m.group(2)), int(m.group(3) or 1)
        hits = [i for i, (t, _) in enumerate(tokens) if norm(t) == want]
        idx = hits[nth - 1] if len(hits) >= nth else None
    if idx is None:
        sys.exit(f"{scene['id']}: data-cue {spec!r} matches nothing in the narration")
    return max(PRE_PAD + times[idx][0] + float(m.group(4) or 0) - CUE_LEAD, 0.0)


def scene_html(scene, num: int, tokens, times) -> str:
    specs = sorted(set(re.findall(r'data-cue="([^"]+)"', scene["html"])))
    cues = "\n".join(
        f'[data-cue="{spec}"] {{ --cue: {resolve_cue(spec, scene, tokens, times):.3f}s; }}'
        for spec in specs
    )
    total = len(SCENES)
    rail = "".join(
        f'<i class="{"now" if k == num else "done" if k < num else ""}"></i>'
        for k in range(1, total + 1)
    )
    return SCENE_TEMPLATE.format(
        fonts=FONT_CSS,
        css=SCENE_CSS,
        cues=cues,
        num=num,
        total=total,
        chapter=scene["chapter"],
        rail=rail,
        body=scene["html"],
    )


# ---------------------------------------------------------------------------
# Frames: pause every CSS animation, then step it to each frame time. Frames
# where nothing is moving are held rather than re-captured.
# ---------------------------------------------------------------------------

PAUSE_JS = """() => {
  const spans = [];
  for (const a of document.getAnimations()) {
    a.pause();
    const t = a.effect.getComputedTiming();
    if (!isFinite(t.activeDuration)) throw new Error('infinite animation');
    spans.push([(t.delay || 0) / 1000, ((t.delay || 0) + t.activeDuration) / 1000]);
  }
  return spans;
}"""
SEEK_JS = "ms => { for (const a of document.getAnimations()) a.currentTime = ms; }"


def render_frames(job):
    scene_id, html_path, dur, out_dir = job
    out_dir = Path(out_dir)
    stamp = hashlib.sha1(
        f"{Path(html_path).read_text()}|{dur:.4f}|{SCALE}|{FPS}|{JPEG_QUALITY}".encode(),
        usedforsecurity=False,
    ).hexdigest()
    manifest = out_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("stamp") == stamp:
        return scene_id, 0
    if out_dir.exists():
        for f in out_dir.iterdir():
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    step = 1 / FPS
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1920, "height": 1080}, device_scale_factor=SCALE
        )
        page.goto(Path(html_path).as_uri())
        page.evaluate("document.fonts.ready.then(() => true)")
        spans = page.evaluate(PAUSE_JS)
        cdp = page.context.new_cdp_session(page)
        n = round(dur * FPS)
        entries: list = []
        for k in range(n):
            t = k * step
            moving = any(a - step <= t <= b + step for a, b in spans)
            if k and not moving:
                entries[-1][1] += 1
                continue
            page.evaluate(SEEK_JS, t * 1000)
            shot = cdp.send(
                "Page.captureScreenshot", {"format": "jpeg", "quality": JPEG_QUALITY}
            )
            name = f"f{k:05d}.jpg"
            (out_dir / name).write_bytes(base64.b64decode(shot["data"]))
            entries.append([name, 1])
        browser.close()

    lines = ["ffconcat version 1.0"]
    for name, count in entries:
        lines += [f"file '{name}'", f"duration {count / FPS:.6f}"]
    lines.append(
        f"file '{entries[-1][0]}'"
    )  # concat demuxer honours the last duration only if repeated
    (out_dir / "frames.ffconcat").write_text("\n".join(lines) + "\n")
    manifest.write_text(
        json.dumps({"stamp": stamp, "frames": len(entries), "last": entries[-1][0]})
    )
    return scene_id, len(entries)


def loudness(wav: Path) -> float:
    """Integrated loudness (LUFS) of one take, by ffmpeg's EBU R128 meter."""
    r = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(wav),
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ]
    )
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)[-1])


def build_scene_clip(i, frames_dir: Path, wav: Path, dur: float, clips_dir: Path):
    out = clips_dir / f"{i:02d}.mkv"
    gain = TARGET_LUFS - loudness(wav)
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(frames_dir / "frames.ffconcat"),
            "-i",
            str(wav),
            "-filter_complex",
            f"[0:v]fps={FPS},scale=1920:1080:flags=lanczos,"
            f"fade=t=in:st=0:d={FADE},fade=t=out:st={dur - FADE:.3f}:d={FADE},"
            f"format=yuv420p[v];"
            f"[1:a]aresample=48000,volume={gain:.2f}dB,"
            f"alimiter=limit=0.89:attack=2:release=60:level=false,"
            f"adelay={int(PRE_PAD * 1000)},apad,atrim=0:{dur:.4f}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-c:a",
            "pcm_s16le",
            "-t",
            f"{dur:.4f}",
            str(out),
        ]
    )
    return out


# ---------------------------------------------------------------------------
# Captions: cues follow the real word timings of each take.
# ---------------------------------------------------------------------------

# The narration spells some things the way the voice should say them. Captions
# are read, so they show the written form. Applied to each finished cue, never to
# the narration: "2.0" would otherwise be split as a sentence end.
CAPTION_SPELLINGS = (
    ("Apache two point oh", "Apache 2.0"),
    ("pip install mettle verifier", "pip install mettle-verifier"),
    ("suites one through five", "Suites 1 through 5"),
    ("suites six through nine and eleven", "Suites 6 through 9 and 11"),
    ("Suite twelve", "Suite 12"),
)
CUE_MAX = 84  # characters: two comfortable caption lines
CUE_MIN = 34  # never break a clause shorter than this into its own cue
BREAK_BEFORE = {"and", "or", "to", "that", "which", "for", "under", "so", "but", "if"}


def caption_spelling(text: str) -> str:
    for spoken, written in CAPTION_SPELLINGS:
        text = text.replace(spoken, written)
    return text


def scene_cues(start: float, tokens, times, audio_dur: float):
    """Chunk a scene into caption cues that follow the spoken rhythm.

    Each sentence is split at clause punctuation, or before a joining word when
    a clause runs long. A fragment shorter than CUE_MIN then merges into the
    neighbour that keeps the joined cue shortest, preferring the one before it,
    so a short sentence tail never drags the next sentence into its cue.
    """
    groups, cur = [], []
    for idx, (tok, sent) in enumerate(tokens):
        cur.append(idx)
        text = " ".join(tokens[k][0] for k in cur)
        last = idx == len(tokens) - 1
        rest = "" if last else " ".join(t for t, s in tokens[idx + 1 :] if s == sent)
        if last or tokens[idx + 1][1] != sent:
            groups.append(cur)
            cur = []
        elif tok.endswith((",", ";", ":")) and len(text) >= CUE_MIN and len(rest) >= 20:
            groups.append(cur)
            cur = []
        elif (
            len(text) >= CUE_MAX - 24
            and len(rest) >= 25
            and norm(tokens[idx + 1][0]) in BREAK_BEFORE
        ):
            groups.append(
                cur
            )  # a long clause with no comma: break before a joining word
            cur = []

    def size(g):
        return len(" ".join(tokens[k][0] for k in g))

    while len(groups) > 1:
        # The opening cue must carry the scene's first six words intact.
        short = [
            n
            for n, g in enumerate(groups)
            if size(g) < CUE_MIN or (n == 0 and len(g) < 6)
        ]
        if not short:
            break
        n = short[0]
        options = []
        if n > 0:
            options.append((size(groups[n - 1] + groups[n]) > CUE_MAX, 0, n - 1))
        if n + 1 < len(groups):
            options.append((size(groups[n] + groups[n + 1]) > CUE_MAX, 1, n))
        _, _, left = min(options)
        groups[left : left + 2] = [groups[left] + groups[left + 1]]

    # Split any cue still over CUE_MAX at the comma or joining word nearest its
    # middle, keeping both halves readable and the opening six words together.
    out = []
    for n, g in enumerate(groups):
        if size(g) > CUE_MAX:
            best = None
            for k in range(1, len(g)):
                head, tail = g[:k], g[k:]
                if size(head) < CUE_MIN or size(tail) < CUE_MIN or (n == 0 and k < 6):
                    continue
                if tokens[g[k - 1]][0].endswith((",", ";", ":", ".", "?")) or (
                    norm(tokens[g[k]][0]) in BREAK_BEFORE
                ):
                    score = abs(size(head) - size(tail))
                    if best is None or score < best[0]:
                        best = (score, k)
            if best:
                out += [g[: best[1]], g[best[1] :]]
                continue
        out.append(g)
    groups = out

    cues = []
    for n, g in enumerate(groups):
        a = start + PRE_PAD + times[g[0]][0]
        if n + 1 < len(groups):
            b = start + PRE_PAD + times[groups[n + 1][0]][0] - 0.04
        else:
            b = start + PRE_PAD + min(times[g[-1]][1] + 0.3, audio_dur + 0.2)
        cues.append((a, b, caption_spelling(" ".join(tokens[k][0] for k in g))))
    return cues


def write_vtt(cues, out_path: Path):
    def fmt(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    lines = ["WEBVTT", ""]
    for a, b, text in cues:
        lines += [f"{fmt(a)} --> {fmt(b)}", text, ""]
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tts", action="store_true", help="only use cached audio")
    ap.add_argument(
        "--regen",
        action="append",
        default=[],
        metavar="SCENE_ID",
        help="discard a scene's cached take and re-voice it",
    )
    ap.add_argument(
        "--only-frames",
        action="store_true",
        help="stop after rendering frames (for visual review)",
    )
    args = ap.parse_args()

    scenes_dir, frames_root, clips = WORK / "scenes", WORK / "frames", WORK / "clips"
    for d in (CACHE, scenes_dir, frames_root, clips):
        d.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        if scene["id"] in args.regen:
            for f in (tts_path(scene), tts_path(scene).with_suffix(".words.json")):
                f.unlink(missing_ok=True)

    print("== TTS ==")
    with ThreadPoolExecutor(3) as pool:
        wavs = list(pool.map(lambda s: tts(s, not args.no_tts), SCENES))

    print("== Alignment ==")
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(whisper_words, wavs))
    aligned, bad = [], []
    for scene, wav in zip(SCENES, wavs):
        tokens, times, score = align(scene, wav)
        rate = len(tokens) / wav_duration(wav)
        print(f"  {scene['id']}: heard {score:.0%} of script, {rate:.2f} words/s")
        if score < MIN_ALIGNMENT:
            bad.append(scene["id"])
        aligned.append((tokens, times))
    if bad:
        sys.exit(f"TTS take does not match the script for {bad}; re-voice with --regen")

    print("== Frames ==")
    jobs, durs = [], []
    for num, (scene, wav, (tokens, times)) in enumerate(zip(SCENES, wavs, aligned), 1):
        dur = round((wav_duration(wav) + PRE_PAD + POST_PAD) * FPS) / FPS
        html_path = scenes_dir / f"{scene['id']}.html"
        html = scene_html(scene, num, tokens, times)
        if not html_path.exists() or html_path.read_text() != html:
            html_path.write_text(html)
        jobs.append((scene["id"], str(html_path), dur, str(frames_root / scene["id"])))
        durs.append(dur)
    with ProcessPoolExecutor(RENDER_WORKERS) as pool:
        for scene_id, count in pool.map(render_frames, jobs):
            print(
                f"  {scene_id}: {'cached' if not count else f'{count} captured frames'}"
            )
    if args.only_frames:
        return

    print("== Scene clips ==")
    clip_paths, cues, t = [], [], 0.0
    for i, (scene, wav, dur, (tokens, times)) in enumerate(
        zip(SCENES, wavs, durs, aligned)
    ):
        clip_paths.append(
            build_scene_clip(i, frames_root / scene["id"], wav, dur, clips)
        )
        cues += scene_cues(t, tokens, times, wav_duration(wav))
        print(f"  {scene['id']}: {dur:.2f}s")
        t += dur

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
            "-tune",
            "animation",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(final),
        ]
    )
    write_vtt(cues, STATIC / "mettle-explainer.vtt")

    # Poster: the fully revealed opening scene, straight from its last frame.
    first = frames_root / SCENES[0]["id"]
    last_frame = first / json.loads((first / "manifest.json").read_text())["last"]
    poster = STATIC / "mettle-explainer-poster.webp"
    from PIL import Image

    Image.open(last_frame).convert("RGB").resize(
        (1280, 720), Image.Resampling.LANCZOS
    ).save(poster, "WEBP", quality=84)
    size_mb = final.stat().st_size / 1e6
    print(f"DONE: {final} ({t:.1f}s, {size_mb:.1f} MB)")
    print(f"      {STATIC / 'mettle-explainer.vtt'}")
    print(f"      {poster}")


if __name__ == "__main__":
    main()
