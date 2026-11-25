# METTLE explainer video

The canonical cinematic remaster is in [`remaster/`](remaster/README.md).
Its grounded narration and 16:9 production brief are in `remaster/brief.json`.
The local 1080p30 film is approximately 6 minutes 11 seconds, with Gemini 3.8 expressive
professor narration, Sunburst reference images, Veo/Seedance motion, native
Manim/Motion Canvas graphics, an actual Hyperframes brand bridge and an original
Lyria instrumental score. ElevenLabs music access returned 401; the user approved
OpenRouter fallback.

Final captions derive from actual measured MAI-Transcribe 2 word timings and
reviewed spelling adjudications. The portable local bundle retains media,
reference originals, provider receipts, caption data and a read-back-validated
OpenTimelineIO edit. Delivery and acceptance evidence are under
`video/build/remaster/`; retain the bundle before cleaning intermediate media.
No publication or deployment is included.

Working if: the exact film and captions pass their source/audio hash checks,
complete decode, visual/motion/audio review and local static-site gates.

## Historical pipeline

`build.py` and its cached earlier Gemini 3.1 narration remain for provenance.
That builder now writes only to `video/build/legacy-export/`, preserving the
accepted website film. Its older recipe and usage follow.

The SaferAgenticAI and Psychopathia explainers used the same recipe (Gemini
Flash TTS narration over branded motion slides) but their pipelines were never
committed. This one is: script, slide design, and assembly all live in
`build.py`, so the video can be regenerated or re-scripted at any time.

## Pipeline

1. **Narration**: scene text in `build.py` `SCENES`, spoken by Gemini
   `gemini-3.1-flash-tts-preview`, voice **Sadaltager**, British documentary
   style prompt. Audio is cached in `tts-cache/` keyed by a hash of
   model+voice+style+text, so unchanged scenes never re-bill.
2. **Alignment and QA**: `whisper-cli` (ggml `base.en`) transcribes each take
   with word timestamps, cached beside the audio as `*.words.json`. The script
   is aligned to what was heard; a take in which whisper hears less than 80% of
   the script fails the build (re-voice it with `--regen <scene-id>`).
3. **Scenes**: each scene is an HTML body on the shared METTLE dark/teal CSS
   (site fonts from `static/fonts/`) with a chapter label and progress rail.
   Elements carry `data-cue` (`s3` = fourth sentence, `w:session` = first time
   "session" is spoken, `@+0.4` = offset), so every reveal lands on the word
   that introduces it.
4. **Frames**: Playwright's headless Chromium pauses every CSS animation and
   steps it to each frame time, capturing 3840x2160 JPEGs only while something
   moves; still stretches are held, not re-captured. Scenes render in parallel.
5. **Assembly**: per scene, frames downsampled with lanczos, 0.35 s fade at
   each cut, narration padded 0.5 s / 0.9 s; ffmpeg concat, x264 crf 22
   (`-tune animation`), AAC 160k, faststart. Captions are cut from the real
   word timings, breaking only at punctuation or before a joining word. The
   poster is the fully revealed opening scene.

## Usage

```bash
export GOOGLE_API_KEY=...   # only needed for uncached narration
python3 video/build.py                       # full build
python3 video/build.py --no-tts              # refuse to spend TTS quota; cache only
python3 video/build.py --only-frames         # stop after frames, for visual review
python3 video/build.py --regen 05-questions  # re-voice one scene
```

Requires ffmpeg, `google-genai`, Python Playwright with its Chromium, Pillow,
and (for uncached alignment only) `whisper-cli` with
`~/.cache/whisper-models/ggml-base.en.bin` (override with
`METTLE_WHISPER_MODEL`).

Editing a scene's narration re-voices and re-aligns only that scene; editing
its HTML re-renders only its frames. `build/` is disposable; `tts-cache/`
(audio and word timings) is kept in git so rebuilds are deterministic and
don't re-spend TTS quota. After a render, update the transcript in
`static/index.html` if the narration changed, then run
`python3 scripts/update_asset_fingerprints.py`.
