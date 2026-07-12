# METTLE Explainer Video Pipeline

Builds `static/mettle-explainer.mp4` (3:38, 1080p30) plus `mettle-explainer.vtt`
captions and `mettle-explainer-poster.webp`.

The SaferAgenticAI and Psychopathia explainers used the same recipe (Gemini
Flash TTS narration over branded motion slides) but their pipelines were never
committed. This one is: script, slide design, and assembly all live in
`build.py`, so the video can be regenerated or re-scripted at any time.

## Pipeline

1. **Narration** — scene text in `build.py` `SCENES`, spoken by Gemini
   `gemini-3.1-flash-tts-preview`, voice **Sadaltager**, British documentary
   style prompt. Audio is cached in `tts-cache/` keyed by a hash of
   model+voice+style+text, so unchanged scenes never re-bill.
2. **Slides** — each scene has an HTML body rendered against the shared METTLE
   dark/teal CSS (site fonts from `static/fonts/`), screenshotted at 3840x2160
   by headless Chrome into `build/slides/`.
3. **Assembly** — per scene: static slide (no zoompan; sub-pixel zoom jitters
   on crisp text), 0.4 s fade to black at each cut, narration padded
   0.45 s / 0.55 s; ffmpeg concat, x264 crf 22, AAC 160k, faststart. Captions
   are generated from the scene timeline with sentence-proportional timing.

## Usage

```bash
export GOOGLE_API_KEY=...   # only needed for uncached narration
python3 video/build.py            # full build
python3 video/build.py --no-tts   # refuse to spend TTS quota; cache only
```

Requires ffmpeg, Google Chrome, `google-genai`, Pillow (poster webp).

Editing a scene's narration re-generates only that scene's audio; editing its
HTML re-renders only that slide. `build/` is disposable; `tts-cache/` is kept
in git so rebuilds are deterministic and don't re-spend TTS quota.
