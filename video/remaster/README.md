# METTLE explainer remaster

The accepted local film is 1920 × 1080, 30fps and 6 minutes 11 seconds, with
clean and burned-caption variants. The local site video, poster, measured caption
track and full transcript have been updated. Original site assets are retained.
The production brief and professor-style narration are in `brief.json` and
`narration.md`. No publication, credit purchase or deployment occurred.

Exact acceptance and local gates are under `video/build/remaster/review/`.
The portable project and films are in `video/build/remaster/delivery/`; the
archive is `video/build/remaster/METTLE-remaster-local.zip`. Confirmed task charges
are $8.44, with a separate $2 reserve for unavailable failed-job accounting.
There are 125 readable cues from 737 positive measured word intervals. Full
decoding, complete-film critique, 27 caption/documentation tests, and local
browser accessibility, performance and playback checks passed.

## Production

* `provider.py`: OpenRouter Gemini 3.8 TTS with delivery metadata separate from
  the spoken script, Sunburst reference-conditioned keyframes, recoverable
  Veo/Seedance jobs, measured ASR word timings and audiovisual review.
  The OpenAI SDK has automatic retries disabled. A submitted receipt without a
  result is a recovery hold, not permission to pay for a duplicate.
* `graphics/`: deterministic HTML/GSAP compositions, an actual Hyperframes
  engine-rendered brand bridge, and Motion Canvas constellation and multi-round
  flow scenes.
* `math_scene.py`: Manim illustration of a quick-API arithmetic challenge.
* `assemble.py`: frame-bound edit, explicit narration placement, a ducked
  instrumental bed, FFmpeg encoding and a read-back-validated OpenTimelineIO
  timeline. Scratch narration is distinguished from the final script.
* `captions.py`: actual ASR word timing, source hash validation, phrase VTT/SRT
  and unchanged measured word data.

The four reusable video skills (script planning, expressive TTS, captions and
quality critic) were already installed in Codex and are used here rather than
duplicated. Their project-specific settings are in the production brief.

Working if: a reviewed animatic precedes full video generation, narration matches
the final script, all references and provider receipts are retained, genuine word
timestamps map to the final audio, and the exact export decodes successfully.

## Credentials and spend

Read credentials only from Keychain at point of use, never from repository
files or chat. No key is included in this package. The user authorised up to
$100 of OpenRouter use; the existing general key had $41.90 remaining, so this
production sets an effective $40 ceiling. `video/build/remaster/spend-ledger.json`
records both confirmed task receipts and shared account changes, with unresolved
billing held separately. Concurrent key use is excluded from the task cost.
ElevenLabs music returned HTTP 401. The user authorised an OpenRouter fallback:
Lyria 3 Pro generated two instrumental cues, joined with a 6-second crossfade. No
credit purchase or account mutation was needed. Task receipt costs are separated
from concurrent account/key use.

Working if: no raw credential reaches a receipt or log and paid calls stop at
the effective ceiling, a credential hold or an unresolved submission.

## Runtime and outputs

Intermediate media and full review evidence are under `video/build/remaster/`,
which is ignored by Git. Local outputs include the animatic, final master,
captioned variant, caption files, word timestamps and `.otio` edit.

The current machine has a usable read-only Manim/OpenTimelineIO Python runtime
and Node render runtime from the prior AI Guardians remaster. Exact Node package
versions and paths are in `graphics/dependencies.json`. These are tooling
resources, not a dependency of the METTLE service. The Python generator needs
OpenAI, httpx and Pillow; the media tools need ffmpeg/ffprobe. A future portable
checkout should provision isolated video tooling, leaving application locks
unchanged.

The animatic uses the new Gemini narration once all takes exist. A cache-only
preflight can use the previous Gemini narration as temporary audio.
Its receipt identifies that scratch source explicitly; it never represents the
new script as voiced or the finished remaster as accepted.

## Acceptance

Check the script against current source, the quick/authenticated API distinction,
all layout frames, representative real-speed motion, actual spoken words, caption
synchronisation, complete decode, stream properties, loudness/true peak, reference
continuity, provider provenance, the separate publication-rights boundary and budget. Preserve the original website video until
those checks pass. Publication requires a separate request.

Working if: the review names exact hashes and evidence, unavailable checks remain
explicit, and a successful encode alone does not close the quality gate.

## Measured alignment

Whisper transcription retained genuine raw word data but quantised 56 words to
point timestamps. Independent `microsoft/mai-transcribe-2` produced 737 positive
word intervals across the actual ten takes. It is the final timing authority,
with exact source/ASR hashes and audio-reviewed spelling adjudications. Original
Whisper receipts remain available. Captions use readable phrases rather than
inventing per-word karaoke intervals.

## Rendering and rights boundaries

The installed Hyperframes CLI lint and 3-second render ran successfully. Its own
browser recovery/forced-close behavior is documented in the graphics notes;
manual renderer browsers were closed normally or their exact owned page reused.
No further CLI recovery runs were used. Existing METTLE fonts remain intentional.
The available Impeccable lint findings were triaged against rendered visibility.

Generated originals, provider receipts and embedded provenance are retained.
A local working export is permitted; this production does not certify external
commercial clearance. OpenRouter defers to supplying-provider terms. General
Google/OpenAI output terms are favourable; exact Seedance model-specific
conditions and preview-provider contractual matching require verification before
external publication. No third-party stock footage is used, so no Creative
Commons clip attribution or yt-dlp download is needed.

Working if: the local deliverable has the exact acceptance hashes, all timeline
references resolve inside its project bundle, and it is never described as live
or commercially cleared without those separate checks.

## Encode compatibility

The installed FFmpeg lacks drawtext and libass. Existing Pillow authors only
transparent typography and caption strips; FFmpeg overlays these on the intact
video. Caption cues retain their measured timings, rounded forward within one
30fps frame for the burned-in variant. No new system package was installed.

The first observatory motion take was rejected for geometry instability. A
gentler, constant-focal-length Seedance take is retained separately, with both
paid job identities and the original rejection evidence. One terminal Veo
service failure also remains in the receipts. Its billing record is unavailable;
a conservative $2 reserve is carried separately from confirmed task charges.

Working if: caption wording matches the full script, all raw word intervals
remain positive, the audio is unchanged in the captioned variant, and failed
provider jobs never disappear from the spending evidence.
