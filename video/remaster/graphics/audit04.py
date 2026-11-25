"""Finite decoded-frame check of chapter04 headline paint and six review frames."""

import json
import pathlib
import subprocess
import sys
import numpy as np

root = pathlib.Path("video/build/remaster/graphics/correction04")
clip = pathlib.Path(sys.argv[1])
label = sys.argv[2]
# The native defect at x800 removes the end of the always-visible headline.
p = subprocess.Popen(
    [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(clip),
        "-vf",
        "crop=500:100:800:135",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ],
    stdout=subprocess.PIPE,
)
counts = []
if p.stdout is None:
    raise RuntimeError("Decoder stdout missing")
while True:
    b = p.stdout.read(500 * 100 * 3)
    if not b:
        break
    if len(b) != 500 * 100 * 3:
        raise RuntimeError("Partial decoded frame")
    a = np.frombuffer(b, np.uint8).reshape(100, 500, 3)
    counts.append(
        int(((a[:, :, 0] > 180) & (a[:, :, 1] > 180) & (a[:, :, 2] > 160)).sum())
    )
if p.wait():
    raise RuntimeError("Decoder failed")
# Ignore deliberate initial reveal; headline stays fully visible thereafter.
fail = [i for i, n in enumerate(counts) if i >= 30 and n < 3000]
report = {
    "clip": str(clip),
    "frames_decoded": len(counts),
    "roi": "x800..1299,y135..234",
    "ivory_pixel_threshold": 3000,
    "minimum_after_1s": min(counts[30:]),
    "failure_frames": fail,
    "counts": counts,
}
(root / f"{label}-headline-audit.json").write_text(json.dumps(report, indent=2))
for t in [3, 8, 12, 16, 25, 33]:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            str(t),
            "-i",
            str(clip),
            "-frames:v",
            "1",
            str(root / f"{label}-{t}s.png"),
        ],
        check=True,
    )
print(
    json.dumps(
        {k: v for k, v in report.items() if k not in ["counts", "failure_frames"]}
        | {"failure_count": len(fail), "first_failure": fail[:3]}
    )
)

if label == "after" and (len(counts) != 1133 or fail):
    raise RuntimeError("Corrected chapter04 headline audit failed")
