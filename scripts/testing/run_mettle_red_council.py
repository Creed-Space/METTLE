#!/usr/bin/env python3
"""Historical METTLE Red Council fixed-demonstration entrypoint (retired).

EVALUATION_MODE is "fixed_demonstration". The former runner never submitted
attack prompts to METTLE, and its verdicts were predetermined constants.
Its pass rates cannot support detection or security claims. Main retired its
instrumented agent; this entrypoint preserves that retirement and emits only
an explicit retirement notice, never an evaluation or provider call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVALUATION_MODE = "fixed_demonstration"
EVALUATION_CAVEAT = (
    "Historical fixed demonstration, not a measurement: attack prompts never "
    "reached METTLE and verdicts were constants. The instrumented agent is "
    "retired; no scenarios are executed and no detection claim is supported."
)


def main() -> int:
    """Print the retirement summary and optionally write its JSON notice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a JSON retirement notice")
    # Recognize historical arguments without acting on them.
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--severity-threshold", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    notice = {
        "evaluation_mode": EVALUATION_MODE,
        "evaluation_caveat": EVALUATION_CAVEAT,
        "status": "retired",
        "scenarios_executed": 0,
    }
    print("METTLE RED COUNCIL: RETIRED")
    print(f"[{EVALUATION_MODE}] {EVALUATION_CAVEAT}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(notice, indent=2) + "\n", encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
