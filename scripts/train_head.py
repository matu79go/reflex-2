"""Train a Reflex-2 head from labelled examples and save it as JSON.

  python scripts/train_head.py --out heads/fall.json \
      --label fall="data/*/Fall/*.mp4" --label adl="data/*/ADL/*.mp4"

Each --label is NAME=GLOB. Any input Reflex-2 can embed works (videos, images, audio files).
"""
from __future__ import annotations

import argparse
import glob
import time

from reflex2 import Reflex2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True, help='NAME=GLOB, e.g. fall="clips/fall/*.mp4"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--wd", type=float, default=0.1, help="L2 strength")
    a = ap.parse_args()
    examples = {}
    for spec in a.label:
        name, pattern = spec.split("=", 1)
        examples[name] = sorted(glob.glob(pattern.strip('"'), recursive=True))
        print(f"{name}: {len(examples[name])} files", flush=True)
    rf = Reflex2(a.model)
    t0 = time.time()
    head = rf.train_head(examples, wd=a.wd)
    head.save(a.out)
    print(f"saved {a.out} ({head.meta['n']} examples, {time.time() - t0:.1f} s incl. embedding)")


if __name__ == "__main__":
    main()
