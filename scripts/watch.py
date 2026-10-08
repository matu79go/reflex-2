"""Watch a video the way a live camera would and report when a head first fires.

  python scripts/watch.py clip.mp4 --head heads/fall.json --label fall

Every --step seconds it judges the last --window seconds (2 frames per second) and prints the probability.
"""
from __future__ import annotations

import argparse
import json

from reflex2 import Head, Reflex2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--head", required=True)
    ap.add_argument("--label", required=True, help="the label to alert on, e.g. fall")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rf = Reflex2()
    head = Head.load(a.head)
    timeline = rf.watch(a.video, head, a.window, a.step)
    first = next((t for t, p in timeline if p[a.label] >= a.threshold), None)
    if a.json:
        print(json.dumps({"timeline": timeline, "first_alert_s": first}))
        return
    for t, p in timeline:
        bar = "#" * int(p[a.label] * 40)
        print(f"{t:6.1f}s  {p[a.label] * 100:5.1f}%  {bar}")
    print(f"first alert: {first} s" if first is not None else "no alert")


if __name__ == "__main__":
    main()
