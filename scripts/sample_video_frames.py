#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=16)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, frame in enumerate(iio.imiter(args.video)):
        if i % max(args.stride, 1) != 0:
            continue
        Image.fromarray(frame).save(out_dir / f"frame_{i:05d}.png")
        saved += 1
        if saved >= args.max_frames:
            break
    print(f"saved {saved} frames")


if __name__ == "__main__":
    main()
