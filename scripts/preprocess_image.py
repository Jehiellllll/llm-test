#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in in_dir.glob("*"):
        if fp.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = Image.open(fp).convert("RGB").resize((args.size, args.size))
        img.save(out_dir / fp.name)

    print(f"processed images -> {out_dir}")


if __name__ == "__main__":
    main()
