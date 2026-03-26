#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    args = p.parse_args()

    out_lines = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["text"] = normalize_text(row.get("text", ""))
            out_lines.append(json.dumps(row, ensure_ascii=False))

    Path(args.output_jsonl).write_text("\n".join(out_lines), encoding="utf-8")
    print(f"saved: {args.output_jsonl}")


if __name__ == "__main__":
    main()
