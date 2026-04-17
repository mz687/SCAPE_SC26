#!/usr/bin/env python3
"""Stream OpenWebText from Hugging Face to JSONL for SCAPE preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream OpenWebText from Hugging Face and write one JSON object with a "
            'single "text" field per line.'
        )
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Path to the JSONL file to create.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Skylion007/openwebtext",
        help="Hugging Face dataset name to stream.",
    )
    parser.add_argument(
        "--dataset-config",
        default="plain_text",
        help="Optional dataset config/subset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Field containing document text.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10_000,
        help="Progress logging interval in written documents.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.log_interval <= 0:
        raise SystemExit("--log-interval must be positive")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - import error path
        raise SystemExit(
            "The 'datasets' package is required. Install it with 'pip install datasets'."
        ) from exc

    output_path = args.output_jsonl.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Streaming {args.dataset_name}"
        f" config={args.dataset_config or '<default>'}"
        f" split={args.split} to {output_path}"
    )

    load_kwargs = {
        "split": args.split,
        "streaming": True,
    }
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config

    dataset = load_dataset(args.dataset_name, **load_kwargs)

    scanned = 0
    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for sample in dataset:
            scanned += 1
            text = sample.get(args.text_key, "")
            if not text:
                continue
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written % args.log_interval == 0:
                print(f"Written {written} documents after scanning {scanned} samples")

    print(
        f"Finished writing {written} documents to {output_path} "
        f"after scanning {scanned} samples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
