#!/usr/bin/env python3
"""Compute average training step time from SCAPE/Megatron log files."""

import argparse
import glob
import re
from pathlib import Path
from statistics import mean
from typing import List, Optional, Sequence, Set, Tuple


TIME_PATTERNS = (
    re.compile(
        r"elapsed time per iteration \((?P<unit>ms|s|sec|seconds)\):\s*(?P<value>\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"time per (?:iteration|step)\s*\((?P<unit>ms|s|sec|seconds)\):\s*(?P<value>\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)
ITERATION_PATTERN = re.compile(r"iteration\s+(?P<iteration>\d+)\s*/", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape training logs and compute average time per step. "
            "Supports individual files, directories, and glob patterns."
        )
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Log files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for log files.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".out", ".log", ".txt"],
        help="File extensions to include when a directory is provided.",
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=0,
        metavar="N",
        help="Ignore the first N matched steps in each file (useful for warmup).",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        metavar="N",
        help="Only include entries with iteration >= N when iteration is present.",
    )
    parser.add_argument(
        "--end-iteration",
        type=int,
        default=None,
        metavar="N",
        help="Only include entries with iteration <= N when iteration is present.",
    )
    return parser.parse_args()


def normalize_extensions(extensions: Sequence[str]) -> Set[str]:
    normalized = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def expand_inputs(paths: Sequence[str], recursive: bool, extensions: Set[str]) -> List[Path]:
    files: List[Path] = []
    seen = set()

    def add_file(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            files.append(resolved)

    for raw in paths:
        path = Path(raw).expanduser()

        if path.exists():
            if path.is_file():
                add_file(path)
                continue
            if path.is_dir():
                iterator = path.rglob("*") if recursive else path.glob("*")
                for item in iterator:
                    if not item.is_file():
                        continue
                    if extensions and item.suffix.lower() not in extensions:
                        continue
                    add_file(item)
                continue

        # Fall back to glob expansion (useful for wildcard input).
        for match in glob.glob(raw, recursive=recursive):
            candidate = Path(match).expanduser()
            if candidate.is_file():
                add_file(candidate)

    return sorted(files)


def extract_time_ms(line: str) -> Optional[float]:
    for pattern in TIME_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        value = float(match.group("value"))
        unit = match.group("unit").lower()
        if unit in {"s", "sec", "seconds"}:
            return value * 1000.0
        return value
    return None


def extract_iteration(line: str) -> Optional[int]:
    match = ITERATION_PATTERN.search(line)
    if not match:
        return None
    return int(match.group("iteration"))


def scrape_file(
    path: Path,
    skip_steps: int,
    start_iteration: Optional[int],
    end_iteration: Optional[int],
) -> List[float]:
    values: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            time_ms = extract_time_ms(line)
            if time_ms is None:
                continue

            iteration = extract_iteration(line)
            if iteration is not None:
                if start_iteration is not None and iteration < start_iteration:
                    continue
                if end_iteration is not None and iteration > end_iteration:
                    continue

            values.append(time_ms)

    if skip_steps > 0:
        values = values[skip_steps:]
    return values


def summarize(values: Sequence[float]) -> Tuple[float, float, float]:
    return mean(values), min(values), max(values)


def format_ms(ms: float) -> str:
    return f"{ms:.3f} ms ({ms / 1000.0:.3f} s)"


def main() -> int:
    args = parse_args()
    extensions = normalize_extensions(args.extensions)
    files = expand_inputs(args.paths, args.recursive, extensions)

    if not files:
        print("No matching files found.")
        return 1

    all_values: List[float] = []
    processed = 0
    for file_path in files:
        values = scrape_file(
            file_path,
            skip_steps=max(args.skip_steps, 0),
            start_iteration=args.start_iteration,
            end_iteration=args.end_iteration,
        )
        if not values:
            continue

        avg_ms, min_ms, max_ms = summarize(values)
        print(
            f"{file_path}\n"
            f"  matched_steps={len(values)}\n"
            f"  avg={format_ms(avg_ms)} | min={format_ms(min_ms)} | max={format_ms(max_ms)}"
        )
        all_values.extend(values)
        processed += 1

    if not all_values:
        print("No timing entries matched in the selected files.")
        return 2

    overall_avg, overall_min, overall_max = summarize(all_values)
    print("\n=== Overall ===")
    print(f"files_with_matches={processed}/{len(files)}")
    print(f"total_matched_steps={len(all_values)}")
    print(
        f"avg={format_ms(overall_avg)} | min={format_ms(overall_min)} | max={format_ms(overall_max)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
