#!/usr/bin/env python3
"""Compact runtime summary for Megatron training logs."""

import argparse
import glob
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Set

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
ALL_GRADS_SYNC_PATTERN = re.compile(
    r"all-grads-sync\s+\.*:\s*\((?P<min>\d+(?:\.\d+)?),\s*(?P<max>\d+(?:\.\d+)?)\)",
    re.IGNORECASE,
)
PARAMS_ALL_GATHER_PATTERN = re.compile(
    r"params-all-gather\s+\.*:\s*\((?P<min>\d+(?:\.\d+)?),\s*(?P<max>\d+(?:\.\d+)?)\)",
    re.IGNORECASE,
)
RESIDUAL_OFFLOAD_PATTERN = re.compile(
    r"topk_adams_residual_cpu_offload\s+\.*\s+(?P<value>True|False)",
    re.IGNORECASE,
)
BENCH_MODE_LINE_PATTERN = re.compile(r"\bBENCH_MODE=(?P<mode>perf|debug)\b", re.IGNORECASE)
BENCH_MODE_PATH_PATTERN = re.compile(r"mode_(?P<mode>perf|debug)", re.IGNORECASE)
TIMING_LOG_LEVEL_PATTERN = re.compile(
    r"timing_log_level\s+\.*\s+(?P<value>\d+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize steady-state iteration, all-grads-sync, params-all-gather, "
            "and residual offload state from training logs."
        )
    )
    parser.add_argument("--paths", nargs="+", required=True, help="Files, directories, or globs")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".out", ".log", ".txt"],
        help="Extensions to include while scanning directories",
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=5,
        help="Skip first N matched samples for steady-state summary",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="Only include iteration entries with iteration >= value",
    )
    parser.add_argument(
        "--end-iteration",
        type=int,
        default=None,
        help="Only include iteration entries with iteration <= value",
    )
    return parser.parse_args()


def normalize_extensions(extensions: Sequence[str]) -> Set[str]:
    normalized: Set[str] = set()
    for ext in extensions:
        item = ext.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        normalized.add(item)
    return normalized


def expand_inputs(paths: Sequence[str], recursive: bool, extensions: Set[str]) -> List[Path]:
    files: List[Path] = []
    seen: Set[Path] = set()

    def add_file(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        if not resolved.is_file():
            return
        seen.add(resolved)
        files.append(resolved)

    for raw in paths:
        p = Path(raw).expanduser()

        if p.exists():
            if p.is_file():
                add_file(p)
                continue
            if p.is_dir():
                iterator: Iterable[Path] = p.rglob("*") if recursive else p.glob("*")
                for item in iterator:
                    if not item.is_file():
                        continue
                    if extensions and item.suffix.lower() not in extensions:
                        continue
                    add_file(item)
                continue

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


def summarize(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "avg": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "n": float(len(values)),
    }


def format_stats(stats: Optional[Dict[str, float]]) -> str:
    if stats is None:
        return "n=0"
    avg = stats["avg"]
    minimum = stats["min"]
    maximum = stats["max"]
    count = int(stats["n"])
    return f"avg={avg:.2f} ms | min={minimum:.2f} ms | max={maximum:.2f} ms | n={count}"


def detect_mode(path: Path, first_lines: List[str]) -> str:
    for line in first_lines:
        match = BENCH_MODE_LINE_PATTERN.search(line)
        if match:
            return match.group("mode").lower()
    match = BENCH_MODE_PATH_PATTERN.search(path.name)
    if match:
        return match.group("mode").lower()
    return "unknown"


def scrape_file(
    path: Path,
    skip_steps: int,
    start_iteration: Optional[int],
    end_iteration: Optional[int],
) -> Dict[str, object]:
    iteration_values: List[float] = []
    iteration_ids: List[int] = []
    all_grads_sync_max: List[float] = []
    params_all_gather_max: List[float] = []
    first_lines: List[str] = []

    residual_offload = "unknown"
    timing_log_level = "unknown"

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if idx < 200:
                first_lines.append(line)

            if residual_offload == "unknown":
                match = RESIDUAL_OFFLOAD_PATTERN.search(line)
                if match:
                    residual_offload = match.group("value").lower()

            if timing_log_level == "unknown":
                match = TIMING_LOG_LEVEL_PATTERN.search(line)
                if match:
                    timing_log_level = match.group("value")

            time_ms = extract_time_ms(line)
            if time_ms is not None:
                iteration = extract_iteration(line)
                if iteration is not None:
                    if start_iteration is not None and iteration < start_iteration:
                        continue
                    if end_iteration is not None and iteration > end_iteration:
                        continue
                    iteration_values.append(time_ms)
                    iteration_ids.append(iteration)

            match = ALL_GRADS_SYNC_PATTERN.search(line)
            if match:
                all_grads_sync_max.append(float(match.group("max")))

            match = PARAMS_ALL_GATHER_PATTERN.search(line)
            if match:
                params_all_gather_max.append(float(match.group("max")))

    skip = max(0, int(skip_steps))
    if skip > 0:
        iteration_values = iteration_values[skip:]
        iteration_ids = iteration_ids[skip:]
        all_grads_sync_max = all_grads_sync_max[skip:]
        params_all_gather_max = params_all_gather_max[skip:]

    iteration_window = "unknown"
    if iteration_ids:
        iteration_window = f"{iteration_ids[0]}..{iteration_ids[-1]}"

    return {
        "path": path,
        "mode": detect_mode(path=path, first_lines=first_lines),
        "timing_log_level": timing_log_level,
        "residual_offload": residual_offload,
        "iteration_window": iteration_window,
        "iteration_stats": summarize(iteration_values),
        "all_grads_sync_stats": summarize(all_grads_sync_max),
        "params_all_gather_stats": summarize(params_all_gather_max),
    }


def compact_value(stats: Optional[Dict[str, float]]) -> str:
    if stats is None:
        return "na"
    avg = stats["avg"]
    return f"{avg:.2f}"


def main() -> int:
    args = parse_args()
    extensions = normalize_extensions(args.extensions)
    files = expand_inputs(args.paths, args.recursive, extensions)

    if not files:
        print("No matching files found.")
        return 1

    rows: List[Dict[str, object]] = []
    for path in files:
        row = scrape_file(
            path=path,
            skip_steps=args.skip_steps,
            start_iteration=args.start_iteration,
            end_iteration=args.end_iteration,
        )
        rows.append(row)

        mode = str(row["mode"])
        timing = str(row["timing_log_level"])
        residual = str(row["residual_offload"])
        iteration_window = str(row["iteration_window"])
        iteration_stats = row["iteration_stats"]
        all_grads_stats = row["all_grads_sync_stats"]
        params_gather_stats = row["params_all_gather_stats"]

        print(path)
        print(f"  mode={mode} | timing_log_level={timing} | residual_cpu_offload={residual}")
        print(f"  steady_iteration_window={iteration_window} | skip_first={max(0, int(args.skip_steps))}")
        print(f"  iteration_ms: {format_stats(iteration_stats)}")
        print(f"  all_grads_sync_max_ms: {format_stats(all_grads_stats)}")
        print(f"  params_all_gather_max_ms: {format_stats(params_gather_stats)}")

    print("\n=== Compact Summary ===")
    for row in rows:
        path = row["path"]
        assert isinstance(path, Path)
        mode = str(row["mode"])
        timing = str(row["timing_log_level"])
        residual = str(row["residual_offload"])
        iteration_stats = row["iteration_stats"]
        all_grads_stats = row["all_grads_sync_stats"]
        params_gather_stats = row["params_all_gather_stats"]

        print(
            " | ".join(
                [
                    path.name,
                    f"mode={mode}",
                    f"timing={timing}",
                    f"residual_offload={residual}",
                    f"iter_avg_ms={compact_value(iteration_stats)}",
                    f"all_grad_sync_avg_ms={compact_value(all_grads_stats)}",
                    f"params_ag_avg_ms={compact_value(params_gather_stats)}",
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
