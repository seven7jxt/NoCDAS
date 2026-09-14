#!/usr/bin/env python3
"""Compare baseline and cNoC cycle counts recorded in NoCDAS logs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TOTAL_RE = re.compile(r"All finished! at cycle\s+(\d+)")
LAYER_RE = re.compile(r"Layer finished\s+(\d+) at cycle\s+(\d+)")
MODEL_LAYER_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([^|]+?)\s*\|", re.MULTILINE
)


@dataclass
class Run:
    total: int
    layers: dict[str, int]


def parse_log(path: Path) -> Run:
    text = path.read_text(encoding="utf-8", errors="replace")
    total_matches = TOTAL_RE.findall(text)
    if not total_matches:
        raise ValueError("total cycle marker was not found")

    layer_types = {
        int(layer): layer_type.strip()
        for layer, layer_type in MODEL_LAYER_RE.findall(text)
    }
    cumulative: dict[int, int] = {}
    for layer, cycle in LAYER_RE.findall(text):
        cumulative[int(layer)] = int(cycle)

    if not cumulative:
        raise ValueError("no layer cycle markers were found")

    previous = 0
    layers: dict[str, int] = {}
    for layer in sorted(cumulative):
        duration = cumulative[layer] - previous
        if duration < 0:
            raise ValueError(f"layer {layer} has a decreasing cumulative cycle")
        layer_type = layer_types.get(layer, f"Layer {layer}")
        layers[layer_type] = layers.get(layer_type, 0) + duration
        previous = cumulative[layer]

    return Run(total=int(total_matches[-1]), layers=layers)


def pair_logs() -> tuple[Path, Path, list[str]]:
    repo_dir = Path(__file__).resolve().parent.parent
    logs_dir = repo_dir / "logs"
    baseline_dir = logs_dir / "baseline"
    cnoc_dir = logs_dir / "cnoc"
    names = sorted(
        {p.name for p in baseline_dir.glob("*.log")}
        & {p.name for p in cnoc_dir.glob("*.log")}
    )
    return baseline_dir, cnoc_dir, names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline and cNoC total and per-layer cycles."
    )
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--cnoc-dir", type=Path)
    args = parser.parse_args()

    if (args.baseline_dir is None) != (args.cnoc_dir is None):
        parser.error("--baseline-dir and --cnoc-dir must be supplied together")

    if args.baseline_dir is None:
        baseline_dir, cnoc_dir, names = pair_logs()
        print(f"Comparing logs in: {baseline_dir.parent}")
    else:
        baseline_dir, cnoc_dir = args.baseline_dir, args.cnoc_dir
        names = sorted(
            {p.name for p in baseline_dir.glob("*.log")}
            & {p.name for p in cnoc_dir.glob("*.log")}
        )
        print(f"Comparing logs in: {baseline_dir} and {cnoc_dir}")

    if not names:
        print("error: no matching log pairs found", file=sys.stderr)
        return 1

    runs: list[tuple[str, Run, Run]] = []
    for name in names:
        try:
            baseline = parse_log(baseline_dir / name)
            cnoc = parse_log(cnoc_dir / name)
        except (OSError, ValueError) as error:
            print(f"{name}: skipped: {error}", file=sys.stderr)
            continue
        if set(baseline.layers) != set(cnoc.layers):
            print(f"{name}: skipped: layer-type sets differ", file=sys.stderr)
            continue
        runs.append((name, baseline, cnoc))

    if not runs:
        print("error: no complete log pairs to compare", file=sys.stderr)
        return 1

    total_rows = sorted(
        ((name, baseline.total, cnoc.total, baseline.total / cnoc.total)
         for name, baseline, cnoc in runs),
        key=lambda row: row[3],
    )
    print("\nTotal cycles (baseline / cNoC, low to high)\n")
    print(f"{'Workload':<28} {'Baseline':>14} {'cNoC':>14} {'Ratio':>10}")
    print("-" * 70)
    for name, baseline, cnoc, ratio in total_rows:
        print(f"{name:<28} {baseline:>14,d} {cnoc:>14,d} {ratio:>9.3f}x")

    layer_rows = []
    for name, baseline, cnoc in runs:
        for layer_type in sorted(baseline.layers):
            b_cycles = baseline.layers[layer_type]
            c_cycles = cnoc.layers[layer_type]
            layer_rows.append((name, layer_type, b_cycles, c_cycles, b_cycles / c_cycles))
    layer_rows.sort(key=lambda row: row[4])

    print("\nLayer-type cycles (same layer types aggregated, baseline / cNoC, low to high)\n")
    print(f"{'Workload':<28} {'Layer type':>14} {'Baseline':>14} {'cNoC':>14} {'Ratio':>10}")
    print("-" * 78)
    for name, layer_type, baseline, cnoc, ratio in layer_rows:
        print(f"{name:<28} {layer_type:>14} {baseline:>14,d} {cnoc:>14,d} {ratio:>9.3f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
