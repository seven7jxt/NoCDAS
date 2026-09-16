#!/usr/bin/env python3
"""Compare total, layer-type and individual-layer flit-hops and byte-hops."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


MODEL_RE = re.compile(r"^\s*(\d+)\s*\|\s*([^|]+?)\s*\|")
LAYER_RE = re.compile(r"^Layer finished\s+(\d+) at cycle\s+\d+\s*\|\s*flit-hops:\s*(\d+)\s+byte-hops:\s*(\d+)")
FINAL_RE = re.compile(r"^All finished! at cycle\s+\d+\s*\|\s*Layer\s+(\d+)\s+flit-hops:\s*(\d+)\s+byte-hops:\s*(\d+)")
TOTAL_RE = re.compile(r"^Total inter-router flit-hops:\s*(\d+)\s*$")
TOTAL_BYTES_RE = re.compile(r"^Total inter-router byte-hops:\s*(\d+)\s*$")


@dataclass
class Run:
    total: int
    layers: dict[int, int]
    types: dict[int, str]
    total_bytes: int
    layer_bytes: dict[int, int]

    def grouped(self, byte_hops: bool = False) -> dict[str, int]:
        result: dict[str, int] = {}
        for layer, hops in (self.layer_bytes if byte_hops else self.layers).items():
            name = self.types[layer]
            result[name] = result.get(name, 0) + hops
        return result


def parse_log(path: Path) -> Run:
    layers: dict[int, int] = {}
    layer_bytes: dict[int, int] = {}
    types: dict[int, str] = {}
    total = None
    total_bytes = None
    completed = False
    starts = 0
    # Stream the log: tensor-output lines can be very large.
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.strip() == "Initialize":
                starts += 1
                if starts > 1:
                    raise ValueError("multiple runs in one log")
            model = MODEL_RE.match(line)
            if model:
                types[int(model[1])] = model[2].strip()
            match = LAYER_RE.match(line)
            if line.startswith("All finished!"):
                if completed:
                    raise ValueError("multiple completion markers")
                completed = True
                match = FINAL_RE.match(line)
                if not match:
                    raise ValueError("final layer flit-hops/byte-hops markers were not found")
            elif line.startswith("Layer finished") and not match:
                raise ValueError("layer flit-hops/byte-hops markers were not found (old log format)")
            if match:
                layer, hops, byte_hops = map(int, match.groups())
                if layer in layers:
                    raise ValueError(f"duplicate layer {layer}")
                layers[layer] = hops
                layer_bytes[layer] = byte_hops
            match = TOTAL_RE.match(line)
            if match:
                if total is not None:
                    raise ValueError("duplicate total flit-hops marker")
                total = int(match[1])
            match = TOTAL_BYTES_RE.match(line)
            if match:
                if total_bytes is not None:
                    raise ValueError("duplicate total byte-hops marker")
                total_bytes = int(match[1])
    if not completed:
        raise ValueError("run has not completed")
    if total is None:
        raise ValueError("total flit-hops marker was not found")
    if sum(layers.values()) != total:
        raise ValueError(f"layer hops sum {sum(layers.values())} differs from total {total}")
    if total_bytes is None:
        raise ValueError("total byte-hops marker was not found")
    if sum(layer_bytes.values()) != total_bytes:
        raise ValueError(f"layer byte-hops sum {sum(layer_bytes.values())} differs from total {total_bytes}")
    return Run(total, layers, {i: types.get(i, f"Layer {i}") for i in layers}, total_bytes, layer_bytes)


def ratio(baseline: int, cnoc: int) -> float:
    return baseline / cnoc if cnoc else (math.inf if baseline else math.nan)


def show_table(title: str, rows: list[tuple[str, str, int, int]], sort_ratio: bool = True) -> None:
    if sort_ratio:
        rows.sort(key=lambda row: ratio(row[2], row[3]) if row[2] or row[3] else -math.inf)
    width = max(28, *(len(row[0]) for row in rows))
    label_width = max(18, *(len(row[1]) for row in rows))
    print(f"\n{title}\n")
    print(f"{'Workload':<{width}} {'Layer':<{label_width}} {'Baseline':>14} {'cNoC':>14} {'Ratio':>10}")
    print("-" * (width + label_width + 42))
    for name, label, baseline, cnoc in rows:
        value = ratio(baseline, cnoc)
        text = "n/a" if math.isnan(value) else "inf" if math.isinf(value) else f"{value:.3f}x"
        print(f"{name:<{width}} {label:<{label_width}} {baseline:>14,d} {cnoc:>14,d} {text:>10}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--cnoc-dir", type=Path)
    parser.add_argument("--baseline", type=Path, help="compare a single baseline log")
    parser.add_argument("--cnoc", type=Path, help="compare a single cNoC log")
    parser.add_argument("--per-layer", action="store_true", help="also show individual layer IDs in execution order")
    args = parser.parse_args(argv)
    if bool(args.baseline_dir) != bool(args.cnoc_dir):
        parser.error("--baseline-dir and --cnoc-dir must be supplied together")
    if bool(args.baseline) != bool(args.cnoc):
        parser.error("--baseline and --cnoc must be supplied together")
    if args.baseline and args.baseline_dir:
        parser.error("choose either a file pair or directories")
    if args.baseline:
        pairs = [(args.baseline.stem, args.baseline, args.cnoc)]
        print(f"Comparing logs: {args.baseline} and {args.cnoc}")
    else:
        root = Path(__file__).resolve().parent.parent / "logs"
        baseline_dir = args.baseline_dir or root / "baseline"
        cnoc_dir = args.cnoc_dir or root / "cnoc"
        names = sorted({p.name for p in baseline_dir.glob("*.log")} &
                       {p.name for p in cnoc_dir.glob("*.log")})
        pairs = [(name, baseline_dir / name, cnoc_dir / name) for name in names]
        print(f"Comparing logs in: {baseline_dir} and {cnoc_dir}")
    if not pairs:
        print("error: no matching log pairs found", file=sys.stderr)
        return 1
    runs = []
    for name, baseline_path, cnoc_path in pairs:
        try:
            baseline, cnoc = parse_log(baseline_path), parse_log(cnoc_path)
            if baseline.types != cnoc.types:
                raise ValueError("layer ID/type mappings differ")
        except (OSError, ValueError) as error:
            print(f"{name}: skipped: {error}", file=sys.stderr)
            continue
        runs.append((name, baseline, cnoc))
    if not runs:
        print("error: no complete log pairs to compare", file=sys.stderr)
        return 1
    print("Metrics: inter-router flit-hops and byte-hops from logs (not packet hops). "
          "Ratio = baseline / cNoC; >1 means lower cNoC traffic for that metric.")
    for byte_hops, metric in [(False, "flit-hops"), (True, "byte-hops")]:
        show_table(f"Total {metric} (ratio low to high)", [
            (name, "Total", b.total_bytes if byte_hops else b.total,
             c.total_bytes if byte_hops else c.total) for name, b, c in runs
        ])
        rows = []
        for name, baseline, cnoc in runs:
            b, c = baseline.grouped(byte_hops), cnoc.grouped(byte_hops)
            rows.extend((name, layer, b[layer], c[layer]) for layer in sorted(b))
        show_table(f"Layer-type {metric} (same types aggregated, ratio low to high)", rows)
        if args.per_layer:
            show_table(f"Individual-layer {metric}", [
                (name, f"{i}: {b.types[i]}",
                 (b.layer_bytes if byte_hops else b.layers)[i],
                 (c.layer_bytes if byte_hops else c.layers)[i])
                for name, b, c in runs for i in sorted(b.layers)
            ], sort_ratio=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
