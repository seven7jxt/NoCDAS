#!/usr/bin/env python3
"""Compare the final numeric vectors in two NoCDAS log files."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)


def extract_output(path: Path, marker: str) -> list[float]:
    """Extract the first non-empty, numeric-only line after *marker*."""
    marker_seen = False

    with path.open("r", encoding="utf-8", errors="replace") as log:
        for line_number, line in enumerate(log, start=1):
            if not marker_seen:
                if marker in line:
                    marker_seen = True
                continue

            fields = line.split()
            if not fields:
                continue
            if all(NUMBER_RE.fullmatch(field) for field in fields):
                values = [float(field) for field in fields]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(
                        f"{path}: output line {line_number} contains NaN or infinity"
                    )
                return values

    if not marker_seen:
        raise ValueError(f"{path}: marker {marker!r} was not found")
    raise ValueError(f"{path}: no numeric output line was found after {marker!r}")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"output lengths differ: {len(left)} values vs {len(right)} values"
        )
    if not left:
        raise ValueError("output vectors are empty")

    dot = math.fsum(a * b for a, b in zip(left, right))
    left_sq = math.fsum(a * a for a in left)
    right_sq = math.fsum(b * b for b in right)
    if left_sq == 0.0 or right_sq == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    # Clamp tiny floating-point excursions outside the mathematical range.
    return max(-1.0, min(1.0, dot / math.sqrt(left_sq * right_sq)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare final output vectors from two NoCDAS logs."
    )
    parser.add_argument("log_a", type=Path, nargs="?", help="first NoCDAS log")
    parser.add_argument("log_b", type=Path, nargs="?", help="second NoCDAS log")
    parser.add_argument(
        "--marker",
        default="All finished!",
        help="text preceding the output vector (default: %(default)s)",
    )
    return parser.parse_args()


def compare_log_pair(log_a: Path, log_b: Path, marker: str) -> tuple[int, float]:
    left = extract_output(log_a, marker)
    right = extract_output(log_b, marker)
    return len(left), cosine_similarity(left, right)


def compare_default_logs(marker: str) -> int:
    # Resolve logs relative to the repository, so invocation from script/ also works.
    repo_dir = Path(__file__).resolve().parent.parent
    logs_dir = repo_dir / "logs"
    baseline_dir = repo_dir / "logs" / "baseline"
    cnoc_dir = repo_dir / "logs" / "cnoc"
    print(f"Comparing logs in: {logs_dir}")
    baseline_logs = {path.name: path for path in baseline_dir.glob("*.log")}
    cnoc_logs = {path.name: path for path in cnoc_dir.glob("*.log")}
    common_names = sorted(baseline_logs.keys() & cnoc_logs.keys())

    if not common_names:
        print(
            f"error: no matching log pairs found in {baseline_dir} and {cnoc_dir}",
            file=sys.stderr,
        )
        return 1

    similarities: list[float] = []
    for name in common_names:
        try:
            length, similarity = compare_log_pair(
                baseline_logs[name], cnoc_logs[name], marker
            )
        except (OSError, ValueError) as error:
            print(f"{name}: error: {error}", file=sys.stderr)
            continue
        similarities.append(similarity)
        print(f"{name}: length={length}, cosine={similarity:.12f}")

    if not similarities:
        print("error: all matching log pairs failed to compare", file=sys.stderr)
        return 1

    print(f"Average cosine similarity: {math.fsum(similarities) / len(similarities):.12f}")
    print(f"Compared pairs:             {len(similarities)}/{len(common_names)}")
    return 0


def main() -> int:
    args = parse_args()
    if args.log_a is None and args.log_b is None:
        return compare_default_logs(args.marker)
    if args.log_a is None or args.log_b is None:
        print("error: provide both log paths, or provide neither for default batch mode", file=sys.stderr)
        return 2
    try:
        length, similarity = compare_log_pair(args.log_a, args.log_b, args.marker)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Vector length:     {length}")
    print(f"Cosine similarity: {similarity:.12f}")
    print(f"Cosine distance:   {1.0 - similarity:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
