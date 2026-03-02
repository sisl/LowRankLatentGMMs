#!/usr/bin/env python3
"""Gather all results.json files from a runs directory, keyed by folder path."""

import argparse
import json
from pathlib import Path


def gather_results(runs_dir: Path) -> dict:
    """Walk runs_dir and collect all results.json files.

    Keys are relative paths like 'celeba-64x64/VPCFM-Normal-16-factors'.
    """
    results = {}
    for results_file in sorted(runs_dir.rglob("results.json")):
        key = str(results_file.parent.relative_to(runs_dir))
        with open(results_file) as f:
            results[key] = json.load(f)
    return results


def main():
    parser = argparse.ArgumentParser(description="Gather results.json files from a runs directory.")
    parser.add_argument("runs_dir", nargs="?", default="./runs", help="Path to runs directory (default: ./runs)")
    parser.add_argument("-o", "--output", help="Write gathered results to this JSON file (default: print to stdout)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"Error: '{runs_dir}' is not a directory.")

    results = gather_results(runs_dir)
    print(f"Found {len(results)} result(s).", flush=True)

    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
