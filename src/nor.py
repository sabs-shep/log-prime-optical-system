#!/usr/bin/env python3
"""
nor.py

Standalone 128-column NOR combiner for debug and file generation.

Purpose:
- Read Input 1 and Input 2 as 128 numbers each.
- Treat any value > 0 as logic 1 / active / light present.
- Compute lane-wise NOR:

      OUT[i] = NOT(INPUT1[i] OR INPUT2[i])

  which means:

      OUT[i] = 1 only when INPUT1[i] == 0 and INPUT2[i] == 0
      OUT[i] = 0 otherwise

- Save the output as exactly 128 numbers.

Important:
- This script operates across all 128 LCD columns.
- It does NOT know which columns are real mapped optical wavelength lanes.
- main.py performs the physical mapped-lane NOR by masking unmapped columns.
- Therefore this script should not calculate optical voltage.

Supported input formats:
- .txt / .csv containing 128 numbers separated by spaces, commas, or newlines
- .json containing either:
    [0, 1, 0, ...]
    {"values": [0, 1, 0, ...]}
    {"output": [0, 1, 0, ...]}
    {"pattern": [[...128...], ...]}

For 2D pattern JSON:
- Each x-column becomes one number.
- If any pixel in the column is > 0, that output lane is 1.

Output files:
- <out>.txt
- <out>.csv
- <out>.json
- <out>.c

Examples:
    py -3.13 nor.py --input1 input1.json --input2 input2.json --out nor_result
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, List

LANE_COUNT = 128


def load_128_values(path: str | Path, active_threshold: float = 0.0) -> List[int]:
    """
    Load exactly 128 logical lane values from a file.

    Returned values are binary:
        0 = inactive / no light / false
        1 = active / light present / true

    Any numeric value greater than active_threshold becomes 1.
    Any numeric value <= active_threshold becomes 0.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        raw_values = load_values_from_json(path)
    else:
        raw_values = load_values_from_text(path)

    if len(raw_values) != LANE_COUNT:
        raise ValueError(
            f"{path} produced {len(raw_values)} values; expected exactly {LANE_COUNT}."
        )

    return [1 if float(value) > active_threshold else 0 for value in raw_values]


def load_values_from_json(path: Path) -> List[float]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return flatten_direct_list(data)

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported JSON structure in {path}")

    for key in ["values", "output", "lanes", "input", "input1", "input2"]:
        if key in data:
            return flatten_direct_list(data[key])

    if "pattern" in data:
        return compress_2d_pattern_to_128(data["pattern"])

    raise ValueError(
        f"JSON file {path} must contain a direct list, a 'values'/'output' list, or a 'pattern' matrix."
    )


def flatten_direct_list(value: Any) -> List[float]:
    if not isinstance(value, list):
        raise ValueError("Expected a list of numeric values.")

    if len(value) == LANE_COUNT and all(not isinstance(item, list) for item in value):
        return [float(item) for item in value]

    if value and all(isinstance(row, list) for row in value):
        return compress_2d_pattern_to_128(value)

    raise ValueError("Expected either a 128-number list or a 2D pattern list.")


def compress_2d_pattern_to_128(pattern: List[List[Any]]) -> List[float]:
    """
    Convert a 2D LCD pattern to 128 lane values.

    Any open pixel in a column means the column/lane is active.
    """
    if not pattern:
        raise ValueError("Pattern is empty.")

    width = len(pattern[0])

    if width != LANE_COUNT:
        raise ValueError(f"Pattern width is {width}; expected {LANE_COUNT}.")

    for y, row in enumerate(pattern):
        if len(row) != LANE_COUNT:
            raise ValueError(f"Pattern row {y} has width {len(row)}; expected {LANE_COUNT}.")

    output: List[float] = []

    for x in range(LANE_COUNT):
        active = any(float(pattern[y][x]) > 0 for y in range(len(pattern)))
        output.append(1.0 if active else 0.0)

    return output


def load_values_from_text(path: Path) -> List[float]:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Input file is empty: {path}")

    if text.startswith("P1"):
        return load_values_from_pbm_text(text)

    tokens = re.split(r"[\s,;]+", text)
    tokens = [token for token in tokens if token != ""]

    return [float(token) for token in tokens]


def load_values_from_pbm_text(text: str) -> List[float]:
    """
    Supports P1 PBM.

    If 128x1, returns the row directly.
    If 128x64 or other 128-wide PBM, compresses by column.
    """
    lines: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)

    if not lines or lines[0] != "P1":
        raise ValueError("Invalid P1 PBM file.")

    if len(lines) < 3:
        raise ValueError("PBM file is missing dimensions or pixel data.")

    width, height = [int(part) for part in lines[1].split()[:2]]

    if width != LANE_COUNT:
        raise ValueError(f"PBM width is {width}; expected {LANE_COUNT}.")

    tokens: List[str] = []
    for line in lines[2:]:
        tokens.extend(line.split())

    values = [float(token) for token in tokens]

    if len(values) != width * height:
        raise ValueError(
            f"PBM has {len(values)} pixels; expected {width * height}."
        )

    if height == 1:
        return values

    pattern: List[List[float]] = []

    for y in range(height):
        start = y * width
        pattern.append(values[start:start + width])

    return compress_2d_pattern_to_128(pattern)


def nor_128(input1: List[int], input2: List[int]) -> List[int]:
    if len(input1) != LANE_COUNT or len(input2) != LANE_COUNT:
        raise ValueError("Both inputs must contain exactly 128 values.")

    return [
        1 if input1[i] == 0 and input2[i] == 0 else 0
        for i in range(LANE_COUNT)
    ]


def save_txt(values: List[int], path: Path) -> None:
    path.write_text(
        " ".join(str(value) for value in values) + "\n",
        encoding="utf-8"
    )


def save_csv(values: List[int], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(values)


def save_json(input1: List[int], input2: List[int], output: List[int], path: Path) -> None:
    data = {
        "lane_count": LANE_COUNT,
        "logic": "OUT[i] = NOT(INPUT1[i] OR INPUT2[i])",
        "convention": "0=inactive/no-light/false, 1=active/light-present/true",
        "input1": input1,
        "input2": input2,
        "output": output,
        "active_output_indices": [
            i for i, value in enumerate(output)
            if value == 1
        ],
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_c_array(values: List[int], path: Path, array_name: str = "nor_output") -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("#include <stdint.h>\n\n")
        file.write(f"const uint8_t {array_name}[{LANE_COUNT}] = {{\n")

        for i in range(0, LANE_COUNT, 16):
            chunk = values[i:i + 16]
            file.write("    " + ", ".join(str(value) for value in chunk))

            if i + 16 < LANE_COUNT:
                file.write(",")

            file.write("\n")

        file.write("};\n")


def save_outputs(
    input1: List[int],
    input2: List[int],
    output: List[int],
    out_prefix: str
) -> None:
    base = Path(out_prefix)

    save_txt(output, base.with_suffix(".txt"))
    save_csv(output, base.with_suffix(".csv"))
    save_json(input1, input2, output, base.with_suffix(".json"))
    save_c_array(output, base.with_suffix(".c"))

    print(f"Saved {base.with_suffix('.txt')}")
    print(f"Saved {base.with_suffix('.csv')}")
    print(f"Saved {base.with_suffix('.json')}")
    print(f"Saved {base.with_suffix('.c')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NOR two 128-column LCD/vector outputs. Debug helper only."
    )

    parser.add_argument("--input1", required=True, help="First 128-number input file")
    parser.add_argument("--input2", required=True, help="Second 128-number input file")
    parser.add_argument("--out", default="nor_result", help="Output file prefix")
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.0,
        help="Values greater than this are treated as logic 1. Default: 0.0"
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    input1 = load_128_values(args.input1, active_threshold=args.active_threshold)
    input2 = load_128_values(args.input2, active_threshold=args.active_threshold)

    output = nor_128(input1, input2)

    save_outputs(input1, input2, output, args.out)

    print()
    print("128-Column NOR Debug Summary")
    print("----------------------------")
    print(f"Input 1 active columns: {sum(input1)}")
    print(f"Input 2 active columns: {sum(input2)}")
    print(f"NOR active columns:     {sum(output)}")
    print(f"Active indices:         {[i for i, value in enumerate(output) if value == 1]}")
    print()
    print("Note:")
    print("  This is a raw 128-column LCD/vector NOR debug result.")
    print("  It is not the physical mapped optical NOR result.")
    print("  main.py masks this down to the mapped wavelength lanes.")
    print("  Use main.py output for the actual optical-core result.")


if __name__ == "__main__":
    main()
