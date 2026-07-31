#!/usr/bin/env python3
"""
main.py

End-to-end runner for the Optical GPU Driver proof-of-concept.

Important model fix:
- The optical system only has lanes at the mapped LCD wavelength columns.
- Unmapped LCD columns are not optical channels and must not participate in NOR.
- Therefore this runner computes a MAPPED NOR result:
      OUT[x] = 1 only if x is a mapped wavelength pixel AND input1[x] == 0 AND input2[x] == 0
      OUT[x] = 0 for every unmapped pixel

Runs in order:
  1. Wavelength_LCD_Encoder.py     -> integer_log_prime_wavelength_map.cfg/.csv
  2. optical_lcd_codec.py encode   -> input1.json
  3. random_input.py               -> input2.json
  4. nor.py                        -> optional external NOR run, mostly for debug output
  5. main.py                       -> overwrites nor_result.* with mapped-channel NOR
  6. nor_voltage_decoder.py        -> decodes unknown mean voltage and compares to mapped NOR

Run:
  py -3.13 main.py
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Set
from decimal import Decimal, getcontext, ROUND_CEILING

LANE_COUNT = 128

DEFAULT_MAP_SCRIPT = "Wavelength_LCD_Encoder.py"
DEFAULT_CODEC_SCRIPT = "optical_lcd_codec.py"
DEFAULT_RANDOM_SCRIPT = "random_input.py"
DEFAULT_NOR_SCRIPT = "nor.py"
DEFAULT_DECODER_SCRIPT = "nor_voltage_decoder.py"

DEFAULT_MAP_CFG = "integer_log_prime_wavelength_map.cfg"
DEFAULT_INPUT1_PREFIX = "input1"
DEFAULT_INPUT2_PREFIX = "input2"
DEFAULT_NOR_PREFIX = "nor_result"
DEFAULT_DECODE_REPORT = "unknown_decode_report.json"
DEFAULT_DECODED_VECTOR = "decoded_unknown_vector.txt"


def run_step(label: str, command: List[str], allow_fail: bool = False) -> subprocess.CompletedProcess:
    print()
    print("=" * 100)
    print(label)
    print("=" * 100)
    print("$ " + " ".join(command))

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Step failed: {label} | exit={result.returncode}")

    return result


def require_file(path: str | Path, label: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing {label}: {file_path}")
    return file_path


def clean_header(header: str) -> str:
    if header is None:
        return ""
    text = str(header).strip()
    text = text.replace('<strong data-lexical-text="true">', '')
    text = text.replace('</strong>', '')
    text = text.replace(',', '')
    return text.strip()


def load_mapped_pixels(map_path: str | Path) -> Set[int]:
    """
    Return the set of actual wavelength-channel LCD columns from the mapper output.
    These are the only columns that are real optical lanes.
    """
    path = Path(map_path)
    if not path.exists():
        raise FileNotFoundError(f"Map file not found: {path}")

    pixels: Set[int] = set()

    if path.suffix.lower() == ".cfg":
        cfg = configparser.ConfigParser()
        cfg.read(path, encoding="utf-8")

        for section in cfg.sections():
            if not section.lower().startswith("channel_"):
                continue
            pixels.add(int(cfg[section]["integer_pixel"]))

    elif path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                clean = {clean_header(key): value for key, value in row.items()}
                pixels.add(int(clean["integer_pixel"]))
    else:
        raise ValueError("Map must be .cfg or .csv")

    for pixel in pixels:
        if not 0 <= pixel < LANE_COUNT:
            raise ValueError(f"Mapped pixel {pixel} is outside 0..{LANE_COUNT - 1}")

    if not pixels:
        raise ValueError("No mapped pixels found in map file.")

    return pixels


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, data: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def compress_pattern_to_128(pattern: List[List[Any]]) -> List[int]:
    if not pattern:
        raise ValueError("Pattern is empty.")

    width = len(pattern[0])
    if width != LANE_COUNT:
        raise ValueError(f"Pattern width is {width}; expected {LANE_COUNT}.")

    for row_index, row in enumerate(pattern):
        if len(row) != LANE_COUNT:
            raise ValueError(f"Pattern row {row_index} has width {len(row)}; expected {LANE_COUNT}.")

    return [1 if any(float(pattern[y][x]) > 0 for y in range(len(pattern))) else 0 for x in range(LANE_COUNT)]


def normalise_128(value: Any) -> List[int]:
    if not isinstance(value, list):
        raise ValueError("Expected list or 2D pattern.")

    if len(value) == LANE_COUNT and all(not isinstance(item, list) for item in value):
        return [1 if float(item) > 0 else 0 for item in value]

    if value and all(isinstance(row, list) for row in value):
        return compress_pattern_to_128(value)

    raise ValueError("Expected 128-number list or 2D pattern.")


def load_128(path: str | Path) -> List[int]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Could not load 128-vector: {file_path}")

    if file_path.suffix.lower() == ".json":
        data = load_json(file_path)

        if isinstance(data, list):
            return normalise_128(data)

        if isinstance(data, dict):
            for key in ["output", "values", "lanes", "input", "input1", "input2"]:
                if key in data:
                    return normalise_128(data[key])

            if "pattern" in data:
                return compress_pattern_to_128(data["pattern"])

        raise ValueError(f"Unsupported JSON vector structure: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()

    if text.startswith("P1"):
        return load_pbm_128(text)

    tokens = [token for token in re.split(r"[\s,;]+", text) if token]
    values = [1 if float(token) > 0 else 0 for token in tokens]

    if len(values) != LANE_COUNT:
        raise ValueError(f"{file_path} produced {len(values)} values; expected {LANE_COUNT}.")

    return values


def load_pbm_128(text: str) -> List[int]:
    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)

    if not lines or lines[0] != "P1":
        raise ValueError("Invalid PBM data. Expected P1 header.")

    width, height = [int(part) for part in lines[1].split()[:2]]
    if width != LANE_COUNT:
        raise ValueError(f"PBM width is {width}; expected {LANE_COUNT}.")

    tokens: List[str] = []
    for line in lines[2:]:
        tokens.extend(line.split())

    values = [1 if float(token) > 0 else 0 for token in tokens]
    if len(values) != width * height:
        raise ValueError(f"PBM has {len(values)} pixels; expected {width * height}.")

    if height == 1:
        return values

    pattern: List[List[int]] = []
    for y in range(height):
        start = y * width
        pattern.append(values[start:start + width])

    return compress_pattern_to_128(pattern)


def mask_to_mapped_lanes(values: List[int], mapped_pixels: Set[int]) -> List[int]:
    """
    Force all unmapped columns to zero.
    """
    if len(values) != LANE_COUNT:
        raise ValueError("Vector must have 128 lanes.")
    return [1 if index in mapped_pixels and values[index] else 0 for index in range(LANE_COUNT)]


def mapped_nor_128(input1: List[int], input2: List[int], mapped_pixels: Set[int]) -> List[int]:
    """
    NOR only over real optical wavelength lanes.
    Unmapped columns are not physical lanes, so they always output zero.
    """
    if len(input1) != LANE_COUNT or len(input2) != LANE_COUNT:
        raise ValueError("Both inputs must have exactly 128 lanes.")

    output = [0 for _ in range(LANE_COUNT)]

    for pixel in mapped_pixels:
        output[pixel] = 1 if input1[pixel] == 0 and input2[pixel] == 0 else 0

    return output


def save_128_txt(path: str | Path, values: List[int]) -> None:
    Path(path).write_text(" ".join(str(value) for value in values) + "\n", encoding="utf-8")


def save_nor_result(prefix: str, input1: List[int], input2: List[int], output: List[int], mapped_pixels: Set[int]) -> None:
    active_indices = [index for index, value in enumerate(output) if value == 1]

    save_128_txt(f"{prefix}.txt", output)
    Path(f"{prefix}.csv").write_text(",".join(str(value) for value in output) + "\n", encoding="utf-8")

    save_json(
        f"{prefix}.json",
        {
            "lane_count": LANE_COUNT,
            "logic": "MAPPED_OUT[x] = mapped(x) AND NOT(INPUT1[x] OR INPUT2[x])",
            "convention": "0=inactive/no-light/false, 1=active/light-present/true",
            "mapped_pixels": sorted(mapped_pixels),
            "input1": input1,
            "input2": input2,
            "output": output,
            "active_output_indices": active_indices,
        },
    )

    with Path(f"{prefix}.c").open("w", encoding="utf-8") as file:
        file.write("#include <stdint.h>\n\n")
        file.write(f"const uint8_t nor_output[{LANE_COUNT}] = {{\n")
        for index in range(0, LANE_COUNT, 16):
            chunk = output[index:index + 16]
            file.write("    " + ", ".join(str(value) for value in chunk))
            if index + 16 < LANE_COUNT:
                file.write(",")
            file.write("\n")
        file.write("};\n")



def decimal_scientific(value: Decimal, significant_digits: int = 6) -> str:
    """Format a Decimal in compact scientific notation."""
    if value == 0:
        return "0"

    sign = "-" if value < 0 else ""
    value = abs(value)
    exponent = value.adjusted()
    mantissa = value.scaleb(-exponent)

    # Keep display short but stable.
    shown = f"{mantissa:.{significant_digits - 1}f}"
    return f"{sign}{shown}e{exponent}"


def compute_readout_accuracy_for_effective_bits(
    effective_bits: int,
    adc_range_volts: str = "3.3",
) -> dict:
    """
    Compute the actual scalar-readout voltage accuracy required for a core
    with a fixed number of effective output bits.

    This is the physically relevant calculation for the practical core model.

    For a core with b independent binary NOR lanes:

        possible states = 2 ** b

    If those states are spread across an ADC/voltmeter range V_range:

        voltage_step = V_range / (2 ** b)

    To decode safely by nearest voltage state:

        minimum required accuracy = voltage_step / 2

    For engineering margin, a conservative target is:

        recommended accuracy = voltage_step / 4

    Example for b=12 and V_range=3.3 V:

        states = 4096
        step = 805.664 µV
        required = ±402.832 µV
        recommended = ±201.416 µV
    """

    if effective_bits <= 0:
        raise ValueError("effective_bits must be greater than zero.")

    adc_range = Decimal(str(adc_range_volts))
    state_count = Decimal(2) ** Decimal(effective_bits)

    voltage_step = adc_range / state_count
    required_abs_voltage = voltage_step / Decimal(2)
    recommended_abs_voltage = voltage_step / Decimal(4)

    return {
        "effective_bits": effective_bits,
        "state_count": int(state_count),
        "adc_range_volts": adc_range,
        "voltage_step": voltage_step,
        "required_abs_voltage": required_abs_voltage,
        "recommended_abs_voltage": recommended_abs_voltage,
    }
def print_core_readout_accuracy(accuracy: dict) -> None:
    print()
    print("Minimum Voltage Accuracy Required for Lane Reconstruction")
    print("Practical scenario")
    print("---------------------------------------------------------")
    print(f"Optical core size:                 {accuracy['effective_bits']} NOR lanes")
    print(f"Possible lane output combinations: {accuracy['state_count']}")
    print(f"ADC / voltmeter range assumed:     {accuracy['adc_range_volts']} V")
    print()
    print(f"Voltage spacing per lane state:    {decimal_scientific(accuracy['voltage_step'], 6)} V")
    print(f"Minimum voltage accuracy:          +/-{decimal_scientific(accuracy['required_abs_voltage'], 6)} V")
    print(f"Recommended safer target:          +/-{decimal_scientific(accuracy['recommended_abs_voltage'], 6)} V")
    print()
    print("Meaning:")
    print("  This is the practical hardware requirement for reconstructing the")
    print("  lane-state output of the scalar optical core.")
    print("  For a 12-lane core over 3.3 V:")
    print("    2^12 = 4096 possible lane output combinations")
    print("    3.3 V / 4096 = about 805.7 uV per state")
    print("    minimum half-step accuracy = about +/-402.8 uV")
    print("    safer quarter-step target = about +/-201.4 uV")
def build_random_command(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        args.random_script,
        "--map",
        args.map_cfg,
        "--amp-min",
        "1",
        "--amp-max",
        "1",
        "--seed",
        str(args.seed),
        "--out",
        args.input2_out,
    ]

    if args.input2_active_probability is not None:
        command += ["--active-probability", str(args.input2_active_probability)]
    else:
        command += ["--active-count", str(args.input2_active_count)]

    return command


def build_decoder_command(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        args.decoder_script,
        "--map",
        args.map_cfg,
        "--actual-nor",
        f"{args.nor_out}.json",
        "--precision",
        str(args.precision),
        "--voltage-tolerance",
        str(args.voltage_tolerance),
        "--out",
        args.decode_out,
        "--vector-out",
        args.decoded_vector_out,
    ]

    if args.measured_voltage is None:
        command += ["--simulate-from-actual"]
    else:
        command += ["--measured-voltage", str(args.measured_voltage)]

    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Optical GPU Driver proof pipeline.")

    parser.add_argument("--map-script", default="Wavelength_LCD_Encoder.py")
    parser.add_argument("--codec-script", default="optical_lcd_codec.py")
    parser.add_argument("--random-script", default="random_input.py")
    parser.add_argument("--nor-script", default="nor.py")
    parser.add_argument("--decoder-script", default="nor_voltage_decoder.py")

    parser.add_argument("--map-cfg", default="integer_log_prime_wavelength_map.cfg")
    parser.add_argument("--force-map", action="store_true", help="Regenerate wavelength map even if cfg already exists.")

    parser.add_argument(
    "--input1-channels",
    default="1,5,12",
    help="Input 1 channels, e.g. 1,5,12. Must exist in the generated map."
)
    parser.add_argument("--input1-out", default="input1")

    parser.add_argument("--input2-out", default="input2")
    parser.add_argument(
    "--input2-active-count",
    type=int,
    default=6,
    help="Number of random active Input 2 channels. For a 12-lane core, keep this <= 12."
)
    parser.add_argument("--input2-active-probability", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--nor-out", default="nor_result")
    parser.add_argument("--skip-external-nor", action="store_true", help="Do not call nor.py; main.py computes mapped NOR itself.")

    parser.add_argument("--measured-voltage", default=None, help="Actual measured voltage. If absent, decoder simulates from actual mapped NOR.")
    parser.add_argument("--precision", type=int, default=5000)
    parser.add_argument("--voltage-tolerance", default="0.000001")
    parser.add_argument("--decode-out", default="unknown_decode_report.json")
    parser.add_argument("--decoded-vector-out", default="decoded_unknown_vector.txt")
    parser.add_argument("--adc-range-volts", default="3.3", help="ADC/voltmeter full-scale range used for ideal bits estimate.")
    parser.add_argument(
        "--effective-bits",
        type=int,
        default=12,
        help="Effective output bits per scalar-readout core. Default: 12."
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    require_file(args.map_script, "wavelength mapper script")
    require_file(args.codec_script, "LCD codec script")
    require_file(args.random_script, "random input script")
    require_file(args.decoder_script, "mean-voltage decoder script")

    if args.force_map or not Path(args.map_cfg).exists():
        run_step("1. Generate wavelength/log-prime LCD map", [sys.executable, args.map_script])
    else:
        print(f"Using existing wavelength map: {args.map_cfg}")

    require_file(args.map_cfg, "wavelength map cfg")
    mapped_pixels = load_mapped_pixels(args.map_cfg)

    print()
    print(f"Mapped optical wavelength lanes: {len(mapped_pixels)}")
    print(f"Mapped LCD pixels: {sorted(mapped_pixels)}")

    run_step(
        "2. Generate Input 1 vector/pattern",
        [
            sys.executable,
            args.codec_script,
            "encode",
            "--map",
            args.map_cfg,
            "--channels",
            args.input1_channels,
            "--out",
            args.input1_out,
        ],
    )

    run_step("3. Generate random Input 2 vector/pattern", build_random_command(args))

    input1_json = f"{args.input1_out}.json"
    input2_json = f"{args.input2_out}.json"
    require_file(input1_json, "Input 1 json")
    require_file(input2_json, "Input 2 json")

    if not args.skip_external_nor:
        require_file(args.nor_script, "NOR script")
        run_step(
            "4. Run external NOR script for debug",
            [
                sys.executable,
                args.nor_script,
                "--input1",
                input1_json,
                "--input2",
                input2_json,
                "--out",
                args.nor_out,
            ],
        )
    else:
        print("Skipping external NOR script; main.py will compute mapped NOR output.")

    raw_input1 = load_128(input1_json)
    raw_input2 = load_128(input2_json)

    input1 = mask_to_mapped_lanes(raw_input1, mapped_pixels)
    input2 = mask_to_mapped_lanes(raw_input2, mapped_pixels)
    mapped_nor = mapped_nor_128(input1, input2, mapped_pixels)

    # This intentionally overwrites nor_result.* so the decoder sees the physical optical NOR result,
    # not the logical complement of all 128 LCD columns.
    save_nor_result(args.nor_out, input1, input2, mapped_nor, mapped_pixels)

    run_step("5. Decode unknown mean voltage back into active mapped channels", build_decoder_command(args))

    decoded_vector = load_128(args.decoded_vector_out)
    decoded_vector = mask_to_mapped_lanes(decoded_vector, mapped_pixels)
    matches = decoded_vector == mapped_nor

    accuracy = compute_readout_accuracy_for_effective_bits(
        effective_bits=args.effective_bits,
        adc_range_volts=args.adc_range_volts,
    )
    print_core_readout_accuracy(accuracy)

    print()
    print("=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)
    print(f"Mapped optical lanes:        {len(mapped_pixels)}")
    print(f"Input 1 active mapped lanes: {sum(input1)}")
    print(f"Input 2 active mapped lanes: {sum(input2)}")
    print(f"NOR active mapped lanes:     {sum(mapped_nor)}")
    print(f"Decoded active mapped lanes: {sum(decoded_vector)}")
    print(f"Decoded matches mapped NOR:  {matches}")
    print(f"Wavelength map:              {args.map_cfg}")
    print(f"Input 1 JSON:                {input1_json}")
    print(f"Input 2 JSON:                {input2_json}")
    print(f"Mapped NOR JSON:             {args.nor_out}.json")
    print(f"Decode report:               {args.decode_out}")
    print(f"Decoded vector:              {args.decoded_vector_out}")

    if not matches:
        raise RuntimeError("Decoded vector did not match mapped optical NOR output.")


if __name__ == "__main__":
    main()
