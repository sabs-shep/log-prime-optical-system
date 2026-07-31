#!/usr/bin/env python3
"""
nor_voltage_decoder.py

Fast unknown mean-voltage decoder for the log-prime optical NOR system.

This is the replacement decoder for the current Optical GPU Driver prototype.

Core model
----------
The optical output voltage is treated as the mean log-prime channel voltage:

    V_measured = V_OFFSET + K * mean(ln(p_i) for active mapped channels)

So:

    L = (V_measured - V_OFFSET) / K

For an unknown active count k:

    k * L = sum(ln(p_i)) = ln(product(p_i))

Therefore:

    product_guess = round(exp(k * L))

Instead of enumerating all subsets, this decoder uses the known prime mapping.
It tries possible active counts, reconstructs the candidate product, then extracts
which mapped primes are present using a product tree and grouped GCD searches.

Why product tree?
-----------------
The simple decoder checks every prime for every active-count guess.
This version groups primes into a binary product tree. For each subtree:

    shared = gcd(target_product, subtree_product)

If shared == 1:
    no active prime in that subtree

If shared == subtree_product:
    the entire subtree is active

Otherwise:
    recurse into the subtree

This avoids brute-force subset enumeration and avoids naive trial division over
every prime in the common cases.

Typical usage
-------------
Decode from actual NOR vector by simulating measured voltage:

    py -3.13 nor_voltage_decoder.py --map integer_log_prime_wavelength_map.cfg --actual-nor nor_result.json --simulate-from-actual

Decode from a real measured voltage and compare against the actual vector:

    py -3.13 nor_voltage_decoder.py --map integer_log_prime_wavelength_map.cfg --measured-voltage 1.543685 --actual-nor nor_result.json

Quiet run:

    py -3.13 nor_voltage_decoder.py --map integer_log_prime_wavelength_map.cfg --actual-nor nor_result.json --simulate-from-actual --quiet
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LANE_COUNT = 128
DEFAULT_V_OFFSET = Decimal("1.0")
DEFAULT_K = Decimal("0.1")


@dataclass(slots=True)
class Channel:
    channel: int
    prime: int
    pixel: int
    ln_prime: Optional[Decimal] = None


@dataclass(slots=True)
class ProductTreeNode:
    product: int
    channels: List[Channel]
    left: Optional["ProductTreeNode"] = None
    right: Optional["ProductTreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------

def short_decimal(value: Decimal, significant_digits: int = 12) -> str:
    """Return a compact scientific-notation string for huge Decimals."""
    if value.is_zero():
        return "0"

    sign = "-" if value < 0 else ""
    value = abs(value)
    exponent = value.adjusted()
    mantissa = value.scaleb(-exponent)
    return f"{sign}{mantissa:.{significant_digits - 1}f}e{exponent}"


def decimal_to_json(value: Decimal) -> str:
    return str(value)


# -----------------------------------------------------------------------------
# Map loading
# -----------------------------------------------------------------------------

def clean_header(header: str) -> str:
    """Make CSV loading tolerant of pasted rich-text headers."""
    if header is None:
        return ""

    text = str(header).strip()
    text = text.replace('<strong data-lexical-text="true">', '')
    text = text.replace('</strong>', '')
    text = text.replace(',', '')
    return text.strip()


def load_map(path: str | Path) -> List[Channel]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Map file not found: {path}")

    if path.suffix.lower() == ".cfg":
        channels = load_map_cfg(path)
    elif path.suffix.lower() == ".csv":
        channels = load_map_csv(path)
    else:
        raise ValueError("Map must be .cfg or .csv")

    return validate_channels(channels)


def load_map_cfg(path: Path) -> List[Channel]:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")

    channels: List[Channel] = []

    for section in cfg.sections():
        if not section.lower().startswith("channel_"):
            continue

        item = cfg[section]
        channels.append(
            Channel(
                channel=int(item["channel"]),
                prime=int(item["prime"]),
                pixel=int(item["integer_pixel"]),
            )
        )

    return channels


def load_map_csv(path: Path) -> List[Channel]:
    channels: List[Channel] = []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            clean = {clean_header(key): value for key, value in row.items()}
            channels.append(
                Channel(
                    channel=int(clean["channel"]),
                    prime=int(clean["prime"]),
                    pixel=int(clean["integer_pixel"]),
                )
            )

    return channels


def validate_channels(channels: List[Channel]) -> List[Channel]:
    if not channels:
        raise ValueError("No channels found in map.")

    seen_pixels: Dict[int, int] = {}
    seen_primes: Dict[int, int] = {}

    for channel in channels:
        if not 0 <= channel.pixel < LANE_COUNT:
            raise ValueError(f"Channel {channel.channel} has invalid pixel {channel.pixel}.")

        if channel.pixel in seen_pixels:
            raise ValueError(
                f"Duplicate pixel {channel.pixel}: "
                f"channel {seen_pixels[channel.pixel]} and {channel.channel}."
            )

        if channel.prime in seen_primes:
            raise ValueError(
                f"Duplicate prime {channel.prime}: "
                f"channel {seen_primes[channel.prime]} and {channel.channel}."
            )

        seen_pixels[channel.pixel] = channel.channel
        seen_primes[channel.prime] = channel.channel

    # Keep logical channel order stable.
    return sorted(channels, key=lambda item: item.channel)


def precompute_logs(channels: List[Channel]) -> None:
    for channel in channels:
        channel.ln_prime = Decimal(channel.prime).ln()


# -----------------------------------------------------------------------------
# 128 vector loading
# -----------------------------------------------------------------------------

def load_128_vector(path: str | Path) -> List[int]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            return normalize_list_or_pattern(data)

        if isinstance(data, dict):
            for key in ["output", "values", "lanes", "input", "input1", "input2"]:
                if key in data:
                    return normalize_list_or_pattern(data[key])

            if "pattern" in data:
                return compress_pattern_to_vector(data["pattern"])

        raise ValueError("JSON must contain output/values/pattern")

    text = path.read_text(encoding="utf-8").strip()

    if text.startswith("P1"):
        return load_pbm_vector(text)

    tokens = [token for token in re.split(r"[\s,;]+", text) if token]
    values = [1 if Decimal(token) > 0 else 0 for token in tokens]

    if len(values) != LANE_COUNT:
        raise ValueError(f"Expected {LANE_COUNT} values, got {len(values)} from {path}")

    return values


def normalize_list_or_pattern(value: Any) -> List[int]:
    if not isinstance(value, list):
        raise ValueError("Expected list")

    if len(value) == LANE_COUNT and all(not isinstance(item, list) for item in value):
        return [1 if Decimal(str(item)) > 0 else 0 for item in value]

    if value and all(isinstance(row, list) for row in value):
        return compress_pattern_to_vector(value)

    raise ValueError("Expected 128-number list or 2D pattern")


def compress_pattern_to_vector(pattern: List[List[Any]]) -> List[int]:
    if not pattern:
        raise ValueError("Pattern is empty")

    if len(pattern[0]) != LANE_COUNT:
        raise ValueError(f"Pattern width expected {LANE_COUNT}, got {len(pattern[0])}")

    for row_index, row in enumerate(pattern):
        if len(row) != LANE_COUNT:
            raise ValueError(f"Pattern row {row_index} has wrong width")

    return [
        1 if any(Decimal(str(pattern[y][x])) > 0 for y in range(len(pattern))) else 0
        for x in range(LANE_COUNT)
    ]


def load_pbm_vector(text: str) -> List[int]:
    lines: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)

    if not lines or lines[0] != "P1":
        raise ValueError("Invalid PBM")

    width, height = [int(part) for part in lines[1].split()[:2]]

    if width != LANE_COUNT:
        raise ValueError(f"Expected PBM width {LANE_COUNT}, got {width}")

    tokens: List[str] = []
    for line in lines[2:]:
        tokens.extend(line.split())

    values = [1 if Decimal(token) > 0 else 0 for token in tokens]

    if len(values) != width * height:
        raise ValueError(f"PBM expected {width * height} pixels, got {len(values)}")

    if height == 1:
        return values

    pattern: List[List[int]] = []

    for y in range(height):
        start = y * width
        pattern.append(values[start:start + width])

    return compress_pattern_to_vector(pattern)


# -----------------------------------------------------------------------------
# Voltage model and vectors
# -----------------------------------------------------------------------------

def mean_voltage_for_channels(active_channels: List[Channel], v_offset: Decimal, k_value: Decimal) -> Decimal:
    if not active_channels:
        return Decimal("0")

    total_log = sum(channel.ln_prime for channel in active_channels if channel.ln_prime is not None)
    mean_log = total_log / Decimal(len(active_channels))
    return v_offset + k_value * mean_log


def vector_from_active_channels(active_channels: List[Channel]) -> List[int]:
    vector = [0 for _ in range(LANE_COUNT)]

    for channel in active_channels:
        vector[channel.pixel] = 1

    return vector


def active_channels_from_vector(channels: List[Channel], vector: List[int]) -> List[Channel]:
    pixel_to_channel = {channel.pixel: channel for channel in channels}
    active: List[Channel] = []

    for pixel, value in enumerate(vector):
        if value and pixel in pixel_to_channel:
            active.append(pixel_to_channel[pixel])

    return sorted(active, key=lambda item: item.channel)


# -----------------------------------------------------------------------------
# Product-tree decoding
# -----------------------------------------------------------------------------

def build_product_tree(channels: List[Channel]) -> ProductTreeNode:
    if not channels:
        raise ValueError("Cannot build product tree from no channels.")

    if len(channels) == 1:
        return ProductTreeNode(
            product=channels[0].prime,
            channels=channels,
        )

    midpoint = len(channels) // 2
    left = build_product_tree(channels[:midpoint])
    right = build_product_tree(channels[midpoint:])

    return ProductTreeNode(
        product=left.product * right.product,
        channels=channels,
        left=left,
        right=right,
    )


def extract_channels_from_product_tree(node: ProductTreeNode, target_product: int) -> List[Channel]:
    """
    Extract active channels from target_product using grouped gcd recursion.

    This is the non-linear factor recovery step:
    - gcd == 1 means no active factor in this subtree.
    - gcd == subtree product means the whole subtree is active.
    - otherwise recurse into the subtree.
    """
    shared = math.gcd(target_product, node.product)

    if shared == 1:
        return []

    if shared == node.product:
        return node.channels.copy()

    if node.is_leaf:
        return node.channels.copy() if shared == node.product else []

    decoded: List[Channel] = []

    if node.left is not None:
        decoded.extend(extract_channels_from_product_tree(node.left, shared))

    if node.right is not None:
        decoded.extend(extract_channels_from_product_tree(node.right, shared))

    return decoded


def product_is_known_subset(product_tree: ProductTreeNode, target_product: int) -> bool:
    if target_product <= 1:
        return False

    shared = math.gcd(target_product, product_tree.product)
    return shared == target_product


# -----------------------------------------------------------------------------
# Active-count pruning
# -----------------------------------------------------------------------------

def build_count_windows(channels: List[Channel]) -> Dict[int, Tuple[Decimal, Decimal]]:
    """
    For each active count k, compute min and max possible mean ln(prime).

    This prunes impossible active counts before expensive Decimal.exp calls.
    """
    logs = sorted(channel.ln_prime for channel in channels if channel.ln_prime is not None)
    n = len(logs)

    prefix = [Decimal("0")]

    for value in logs:
        prefix.append(prefix[-1] + value)

    windows: Dict[int, Tuple[Decimal, Decimal]] = {}

    for k in range(1, n + 1):
        min_mean = (prefix[k] - prefix[0]) / Decimal(k)
        max_mean = (prefix[n] - prefix[n - k]) / Decimal(k)
        windows[k] = (min_mean, max_mean)

    return windows


def candidate_counts_for_mean(
    normalized_mean_log: Decimal,
    windows: Dict[int, Tuple[Decimal, Decimal]],
    log_tolerance: Decimal,
) -> List[int]:
    candidates: List[Tuple[Decimal, int]] = []

    for active_count, (min_mean, max_mean) in windows.items():
        if min_mean - log_tolerance <= normalized_mean_log <= max_mean + log_tolerance:
            midpoint = (min_mean + max_mean) / Decimal(2)
            candidates.append((abs(normalized_mean_log - midpoint), active_count))

    candidates.sort(key=lambda item: item[0])
    return [active_count for _, active_count in candidates]


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------

def decode_unknown_mean_voltage(
    channels: List[Channel],
    measured_voltage: Decimal,
    v_offset: Decimal,
    k_value: Decimal,
    voltage_tolerance: Decimal,
    product_relative_tolerance: Decimal,
) -> Tuple[List[Channel], Decimal, Decimal, int, Dict[str, Any]]:
    """
    Decode measured mean voltage without enumerating subsets.

    Main steps:
    1. Convert voltage to mean log value.
    2. Prune impossible active counts using min/max mean-log windows.
    3. For each candidate active count, reconstruct candidate product.
    4. Use product-tree gcd extraction to recover channels.
    5. Verify expected voltage and return.
    """
    if measured_voltage == 0:
        return [], Decimal("0"), Decimal("0"), 0, {
            "candidate_counts": 0,
            "exp_attempts": 0,
            "factor_attempts": 0,
        }

    product_tree = build_product_tree(channels)
    windows = build_count_windows(channels)

    normalized_mean_log = (measured_voltage - v_offset) / k_value
    log_tolerance = abs(voltage_tolerance / k_value)
    candidate_counts = candidate_counts_for_mean(normalized_mean_log, windows, log_tolerance)

    best_active: List[Channel] = []
    best_expected = Decimal("0")
    best_error: Optional[Decimal] = None
    best_count = 0

    exp_attempts = 0
    factor_attempts = 0
    gcd_extractions = 0

    for active_count in candidate_counts:
        target_log_product = normalized_mean_log * Decimal(active_count)

        exp_attempts += 1
        target_product_decimal = target_log_product.exp()
        target_product_int = int(target_product_decimal.to_integral_value(rounding=ROUND_HALF_EVEN))

        if target_product_int <= 1:
            continue

        rounded_distance = abs(Decimal(target_product_int) - target_product_decimal)
        allowed_distance = max(Decimal("1"), abs(target_product_decimal) * product_relative_tolerance)

        if rounded_distance > allowed_distance:
            continue

        factor_attempts += 1

        if not product_is_known_subset(product_tree, target_product_int):
            continue

        gcd_extractions += 1
        decoded = extract_channels_from_product_tree(product_tree, target_product_int)

        if len(decoded) != active_count:
            continue

        # Defensive exact reconstruction check.
        reconstructed = 1
        for channel in decoded:
            reconstructed *= channel.prime

        if reconstructed != target_product_int:
            continue

        expected_voltage = mean_voltage_for_channels(decoded, v_offset, k_value)
        error = abs(expected_voltage - measured_voltage)

        if best_error is None or error < best_error:
            best_active = decoded
            best_expected = expected_voltage
            best_error = error
            best_count = active_count

        if error <= voltage_tolerance:
            return decoded, expected_voltage, error, active_count, {
                "candidate_counts": len(candidate_counts),
                "exp_attempts": exp_attempts,
                "factor_attempts": factor_attempts,
                "gcd_extractions": gcd_extractions,
            }

    if best_error is None:
        raise ValueError(
            "No valid subset recovered from measured voltage. "
            f"Candidate active counts after pruning: {candidate_counts}"
        )

    raise ValueError(
        f"Closest recovered subset had error {best_error}, exceeding tolerance {voltage_tolerance}. "
        f"Recovered count was {best_count}."
    )


# -----------------------------------------------------------------------------
# Accuracy estimate
# -----------------------------------------------------------------------------

def required_voltage_accuracy_for_subset(
    decoded_channels: List[Channel],
    k_value: Decimal,
) -> Dict[str, Any]:
    """
    Estimate voltage accuracy required for product rounding to give the same integer product.

    For active count n and product P:
        product_guess = round(exp(n * (V - V_OFFSET) / K))

    To keep product_guess rounding to P:
        abs(dV) approximately < (K / n) * ln(1 + 0.5 / P)

    The returned number is a mathematical ideal bound for the integer-product decoder.
    """
    if not decoded_channels:
        return {
            "active_count": 0,
            "product_digits": 1,
            "required_abs_voltage": Decimal("0"),
            "positive_bound": Decimal("0"),
            "negative_bound": Decimal("0"),
        }

    product = 1
    for channel in decoded_channels:
        product *= channel.prime

    product_decimal = Decimal(product)
    active_count = Decimal(len(decoded_channels))
    half_over_product = Decimal("0.5") / product_decimal

    positive_bound = (k_value / active_count) * (Decimal(1) + half_over_product).ln()
    negative_bound = (k_value / active_count) * (-(Decimal(1) - half_over_product).ln())
    required_abs_voltage = min(positive_bound, negative_bound)

    return {
        "active_count": len(decoded_channels),
        "product_digits": len(str(product)),
        "product_log10": product_decimal.log10(),
        "required_abs_voltage": required_abs_voltage,
        "positive_bound": positive_bound,
        "negative_bound": negative_bound,
    }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_report(
    path: str | Path,
    measured_voltage: Decimal,
    expected_voltage: Decimal,
    error: Decimal,
    decoded_channels: List[Channel],
    decoded_vector: List[int],
    actual_vector: Optional[List[int]],
    v_offset: Decimal,
    k_value: Decimal,
    precision: int,
    stats: Dict[str, Any],
    accuracy: Dict[str, Any],
) -> None:
    decoded_indices = [index for index, value in enumerate(decoded_vector) if value]
    actual_indices = [index for index, value in enumerate(actual_vector) if value] if actual_vector is not None else None

    data = {
        "decoder": "unknown_mean_voltage_decoder_product_tree",
        "model": "V = V_OFFSET + K * mean(ln(prime) for active channels)",
        "precision_digits": precision,
        "v_offset": str(v_offset),
        "k": str(k_value),
        "measured_voltage": str(measured_voltage),
        "expected_voltage_from_decoded_subset": str(expected_voltage),
        "absolute_error": str(error),
        "decoded_active_count": len(decoded_channels),
        "decoded_channels": [channel.channel for channel in decoded_channels],
        "decoded_pixels": [channel.pixel for channel in decoded_channels],
        "decoded_primes": [str(channel.prime) for channel in decoded_channels],
        "decoded_128_vector": decoded_vector,
        "decoded_active_pixel_indices": decoded_indices,
        "actual_128_vector": actual_vector,
        "actual_active_pixel_indices": actual_indices,
        "matches_actual": None if actual_vector is None else decoded_vector == actual_vector,
        "stats": stats,
        "accuracy": {key: str(value) for key, value in accuracy.items()},
    }

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast product-tree decoder for unknown optical NOR subset from mean log-prime voltage."
    )

    parser.add_argument("--map", required=True, help="Channel map .cfg or .csv")
    parser.add_argument("--measured-voltage", default=None, help="Measured mean voltage")
    parser.add_argument("--actual-nor", default=None, help="Optional actual NOR vector to compare against")
    parser.add_argument("--simulate-from-actual", action="store_true", help="Compute measured voltage from --actual-nor")

    parser.add_argument("--v-offset", default="1.0", help="V_OFFSET")
    parser.add_argument("--k", default="0.1", help="K multiplier")
    parser.add_argument("--voltage-tolerance", default="0.000001", help="Allowed voltage error")
    parser.add_argument("--product-relative-tolerance", default="1e-80", help="Rounded product relative tolerance")
    parser.add_argument("--precision", type=int, default=1200, help="Decimal precision digits")

    parser.add_argument("--out", default="unknown_decode_report.json", help="Output report JSON")
    parser.add_argument("--vector-out", default="decoded_unknown_vector.txt", help="Output decoded 128-vector txt")
    parser.add_argument("--quiet", action="store_true", help="Hide decoded channel/pixel lists")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    getcontext().prec = args.precision

    total_start = time.perf_counter()

    channels = load_map(args.map)
    precompute_logs(channels)

    v_offset = Decimal(args.v_offset)
    k_value = Decimal(args.k)
    voltage_tolerance = Decimal(args.voltage_tolerance)
    product_relative_tolerance = Decimal(args.product_relative_tolerance)

    actual_vector = load_128_vector(args.actual_nor) if args.actual_nor else None

    if args.simulate_from_actual:
        if actual_vector is None:
            raise ValueError("--simulate-from-actual requires --actual-nor")

        actual_active = active_channels_from_vector(channels, actual_vector)
        measured_voltage = mean_voltage_for_channels(actual_active, v_offset, k_value)

    elif args.measured_voltage is not None:
        measured_voltage = Decimal(str(args.measured_voltage))

    else:
        raise ValueError("Supply --measured-voltage or use --simulate-from-actual with --actual-nor")

    decode_start = time.perf_counter()

    decoded_channels, expected_voltage, error, recovered_count, stats = decode_unknown_mean_voltage(
        channels=channels,
        measured_voltage=measured_voltage,
        v_offset=v_offset,
        k_value=k_value,
        voltage_tolerance=voltage_tolerance,
        product_relative_tolerance=product_relative_tolerance,
    )

    decode_seconds = time.perf_counter() - decode_start
    total_seconds = time.perf_counter() - total_start

    decoded_vector = vector_from_active_channels(decoded_channels)

    Path(args.vector_out).write_text(
        " ".join(str(value) for value in decoded_vector) + "\n",
        encoding="utf-8"
    )

    accuracy = required_voltage_accuracy_for_subset(decoded_channels, k_value)

    stats["decode_seconds"] = decode_seconds
    stats["total_seconds"] = total_seconds

    save_report(
        path=args.out,
        measured_voltage=measured_voltage,
        expected_voltage=expected_voltage,
        error=error,
        decoded_channels=decoded_channels,
        decoded_vector=decoded_vector,
        actual_vector=actual_vector,
        v_offset=v_offset,
        k_value=k_value,
        precision=args.precision,
        stats=stats,
        accuracy=accuracy,
    )

    print()
    print("Product-Tree Unknown Mean-Voltage Decode")
    print("----------------------------------------")
    print(f"Mapped channels:              {len(channels)}")
    print(f"Recovered active count:       {recovered_count}")
    print(f"Measured voltage:             {short_decimal(measured_voltage)} V")
    print(f"Expected decoded voltage:     {short_decimal(expected_voltage)} V")
    print(f"Absolute error:               {short_decimal(error)} V")
    print(f"Candidate counts after prune: {stats['candidate_counts']}")
    print(f"Decimal exp attempts:         {stats['exp_attempts']}")
    print(f"Factor attempts:              {stats['factor_attempts']}")
    print(f"Product-tree extractions:     {stats['gcd_extractions']}")
    print(f"Decode time:                  {decode_seconds:.6f} s")
    print(f"Total time:                   {total_seconds:.6f} s")

    print()
    print("Minimum Voltage Accuracy Required for Exact Prime Reconstruction")
    print("Diagnostic only")
    print("----------------------------------------------------------------")
    print(f"Prime product digits:              {accuracy['product_digits']}")

    if decoded_channels:
        print(f"log10(product):                    {accuracy['product_log10']:.6f}")
        print(f"Minimum voltage accuracy:          +/-{short_decimal(accuracy['required_abs_voltage'])} V")
        print(f"Positive-side safe error:          +{short_decimal(accuracy['positive_bound'])} V")
        print(f"Negative-side safe error:          -{short_decimal(accuracy['negative_bound'])} V")
        print()
        print("Meaning:")
        print("  This is the strict mathematical tolerance needed for the decoder")
        print("  to reconstruct the exact prime product using round(exp(kL)).")
        print("  This is useful as a diagnostic/proof metric, but it is NOT the")
        print("  practical voltage target for reconstructing the 12-lane core output.")
    else:
        print("Empty optical output: use photodiode dark-current threshold instead of voltage decode.")

    if not args.quiet:
        print()
        print(f"Decoded active channels:      {[channel.channel for channel in decoded_channels]}")
        print(f"Decoded active pixels:        {[channel.pixel for channel in decoded_channels]}")

    if actual_vector is not None:
        print(f"Matches actual NOR vector:    {decoded_vector == actual_vector}")

    print(f"Saved decoded vector:         {args.vector_out}")
    print(f"Saved report:                 {args.out}")


if __name__ == "__main__":
    main()
