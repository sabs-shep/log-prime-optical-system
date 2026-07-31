#!/usr/bin/env python3
"""
random_input2_generator.py

Standalone random Input 2 pattern generator for the optical LCD core.

Purpose:
- Load your log-prime wavelength map from .cfg or .csv.
- Randomly choose channels for Input 2.
- Generate a 128x64 LCD pattern.
- Save the pattern as:
    input2_random.json  machine-readable
    input2_random.txt   human-readable
    input2_random.pbm   monochrome PBM preview
    input2_random.c     SSD1306-style byte array

Conventions:
- x = 0..127 = wavelength column axis.
- y = 0..63  = amplitude axis.
- 1 = open / pass light.
- 0 = closed / block light.

Examples:
    python random_input2_generator.py --map integer_log_prime_wavelength_map.cfg --active-count 8 --out input2_random
    python random_input2_generator.py --map integer_log_prime_wavelength_map.csv --active-probability 0.25 --seed 1234 --out input2_seeded
    python random_input2_generator.py --map map.cfg --active-count 12 --amp-min 8 --amp-max 64 --vertical-mode bottom
"""

import argparse
import configparser
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

LCD_WIDTH = 128
LCD_HEIGHT = 1


@dataclass(frozen=True)
class ChannelInfo:
    channel: int
    prime: int
    log2_prime: float
    integer_pixel: int
    actual_pixel_wavelength_nm: Optional[float] = None
    actual_pixel_frequency_thz: Optional[float] = None


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def clean_header(header: str) -> str:
    """
    Makes CSV loading tolerant of pasted rich-text headers.
    """
    if header is None:
        return ""

    text = str(header).strip()
    text = text.replace('<strong data-lexical-text="true">', '')
    text = text.replace('</strong>', '')
    text = text.replace(',', '')
    return text.strip()


def optional_float(mapping, key: str) -> Optional[float]:
    if key not in mapping:
        return None

    value = mapping[key]

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return float(value)


def load_channel_map(path: str | Path) -> Dict[int, ChannelInfo]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Map file not found: {path}")

    if path.suffix.lower() == ".cfg":
        channels = load_channel_map_cfg(path)
    elif path.suffix.lower() == ".csv":
        channels = load_channel_map_csv(path)
    else:
        raise ValueError("Map file must be .cfg or .csv")

    validate_channel_map(channels)
    return channels


def load_channel_map_cfg(path: Path) -> Dict[int, ChannelInfo]:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")

    channels: Dict[int, ChannelInfo] = {}

    for section in cfg.sections():
        if not section.lower().startswith("channel_"):
            continue

        item = cfg[section]

        channel = int(item["channel"])
        prime = int(item["prime"])
        log2_prime = float(item["log2_prime"])
        integer_pixel = int(item["integer_pixel"])

        channels[channel] = ChannelInfo(
            channel=channel,
            prime=prime,
            log2_prime=log2_prime,
            integer_pixel=integer_pixel,
            actual_pixel_wavelength_nm=optional_float(item, "actual_pixel_wavelength_nm"),
            actual_pixel_frequency_thz=optional_float(item, "actual_pixel_frequency_thz"),
        )

    return channels


def load_channel_map_csv(path: Path) -> Dict[int, ChannelInfo]:
    channels: Dict[int, ChannelInfo] = {}

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            clean = {clean_header(key): value for key, value in row.items()}

            channel = int(clean["channel"])
            prime = int(clean["prime"])
            log2_prime = float(clean["log2_prime"])
            integer_pixel = int(clean["integer_pixel"])

            channels[channel] = ChannelInfo(
                channel=channel,
                prime=prime,
                log2_prime=log2_prime,
                integer_pixel=integer_pixel,
                actual_pixel_wavelength_nm=optional_float(clean, "actual_pixel_wavelength_nm"),
                actual_pixel_frequency_thz=optional_float(clean, "actual_pixel_frequency_thz"),
            )

    return channels


def validate_channel_map(channels: Dict[int, ChannelInfo]) -> None:
    if not channels:
        raise ValueError("No channel sections/rows found in map file.")

    used_pixels: Dict[int, int] = {}

    for channel, info in channels.items():
        if not 0 <= info.integer_pixel < LCD_WIDTH:
            raise ValueError(
                f"Channel {channel} maps to invalid pixel {info.integer_pixel}; expected 0..{LCD_WIDTH - 1}."
            )

        if info.integer_pixel in used_pixels:
            other_channel = used_pixels[info.integer_pixel]
            raise ValueError(
                f"Duplicate LCD x pixel {info.integer_pixel}: channel {other_channel} and channel {channel}."
            )

        used_pixels[info.integer_pixel] = channel


def generate_random_selection(
    channels: Dict[int, ChannelInfo],
    active_count: Optional[int],
    active_probability: Optional[float],
    amp_min: int,
    amp_max: int,
    seed: Optional[int],
) -> Dict[int, int]:
    """
    Return a random Input 2 selection as:
        {channel: amplitude_level}

    Use either active_count or active_probability.
    If neither is supplied, default to roughly 25 percent of channels.
    """
    rng = random.Random(seed)
    available = sorted(channels.keys())

    amp_min = clamp_int(amp_min, 0, LCD_HEIGHT)
    amp_max = clamp_int(amp_max, 0, LCD_HEIGHT)

    if amp_min > amp_max:
        raise ValueError("--amp-min cannot be greater than --amp-max")

    if active_count is not None and active_probability is not None:
        raise ValueError("Use either --active-count or --active-probability, not both.")

    selected: Dict[int, int] = {}

    if active_count is None and active_probability is None:
        active_count = max(1, len(available) // 4)

    if active_count is not None:
        active_count = clamp_int(active_count, 0, len(available))
        chosen = rng.sample(available, active_count)

        for channel in chosen:
            selected[channel] = rng.randint(amp_min, amp_max)

        return selected

    assert active_probability is not None

    if not 0.0 <= active_probability <= 1.0:
        raise ValueError("--active-probability must be between 0.0 and 1.0")

    for channel in available:
        if rng.random() <= active_probability:
            selected[channel] = rng.randint(amp_min, amp_max)

    return selected


def amplitude_to_y_indices(amplitude: int, vertical_mode: str) -> List[int]:
    amplitude = clamp_int(amplitude, 0, LCD_HEIGHT)

    if amplitude == 0:
        return []

    if vertical_mode == "full":
        return list(range(LCD_HEIGHT))

    if vertical_mode == "bottom":
        return list(range(LCD_HEIGHT - amplitude, LCD_HEIGHT))

    if vertical_mode == "top":
        return list(range(0, amplitude))

    if vertical_mode == "center":
        start = (LCD_HEIGHT - amplitude) // 2
        return list(range(start, start + amplitude))

    raise ValueError("vertical_mode must be bottom, top, center, or full")


def build_pattern(
    channels: Dict[int, ChannelInfo],
    selected: Dict[int, int],
    vertical_mode: str,
) -> List[List[int]]:
    pattern = [[0 for _ in range(LCD_WIDTH)] for _ in range(LCD_HEIGHT)]

    for channel, amplitude in selected.items():
        info = channels[channel]
        x = info.integer_pixel

        for y in amplitude_to_y_indices(amplitude, vertical_mode):
            pattern[y][x] = 1

    return pattern


def save_txt(pattern: List[List[int]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in pattern:
            file.write("".join("#" if cell else "." for cell in row) + "\n")


def save_pbm(pattern: List[List[int]], path: Path) -> None:
    with path.open("w", encoding="ascii") as file:
        file.write("P1\n")
        file.write(f"{LCD_WIDTH} {LCD_HEIGHT}\n")
        for row in pattern:
            file.write(" ".join(str(cell) for cell in row) + "\n")


def save_json(
    pattern: List[List[int]],
    channels: Dict[int, ChannelInfo],
    selected: Dict[int, int],
    path: Path,
    seed: Optional[int],
    vertical_mode: str,
) -> None:
    active = []

    for channel in sorted(selected):
        info = channels[channel]
        active.append({
            "channel": channel,
            "amplitude": selected[channel],
            "x_pixel": info.integer_pixel,
            "prime": info.prime,
            "log2_prime": info.log2_prime,
            "actual_pixel_wavelength_nm": info.actual_pixel_wavelength_nm,
            "actual_pixel_frequency_thz": info.actual_pixel_frequency_thz,
        })

    data = {
        "name": "random_input2_pattern",
        "lcd_width": LCD_WIDTH,
        "lcd_height": LCD_HEIGHT,
        "seed": seed,
        "vertical_mode": vertical_mode,
        "convention": "pattern[y][x], 1=open/pass light, 0=closed/block light",
        "active_channels": active,
        "pattern": pattern,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_c_array(pattern: List[List[int]], path: Path, array_name: str = "input2_pattern") -> None:
    """
    Save SSD1306-style page bytes:
    page 0 = y 0..7, x 0..127
    page 1 = y 8..15, x 0..127
    ...
    """
    bytes_out: List[int] = []

    for page in range(LCD_HEIGHT // 8):
        for x in range(LCD_WIDTH):
            value = 0
            for bit in range(8):
                y = page * 8 + bit
                if pattern[y][x]:
                    value |= 1 << bit
            bytes_out.append(value)

    with path.open("w", encoding="utf-8") as file:
        file.write("#include <stdint.h>\n\n")
        file.write(f"const uint8_t {array_name}[{len(bytes_out)}] = {{\n")

        for i in range(0, len(bytes_out), 16):
            chunk = bytes_out[i:i + 16]
            file.write("    " + ", ".join(f"0x{byte:02X}" for byte in chunk))
            if i + 16 < len(bytes_out):
                file.write(",")
            file.write("\n")

        file.write("};\n")


def save_outputs(
    pattern: List[List[int]],
    channels: Dict[int, ChannelInfo],
    selected: Dict[int, int],
    out_prefix: str,
    seed: Optional[int],
    vertical_mode: str,
) -> None:
    base = Path(out_prefix)

    save_txt(pattern, base.with_suffix(".txt"))
    save_pbm(pattern, base.with_suffix(".pbm"))
    save_json(pattern, channels, selected, base.with_suffix(".json"), seed, vertical_mode)
    save_c_array(pattern, base.with_suffix(".c"))

    print(f"Saved {base.with_suffix('.txt')}")
    print(f"Saved {base.with_suffix('.pbm')}")
    print(f"Saved {base.with_suffix('.json')}")
    print(f"Saved {base.with_suffix('.c')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone random Input 2 LCD pattern generator.")

    parser.add_argument("--map", required=True, help="Input wavelength map .cfg or .csv")
    parser.add_argument("--out", default="input2_random", help="Output file prefix")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--active-count", type=int, default=None, help="Exact number of channels to activate")
    group.add_argument("--active-probability", type=float, default=None, help="Chance each channel is active, 0.0 to 1.0")

    parser.add_argument("--amp-min", type=int, default=1, help="Minimum amplitude level, 0..64")
    parser.add_argument("--amp-max", type=int, default=LCD_HEIGHT, help="Maximum amplitude level, 0..64")
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable random patterns")
    parser.add_argument(
        "--vertical-mode",
        choices=["bottom", "top", "center", "full"],
        default="bottom",
        help="How amplitude maps onto the 64-pixel vertical axis"
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    channels = load_channel_map(args.map)

    selected = generate_random_selection(
        channels=channels,
        active_count=args.active_count,
        active_probability=args.active_probability,
        amp_min=args.amp_min,
        amp_max=args.amp_max,
        seed=args.seed,
    )

    pattern = build_pattern(
        channels=channels,
        selected=selected,
        vertical_mode=args.vertical_mode,
    )

    save_outputs(
        pattern=pattern,
        channels=channels,
        selected=selected,
        out_prefix=args.out,
        seed=args.seed,
        vertical_mode=args.vertical_mode,
    )

    print()
    print("Random Input 2 Summary")
    print("----------------------")
    print(f"Mapped channels available: {len(channels)}")
    print(f"Active channels selected:  {len(selected)}")
    print(f"Seed:                      {args.seed}")
    print(f"Amplitude range:           {args.amp_min}..{args.amp_max}")
    print(f"Vertical mode:             {args.vertical_mode}")
    print()

    for channel in sorted(selected):
        info = channels[channel]
        print(
            f"channel={channel:>3}  "
            f"amp={selected[channel]:>2}  "
            f"x={info.integer_pixel:>3}  "
            f"prime={info.prime}  "
            f"wavelength_nm={info.actual_pixel_wavelength_nm}"
        )


if __name__ == "__main__":
    main()
