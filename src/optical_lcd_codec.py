#!/usr/bin/env python3
"""
optical_lcd_codec.py

Encoder/decoder for the log-prime optical LCD wavelength map.

What this does right now, with no ADM:
- Loads your wavelength/channel map from a .cfg or .csv file.
- Encodes selected channel numbers into a 128x64 LCD mask.
- Each selected channel opens its mapped integer x-column.
- Optional amplitude level 0..64 opens that many vertical pixels in the column.
- Saves the LCD pattern to files:
    - .txt  human-readable 128-column by 64-row mask
    - .json machine-readable pattern and selected channels
    - .pbm  monochrome image file, useful for previewing
    - .c    C/Arduino-style byte array, pages of 8 vertical pixels
- Decodes a saved pattern back into channel IDs by checking which mapped columns are open.

Coordinate convention:
- x = 0..127 is the horizontal wavelength axis.
- y = 0..63 is the vertical amplitude axis.
- pattern[y][x] is 1 for open/pass light, 0 for blocked.

Example usage:
    python optical_lcd_codec.py encode --map integer_log_prime_wavelength_map.cfg --channels 1,2,3 --out lcd_pattern
    python optical_lcd_codec.py encode --map integer_log_prime_wavelength_map.csv --channels 1:64,2:32,40:12 --out lcd_pattern
    python optical_lcd_codec.py decode --map integer_log_prime_wavelength_map.cfg --pattern lcd_pattern.json
"""

import argparse
import configparser
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


LCD_WIDTH = 128
LCD_HEIGHT = 1


@dataclass(frozen=True)
class ChannelInfo:
    channel: int
    prime: int
    log2_prime: float
    integer_pixel: int
    target_wavelength_nm: float | None = None
    actual_pixel_wavelength_nm: float | None = None
    target_frequency_thz: float | None = None
    actual_pixel_frequency_thz: float | None = None


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def load_channel_map(path: str | Path) -> Dict[int, ChannelInfo]:
    """
    Load channel mapping from either .cfg or .csv.

    Required per channel:
        channel
        prime
        log2_prime
        integer_pixel

    Optional per channel:
        target_wavelength_nm
        actual_pixel_wavelength_nm
        target_frequency_thz
        actual_pixel_frequency_thz
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Map file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".cfg":
        return load_channel_map_cfg(path)

    if suffix == ".csv":
        return load_channel_map_csv(path)

    raise ValueError(f"Unsupported map file type: {path.suffix}. Use .cfg or .csv")


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
            target_wavelength_nm=optional_float(item, "target_wavelength_nm"),
            actual_pixel_wavelength_nm=optional_float(item, "actual_pixel_wavelength_nm"),
            target_frequency_thz=optional_float(item, "target_frequency_thz"),
            actual_pixel_frequency_thz=optional_float(item, "actual_pixel_frequency_thz"),
        )

    validate_channel_map(channels)
    return channels


def load_channel_map_csv(path: Path) -> Dict[int, ChannelInfo]:
    channels: Dict[int, ChannelInfo] = {}

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            clean = {clean_header(k): v for k, v in row.items()}

            channel = int(clean["channel"])
            prime = int(clean["prime"])
            log2_prime = float(clean["log2_prime"])
            integer_pixel = int(clean["integer_pixel"])

            channels[channel] = ChannelInfo(
                channel=channel,
                prime=prime,
                log2_prime=log2_prime,
                integer_pixel=integer_pixel,
                target_wavelength_nm=optional_float(clean, "target_wavelength_nm"),
                actual_pixel_wavelength_nm=optional_float(clean, "actual_pixel_wavelength_nm"),
                target_frequency_thz=optional_float(clean, "target_frequency_thz"),
                actual_pixel_frequency_thz=optional_float(clean, "actual_pixel_frequency_thz"),
            )

    validate_channel_map(channels)
    return channels


def clean_header(header: str) -> str:
    """
    Makes the CSV reader tolerant of pasted rich-text headers such as:
        <strong data-lexical-text="true">actual_pixel_wavelength_nm,</strong>
    """
    if header is None:
        return ""

    header = header.strip()
    header = header.replace("<strong data-lexical-text=\"true\">", "")
    header = header.replace("</strong>", "")
    header = header.replace(",", "")
    return header.strip()


def optional_float(mapping, key: str) -> float | None:
    if key not in mapping:
        return None

    value = mapping[key]

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return float(value)


def validate_channel_map(channels: Dict[int, ChannelInfo]) -> None:
    if not channels:
        raise ValueError("No channels found in map file.")

    seen_pixels: Dict[int, int] = {}

    for channel, info in channels.items():
        if not (0 <= info.integer_pixel < LCD_WIDTH):
            raise ValueError(
                f"Channel {channel} maps to invalid x pixel {info.integer_pixel}; "
                f"expected 0..{LCD_WIDTH - 1}."
            )

        if info.integer_pixel in seen_pixels:
            other = seen_pixels[info.integer_pixel]
            raise ValueError(
                f"Duplicate integer pixel {info.integer_pixel}: "
                f"channel {other} and channel {channel}."
            )

        seen_pixels[info.integer_pixel] = channel


def parse_channel_spec(spec: str, max_amplitude: int = LCD_HEIGHT) -> Dict[int | str, int]:
    """
    Parse channel selections.

    Supported forms:
        "1,2,3"          -> channels 1,2,3 at full amplitude 64
        "1:64,2:32"      -> channel 1 amplitude 64, channel 2 amplitude 32
        "all"            -> every mapped channel at full amplitude

    Return:
        {channel_id_or_ALL: amplitude_level_0_to_64}
    """
    spec = spec.strip()

    if not spec:
        raise ValueError("Empty channel spec.")

    if spec.lower() == "all":
        return {"ALL": max_amplitude}

    selected: Dict[int | str, int] = {}

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            channel_text, amplitude_text = part.split(":", 1)
            channel = int(channel_text.strip())
            amplitude = int(amplitude_text.strip())
        else:
            channel = int(part)
            amplitude = max_amplitude

        selected[channel] = clamp_int(amplitude, 0, max_amplitude)

    return selected



def encode_lcd_pattern(
    channels: Dict[int, ChannelInfo],
    selected: Dict[int | str, int],
    lcd_width: int = LCD_WIDTH,
    lcd_height: int = LCD_HEIGHT,
    vertical_mode: str = "bottom"
) -> List[List[int]]:
    """
    Convert selected channels into a 2D LCD mask.
    """
    pattern = [[0 for _ in range(lcd_width)] for _ in range(lcd_height)]

    if "ALL" in selected:
        selected = {channel: selected["ALL"] for channel in channels}

    for channel, amplitude in selected.items():
        if channel not in channels:
            raise KeyError(f"Channel {channel} is not present in the map file.")

        amplitude = clamp_int(int(amplitude), 0, lcd_height)
        if amplitude == 0:
            continue

        x = channels[channel].integer_pixel
        y_indices = amplitude_to_y_indices(amplitude, lcd_height, vertical_mode)

        for y in y_indices:
            pattern[y][x] = 1

    return pattern



def amplitude_to_y_indices(amplitude: int, lcd_height: int, mode: str) -> List[int]:
    amplitude = clamp_int(amplitude, 0, lcd_height)

    if amplitude == 0:
        return []

    if mode == "full":
        return list(range(lcd_height))

    if mode == "bottom":
        return list(range(lcd_height - amplitude, lcd_height))

    if mode == "top":
        return list(range(0, amplitude))

    if mode == "center":
        start = (lcd_height - amplitude) // 2
        return list(range(start, start + amplitude))

    raise ValueError("vertical_mode must be one of: bottom, top, center, full")


def decode_lcd_pattern(
    channels: Dict[int, ChannelInfo],
    pattern: List[List[int]],
    threshold_open_pixels: int = 1
) -> Dict[int, int]:
    """
    Decode a 2D LCD mask back into channel amplitudes.

    Returns:
        {channel_id: number_of_open_pixels_in_mapped_column}
    """
    validate_pattern_shape(pattern)

    decoded: Dict[int, int] = {}

    for channel, info in channels.items():
        x = info.integer_pixel
        open_count = sum(1 for y in range(LCD_HEIGHT) if pattern[y][x])

        if open_count >= threshold_open_pixels:
            decoded[channel] = open_count

    return decoded


def validate_pattern_shape(pattern: List[List[int]]) -> None:
    if len(pattern) != LCD_HEIGHT:
        raise ValueError(f"Pattern has {len(pattern)} rows; expected {LCD_HEIGHT}.")

    for y, row in enumerate(pattern):
        if len(row) != LCD_WIDTH:
            raise ValueError(f"Pattern row {y} has {len(row)} columns; expected {LCD_WIDTH}.")


def save_pattern_txt(pattern: List[List[int]], path: str | Path) -> None:
    """
    Save as plain text. Each row is 128 chars.
    # = open/pass light
    . = closed/block light
    """
    validate_pattern_shape(pattern)
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        for row in pattern:
            f.write("".join("#" if cell else "." for cell in row) + "\n")

def save_pattern_json(
    pattern: List[List[int]],
    channels: Dict[int, ChannelInfo],
    selected: Dict[int | str, int],
    path: str | Path
) -> None:
    validate_pattern_shape(pattern)
    path = Path(path)

    if "ALL" in selected:
        selected = {channel: selected["ALL"] for channel in channels}

    active_columns = []

    for channel, amplitude in selected.items():
        if channel not in channels:
            continue

        info = channels[channel]
        active_columns.append({
            "channel": channel,
            "amplitude": amplitude,
            "x_pixel": info.integer_pixel,
            "prime": info.prime,
            "actual_pixel_wavelength_nm": info.actual_pixel_wavelength_nm,
            "actual_pixel_frequency_thz": info.actual_pixel_frequency_thz,
        })

    data = {
        "lcd_width": LCD_WIDTH,
        "lcd_height": LCD_HEIGHT,
        "coordinate_system": "pattern[y][x], x=0..127 wavelength axis, y=0..63 amplitude axis",
        "active_columns": sorted(active_columns, key=lambda item: item["x_pixel"]),
        "pattern": pattern,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



def load_pattern_json(path: str | Path) -> List[List[int]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pattern = data["pattern"]
    validate_pattern_shape(pattern)
    return pattern


def save_pattern_pbm(pattern: List[List[int]], path: str | Path) -> None:
    """
    Save as ASCII PBM image.
    PBM convention: 1 = black, 0 = white.
    Here we use 1 for open pixels, so open pixels preview as black.
    """
    validate_pattern_shape(pattern)
    path = Path(path)

    with path.open("w", encoding="ascii") as f:
        f.write("P1\n")
        f.write(f"{LCD_WIDTH} {LCD_HEIGHT}\n")
        for row in pattern:
            f.write(" ".join(str(int(cell)) for cell in row) + "\n")


def save_pattern_c_array(pattern: List[List[int]], path: str | Path, array_name: str = "lcd_pattern") -> None:
    """
    Save as a C byte array using SSD1306-like pages.

    Byte layout:
        page 0 = y 0..7, all x 0..127
        page 1 = y 8..15, all x 0..127
        ...
        page 7 = y 56..63, all x 0..127

    Bit 0 is the first row in the page.
    """
    validate_pattern_shape(pattern)
    path = Path(path)

    pages = []

    for page in range(LCD_HEIGHT // 8):
        for x in range(LCD_WIDTH):
            byte = 0
            for bit in range(8):
                y = page * 8 + bit
                if pattern[y][x]:
                    byte |= (1 << bit)
            pages.append(byte)

    with path.open("w", encoding="utf-8") as f:
        f.write("#include <stdint.h>\n\n")
        f.write(f"const uint8_t {array_name}[{len(pages)}] = {{\n")

        for i in range(0, len(pages), 16):
            chunk = pages[i:i + 16]
            f.write("    " + ", ".join(f"0x{value:02X}" for value in chunk))
            if i + 16 < len(pages):
                f.write(",")
            f.write("\n")

        f.write("};\n")

def save_all_pattern_outputs(
    pattern: List[List[int]],
    channels: Dict[int, ChannelInfo],
    selected: Dict[int | str, int],
    out_prefix: str | Path
) -> None:
    out_prefix = Path(out_prefix)

    save_pattern_txt(pattern, out_prefix.with_suffix(".txt"))
    save_pattern_json(pattern, channels, selected, out_prefix.with_suffix(".json"))
    save_pattern_pbm(pattern, out_prefix.with_suffix(".pbm"))
    save_pattern_c_array(pattern, out_prefix.with_suffix(".c"))

    print(f"Saved: {out_prefix.with_suffix('.txt')}")
    print(f"Saved: {out_prefix.with_suffix('.json')}")
    print(f"Saved: {out_prefix.with_suffix('.pbm')}")
    print(f"Saved: {out_prefix.with_suffix('.c')}")


def command_encode(args: argparse.Namespace) -> None:
    channels = load_channel_map(args.map)
    selected = parse_channel_spec(args.channels, max_amplitude=LCD_HEIGHT)

    pattern = encode_lcd_pattern(
        channels=channels,
        selected=selected,
        vertical_mode=args.vertical_mode,
    )

    save_all_pattern_outputs(
        pattern=pattern,
        channels=channels,
        selected=selected,
        out_prefix=args.out,
    )

    decoded = decode_lcd_pattern(channels, pattern)
    print("\nEncoded channels decoded back from generated pattern:")
    for channel in sorted(decoded):
        print(f"  channel {channel}: amplitude {decoded[channel]}, x={channels[channel].integer_pixel}")


def command_decode(args: argparse.Namespace) -> None:
    channels = load_channel_map(args.map)
    pattern = load_pattern_json(args.pattern)
    decoded = decode_lcd_pattern(channels, pattern, threshold_open_pixels=args.threshold)

    print("Decoded channels:")
    for channel in sorted(decoded):
        info = channels[channel]
        print(
            f"  channel {channel}: "
            f"amplitude={decoded[channel]}, "
            f"x={info.integer_pixel}, "
            f"prime={info.prime}, "
            f"actual_wavelength_nm={info.actual_pixel_wavelength_nm}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encoder/decoder for log-prime optical LCD wavelength maps."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Create LCD pattern files from selected channels.")
    encode_parser.add_argument("--map", required=True, help="Input .cfg or .csv channel map file.")
    encode_parser.add_argument(
        "--channels",
        required=True,
        help="Channel selection. Examples: '1,2,3', '1:64,2:32', or 'all'."
    )
    encode_parser.add_argument("--out", default="lcd_pattern", help="Output file prefix.")
    encode_parser.add_argument(
        "--vertical-mode",
        default="bottom",
        choices=["bottom", "top", "center", "full"],
        help="How amplitude level maps onto the 64 vertical pixels."
    )
    encode_parser.set_defaults(func=command_encode)

    decode_parser = subparsers.add_parser("decode", help="Decode saved LCD JSON pattern back into channels.")
    decode_parser.add_argument("--map", required=True, help="Input .cfg or .csv channel map file.")
    decode_parser.add_argument("--pattern", required=True, help="Input .json LCD pattern file.")
    decode_parser.add_argument("--threshold", type=int, default=1, help="Minimum open pixels in a mapped column.")
    decode_parser.set_defaults(func=command_decode)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
