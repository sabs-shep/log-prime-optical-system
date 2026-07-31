from __future__ import annotations
import math
import csv
from dataclasses import dataclass
import configparser


C = 299_792_458.0


@dataclass
class OpticalConfig:
    led_min_nm: float = 380.0
    led_max_nm: float = 740.0

    lcd_pixels_x: int = 128
    lcd_width_mm: float = 21.5

    grating_lines_per_mm: float = 600.0
    diffraction_order: int = 1

    lens_options_mm: tuple = (80.0, 100.0)

    min_channels: int = 32
    max_channels: int = 64

    # Integer gap between actual selected LCD columns.
    # 1 means adjacent columns allowed.
    # 2 means at least one blank/guard column between channels.
    min_integer_pixel_gap: int = 2

    # Maximum allowed snapping error from ideal log-prime wavelength position
    # to chosen integer LCD pixel.
    max_snap_error_px: float = 0.5

    allow_cropped_windows: bool = True
    crop_step_nm: float = 1.0


def is_prime_64bit(n: int) -> bool:
    if n < 2:
        return False

    small_primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
        37, 41, 43, 47
    ]

    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0

    while d % 2 == 0:
        s += 1
        d //= 2

    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue

        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        passed = False

        for _ in range(s - 1):
            x = pow(x, 2, n)

            if x == n - 1:
                passed = True
                break

        if not passed:
            return False

    return True


def next_prime(n: int) -> int:
    if n <= 2:
        return 2

    if n % 2 == 0:
        n += 1

    while not is_prime_64bit(n):
        n += 2

    return n

def prime_ladder(channel_count: int) -> list:
    """
    Return the first `channel_count` raw prime numbers.

    This minimises the prime product size compared with the old power-of-two ladder.

    Old:
        p_i ~= next_prime(2 ** i)

    New:
        p_i = ith prime

    This reduces the required voltage precision for the mean-voltage decoder,
    but it also makes wavelength positions bunch together more, so fewer channels
    may fit without duplicate LCD pixels.
    """

    primes: list[int] = []
    candidate = 2

    while len(primes) < channel_count:
        if is_prime_64bit(candidate):
            primes.append(candidate)

        candidate += 1

    return primes




def frequency_thz_from_wavelength_nm(wavelength_nm: float) -> float:
    return (C / (wavelength_nm * 1e-9)) / 1e12


def wavelength_nm_from_frequency_thz(frequency_thz: float) -> float:
    return (C / (frequency_thz * 1e12)) * 1e9


def grating_x_position_mm(
    wavelength_nm: float,
    focal_length_mm: float,
    grating_lines_per_mm: float,
    diffraction_order: int
) -> float:
    d_mm = 1.0 / grating_lines_per_mm
    wavelength_mm = wavelength_nm * 1e-6

    argument = diffraction_order * wavelength_mm / d_mm

    if argument < -1.0 or argument > 1.0:
        raise ValueError(
            f"Wavelength {wavelength_nm} nm is invalid for order "
            f"{diffraction_order} with {grating_lines_per_mm} lines/mm."
        )

    theta_rad = math.asin(argument)
    return focal_length_mm * math.tan(theta_rad)


def wavelength_to_pixel_float(
    wavelength_nm: float,
    anchor_wavelength_nm: float,
    anchor_pixel: float,
    focal_length_mm: float,
    config: OpticalConfig
) -> float:
    pixel_pitch_mm = config.lcd_width_mm / config.lcd_pixels_x

    x = grating_x_position_mm(
        wavelength_nm=wavelength_nm,
        focal_length_mm=focal_length_mm,
        grating_lines_per_mm=config.grating_lines_per_mm,
        diffraction_order=config.diffraction_order
    )

    x_anchor = grating_x_position_mm(
        wavelength_nm=anchor_wavelength_nm,
        focal_length_mm=focal_length_mm,
        grating_lines_per_mm=config.grating_lines_per_mm,
        diffraction_order=config.diffraction_order
    )

    return anchor_pixel + (x - x_anchor) / pixel_pitch_mm


def pixel_to_wavelength_nm(
    pixel: float,
    anchor_wavelength_nm: float,
    anchor_pixel: float,
    focal_length_mm: float,
    config: OpticalConfig
) -> float:
    pixel_pitch_mm = config.lcd_width_mm / config.lcd_pixels_x

    x_anchor = grating_x_position_mm(
        wavelength_nm=anchor_wavelength_nm,
        focal_length_mm=focal_length_mm,
        grating_lines_per_mm=config.grating_lines_per_mm,
        diffraction_order=config.diffraction_order
    )

    x = x_anchor + (pixel - anchor_pixel) * pixel_pitch_mm
    theta_rad = math.atan(x / focal_length_mm)

    d_mm = 1.0 / config.grating_lines_per_mm
    wavelength_mm = (d_mm / config.diffraction_order) * math.sin(theta_rad)

    return wavelength_mm * 1e6


def fit_log_prime_equation(
    primes: list[int],
    wavelength_min_nm: float,
    wavelength_max_nm: float
) -> tuple[float, float]:
    """
    Fit exact endpoint equation:

        frequency_THz = A * log2(P) + B

    First prime maps to wavelength_max_nm.
    Last prime maps to wavelength_min_nm.
    """
    f_low_thz = frequency_thz_from_wavelength_nm(wavelength_max_nm)
    f_high_thz = frequency_thz_from_wavelength_nm(wavelength_min_nm)

    log_first = math.log2(primes[0])
    log_last = math.log2(primes[-1])

    A_thz = (f_high_thz - f_low_thz) / (log_last - log_first)
    B_thz = f_low_thz - A_thz * log_first

    return A_thz, B_thz

def prime_product_log10(rows: list[dict]) -> float:
    """
    Estimate log10(product of all selected channel primes).

    Smaller is better because smaller prime products reduce the voltage accuracy
    required by the mean-voltage product reconstruction decoder.
    """

    total = 0.0

    for row in rows:
        total += math.log10(row["prime"])

    return total

def first_n_primes(count: int) -> list[int]:
    """
    Return the first `count` raw prime numbers.
    """
    primes: list[int] = []
    candidate = 2

    while len(primes) < count:
        if is_prime_64bit(candidate):
            primes.append(candidate)
        candidate += 1

    return primes


def prime_product_log10_from_primes(primes: list[int]) -> float:
    return sum(math.log10(prime) for prime in primes)


def make_rows_for_primes(
    primes: list[int],
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    focal_length_mm: float,
    config: OpticalConfig
) -> tuple[list[dict], float, float]:
    """
    Map each prime to an LCD pixel using the fitted log-prime wavelength model.
    """
    A_thz, B_thz = fit_log_prime_equation(
        primes=primes,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm
    )

    rows = []

    for channel, prime in enumerate(primes, start=1):
        log2_prime = math.log2(prime)

        target_frequency_thz = A_thz * log2_prime + B_thz
        target_wavelength_nm = wavelength_nm_from_frequency_thz(target_frequency_thz)

        float_pixel = wavelength_to_pixel_float(
            wavelength_nm=target_wavelength_nm,
            anchor_wavelength_nm=wavelength_min_nm,
            anchor_pixel=0.0,
            focal_length_mm=focal_length_mm,
            config=config
        )

        integer_pixel = int(round(float_pixel))

        actual_wavelength_nm = pixel_to_wavelength_nm(
            pixel=integer_pixel,
            anchor_wavelength_nm=wavelength_min_nm,
            anchor_pixel=0.0,
            focal_length_mm=focal_length_mm,
            config=config
        )

        actual_frequency_thz = frequency_thz_from_wavelength_nm(actual_wavelength_nm)

        rows.append({
            "channel": channel,
            "prime": prime,
            "log2_prime": log2_prime,
            "target_frequency_thz": target_frequency_thz,
            "target_wavelength_nm": target_wavelength_nm,
            "float_pixel": float_pixel,
            "integer_pixel": integer_pixel,
            "snap_error_px": integer_pixel - float_pixel,
            "actual_pixel_wavelength_nm": actual_wavelength_nm,
            "actual_pixel_frequency_thz": actual_frequency_thz,
            "wavelength_error_nm": actual_wavelength_nm - target_wavelength_nm,
            "frequency_error_thz": actual_frequency_thz - target_frequency_thz,
            "pixel_left_boundary_nm": pixel_to_wavelength_nm(
                pixel=integer_pixel - 0.5,
                anchor_wavelength_nm=wavelength_min_nm,
                anchor_pixel=0.0,
                focal_length_mm=focal_length_mm,
                config=config
            ),
            "pixel_right_boundary_nm": pixel_to_wavelength_nm(
                pixel=integer_pixel + 0.5,
                anchor_wavelength_nm=wavelength_min_nm,
                anchor_pixel=0.0,
                focal_length_mm=focal_length_mm,
                config=config
            )
        })

    return rows, A_thz, B_thz


def select_small_prime_subset(
    channel_count: int,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    focal_length_mm: float,
    config: OpticalConfig,
    pool_size: int | None = None
) -> list[int]:
    """
    Select a small-prime subset that maps to unique LCD pixels.
    """
    if pool_size is None:
        pool_size = max(256, channel_count * 16)

    prime_pool = first_n_primes(pool_size)

    best_primes: list[int] | None = None
    best_score: tuple | None = None

    first_prime = prime_pool[0]

    for endpoint_index in range(channel_count - 1, len(prime_pool)):
        endpoint_prime = prime_pool[endpoint_index]

        endpoint_primes = [first_prime, endpoint_prime]

        A_thz, B_thz = fit_log_prime_equation(
            primes=endpoint_primes,
            wavelength_min_nm=wavelength_min_nm,
            wavelength_max_nm=wavelength_max_nm
        )

        pixel_to_prime: dict[int, tuple[int, float]] = {}

        for prime in prime_pool[:endpoint_index + 1]:
            log2_prime = math.log2(prime)
            target_frequency_thz = A_thz * log2_prime + B_thz
            target_wavelength_nm = wavelength_nm_from_frequency_thz(target_frequency_thz)

            try:
                float_pixel = wavelength_to_pixel_float(
                    wavelength_nm=target_wavelength_nm,
                    anchor_wavelength_nm=wavelength_min_nm,
                    anchor_pixel=0.0,
                    focal_length_mm=focal_length_mm,
                    config=config
                )
            except ValueError:
                continue

            integer_pixel = int(round(float_pixel))
            snap_error = abs(integer_pixel - float_pixel)

            if integer_pixel < 0 or integer_pixel >= config.lcd_pixels_x:
                continue

            if snap_error > config.max_snap_error_px:
                continue

            existing = pixel_to_prime.get(integer_pixel)
            if existing is None or prime < existing[0]:
                pixel_to_prime[integer_pixel] = (prime, snap_error)

        candidate_items = [
            {"pixel": pixel, "prime": prime, "snap_error": snap_error}
            for pixel, (prime, snap_error) in pixel_to_prime.items()
        ]

        if len(candidate_items) < channel_count:
            continue

        candidate_items.sort(key=lambda item: item["prime"])

        selected: list[dict] = []

        def pixel_ok(pixel: int) -> bool:
            return all(abs(pixel - item["pixel"]) >= config.min_integer_pixel_gap for item in selected)

        forced_primes = {first_prime, endpoint_prime}

        for prime in forced_primes:
            forced_matches = [item for item in candidate_items if item["prime"] == prime]
            if not forced_matches:
                break
            item = forced_matches[0]
            if pixel_ok(item["pixel"]):
                selected.append(item)

        if len(selected) != len(forced_primes):
            continue

        for item in candidate_items:
            if item["prime"] in forced_primes:
                continue
            if pixel_ok(item["pixel"]):
                selected.append(item)
            if len(selected) >= channel_count:
                break

        if len(selected) < channel_count:
            continue

        selected = selected[:channel_count]
        selected_primes = sorted(item["prime"] for item in selected)

        if first_prime not in selected_primes:
            continue
        if endpoint_prime not in selected_primes:
            continue

        product_log10 = prime_product_log10_from_primes(selected_primes)
        max_prime = max(selected_primes)
        pixel_span = max(item["pixel"] for item in selected) - min(item["pixel"] for item in selected)

        score = (product_log10, max_prime, -pixel_span)

        if best_score is None or score < best_score:
            best_score = score
            best_primes = selected_primes

    if best_primes is None:
        raise ValueError("Could not find a valid small-prime subset for this optical window.")

    return best_primes

def generate_integer_channel_map(
    channel_count: int,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    focal_length_mm: float,
    config: OpticalConfig
) -> dict:
    if channel_count > config.lcd_pixels_x:
        raise ValueError("Channel count cannot exceed horizontal LCD pixel count.")

    primes = select_small_prime_subset(
        channel_count=channel_count,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        focal_length_mm=focal_length_mm,
        config=config
    )

    rows, A_thz, B_thz = make_rows_for_primes(
        primes=primes,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        focal_length_mm=focal_length_mm,
        config=config
    )

    rows_by_pixel = sorted(rows, key=lambda r: r["integer_pixel"])

    integer_pixels = [row["integer_pixel"] for row in rows_by_pixel]
    unique_pixels = sorted(set(integer_pixels))

    duplicate_pixels = len(unique_pixels) != len(integer_pixels)

    integer_gaps = [
        unique_pixels[i + 1] - unique_pixels[i]
        for i in range(len(unique_pixels) - 1)
    ]

    min_integer_gap = min(integer_gaps) if integer_gaps else 999999
    max_snap_error = max(abs(row["snap_error_px"]) for row in rows)

    fits_lcd = (
        min(integer_pixels) >= 0
        and max(integer_pixels) <= config.lcd_pixels_x - 1
    )

    passes_integer_gap = min_integer_gap >= config.min_integer_pixel_gap
    passes_snap_error = max_snap_error <= config.max_snap_error_px

    valid = (
        fits_lcd
        and not duplicate_pixels
        and passes_integer_gap
        and passes_snap_error
    )

    return {
        "channel_count": channel_count,
        "wavelength_min_nm": wavelength_min_nm,
        "wavelength_max_nm": wavelength_max_nm,
        "focal_length_mm": focal_length_mm,

        "A_thz": A_thz,
        "B_thz": B_thz,

        "min_integer_pixel": min(integer_pixels),
        "max_integer_pixel": max(integer_pixels),
        "integer_pixel_span": max(integer_pixels) - min(integer_pixels),

        "min_integer_gap": min_integer_gap,
        "max_snap_error_px": max_snap_error,

        "duplicate_pixels": duplicate_pixels,
        "fits_lcd": fits_lcd,
        "passes_integer_gap": passes_integer_gap,
        "passes_snap_error": passes_snap_error,
        "valid": valid,

        "prime_product_log10_all_channels": prime_product_log10_from_primes(primes),
        "max_prime": max(primes),
        "min_prime": min(primes),

        "rows": rows_by_pixel
    }

def max_wavelength_that_fits_from_min(
    wavelength_min_nm: float,
    focal_length_mm: float,
    config: OpticalConfig
) -> float:
    return pixel_to_wavelength_nm(
        pixel=config.lcd_pixels_x - 1,
        anchor_wavelength_nm=wavelength_min_nm,
        anchor_pixel=0.0,
        focal_length_mm=focal_length_mm,
        config=config
    )


def find_candidates(config: OpticalConfig) -> list:
    candidates = []

    for focal_length_mm in config.lens_options_mm:
        for channel_count in range(config.min_channels, config.max_channels + 1):
            try:
                full = generate_integer_channel_map(
                    channel_count=channel_count,
                    wavelength_min_nm=config.led_min_nm,
                    wavelength_max_nm=config.led_max_nm,
                    focal_length_mm=focal_length_mm,
                    config=config
                )
                full["mode"] = "full_led_range"
                candidates.append(full)
            except ValueError:
                pass

        if config.allow_cropped_windows:
            start_nm = config.led_min_nm

            while start_nm < config.led_max_nm:
                end_nm = max_wavelength_that_fits_from_min(
                    wavelength_min_nm=start_nm,
                    focal_length_mm=focal_length_mm,
                    config=config
                )

                end_nm = min(end_nm, config.led_max_nm)

                if end_nm <= start_nm + 10:
                    start_nm += config.crop_step_nm
                    continue

                for channel_count in range(config.min_channels, config.max_channels + 1):
                    try:
                        cropped = generate_integer_channel_map(
                            channel_count=channel_count,
                            wavelength_min_nm=start_nm,
                            wavelength_max_nm=end_nm,
                            focal_length_mm=focal_length_mm,
                            config=config
                        )
                        cropped["mode"] = "cropped_window"
                        candidates.append(cropped)
                    except ValueError:
                        pass

                start_nm += config.crop_step_nm

    def score_valid(result: dict) -> tuple:
        return (
            result["channel_count"],
            result["min_integer_gap"],
            -result["max_snap_error_px"],
            -result.get("prime_product_log10_all_channels", 0.0),
            result["integer_pixel_span"]
        )

    def score_invalid(result: dict) -> tuple:
        return (
            result["channel_count"],
            -int(result["duplicate_pixels"]),
            int(result["fits_lcd"]),
            int(result["passes_snap_error"]),
            result["integer_pixel_span"]
        )

    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    invalid_candidates = [candidate for candidate in candidates if not candidate["valid"]]

    valid_candidates.sort(key=score_valid, reverse=True)
    invalid_candidates.sort(key=score_invalid, reverse=True)

    return valid_candidates + invalid_candidates


def print_result(result: dict) -> None:
    print()
    print("=" * 90)
    print(f"Mode:                  {result['mode']}")
    print(f"Channels:              {result['channel_count']}")
    print(f"Lens:                  {result['focal_length_mm']} mm")
    print(f"Wavelength window:     {result['wavelength_min_nm']:.3f} to {result['wavelength_max_nm']:.3f} nm")
    print(f"Equation:              frequency_THz = A * log2(P) + B")
    print(f"A:                     {result['A_thz']:.9f} THz")
    print(f"B:                     {result['B_thz']:.9f} THz")
    print(f"Integer pixel range:   {result['min_integer_pixel']} to {result['max_integer_pixel']}")
    print(f"Integer pixel span:    {result['integer_pixel_span']} px")
    print(f"Minimum integer gap:   {result['min_integer_gap']} px")
    print(f"Max snap error:        {result['max_snap_error_px']:.4f} px")
    print(f"Duplicate pixels:      {result['duplicate_pixels']}")
    print(f"Fits LCD:              {result['fits_lcd']}")
    print(f"Passes integer gaps:   {result['passes_integer_gap']}")
    print(f"Passes snap error:     {result['passes_snap_error']}")
    print(f"VALID:                 {result['valid']}")
    print("=" * 90)

    print()
    print("First 10 mapped integer columns:")
    for row in result["rows"][:10]:
        print(
            f"px={row['integer_pixel']:>3}  "
            f"ch={row['channel']:>2}  "
            f"P={row['prime']}  "
            f"targetlambda={row['target_wavelength_nm']:.3f}nm  "
            f"actuallambda={row['actual_pixel_wavelength_nm']:.3f}nm  "
            f"snap={row['snap_error_px']:+.3f}px  "
            f"gap-boundary={row['pixel_left_boundary_nm']:.3f}-{row['pixel_right_boundary_nm']:.3f}nm"
        )

    print()
    print("Last 10 mapped integer columns:")
    for row in result["rows"][-10:]:
        print(
            f"px={row['integer_pixel']:>3}  "
            f"ch={row['channel']:>2}  "
            f"P={row['prime']}  "
            f"targetlambda={row['target_wavelength_nm']:.3f}nm  "
            f"actuallambda={row['actual_pixel_wavelength_nm']:.3f}nm  "
            f"snap={row['snap_error_px']:+.3f}px  "
            f"gap-boundary={row['pixel_left_boundary_nm']:.3f}-{row['pixel_right_boundary_nm']:.3f}nm"
        )


def save_csv(result: dict, filename: str) -> None:
    fieldnames = [
        "channel",
        "prime",
        "log2_prime",
        "target_frequency_thz",
        "target_wavelength_nm",
        "float_pixel",
        "integer_pixel",
        "snap_error_px",
        "actual_pixel_wavelength_nm",
        "actual_pixel_frequency_thz",
        "wavelength_error_nm",
        "frequency_error_thz",
        "pixel_left_boundary_nm",
        "pixel_right_boundary_nm"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in result["rows"]:
            writer.writerow(row)

    print(f"\nSaved CSV: {filename}")

def save_cfg(result: dict, filename: str) -> None:
    """
    Save the generated integer log-prime wavelength map as a .cfg file.

    The file stores:
    - global optical setup
    - fitted log-prime equation constants
    - integer LCD pixel mapping
    - per-channel prime/wavelength/frequency data

    Output equation:
        frequency_THz = A_thz * log2(P) + B_thz
    """

    cfg = configparser.ConfigParser()

    cfg["setup"] = {
        "mode": str(result.get("mode", "unknown")),
        "channel_count": str(result["channel_count"]),
        "focal_length_mm": str(result["focal_length_mm"]),
        "wavelength_min_nm": str(result["wavelength_min_nm"]),
        "wavelength_max_nm": str(result["wavelength_max_nm"]),
        "min_integer_pixel": str(result["min_integer_pixel"]),
        "max_integer_pixel": str(result["max_integer_pixel"]),
        "integer_pixel_span": str(result["integer_pixel_span"]),
        "min_integer_gap": str(result["min_integer_gap"]),
        "max_snap_error_px": str(result["max_snap_error_px"]),
        "duplicate_pixels": str(result["duplicate_pixels"]),
        "fits_lcd": str(result["fits_lcd"]),
        "passes_integer_gap": str(result["passes_integer_gap"]),
        "passes_snap_error": str(result["passes_snap_error"]),
        "valid": str(result["valid"])
    }

    cfg["log_prime_equation"] = {
        "equation": "frequency_THz = A_thz * log2(P) + B_thz",
        "A_thz": str(result["A_thz"]),
        "B_thz": str(result["B_thz"])
    }

    for row in result["rows"]:
        section_name = f"channel_{row['channel']:03d}"

        cfg[section_name] = {
            "channel": str(row["channel"]),
            "prime": str(row["prime"]),
            "log2_prime": str(row["log2_prime"]),

            "target_frequency_thz": str(row["target_frequency_thz"]),
            "target_wavelength_nm": str(row["target_wavelength_nm"]),

            "float_pixel": str(row["float_pixel"]),
            "integer_pixel": str(row["integer_pixel"]),
            "snap_error_px": str(row["snap_error_px"]),

            "actual_pixel_wavelength_nm": str(row["actual_pixel_wavelength_nm"]),
            "actual_pixel_frequency_thz": str(row["actual_pixel_frequency_thz"]),

            "wavelength_error_nm": str(row["wavelength_error_nm"]),
            "frequency_error_thz": str(row["frequency_error_thz"]),

            "pixel_left_boundary_nm": str(row["pixel_left_boundary_nm"]),
            "pixel_right_boundary_nm": str(row["pixel_right_boundary_nm"])
        }

    with open(filename, "w", encoding="utf-8") as file:
        cfg.write(file)

    print(f"Saved CFG: {filename}")


def main():
    config = OpticalConfig(
        led_min_nm=380.0,
        led_max_nm=740.0,

        lcd_pixels_x=128,
        lcd_width_mm=21.5,

        grating_lines_per_mm=600.0,
        diffraction_order=1,

        lens_options_mm=(80.0, 100.0),

        # 12-lane scalar-readout core.
        # This keeps one optical core around 12 effective bits / 4096 states.
        min_channels=12,
        max_channels=12,

        # Use 1 for aggressive proof-of-concept density.
        # Use 2 if you want guard pixels for a more forgiving physical setup.
        min_integer_pixel_gap=1,
        max_snap_error_px=0.5,

        allow_cropped_windows=True,

        # Faster search. Use 1.0 only for final map polishing.
        crop_step_nm=5.0
    )

    candidates = find_candidates(config)

    if not candidates:
        raise RuntimeError("No candidates were generated at all.")

    best = candidates[0]

    if not best["valid"]:
        print()
        print("=" * 90)
        print("NO VALID 12-LANE MAP FOUND")
        print("=" * 90)
        print("The best candidate was still invalid, so the mapper will not save it.")
        print()
        print_result(best)

        print()
        print("Try:")
        print("  1. min_integer_pixel_gap=1")
        print("  2. crop_step_nm=1.0 for a finer search")
        print("  3. a different lens option")
        print("  4. a smaller wavelength window")
        print("=" * 90)

        raise RuntimeError("No valid 12-lane wavelength map found.")

    print_result(best)

    print()
    print(f"Prime product log10:   {best.get('prime_product_log10_all_channels', 0.0):.3f}")

    save_csv(best, "integer_log_prime_wavelength_map.csv")
    save_cfg(best, "integer_log_prime_wavelength_map.cfg")

    print()
    print("Top candidates:")
    for i, candidate in enumerate(candidates[:10], start=1):
        print(
            f"{i:>2}. "
            f"valid={candidate['valid']}  "
            f"mode={candidate['mode']:<16}  "
            f"channels={candidate['channel_count']:<2}  "
            f"lens={candidate['focal_length_mm']:<5.0f}mm  "
            f"range={candidate['wavelength_min_nm']:.1f}-{candidate['wavelength_max_nm']:.1f}nm  "
            f"px={candidate['min_integer_pixel']}-{candidate['max_integer_pixel']}  "
            f"gap={candidate['min_integer_gap']}  "
            f"snap={candidate['max_snap_error_px']:.3f}px  "
            f"log10P={candidate.get('prime_product_log10_all_channels', 0.0):.2f}"
        )
if __name__ == "__main__":
    main()