"""
Fourier Tiler — Command-line version
Usage:  python FourierTiler.py Input.txt
   or:  python FourierTiler.py ( it looks for Input.txt in the working directory)

All parameters are read from the input file. Run with --template to write a
commented template input file and exit.
"""

import sys
import os
import random
import io
import datetime

import numpy as np
import numpy.fft as fft
from PIL import Image, ImageOps, PngImagePlugin
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Input-file parser
# ---------------------------------------------------------------------------

DEFAULTS = {
    "n":                    42,
    "mc_cycles":            1,
    "crop_shape":           "faded",
    "crop_radius_factor":   1.0,
    "output_folder":        ".",
    "fft_zoom_factor":      0.3,
    "gaussian_sigma":       1.0,
    "colormap":             "afmhot",
    "intensity_low":        0.0,
    "intensity_high":       100.0,
    "export_size":          3000,
    "save_mc_energy_plot":  True,
    "save_diff_ff":         True,
    "save_avg_structure":   False,
}

VALID_CROP_SHAPES = {"none", "circle", "faded", "ellipse", "square", "pentagonal"}


def parse_input_file(path: str) -> dict:
    """
    Parse a key = value input file.

    Special multi-value keys:
        tile        = <occupancy>, <image_path>
        interaction = <from_index>, <to_index>, <dx>, <dy>, <energy>

    Tile indices in interactions are 1-based (matching tile order in the file).
    Lines starting with # are comments; blank lines are ignored.
    """
    params = dict(DEFAULTS)
    params["tiles"]        = []   # list of (occupancy, path)
    params["interactions"] = []   # list of (i, j, dx, dy, energy)  — 0-based indices

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                _warn(f"Line {lineno}: no '=' found — skipped: {raw.rstrip()}")
                continue

            key, _, rest = line.partition("=")
            key  = key.strip().lower()
            rest = rest.strip()

            if key == "tile":
                parts = [p.strip() for p in rest.split(",", 1)]
                if len(parts) != 2:
                    _warn(f"Line {lineno}: 'tile' needs 'occupancy, path' — skipped.")
                    continue
                try:
                    occ = float(parts[0])
                except ValueError:
                    _warn(f"Line {lineno}: occupancy '{parts[0]}' is not a number — skipped.")
                    continue
                params["tiles"].append((occ, parts[1]))

            elif key == "interaction":
                parts = [p.strip() for p in rest.split(",")]
                if len(parts) != 5:
                    _warn(f"Line {lineno}: 'interaction' needs 5 comma-separated values — skipped.")
                    continue
                try:
                    i_from = int(parts[0]) - 1   # convert to 0-based
                    j_to   = int(parts[1]) - 1
                    dx     = int(float(parts[2]))
                    dy     = int(float(parts[3]))
                    energy = float(parts[4])
                except ValueError as exc:
                    _warn(f"Line {lineno}: bad interaction values ({exc}) — skipped.")
                    continue
                params["interactions"].append((i_from, j_to, dx, dy, energy))

            elif key == "n":
                params["n"] = _int(key, rest, lineno, DEFAULTS["n"])

            elif key == "mc_cycles":
                params["mc_cycles"] = _int(key, rest, lineno, DEFAULTS["mc_cycles"])

            elif key == "export_size":
                params["export_size"] = _int(key, rest, lineno, DEFAULTS["export_size"])

            elif key in ("crop_radius_factor", "fft_zoom_factor",
                         "gaussian_sigma", "intensity_low", "intensity_high"):
                params[key] = _float(key, rest, lineno, DEFAULTS[key])

            elif key == "crop_shape":
                val = rest.lower()
                if val not in VALID_CROP_SHAPES:
                    _warn(f"Line {lineno}: unknown crop_shape '{rest}'. "
                          f"Valid options: {sorted(VALID_CROP_SHAPES)}. Using default.")
                else:
                    params["crop_shape"] = val

            elif key == "colormap":
                if rest not in matplotlib.colormaps:
                    _warn(f"Line {lineno}: colormap '{rest}' not recognised. "
                          "Using default 'afmhot'.")
                else:
                    params["colormap"] = rest

            elif key == "output_folder":
                params["output_folder"] = rest

            elif key in ("save_mc_energy_plot", "save_diff_ff", "save_avg_structure"):
                params[key] = _bool(key, rest, lineno, DEFAULTS[key])

            else:
                _warn(f"Line {lineno}: unknown key '{key}' — skipped.")

    return params


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _warn(msg):
    print(f"  [WARNING] {msg}", file=sys.stderr)


def _int(key, val, lineno, default):
    try:
        return int(val)
    except ValueError:
        _warn(f"Line {lineno}: '{key}' value '{val}' is not an integer. Using {default}.")
        return default


def _float(key, val, lineno, default):
    try:
        return float(val)
    except ValueError:
        _warn(f"Line {lineno}: '{key}' value '{val}' is not a number. Using {default}.")
        return default


def _bool(key, val, lineno, default):
    if val.lower() in ("true", "yes", "1"):
        return True
    if val.lower() in ("false", "no", "0"):
        return False
    _warn(f"Line {lineno}: '{key}' value '{val}' is not True/False. Using {default}.")
    return default


# ---------------------------------------------------------------------------
# Image-processing helpers  (unchanged logic from the GUI version)
# ---------------------------------------------------------------------------

def crop_to_square_center(img: Image.Image) -> Image.Image:
    """Crop a rectangular image to a centred square."""
    w, h = img.size
    if w == h:
        return img
    side   = min(w, h)
    left   = (w - side) // 2
    top    = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def pad_to_minimum(arr: np.ndarray, target: int = 5000) -> np.ndarray:
    """Centre-pad a (square) array to at least target×target pixels."""
    h, w = arr.shape[:2]
    if h >= target and w >= target:
        return arr
    new_h, new_w = max(h, target), max(w, target)
    pt = (new_h - h) // 2;  pb = new_h - h - pt
    pl = (new_w - w) // 2;  pr = new_w - w - pl
    kw = dict(mode="constant", constant_values=0)
    if arr.ndim == 2:
        return np.pad(arr, ((pt, pb), (pl, pr)), **kw)
    return np.pad(arr, ((pt, pb), (pl, pr), (0, 0)), **kw)


def apply_mask_crop(image: Image.Image, crop_radius_factor: float, shape: str) -> np.ndarray:
    """Return a numpy array with the requested mask/crop applied."""
    arr         = np.array(image.convert("L"))
    h, w        = arr.shape
    cx, cy      = w // 2, h // 2
    max_r       = min(w, h) // 2
    radius      = int(max_r * crop_radius_factor)
    Y, X        = np.ogrid[:h, :w]

    if shape == "none":
        return arr

    if shape == "circle":
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
        return np.where(mask, arr, 0)

    if shape == "faded":
        mask        = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
        radial_dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        r_start     = radius * 0.8
        sigma       = (radius - r_start) / 2
        fade        = np.ones_like(arr, dtype=float)
        zone        = (radial_dist > r_start) & (radial_dist <= radius)
        fade[zone]  = np.exp(-((radial_dist[zone] - r_start) ** 2) / (2 * sigma ** 2))
        return (arr * fade * mask).astype(np.uint8)

    if shape == "ellipse":
        rx   = min(radius, w // 2 - 1)
        ry   = min(int(radius * 0.75), h // 2 - 1)
        mask = ((X - cx) ** 2 / rx ** 2) + ((Y - cy) ** 2 / ry ** 2) <= 1

    elif shape == "square":
        mask = (np.abs(X - cx) + np.abs(Y - cy)) <= radius

    elif shape == "pentagonal":
        angles  = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        polygon = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
        xx, yy  = np.meshgrid(np.arange(w), np.arange(h))
        coords  = np.column_stack((xx.ravel(), yy.ravel()))
        mask    = Path(polygon).contains_points(coords).reshape((h, w))

    else:
        mask = np.ones_like(arr, dtype=bool)

    return np.where(mask, arr, 0)


def apply_fft_processing(magnitude: np.ndarray, params: dict) -> np.ndarray:
    """Apply smoothing, zoom, percentile clip, and intensity-range rescaling."""
    sigma = float(params["gaussian_sigma"])
    if sigma > 0:
        magnitude = gaussian_filter(magnitude, sigma=sigma)

    zoom   = max(0.01, min(float(params["fft_zoom_factor"]), 1.0))
    h, w   = magnitude.shape
    zh, zw = int(h * zoom), int(w * zoom)
    sy, sx = (h - zh) // 2, (w - zw) // 2
    magnitude = magnitude[sy:sy + zh, sx:sx + zw]

    magnitude = np.clip(magnitude, 0, np.percentile(magnitude, 99))
    if magnitude.max() != 0:
        magnitude /= magnitude.max()

    lo = float(params["intensity_low"])  / 100.0
    hi = float(params["intensity_high"]) / 100.0
    magnitude = np.clip(magnitude, lo, hi)
    magnitude = (magnitude - lo) / (hi - lo + 1e-8)
    return magnitude


# ---------------------------------------------------------------------------
# File-naming helper
# ---------------------------------------------------------------------------

def next_available_index(folder: str, prefix: str) -> int:
    existing = []
    for fname in os.listdir(folder):
        if fname.startswith(prefix + "_"):
            try:
                existing.append(int(fname.split("_")[1].split(".")[0]))
            except Exception:
                pass
    if not existing:
        return 1
    for expected, num in enumerate(sorted(existing), start=1):
        if num != expected:
            return expected
    return len(existing) + 1


# ---------------------------------------------------------------------------
# Energy calculation
# ---------------------------------------------------------------------------

def calculate_energy(g: np.ndarray, name_to_idx: dict, interactions: list) -> float:
    energy   = 0.0
    idx_grid = np.vectorize(lambda nm: name_to_idx[nm])(g)
    rows, cols = g.shape
    for (i_from, j_to, dx, dy, e_val) in interactions:
        dx_np = -dy
        dy_np =  dx
        i0 = max(0, -dx_np);  i1 = min(rows - 1, rows - 1 - dx_np)
        j0 = max(0, -dy_np);  j1 = min(cols - 1, cols - 1 - dy_np)
        if i0 > i1 or j0 > j1:
            continue
        src = idx_grid[i0:i1 + 1, j0:j1 + 1]
        nb  = idx_grid[i0 + dx_np:i1 + dx_np + 1, j0 + dy_np:j1 + dy_np + 1]
        energy += e_val * np.count_nonzero((src == i_from) & (nb == j_to))
    return energy


# ---------------------------------------------------------------------------
# Optional-output helpers
# ---------------------------------------------------------------------------

def compute_diff_ff(paths: list, min_size: int, params: dict) -> np.ndarray:
    """
    Compute |FT(tile1) − FT(tile2)|² using the first two valid tile paths,
    then apply the same FFT processing (smoothing, zoom, intensity range)
    as the main pattern.  Returns a processed magnitude array (0–1).
    """
    TARGET = 5000
    imgs = []
    for p in paths[:2]:
        img = crop_to_square_center(Image.open(p).convert("L"))
        if img.size[0] != min_size:
            img = img.resize((min_size, min_size), Image.LANCZOS)
        arr   = np.array(img, dtype=float)
        pad_y = (TARGET - arr.shape[0]) // 2
        pad_x = (TARGET - arr.shape[1]) // 2
        padded = np.pad(
            arr,
            ((pad_y, TARGET - arr.shape[0] - pad_y),
             (pad_x, TARGET - arr.shape[1] - pad_x)),
            mode="constant", constant_values=0,
        )
        imgs.append(padded)

    F1   = fft.fftshift(fft.fft2(imgs[0]))
    F2   = fft.fftshift(fft.fft2(imgs[1]))
    diff = np.abs(F2 - F1) ** 2
    return apply_fft_processing(diff, params)


def compute_avg_structure(paths: list, occs: list, min_size: int) -> Image.Image:
    """
    Return a greyscale PIL Image of the occupancy-weighted average tile.
    """
    arrays = []
    for p in paths:
        img = crop_to_square_center(Image.open(p).convert("L"))
        if img.size[0] != min_size:
            img = img.resize((min_size, min_size), Image.LANCZOS)
        arrays.append(np.array(img, dtype=float))

    avg  = sum(arr * occ for arr, occ in zip(arrays, occs))
    pmin, pmax = avg.min(), avg.max()
    if pmax > pmin:
        avg = (avg - pmin) / (pmax - pmin)
    return Image.fromarray((avg * 255).astype(np.uint8), mode="L")




def run(params: dict):
    # ── Validate / clamp ────────────────────────────────────────────────────
    n      = max(1, min(params["n"], 1000))
    cycles = max(0, params["mc_cycles"])
    tiles_raw  = params["tiles"]
    interactions_raw = params["interactions"]

    if not tiles_raw:
        sys.exit("ERROR: No tiles defined. Add at least one 'tile = occupancy, path' line.")

    # Normalise occupancies
    occs  = [t[0] for t in tiles_raw]
    paths = [t[1] for t in tiles_raw]
    total = sum(occs)
    if total <= 0:
        sys.exit("ERROR: All tile occupancies are zero.")
    occs = [o / total for o in occs]

    # Check files exist
    for p in paths:
        if not os.path.isfile(p):
            sys.exit(f"ERROR: Tile image not found: {p}")

    # Check for duplicate tiles
    print("Loading tiles...")
    loaded_arrays = []
    for p in paths:
        img = crop_to_square_center(Image.open(p).convert("RGB"))
        loaded_arrays.append((p, np.array(img)))

    for i in range(len(loaded_arrays)):
        for j in range(i + 1, len(loaded_arrays)):
            if np.array_equal(loaded_arrays[i][1], loaded_arrays[j][1]):
                sys.exit(
                    f"ERROR: Tiles appear identical:\n"
                    f"  {loaded_arrays[i][0]}\n  {loaded_arrays[j][0]}\n"
                    "Please remove one of them."
                )

    # Find smallest square tile size; resize all to that
    sizes     = [la[1].shape[0] for la in loaded_arrays]
    min_size  = min(sizes)
    tile_names = []
    tiles      = {}
    used_occs  = []

    for occ, (p, _) in zip(occs, loaded_arrays):
        img  = crop_to_square_center(Image.open(p).convert("RGB"))
        if img.size[0] != min_size:
            img = img.resize((min_size, min_size), Image.LANCZOS)
        name = os.path.splitext(os.path.basename(p))[0]
        tile_names.append(name)
        tiles[name] = img
        used_occs.append(occ)

    # Validate interaction tile indices
    n_tiles = len(tile_names)
    interactions = []
    for (i_from, j_to, dx, dy, energy) in interactions_raw:
        if not (0 <= i_from < n_tiles and 0 <= j_to < n_tiles):
            _warn(f"Interaction ({i_from+1}, {j_to+1}, {dx}, {dy}, {energy}): "
                  f"tile index out of range (1..{n_tiles}) — skipped.")
            continue
        interactions.append((i_from, j_to, dx, dy, energy))

    # ── Build initial grid ──────────────────────────────────────────────────
    print(f"Building {n}×{n} grid...")
    tile_counts = [int(n * n * occ) for occ in used_occs]
    while sum(tile_counts) < n * n:
        tile_counts[-1] += 1

    grid_flat = []
    for name, count in zip(tile_names, tile_counts):
        grid_flat.extend([name] * count)
    random.shuffle(grid_flat)
    grid = np.array(grid_flat).reshape((n, n))

    # ── MC simulation ───────────────────────────────────────────────────────
    name_to_idx  = {name: idx for idx, name in enumerate(tile_names)}
    mc_energies  = []
    out_dir      = params["output_folder"]
    os.makedirs(out_dir, exist_ok=True)
    log_path     = os.path.join(out_dir, "MC_global_energy_log.txt")

    print(f"Running MC simulation ({cycles} cycles)...")
    with open(log_path, "w") as log_file:
        log_file.write("cycle\told_energy\tnew_energy\tcurrent_energy\n")

        for cycle in range(cycles):
            i1, j1 = random.randint(0, n - 1), random.randint(0, n - 1)
            i2, j2 = random.randint(0, n - 1), random.randint(0, n - 1)

            old_energy = calculate_energy(grid, name_to_idx, interactions)
            grid[i1, j1], grid[i2, j2] = grid[i2, j2], grid[i1, j1]
            new_energy = calculate_energy(grid, name_to_idx, interactions)

            if new_energy < old_energy:
                current_energy = new_energy          # keep the swap
            else:
                grid[i1, j1], grid[i2, j2] = grid[i2, j2], grid[i1, j1]   # revert
                current_energy = old_energy

            log_file.write(
                f"{cycle + 1}\t\t{old_energy}\t\t{new_energy}\t\t{current_energy}\n"
            )
            mc_energies.append(current_energy)

            if cycles >= 100 and cycle % max(1, cycles // 100) == 0:
                pct = int((cycle / cycles) * 100)
                print(f"  {pct}%", end="\r", flush=True)

    if cycles >= 100:
        print("  100%")

    # ── Compose tiling image ────────────────────────────────────────────────
    print("Composing tiling image...")
    total_pixels = (n * min_size) ** 2
    MAX_PIXELS   = 100_000_000

    final_img = Image.new("RGB", (n * min_size, n * min_size))
    for i in range(n):
        for j in range(n):
            final_img.paste(tiles[grid[i, j]], (j * min_size, i * min_size))

    if total_pixels > MAX_PIXELS:
        print(f"  Image has {total_pixels:,} pixels — resizing to 100 Mpx automatically.")
        final_img = final_img.resize((10_000, 10_000), Image.LANCZOS)

    TARGET_PIXELS = 5000
    final_img_resized = final_img.resize((TARGET_PIXELS, TARGET_PIXELS), Image.LANCZOS)

    # ── FFT ─────────────────────────────────────────────────────────────────
    print("Computing FFT...")
    crop_shape  = params["crop_shape"]
    crop_radius = float(params["crop_radius_factor"])
    cropped     = apply_mask_crop(final_img, crop_radius, crop_shape)
    cropped_arr = pad_to_minimum(
        np.array(Image.fromarray(cropped), dtype=np.uint8), target=5000
    )

    fft_shifted      = fft.fftshift(fft.fft2(cropped_arr))
    fft_magnitude_raw = np.abs(fft_shifted) ** 2

    magnitude = apply_fft_processing(fft_magnitude_raw.copy(), params)

    # ── Save outputs ─────────────────────────────────────────────────────────
    export_size = int(params["export_size"])
    TARGET_DPI  = 600
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signature   = (
        f"Fourier Tiler | Saved on {timestamp} | "
        "Designed and developed by Stefano Canossa, supported by EU4MOFs COST action."
    )

    meta = PngImagePlugin.PngInfo()
    meta.add_text("FourierTilerSignature", signature)

    # Tiling
    idx_img   = next_available_index(out_dir, "Image")
    img_path  = os.path.join(out_dir, f"Image_{idx_img:02d}.png")
    tiling_export = final_img_resized.resize((export_size, export_size), Image.LANCZOS)
    tiling_export.save(img_path, dpi=(TARGET_DPI, TARGET_DPI), pnginfo=meta)
    print(f"  Saved tiling        →  {img_path}")

    # FFT intensity
    cmap_name = params["colormap"]
    cmap      = plt.get_cmap(cmap_name)
    rgba      = cmap(magnitude)
    rgb       = (rgba[:, :, :3] * 255).astype(np.uint8)
    fft_img   = Image.fromarray(rgb, mode="RGB").resize(
        (export_size, export_size), Image.LANCZOS
    )
    idx_int  = next_available_index(out_dir, "Intensity")
    fft_path = os.path.join(out_dir, f"Intensity_{idx_int:02d}.png")
    fft_img.save(fft_path, dpi=(TARGET_DPI, TARGET_DPI), pnginfo=meta)
    print(f"  Saved FFT           →  {fft_path}")

    # ── Difference form factor |f1 − f2|² ────────────────────────────────────
    if params["save_diff_ff"]:
        if len(paths) < 2:
            _warn("save_diff_ff = True but fewer than 2 tiles are loaded — skipped.")
        else:
            print("Computing difference form factor...")
            diff_mag  = compute_diff_ff(paths, min_size, params)
            diff_rgba = cmap(diff_mag)
            diff_rgb  = (diff_rgba[:, :, :3] * 255).astype(np.uint8)
            diff_img  = Image.fromarray(diff_rgb, mode="RGB").resize(
                (export_size, export_size), Image.LANCZOS
            )
            idx_diff  = next_available_index(out_dir, "DiffFF")
            diff_path = os.path.join(out_dir, f"DiffFF_{idx_diff:02d}.png")
            diff_img.save(diff_path, dpi=(TARGET_DPI, TARGET_DPI), pnginfo=meta)
            print(f"  Saved |f1−f2|²      →  {diff_path}")

    # ── Average structure ─────────────────────────────────────────────────────
    if params["save_avg_structure"]:
        print("Computing average structure...")
        avg_img  = compute_avg_structure(paths, used_occs, min_size)
        avg_img  = avg_img.resize((export_size, export_size), Image.LANCZOS)
        idx_avg  = next_available_index(out_dir, "AvgStructure")
        avg_path = os.path.join(out_dir, f"AvgStructure_{idx_avg:02d}.png")
        avg_img.save(avg_path, dpi=(TARGET_DPI, TARGET_DPI), pnginfo=meta)
        print(f"  Saved avg structure →  {avg_path}")

    # ── MC energy plot ────────────────────────────────────────────────────────
    has_nonzero = any(e != 0 for (_, _, _, _, e) in interactions)
    if params["save_mc_energy_plot"]:
        if not (has_nonzero and len(mc_energies) >= 2):
            _warn("save_mc_energy_plot = True but no nonzero interactions or fewer than "
                  "2 MC cycles ran — energy plot skipped.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.plot(range(1, len(mc_energies) + 1), mc_energies, color="#2F24D8", linewidth=2)
            ax.set_xlabel("\nMC cycle number\n")
            ax.set_ylabel("\nGlobal energy\n")
            ax.set_title("Global energy plot", fontsize=14, pad=12)
            ax.text(
                0.5, -0.18,
                "\nTo make full use of interactions, ensure cycles are sufficient\n"
                "for the energy to reach a plateau (equilibration).",
                fontsize=8, ha="center", va="top", transform=ax.transAxes,
            )
            ax.grid(True, alpha=0.3)
            idx_en    = next_available_index(out_dir, "MC_energy")
            plot_path = os.path.join(out_dir, f"MC_energy_{idx_en:02d}.png")
            fig.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  Saved MC energy     →  {plot_path}")

    print(f"\nAll done. Output folder: {os.path.abspath(out_dir)}")


# ---------------------------------------------------------------------------
# Template writer
# ---------------------------------------------------------------------------

TEMPLATE = """\
# ============================================================
#  Fourier Tiler — Input file
#  Usage: python FourierTiler.py Input.txt
#
#  Lines beginning with # are comments.
#  Key names are case-insensitive.
# ============================================================


# ── Grid ─────────────────────────────────────────────────────
# Side length of the square tiling in number of tiles.
# A value of 42 produces a 42×42 mosaic (1764 tiles total).
n = 42

# Number of Monte Carlo swap cycles.
# Use 0 for a purely random arrangement.
mc_cycles = 1000

# Shape used to crop the tiling before the FFT is computed.
# Options: none | circle | faded | ellipse | square | pentagonal
# "faded" (default) applies a Gaussian-tapered circular window
# that reduces edge artefacts in the diffraction pattern.
crop_shape = faded

# Fraction of the half-width used as the crop radius (0 < value ≤ 1).
crop_radius_factor = 1.0


# ── Output ───────────────────────────────────────────────────
# Folder where all output files are written (created if absent).
output_folder = ./output


# ── FFT / visualisation ──────────────────────────────────────
# Fraction of the FFT to display / export (centred zoom).
# 1.0 = full pattern; 0.3 = central 30 %.
fft_zoom_factor = 0.3

# Standard deviation of the Gaussian smoothing applied to the
# intensity pattern. Use 0 for no smoothing.
gaussian_sigma = 1.0

# Matplotlib colormap name for the intensity pattern.
# Examples: afmhot, inferno, viridis, magma, hot, gray, jet ...
colormap = afmhot

# Intensity-range clipping (percentages, 0 – 100).
# Equivalent to the two slider handles in the GUI.
intensity_low  = 0
intensity_high = 100

# Side length (pixels) of the exported PNG files.
# Allowed values: 1000 | 2000 | 3000 | 4000 | 5000
export_size = 3000

# ── Optional outputs ──────────────────────────────────────────
# Save the MC global-energy plot (PNG).
# Requires nonzero interaction energies and at least 2 MC cycles.
save_mc_energy_plot = True

# Save the squared difference form factor |FT(tile1) − FT(tile2)|²  (PNG).
# Uses the first two tiles listed below. Requires at least 2 tiles.
save_diff_ff = True

# Save the occupancy-weighted average structure (PNG).
save_avg_structure = False


# ── Tiles ────────────────────────────────────────────────────
# Format:  tile = <occupancy>, <path/to/image>
# Occupancies are normalised automatically if they do not sum to 1.
# Any non-square image is centre-cropped to a square.
# Images of different sizes are resized to the smallest one.
# Add or remove lines freely.
tile = 0.5, path/to/tile1.png
tile = 0.5, path/to/tile2.png


# ── Pair interactions ────────────────────────────────────────
# Format:  interaction = <from_tile>, <to_tile>, <dx>, <dy>, <energy>
#
#  from_tile / to_tile : 1-based tile index (order as listed above).
#  dx, dy              : displacement vector components (integers).
#  energy              : real number; negative = favoured, positive = disfavoured.
#
# No symmetry is applied automatically — add each direction explicitly.
# Comment out or delete these lines if no interactions are needed.
#
# Example — tile 1 dislikes tile 2 along x and y, likes it on diagonals:
# interaction = 1, 2,  1,  0,  10.0
# interaction = 1, 2,  0,  1,  10.0
# interaction = 1, 2, -1,  0,  10.0
# interaction = 1, 2,  0, -1,  10.0
# interaction = 1, 2,  1,  1,  -2.0
# interaction = 1, 2,  1, -1,  -2.0
# interaction = 1, 2, -1,  1,  -2.0
# interaction = 1, 2, -1, -1,  -2.0
"""


def write_template(dest: str = "Input.txt"):
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(TEMPLATE)
    print(f"Template input file written to: {os.path.abspath(dest)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Handle --template flag
    if "--template" in sys.argv:
        dest = "Input.txt"
        for arg in sys.argv[1:]:
            if not arg.startswith("-"):
                dest = arg
                break
        write_template(dest)
        return

    # Resolve input file
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        input_path = sys.argv[1]
    else:
        input_path = "Input.txt"

    if not os.path.isfile(input_path):
        print(
            f"ERROR: Input file not found: '{input_path}'\n\n"
            "Usage:\n"
            "  python FourierTiler.py Input.txt\n"
            "  python FourierTiler.py --template   (writes a template Input.txt)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading parameters from: {input_path}")
    params = parse_input_file(input_path)

    # Echo key settings
    print(
        f"\n  Grid              : {params['n']}×{params['n']}\n"
        f"  MC cycles         : {params['mc_cycles']}\n"
        f"  Crop shape        : {params['crop_shape']}\n"
        f"  Colormap          : {params['colormap']}\n"
        f"  Export size       : {params['export_size']}×{params['export_size']} px\n"
        f"  Output folder     : {params['output_folder']}\n"
        f"  Tiles             : {len(params['tiles'])}\n"
        f"  Interactions      : {len(params['interactions'])}\n"
        f"  Save MC energy    : {params['save_mc_energy_plot']}\n"
        f"  Save |f1−f2|²     : {params['save_diff_ff']}\n"
        f"  Save avg structure: {params['save_avg_structure']}\n"
    )

    run(params)


if __name__ == "__main__":
    main()
