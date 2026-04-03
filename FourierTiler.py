"""
Fourier Tiler — Command-line version
Usage:  python FourierTiler.py Input.txt
        python FourierTiler.py               (looks for Input.txt in the working directory)

All parameters are read from the input file. Run with --template to write a
commented template input file and exit.
"""

import sys
import os
import random
import datetime
import math

import numpy as np
import numpy.fft as fft
from PIL import Image, ImageDraw, ImageOps, PngImagePlugin


# ---------------------------------------------------------------------------
# Colormap LUT engine  (replaces matplotlib.colormaps + plt.get_cmap)
# ---------------------------------------------------------------------------

def _c(v): return max(0.0, min(1.0, float(v)))
def _lerp_ctrl(ctrl, t):
    if t <= ctrl[0][0]:  return ctrl[0][1], ctrl[0][2], ctrl[0][3]
    if t >= ctrl[-1][0]: return ctrl[-1][1], ctrl[-1][2], ctrl[-1][3]
    for j in range(len(ctrl)-1):
        t0,r0,g0,b0 = ctrl[j]; t1,r1,g1,b1 = ctrl[j+1]
        if t0 <= t <= t1:
            u = (t-t0)/(t1-t0) if t1>t0 else 0.0
            return r0+u*(r1-r0), g0+u*(g1-g0), b0+u*(b1-b0)
    return ctrl[-1][1], ctrl[-1][2], ctrl[-1][3]
def _make_lut(fn, n=256):
    out = np.empty((n,3), dtype=np.uint8)
    for i in range(n):
        t=i/255.0; r,g,b=fn(t)
        out[i]=(int(_c(r)*255), int(_c(g)*255), int(_c(b)*255))
    return out

_VI=[(0.0,0.267,0.005,0.329),(0.1,0.283,0.141,0.458),(0.2,0.254,0.265,0.530),(0.3,0.207,0.372,0.553),(0.4,0.164,0.471,0.558),(0.5,0.128,0.566,0.551),(0.6,0.135,0.659,0.518),(0.7,0.267,0.749,0.441),(0.8,0.478,0.821,0.318),(0.9,0.741,0.873,0.150),(1.0,0.993,0.906,0.144)]
_IN=[(0.0,0.001,0.000,0.014),(0.1,0.116,0.052,0.240),(0.2,0.307,0.062,0.425),(0.3,0.494,0.090,0.418),(0.4,0.652,0.167,0.342),(0.5,0.798,0.280,0.225),(0.6,0.910,0.418,0.104),(0.7,0.974,0.572,0.024),(0.8,0.992,0.733,0.137),(0.9,0.978,0.901,0.349),(1.0,0.988,0.998,0.645)]
_MA=[(0.0,0.001,0.000,0.014),(0.1,0.107,0.049,0.202),(0.2,0.269,0.067,0.401),(0.3,0.432,0.099,0.468),(0.4,0.580,0.141,0.480),(0.5,0.728,0.195,0.473),(0.6,0.864,0.285,0.420),(0.7,0.956,0.448,0.384),(0.8,0.992,0.634,0.490),(0.9,0.997,0.825,0.682),(1.0,0.988,0.992,0.880)]
_PL=[(0.0,0.051,0.030,0.529),(0.1,0.286,0.017,0.603),(0.2,0.458,0.006,0.640),(0.3,0.612,0.091,0.624),(0.4,0.736,0.209,0.563),(0.5,0.845,0.323,0.470),(0.6,0.932,0.441,0.365),(0.7,0.976,0.576,0.248),(0.8,0.987,0.727,0.133),(0.9,0.969,0.882,0.104),(1.0,0.940,1.000,0.150)]
_TU=[(0.0,0.190,0.071,0.230),(0.1,0.274,0.365,0.886),(0.2,0.165,0.640,0.992),(0.3,0.071,0.845,0.768),(0.4,0.176,0.950,0.431),(0.5,0.490,0.991,0.137),(0.6,0.784,0.929,0.075),(0.7,0.970,0.750,0.090),(0.8,0.991,0.490,0.063),(0.9,0.916,0.224,0.047),(1.0,0.735,0.041,0.039)]
_SP=[(0.0,0.620,0.004,0.259),(0.1,0.835,0.243,0.310),(0.2,0.957,0.427,0.263),(0.3,0.992,0.682,0.380),(0.4,0.996,0.878,0.545),(0.5,1.000,1.000,0.749),(0.6,0.902,0.961,0.596),(0.7,0.671,0.867,0.643),(0.8,0.400,0.761,0.647),(0.9,0.196,0.533,0.741),(1.0,0.369,0.310,0.635)]
_RB=[(0.0,0.647,0.000,0.149),(0.17,0.843,0.188,0.153),(0.33,0.957,0.647,0.510),(0.50,0.969,0.969,0.969),(0.67,0.573,0.773,0.871),(0.83,0.216,0.525,0.753),(1.0,0.020,0.188,0.380)]
_PY=[(0.0,0.557,0.004,0.322),(0.2,0.867,0.400,0.690),(0.4,0.980,0.749,0.906),(0.5,0.969,0.969,0.969),(0.6,0.851,0.941,0.827),(0.8,0.498,0.737,0.435),(1.0,0.153,0.392,0.098)]
_PR=[(0.0,0.251,0.000,0.294),(0.2,0.541,0.298,0.647),(0.4,0.847,0.761,0.871),(0.5,0.969,0.969,0.969),(0.6,0.718,0.886,0.698),(0.8,0.220,0.671,0.380),(1.0,0.000,0.267,0.106)]
_BB=[(0.0,0.329,0.188,0.020),(0.2,0.698,0.510,0.231),(0.4,0.902,0.812,0.616),(0.5,0.961,0.961,0.961),(0.6,0.718,0.886,0.878),(0.8,0.200,0.600,0.561),(1.0,0.004,0.235,0.188)]
_PO=[(0.0,0.498,0.231,0.031),(0.2,0.847,0.600,0.173),(0.4,0.992,0.859,0.635),(0.5,0.969,0.969,0.969),(0.6,0.847,0.741,0.906),(0.8,0.596,0.439,0.792),(1.0,0.176,0.000,0.294)]
_RG=[(0.0,0.404,0.000,0.122),(0.2,0.816,0.239,0.306),(0.4,0.980,0.757,0.651),(0.5,1.000,1.000,1.000),(0.6,0.878,0.878,0.878),(0.8,0.529,0.529,0.529),(1.0,0.251,0.251,0.251)]
_RYB=[(0.0,0.647,0.000,0.149),(0.2,0.910,0.306,0.145),(0.4,0.988,0.816,0.490),(0.5,1.000,1.000,0.749),(0.6,0.671,0.851,0.914),(0.8,0.271,0.604,0.773),(1.0,0.192,0.212,0.584)]
_RYG=[(0.0,0.647,0.000,0.149),(0.2,0.910,0.306,0.145),(0.4,0.988,0.816,0.490),(0.5,1.000,1.000,0.749),(0.6,0.741,0.902,0.490),(0.8,0.290,0.745,0.439),(1.0,0.000,0.408,0.216)]
_CW=[(0.0,0.086,0.404,0.859),(0.25,0.573,0.773,0.871),(0.5,0.969,0.969,0.969),(0.75,0.957,0.647,0.510),(1.0,0.706,0.016,0.016)]
_TW=[(0.0,0.886,0.816,0.859),(0.25,0.376,0.298,0.647),(0.5,0.141,0.141,0.141),(0.75,0.471,0.271,0.133),(1.0,0.886,0.816,0.859)]
_OC=[(0.0,0.0,0.0,0.4),(0.5,0.0,0.5,0.5),(1.0,0.0,0.8,0.0)]
_GE=[(0.0,0.0,0.2,0.0),(0.2,0.1,0.4,0.1),(0.4,0.4,0.6,0.2),(0.6,0.7,0.7,0.5),(0.8,0.9,0.85,0.7),(1.0,1.0,1.0,1.0)]
_TE=[(0.0,0.2,0.2,0.8),(0.15,0.0,0.5,1.0),(0.25,0.0,0.8,0.4),(0.5,0.6,0.8,0.3),(0.75,0.8,0.7,0.5),(1.0,1.0,1.0,1.0)]
_CM=[(0.0,0.0,0.0,0.0),(0.2,0.2,0.0,0.6),(0.4,0.0,0.4,0.8),(0.6,0.6,0.8,0.0),(0.8,1.0,0.6,0.0),(1.0,1.0,1.0,1.0)]
_GS=[(0.0,0.0,0.0,0.0),(0.09,1.0,0.0,0.0),(0.10,0.0,0.0,0.0),(0.5,0.5,0.5,0.5),(1.0,1.0,1.0,1.0)]
_NI=[(0.0,0.0,0.0,0.0),(0.1,0.5,0.0,0.5),(0.2,0.0,0.0,1.0),(0.35,0.0,0.8,1.0),(0.5,0.0,0.7,0.0),(0.65,0.8,0.8,0.0),(0.75,1.0,0.5,0.0),(0.9,1.0,0.0,0.0),(0.95,1.0,1.0,1.0),(1.0,1.0,1.0,1.0)]
_NC=[(0.0,0.0,0.0,0.5),(0.1,0.0,0.5,1.0),(0.2,0.0,1.0,1.0),(0.3,0.5,1.0,0.0),(0.4,1.0,1.0,0.0),(0.5,1.0,0.5,0.0),(0.6,1.0,0.0,0.0),(0.7,0.8,0.0,0.5),(0.8,0.5,0.0,0.8),(0.9,0.8,0.5,0.8),(1.0,1.0,1.0,1.0)]
_ORD=[(0.0,1.0,0.97,0.94),(0.5,0.99,0.55,0.24),(1.0,0.55,0.00,0.00)]
_PUD=[(0.0,0.97,0.96,0.98),(0.5,0.78,0.49,0.72),(1.0,0.45,0.00,0.33)]
_RPU=[(0.0,1.0,0.97,0.97),(0.5,0.97,0.41,0.60),(1.0,0.49,0.00,0.27)]
_BPU=[(0.0,0.97,0.94,0.98),(0.5,0.55,0.59,0.78),(1.0,0.30,0.00,0.29)]
_GNB=[(0.0,0.97,0.99,0.94),(0.5,0.40,0.76,0.64),(1.0,0.03,0.25,0.50)]
_PUB=[(0.0,1.0,0.97,0.98),(0.5,0.39,0.67,0.81),(1.0,0.02,0.22,0.45)]
_YGB=[(0.0,1.0,1.0,0.85),(0.33,0.36,0.82,0.64),(0.67,0.04,0.52,0.70),(1.0,0.03,0.11,0.35)]
_PBG=[(0.0,1.0,0.97,0.98),(0.33,0.39,0.67,0.81),(0.67,0.10,0.70,0.54),(1.0,0.00,0.27,0.11)]
_BGN=[(0.0,0.97,0.99,0.98),(0.5,0.40,0.83,0.74),(1.0,0.00,0.27,0.11)]
_YGN=[(0.0,1.0,1.0,0.90),(0.5,0.42,0.76,0.39),(1.0,0.00,0.27,0.11)]

def _hsv(h,s=1.0,v=1.0):
    h=h%360; i=int(h/60)%6; f=h/60-int(h/60)
    p,q,t2=v*(1-s),v*(1-f*s),v*(1-(1-f)*s)
    return [(v,t2,p),(q,v,p),(p,v,t2),(p,q,v),(t2,p,v),(v,p,q)][i]
def _cubehelix(t,start=0.5,rot=-1.5,hue=1.0,gamma=1.0):
    t=t**gamma; a=2*math.pi*(start/3+rot*t); amp=hue*t*(1-t)/2
    return (_c(t+amp*(-0.14861*math.cos(a)+1.78277*math.sin(a))),
            _c(t+amp*(-0.29227*math.cos(a)-0.90649*math.sin(a))),
            _c(t+amp*(1.97294*math.cos(a))))
def _build_lut(name):
    L=lambda ctrl:(lambda t:_lerp_ctrl(ctrl,t))
    fns={
        "viridis":L(_VI),"inferno":L(_IN),"magma":L(_MA),"plasma":L(_PL),"turbo":L(_TU),
        "OrRd":L(_ORD),"PuRd":L(_PUD),"RdPu":L(_RPU),"BuPu":L(_BPU),"GnBu":L(_GNB),
        "PuBu":L(_PUB),"YlGnBu":L(_YGB),"PuBuGn":L(_PBG),"BuGn":L(_BGN),"YlGn":L(_YGN),
        "gist_gray":lambda t:(t,t,t),"gist_yarg":lambda t:(1-t,1-t,1-t),
        "hot":lambda t:(_c(t/0.3333),_c((t-0.3333)/0.3333),_c((t-0.6667)/0.3333)),
        "afmhot":lambda t:(_c(t/0.5),_c((t-0.25)/0.5),_c((t-0.5)/0.5)),
        "gist_heat":lambda t:(_c(t/0.4),_c((t-0.4)/0.35),_c((t-0.75)/0.25)),
        "bone":lambda t:(_c(t*0.875+(t-0.75)*0.125 if t>0.75 else t*0.875),
                         _c(t*0.875+(t-0.75)*0.125 if t>0.75 else t*0.875),
                         _c(t*0.875+t*0.125/0.75 if t<0.75 else 0.875+0.125*(t-0.75)/0.25)),
        "pink":lambda t:(_c(math.sqrt(max(0,t*2/3+(t/3 if t>0.5 else 0)))),
                         _c(math.sqrt(max(0,t*2/3+(t/3 if t<0.5 else 0)))),
                         _c(math.sqrt(t) if t>0.5 else 0)),
        "copper":lambda t:(_c(t*1.25),_c(t*0.7812),_c(t*0.4975)),
        "Spectral":L(_SP),"RdBu":L(_RB),"PiYG":L(_PY),"PRGn":L(_PR),"BrBG":L(_BB),
        "PuOr":L(_PO),"RdGy":L(_RG),"RdYlBu":L(_RYB),"RdYlGn":L(_RYG),"coolwarm":L(_CW),
        "seismic":lambda t:((_c(t*2),_c(t*2),1.0) if t<0.5 else (1.0,_c(2-t*2),_c(2-t*2))),
        "hsv":lambda t:_hsv(t*360),"twilight":L(_TW),
        "twilight_shifted":lambda t:_lerp_ctrl(_TW,(t+0.5)%1.0),
        "cool":lambda t:(t,1-t,1.0),"spring":lambda t:(1.0,t,1-t),
        "summer":lambda t:(t,_c(0.5+0.5*t),0.4),"autumn":lambda t:(1.0,t,0.0),
        "winter":lambda t:(0.0,t,_c(1-0.5*t)),
        "rainbow":lambda t:_hsv((1-t)*270),"gist_rainbow":lambda t:_hsv((1-t)*300),
        "jet":lambda t:(_c(1.5-abs(4*t-3)),_c(1.5-abs(4*t-2)),_c(1.5-abs(4*t-1))),
        "brg":lambda t:(_c(2-4*t) if t<0.5 else _c(4*t-2),0.0,
                        _c(4*t) if t<0.25 else (_c(2-4*t) if t<0.5 else 0.0)),
        "prism":lambda t:_hsv((t*10%1)*360),
        "ocean":L(_OC),"gist_earth":L(_GE),"terrain":L(_TE),"CMRmap":L(_CM),
        "gist_stern":L(_GS),"gnuplot":lambda t:(_c(t),0.0,_c(t*t)),
        "gnuplot2":lambda t:(_c(2*t),_c(2*t-1),_c(4*t-3)),
        "cubehelix":lambda t:_cubehelix(t),"nipy_spectral":L(_NI),"gist_ncar":L(_NC),
    }
    return _make_lut(fns.get(name, fns["inferno"]))

_ALL_CMAPS = [
    "inferno","viridis","magma","OrRd","PuRd","RdPu","BuPu","GnBu","PuBu","YlGnBu",
    "PuBuGn","BuGn","YlGn","gist_yarg","gist_gray","bone","pink","hot","afmhot",
    "gist_heat","PiYG","PRGn","BrBG","PuOr","RdGy","RdBu","RdYlBu","RdYlGn",
    "Spectral","coolwarm","seismic","twilight","twilight_shifted","hsv","prism",
    "ocean","gist_earth","terrain","gist_stern","gnuplot","gnuplot2","CMRmap",
    "cubehelix","brg","gist_rainbow","rainbow","jet","turbo","nipy_spectral","gist_ncar",
]
_LUT_CACHE = {}
def get_lut(name):
    if name not in _LUT_CACHE:
        _LUT_CACHE[name] = _build_lut(name)
    return _LUT_CACHE[name]

def apply_lut(magnitude_01, cmap_name):
    """Apply colormap LUT to a [0,1] float array; return (H,W,3) uint8 array."""
    lut = get_lut(cmap_name)
    return lut[(np.clip(magnitude_01, 0, 1) * 255).astype(np.uint8)]

# Pre-warm all LUTs at startup
for _n in _ALL_CMAPS:
    get_lut(_n)


# ---------------------------------------------------------------------------
# Gaussian filter  (replaces scipy.ndimage.gaussian_filter)
# ---------------------------------------------------------------------------

def gaussian_filter_np(arr, sigma):
    if sigma <= 0:
        return arr
    radius  = max(1, int(math.ceil(3 * sigma)))
    x       = np.arange(-radius, radius + 1, dtype=float)
    kernel  = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    arr = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), axis=1, arr=arr)
    arr = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), axis=0, arr=arr)
    return arr


# ---------------------------------------------------------------------------
# Point-in-polygon  (replaces matplotlib.path.Path.contains_points)
# ---------------------------------------------------------------------------

def _points_in_polygon(coords, polygon):
    result = np.zeros(len(coords), dtype=bool)
    cx, cy = coords[:, 0], coords[:, 1]
    n = len(polygon); j = n - 1
    for i in range(n):
        xi, yi = polygon[i]; xj, yj = polygon[j]
        cond1 = (cy > yi) != (cy > yj)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (xj - xi) * (cy - yi) / (yj - yi) + xi
        result ^= cond1 & (cx < x_int)
        j = i
    return result


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
                if rest not in _ALL_CMAPS:
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
        coords  = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        mask    = _points_in_polygon(coords, polygon).reshape((h, w))

    else:
        mask = np.ones_like(arr, dtype=bool)

    return np.where(mask, arr, 0)


def apply_fft_processing(magnitude: np.ndarray, params: dict) -> np.ndarray:
    """Apply smoothing, zoom, percentile clip, and intensity-range rescaling."""
    sigma = float(params["gaussian_sigma"])
    if sigma > 0:
        magnitude = gaussian_filter_np(magnitude, sigma)

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

    print(f"Running MC simulation ({cycles} cycles)...")
    if cycles >= 2:
        log_path = os.path.join(out_dir, "MC_global_energy_log.txt")
        with open(log_path, "w") as log_file:
            log_file.write("cycle\told_energy\tnew_energy\tcurrent_energy\n")

            for cycle in range(cycles):
                i1, j1 = random.randint(0, n - 1), random.randint(0, n - 1)
                i2, j2 = random.randint(0, n - 1), random.randint(0, n - 1)

                old_energy = calculate_energy(grid, name_to_idx, interactions)
                grid[i1, j1], grid[i2, j2] = grid[i2, j2], grid[i1, j1]
                new_energy = calculate_energy(grid, name_to_idx, interactions)

                if new_energy < old_energy:
                    current_energy = new_energy
                else:
                    grid[i1, j1], grid[i2, j2] = grid[i2, j2], grid[i1, j1]
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
    else:
        print("  MC simulation skipped (cycles < 2).")

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
    rgb       = apply_lut(magnitude, cmap_name)
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
            diff_rgb  = apply_lut(diff_mag, cmap_name)
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
            W, H = 600, 400
            pad_l, pad_r, pad_t, pad_b = 72, 20, 44, 70
            pw = W - pad_l - pad_r
            ph = H - pad_t - pad_b
            mn, mx = min(mc_energies), max(mc_energies)
            if mx == mn: mx = mn + 1.0
            plot_img  = Image.new("RGB", (W, H), "white")
            draw      = ImageDraw.Draw(plot_img)
            # Grid + Y ticks
            for i in range(5):
                y   = pad_t + ph - int(i * ph / 4)
                val = mn + (mx - mn) * i / 4
                draw.line([(pad_l, y), (pad_l + pw, y)], fill="#cccccc")
                draw.text((pad_l - 4, y), f"{val:.4g}", fill="black", anchor="rm")
            # Axes box
            draw.rectangle([pad_l, pad_t, pad_l + pw, pad_t + ph], outline="black")
            # Labels
            draw.text((W // 2, pad_t // 2), "Global energy plot", fill="black", anchor="mm")
            draw.text((W // 2, H - 8), "MC cycle number", fill="black", anchor="mm")
            draw.text((10, pad_t + ph // 2), "Global energy", fill="black", anchor="mm")
            # Data line
            if len(mc_energies) >= 2:
                pts = []
                for i, e in enumerate(mc_energies):
                    x = pad_l + int(i / (len(mc_energies) - 1) * pw)
                    y = pad_t + ph - int((e - mn) / (mx - mn) * ph)
                    pts.append((x, y))
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i + 1]], fill="#2F24D8", width=2)
            idx_en    = next_available_index(out_dir, "MC_energy")
            plot_path = os.path.join(out_dir, f"MC_energy_{idx_en:02d}.png")
            plot_img.save(plot_path, dpi=(150, 150))
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
