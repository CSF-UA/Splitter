"""Auto splitting: period -> phase template -> one window per extremum in every cycle.

A window is the part of the template beyond a level between the extremum and its
neighbouring opposite extremum. It is mapped onto every cycle, re-centred by matching
the template, and dropped if it has too few points, a gap, a missing branch, or points
already used by a previous layer.
"""

import os

import numpy as np
from scipy.signal import find_peaks

from src.core import get_data, save_data

NB = 200  # template phase bins


def _dispersion(x, y, period, nb=100):
    """Within-bin variance of the folded curve over the total variance (PDM theta)."""
    idx = ((x / period) % 1.0 * nb).astype(int) % nb
    n = np.bincount(idx, minlength=nb)
    s = np.bincount(idx, y, nb)
    s2 = np.bincount(idx, y * y, nb)
    ok = n > 1
    return (s2[ok] - s[ok] ** 2 / n[ok]).sum() / max(n[ok].sum() - ok.sum(), 1) / np.var(y)


def find_period(x, y, pmin=0.05):
    """Periodogram peak, refined by PDM, then multiplied by 2 or 3 while that folds better
    (the peak may be a harmonic of narrow eclipses, or P/2 of two different minima).
    The periodogram is an FFT of the light curve averaged onto a regular time grid.
    Returns 0.0 when there is no reliable period: longer than the data allow, or no signal."""
    span = x[-1] - x[0]
    dt = max(float(np.median(np.diff(x))), span / 1e6)  # ponytail: grid capped at 1e6 cells
    k = ((x - x[0]) / dt).astype(int)
    grid = np.bincount(k, y - y.mean()) / np.maximum(np.bincount(k), 1)  # empty cells stay 0
    power = np.abs(np.fft.rfft(grid, 5 * grid.size)) ** 2  # 5x zero padding: 5 samples per peak
    freqs = np.fft.rfftfreq(5 * grid.size, dt)
    band = (freqs >= 2 / span) & (freqs <= 1 / pmin)
    peak = int(np.argmax(power[band]))
    if peak < 3:  # at the long-period edge of the band: the period is comparable to the data span
        return 0.0
    f0 = freqs[band][peak]
    trial = np.arange(f0 - 0.2 / span, f0 + 0.2 / span, 0.01 / span)
    period = 1 / trial[np.argmin([_dispersion(x, y, 1 / f, NB) for f in trial])]
    improved = True
    while improved:  # compare with the same bin width in time: k*NB bins for k*P
        improved = False
        for k in (2, 3):
            if k * period <= span / 2 and _dispersion(x, y, k * period, k * NB) < 0.85 * _dispersion(x, y, period, NB):
                period, improved = k * period, True
                break
    return period if _dispersion(x, y, period, NB) < 0.95 else 0.0  # pure noise folds to theta ~ 1


def _template(x, y, period):
    """Median light curve in NB phase bins (lightly smoothed) and the noise of one bin."""
    idx = ((x / period) % 1.0 * NB).astype(int) % NB
    tpl = np.array([np.median(y[idx == k]) if np.any(idx == k) else np.nan for k in range(NB)])
    ok = np.isfinite(tpl)
    tpl = np.interp(np.arange(NB), np.flatnonzero(ok), tpl[ok], period=NB)
    tpl = (np.roll(tpl, 1) + tpl + np.roll(tpl, -1)) / 3
    resid = y - tpl[idx]
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) / np.sqrt(max(x.size / NB, 1))
    return tpl, noise


def _extrema(tpl, noise):
    """[(bin, 'min'|'max')]; 'min' = brightness minimum = magnitude maximum."""
    prom = max(0.05 * np.ptp(tpl), 5 * noise)
    ext = []
    for kind, v in (("min", tpl), ("max", -tpl)):
        peaks, _ = find_peaks(np.tile(v, 3), prominence=prom)
        ext += [(int(b) - NB, kind) for b in peaks if NB <= b < 2 * NB]
    return sorted(ext)


def _star_type(tpl, ext, noise):
    flat = np.mean(tpl < tpl.min() + max(0.02 * np.ptp(tpl), 3 * noise))
    if flat > 0.3:  # ponytail: naive flat-maxima heuristic; calibrate on a VSX-typed sample
        return "EA"
    depths = sorted((tpl[b] - tpl.min() for b, k in ext if k == "min"), reverse=True)
    if len(depths) < 2:
        return "pulsator"
    return "EW" if depths[1] > 0.8 * depths[0] else "EB"


def _half_widths(tpl, b0, kind, frac, ext):
    """Bins left/right of b0 where the template is beyond y0 - frac * (y0 - ref),
    ref = the less extreme neighbouring opposite extremum. Each side <= 2x the other."""
    y = tpl if kind == "min" else -tpl
    opp = [b for b, k in ext if k != kind]
    if opp:
        left = min(opp, key=lambda b: (b0 - b) % NB)
        right = min(opp, key=lambda b: (b - b0) % NB)
        ref = max(y[left], y[right])
    else:
        ref = y.min()
    level = y[b0] - frac * (y[b0] - ref)
    lo = hi = 0
    while lo < NB // 2 and y[(b0 - lo - 1) % NB] >= level:
        lo += 1
    while hi < NB // 2 and y[(b0 + hi + 1) % NB] >= level:
        hi += 1
    return min(lo, 2 * max(hi, 1)), min(hi, 2 * max(lo, 1))


def auto_split(x, y, period=0.0, frac=0.0, min_points=15, minima_only=False, excluded=None):
    """Return (start, finish, kinds, info): inclusive indices, kinds 'min'/'max' (brightness),
    info = {period, type, rejected}. period <= 0 means: find it (type 'no period' and no windows
    if there is none); frac 0 means 0.95, the width that gave the best O-C on real TESS stars."""
    period = period if period > 0 else find_period(x, y)
    if period <= 0:
        return [], [], [], {"period": 0.0, "type": "no period", "rejected": {}}
    tpl, noise = _template(x, y, period)
    ext = _extrema(tpl, noise)
    star = _star_type(tpl, ext, noise)
    frac = frac or 0.95
    used = np.zeros(x.size, bool)
    if excluded:
        used[list(excluded)] = True
    out, rejected = [], {}
    for b0, kind in ext:
        if kind == "max" and (minima_only or star == "EA"):
            continue
        lo, hi = _half_widths(tpl, b0, kind, frac, ext)
        center = (b0 + 0.5) / NB
        for n in range(int(x[0] / period) - 1, int(x[-1] / period) + 2):
            tc = (n + center) * period
            a, b = tc - (lo + 0.5) / NB * period, tc + (hi + 0.5) / NB * period
            if a < x[0] or b > x[-1]:
                continue
            w = b - a
            j0, j1 = np.searchsorted(x, [a - 0.3 * w, b + 0.3 * w])  # x is sorted: slice, don't scan
            if j1 - j0 >= min_points:  # re-centre by template matching: period error, O-C drift
                xn, yn = x[j0:j1], y[j0:j1]
                shifts = np.linspace(-0.25, 0.25, 51) * w
                bins = [(((xn - d) / period) % 1.0 * NB).astype(int) % NB for d in shifts]
                d = shifts[int(np.argmin([np.var(yn - tpl[i]) for i in bins]))]
                a, b, tc = a + d, b + d, tc + d
            i0, i1 = int(np.searchsorted(x, a)), int(np.searchsorted(x, b, side="right")) - 1
            seg = x[i0 : i1 + 1]
            if seg.size < min_points:
                reason = "too few points"
            elif np.diff(np.r_[a, seg, b]).max() > 0.2 * w:
                reason = "gap"
            elif min((seg < tc).sum(), (seg > tc).sum()) < 0.2 * seg.size:
                reason = "one branch missing"
            elif used[i0 : i1 + 1].any():
                reason = "used by previous layer"
            else:
                out.append((i0, i1, kind))
                continue
            rejected[reason] = rejected.get(reason, 0) + 1
    out.sort()
    info = {"period": float(period), "type": star, "rejected": rejected}
    return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out], info


def batch(paths, period=0.0):
    """Headless Auto: write <name>_intervals.txt next to every light curve (period 0 = find it)."""
    for path in paths:
        try:
            x, y = get_data(path)
            s, f, kinds, info = auto_split(x, y, period)
            out = save_data(s, f, kinds, os.path.splitext(path)[0] + "_intervals.txt")
            print(f"{path}: P={info['period']:.6f} d, {info['type']}, {kinds.count('min')} min + "
                  f"{kinds.count('max')} max, rejected {info['rejected']} -> {out}")
        except Exception as e:  # one bad file must not stop the batch
            print(f"{path}: FAILED: {e}")
