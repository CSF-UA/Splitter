"""Auto splitting: period -> phase template -> one window per extremum in every cycle.

Narrow one-sided dips (eclipses) on slower variability (spots, trends) are analysed after
removing that variability, so the period is the orbital one, not the rotation of the spots.
A window is the part of the template beyond a level between the extremum and its
neighbouring opposite extremum. It is mapped onto every cycle, re-centred by matching
the template, and dropped if it has too few points, a gap, a missing branch, or points
already used by a previous layer.
"""

import os

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_prominences

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


def _refine(x, y, f0, span, fine=False):
    """Period with the least PDM dispersion near frequency f0, on a coarse grid (then a fine one)."""
    for half, step in ((0.2, 0.01), (0.01, 0.001))[: 1 + fine]:
        trial = np.arange(f0 - half / span, f0 + half / span, step / span)
        f0 = trial[np.argmin([_dispersion(x, y, 1 / f, NB) for f in trial])]
    return 1 / f0


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
    period = _refine(x, y, freqs[band][peak], span)
    improved = True
    while improved:  # compare with the same bin width in time: k*NB bins for k*P
        improved = False
        for k in (2, 3):
            if k * period <= span / 2 and _dispersion(x, y, k * period, k * NB) < 0.85 * _dispersion(x, y, period, NB):
                period, improved = k * period, True
                break
    return period if _dispersion(x, y, period, NB) < 0.95 else 0.0  # pure noise folds to theta ~ 1


def _eclipses(x, y, width=0.5):
    """(trend, period, dip times, tol) if the curve has periodic narrow dips that only go fainter
    (eclipses) deeper than the slower variability (spots, pulsations, trends), else None. The trend is a running
    median over `width` days in each gap-free segment with outliers on both sides masked. The dip
    times give the period, PDM refines it; tol is half the median dip width. Near segment edges the
    median is one-sided, so dips there are not used."""
    if x.size < 100:
        return None
    dt = np.diff(x)
    cad = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.0  # repeated time stamps: the real step
    if cad <= 0:
        return None
    n = max(int(width / cad) | 1, 5)
    seg = np.r_[0, np.cumsum(dt > 10 * cad)]  # gap-free segments
    pos = np.arange(x.size) - np.flatnonzero(np.diff(np.r_[-1, seg]))[seg]  # index inside the segment
    edge = np.minimum(pos, np.bincount(seg)[seg] - 1 - pos) < n // 2
    keep, trend = np.ones(x.size, bool), np.empty(x.size)
    for _ in range(3):
        for i in range(seg[-1] + 1):
            m = seg == i
            k = m & keep
            trend[m] = np.interp(x[m], x[k], median_filter(y[k], n, mode="nearest")) if k.any() else np.median(y[m])
        r = y - trend
        sig = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        if sig == 0:
            return None  # no noise (simulated data): no dips to tell apart
        keep = np.abs(r) < 3 * sig  # both sides: a faint extremum masked alone would drag the median down
    dips, bright = (r[~edge] > 6 * sig).sum(), (r[~edge] < -6 * sig).sum()
    if dips < 10 or bright >= 0.1 * dips:
        return None
    rs = np.convolve(r, np.ones(5) / 5, "same")  # a dip: a run of the 5-point mean above 6 of its sigmas
    on = np.diff(np.r_[0, rs > 6 * sig / np.sqrt(5), 0].astype(int))
    runs, last = [], (0, 0)
    for a, b in zip(np.flatnonzero(on == 1), np.flatnonzero(on == -1)):
        if runs and seg[a] == seg[last[1] - 1] and x[a] - x[last[1] - 1] < max(x[last[1] - 1] - x[last[0]], x[b - 1] - x[a]):
            runs[-1] = (runs[-1][0], b)  # one eclipse split by noise or pulsations
        else:
            runs.append((a, b))
        last = (a, b)
    runs = [(a, b) for a, b in runs if b - a >= 5 and not edge[a:b].any() and seg[a] == seg[b - 1]]
    t = np.array([(x[a] + x[b - 1]) / 2 for a, b in runs])  # mid-point at that level: robust to flat bottoms
    depth = np.array([rs[a:b].max() for a, b in runs])
    tol = float(np.median([x[b - 1] - x[a] for a, b in runs])) / 2 if runs else 0.0
    if not runs or np.percentile(trend, 99) - np.percentile(trend, 1) >= np.median(depth):
        return None  # the slower variability dominates: its own sharp features are no eclipses to time
    local = t < t[0] + 30 if t.size else t.astype(bool)  # one sector: the rough period error stays small
    inner = x[~edge]  # cycles count as covered only where a dip could have been found
    period = _dip_period(inner[inner < t[0] + 30 + tol], t[local], depth[local], tol) if local.sum() >= 3 else 0.0
    if not period:
        return None  # dips, but not periodic ones: leave the curve as it is
    return trend, _refine(x[~edge], r[~edge], 1 / period, x[-1] - x[0], fine=True), t, tol


def _dip_period(x, t, depth, tol):
    """Smallest period that puts the dips (80 % if there are strays) into one or two phase groups,
    the main group one kind of eclipse (depths within 1.5x) seen in 80 % of the cycles the data
    cover (a harmonic misses most of them). With one group, doubled if depths or times alternate:
    two different eclipses. 0 if there is none."""
    cands = {(t[j] - t[i]) / m for i in range(min(6, t.size)) for j in range(i + 1, min(i + 4, t.size)) for m in range(1, 21)}
    for P in sorted(c for c in cands if c > 4 * tol):
        o = np.sort((t - t[0]) / P % 1)
        cut = np.flatnonzero(np.diff(np.r_[o, o[0] + 1]) > tol / P)  # gaps between groups, circular
        if cut.size == 0:
            continue  # dips at every phase: P is too short
        sizes = np.diff(np.r_[cut, cut[0] + o.size])  # group k: sorted phases cut[k]+1 .. cut[k+1]
        if sizes.size > 2 and ((sizes >= 2).sum() > 2 or sizes[sizes >= 2].sum() < 0.8 * t.size):
            continue  # one or two groups of any size, or up to 20 % stray dips
        k = int(np.argmax(sizes))
        g = o[(cut[k] + 1 + np.arange(sizes[k])) % o.size]
        c = g[0] + ((g - g[0]) % 1).mean()  # circular mean phase of the main group
        main = np.abs(((t - t[0]) / P - c + 0.5) % 1 - 0.5) < tol / P
        if depth[main].max() > 1.5 * depth[main].min():
            continue  # a primary and a secondary eclipse in one group
        T = t[0] + (np.arange(np.floor((x[0] - t[0]) / P - c), np.ceil((x[-1] - t[0]) / P - c) + 1) + c) * P
        lo, hi = np.searchsorted(x, T - 1.5 * tol), np.searchsorted(x, T + 1.5 * tol)
        T = T[[b - a >= 3 and x[a] - tc < -1.4 * tol and x[b - 1] - tc > 1.4 * tol and np.diff(x[a:b]).max() < tol / 2
               for tc, a, b in zip(T, lo, hi)]]  # cycles whose whole dip is in the data: no gap in +-1.5 tol
        if T.size and (np.abs(t[None, :] - T[:, None]).min(axis=1) < tol).mean() >= 0.8:  # P/2: about half
            break
    else:
        return 0.0
    e = np.round((t[main] - t[0]) / P - c)
    if np.ptp(e) == 0 or (sizes >= 2).sum() == 2 or sizes.size == 2:  # two groups: two eclipses already apart
        return float(np.polyfit(e, t[main], 1)[0]) if np.ptp(e) else float(P)
    fit = np.polyfit(e, t[main], 1)
    odd = e.astype(int) % 2 == 1

    def alternate(v, floor):  # medians and MADs: one odd dip must not decide
        a, b = v[~odd], v[odd]
        mad = lambda u: 1.4826 * np.median(np.abs(u - np.median(u)))
        return min(a.size, b.size) >= 2 and abs(np.median(a) - np.median(b)) > max(
            6 * np.sqrt(mad(a) ** 2 / a.size + mad(b) ** 2 / b.size), floor)

    two = alternate(depth[main], 0.03 * depth[main].max()) or alternate(t[main] - np.polyval(fit, e), tol / 4)
    return float(fit[0] * (2 if two else 1))


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
    info = {period, type, eclipses, rejected}. period <= 0 means: find it (type 'no period' and no
    windows if there is none); frac 0 means 0.95, the width that gave the best O-C on real TESS stars."""
    ecl = _eclipses(x, y)
    if ecl is not None and period > 0:
        k = max(period / ecl[1], ecl[1] / period)
        if round(k) > 3 or abs(k - round(k)) > 0.01 * k:
            ecl = None  # not the eclipse period or 2, 3 times it or a half, third: e.g. the spot wave
    if ecl is not None:
        y = y - ecl[0]
    period = period if period > 0 else ecl[1] if ecl else find_period(x, y)
    if period <= 0:
        return [], [], [], {"period": 0.0, "type": "no period", "eclipses": ecl is not None, "rejected": {}}
    tpl, noise = _template(x, y, period)
    if ecl is not None:  # what is left between the eclipses (spots, pulsations) is correlated within a cycle:
        noise *= np.sqrt(max(x.size / NB / max((x[-1] - x[0]) / period, 1), 1))  # count cycles, not points
    ext = _extrema(tpl, noise)
    star = _star_type(tpl, ext, noise)
    if ecl is not None:  # eclipses, or extrema of 20 % of the amplitude (not what is left of spots, pulsations)
        dip_ph = ecl[2] / period % 1
        prom = lambda b, k: peak_prominences(np.tile(tpl if k == "min" else -tpl, 3), [b + NB])[0][0]
        ext = [(b, k) for b, k in ext if prom(b, k) >= 0.2 * np.ptp(tpl) or
               k == "min" and np.abs(((b + 0.5) / NB - dip_ph + 0.5) % 1 - 0.5).min() < ecl[3] / period + 2 / NB]
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
    info = {"period": float(period), "type": star, "eclipses": ecl is not None, "rejected": rejected}
    return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out], info


def batch(paths, period=0.0):
    """Headless Auto: write <name>_intervals.txt next to every light curve (period 0 = find it)."""
    for path in paths:
        try:
            x, y = get_data(path)
            s, f, kinds, info = auto_split(x, y, period)
            out = save_data(s, f, kinds, os.path.splitext(path)[0] + "_intervals.txt")
            print(f"{path}: P={info['period']:.6f} d, {info['type']}{', eclipse mode' * info['eclipses']}, "
                  f"{kinds.count('min')} min + "
                  f"{kinds.count('max')} max, rejected {info['rejected']} -> {out}")
        except Exception as e:  # one bad file must not stop the batch
            print(f"{path}: FAILED: {e}")
