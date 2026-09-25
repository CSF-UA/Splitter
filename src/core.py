"""Light-curve I/O and the v4 splitting algorithms: S-DIPS and GB-AT."""

import numpy as np


def get_data(name: str, mmin: float = -1000, mmax: float = 1000):
    """Read a two-column text file (JD mag); '#' lines are comments, unparsable lines are skipped."""
    try:
        data = np.loadtxt(name, comments="#", ndmin=2)
    except (FileNotFoundError, OSError):
        raise FileNotFoundError(f"File not found: {name}")
    except ValueError:
        rows = []
        with open(name, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    a = line.split()
                    rows.append((float(a[0]), float(a[1])))
                except (ValueError, IndexError):
                    pass
        data = np.array(rows).reshape(-1, 2)
    if data.shape[1] < 2:
        return np.array([]), np.array([])
    x, m = data[:, 0], data[:, 1]
    keep = (m >= mmin) & (m <= mmax)
    return x[keep], m[keep]


def vectorized_sm(y: np.ndarray, N: int) -> np.ndarray:
    """Moving average over 2N+1 points; the first and last N points stay unchanged."""
    if N <= 0 or len(y) < 2 * N + 1:
        return y.copy()
    Y = y.copy()
    Y[N : len(y) - N] = np.convolve(y, np.ones(2 * N + 1) / (2 * N + 1), mode="valid")
    return Y


def NN(T0: float, alpha: float, cadence: float = 2 / 1440):
    """Half-widths of the smoothing cascade: int(0.7734 e^(0.4484 k)), k = 2, 3, ...
    up to alpha * P * points-per-day; 584 was tuned on TESS 2-min data."""
    nmax = alpha * T0 * 584 * (2 / 1440) / cadence
    N, nn, k = [], 0, 2
    while nn < nmax:
        nn = int(0.7734 * np.exp(0.4484 * k))
        N.append(nn)
        k += 1
    return N


def smooth(T0: float, alpha: float, y, cadence: float = 2 / 1440):
    """Cascade of boxcars with growing, then shrinking windows; returns (smoothed, largest half-width)."""
    N = NN(T0, alpha, cadence)
    for n in N + N[::-1]:
        y = vectorized_sm(y, n)
    return y, max(N, default=0)


def splitting_normal(x, y, T0: float, alpha: float = 0.12, excluded_indices: set = None):
    """S-DIPS: split where the smoothed 2nd derivative changes sign or the time gap exceeds P/2."""
    if len(x) < 3:
        return [], []
    yy, nmax = smooth(T0, alpha, y, float(np.median(np.diff(x))))
    dx = np.diff(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.diff(yy) / dx
        d[~np.isfinite(d)] = 0.0
        for n in (3, 5, 9, 13, 9, 5, 3):
            d = vectorized_sm(d, n)
        dd = np.diff(d) / dx[1:]
        dd[~np.isfinite(dd)] = 0.0
    for n in (3, 5, 9, 13, 9, 5, 3):
        dd = vectorized_sm(dd, n)
    first, last = nmax + 1, len(x) - nmax - 1  # the edges were never smoothed
    i = np.arange(first, last - 1)
    i = i[(i < len(dd)) & (i < len(dx))]
    if first >= last or i.size == 0:
        return [], []
    cut = i[(dx[i] > 0.5 * T0) | (dd[i] * dd[i - 1] < 0)]
    pairs = list(zip([first, *(cut + 1)], [*cut, last]))
    if excluded_indices:
        used = np.zeros(len(x), bool)
        used[list(excluded_indices)] = True
        pairs = [(s, f) for s, f in pairs if not used[s : f + 1].any()]
    return [s for s, _ in pairs], [f for _, f in pairs]


def extremum_kind(x, y):
    """Fit a parabola; return 'min'/'max' (brightness) if its vertex is strictly inside x, else None."""
    xc = x - x.mean()
    try:
        a, b, _ = np.polyfit(xc, y, 2)
    except np.linalg.LinAlgError:
        return None
    if abs(a) < 1e-9 or not xc[0] < -b / (2 * a) < xc[-1]:
        return None
    return "min" if a < 0 else "max"  # magnitudes: a < 0 means a faint peak


def check_up(x, y, start, finish, T0: float):
    """Keep intervals with >= 5 points, 0.003 P <= duration <= P and a parabola vertex inside."""
    kept = []
    for s, f in zip(start, finish):
        if not 0 <= s < f < len(x) or f - s + 1 < 5:
            continue
        if not 0.003 * T0 <= x[f] - x[s] <= T0:
            continue
        kind = extremum_kind(x[s : f + 1], y[s : f + 1])
        if kind:
            kept.append((int(s), int(f), kind))
    return tuple(list(v) for v in zip(*kept)) if kept else ([], [], [])


def splitting_algol_configurable(x, y, index_gap: int = 2, cut_ratio: float = 0.25,
                                 min_interval_points: int = 5, excluded_indices: set = None,
                                 is_inverted: bool = False, fill_remaining: bool = False):
    """GB-AT: points beyond base + r * amplitude (5th..99.5th percentile), grouped while the
    index step is <= index_gap. Leftover runs (fill_remaining) get the opposite kind."""
    if len(y) == 0:
        return [], [], []
    base, deep = np.percentile(y, [5, 99.5])
    amp = deep - base
    if amp <= 0:
        return [], [], []
    if is_inverted:
        idx = np.flatnonzero(y < base + amp * (1 - cut_ratio))
    else:
        idx = np.flatnonzero(y > base + amp * cut_ratio)
    if excluded_indices:
        idx = idx[~np.isin(idx, list(excluded_indices))]
    if idx.size == 0:
        return [], [], []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > index_gap) + 1)
    main, other = ("max", "min") if is_inverted else ("min", "max")
    rows = [(int(g[0]), int(g[-1]), main) for g in groups if g.size >= min_interval_points]
    if fill_remaining and rows:
        covered = np.zeros(len(x), bool)
        for s, f, _ in rows:
            covered[s : f + 1] = True
        if excluded_indices:
            covered[list(excluded_indices)] = True
        free = np.flatnonzero(~covered)
        runs = np.split(free, np.flatnonzero(np.diff(free) != 1) + 1)
        rows += [(int(g[0]), int(g[-1]), other) for g in runs if g.size]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def save_data(start, finish, kinds, fname: str):
    """Write 'start end kind' lines: inclusive point indices, kind = min | max (brightness)."""
    out = fname.replace(".tess", ".txt")
    if out == fname and not fname.lower().endswith(".txt"):
        out += ".txt"
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(f"{int(s)} {int(e)} {k}\n" for s, e, k in zip(start, finish, kinds))
    return out
