"""Self-checks on synthetic light curves. Run: uv run python tests/test_core.py"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auto import auto_split, batch  # noqa: E402
from src.core import NN, check_up, get_data, save_data, splitting_algol_configurable, splitting_normal  # noqa: E402

rng = np.random.default_rng(0)
X = np.arange(0, 27, 2 / 1440)
X = X[(X < 13) | (X > 14)]  # TESS-like sector with a mid-sector gap


def dip(ph, center, width, depth):
    d = np.abs((ph - center + 0.5) % 1 - 0.5)
    return np.where(d < width, depth * (1 - d / width), 0.0)


def true_centers(period, phases, x=X):
    return [(n + p) * period for n in range(-1, int(x[-1] / period) + 2) for p in phases]


def one_extremum_each(x, start, finish, centers):
    c = np.array(centers)
    return all(np.sum((c >= x[s]) & (c <= x[f])) == 1 for s, f in zip(start, finish))


def test_sdips_kinds():
    P = 0.5
    y = -200 * np.sin(2 * np.pi * X / P) + rng.normal(0, 5, X.size)  # magnitude max (= min) at phase 0.75
    s, f, kinds = check_up(X, y, *splitting_normal(X, y, P), P)
    assert len(s) > 80 and set(kinds) == {"min", "max"}
    minima = true_centers(P, (0.75,))
    for a, b, k in zip(s, f, kinds):
        assert (k == "min") == any(X[a] <= c <= X[b] for c in minima)


def test_nn_scales_with_cadence():
    assert NN(1.0, 0.12) == NN(1.0, 0.12, cadence=2 / 1440)  # TESS 2-min: unchanged
    assert max(NN(1.0, 0.12, cadence=20 / 86400)) > max(NN(1.0, 0.12))  # 20 s: more points per day


def test_gbat_kinds_and_fill():
    P = 1.3
    y = dip(X / P % 1, 0.3, 0.03, 500) + rng.normal(0, 3, X.size)  # phase 0.3: no dip cut by the gap
    s, f, kinds = splitting_algol_configurable(X, y)
    assert len(s) >= 18 and set(kinds) == {"min"}
    assert one_extremum_each(X, s, f, true_centers(P, (0.3,)))
    s2, f2, kinds2 = splitting_algol_configurable(X, y, fill_remaining=True)
    assert kinds2[: len(s)] == kinds and set(kinds2[len(s) :]) == {"max"}
    assert sum(b - a + 1 for a, b in zip(s2, f2)) == X.size  # fill covers every point once


def test_get_data_ragged_file_and_save_format():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "lc.tess"
        src.write_text("# header\n1.0 2.0\n2.0 3.0 extra\nbad line\n3.0\n")
        x, m = get_data(str(src))
        assert list(x) == [1.0, 2.0] and list(m) == [2.0, 3.0]
        out = save_data([0, 10], [5, 20], ["min", "max"], str(Path(d) / "iv.txt"))
        assert Path(out).read_text() == "0 5 min\n10 20 max\n"


def test_auto_ea():
    P = 1.3
    ph = X / P % 1
    y = dip(ph, 0.0, 0.03, 500) + dip(ph, 0.5, 0.03, 150) + rng.normal(0, 3, X.size)
    s, f, kinds, info = auto_split(X, y)
    assert abs(info["period"] - P) < 1e-3, info
    assert info["type"] == "EA" and set(kinds) == {"min"}, info
    assert one_extremum_each(X, s, f, true_centers(P, (0.0, 0.5)))
    assert len(s) >= 36  # ~40 minima in the data, a few lost at the gap and the edges


def test_auto_ew():
    P = 0.35
    y = 300 * np.cos(4 * np.pi * X / P) + 60 * np.cos(2 * np.pi * X / P) + rng.normal(0, 5, X.size)
    s, f, kinds, info = auto_split(X, y)
    assert abs(info["period"] - P) < 1e-3, info
    assert kinds.count("min") >= 130 and kinds.count("max") >= 130, (len(kinds), info)
    assert one_extremum_each(X, s, f, true_centers(P, (0.0, 0.25, 0.5, 0.75)))


def test_auto_pulsator_minima_only_and_given_period():
    P = 0.5
    y = -200 * np.sin(2 * np.pi * X / P) + rng.normal(0, 5, X.size)
    s, f, kinds, info = auto_split(X, y, period=P, minima_only=True)
    assert info["period"] == P
    assert set(kinds) == {"min"} and len(s) >= 45
    assert one_extremum_each(X, s, f, true_centers(P, (0.75,)))


def test_auto_ten_minute_cadence():
    P = 0.5
    x = np.arange(0, 27, 10 / 1440)
    y = -200 * np.sin(2 * np.pi * x / P) + rng.normal(0, 5, x.size)
    s, f, kinds, info = auto_split(x, y)
    assert abs(info["period"] - P) < 1e-3 and len(s) >= 100, (len(s), info)
    assert one_extremum_each(x, s, f, true_centers(P, (0.25, 0.75), x))


def test_auto_excludes_previous_layers():
    P = 0.5
    y = -200 * np.sin(2 * np.pi * X / P) + rng.normal(0, 5, X.size)
    s1, f1, _, _ = auto_split(X, y, period=P, minima_only=True)
    used = set(range(s1[0], f1[0] + 1))
    s2, f2, _, info = auto_split(X, y, period=P, minima_only=True, excluded=used)
    assert len(s2) == len(s1) - 1 and info["rejected"].get("used by previous layer") == 1
    assert all(not used & set(range(a, b + 1)) for a, b in zip(s2, f2))


def test_batch_writes_intervals():
    P = 0.5
    y = -200 * np.sin(2 * np.pi * X / P) + rng.normal(0, 5, X.size)
    with tempfile.TemporaryDirectory() as d:
        lc = Path(d) / "star.tess"
        np.savetxt(lc, np.c_[X, y])
        batch([str(lc), str(Path(d) / "missing.tess")])  # a bad file must not stop the batch
        lines = (Path(d) / "star_intervals.txt").read_text().split("\n")[:-1]
        assert len(lines) >= 90 and all(len(l.split()) == 3 for l in lines)


def test_auto_irregular_sampling():
    P = 0.5
    x = np.sort(np.random.default_rng(1).uniform(0, 27, 6000))  # ground-based-like, no regular cadence
    y = -200 * np.sin(2 * np.pi * x / P) + rng.normal(0, 5, x.size)
    s, f, kinds, info = auto_split(x, y)
    assert abs(info["period"] - P) < 1e-3 and len(s) >= 60, (len(s), info)
    assert one_extremum_each(x, s, f, true_centers(P, (0.25, 0.75), x))


def test_auto_default_windows_cover_both_branches():
    P = 0.5  # windows must be wide enough for a good polynomial fit (O-C scatter on real stars)
    y = -200 * np.sin(2 * np.pi * X / P) + rng.normal(0, 5, X.size)
    s, f, kinds, _ = auto_split(X, y, period=P)
    assert np.median([(X[b] - X[a]) / P for a, b in zip(s, f)]) > 0.83  # frac 0.95 on a sinusoid: 0.86 P


def test_auto_ea_narrow_eclipses_not_a_harmonic():
    P = 1.5
    for secondary in (0.0, 20.0):  # the periodogram peak is a harmonic of such narrow eclipses
        ph = X / P % 1
        y = dip(ph, 0.0, 0.015, 400) + dip(ph, 0.5, 0.015, secondary) + rng.normal(0, 3, X.size)
        s, f, kinds, info = auto_split(X, y)
        assert abs(info["period"] - P) < 1e-3, (secondary, info)
        assert len(s) >= 15 and one_extremum_each(X, s, f, true_centers(P, (0.0, 0.5)))


def test_auto_period_longer_than_data():
    y = 100 * np.sin(2 * np.pi * X / 40.0) + rng.normal(0, 3, X.size)
    s, f, kinds, info = auto_split(X, y)
    assert s == [] and info["type"] == "no period" and info["period"] == 0.0, (len(s), info)


def test_auto_long_baseline_is_fast():
    import time
    x = np.concatenate([np.arange(0, 27, 2 / 1440) + 60 * k for k in range(13)])  # 13 sectors over 2 years
    y = -200 * np.sin(2 * np.pi * x / 0.3) + rng.normal(0, 5, x.size)
    t = time.perf_counter()
    s, f, kinds, info = auto_split(x, y)
    assert time.perf_counter() - t < 10 and len(s) > 2000, (time.perf_counter() - t, len(s))


def test_auto_shallow_narrow_ea():
    P = 2.2  # 60 mmag eclipses 2 % of the period long in 8 mmag noise: theta ~0.7 at the right period
    y = dip(X / P % 1, 0.0, 0.01, 60) + rng.normal(0, 8, X.size)
    s, f, kinds, info = auto_split(X, y)
    assert abs(info["period"] - P) < 2e-3, info
    assert len(s) >= 9 and one_extremum_each(X, s, f, true_centers(P, (0.0, 0.5)))


def test_auto_pure_noise_no_windows():
    s, f, kinds, info = auto_split(X, rng.normal(0, 5, X.size))
    assert s == [] and info["type"] == "no period", (len(s), info)


def test_batch_uses_given_period():
    P = 2.2
    y = dip(X / P % 1, 0.0, 0.01, 60) + rng.normal(0, 8, X.size)
    with tempfile.TemporaryDirectory() as d:
        lc = Path(d) / "ea.tess"
        np.savetxt(lc, np.c_[X, y])
        batch([str(lc)], period=P)
        s, f = zip(*[map(int, l.split()[:2]) for l in (Path(d) / "ea_intervals.txt").read_text().splitlines()])
        assert len(s) >= 9 and one_extremum_each(X, s, f, true_centers(P, (0.0,)))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
