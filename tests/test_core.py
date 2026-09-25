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


def test_auto_spotted_ea():
    P, t0 = 9.44, 1.0  # narrow eccentric eclipses on a 10 mmag spot wave that rotates in 2.21 d (TIC 140659980)
    ph = (X - t0) / P % 1
    y = dip(ph, 0.0, 0.009, 60) + dip(ph, 0.473, 0.01, 35) + 5 * np.sin(2 * np.pi * X / 2.21) + rng.normal(0, 0.8, X.size)
    s, f, kinds, info = auto_split(X, y)
    assert abs(info["period"] - P) < 0.01 and info["type"] == "EA" and info["eclipses"], info
    assert len(s) >= 5 and one_extremum_each(X, s, f, true_centers(P, (t0 / P, t0 / P + 0.473)))
    s2, f2, _, _ = auto_split(X, y, period=info["period"])  # the GUI passes the found P back on the next run
    assert (s2, f2) == (s, f)


def test_auto_spotted_ea_three_plus_two_eclipses():
    P, t0, x = 9.44, 1.0, X[X < 24]  # 3 primaries + 2 secondaries: at P/2 the primaries alone fill 3 of 5 cycles
    ph = (x - t0) / P % 1
    y = dip(ph, 0.0, 0.009, 60) + dip(ph, 0.473, 0.01, 35) + 5 * np.sin(2 * np.pi * x / 2.21) + rng.normal(0, 0.8, x.size)
    s, f, kinds, info = auto_split(x, y)
    assert abs(info["period"] - P) < 0.01, info
    assert len(s) == 5 and one_extremum_each(x, s, f, true_centers(P, (t0 / P, t0 / P + 0.473), x))


def test_auto_narrow_eclipses_any_noise():
    P = 3.54  # 30 and 20 mmag eclipses 2.4 % of P long: the periodogram is a flat comb of harmonics
    for seed in range(5):
        r = np.random.default_rng(seed)
        ph = X / P % 1
        y = dip(ph, 0.0, 0.012, 30) + dip(ph, 0.5, 0.012, 20) + 4 * np.sin(2 * np.pi * X / 4.7) + r.normal(0, 1.5, X.size)
        s, f, kinds, info = auto_split(X, y)
        assert abs(info["period"] - P) < 2e-3 and len(s) >= 10, (seed, info)
        assert one_extremum_each(X, s, f, true_centers(P, (0.0, 0.5)))


def test_auto_eccentric_equal_eclipses():
    P = 4.3  # equal eclipses at phases 0 and 0.38: no single fold at P/2 or P/k puts them together
    ph = X / P % 1
    y = dip(ph, 0.0, 0.01, 150) + dip(ph, 0.38, 0.01, 150) + rng.normal(0, 2, X.size)
    s, f, kinds, info = auto_split(X, y)
    assert abs(info["period"] - P) < 2e-3 and len(s) >= 9, info
    assert one_extremum_each(X, s, f, true_centers(P, (0.0, 0.38)))


def test_auto_ea_with_pulsations():
    P = 1.9  # eclipses of a star that pulsates (delta Sct-like, 10 mmag): windows only at the eclipses
    ph = X / P % 1
    y = (dip(ph, 0.0, 0.025, 120) + dip(ph, 0.5, 0.025, 50) + 10 * np.sin(2 * np.pi * X / 0.0437)
         + 4 * np.sin(2 * np.pi * X / 0.0611) + rng.normal(0, 2, X.size))
    s, f, kinds, info = auto_split(X, y)
    assert abs(info["period"] - P) < 2e-3 and 20 <= len(s) <= 30, (len(s), info)
    assert one_extremum_each(X, s, f, true_centers(P, (0.0, 0.5)))


def test_auto_noiseless_curves():
    y = 100 * np.sin(2 * np.pi * X / 40.0)  # simulated data without noise: nothing to divide by
    assert auto_split(X, y)[3]["type"] == "no period"
    s, f, kinds, info = auto_split(X, 100 * np.sin(2 * np.pi * X / 0.5))
    assert abs(info["period"] - 0.5) < 1e-3 and len(s) > 90, info


def test_eclipse_mode_leaves_pulsators_alone():
    y = -200 * np.sin(2 * np.pi * X / 2.0) + rng.normal(0, 5, X.size)  # review: the gap once made this "eclipses"
    s, f, kinds, info = auto_split(X, y)
    assert abs(info["period"] - 2.0) < 1e-2 and not info["eclipses"] and len(s) >= 20, info
    assert one_extremum_each(X, s, f, true_centers(2.0, (0.25, 0.75)))
    y = -50 * np.sin(2 * np.pi * X / 3.0) + rng.normal(0, 3, X.size)
    y[np.searchsorted(X, 5.0) : np.searchsorted(X, 5.0) + 10] += 40  # one faint 20-min blip
    s, f, kinds, info = auto_split(X, y)
    assert abs(info["period"] - 3.0) < 2e-2 and kinds.count("max") >= 6, info
    y = 300 * np.sin(2 * np.pi * X / 5.0) + dip(X / 2.3 % 1, 0.0, 0.01, 60) + rng.normal(0, 3, X.size)
    assert not auto_split(X, y)[3]["eclipses"]  # dips shallower than the slow wave: not eclipse mode
    s, f, kinds, info = auto_split(X, -10 * np.sin(2 * np.pi * X / 10) + rng.normal(0, 3, X.size), period=10.0)
    assert info["type"] != "EA" and kinds.count("max") >= 1, info  # long period, low S/N: keeps its maxima


def test_eclipse_mode_near_gaps_and_edges():
    ph = X / 2.87 % 1  # 0.4-d eclipses, one cut by the mid-sector gap
    s, f, kinds, info = auto_split(X, dip(ph, 0.0, 0.07, 500) + dip(ph, 0.5, 0.07, 200) + rng.normal(0, 3, X.size))
    assert abs(info["period"] - 2.87) < 2e-3 and one_extremum_each(X, s, f, true_centers(2.87, (0.0, 0.5))), info
    x = np.arange(0, 27, 2 / 1440)  # an eclipse cut by the start of the data must not bias P
    s, f, kinds, info = auto_split(x, dip(x / 5 % 1, 0.0, 0.06, 500) + dip(x / 5 % 1, 0.5, 0.06, 200) + rng.normal(0, 3, x.size))
    assert abs(info["period"] - 5.0) < 2e-3, info
    x = np.sort(np.r_[X, X])  # the same sector twice: repeated time stamps
    s, f, kinds, info = auto_split(x, dip(x / 1.3 % 1, 0.0, 0.03, 300) + rng.normal(0, 3, x.size))
    assert abs(info["period"] - 1.3) < 2e-3, info


def test_auto_slow_low_amplitude_variable_keeps_its_extrema():
    y = -4 * np.sin(2 * np.pi * X / 5.0) + rng.normal(0, 5, X.size)  # review 2: 0 windows once
    s, f, kinds, info = auto_split(X, y)
    assert abs(info["period"] - 5.0) < 0.05 and len(s) >= 4, (len(s), info)


def test_eclipse_mode_dip_lost_at_a_gap_is_no_missed_cycle():
    x = np.arange(0, 27, 2 / 1440)
    x = x[~(((x > 9.94) & (x < 10.14)) | ((x > 14.405) & (x < 14.605)))]  # gaps ending 0.3 d before two eclipses
    P, t0 = 9.44, 1.0
    ph = (x - t0) / P % 1
    y = dip(ph, 0.0, 0.009, 60) + dip(ph, 0.473, 0.01, 35) + 5 * np.sin(2 * np.pi * x / 2.21) + rng.normal(0, 0.8, x.size)
    assert abs(auto_split(x, y)[3]["period"] - P) < 0.01


def test_eclipse_mode_no_doubling_on_small_depth_differences():
    P = 1.25  # one eclipse per cycle; depths alternate by 1.5 % (spots), which is no second eclipse
    ph = X / P % 1
    y = dip(ph, 0.0, 0.03, 106) * np.where(np.round(X / P) % 2 == 0, 1.0, 0.98) + rng.normal(0, 0.5, X.size)
    assert abs(auto_split(X, y)[3]["period"] - P) < 2e-3


def test_given_period_other_than_the_eclipses_turns_eclipse_mode_off():
    P, t0 = 9.44, 1.0  # to time the spot wave the user enters its period
    ph = (X - t0) / P % 1
    y = dip(ph, 0.0, 0.009, 60) + dip(ph, 0.473, 0.01, 35) + 5 * np.sin(2 * np.pi * X / 2.21) + rng.normal(0, 0.8, X.size)
    s, f, kinds, info = auto_split(X, y, period=2.21)
    assert not info["eclipses"] and len(s) >= 10, (len(s), info)


def test_eclipse_mode_one_group_is_one_kind_of_eclipse():
    x = np.arange(0, 20.3, 2 / 1440)  # TIC 140659980 sector 3: two secondaries and one usable primary
    x = x[(x < 8.72) | (x > 10.71)]
    P, t0 = 9.44, 17.86
    ph = (x - t0) / P % 1
    y = dip(ph, 0.0, 0.009, 60) + dip(ph, 0.4732, 0.01, 35) + 5 * np.sin(2 * np.pi * x / 2.21) + rng.normal(0, 0.8, x.size)
    assert abs(auto_split(x, y)[3]["period"] - P) < 0.02  # not 7.21: a primary and a secondary in one group


def spotted_ea(x, r=rng):
    P, t0 = 9.44, 1.0
    ph = (x - t0) / P % 1
    return dip(ph, 0.0, 0.009, 60) + dip(ph, 0.473, 0.01, 35) + 5 * np.sin(2 * np.pi * x / 2.21) + r.normal(0, 0.8, x.size)


def test_eclipse_mode_with_repeated_or_missing_points():
    x = np.sort(np.r_[X, X])  # every time stamp twice
    assert abs(auto_split(x, spotted_ea(x))[3]["period"] - 9.44) < 0.01
    x = X[np.random.default_rng(3).random(X.size) > 0.25]  # a quarter of the points missing at random
    s, f, kinds, info = auto_split(x, spotted_ea(x))
    assert info["eclipses"] and abs(info["period"] - 9.44) < 0.01, info


def test_eclipse_mode_keeps_a_given_multiple_of_the_period():
    y = spotted_ea(X)
    for k in (2, 0.5):
        s, f, kinds, info = auto_split(X, y, period=9.44 * k)
        assert info["eclipses"] and set(kinds) == {"min"}, (k, info)  # still the eclipses, not the spot wave


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
