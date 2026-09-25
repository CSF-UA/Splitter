# Splitter

Splits the light curve of a periodic variable star into intervals that each contain one
extremum, for O−C analysis. The intervals are then fitted in astrolab `approximation` or MAVKA.

## Run

```
uv sync
uv run main.py                         # GUI
uv run main.py --batch data/*.tess     # no GUI: Auto on every file
uv run main.py --batch --period 6.29 star/*/*.tess   # same, with a known period
uv run python tests/test_core.py       # self-checks
```

## Input and output

- Input: a text file with two columns, time (JD/BJD) and magnitude (mmag). Lines starting with `#` are comments.
- Output: `<name>_intervals.txt` with one line per interval, `start end kind`.
  - `start` and `end` are 0-based indices of the first and last point (inclusive).
  - `kind` is `min` (brightness minimum) or `max`.
  - Readers that take only the first two columns keep working.

## Algorithms

- **Auto** (default):
  - Eclipses on slower variability (spots, pulsations, trends): if the curve has narrow dips that
    only go fainter, Auto removes the slower variability (a 0.5-d running median with the dips
    masked) and works on the rest. The period then comes from the dip times: the shortest one that
    puts the dips into one or two phase groups seen in most cycles, doubled if alternate dips
    differ in depth or timing, refined by PDM. Windows go only to the eclipses, not to spot waves.
    The info panel says "Eclipse mode: P from the eclipses". A P you enter that differs from it
    turns eclipse mode off (e.g. to time the spot wave instead).
  - Otherwise it finds the period unless P is entered: an FFT periodogram of the light curve averaged onto a
    regular time grid, refined by PDM, then multiplied by 2 or 3 while that folds better
    (harmonics of narrow eclipses, two different minima).
  - If there is no reliable period (longer than about half the data span, or no signal), Auto says
    so and makes no windows: enter P by hand. It also cannot find periods shorter than 0.05 d or
    shorter than about twice the typical sampling step.
  - Builds the phase template and cuts a window around every minimum and maximum in every cycle.
  - Drops windows with a gap, too few points or a missing branch.
  - Parameters:
    - *Frac (0=auto)*: window edge as a fraction of the depth towards the neighbouring opposite
      extremum; 0 means 0.95, which gave the smallest O−C scatter on real TESS stars. Lower it
      if the windows are too wide for your fitting method.
    - *Min Points*: default 15. Lower it for 30-min FFI data with short eclipses.
    - *Minima only*.
- **GB-AT / M-inverted GB-AT**: magnitude threshold for minima / maxima.
- **S-DIPS**: multi-scale smoothing and sign changes of the second derivative. Needs P.

## GUI

- **Layers**: a later layer skips points used by the earlier ones.
- **Intervals**: click to select, drag the borders to resize, `D` to delete, `S` to save, `R` to reset the view.
- **Period**: `P [+]` measures the period with two clicks, `Esc` exits.
