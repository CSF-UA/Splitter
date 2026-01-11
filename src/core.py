import numpy as np


def splitting_algol_configurable(
    x,
    y,
    index_gap: int = 2,
    cut_ratio: float = 0.25,
    min_interval_points: int = 5,
    excluded_indices: set = None,
    is_inverted: bool = False,
    fill_remaining: bool = False,
):
    """Threshold splitting for Algol-type eclipsing binaries (EA)."""
    if len(y) == 0:
        return [], []

    base_mag = np.percentile(y, 5)
    deep_mag = np.percentile(y, 99.5)
    amplitude = deep_mag - base_mag
    if amplitude <= 0:
        return [], []

    if is_inverted:
        cutoff_level = base_mag + (amplitude * (1 - cut_ratio))
        indices = np.where(y < cutoff_level)[0]
    else:
        cutoff_level = base_mag + (amplitude * cut_ratio)
        indices = np.where(y > cutoff_level)[0]
    if len(indices) == 0:
        return [], []

    # Exclude indices that were already processed in previous layers
    if excluded_indices is not None:
        mask = np.array([i not in excluded_indices for i in indices])
        indices = indices[mask]
        if len(indices) == 0:
            return [], []

    start = []
    finish = []
    current_start = int(indices[0])
    for i in range(1, len(indices)):
        if indices[i] > indices[i - 1] + index_gap:
            start.append(current_start)
            finish.append(int(indices[i - 1]))
            current_start = int(indices[i])
    start.append(current_start)
    finish.append(int(indices[-1]))

    filtered_start = []
    filtered_finish = []
    for s, f in zip(start, finish):
        points_in_interval = np.sum((indices >= s) & (indices <= f))
        if points_in_interval >= min_interval_points:
            filtered_start.append(s)
            filtered_finish.append(f)

    start, finish = filtered_start, filtered_finish

    final_start = list(start)
    final_finish = list(finish)

    # Fill remaining points if requested
    if fill_remaining and len(final_start) > 0:
        # Create a set of all covered indices
        covered_indices = set()
        for s, f in zip(final_start, final_finish):
            covered_indices.update(range(s, f + 1))

        # Add excluded indices to covered set
        if excluded_indices is not None:
            covered_indices.update(excluded_indices)

        # Find gaps (consecutive uncovered indices)
        all_indices = set(range(len(x)))
        uncovered_indices = sorted(list(all_indices - covered_indices))

        if len(uncovered_indices) > 0:
            # Group consecutive uncovered indices into intervals
            gap_start = [uncovered_indices[0]]
            gap_finish = []

            for i in range(1, len(uncovered_indices)):
                if uncovered_indices[i] != uncovered_indices[i - 1] + 1:
                    gap_finish.append(uncovered_indices[i - 1])
                    gap_start.append(uncovered_indices[i])
            gap_finish.append(uncovered_indices[-1])

            # Add gap intervals to the result
            final_start.extend(gap_start)
            final_finish.extend(gap_finish)

    return final_start, final_finish


# ==============================================================================
# I. Optimized Data Loading
# ==============================================================================


def get_data(name: str, mmin: float = -1000, mmax: float = 1000):
    """
    Read two-column text file: JD mag.
    Optimized to use np.loadtxt where possible for C-speed parsing.
    """
    try:
        # Attempt fast C-based loading first
        # comments='#' handles standard headers
        data = np.loadtxt(name, comments="#")
    except (FileNotFoundError, OSError):
        raise FileNotFoundError(f"File not found: {name}")
    except ValueError:
        # Fallback to python parsing if lines are ragged or messy
        with open(name, "r", encoding="utf-8", errors="ignore") as f:
            S = f.readlines()
        JD, mag = [], []
        for line in S:
            a = line.split()
            if len(a) < 2:
                continue
            try:
                x, m = float(a), float(a)
            except ValueError:
                continue
            if mmin <= m <= mmax:
                JD.append(x)
                mag.append(m)
        return np.array(JD, dtype=float), np.array(mag, dtype=float)

    # If loadtxt succeeded, filter and return
    if data.ndim != 2 or data.shape[1] < 2:
        return np.array([]), np.array([])

    x = data[:, 0]
    m = data[:, 1]

    # Vectorized boolean masking is faster than appending to lists
    mask = (m >= mmin) & (m <= mmax)
    return x[mask], m[mask]


# ==============================================================================
# II. Vectorized Smoothing Kernel
# ==============================================================================


def vectorized_sm(y: np.ndarray, N: int) -> np.ndarray:
    """
    Computes moving average using vectorized convolution.
    Replicates the exact edge behavior of the original 'sm' function:
    - Window size = 2*N + 1
    - The first N and last N pixels are left UNCHANGED (raw data).

    Performance: ~100x faster than looping.
    """
    if N <= 0:
        return y.copy()

    window_size = 2 * N + 1
    if len(y) < window_size:
        return y.copy()

    # Create normalized boxcar kernel
    kernel = np.ones(window_size) / window_size

    # 'valid' mode returns only the fully overlapped convolution
    # Output length = L - (2N+1) + 1 = L - 2N
    convolved = np.convolve(y, kernel, mode="valid")

    Y = y.copy()
    # Fill the center with the smoothed signal
    # Indices [N : len(y)-N] have length L-2N, matching 'convolved'
    Y[N : len(y) - N] = convolved
    return Y


def NN(T0: float, alpha: float):
    """
    Generates sequence of window sizes for the smoothing cascade.
    Exponential growth mimics a Gaussian filter response.
    """
    N = []
    Nmax = alpha * T0 * 584
    n = 2
    nn = 0
    while nn < Nmax:
        nn = int(0.7734 * np.exp(0.4484 * n))
        N.append(nn)
        n += 1
    return N


def smooth(T0: float, alpha: float, y):
    """
    Applies the cascading smooth using the vectorized kernel.
    """
    N_vals = NN(T0, alpha)
    Y = y.copy()

    # Forward pass: iteratively smooth with increasing windows
    for n_val in N_vals:
        Y = vectorized_sm(Y, n_val)

    # Backward pass: symmetric smoothing to cancel phase shift
    for n_val in reversed(N_vals):
        Y = vectorized_sm(Y, n_val)

    return Y, (max(N_vals) if len(N_vals) > 0 else 0)


# ==============================================================================
# III. Optimized Geometric Fitting (Linear Algebra)
# ==============================================================================


def fast_parabolic_approximation(xxx: np.ndarray, y: np.ndarray):
    """
    Fits a parabola y = Ax^2 + Bx + C using deterministic Linear Least Squares.
    Replaces the iterative, random-restart Levenberg-Marquardt solver.

    Returns:
       P: Coefficients
       x_centered: The x-axis shifted to mean=0 (for numerical stability)
    """
    if len(xxx) == 0:
        return [0.0, 0.0, 0.0], xxx

    # Center x to avoid large number squaring (e.g. JD^2)
    x_mean = np.mean(xxx)
    x_centered = xxx - x_mean

    try:
        # np.polyfit uses SVD via LAPACK. Robust and fast.
        P = np.polyfit(x_centered, y, 2)
    except np.linalg.LinAlgError:
        return [0.0, 0.0, 0.0], x_centered

    return P, x_centered


def if_extr_is_inside_interval(xxx, y):
    """
    Checks if the parabola vertex lies strictly inside the data interval.
    Uses the fast fit, removing the need for 10 random restarts.
    """
    par, x = fast_parabolic_approximation(xxx, y)

    # If A (curvature) is near 0, it's a line -> no extremum.
    if abs(par[0]) < 1e-9:
        return False

    # Vertex formula: x = -B / 2A
    X_vertex = -0.5 * par[1] / par[0]

    # Check bounds. Note: x[0], x[-1], and X_vertex are all in the
    # centered coordinate system, so the comparison is valid.
    return (X_vertex > x[0]) and (X_vertex < x[-1])


def check_up(x, y, start, finish, T0: float):
    """
    Vectorized/Fast wrapper for the check_up logic.
    Validates each detected interval.
    """
    valid_start = []
    valid_finish = []

    # This loop runs over *intervals* (rare), not pixels (dense),
    # so keeping it as a Python loop is acceptable.
    for s, f in zip(start, finish):
        # Basic index sanity checks
        if s >= f or s < 0 or f >= len(x):
            continue

        xx = x[s : f + 1]
        yy = y[s : f + 1]

        # Original ckeck_and_plot logic inline
        if len(xx) < 5:
            continue

        duration = xx[-1] - xx[0]
        if (duration < 0.003 * T0) or (duration > T0):
            continue

        # Use the fast parabolic check
        if if_extr_is_inside_interval(xx, yy):
            valid_start.append(int(s))
            valid_finish.append(int(f))

    return valid_start, valid_finish


# ==============================================================================
# IV. The Main Logic: Vectorized Feature Extraction
# ==============================================================================


def splitting_normal(
    x, y, T0: float, alpha: float = 0.12, excluded_indices: set = None
):
    """
    Fully vectorized implementation of the segmentation algorithm.
    Replaces list comprehensions and scalar loops with NumPy array operations.
    """
    # 1. Smooth the signal
    yy, Nmax = smooth(T0, alpha, y)

    if len(x) < 3:
        return [], []

    # 2. First Derivative (Vectorized)
    # d = dy / dx
    dx = x[1:] - x[:-1]
    dy = yy[1:] - yy[:-1]

    # Safe division handling
    with np.errstate(divide="ignore", invalid="ignore"):
        d = dy / dx
        d[~np.isfinite(d)] = 0.0

    # Smooth First Derivative
    # Using the sequence from original code: 3, 5, 9, 13, 9, 5, 3
    for N in [3, 5, 9, 13, 9, 5, 3]:
        d = vectorized_sm(d, N)

    # 3. Second Derivative (Vectorized)
    # The original code calculated dd[i] using d[i+1]-d[i] and x[i+2]-x[i+1].
    # This implies using the 'next' dx interval for scaling.

    d_diff = d[1:] - d[:-1]
    dx_shifted = dx[1:]  # Corresponds to x[i+2] - x[i+1] in original indexing

    with np.errstate(divide="ignore", invalid="ignore"):
        dd = d_diff / dx_shifted
        dd[~np.isfinite(dd)] = 0.0

    # Smooth Second Derivative
    for N in [3, 5, 9, 13, 9, 5, 3]:
        dd = vectorized_sm(dd, N)

    # 4. Interval Search (Vectorized Logic)
    # We must scan indices from start_idx to end_idx - 1.
    start_idx = Nmax + 1
    end_idx = len(x) - Nmax - 1

    if start_idx >= end_idx:
        return [], []

    # Generate the indices we need to test
    # These indices 'i' correspond to the loop counter in the original code.
    scan_indices = np.arange(start_idx, end_idx - 1)

    # Ensure indices don't go out of bounds of our dd array
    # dd has length len(x)-2.
    valid_mask = (
        (scan_indices < len(dd))
        & ((scan_indices - 1) < len(dd))
        & (scan_indices < len(dx))
    )
    scan_indices = scan_indices[valid_mask]

    if len(scan_indices) == 0:
        return [], []

    # Condition A: Gaps
    # Original: if x[i+1] - x[i] > T0 * 0.5
    # This corresponds to checking dx[i]
    gaps = dx[scan_indices] > (T0 * 0.5)

    # Condition B: Zero Crossing (Inflection Points)
    # Original: if dd[i] * dd[i-1] < 0
    # We compare dd at the current index vs the previous index
    crossings = (dd[scan_indices] * dd[scan_indices - 1]) < 0

    # Combined Breakpoints
    # We split if either a gap or a crossing occurs
    breakpoints = gaps | crossings

    # Extract the indices where breakpoints happened
    split_indices = scan_indices[breakpoints]

    # Construct the result lists
    intervals_start = [start_idx]
    intervals_finish = []

    if len(split_indices) > 0:
        intervals_finish.extend(split_indices)
        intervals_start.extend(split_indices + 1)

    intervals_finish.append(end_idx)

    # Filter out intervals that contain excluded indices
    if excluded_indices is not None:
        filtered_start = []
        filtered_finish = []
        for s, f in zip(intervals_start, intervals_finish):
            # Check if this interval contains any excluded indices
            interval_indices = set(range(s, f + 1))
            if not interval_indices.intersection(excluded_indices):
                filtered_start.append(s)
                filtered_finish.append(f)
        intervals_start, intervals_finish = filtered_start, filtered_finish

    return intervals_start, intervals_finish


def save_data(start, finish, x, fname: str):
    """Helper to save results (unchanged logic, just type hints)."""
    fname_out = fname.replace(".tess", ".da!")
    if fname_out == fname:
        fname_out = fname + ".da!"

    # Using list comprehension and join is faster than string concatenation in a loop
    lines = [f"{s} {x[s]} {f} {x[f]}" for s, f in zip(start, finish)]
    data = "\n".join(lines) + "\n"

    with open(fname_out, "w", encoding="utf-8") as f:
        f.write(data)
    return fname_out
