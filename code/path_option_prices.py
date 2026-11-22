"""Price arithmetic Asian and down-and-out puts with 200,000 paths per mesh.

Run with NumPy, SciPy, Numba and the dependencies of marked_inar_smiles.py.
Direct convolution accounts for lags within each leaf. An FFT transfers each
completed left block's contribution to the following right block. Together the
two sums include every lag of the grid kernel. The fine-grid reference is in
path_option_prices_200k.csv. Use --timing-runs 3 to reproduce the paper's
grid-refinement experiment and its median wall-clock timings.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from numba import njit
from scipy.fft import rfft, irfft

from marked_inar_smiles import JUMP_LOAD, PRICING_STEPS, lattice

ALPHAS = (0.62, 0.80)
STEPS = PRICING_STEPS
PATHS = 200_000
STRIKE, BARRIER = 1.0, 0.85
CONTRACTS = ("asian", "down_out")
LEAF = 64
BATCH = 64
WORKERS = 10
DIRECTORY = Path(__file__).resolve().parent


@njit(nogil=True)
def simulate_leaf(start, stop, lag, activity, history, state, coefficients, rng):
    rp, rm, ap, am, jump_weight, up, down, drift, h = coefficients
    for path in range(len(state)):
        log_spot, spot, minimum, total = state[path]
        for n in range(start, stop):
            variance = activity[path, n]
            for j in range(start, n):
                variance += lag[n-j] * history[path, j]
            plus = rng.poisson(rp * variance)
            minus = rng.poisson(rm * variance)
            count = rng.poisson(h * variance)
            marks = rng.standard_gamma(count) if count else 0.0
            history[path, n] = ap * plus + am * minus + jump_weight * marks
            total += spot
            log_spot += up * plus + down * minus - JUMP_LOAD * marks - drift * variance
            spot = np.exp(log_spot)
            minimum = min(minimum, spot)
        state[path] = log_spot, spot, minimum, total


def kernel_spectra(lag, steps):
    spectra = {}
    width = LEAF
    while width < steps:
        spectra[width] = rfft(lag[:2 * width], n=2 * width)
        width *= 2
    return spectra


def add_left_history(stop, width, history, activity, spectrum):
    # Wraparound of a length-2w circular convolution affects only its left half.
    data = rfft(history[:, stop - width:stop], n=2 * width, axis=1)
    data *= spectrum
    right_size = min(width, activity.shape[1] - stop)
    activity[:, stop:stop + right_size] += irfft(data, axis=1)[:, width:width + right_size]


def simulate_batch(batch_id, *, alpha, steps, paths, lag, baseline, coefficients, spectra):
    size = min(BATCH, paths - BATCH * batch_id)
    seed = np.random.SeedSequence([76143, round(100 * alpha), steps, batch_id])
    rng = np.random.default_rng(seed)
    history = np.empty((size, steps))
    activity = np.tile(baseline, (size, 1))
    state = np.zeros((size, 4))
    state[:, 1:3] = 1.0
    for start in range(0, steps, LEAF):
        stop = min(start + LEAF, steps)
        simulate_leaf(start, stop, lag, activity, history, state, coefficients, rng)
        if stop < steps:
            # The lowest set bit gives the size of the completed left child.
            width = stop & -stop
            add_left_history(stop, width, history, activity, spectra[width])
    spot, minimum, average = state[:, 1], state[:, 2], state[:, 3] / steps
    puts = np.column_stack((np.maximum(STRIKE - average, 0),
                            np.maximum(STRIKE - spot, 0) * (minimum > BARRIER)))
    return puts.sum(axis=0), (puts * puts).sum(axis=0)


def estimate(alpha, steps, paths=PATHS, workers=WORKERS):
    if steps <= 0 or paths < 2:
        raise ValueError("the grid requires at least one bin and two Monte Carlo paths")
    phi, baseline, *coefficients = lattice(alpha, steps)
    lag = np.r_[0.0, phi]
    simulate = partial(simulate_batch, alpha=alpha, steps=steps, paths=paths,
                       lag=lag, baseline=baseline, coefficients=tuple(coefficients),
                       spectra=kernel_spectra(lag, steps))
    batches = (paths + BATCH - 1) // BATCH
    with ThreadPoolExecutor(max_workers=workers) as pool:
        moments = np.sum(list(pool.map(simulate, range(batches))), axis=0)
    means = moments[0] / paths
    errors = np.sqrt((moments[1] / paths - means**2) / (paths - 1))
    row = dict(alpha=alpha, bins=steps, paths=paths)
    for name, mean, error in zip(CONTRACTS, means, errors):
        row[name] = mean
        row[name + "_se"] = error
    return row


def save_prices(path, rows):
    stream = NamedTemporaryFile(mode="w", newline="", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["alpha"], row["bins"])))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, nargs="+", default=STEPS,
                        help="grid sizes (default: %(default)s)")
    parser.add_argument("--timing-runs", type=int, default=1,
                        help="number of runs with identical seeds for the median wall time")
    parser.add_argument("--output", type=Path, help="destination CSV for this run")
    args = parser.parse_args()
    if args.timing_runs < 1 or min(args.bins) < 1:
        parser.error("timing runs and grid sizes must be positive")
    estimate(ALPHAS[0], LEAF, paths=2, workers=1)  # Compile the simulation before timing.
    rows = []
    for steps in args.bins:
        for alpha in ALPHAS:
            elapsed = []
            for _ in range(args.timing_runs):
                start = perf_counter()
                row = estimate(alpha, steps)
                elapsed.append(perf_counter() - start)
            row["timing_runs"] = args.timing_runs
            row["wall_seconds"] = median(elapsed)
            print(row, flush=True)
            rows.append(row)
    grid_label = "_".join(map(str, args.bins))
    save_prices(args.output or DIRECTORY / f"path_option_prices_{grid_label}.csv", rows)


if __name__ == "__main__":
    main()
