"""Run the multifactor Euler comparison adapted from Serva, Chapter 3.

Use geometric kernel quadrature as in Section 2.1, with one constant factor
for the low-rate tail. Freeze the continuous contribution to the intensity
over each step and thin against its sum with the decaying jump contribution.
The Euler diffusion and drift use the coefficients of the continuous model
(Serva's equations (1.1)--(1.3)) with raw jump marks.
"""

import argparse
import csv
from math import exp, gamma, sqrt
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from marked_inar_smiles import (
    ALPHAS,
    DRIFT,
    IMMIGRATION,
    INITIAL,
    JUMP_LOAD,
    LOG_STRIKES,
    OUT,
    PRICING_STEPS,
    RHO,
    VAR_COEF,
    implied,
    smile_payoff_statistics,
)


PATHS = 2**20
STEPS = PRICING_STEPS[1]
FACTORS = 64
VARIANCE_FLOOR = 1e-5


def geometric_factors(alpha, count):
    log_ratio = np.log1p(10 * count**(-0.9))
    lower = np.exp(log_ratio * (np.arange(count) - count / 2))
    mass_ratio = np.expm1((1 - alpha) * log_ratio)
    normalization = gamma(alpha) * gamma(2 - alpha)
    weights = lower**(1 - alpha) * mass_ratio / normalization
    rates = lower * (1 - alpha) / (2 - alpha)
    rates *= np.expm1((2 - alpha) * log_ratio) / mass_ratio
    low_mass = lower[0]**(1 - alpha) / normalization
    return np.r_[low_mass, weights], np.r_[0.0, rates]


@njit
def thin_jumps(base, intensity, jump_factors, weights, rates, step):
    """Return total marks and advance the jump factors to the next node."""
    elapsed = 0.0
    marks = 0.0
    while True:
        waiting = np.random.exponential() / intensity
        if waiting >= step - elapsed:
            for j in range(len(rates)):
                jump_factors[j] *= exp(-rates[j] * (step - elapsed))
            return marks

        elapsed += waiting
        candidate_intensity = base
        for j in range(len(rates)):
            jump_factors[j] *= exp(-rates[j] * waiting)
            candidate_intensity += jump_factors[j]
        candidate_intensity = max(VARIANCE_FLOOR, candidate_intensity)
        if np.random.random() * intensity <= candidate_intensity:
            mark = np.random.exponential()
            marks += mark
            for j in range(len(rates)):
                jump_factors[j] += weights[j] * mark
            # If the floor was active, compute the new intensity from the raw sum.
            candidate_intensity = max(VARIANCE_FLOOR, base + np.sum(jump_factors))
        intensity = candidate_intensity


@njit(parallel=True)
def terminal_spots(initial_curve, weights, rates, paths, seed):
    step = 1 / len(initial_curve)
    decay = np.exp(-rates * step)
    independent_loading = sqrt(1 - RHO**2)
    price_drift = -0.5 + JUMP_LOAD / (1 + JUMP_LOAD)
    spots = np.empty(paths)
    floor_count = np.empty(paths, dtype=np.int64)
    batch_size = 1024
    for batch in prange((paths + batch_size - 1) // batch_size):
        np.random.seed(seed + batch)
        continuous_factors = np.empty(len(rates))
        jump_factors = np.empty(len(rates))
        for path in range(batch * batch_size, min(paths, (batch + 1) * batch_size)):
            continuous_factors[:] = 0.0
            jump_factors[:] = 0.0
            log_price = 0.0
            has_jumped = False
            floor_count[path] = 0
            for initial in initial_curve:
                base = initial + np.sum(continuous_factors)
                raw_variance = base
                if has_jumped:
                    raw_variance += np.sum(jump_factors)
                variance = max(VARIANCE_FLOOR, raw_variance)
                floor_count[path] += raw_variance < VARIANCE_FLOOR
                variance_step = variance * step
                variance_noise = np.random.standard_normal()
                price_noise = RHO * variance_noise + independent_loading * np.random.standard_normal()

                # Before the first jump the intensity is constant within a step.
                # Generate that proposal separately to avoid advancing zero factors.
                if has_jumped:
                    marks = thin_jumps(base, variance, jump_factors, weights, rates, step)
                else:
                    waiting = np.random.exponential() / variance
                    marks = 0.0
                    if waiting < step:
                        mark = np.random.exponential()
                        jump_factors[:] = weights * mark
                        after_jump = max(VARIANCE_FLOOR, base + np.sum(jump_factors))
                        marks = mark + thin_jumps(
                            base, after_jump, jump_factors, weights, rates, step - waiting
                        )
                        has_jumped = True

                log_price += price_drift * variance_step + sqrt(variance_step) * price_noise - JUMP_LOAD * marks
                forcing = (DRIFT - 1) * variance_step + sqrt(VAR_COEF * variance_step) * variance_noise
                for j in range(len(rates)):
                    continuous_factors[j] = decay[j] * continuous_factors[j] + weights[j] * forcing
            spots[path] = exp(log_price)
    return spots, floor_count


def estimate(alpha, steps, paths, seed):
    weights, rates = geometric_factors(alpha, FACTORS)
    grid = np.arange(steps) / steps
    initial_curve = INITIAL + IMMIGRATION * grid**alpha / gamma(1 + alpha)
    spots, floor_count = terminal_spots(initial_curve, weights, rates, paths, seed)
    if not np.all(np.isfinite(spots)):
        raise ArithmeticError("nonfinite terminal prices")
    call_prices, error, forward, forward_se = smile_payoff_statistics(spots)
    intervals = implied(call_prices), implied(call_prices - error), implied(call_prices + error)
    return intervals, forward, forward_se, floor_count.sum() / (paths * steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=PATHS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output", type=Path, default=OUT / "serva_smile_values.csv")
    args = parser.parse_args()
    if args.paths < 2 or args.steps < 1:
        parser.error("use at least two paths and one time step")
    set_num_threads(min(get_num_threads(), 8))
    rows = []
    for index, alpha in enumerate(ALPHAS):
        (iv, low, high), forward, forward_se, floor_rate = estimate(
            alpha, args.steps, args.paths, 982647 + index * 100000
        )
        print(f"alpha={alpha:.2f}, steps={args.steps}, forward={forward:.6f} "
              f"(SE {forward_se:.6f}), variance floor={floor_rate:.2%}", flush=True)
        rows.extend((alpha, strike, mid, lo, hi, args.steps, args.paths, forward, forward_se, floor_rate)
                    for strike, mid, lo, hi in zip(LOG_STRIKES, iv, low, high))
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["alpha", "log_strike", "iv", "low", "high", "steps", "paths",
                         "forward", "forward_se", "variance_floor_rate"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
