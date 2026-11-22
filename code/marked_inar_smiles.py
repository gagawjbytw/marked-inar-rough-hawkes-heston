"""Compute smile_values.csv for the one-year marked-INAR smile experiment.

Run plot_smiles.py with the saved values to generate figures/smiles.pdf.
Dependencies: numpy, scipy, numba, threadpoolctl (CPU).
Prices are in forward units.
The marked grid and its rough Hawkes--Heston limit are defined in main.tex.

At each step, conditional on the past, regular Poisson counts N+, N- have means
V*l*w+, V*l*w-. A separate Poisson(h*V) count gives independent Exp(1) marks
with total Z. With Y=A+*N+ + A-*N- + B*Z, the predictable variance proxy
is V[n]=H[n]+sum_{k=1}^n phi[k]*Y[n-k], using zero prehistory and indices n>=0.
The log return is q*tick*N+ - tick*N- - jump_load*Z - drift*V, with the exact
conditional exponential compensator. The weights phi are from SIFIN Def. 2.1.

The fractional reference uses the Bondi-Pulido-Scotti log-price Riccati function,
a positive approximation of the fractional kernel by exponentials, linear
product integration in time, and Lewis inversion on Re(w)=1/2.
"""

import csv
from math import gamma, pi
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
from scipy.optimize import brentq
from scipy.special import ndtr, roots_legendre


PRICING_STEPS = (128, 512, 2048)
ALPHAS = (0.55, 0.62, 0.80, 0.95)
LOG_STRIKES = np.linspace(-0.5, 0.5, 11)
RHO, DRIFT, VAR_COEF, JUMP_LOAD = -0.731, -1.812, 0.115, 0.276
IMMIGRATION, INITIAL = 0.049, 0.0079
ASYMMETRY = 8.0
OUT = Path(__file__).resolve().parent


def lattice(alpha, steps):
    h = 1 / steps
    beta = ASYMMETRY
    def correlation(q):
        return -(beta-1)*np.sqrt(q/((1+q)*(1+beta**2*q)))
    q = brentq(lambda q: correlation(q)-RHO, 0, 1/beta)
    wp, wm = 2/(1+q), 2*q/(1+q)
    normalizer = wp + beta*wm
    ell = (wp + beta**2*wm)/normalizer**2 * h**(1-2*alpha)/VAR_COEF
    # With unit variance jump loading and Exp(1) marks, the drift before
    # jump compensation is b-1. The finite mean reproduction is a*(1+h**alpha).
    a = 1/(1+(1-DRIFT)*h**alpha)
    assert a*(1+h**alpha) < 1
    tail = np.ones(steps)
    tail[1:] = np.arange(1, steps, dtype=float)**(-alpha)/gamma(1-alpha)
    phi = -np.diff(tail)
    baseline = a*(INITIAL*tail + IMMIGRATION*h**alpha)
    tick = np.sqrt(h/(ell*(q*q*wp+wm)))
    up, down = q*tick, -tick
    rp, rm = ell*wp, ell*wm
    drift = rp*np.expm1(up)+rm*np.expm1(down)-h*JUMP_LOAD/(1+JUMP_LOAD)
    aplus = a/(ell*normalizer)
    return phi, baseline, rp, rm, aplus, beta*aplus, a*h**(alpha-1), up, down, drift, h


@njit(parallel=True)
def discrete_transform(w_array, phi, baseline, rp, rm, ap, am, bj, up, down, drift, h):
    steps = len(baseline)
    out = np.empty_like(w_array)
    for idx in prange(len(w_array)):
        w = w_array[idx]
        cumulant = np.empty(steps, dtype=np.complex128)
        total = 0j
        for n in range(steps):
            future = 0j
            for k in range(n):
                future += phi[k]*cumulant[n-1-k]
            mark = bj*future-JUMP_LOAD*w
            cumulant[n] = (rp*np.expm1(up*w+ap*future) + rm*np.expm1(down*w+am*future)
                           + h*mark/(1-mark) - w*drift)
            total += baseline[steps-1-n]*cumulant[n]
        out[idx] = np.exp(total)
    return out


def factors(alpha, dt, order=4):
    # K(t) = sin(pi*alpha)/pi * integral exp(-x*t) x**(-alpha) dx.
    edges = np.linspace(-25, 38, 33)
    nodes, weights = roots_legendre(order)
    mids, halves = (edges[1:]+edges[:-1])/2, np.diff(edges)/2
    logx = mids[:, None]+halves[:, None]*nodes
    normalizer = np.sin(pi*alpha)/pi
    mass = normalizer*halves[:, None]*weights*np.exp((1-alpha)*logx)
    low_mass = normalizer*np.exp((1-alpha)*edges[0])/(1-alpha)
    mass = np.r_[low_mass, mass.ravel()]
    y = dt*np.r_[0.0, np.exp(logx.ravel())]
    decay = np.exp(-y)
    small = y < 1e-3
    old, new = np.empty_like(y), np.empty_like(y)
    ys = y[small]
    old[small] = 0.5-ys/3+ys**2/8-ys**3/30
    new[small] = 0.5-ys/6+ys**2/24-ys**3/120
    yl = y[~small]
    old[~small] = (-np.expm1(-yl)-yl*decay[~small])/yl**2
    new[~small] = (yl+np.expm1(-yl))/yl**2
    return decay, dt*mass*old, dt*mass*new


@njit(parallel=True)
def fractional_transform(w_array, steps, maturity, decay, old, new):
    out = np.empty_like(w_array)
    dt = maturity/steps
    for idx in prange(len(w_array)):
        w = w_array[idx]
        # R(w,v) = constant + linear*v + c*v**2/2 + 1/(1+Lambda*w-v).
        constant = (w*w-w)/2 + w*JUMP_LOAD/(1+JUMP_LOAD)-1
        linear = DRIFT + RHO*np.sqrt(VAR_COEF)*w - 1
        denominator = 1+JUMP_LOAD*w
        v = 0j
        forcing = constant+1/denominator
        factor_state = np.zeros(len(decay), dtype=np.complex128)
        implicit_weight = np.sum(new)
        integral = 0j
        for n in range(steps):
            previous_v, previous_forcing = v, forcing
            history = 0j
            for j in range(len(decay)):
                factor_state[j] = decay[j]*factor_state[j]+old[j]*previous_forcing
                history += factor_state[j]
            for iteration in range(40):
                inverse = 1/(denominator-v)
                forcing = constant + linear*v + VAR_COEF*v*v/2 + inverse
                derivative = linear+VAR_COEF*v+inverse*inverse
                change = (v-history-implicit_weight*forcing)/(1-implicit_weight*derivative)
                v -= change
                if abs(change) < 1e-12*(1+abs(v)):
                    break
            else:
                v = complex(np.nan, np.nan)
            forcing = constant + linear*v + VAR_COEF*v*v/2 + 1/(denominator-v)
            for j in range(len(decay)):
                factor_state[j] += new[j]*forcing
            # For G0(t)=v0+immigration*I^alpha(1), log transform equals
            # integral [v0*R(w,psi) + immigration*psi] by stochastic-model affinity
            # and associativity of the deterministic fractional integral.
            integral += (INITIAL*(previous_forcing+forcing)+IMMIGRATION*(previous_v+v))*dt/2
        out[idx] = np.exp(integral)
    return out


def fourier_grid(upper=480, width=4, order=16):
    nodes, weights = roots_legendre(order)
    # Resolve the Lewis denominator u^2+1/4 especially accurately near zero.
    near_nodes, near_weights = roots_legendre(2*order)
    u = (np.arange(width, upper, width)[:, None]+width*(nodes+1)/2).ravel()
    return (np.r_[width*(near_nodes+1)/2, u],
            width/2*np.r_[near_weights, np.tile(weights, upper//width-1)])


def call_prices(cf, u, weights, *, log_strikes=LOG_STRIKES):
    k = np.asarray(log_strikes)
    integrand = np.real(cf[:, None]*np.exp(-1j*np.outer(u,k)))
    return 1-np.exp(k/2)/pi*((weights/(u*u+0.25))@integrand)


def call_iv(call, log_strike):
    k, strike = log_strike, np.exp(log_strike)
    intrinsic = max(0, 1-strike)
    if call <= intrinsic:
        return 0.0
    def residual(vol):
        d1 = -k/vol+vol/2
        return ndtr(d1)-strike*ndtr(d1-vol)-call
    return brentq(residual, 1e-6, 5)


def implied(prices, *, log_strikes=LOG_STRIKES):
    return np.array([call_iv(c, k) for c, k in zip(prices, log_strikes)])


def log_price_steps(params, size, rng):
    """Yield the updated log price and mark count at each right endpoint."""
    phi, baseline, rp, rm, ap, am, bj, up, down, drift, h = params
    reversed_phi = phi[::-1].copy()
    history = np.empty((len(baseline), size))
    log_spot = np.zeros(size)
    for n, base in enumerate(baseline):
        variance = np.full(size, base)
        if n:
            variance += reversed_phi[-n:] @ history[:n]
        plus, minus = rng.poisson(rp * variance), rng.poisson(rm * variance)
        count = rng.poisson(h * variance)
        marks = rng.standard_gamma(count)
        history[n] = ap * plus + am * minus + bj * marks
        log_spot += up * plus + down * minus - JUMP_LOAD * marks - drift * variance
        yield log_spot, count


def smile_payoff_statistics(samples):
    """Return call-equivalent prices, 95% half widths and forward diagnostics."""
    strikes = np.exp(LOG_STRIKES)
    put = LOG_STRIKES < 0
    payoffs = np.maximum((samples[:, None]-strikes)*np.where(put,-1,1), 0)
    calls = payoffs.mean(axis=0)+np.where(put, 1-strikes, 0)
    root_paths = np.sqrt(len(samples))
    err = 1.96*payoffs.std(axis=0, ddof=1)/root_paths
    return calls, err, samples.mean(), samples.std(ddof=1)/root_paths


def monte_carlo(params, paths, seed):
    rng = np.random.default_rng(seed)
    samples = np.empty(paths)
    for start in range(0, paths, 2048):
        size = min(paths - start, 2048)
        for log_spot, _ in log_price_steps(params, size, rng):
            pass
        samples[start:start + size] = np.exp(log_spot)
    return smile_payoff_statistics(samples)


def main():
    set_num_threads(min(get_num_threads(), 6))
    from threadpoolctl import threadpool_limits
    u, weights = fourier_grid()
    w = np.ascontiguousarray(0.5+1j*u)
    paths = 1048576
    mc_steps = PRICING_STEPS[1]
    rows = []
    for idx, alpha in enumerate(ALPHAS):
        ref = []
        for steps, order in [(512,3),(1024,4)]:
            cf = fractional_transform(w, steps, 1., *factors(alpha, 1/steps, order))
            assert np.all(np.isfinite(cf))
            ref.append(implied(call_prices(cf,u,weights)))
        ivs = {}
        for steps in PRICING_STEPS:
            params = lattice(alpha, steps)
            cf = discrete_transform(w, *params)
            assert np.all(np.isfinite(cf))
            ivs[steps] = implied(call_prices(cf,u,weights))
            # Transform recursions must preserve total probability and the forward.
            assert np.allclose(discrete_transform(np.array([0j,1+0j]),*params),1,atol=2e-11,rtol=0)
        with threadpool_limits(limits=1, user_api="blas"):
            mc, err, forward, forward_se = monte_carlo(lattice(alpha,mc_steps),paths,76143+idx)
        mciv, lo, hi = implied(mc), implied(mc-err), implied(mc+err)
        print(f"alpha={alpha:.2f}, forward={forward:.6f} (SE {forward_se:.6f}), "
              f"reference refinement={np.max(abs(ref[1]-ref[0])):.7f}, "
              f"max lattice errors={[round(max(abs(ivs[n]-ref[1])),6) for n in ivs]}",flush=True)
        for j,k in enumerate(LOG_STRIKES):
            rows.append([alpha,k,ref[1][j],*(ivs[n][j] for n in PRICING_STEPS),
                         mciv[j],lo[j],hi[j],ref[0][j]])
    with (OUT/"smile_values.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["alpha","log_strike","affine_iv",*(f"inar_{n}_iv" for n in PRICING_STEPS),
                         f"mc_{mc_steps}_iv",f"mc_{mc_steps}_low",f"mc_{mc_steps}_high","affine_coarser_iv"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
