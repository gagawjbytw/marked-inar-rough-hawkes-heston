"""Generate the paper's vector smile figure from the transforms and saved Monte Carlo estimates."""

from pathlib import Path

import matplotlib
import numpy as np
from numba import get_num_threads, set_num_threads
from threadpoolctl import threadpool_limits

from marked_inar_smiles import (
    ALPHAS,
    LOG_STRIKES,
    PRICING_STEPS,
    call_prices,
    discrete_transform,
    factors,
    fourier_grid,
    fractional_transform,
    implied,
    lattice,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PILOT_DIR = Path(__file__).resolve().parent
FIGURE_PATH = PILOT_DIR.parent / "figures" / "smiles.pdf"
PLOT_STRIKES = np.linspace(-0.5, 0.5, 101)


def main():
    set_num_threads(min(get_num_threads(), 6))
    samples = np.genfromtxt(PILOT_DIR / "smile_values.csv", delimiter=",", names=True)
    serva_samples = np.genfromtxt(PILOT_DIR / "serva_smile_values.csv", delimiter=",", names=True)
    frequencies, weights = fourier_grid()
    arguments = 0.5 + 1j * frequencies
    reference_steps = 1024
    grid_bins = PRICING_STEPS[1]

    with threadpool_limits(limits=1, user_api="blas"), matplotlib.rc_context({
        "font.family": "STIXGeneral",
        "font.size": 9,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(6.45, 3.65), sharex=True, sharey=True)
        for ax, alpha in zip(axes.flat, ALPHAS):
            continuous = fractional_transform(
                arguments, reference_steps, 1.0, *factors(alpha, 1 / reference_steps)
            )
            discrete = discrete_transform(arguments, *lattice(alpha, grid_bins))
            continuous_prices = call_prices(continuous, frequencies, weights, log_strikes=PLOT_STRIKES)
            discrete_prices = call_prices(discrete, frequencies, weights, log_strikes=PLOT_STRIKES)
            continuous_iv = implied(continuous_prices, log_strikes=PLOT_STRIKES)
            discrete_iv = implied(discrete_prices, log_strikes=PLOT_STRIKES)

            sample = samples[samples["alpha"] == alpha]
            np.testing.assert_allclose(sample["log_strike"], LOG_STRIKES, atol=1e-12)
            estimate = sample[f"mc_{grid_bins}_iv"]
            errors = np.array([estimate - sample[f"mc_{grid_bins}_low"], sample[f"mc_{grid_bins}_high"] - estimate])
            serva = serva_samples[serva_samples["alpha"] == alpha]
            np.testing.assert_allclose(serva["log_strike"], LOG_STRIKES, atol=1e-12)
            np.testing.assert_array_equal(serva["steps"], np.full(len(LOG_STRIKES), grid_bins))
            serva_errors = np.array([serva["iv"] - serva["low"], serva["high"] - serva["iv"]])

            ax.plot(PLOT_STRIKES, 100 * continuous_iv, color="black", lw=1, label="Affine limit")
            ax.plot(PLOT_STRIKES, 100 * discrete_iv, "--", color="#a0462a", lw=1, label="INAR transform")
            ax.errorbar(
                serva["log_strike"], 100 * serva["iv"], yerr=100 * serva_errors, fmt="s",
                color="#5147a6", markerfacecolor="none", markersize=3.7, markeredgewidth=0.85,
                elinewidth=0.7, capsize=1.25, label="Serva Euler (adapted)",
            )
            ax.errorbar(
                sample["log_strike"], 100 * estimate, yerr=100 * errors, fmt="o",
                color="#00756e", markersize=2.2, elinewidth=0.7, capsize=1.25,
                label="INAR simulation",
            )
            ax.set_title(rf"$\alpha={alpha:.2f}$", pad=3)
            ax.set(xlim=(-0.53, 0.53), ylim=(7.5, 26.5), xticks=(-0.4, -0.2, 0, 0.2, 0.4),
                   yticks=(10, 15, 20, 25))
            ax.grid(color="#dedede", linewidth=0.45)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

        for ax in axes[-1]:
            ax.set_xlabel(r"$\log(K/F)$")
        for ax in axes[:, 0]:
            ax.set_ylabel("Implied volatility (%)")
        fig.subplots_adjust(left=0.095, right=0.987, top=0.943, bottom=0.205, wspace=0.14, hspace=0.33)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=4, frameon=False, fontsize=8.5, columnspacing=1.25, handletextpad=0.5)
        FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_PATH)
        plt.close(fig)


if __name__ == "__main__":
    main()
