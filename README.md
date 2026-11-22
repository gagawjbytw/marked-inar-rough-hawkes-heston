# Marked INAR model and the rough Hawkes–Heston limit

Python code and numerical results for the paper of the same title by Xiaoyu Wang,
Yingli Wang, and Lingjiong Zhu. The experiments cover one-year implied-volatility
smiles, arithmetic Asian puts, and down-and-out puts.

## Setup

Run the following commands from the repository root. The dependency versions
were checked with Python 3.12.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Implied-volatility smiles

The two supplied smile CSVs contain the estimates used in the paper. To draw the
figure from those estimates and the recomputed transform curves, run

```sh
python code/plot_smiles.py
```

The plot is saved to `figures/smiles.pdf`. The following scripts rerun the
underlying calculations and overwrite the corresponding smile CSVs.

```sh
python code/marked_inar_smiles.py
python code/serva_smiles.py
```

The second script implements the comparison based on the multifactor Euler
scheme in Giorgio Enrico Maria Serva's thesis.
[Numerical Methods and SSR for Rough Hawkes Heston Model](https://hdl.handle.net/10589/246994)

## Path-dependent put options

The first command reruns the grid-refinement experiment with 200,000 Monte Carlo
paths for each pair of grid size and roughness level. The second runs the
16,384-bin reference and writes it to a separate CSV.

```sh
python code/path_option_prices.py --bins 128 512 2048 --timing-runs 3
python code/path_option_prices.py --bins 16384 --output path_option_reference.csv
```

The grid-refinement command writes `code/path_option_prices_128_512_2048.csv`.
The supplied `code/path_option_prices_200k.csv` includes the reference estimates.
Wall-clock timings depend on the machine.

## License

[MIT](LICENSE)
