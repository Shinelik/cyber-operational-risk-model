"""Run the baseline vs. improved-controls analysis and save tables and charts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Params, improved_controls, risk_metrics, simulate_annual_losses, theoretical_mean

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)
N_SIMS = 500_000


def main() -> None:
    base = Params()
    better = improved_controls(base)

    loss_base = simulate_annual_losses(base, N_SIMS, seed=1)
    loss_better = simulate_annual_losses(better, N_SIMS, seed=1)

    # 1) Sanity check: simulated mean vs. closed-form mean
    print(f"Simulated mean : {loss_base.mean():,.0f} EUR")
    print(f"Theoretical    : {theoretical_mean(base):,.0f} EUR")

    # 2) Metrics table
    table = pd.DataFrame({"Baseline": risk_metrics(loss_base),
                          "Improved controls": risk_metrics(loss_better)})
    table["Change"] = table["Improved controls"] / table["Baseline"] - 1
    table.to_csv(OUT / "risk_metrics.csv")
    print("\n", table.round(3).to_string())

    # 3) Stability: how much does the 99% VaR / ES move across random seeds?
    seeds = range(10)
    stab = pd.DataFrame([risk_metrics(simulate_annual_losses(base, N_SIMS, s)) for s in seeds])
    summary = stab[["expected_loss", "VaR_99.0%", "ES_99.0%"]].agg(["mean", "std"]).T
    summary["rel_std"] = summary["std"] / summary["mean"]
    summary.to_csv(OUT / "seed_stability.csv")
    print("\nSeed stability (10 runs):\n", summary.round(3).to_string())

    # 4) Sensitivity of 99% VaR to the main assumptions
    rows = []
    for name, grid in {"lam": [1.5, 3, 4.5], "sigma": [1.2, 1.5, 1.8],
                       "tail_alpha": [1.4, 1.7, 2.2], "p_tail": [0.02, 0.05, 0.10]}.items():
        for v in grid:
            p = Params(**{**base.__dict__, name: v})
            m = risk_metrics(simulate_annual_losses(p, 200_000, seed=7))
            rows.append({"parameter": name, "value": v,
                         "expected_loss": m["expected_loss"], "VaR_99.0%": m["VaR_99.0%"]})
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "sensitivity.csv", index=False)
    print("\nSensitivity:\n", sens.round({"expected_loss": 0, "VaR_99.0%": 0}).to_string(index=False))

    # 5) Charts
    fig, ax = plt.subplots(figsize=(7, 4))
    cap = np.quantile(loss_base, 0.995)
    ax.hist(loss_base / 1e6, bins=80, range=(0, cap / 1e6), alpha=.6, density=True, label="Baseline")
    ax.hist(loss_better / 1e6, bins=80, range=(0, cap / 1e6), alpha=.6, density=True, label="Improved controls")
    ax.axvline(np.quantile(loss_base, .99) / 1e6, color="k", ls="--", lw=1, label="Baseline 99% VaR")
    ax.set_xlabel("Annual cyber loss (EUR million, shown up to the 99.5th percentile)")
    ax.set_ylabel("Density"); ax.set_title("Simulated annual cyber loss distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "loss_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, arr in [("Baseline", loss_base), ("Improved controls", loss_better)]:
        s = np.sort(arr)[::-1]
        s = s[s > 0]
        ax.loglog(s / 1e6, np.arange(1, len(s) + 1) / len(arr), label=lab)
    ax.set_xlabel("Annual loss (EUR million)"); ax.set_ylabel("Probability of exceeding")
    ax.set_title("Loss exceedance curve"); ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "loss_exceedance.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
