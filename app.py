"""Interactive dashboard: run with  streamlit run app.py"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model import Params, improved_controls, risk_metrics, simulate_annual_losses

st.set_page_config(page_title="Cyber Loss Model", layout="wide")
st.title("Cyber loss model (frequency-severity)")
st.caption("Illustrative assumptions only. Not calibrated to any real bank and not a risk forecast.")

with st.sidebar:
    st.header("Assumptions")
    lam = st.slider("Material incidents per year (λ)", 0.5, 10.0, 3.0, 0.5)
    median = st.slider("Median ordinary loss (EUR thousand)", 50, 2000, 400, 50) * 1000
    sigma = st.slider("Lognormal sigma (spread)", 0.5, 2.5, 1.5, 0.1)
    p_tail = st.slider("Share of extreme incidents", 0.0, 0.2, 0.05, 0.01)
    alpha = st.slider("Pareto tail index α (lower = fatter tail)", 1.2, 3.0, 1.7, 0.1)
    st.header("Controls scenario")
    f_red = st.slider("Reduction in incident frequency", 0.0, 0.8, 0.30, 0.05)
    s_red = st.slider("Reduction in typical loss size", 0.0, 0.8, 0.10, 0.05)
    n_sims = st.select_slider("Simulations", [50_000, 100_000, 200_000, 500_000], 200_000)

base = Params(lam=lam, median_loss=median, sigma=sigma, p_tail=p_tail, tail_alpha=alpha)
better = improved_controls(base, f_red, s_red)


@st.cache_data
def run(p: Params, n: int):
    losses = simulate_annual_losses(p, n, seed=1)
    return losses, risk_metrics(losses)


loss_b, m_b = run(base, n_sims)
loss_c, m_c = run(better, n_sims)

c1, c2, c3 = st.columns(3)
c1.metric("Expected annual loss", f"€{m_b['expected_loss']/1e6:,.2f}m",
          f"{m_c['expected_loss']/m_b['expected_loss']-1:+.0%} with controls", delta_color="inverse")
c2.metric("99% VaR", f"€{m_b['VaR_99.0%']/1e6:,.1f}m",
          f"{m_c['VaR_99.0%']/m_b['VaR_99.0%']-1:+.0%} with controls", delta_color="inverse")
c3.metric("99% Expected Shortfall", f"€{m_b['ES_99.0%']/1e6:,.1f}m",
          f"{m_c['ES_99.0%']/m_b['ES_99.0%']-1:+.0%} with controls", delta_color="inverse")

fig, ax = plt.subplots(figsize=(8, 3.8))
cap = np.quantile(loss_b, 0.995)
ax.hist(loss_b / 1e6, bins=80, range=(0, cap / 1e6), alpha=.6, density=True, label="Baseline")
ax.hist(loss_c / 1e6, bins=80, range=(0, cap / 1e6), alpha=.6, density=True, label="Improved controls")
ax.set_xlabel("Annual loss (EUR million, shown up to the 99.5th percentile)")
ax.set_ylabel("Density"); ax.legend()
st.pyplot(fig)

st.subheader("All metrics")
tbl = pd.DataFrame({"Baseline": m_b, "Improved controls": m_c}).drop("prob_zero_loss")
st.dataframe((tbl / 1e6).round(2).rename_axis("EUR million"))
st.info("With a heavy tail (α < 2) the variance is infinite, so tail metrics like ES are noisy. "
        "Treat them as ranges, not exact numbers.")
