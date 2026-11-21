# Cyber Loss Model (Frequency-Severity)

A Monte Carlo model of **annual cyber losses at a hypothetical bank**, built with free, open-source tools (Python, NumPy, SciPy, pandas, Matplotlib, Streamlit). It shows how banks and supervisors quantify operational risk, and how much better security controls could reduce expected losses and tail risk.

> **Important:** All inputs are **illustrative assumptions**. They are not calibrated to any real bank, and the results are not a forecast. The project demonstrates the method, not a real risk number.

## Why this topic

The ECB's supervisory priorities for 2026-28 include operational resilience and ICT capabilities, and the EBA reports that operational risk is rising, driven by cyber risk, data security and fraud. Banks typically measure this kind of risk with a loss distribution approach: model *how often* incidents happen and *how large* each one is, then combine them.

## Model

Annual aggregate loss: `S = X_1 + ... + X_N`

- **Frequency:** `N ~ Poisson(λ)`, the number of material incidents per year
- **Severity:** a mixture of
  - a **lognormal** body for ordinary incidents, and
  - a **Pareto** tail (probability `p_tail`) for rare, extreme incidents
- **Outputs:** expected loss, Value-at-Risk (VaR) and Expected Shortfall (ES) at several confidence levels

**Baseline assumptions (all editable):**

| Parameter | Value |
|---|---|
| Incidents per year (λ) | 3.0 |
| Median ordinary loss | €400,000 |
| Lognormal sigma | 1.5 |
| Share of extreme incidents | 5% |
| Pareto minimum / tail index α | €2m / 1.7 |

**Controls scenario:** 30% fewer incidents and 10% smaller typical loss.

## Results (500,000 simulated years)

| Metric | Baseline | Improved controls | Change |
|---|---|---|---|
| Expected annual loss | €4.25m | €2.75m | -35% |
| Median annual loss | €2.29m | €1.12m | -51% |
| 95% VaR | €14.2m | €10.0m | -29% |
| 99% VaR | €30.3m | €22.3m | -27% |
| 99% Expected Shortfall | €53.4m | €43.1m | -19% |

Validation checks:

- The simulated mean (€4.246m) matches the closed-form mean (€4.240m).
- Across 10 random seeds, expected loss varies by about 0.4%, 99% VaR by 0.7%, and 99% ES by 2.5%.
- Sensitivity analysis shows the 99% VaR is most sensitive to the lognormal sigma and the incident frequency.

Charts are in `outputs/` (`loss_distribution.png`, `loss_exceedance.png`), along with CSV tables.

## Run it

```bash
git clone <your-repo-url>
cd cyber-loss-model
pip install -r requirements.txt

python -m pytest              # run the tests
cd src && python run_analysis.py   # generate tables and charts in outputs/
cd .. && streamlit run app.py      # interactive dashboard
```

## Project structure

```
├── app.py                 # Streamlit dashboard with sliders
├── src/model.py           # Frequency-severity model, VaR/ES, scenarios
├── src/run_analysis.py    # Baseline vs. controls, stability, sensitivity, charts
├── tests/test_model.py    # Unit tests
├── outputs/               # Generated tables and charts
└── requirements.txt
```

## Limitations

- **Assumptions are not calibrated.** Real bank loss data is confidential. Parameters should be calibrated to published sources before drawing conclusions.
- **Heavy tail.** With a tail index below 2 the variance is infinite, so tail metrics (especially ES) are noisy. Treat them as ranges.
- **Independence.** Incidents are assumed independent and identically distributed. In reality, attacks cluster and a single provider failure can hit many systems at once.
- **No dependence on the economic cycle, insurance, or recovery.**
- Frequency is Poisson, which can understate year-to-year variation. A negative binomial distribution is a natural extension.

## Possible extensions

- Calibrate the frequency to public incident data, e.g. the free [VERIS Community Database](https://github.com/vz-risk/VCDB), and the severity to published breach-cost studies
- Add a negative binomial frequency and compare with Poisson
- Add insurance layers (deductible and limit) and see how they change tail risk
- Combine with the Gordon-Loeb model to find how much a bank should spend on security
- Add third-party concentration risk (one provider failure affecting many banks)

## Data and sources

- ECB Banking Supervision: supervisory priorities 2026-28
- EBA: Risk Assessment Report, June 2026

## Author

Prabhneet Kaur
