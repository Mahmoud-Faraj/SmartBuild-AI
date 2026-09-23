# SmartBuild AI Recruiter Copy

## GitHub repository description

Explainable building-energy forecasting and residual anomaly detection with chronological evaluation,
Streamlit, scikit-learn, and transparent synthetic-data metrics.

## Resume project entry

**SmartBuild AI | Python, Pandas, scikit-learn, Streamlit**

- Built an end-to-end building-energy intelligence application using leakage-controlled time-series features,
  Random Forest regression, residual anomaly scoring, and an interactive Streamlit dashboard.
- Designed a chronological 70/10/20 train-calibration-holdout workflow; achieved 1.40 kWh MAE and 0.913 R
  squared on synthetic holdout telemetry, with 71.4% anomaly precision and 41.7% F1 against injected labels.
- Added reproducible data generation, CSV validation, baseline comparison, permutation importance, automated
  tests, continuous integration, and inspectable prediction exports.

## LinkedIn project description

SmartBuild AI is an end-to-end machine-learning prototype for commercial-building energy analysis. The
application forecasts expected hourly electricity use, flags unusual forecast residuals, explains global model
drivers, and presents cautious operational guidance in a Streamlit dashboard. I designed the evaluation to
preserve time order: 70% training, 10% threshold calibration, and 20% untouched holdout testing. The current
release reports both forecasting and anomaly-detection metrics against reproducible synthetic telemetry.

Current scope: controlled synthetic demonstration. Next validation step: a properly licensed real-world
building-energy dataset.

## Short LinkedIn post

I have completed SmartBuild AI, an end-to-end portfolio project for explainable building-energy forecasting
and operational anomaly screening.

The application combines a reproducible telemetry generator, leakage-controlled time-series features, Random
Forest regression, residual anomaly scoring, permutation importance, automated testing, and an interactive
Streamlit dashboard.

On the final synthetic holdout, the model achieved 1.40 kWh MAE and 0.913 R squared. The anomaly screen
achieved 71.4% precision and 41.7% F1 against known injected labels. These are controlled synthetic results,
not claims about an unseen real facility. The next step is validation on a properly licensed real-world
building dataset.

## Thirty-second recruiter explanation

SmartBuild AI predicts expected hourly building electricity use and compares the forecast with actual demand.
Large residuals become anomaly candidates for human review. I used a chronological split to prevent future
data from leaking into training, calibrated the anomaly threshold on a separate period, and evaluated once on
an untouched final holdout. The project demonstrates modeling, software organization, transparent evaluation,
testing, and dashboard delivery. Its current limitation is that the public dataset is synthetic, which I state
clearly throughout the repository.
