# SmartBuild AI Model Card

## Model purpose

SmartBuild AI estimates expected hourly commercial-building electricity use and screens large forecast
residuals for operational review. It is a portfolio prototype and decision-support demonstration, not a
fault-diagnosis or safety system.

## Model and features

- Estimator: Random Forest regressor with 240 trees
- Target: `total_energy_kwh`
- Inputs: weather, occupancy, cyclic calendar variables, one-hour and 24-hour lags, and a shifted 24-hour
  rolling mean
- Interpretation: holdout permutation importance

Subsystem loads are excluded from the predictor set because they nearly sum to the target and would create
target leakage.

## Evaluation protocol

Observations remain in time order. The earliest 70% trains the model, the next 10% calibrates the anomaly
threshold, and the last 20% is an untouched holdout. The demonstration threshold is selected on calibration
data to maximize F1 against known injected labels. Unlabelled uploads use the 97.5th percentile of calibration
residual magnitude.

## Current demonstration results

| Metric | Holdout result |
|---|---:|
| MAE | 1.397 kWh |
| RMSE | 1.837 kWh |
| R squared | 0.913 |
| Baseline MAE | 11.160 kWh |
| MAE improvement | 87.5% |
| Anomaly precision | 71.4% |
| Anomaly recall | 29.4% |
| Anomaly F1 | 41.7% |

The detector found 5 of 17 injected holdout events and raised 2 false alerts. These results show a useful but
conservative screen: most alerts correspond to injected anomalies, but several injected events are missed.

## Intended use

- Demonstrating an end-to-end time-series machine-learning workflow
- Exploring expected demand and residual anomaly signals
- Supporting human review of building schedules and energy patterns

## Uses outside scope

- Confirming equipment faults
- Safety-critical control or automated maintenance dispatch
- Claiming energy savings without a controlled real-building study
- Representing synthetic results as real-facility performance

## Limitations

The dataset is synthetic, one demonstration building is represented, and the generator simplifies real sensor
noise and operational change. The Random Forest does not quantify predictive uncertainty. The what-if tool
holds historical lag values constant. External validation and drift controls are required before operational
deployment.
