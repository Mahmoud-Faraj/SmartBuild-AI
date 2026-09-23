from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from smartbuild.anomalies import operational_explanation  # noqa: E402
from smartbuild.config import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, TARGET  # noqa: E402
from smartbuild.data import clean_input, generate_demo_data  # noqa: E402
from smartbuild.features import build_features  # noqa: E402
from smartbuild.modeling import load_model, train_model  # noqa: E402

st.set_page_config(page_title="SmartBuild AI", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background:#f4f7fb; border:1px solid #dce5f1; padding:14px; border-radius:12px;}
    .small-note {color:#5d6b7a; font-size:0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def demo_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    data = generate_demo_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    return data


@st.cache_resource(show_spinner="Training the demonstration forecasting model…")
def demo_model_artifact() -> dict:
    if not MODEL_PATH.exists():
        train_model(demo_data(), MODEL_PATH)
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Training a model for the uploaded dataset…")
def uploaded_model_artifact(csv_bytes: bytes) -> dict:
    uploaded_data = clean_input(pd.read_csv(pd.io.common.BytesIO(csv_bytes)))
    upload_model_path = ROOT / "models" / "uploaded_energy_model.joblib"
    train_model(uploaded_data, upload_model_path)
    return load_model(upload_model_path)


def evaluate(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    featured = build_features(data)
    prediction = artifact["model"].predict(featured[artifact["feature_columns"]])
    output = featured.copy()
    output["predicted_energy_kwh"] = prediction
    output["residual_kwh"] = output[TARGET] - prediction
    output["anomaly_score"] = output["residual_kwh"].abs() / artifact["residual_threshold"]
    output["is_anomaly"] = (output["anomaly_score"] > 1).astype(int)
    return output


st.title("⚡ SmartBuild AI")
st.caption("Explainable building-energy forecasting and operational anomaly detection")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload hourly building data", type=["csv"])
    st.markdown(
        '<p class="small-note">Required: timestamp, outdoor_temp_c, humidity_pct, '
        "occupancy, total_energy_kwh.</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    electricity_rate = st.number_input("Electricity rate ($/kWh)", 0.01, 1.00, 0.18, 0.01)
    show_subsystems = st.toggle("Show subsystem profile", value=True)
    st.info("Demo telemetry is synthetic and contains labelled operational anomalies.")

try:
    if uploaded:
        uploaded_bytes = uploaded.getvalue()
        raw = clean_input(pd.read_csv(pd.io.common.BytesIO(uploaded_bytes)))
        artifact = uploaded_model_artifact(uploaded_bytes)
        source_label = "Uploaded dataset"
    else:
        raw = demo_data()
        artifact = demo_model_artifact()
        source_label = "Reproducible demonstration building"
except ValueError as exc:
    st.error(str(exc))
    st.stop()

all_results = evaluate(raw, artifact)
holdout_start = pd.Timestamp(artifact["holdout_start"])
results = all_results.loc[all_results["timestamp"] >= holdout_start].copy()
metrics = artifact["metrics"]
anomalies = results[results["is_anomaly"] == 1].copy()

st.markdown(
    f"**Source:** {source_label} · **Period:** "
    f"{results.timestamp.min():%Y-%m-%d} to {results.timestamp.max():%Y-%m-%d}"
)

st.caption("Charts and alerts below use the untouched chronological holdout period.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast MAE", f"{metrics['mae']:.2f} kWh")
col2.metric("Model R²", f"{metrics['r2']:.3f}")
col3.metric("Flagged hours", f"{len(anomalies):,}")
estimated_excess = anomalies["residual_kwh"].clip(lower=0).sum()
col4.metric("Estimated excess cost", f"${estimated_excess * electricity_rate:,.0f}")

overview, anomalies_tab, drivers_tab, simulator_tab, methodology_tab = st.tabs(
    ["Overview", "Anomalies", "Model drivers", "What-if simulator", "Methodology"]
)

with overview:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results.timestamp, y=results[TARGET], name="Actual", line=dict(color="#144A74", width=1.5)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.timestamp,
            y=results.predicted_energy_kwh,
            name="Expected",
            line=dict(color="#F28E2B", width=1.3),
        )
    )
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies.timestamp,
                y=anomalies[TARGET],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#D62728", size=7),
            )
        )
    fig.update_layout(
        title="Actual and expected hourly energy",
        xaxis_title=None,
        yaxis_title="Energy (kWh)",
        hovermode="x unified",
        height=430,
    )
    st.plotly_chart(fig, width="stretch")

    if show_subsystems and {"hvac_kwh", "lighting_kwh", "plug_load_kwh", "critical_load_kwh"}.issubset(
        raw.columns
    ):
        daily = (
            raw.set_index("timestamp")[["hvac_kwh", "lighting_kwh", "plug_load_kwh", "critical_load_kwh"]]
            .resample("D")
            .sum()
            .reset_index()
        )
        melted = daily.melt("timestamp", var_name="Subsystem", value_name="Energy (kWh)")
        st.plotly_chart(
            px.area(
                melted,
                x="timestamp",
                y="Energy (kWh)",
                color="Subsystem",
                title="Daily subsystem energy profile",
            ),
            width="stretch",
        )

with anomalies_tab:
    if "anomaly_precision" in metrics:
        p1, p2, p3 = st.columns(3)
        p1.metric("Detection precision", f"{metrics['anomaly_precision']:.1%}")
        p2.metric("Detection recall", f"{metrics['anomaly_recall']:.1%}")
        p3.metric("Detection F1", f"{metrics['anomaly_f1']:.1%}")
        st.caption("Available only because the synthetic demo includes known injected-anomaly labels.")
    if anomalies.empty:
        st.success("No observations exceeded the current anomaly threshold.")
    else:
        selected = st.selectbox(
            "Inspect a flagged observation",
            anomalies.index,
            format_func=lambda index: (
                f"{anomalies.loc[index, 'timestamp']:%Y-%m-%d %H:%M} — "
                f"score {anomalies.loc[index, 'anomaly_score']:.1f}×"
            ),
        )
        row = anomalies.loc[selected]
        a, b, c = st.columns(3)
        a.metric("Actual", f"{row[TARGET]:.1f} kWh")
        b.metric("Expected", f"{row['predicted_energy_kwh']:.1f} kWh")
        c.metric("Difference", f"{row['residual_kwh']:+.1f} kWh")
        st.warning(operational_explanation(row))
        display_columns = ["timestamp", TARGET, "predicted_energy_kwh", "residual_kwh", "anomaly_score"]
        st.dataframe(
            anomalies[display_columns].sort_values("anomaly_score", ascending=False).head(50),
            width="stretch",
            hide_index=True,
        )

with drivers_tab:
    importance = artifact["feature_importance"].head(10).sort_values("importance")
    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Permutation importance on chronological holdout data",
    )
    fig.update_layout(xaxis_title="Increase in MAE when shuffled", yaxis_title=None, height=430)
    st.plotly_chart(fig, width="stretch")
    st.caption("Importance describes model behaviour; it does not establish causation.")

with simulator_tab:
    latest = results.iloc[-1].copy()
    c1, c2 = st.columns(2)
    simulated_temp = c1.slider("Outdoor temperature (°C)", -20.0, 40.0, float(latest.outdoor_temp_c), 1.0)
    simulated_occupancy = c2.slider("Occupancy", 0, 200, int(latest.occupancy), 5)
    scenario = latest.copy()
    scenario["outdoor_temp_c"] = simulated_temp
    scenario["occupancy"] = simulated_occupancy
    scenario_frame = pd.DataFrame([scenario[FEATURE_COLUMNS]])
    scenario_prediction = float(artifact["model"].predict(scenario_frame)[0])
    reference_prediction = float(latest.predicted_energy_kwh)
    difference = scenario_prediction - reference_prediction
    c1, c2, c3 = st.columns(3)
    c1.metric("Scenario demand", f"{scenario_prediction:.1f} kWh")
    c2.metric("Change", f"{difference:+.1f} kWh")
    c3.metric("Hourly cost", f"${scenario_prediction * electricity_rate:.2f}")
    st.caption(
        "This scenario changes temperature and occupancy while holding historical lag features constant."
    )

with methodology_tab:
    detection_results = ""
    if "anomaly_precision" in metrics:
        detection_results = f"""
        - Anomaly precision: **{metrics["anomaly_precision"]:.1%}**
        - Anomaly recall: **{metrics["anomaly_recall"]:.1%}**
        - Anomaly F1: **{metrics["anomaly_f1"]:.1%}**
        """
    st.markdown(
        f"""
        **Evaluation design**

        - The first 70% trains the model, the next 10% calibrates the anomaly threshold,
          and the final 20% is untouched holdout data.
        - Time-series observations are never randomly shuffled.
        - Random-forest forecast compared with a mean-demand baseline.
        - With synthetic labels, the residual threshold is selected on calibration data
          to maximize F1; unlabeled uploads use the calibration residual percentile.
        - Operational explanations are cautious, rule-based interpretations—not confirmed diagnoses.

        **Holdout results**

        - MAE: **{metrics["mae"]:.2f} kWh**
        - RMSE: **{metrics["rmse"]:.2f} kWh**
        - R²: **{metrics["r2"]:.3f}**
        - MAE improvement over baseline: **{metrics["mae_improvement_pct"]:.1f}%**
        {detection_results}
        """
    )
