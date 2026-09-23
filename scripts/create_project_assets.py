"""Create a recruiter-facing dashboard preview from the exported holdout data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    predictions = pd.read_csv(ROOT / "outputs" / "holdout_predictions.csv", parse_dates=["timestamp"])
    window = predictions.tail(24 * 14)
    flagged = window[window["is_anomaly"] == 1]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(14, 8), dpi=150, facecolor="#F7F9FC")
    grid = figure.add_gridspec(2, 4, height_ratios=[0.28, 1], hspace=0.35, wspace=0.25)
    cards = [
        ("Forecast MAE", "1.40 kWh"),
        ("Model R squared", "0.913"),
        ("Detection precision", "71.4%"),
        ("Detection F1", "41.7%"),
    ]
    for index, (label, value) in enumerate(cards):
        axis = figure.add_subplot(grid[0, index])
        axis.set_facecolor("#FFFFFF")
        axis.text(0.06, 0.68, label, color="#5D6B7A", fontsize=10, transform=axis.transAxes)
        axis.text(0.06, 0.22, value, color="#172033", fontsize=20, weight="bold", transform=axis.transAxes)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#DCE5F1")

    axis = figure.add_subplot(grid[1, :])
    axis.set_facecolor("#FFFFFF")
    axis.plot(window["timestamp"], window["total_energy_kwh"], color="#144A74", lw=1.5, label="Actual")
    axis.plot(
        window["timestamp"],
        window["predicted_energy_kwh"],
        color="#F28E2B",
        lw=1.3,
        label="Expected",
    )
    axis.scatter(
        flagged["timestamp"],
        flagged["total_energy_kwh"],
        color="#D62728",
        s=34,
        label="Flagged",
        zorder=4,
    )
    axis.set_title(
        "Actual and expected hourly energy on final holdout", loc="left", fontsize=14, weight="bold"
    )
    axis.set_ylabel("Energy kWh")
    axis.legend(frameon=False, ncol=3, loc="upper right")
    axis.grid(color="#E7ECF2", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("SmartBuild AI", x=0.055, y=0.98, ha="left", fontsize=24, weight="bold", color="#172033")
    figure.text(
        0.055,
        0.93,
        "Explainable building-energy forecasting and operational anomaly detection",
        fontsize=11,
        color="#5D6B7A",
    )
    output = ROOT / "assets" / "dashboard_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
