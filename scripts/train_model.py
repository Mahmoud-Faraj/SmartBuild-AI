import pandas as pd

from smartbuild.config import DATA_PATH, MODEL_PATH
from smartbuild.modeling import train_model


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    result = train_model(data, MODEL_PATH)
    print("Model evaluation (chronological holdout)")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.3f}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
