from smartbuild.config import DATA_PATH
from smartbuild.data import generate_demo_data


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate_demo_data()
    data.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(data):,} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
