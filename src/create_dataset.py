"""Завантажує Wine dataset зі sklearn та зберігає у data/raw/wine.csv."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine


def createAndSaveDataset(outputPath: Path = Path("data/raw/wine.csv")) -> pd.DataFrame:
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputPath, index=False)

    print(f"Wine dataset saved to {outputPath}")
    print(f"Shape: {df.shape}")
    print(f"Classes: {sorted(df['target'].unique().tolist())}")
    print(f"Features: {list(wine.feature_names)}")
    return df


if __name__ == "__main__":
    createAndSaveDataset()
