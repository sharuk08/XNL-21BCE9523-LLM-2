import pandas as pd
import os

def download_dataset():
    df = pd.read_csv("hf://datasets/odunola/sentimenttweets/train_preprocessed-sample.csv")
    df.to_csv("data/raw.csv", index=False)
    print(f"Dataset downloaded and saved to data/raw.csv, total rows: {len(df)}")

download_dataset()