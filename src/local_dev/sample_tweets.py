import pandas as pd
import random

CSV_PATH = "../data/Suicide_Ideation_Dataset_Twitter-based_.csv"
SAMPLE_SIZE = 100
RANDOM_SEED = None  # set an integer here if you want reproducible samples during testing

def load_and_sample(csv_path=CSV_PATH, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    df = pd.read_csv(csv_path)

    # Drop rows with missing tweet text - can't classify an empty tweet
    before = len(df)
    df = df.dropna(subset=["Tweet"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with missing Tweet text.")

    # Clean up trailing whitespace in labels (source data has "Potential Suicide post " with trailing space)
    df["Suicide"] = df["Suicide"].str.strip()

    if seed is not None:
        random.seed(seed)

    sampled = df.sample(n=sample_size, replace=False, random_state=seed)
    return sampled.reset_index(drop=True)

if __name__ == "__main__":
    sample = load_and_sample()
    print(f"Sampled {len(sample)} tweets.")
    print(sample.head())