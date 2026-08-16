import pandas as pd
import random

CSV_PATH = "../../data/Suicide_Ideation_Dataset_Twitter-based_.csv"
SAMPLE_SIZE = 100
RANDOM_SEED = None  # set an integer here for reproducible samples during testing

def load_and_sample(csv_path=CSV_PATH, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED, sample_all=False):
    """
    Loads tweets from csv_path and returns a sample.

    sample_all=True processes every row in the file instead of randomly
    sampling. This is used for the interview assessment file (11 tweets),
    where the goal is to classify every provided tweet, not a random subset.
    """
    df = pd.read_csv(csv_path)

    before = len(df)
    df = df.dropna(subset=["Tweet"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with missing Tweet text.")

    # fillna("") before strip() - the interview file has a blank Suicide
    # column, and calling .str.strip() directly on NaN values would raise
    df["Suicide"] = df["Suicide"].fillna("").str.strip()

    if seed is not None:
        random.seed(seed)

    if sample_all:
        return df.reset_index(drop=True)

    # Never request more rows than exist in the source file
    actual_sample_size = min(sample_size, len(df))
    sampled = df.sample(n=actual_sample_size, replace=False, random_state=seed)
    return sampled.reset_index(drop=True)

if __name__ == "__main__":
    sample = load_and_sample()
    print(f"Sampled {len(sample)} tweets.")
    print(sample.head())