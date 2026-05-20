import pandas as pd
from pathlib import Path

RAW_TXT = Path("data/raw/household_power_consumption.txt")
CLEAN_PARQUET = Path("data/raw/clean.parquet")


def clean_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["datetime", "Global_active_power"]].copy()
    df = df.dropna(subset=["Global_active_power"])
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.dropna()
    df = df.set_index("datetime")
    df = df.resample("h").mean()
    df = df.rename(columns={"Global_active_power": "consumption_kwh"})
    df = df.dropna()
    return df


def ingest() -> pd.DataFrame:
    df_raw = pd.read_csv(
        RAW_TXT,
        sep=";",
        na_values="?",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
        low_memory=False,
    )
    df = clean_and_aggregate(df_raw)
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET)
    print(f"Saved {len(df)} hourly records to {CLEAN_PARQUET}")
    return df


if __name__ == "__main__":
    ingest()
