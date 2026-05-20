# tests/test_ingest.py
import pandas as pd
import pytest
from io import StringIO
from src.pipeline.ingest import clean_and_aggregate

SAMPLE_CSV = """Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000
16/12/2006;17:25:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000
16/12/2006;17:26:00;5.374;0.498;233.290;23.000;0.000;2.000;17.000
16/12/2006;17:27:00;5.388;0.502;233.740;23.000;0.000;1.000;17.000
"""

def test_clean_and_aggregate_returns_hourly_dataframe():
    df_raw = pd.read_csv(
        StringIO(SAMPLE_CSV),
        sep=";",
        na_values="?",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
    )
    result = clean_and_aggregate(df_raw)
    assert isinstance(result, pd.DataFrame)
    assert "consumption_kwh" in result.columns
    assert result.index.freq == "h" or result.index.inferred_freq in ("h", "H")
    assert result["consumption_kwh"].isna().sum() == 0
    assert len(result) == 1  # 4 minutes → 1 hourly bucket

def test_clean_and_aggregate_drops_missing_values():
    df_raw = pd.read_csv(
        StringIO(SAMPLE_CSV.replace("4.216", "?")),
        sep=";",
        na_values="?",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
    )
    result = clean_and_aggregate(df_raw)
    assert result["consumption_kwh"].isna().sum() == 0
