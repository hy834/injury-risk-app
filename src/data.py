import os
import pandas as pd
import streamlit as st

DATA_DIR = "data"
TESTS_DIR = "tests"

REQUIRED_DATA_COLS = [
    "player_id", "sport", "week", "acwr", "sleep_hours", 
    "soreness", "hrv_ms", "prior_injury", "age", "high_risk"
]

DEMO_DATA_COLS = REQUIRED_DATA_COLS + ["case_type"]

def verify_file_exists(path: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: `{path}`. Please verify the file is present.")
        st.stop()

def verify_columns(df: pd.DataFrame, required_cols: list[str], filename: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"File `{filename}` is missing required column(s): {', '.join(missing)}.")
        st.stop()

@st.cache_data
def load_all_datasets():
    """Defensively loads and validates all required CSV datasets."""
    files = {
        "train": os.path.join(DATA_DIR, "train_players.csv"),
        "test1": os.path.join(DATA_DIR, "test1_standard.csv"),
        "test2": os.path.join(DATA_DIR, "test2_demo_cases.csv"),
        "dict": os.path.join(DATA_DIR, "data_dictionary.csv"),
    }
    
    # Check that all files exist
    for f in files.values():
        verify_file_exists(f)
    
    # Check additional markdown and test files exist
    verify_file_exists("club_protocol.md")
    verify_file_exists("briefing_rules.md")
    verify_file_exists(os.path.join(TESTS_DIR, "injection_tests.csv"))
    verify_file_exists("business_case.md")

    try:
        train_df = pd.read_csv(files["train"])
        test1_df = pd.read_csv(files["test1"])
        test2_df = pd.read_csv(files["test2"])
        dict_df = pd.read_csv(files["dict"])
    except Exception as e:
        st.error(f"Error reading CSV files: {str(e)}")
        st.stop()

    # Validate columns
    verify_columns(train_df, REQUIRED_DATA_COLS, files["train"])
    verify_columns(test1_df, REQUIRED_DATA_COLS, files["test1"])
    verify_columns(test2_df, DEMO_DATA_COLS, files["test2"])

    return train_df, test1_df, test2_df, dict_df

