import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import streamlit as st

BASE_FEATURES = ["acwr", "sleep_hours", "soreness", "hrv_ms", "prior_injury", "age"]
SPORTS = ["football", "cricket", "basketball", "kabaddi"]

FEATURE_NAME_MAP = {
    "is_kabaddi": "Sport: kabaddi",
    "is_basketball": "Sport: basketball",
    "is_cricket": "Sport: cricket",
    "acwr": "Workload ratio",
    "acwr_x_football": "Workload × football",
    "acwr_x_cricket": "Workload × cricket",
    "acwr_x_basketball": "Workload × basketball",
    "acwr_x_kabaddi": "Workload × kabaddi",
    "soreness_x_football": "Soreness × football",
    "soreness_x_cricket": "Soreness × cricket",
    "soreness_x_basketball": "Soreness × basketball",
    "soreness_x_kabaddi": "Soreness × kabaddi",
    "prior_injury_x_football": "Past injury × football",
    "prior_injury_x_cricket": "Past injury × cricket",
    "prior_injury_x_basketball": "Past injury × basketball",
    "prior_injury_x_kabaddi": "Past injury × kabaddi",
    "prior_injury": "Past injury",
    "hrv_ms": "HRV (recovery)",
    "sleep_hours": "Sleep (hours)",
    "soreness": "Soreness rating",
    "age": "Age"
}

def add_sport_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds sport-aware features:
    - 6 base features
    - is_cricket, is_basketball, is_kabaddi (football is reference)
    - 12 interaction columns: {feature}_x_{sport} for acwr, soreness, prior_injury
    """
    res = df[BASE_FEATURES].copy()
    res["is_cricket"] = (df["sport"] == "cricket").astype(int)
    res["is_basketball"] = (df["sport"] == "basketball").astype(int)
    res["is_kabaddi"] = (df["sport"] == "kabaddi").astype(int)
    
    inter_features = ["acwr", "soreness", "prior_injury"]
    for s in SPORTS:
        mask = (df["sport"] == s)
        for f in inter_features:
            res[f"{f}_x_{s}"] = np.where(mask, df[f], 0.0)
    return res

@st.cache_resource
def train_models(train_df: pd.DataFrame):
    """Trains the baseline logistic regression models on train_players.csv."""
    y_train = train_df["high_risk"]

    # Model 1: Without sport
    X_train_no = train_df[BASE_FEATURES]
    pipe_no = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe_no.fit(X_train_no, y_train)

    # Model 2: With sport (Selected model)
    X_train_sp = add_sport_features(train_df)
    pipe_sp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe_sp.fit(X_train_sp, y_train)

    return pipe_no, pipe_sp

def evaluate_models(pipe_no, pipe_sp, test_df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """Calculates metrics including Accuracy for Baseline A, Baseline B, and both logistic regression models."""
    y_test = test_df["high_risk"]

    # Baseline A: predict all Low
    pred_a = np.zeros(len(test_df), dtype=int)
    cm_a = confusion_matrix(y_test, pred_a, labels=[0, 1])
    fa_a = int(cm_a[0, 1])
    mi_a = int(cm_a[1, 0])
    acc_a = accuracy_score(y_test, pred_a)
    p_a = precision_score(y_test, pred_a, zero_division=0)
    r_a = recall_score(y_test, pred_a, zero_division=0)
    f1_a = f1_score(y_test, pred_a, zero_division=0)

    # Baseline B: workload ratio > 1.3
    pred_b = (test_df["acwr"] > 1.3).astype(int)
    cm_b = confusion_matrix(y_test, pred_b, labels=[0, 1])
    fa_b = int(cm_b[0, 1])
    mi_b = int(cm_b[1, 0])
    acc_b = accuracy_score(y_test, pred_b)
    p_b = precision_score(y_test, pred_b, zero_division=0)
    r_b = recall_score(y_test, pred_b, zero_division=0)
    f1_b = f1_score(y_test, pred_b, zero_division=0)

    # Model without sport
    X_no = test_df[BASE_FEATURES]
    prob_no = pipe_no.predict_proba(X_no)[:, 1]
    pred_no = (prob_no >= threshold).astype(int)
    cm_no = confusion_matrix(y_test, pred_no, labels=[0, 1])
    fa_no = int(cm_no[0, 1])
    mi_no = int(cm_no[1, 0])
    acc_no = accuracy_score(y_test, pred_no)
    p_no = precision_score(y_test, pred_no, zero_division=0)
    r_no = recall_score(y_test, pred_no, zero_division=0)
    f1_no = f1_score(y_test, pred_no, zero_division=0)
    auc_no = roc_auc_score(y_test, prob_no)

    # Model with sport
    X_sp = add_sport_features(test_df)
    prob_sp = pipe_sp.predict_proba(X_sp)[:, 1]
    pred_sp = (prob_sp >= threshold).astype(int)
    cm_sp = confusion_matrix(y_test, pred_sp, labels=[0, 1])
    fa_sp = int(cm_sp[0, 1])
    mi_sp = int(cm_sp[1, 0])
    acc_sp = accuracy_score(y_test, pred_sp)
    p_sp = precision_score(y_test, pred_sp, zero_division=0)
    r_sp = recall_score(y_test, pred_sp, zero_division=0)
    f1_sp = f1_score(y_test, pred_sp, zero_division=0)
    auc_sp = roc_auc_score(y_test, prob_sp)

    data = [
        {
            "Model": "Baseline A: predict all Low",
            "Accuracy": f"{acc_a:.2f}",
            "Precision": f"{p_a:.2f}",
            "Recall": f"{r_a:.2f}",
            "F1": f"{f1_a:.2f}",
            "False alarms": fa_a,
            "Missed injuries": mi_a,
            "AUC": "—"
        },
        {
            "Model": "Baseline B: workload ratio > 1.3",
            "Accuracy": f"{acc_b:.2f}",
            "Precision": f"{p_b:.2f}",
            "Recall": f"{r_b:.2f}",
            "F1": f"{f1_b:.2f}",
            "False alarms": fa_b,
            "Missed injuries": mi_b,
            "AUC": "—"
        },
        {
            "Model": "Logistic regression, no sport",
            "Accuracy": f"{acc_no:.2f}",
            "Precision": f"{p_no:.2f}",
            "Recall": f"{r_no:.2f}",
            "F1": f"{f1_no:.2f}",
            "False alarms": fa_no,
            "Missed injuries": mi_no,
            "AUC": f"{auc_no:.3f}"
        },
        {
            "Model": "Logistic regression, with sport",
            "Accuracy": f"{acc_sp:.2f}",
            "Precision": f"{p_sp:.2f}",
            "Recall": f"{r_sp:.2f}",
            "F1": f"{f1_sp:.2f}",
            "False alarms": fa_sp,
            "Missed injuries": mi_sp,
            "AUC": f"{auc_sp:.3f}"
        }
    ]
    return pd.DataFrame(data)

def evaluate_by_sport(pipe_no, pipe_sp, df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """Evaluates error counts segmented by sport with and without sport features with plain-English column headers."""
    results = []
    X_no_all = df[BASE_FEATURES]
    X_sp_all = add_sport_features(df)
    
    prob_no_all = pipe_no.predict_proba(X_no_all)[:, 1]
    prob_sp_all = pipe_sp.predict_proba(X_sp_all)[:, 1]

    pred_no_all = (prob_no_all >= threshold).astype(int)
    pred_sp_all = (prob_sp_all >= threshold).astype(int)

    for sport in SPORTS:
        mask = (df["sport"] == sport)
        y_sub = df.loc[mask, "high_risk"]
        
        pred_no_sub = pred_no_all[mask]
        pred_sp_sub = pred_sp_all[mask]

        cm_no = confusion_matrix(y_sub, pred_no_sub, labels=[0, 1])
        cm_sp = confusion_matrix(y_sub, pred_sp_sub, labels=[0, 1])

        results.append({
            "Sport": sport.capitalize(),
            "Players": int(mask.sum()),
            "Actual Injuries": int(y_sub.sum()),
            "Missed injuries (no sport)": int(cm_no[1, 0]),
            "False alarms (no sport)": int(cm_no[0, 1]),
            "Missed injuries (with sport)": int(cm_sp[1, 0]),
            "False alarms (with sport)": int(cm_sp[0, 1]),
        })

    return pd.DataFrame(results)

def evaluate_fairness(pipe_sp, df: pd.DataFrame, threshold: float = 0.35) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes fairness table (recall and false-alarm rate) on test 1 by sport and by age group."""
    X_sp = add_sport_features(df)
    prob_sp = pipe_sp.predict_proba(X_sp)[:, 1]
    pred_sp = (prob_sp >= threshold).astype(int)

    # 1. By Sport
    sport_rows = []
    for sport in SPORTS:
        mask = (df["sport"] == sport)
        y_sub = df.loc[mask, "high_risk"]
        pred_sub = pred_sp[mask]
        cm = confusion_matrix(y_sub, pred_sub, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        sport_rows.append({
            "Sport": sport.capitalize(),
            "Players": int(mask.sum()),
            "Actual Injuries": int(y_sub.sum()),
            "Recall": f"{recall:.2f}",
            "False-alarm rate": f"{far:.2f}"
        })
    df_sport = pd.DataFrame(sport_rows)

    # 2. By Age Group
    age_groups = [
        ("18–23", (df["age"] >= 18) & (df["age"] <= 23)),
        ("24–29", (df["age"] >= 24) & (df["age"] <= 29)),
        ("30–35", (df["age"] >= 30) & (df["age"] <= 35)),
    ]
    age_rows = []
    for label, mask in age_groups:
        y_sub = df.loc[mask, "high_risk"]
        pred_sub = pred_sp[mask]
        cm = confusion_matrix(y_sub, pred_sub, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        age_rows.append({
            "Age Group": label,
            "Players": int(mask.sum()),
            "Actual Injuries": int(y_sub.sum()),
            "Recall": f"{recall:.2f}",
            "False-alarm rate": f"{far:.2f}"
        })
    df_age = pd.DataFrame(age_rows)

    return df_sport, df_age

def get_coefficients(pipe_sp) -> pd.DataFrame:
    """Returns sorted coefficients of the selected model with plain-English feature labels."""
    feature_cols = (
        BASE_FEATURES + 
        ["is_cricket", "is_basketball", "is_kabaddi"] + 
        [f"{f}_x_{s}" for s in SPORTS for f in ["acwr", "soreness", "prior_injury"]]
    )
    coefs = pipe_sp.named_steps["logisticregression"].coef_[0]
    labels = [FEATURE_NAME_MAP.get(c, c) for c in feature_cols]
    
    df = pd.DataFrame({
        "Feature": labels,
        "Coefficient": coefs
    }).sort_values(by="Coefficient", ascending=False)
    return df

def compute_cost_curve(train_df: pd.DataFrame, pipe_sp, cost_ratio: int = 5):
    """Calculates total cost = N * missed + false alarms across cut-offs 0.10 to 0.70 on train data."""
    X_train = add_sport_features(train_df)
    prob_train = pipe_sp.predict_proba(X_train)[:, 1]
    y_train = train_df["high_risk"]

    thresholds = np.round(np.arange(0.10, 0.71, 0.01), 2)
    costs = []
    missed_list = []
    fa_list = []

    for t in thresholds:
        pred = (prob_train >= t).astype(int)
        cm = confusion_matrix(y_train, pred, labels=[0, 1])
        fa = int(cm[0, 1])
        mi = int(cm[1, 0])
        total_cost = cost_ratio * mi + fa
        costs.append(total_cost)
        missed_list.append(mi)
        fa_list.append(fa)

    min_idx = int(np.argmin(costs))
    best_t = thresholds[min_idx]
    best_cost = costs[min_idx]

    df_cost = pd.DataFrame({
        "Threshold": thresholds,
        "Total Cost": costs,
        "Missed Injuries": missed_list,
        "False Alarms": fa_list
    })
    return df_cost, best_t, best_cost
