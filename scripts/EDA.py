import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from catboost import CatBoostRegressor, Pool
import pynvml

# pd.set_option("print.max_columns", 200)

COMPETITION = "deescuy"
RANDOM_STATE = 42
DATA_DIR = Path("data") / COMPETITION
SUBMISSION_DIR = Path("submissions")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Bobot turnamen berdasarkan prestise.
TOURNAMENT_WEIGHT_RULES = {
    "fifa world cup": 2.00,
    "afc championship": 1.80,
    "afc asian cup": 1.90,
    "friendly": 0.96,
}
DEFAULT_TOURNAMENT_WEIGHT = 1.20


def get_tournament_weight(tournament_name: str) -> float:
    name = str(tournament_name).strip().lower()
    for key, weight in TOURNAMENT_WEIGHT_RULES.items():
        if key in name:
            return weight
    return DEFAULT_TOURNAMENT_WEIGHT


def compute_aw_mae(
    y_true_team,
    y_true_opp,
    y_pred_team,
    y_pred_opp,
    tournament_names,
    exact_weight: float = 0.30,
    outcome_weight: float = 0.25,
    gd_weight: float = 0.15,
    outcome_multiplier: float = 1.5,
    error_power: float = 1.5,
):
    """
    AW-MAE sesuai deskripsi kompetisi:
    1) MAE dasar dua target gol
    2) Bonus/Penalti exact, outcome, dan goal difference
    3) Jika outcome salah -> error x 1.5
    4) Error dipangkatkan 1.5
    5) Diakumulasi dengan bobot turnamen
    """
    y_true_team = np.asarray(y_true_team, dtype=float)
    y_true_opp = np.asarray(y_true_opp, dtype=float)
    y_pred_team = np.asarray(y_pred_team, dtype=float)
    y_pred_opp = np.asarray(y_pred_opp, dtype=float)

    y_true_team_int = np.rint(np.clip(y_true_team, 0, None)).astype(int)
    y_true_opp_int = np.rint(np.clip(y_true_opp, 0, None)).astype(int)
    y_pred_team_int = np.rint(np.clip(y_pred_team, 0, None)).astype(int)
    y_pred_opp_int = np.rint(np.clip(y_pred_opp, 0, None)).astype(int)

    base_mae = (np.abs(y_true_team - y_pred_team) + np.abs(y_true_opp - y_pred_opp)) / 2.0

    exact_match = (y_true_team_int == y_pred_team_int) & (y_true_opp_int == y_pred_opp_int)

    outcome_true = np.sign(y_true_team_int - y_true_opp_int)
    outcome_pred = np.sign(y_pred_team_int - y_pred_opp_int)
    outcome_match = outcome_true == outcome_pred

    gd_true = y_true_team_int - y_true_opp_int
    gd_pred = y_pred_team_int - y_pred_opp_int
    gd_match = gd_true == gd_pred

    # Bonus jika benar, penalti jika salah. Error dijaga tetap non-negatif.
    aug_error = base_mae.copy()
    aug_error += np.where(exact_match, -exact_weight, exact_weight)
    aug_error += np.where(outcome_match, -outcome_weight, outcome_weight)
    aug_error += np.where(gd_match, -gd_weight, gd_weight)
    aug_error = np.clip(aug_error, 0.0, None)

    aug_error = np.where(~outcome_match, aug_error * outcome_multiplier, aug_error)
    loss = np.power(aug_error, error_power)

    weights = np.asarray([get_tournament_weight(t) for t in tournament_names], dtype=float)
    weighted_score = float(np.sum(loss * weights) / np.sum(weights))

    detail = {
        "base_mae_mean": float(np.mean(base_mae)),
        "aug_error_mean": float(np.mean(aug_error)),
        "exact_match_rate": float(np.mean(exact_match)),
        "outcome_match_rate": float(np.mean(outcome_match)),
        "gd_match_rate": float(np.mean(gd_match)),
        "mean_tournament_weight": float(np.mean(weights)),
    }
    return weighted_score, detail

# Identifikasi file utama
all_csv = sorted(DATA_DIR.rglob("*.csv"))
if not all_csv:
    raise FileNotFoundError(f"Tidak ada file CSV di {DATA_DIR}")

print("CSV ditemukan:")
for f in all_csv:
    print("-", f.relative_to(DATA_DIR))

def pick_file(candidates, keywords):
    for f in candidates:
        name = f.name.lower()
        if all(k in name for k in keywords):
            return f
    return None

sample_sub_path = pick_file(all_csv, ["sample", "submission"])
train_path = pick_file(all_csv, ["train"])
test_path = pick_file(all_csv, ["test"])

# Fallback heuristik jika penamaan file tidak standar
if train_path is None or test_path is None:
    non_sample = [f for f in all_csv if f != sample_sub_path]
    if len(non_sample) >= 2:
        dfs = []
        for f in non_sample:
            df_tmp = pd.read_csv(f, nrows=200)
            dfs.append((f, df_tmp.shape[1], set(df_tmp.columns)))

        if sample_sub_path is not None:
            ss = pd.read_csv(sample_sub_path)
            pred_col = [c for c in ss.columns if c.lower() not in ["id", "index"]]
            pred_col = pred_col[0] if pred_col else ss.columns[-1]

            for f, _, cols in dfs:
                if pred_col in cols:
                    train_path = f
                    break
            for f, _, cols in dfs:
                if f == train_path:
                    continue
                if pred_col not in cols:
                    test_path = f
                    break

if train_path is None or test_path is None:
    raise RuntimeError("Gagal mengidentifikasi file train/test secara otomatis. Silakan cek nama file.")

print("\nFile terpilih:")
print("train:", train_path.relative_to(DATA_DIR))
print("test :", test_path.relative_to(DATA_DIR))
print("sample_submission:", sample_sub_path.relative_to(DATA_DIR) if sample_sub_path else "(tidak ditemukan)")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("\nShape train:", train_df.shape)
print("Shape test :", test_df.shape)

print(train_df.head())

# Tentukan kolom target, id, dan kandidat fitur
sample_sub = pd.read_csv(sample_sub_path) if sample_sub_path else None

id_candidates = [c for c in train_df.columns if c.lower() in ["id", "index"]]
id_col = id_candidates[0] if id_candidates else None

if sample_sub is not None:
    submission_target_cols = [c for c in sample_sub.columns if c != id_col]
    target_cols = [c for c in submission_target_cols if c in train_df.columns]

    # Fallback jika nama kolom submission berbeda.
    if len(target_cols) < 2:
        fallback = [c for c in train_df.columns if c not in test_df.columns and "goal" in c.lower()]
        for c in fallback:
            if c not in target_cols:
                target_cols.append(c)
else:
    target_cols = [c for c in train_df.columns if c not in test_df.columns and "goal" in c.lower()]

if len(target_cols) < 2:
    raise RuntimeError(
        "Target skor tidak lengkap. Diperlukan minimal dua kolom target (misal team_goals dan opp_goals)."
    )

team_goals_col = next((c for c in target_cols if c.lower() == "team_goals"), target_cols[0])
opp_goals_col = next(
    (c for c in target_cols if c.lower() == "opp_goals" and c != team_goals_col),
    None,
)
if opp_goals_col is None:
    remaining = [c for c in target_cols if c != team_goals_col]
    opp_goals_col = remaining[0]

target_cols = [team_goals_col, opp_goals_col]

feature_cols_train = [c for c in train_df.columns if c not in target_cols]
feature_cols_common = [c for c in feature_cols_train if c in test_df.columns]

if id_col and id_col in feature_cols_common:
    feature_cols_wo_id = [c for c in feature_cols_common if c != id_col]
else:
    feature_cols_wo_id = feature_cols_common

train_only_features = [c for c in feature_cols_train if c not in test_df.columns]

if not feature_cols_wo_id:
    raise RuntimeError("Tidak ada fitur yang overlap antara train dan test.")

print("id_col          :", id_col)
print("target_cols     :", target_cols)
print("n_features_train:", len(feature_cols_train))
print("n_features_used :", len(feature_cols_wo_id))
print("dropped_features:", len(train_only_features))
if train_only_features:
    print("Contoh dropped:", train_only_features[:10])

# EDA ringkas
eda_cols = feature_cols_train + target_cols
print("\nInfo train:")
print(train_df[eda_cols].info())

print("\nMissing value (top 20):")
missing = train_df[eda_cols].isna().sum().sort_values(ascending=False)
print(missing.head(20))

print("\nStatistik deskriptif numerik:")
print(train_df[eda_cols].describe(include=[np.number]).T.head(20))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, target_cols):
    sns.histplot(train_df[col], kde=True, ax=ax)
    ax.set_title(f"Distribusi target: {col}")
plt.tight_layout()
plt.show()