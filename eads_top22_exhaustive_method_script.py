# ============================================================
# Exhaustive train/test feature-subset search for Eads prediction
# ============================================================
#
# Goal:
#   Use predefined top 22 features from feature-importance plot.
#   Test all unique feature combinations from MIN_FEATURES to MAX_FEATURES.
#   Rank combinations by Test MAE.
#
# This script does NOT save millions of rows.
# It saves only:
#   1. Top 50 combinations based on Test MAE
#   2. Best combination for each feature count
#   3. Best overall summary
#   4. Predictions for best combination
#   5. Progress checkpoint for resume
#
# Main model:
#   ExtraTreesRegressor
#
# ============================================================

import os
import re
import json
import time
import heapq
import itertools
import warnings
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")


# ============================================================
# USER SETTINGS
# ============================================================

input_xlsx = "additives_RDKit_adsorption_unique_descriptors_high_adsorption.xlsx"
input_sheet = "ml_ready_adsorption"

target_col = "Eads"

metadata_cols = [
    "query_name",
    "SMILES_used",
    "descriptor_species_used",
    "descriptor_note",
    "status"
]

# Train/test split
test_size = 0.20
random_state = 42

# Feature combination range
MIN_FEATURES = 7
MAX_FEATURES = 22

# Number of best combinations to save
TOP_K = 50

# Parallel settings
N_JOBS = 112

# Batch size.
# Smaller batch = safer checkpointing.
# Larger batch = less overhead.
BATCH_SIZE = 1000

# Resume if interrupted
RESUME = True

# Output files
output_summary_xlsx = "Eads_top22_subset_search_summary.xlsx"
output_top50_csv = "Eads_top50_feature_combinations.csv"
output_best_by_count_csv = "Eads_best_by_feature_count.csv"
output_predictions_csv = "Eads_best_combination_predictions.csv"
progress_file = "Eads_subset_search_progress.json"
log_file = "Eads_subset_search_progress.log"

# Choose model.
# Recommended: ExtraTrees based on previous results.
MODEL_NAME = "ExtraTrees"

# ExtraTrees settings.
# Increase n_estimators for final high-quality run.
# For faster testing, use 100 or 200.
N_ESTIMATORS = 300

# Predefined top 22 features from your feature-importance plot
PREDEFINED_TOP22_FEATURES = [
    "NHOHCount",
    "n_N",
    "fr_NH2",
    "Halogen_fraction",
    "MR_per_volume",
    "fr_Al_OH",
    "n_chalcogen_total",
    "MolVolume_3D",
    "PMI1_3D",
    "MaxPartialCharge",
    "SPS",
    "MolLogP",
    "InertialShapeFactor_3D",
    "MaxEStateIndex",
    "BertzCT",
    "MinAbsEStateIndex",
    "RotatableBond_fraction",
    "coordination_atom_fraction",
    "fr_C_O",
    "n_F",
    "NumHAcceptors",
    "PartialCharge_range"
]


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def write_log(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(str(message) + "\n")


def clean_numeric_value(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    s = s.replace("−", "-")

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)

    if match:
        return float(match.group())

    return np.nan


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_model(model_name, random_state):
    if model_name == "ExtraTrees":
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=random_state,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=1
        )

    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=random_state,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=1
        )

    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.85,
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown MODEL_NAME: {model_name}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])


def evaluate_subset(features, X_train, X_test, y_train, y_test):
    features = list(features)

    Xtr = X_train[features].copy()
    Xte = X_test[features].copy()

    model = build_model(MODEL_NAME, random_state)

    try:
        model.fit(Xtr, y_train)

        y_train_pred = model.predict(Xtr)
        y_test_pred = model.predict(Xte)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = rmse(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = rmse(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        return {
            "N_features": len(features),
            "Features": "; ".join(features),
            "Train_MAE": train_mae,
            "Train_RMSE": train_rmse,
            "Train_R2": train_r2,
            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2,
            "Status": "OK"
        }

    except Exception as e:
        return {
            "N_features": len(features),
            "Features": "; ".join(features),
            "Train_MAE": np.nan,
            "Train_RMSE": np.nan,
            "Train_R2": np.nan,
            "Test_MAE": np.nan,
            "Test_RMSE": np.nan,
            "Test_R2": np.nan,
            "Status": f"ERROR: {e}"
        }


def total_combinations(n_features, min_k, max_k):
    import math
    return sum(math.comb(n_features, k) for k in range(min_k, max_k + 1))


def combinations_for_k(features, k, start_index=0):
    """
    Generator for combinations of size k.
    Skips first start_index combinations for resume.
    """
    return itertools.islice(itertools.combinations(features, k), start_index, None)


def save_progress(current_k, processed_in_current_k, total_processed):
    progress = {
        "current_k": current_k,
        "processed_in_current_k": processed_in_current_k,
        "total_processed": total_processed,
        "MIN_FEATURES": MIN_FEATURES,
        "MAX_FEATURES": MAX_FEATURES,
        "MODEL_NAME": MODEL_NAME,
        "N_ESTIMATORS": N_ESTIMATORS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def load_progress():
    if not RESUME:
        return None

    if not os.path.exists(progress_file):
        return None

    try:
        with open(progress_file, "r") as f:
            return json.load(f)

    except Exception:
        return None


def load_existing_top_results():
    if os.path.exists(output_top50_csv):
        try:
            df = pd.read_csv(output_top50_csv)
            return df
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def update_top_k(existing_top_df, new_rows_df, top_k):
    combined = pd.concat([existing_top_df, new_rows_df], ignore_index=True)

    combined = combined.dropna(subset=["Test_MAE", "Test_RMSE"])

    combined = combined.drop_duplicates(subset=["Features"], keep="first")

    combined = combined.sort_values(
        ["Test_MAE", "Test_RMSE", "N_features", "Test_R2"],
        ascending=[True, True, True, False]
    ).head(top_k).reset_index(drop=True)

    combined["Rank_by_Test_MAE"] = np.arange(1, len(combined) + 1)

    return combined


def update_best_by_count(existing_df, new_rows_df):
    combined = pd.concat([existing_df, new_rows_df], ignore_index=True)

    combined = combined.dropna(subset=["Test_MAE", "Test_RMSE"])

    combined = combined.sort_values(
        ["N_features", "Test_MAE", "Test_RMSE", "Test_R2"],
        ascending=[True, True, True, False]
    )

    best = combined.groupby("N_features", as_index=False).first()

    best = best.sort_values("N_features").reset_index(drop=True)

    return best


def load_existing_best_by_count():
    if os.path.exists(output_best_by_count_csv):
        try:
            return pd.read_csv(output_best_by_count_csv)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def fit_best_model_and_save_predictions(best_features, X_train, X_test, y_train, y_test, train_idx, test_idx, df_clean):
    model = build_model(MODEL_NAME, random_state)

    Xtr = X_train[best_features].copy()
    Xte = X_test[best_features].copy()

    model.fit(Xtr, y_train)

    y_train_pred = model.predict(Xtr)
    y_test_pred = model.predict(Xte)

    rows = []

    for idx, true_val, pred_val in zip(train_idx, y_train, y_train_pred):
        rows.append({
            "Split": "train",
            "Original_index": idx,
            "Eads_true": true_val,
            "Eads_pred": pred_val,
            "Residual": true_val - pred_val
        })

    for idx, true_val, pred_val in zip(test_idx, y_test, y_test_pred):
        rows.append({
            "Split": "test",
            "Original_index": idx,
            "Eads_true": true_val,
            "Eads_pred": pred_val,
            "Residual": true_val - pred_val
        })

    pred_df = pd.DataFrame(rows)

    meta_cols = [
        c for c in ["query_name", "SMILES_used", "descriptor_species_used", "status"]
        if c in df_clean.columns
    ]

    meta_df = df_clean[meta_cols].copy()
    meta_df["Original_index"] = meta_df.index

    pred_df = pred_df.merge(meta_df, on="Original_index", how="left")

    pred_df.to_csv(output_predictions_csv, index=False)

    return pred_df


# ============================================================
# MAIN WORKFLOW
# ============================================================

start_time = time.time()

write_log("================================================")
write_log("Starting predefined top-22 exhaustive subset search")
write_log("================================================")

if not os.path.exists(input_xlsx):
    raise FileNotFoundError(f"Input file not found: {input_xlsx}")

df = pd.read_excel(input_xlsx, sheet_name=input_sheet, engine="openpyxl")

if target_col not in df.columns:
    raise ValueError(f"Target column not found: {target_col}")

df[target_col] = df[target_col].apply(clean_numeric_value)
df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)

write_log(f"Rows used after target cleaning: {len(df_clean)}")

# Validate predefined features
missing_features = [f for f in PREDEFINED_TOP22_FEATURES if f not in df_clean.columns]

if missing_features:
    missing_df = pd.DataFrame({"Missing_feature": missing_features})
    missing_df.to_csv("Eads_missing_predefined_features.csv", index=False)

    raise ValueError(
        "Some predefined top-22 features are missing from the input file. "
        "See Eads_missing_predefined_features.csv"
    )

features = PREDEFINED_TOP22_FEATURES.copy()

if MAX_FEATURES > len(features):
    MAX_FEATURES = len(features)

if MIN_FEATURES < 1:
    MIN_FEATURES = 1

if MIN_FEATURES > MAX_FEATURES:
    raise ValueError("MIN_FEATURES cannot be greater than MAX_FEATURES.")

X = df_clean[features].copy()

for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

y = df_clean[target_col].astype(float)

X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X,
    y,
    X.index,
    test_size=test_size,
    random_state=random_state
)

write_log(f"Training set size: {len(X_train)}")
write_log(f"Test set size: {len(X_test)}")
write_log(f"Model: {MODEL_NAME}")
write_log(f"N_ESTIMATORS: {N_ESTIMATORS}")
write_log(f"Feature range: {MIN_FEATURES} to {MAX_FEATURES}")

n_total = total_combinations(len(features), MIN_FEATURES, MAX_FEATURES)

write_log(f"Total feature-combination tasks: {n_total}")

# Load previous progress and existing results
progress = load_progress()

top50_df = load_existing_top_results()
best_by_count_df = load_existing_best_by_count()

if progress is not None:
    start_k = int(progress["current_k"])
    processed_in_current_k_start = int(progress["processed_in_current_k"])
    total_processed = int(progress["total_processed"])

    write_log("Resume mode ON.")
    write_log(f"Resuming from k = {start_k}")
    write_log(f"Already processed in current k = {processed_in_current_k_start}")
    write_log(f"Total previously processed = {total_processed}")

else:
    start_k = MIN_FEATURES
    processed_in_current_k_start = 0
    total_processed = 0

    write_log("Starting fresh run.")

# Main loop
for k in range(start_k, MAX_FEATURES + 1):

    if progress is not None and k == start_k:
        start_index_for_k = processed_in_current_k_start
    else:
        start_index_for_k = 0

    write_log("------------------------------------------------")
    write_log(f"Processing feature-count k = {k}")
    write_log(f"Starting from combination index = {start_index_for_k}")
    write_log("------------------------------------------------")

    combo_iter = combinations_for_k(features, k, start_index=start_index_for_k)

    batch = []
    processed_in_current_k = start_index_for_k

    for combo in combo_iter:
        batch.append(combo)

        if len(batch) >= BATCH_SIZE:
            batch_start_time = time.time()

            rows = Parallel(
                n_jobs=N_JOBS,
                backend="loky",
                verbose=0,
                pre_dispatch="2*n_jobs"
            )(
                delayed(evaluate_subset)(
                    c,
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )
                for c in batch
            )

            batch_df = pd.DataFrame(rows)

            # Update top 50 and best-by-count only
            top50_df = update_top_k(top50_df, batch_df, TOP_K)
            best_by_count_df = update_best_by_count(best_by_count_df, batch_df)

            top50_df.to_csv(output_top50_csv, index=False)
            best_by_count_df.to_csv(output_best_by_count_csv, index=False)

            processed_in_current_k += len(batch)
            total_processed += len(batch)

            save_progress(k, processed_in_current_k, total_processed)

            batch_time = time.time() - batch_start_time

            best_mae_so_far = top50_df.iloc[0]["Test_MAE"] if len(top50_df) > 0 else np.nan
            best_rmse_so_far = top50_df.iloc[0]["Test_RMSE"] if len(top50_df) > 0 else np.nan

            write_log(
                f"k={k} | total_processed={total_processed}/{n_total} | "
                f"processed_in_k={processed_in_current_k} | "
                f"batch_time_sec={batch_time:.1f} | "
                f"best_Test_MAE={best_mae_so_far:.6f} | "
                f"best_Test_RMSE={best_rmse_so_far:.6f}"
            )

            batch = []

    # Final batch for this k
    if len(batch) > 0:
        batch_start_time = time.time()

        rows = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            verbose=0,
            pre_dispatch="2*n_jobs"
        )(
            delayed(evaluate_subset)(
                c,
                X_train,
                X_test,
                y_train,
                y_test
            )
            for c in batch
        )

        batch_df = pd.DataFrame(rows)

        top50_df = update_top_k(top50_df, batch_df, TOP_K)
        best_by_count_df = update_best_by_count(best_by_count_df, batch_df)

        top50_df.to_csv(output_top50_csv, index=False)
        best_by_count_df.to_csv(output_best_by_count_csv, index=False)

        processed_in_current_k += len(batch)
        total_processed += len(batch)

        save_progress(k, processed_in_current_k, total_processed)

        batch_time = time.time() - batch_start_time

        best_mae_so_far = top50_df.iloc[0]["Test_MAE"] if len(top50_df) > 0 else np.nan
        best_rmse_so_far = top50_df.iloc[0]["Test_RMSE"] if len(top50_df) > 0 else np.nan

        write_log(
            f"k={k} final batch | total_processed={total_processed}/{n_total} | "
            f"processed_in_k={processed_in_current_k} | "
            f"batch_time_sec={batch_time:.1f} | "
            f"best_Test_MAE={best_mae_so_far:.6f} | "
            f"best_Test_RMSE={best_rmse_so_far:.6f}"
        )

    # Mark next k start cleanly
    save_progress(k + 1, 0, total_processed)

# Final summary
runtime_min = (time.time() - start_time) / 60

top50_df = pd.read_csv(output_top50_csv)
best_by_count_df = pd.read_csv(output_best_by_count_csv)

top50_df = top50_df.sort_values(
    ["Test_MAE", "Test_RMSE", "N_features", "Test_R2"],
    ascending=[True, True, True, False]
).reset_index(drop=True)

top50_df["Rank_by_Test_MAE"] = np.arange(1, len(top50_df) + 1)

best_row = top50_df.iloc[0].to_dict()
best_features = [x.strip() for x in best_row["Features"].split(";")]

pred_df = fit_best_model_and_save_predictions(
    best_features,
    X_train,
    X_test,
    y_train,
    y_test,
    train_idx,
    test_idx,
    df_clean
)

summary_df = pd.DataFrame([{
    "Best_model": MODEL_NAME,
    "N_estimators": N_ESTIMATORS,
    "Rows_used": len(df_clean),
    "N_train": len(X_train),
    "N_test": len(X_test),
    "Total_predefined_features": len(features),
    "Subset_size_min": MIN_FEATURES,
    "Subset_size_max": MAX_FEATURES,
    "Total_combinations_evaluated": n_total,
    "Best_N_features": best_row["N_features"],
    "Best_features": best_row["Features"],
    "Best_Test_MAE": best_row["Test_MAE"],
    "Best_Test_RMSE": best_row["Test_RMSE"],
    "Best_Test_R2": best_row["Test_R2"],
    "Best_Train_MAE": best_row["Train_MAE"],
    "Best_Train_RMSE": best_row["Train_RMSE"],
    "Best_Train_R2": best_row["Train_R2"],
    "Runtime_minutes": runtime_min
}])

features_df = pd.DataFrame({
    "Predefined_top22_features": features,
    "Initial_rank_from_plot": np.arange(1, len(features) + 1)
})

with pd.ExcelWriter(output_summary_xlsx, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="best_summary", index=False)
    top50_df.to_excel(writer, sheet_name="top50_by_Test_MAE", index=False)
    best_by_count_df.to_excel(writer, sheet_name="best_by_feature_count", index=False)
    features_df.to_excel(writer, sheet_name="predefined_features", index=False)

write_log("================================================")
write_log("Subset search completed.")
write_log("================================================")
write_log(f"Summary Excel: {output_summary_xlsx}")
write_log(f"Top 50 CSV: {output_top50_csv}")
write_log(f"Best by count CSV: {output_best_by_count_csv}")
write_log(f"Predictions CSV: {output_predictions_csv}")
write_log(f"Runtime minutes: {runtime_min:.2f}")

write_log("Best feature combination:")
write_log(best_row["Features"])
write_log(f"Best Test MAE: {best_row['Test_MAE']}")
write_log(f"Best Test RMSE: {best_row['Test_RMSE']}")
write_log(f"Best Test R2: {best_row['Test_R2']}")

write_log("Done.")



