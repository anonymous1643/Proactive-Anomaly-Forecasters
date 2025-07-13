import shutil
import os
import torch
import sys
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.BackwardReconstructionModel import (
    model, test_loader, train_loader, labels, window_size,
    forecast_horizon, batch_size, g, TimeSeriesDataset, test_data, train_data
)
from Code.scoring_metrics import get_score_2, ano_evaluator, return_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_dir = './pretrained'
if os.path.isdir(pretrained_dir):
    print(f"Removing entire pretrained directory: {pretrained_dir}")
    shutil.rmtree(pretrained_dir)

# --- Prediction ---
def collect_predicted_normals(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            pred, _ = model(x)
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)  # shape: [N, H, D]
    
# --- Predict future values for test ---
test_pred_raw = collect_predicted_normals(model, test_loader)

# --- Align labels ---
aligned_labels = labels[window_size + forecast_horizon - 1:]
aligned_labels = aligned_labels[:test_pred_raw.shape[0]]
test_pred_raw = test_pred_raw[:len(aligned_labels)]

# --- Prepare train set for ECOD fitting ---
train_loader_eval = DataLoader(TimeSeriesDataset(train_data, window_size, forecast_horizon), batch_size=batch_size, shuffle=False, generator=g)
train_pred_raw = collect_predicted_normals(model, train_loader_eval)

train_true = []
with torch.no_grad():
    for i in range(len(train_data) - window_size - forecast_horizon + 1):
        y = train_data[i + window_size : i + window_size + forecast_horizon]
        train_true.append(y)
train_true = np.array(train_true)

# --- Forecast Errors ---
train_forecast_errors = np.mean((train_pred_raw - train_true) ** 2, axis=(1, 2)).reshape(-1, 1)
train_pred_init = train_pred_raw.reshape(train_pred_raw.shape[0], -1)

# --- Run Evaluation ---
results = get_score_2(
    test_pred_init=test_pred_raw.reshape(test_pred_raw.shape[0], -1),
    test_labels=aligned_labels,
    train_orig=train_pred_init,
    dataset="MSL_mixed",
    save_folder="./hybrid_model_mixed_eval"
)

# --- Print Results ---
for model_result in results:
    (
        model_name, f1, prec, rec,
        f1_pak, prec_pak, rec_pak,
        f1_comp, prec_comp, rec_comp,
        f1_range, prec_range, rec_range
    ) = model_result

    print(f"{model_name} Scores:")
    print(f"  Naive F1:       {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  PAK-AUC F1:     {f1_pak:.4f} | Precision: {prec_pak:.4f} | Recall: {rec_pak:.4f}")
    print(f"  Composite F1:   {f1_comp:.4f} | Precision: {prec_comp:.4f} | Recall: {rec_comp:.4f}")
    print(f"  Range-Based F1: {f1_range:.4f} | Precision: {prec_range:.4f} | Recall: {rec_range:.4f}")
    print("-" * 60)

# --- Real-World Forecast-Based Anomaly Detection ---
# --- Reverse test data and prepare dataset ---
backward_dataset = TimeSeriesDataset(test_data[::-1].copy(), window_size, forecast_horizon)
backward_loader = DataLoader(backward_dataset, batch_size=batch_size, shuffle=False, generator=g)

# --- Predict in batch ---
model.eval()
true_preds = []
true_gt = []

with torch.no_grad():
    for x, y in backward_loader:
        x = x.to(device)
        pred, _ = model(x)
        true_preds.append(pred.cpu().numpy())
        true_gt.append(y.numpy())

true_preds = np.vstack(true_preds)  # shape: [N, H, D]
true_gt = np.vstack(true_gt)        # shape: [N, H, D]


forecast_errors = np.mean((true_preds - true_gt) ** 2, axis=(1, 2))
forecast_errors = np.nan_to_num(forecast_errors, nan=0.0, posinf=1e3, neginf=0.0)
forecast_errors = np.log1p(forecast_errors)

# --- Per-feature Errors ---
per_feature_errors = np.mean((true_preds - true_gt) ** 2, axis=1)  # [N, D]

# --- Compute thresholds per feature from training data ---
train_preds, train_gt = collect_preds_and_targets(model, train_loader)
train_per_feature_errors = np.mean((train_preds - train_gt) ** 2, axis=1)  # [N_train, D]
feature_thresholds = np.percentile(train_per_feature_errors, 99, axis=0)   # [D]

# --- Per-feature anomaly flags ---
anomaly_flags_matrix = (per_feature_errors >= feature_thresholds)   # [N, D]
per_feature_flags = anomaly_flags_matrix.any(axis=1).astype(int)    # [N]

# --- Align Labels ---
aligned_labels_real = labels[::-1][window_size + forecast_horizon - 1:]
aligned_labels_real = aligned_labels_real[:len(forecast_errors)]
K = int(aligned_labels_real.sum())

# --- Evaluate Per-Feature Thresholding ---
evaluator_perfeature = ano_evaluator(per_feature_flags, aligned_labels_real)
scores_perfeature = return_scores(evaluator_perfeature)

# --- Print Results ---
print("\nReal-World Forecast-Based Detection (Per-Feature Thresholding):")
print(f"  Naive F1:       {scores_perfeature[0]:.4f} | Precision: {scores_perfeature[1]:.4f} | Recall: {scores_perfeature[2]:.4f}")
print(f"  PAK-AUC F1:     {scores_perfeature[3]:.4f}")
print(f"  Composite F1:   {scores_perfeature[6]:.4f}")
print(f"  Range-Based F1: {scores_perfeature[9]:.4f}")
print("-" * 60)

