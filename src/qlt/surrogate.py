from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class QuboModel:
    n: int
    intercept: float
    linear: np.ndarray
    quadratic: np.ndarray
    diagnostics: dict[str, float]

    def energy(self, schedules: np.ndarray) -> np.ndarray:
        x = np.asarray(schedules, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        e = self.intercept + x @ self.linear
        e = e + np.einsum("bi,ij,bj->b", x, self.quadratic, x)
        return e

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "intercept": self.intercept,
            "linear": self.linear.tolist(),
            "quadratic_upper": self.quadratic.tolist(),
            "diagnostics": self.diagnostics,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def design_matrix(schedules: np.ndarray) -> np.ndarray:
    x = np.asarray(schedules, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    cols = [np.ones(x.shape[0])]
    cols.extend([x[:, i] for i in range(x.shape[1])])
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            cols.append(x[:, i] * x[:, j])
    return np.column_stack(cols)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_rank_correlation(true: np.ndarray, predicted: np.ndarray) -> float:
    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if true.size < 2:
        return 0.0
    true_ranks = _average_ranks(true)
    predicted_ranks = _average_ranks(predicted)
    true_centered = true_ranks - np.mean(true_ranks)
    predicted_centered = predicted_ranks - np.mean(predicted_ranks)
    denom = float(np.linalg.norm(true_centered) * np.linalg.norm(predicted_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(true_centered, predicted_centered) / denom)


def pairwise_order_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    correct = 0.0
    total = 0
    for i in range(true.size):
        for j in range(i + 1, true.size):
            true_sign = np.sign(true[i] - true[j])
            if true_sign == 0:
                continue
            pred_sign = np.sign(predicted[i] - predicted[j])
            total += 1
            if pred_sign == true_sign:
                correct += 1.0
            elif pred_sign == 0:
                correct += 0.5
    return float(correct / total) if total else 0.0


def top_k_recall(true: np.ndarray, predicted: np.ndarray, k: int) -> float:
    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if true.size == 0:
        return 0.0
    k = max(1, min(int(k), true.size))
    true_top = set(np.argsort(true, kind="mergesort")[:k].astype(int).tolist())
    predicted_top = set(np.argsort(predicted, kind="mergesort")[:k].astype(int).tolist())
    return float(len(true_top.intersection(predicted_top)) / k)


def fit_qubo(
    schedules: np.ndarray,
    objectives: np.ndarray,
    ridge: float = 1e-4,
    validation_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[QuboModel, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    x = np.asarray(schedules, dtype=float)
    y = np.asarray(objectives, dtype=float)
    order = rng.permutation(len(y))
    n_val = max(1, int(round(validation_fraction * len(y))))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    phi_train = design_matrix(x[train_idx])
    phi_val = design_matrix(x[val_idx])
    reg = ridge * np.eye(phi_train.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(phi_train.T @ phi_train + reg, phi_train.T @ y[train_idx])
    pred_train = phi_train @ beta
    pred_val = phi_val @ beta

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def r2(a: np.ndarray, b: np.ndarray) -> float:
        den = float(np.sum((a - np.mean(a)) ** 2))
        return 1.0 - float(np.sum((a - b) ** 2)) / den if den > 0 else 0.0

    n = x.shape[1]
    linear = beta[1 : 1 + n].copy()
    quadratic = np.zeros((n, n), dtype=float)
    k = 1 + n
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[i, j] = 0.5 * beta[k]
            quadratic[j, i] = 0.5 * beta[k]
            k += 1
    val_top_k = max(1, min(5, int(np.ceil(0.2 * len(val_idx)))))
    diagnostics = {
        "ridge": float(ridge),
        "train_rmse": rmse(y[train_idx], pred_train),
        "val_rmse": rmse(y[val_idx], pred_val),
        "train_r2": r2(y[train_idx], pred_train),
        "val_r2": r2(y[val_idx], pred_val),
        "val_spearman": spearman_rank_correlation(y[val_idx], pred_val),
        "val_pairwise_order_accuracy": pairwise_order_accuracy(y[val_idx], pred_val),
        "val_top_k": int(val_top_k),
        "val_top_k_recall": top_k_recall(y[val_idx], pred_val, val_top_k),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }
    model = QuboModel(n=n, intercept=float(beta[0]), linear=linear, quadratic=quadratic, diagnostics=diagnostics)
    fit_rows = {
        "train_true": y[train_idx],
        "train_pred": pred_train,
        "val_true": y[val_idx],
        "val_pred": pred_val,
    }
    return model, fit_rows


def all_bitstrings(n: int) -> np.ndarray:
    values = np.arange(2**n, dtype=np.uint32)
    return ((values[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)
