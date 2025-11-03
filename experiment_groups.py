"""
EXPERIMENTAL FRAMEWORK: Transport-Weighted SGD (TW-SGD) vs Baselines
---------------------------------------------------------------------

This script studies optimization over critical groups, not federated clients.

Implements:
 - Adult dataset experiments (fairness across demographic groups)
 - Synthetic financial-style data with group biases
 - Ablation: effect of mini-batch group size K
 - Infeasibility study: varying Sinkhorn regularizer ε
 - Comparison with baselines (GroupAvg, Oversampling, Undersampling, GroupDRO, Importance weighting)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import combinations
from sklearn.datasets import fetch_openml, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# ================================================================
# CONFIG
# ================================================================
SAVE_DIR = "./results_TW_SGD"
os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
GROUP_BATCH_SIZES = [2, 4, 8, 16]
REGULARIZERS = [1e-3, 1e-2, 1e-1, 1.0]   # for infeasible / entropic study
ROUNDS = 1000
LOCAL_EPOCHS = 5
LR = 0.01
TOL = 1e-12
MAX_ITERS = 10000

# ================================================================
# 1. DATASETS
# ================================================================
def load_adult_dataset():
    """
    Adult dataset with protected groups (sex × race).
    Task: predict income > 50K.
    Groups = intersection of sex and race categories.
    """
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame.dropna()
    df["income"] = (df["class"] == ">50K").astype(int)

    num_cols = df.select_dtypes(include="number").columns
    X = df[num_cols].values
    y = df["income"].values

    df["group"] = df["sex"].astype(str) + "_" + df["race"].astype(str)
    group_labels, group_ids = np.unique(df["group"], return_inverse=True)

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, group_ids, test_size=0.2, random_state=42, stratify=group_ids
    )
    return (X_train, y_train, g_train), (X_test, y_test, g_test), group_labels


def load_synthetic_financial_data(num_groups=20, dim=50, samples_per_group=150):
    """
    Synthetic regression dataset with biased group sampling (ν) vs. target importance (μ).
    """
    X_full, y_full = make_regression(
        n_samples=num_groups * samples_per_group, n_features=dim, noise=0.1, random_state=42
    )
    X_full = StandardScaler().fit_transform(X_full)
    group_data = [
        (X_full[i * samples_per_group:(i + 1) * samples_per_group],
         y_full[i * samples_per_group:(i + 1) * samples_per_group])
        for i in range(num_groups)
    ]
    idx = np.arange(1, num_groups + 1, dtype=float)
    mu = np.power(idx[::-1], 3); mu /= mu.sum()   # target importance
    nu = np.power(idx, 3); nu /= nu.sum()         # observed frequency
    return group_data, mu, nu


# ================================================================
# 2. OPTIMIZATION HELPERS
# ================================================================
def local_train_lr(X, y, w_init, epochs=1, lr=0.01):
    """Simple linear regression within each group."""
    w = w_init.copy()
    for _ in range(epochs):
        preds = X @ w
        grad = (X.T @ (preds - y)) / len(y)
        w -= lr * grad
    return w

def global_weighted_loss(w, group_data, mu):
    """Compute μ-weighted MSE across groups."""
    loss = 0.0
    for i, (Xi, yi) in enumerate(group_data):
        preds = Xi @ w
        loss += mu[i] * mean_squared_error(yi, preds)
    return loss


# ================================================================
# 3. TRANSPORT-WEIGHTED SGD
# ================================================================
from ipfp_utils import (
    solve_T_with_given_subsets,
    all_K_subsets_1based,
    estimate_q_by_mc,
    column_users_and_weights,
)

def run_transport_weighted_sgd(group_data, mu, nu, K, regularizer=1e-3, seed=0):
    """
    Perform stochastic optimization where mini-batches contain K groups,
    reweighted using masked Sinkhorn transport alignment between ν and μ.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    N = len(group_data)
    subsets_K = all_K_subsets_1based(N, K)
    q = estimate_q_by_mc(subsets_K, nu, N, K, num_samples=100000, rng=rng)
    try:
        T, subsets_used, M_mask, info = solve_T_with_given_subsets(
            mu, q, subsets_K, tol=TOL, max_iter=MAX_ITERS, verbose=False
        )
        feasible = True
    except Exception:
        feasible = False
        T, subsets_used, M_mask, info = solve_T_with_given_subsets(
            mu, q, subsets_K, tol=TOL * 10, max_iter=MAX_ITERS // 10, verbose=False
        )

    d = group_data[0][0].shape[1]
    w = np.zeros(d)
    losses = []
    q_cum = np.cumsum(q)
    Xs = [gd[0] for gd in group_data]
    ys = [gd[1] for gd in group_data]

    for t in range(ROUNDS):
        j = np.searchsorted(q_cum, rng.rand(), side="right")
        groups_S, weights_T = column_users_and_weights(T, subsets_used, j)
        local_models = [local_train_lr(Xs[g], ys[g], w, epochs=LOCAL_EPOCHS, lr=LR) for g in groups_S]
        w_next = np.zeros_like(w)
        for coeff, theta in zip(weights_T, local_models):
            w_next += coeff * theta
        w = w_next
        losses.append(global_weighted_loss(w, group_data, mu))
    return np.array(losses), info, feasible


# ================================================================
# 4. ABLATION STUDIES
# ================================================================
def ablation_batch_size():
    """Study the effect of mini-batch group size K."""
    group_data, mu, nu = load_synthetic_financial_data()
    results = {}
    for K in GROUP_BATCH_SIZES:
        print(f"\n--- Ablation: batch size K={K} ---")
        seed_losses = []
        for seed in SEEDS:
            losses, _, _ = run_transport_weighted_sgd(group_data, mu, nu, K, seed=seed)
            seed_losses.append(losses)
        mean_loss = np.mean(seed_losses, axis=0)
        results[K] = mean_loss
        np.save(f"{SAVE_DIR}/loss_K{K}.npy", mean_loss)

    plt.figure(figsize=(8, 5))
    for K, L in results.items():
        plt.plot(L, label=f"K={K}")
    plt.yscale("log")
    plt.legend(); plt.grid(True)
    plt.xlabel("Iteration"); plt.ylabel("μ-weighted loss")
    plt.title("TW-SGD — Effect of Group Batch Size K")
    plt.savefig(f"{SAVE_DIR}/ablation_batchsize.png", dpi=300)
    plt.show()


def infeasibility_study():
    """Show how marginal gap ||μ - μ̂||∞ changes with Sinkhorn regularizer."""
    group_data, mu, nu = load_synthetic_financial_data()
    K = 4
    gaps = []
    for reg in REGULARIZERS:
        _, info, feasible = run_transport_weighted_sgd(group_data, mu, nu, K, regularizer=reg)
        err = info.get("p_match_err_inf", np.nan)
        gaps.append(err)
        print(f"ε={reg:.1e} | feasible={feasible} | ||μ - μ̂||∞={err:.3e}")
    plt.figure()
    plt.plot(REGULARIZERS, gaps, marker="o")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Regularizer ε"); plt.ylabel(r"Marginal gap $\|\mu-\hat{\mu}\|_\infty$")
    plt.title("Entropy-Regularized Projection — Gap vs ε")
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/infeasibility_gap.png", dpi=300)
    plt.show()


# ================================================================
# 5. COMPARISON WITH BASELINES
# ================================================================
from baselines_groups import (
    run_groupavg, run_oversampling, run_undersampling, run_groupdro, run_importance_sgd
)

def compare_with_baselines():
    """Compare TW-SGD vs standard group-based baselines."""
    group_data, mu, nu = load_synthetic_financial_data()
    K = 4
    results = {}

    results["TW-SGD"] = run_transport_weighted_sgd(group_data, mu, nu, K, seed=0)[0]
    results["GroupAvg"] = run_groupavg(group_data, mu, nu, K)
    results["Oversampling"] = run_oversampling(group_data, mu, nu, K)
    results["Undersampling"] = run_undersampling(group_data, mu, nu, K)
    results["GroupDRO"] = run_groupdro(group_data, mu, nu, K)
    results["ImportanceSGD"] = run_importance_sgd(group_data, mu, nu, K)

    df = pd.DataFrame({k: v for k, v in results.items()})
    df.to_csv(f"{SAVE_DIR}/baseline_comparison.csv", index=False)

    plt.figure(figsize=(8, 5))
    for name, L in results.items():
        plt.plot(L, label=name)
    plt.yscale("log"); plt.grid(True)
    plt.xlabel("Iteration"); plt.ylabel("μ-weighted loss")
    plt.title("TW-SGD vs Baselines on Group-Structured Regression")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/baseline_comparison.png", dpi=300)
    plt.show()


# ================================================================
# 6. MAIN ENTRY
# ================================================================
if __name__ == "__main__":
    ablation_batch_size()
    infeasibility_study()
    compare_with_baselines()
