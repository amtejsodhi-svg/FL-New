"""
Baselines for Group-Structured Optimization
-------------------------------------------
These methods treat critical features (groups) instead of federated clients.

Implements:
 - GroupAvg (standard)
 - Oversampling
 - Undersampling
 - Importance-weighted SGD
 - GroupDRO (worst-group optimization)
"""

import numpy as np
from sklearn.metrics import mean_squared_error

# ================================================================
# Shared helpers
# ================================================================
def local_train_lr(X, y, w_init, epochs=1, lr=0.01):
    w = w_init.copy()
    for _ in range(epochs):
        preds = X @ w
        grad = (X.T @ (preds - y)) / len(y)
        w -= lr * grad
    return w

def global_weighted_loss(w, group_data, mu):
    loss = 0.0
    for i, (Xi, yi) in enumerate(group_data):
        preds = Xi @ w
        loss += mu[i] * mean_squared_error(yi, preds)
    return loss

def sample_groups(weights, K, rng):
    return rng.choice(len(weights), size=K, replace=False, p=weights / weights.sum())


# ================================================================
# 1. GroupAvg (uniform aggregation)
# ================================================================
def run_groupavg(group_data, mu, nu, K, rounds=1000, lr=0.01, epochs=5, seed=0):
    """
    Standard group-average SGD:
    sample groups ∝ observed frequencies ν, aggregate uniformly.
    """
    rng = np.random.RandomState(seed)
    N = len(group_data)
    w = np.zeros(group_data[0][0].shape[1])
    losses = []
    Xs, ys = [gd[0] for gd in group_data], [gd[1] for gd in group_data]

    for _ in range(rounds):
        groups = sample_groups(nu, K, rng)
        local_models = [local_train_lr(Xs[g], ys[g], w, epochs=epochs, lr=lr) for g in groups]
        w = np.mean(local_models, axis=0)
        losses.append(global_weighted_loss(w, group_data, mu))
    return np.array(losses)


# ================================================================
# 2. Oversampling
# ================================================================
def run_oversampling(group_data, mu, nu, K, rounds=1000, lr=0.01, epochs=5, seed=0):
    """
    Oversample underrepresented groups ∝ μ_i / ν_i.
    """
    rng = np.random.RandomState(seed)
    adj = mu / (nu + 1e-12)
    adj /= adj.sum()
    N = len(group_data)
    w = np.zeros(group_data[0][0].shape[1])
    losses = []
    Xs, ys = [gd[0] for gd in group_data], [gd[1] for gd in group_data]

    for _ in range(rounds):
        groups = sample_groups(adj, K, rng)
        local_models = [local_train_lr(Xs[g], ys[g], w, epochs=epochs, lr=lr) for g in groups]
        w = np.mean(local_models, axis=0)
        losses.append(global_weighted_loss(w, group_data, mu))
    return np.array(losses)


# ================================================================
# 3. Undersampling
# ================================================================
def run_undersampling(group_data, mu, nu, K, rounds=1000, lr=0.01, epochs=5, seed=0):
    """
    Undersample dominant groups ∝ min(μ_i, ν_i).
    """
    rng = np.random.RandomState(seed)
    weights = np.minimum(mu, nu)
    weights /= weights.sum()
    N = len(group_data)
    w = np.zeros(group_data[0][0].shape[1])
    losses = []
    Xs, ys = [gd[0] for gd in group_data], [gd[1] for gd in group_data]

    for _ in range(rounds):
        groups = sample_groups(weights, K, rng)
        local_models = [local_train_lr(Xs[g], ys[g], w, epochs=epochs, lr=lr) for g in groups]
        w = np.mean(local_models, axis=0)
        losses.append(global_weighted_loss(w, group_data, mu))
    return np.array(losses)


# ================================================================
# 4. Importance-weighted SGD
# ================================================================
def run_importance_sgd(group_data, mu, nu, K, rounds=1000, lr=0.01, epochs=5, seed=0):
    """
    Each batch: sample groups ∝ ν_i; aggregate using weights α_i = μ_i / ν_i.
    """
    rng = np.random.RandomState(seed)
    N = len(group_data)
    alpha = mu / (nu + 1e-12)
    w = np.zeros(group_data[0][0].shape[1])
    losses = []
    Xs, ys = [gd[0] for gd in group_data], [gd[1] for gd in group_data]

    for _ in range(rounds):
        groups = sample_groups(nu, K, rng)
        local_models = [local_train_lr(Xs[g], ys[g], w, epochs=epochs, lr=lr) for g in groups]
        weights_sel = alpha[groups]; weights_sel /= weights_sel.sum()
        w_next = np.zeros_like(w)
        for coeff, theta in zip(weights_sel, local_models):
            w_next += coeff * theta
        w = w_next
        losses.append(global_weighted_loss(w, group_data, mu))
    return np.array(losses)


# ================================================================
# 5. GroupDRO
# ================================================================
def run_groupdro(group_data, mu, nu, K, rounds=1000, lr=0.01, epochs=5, seed=0, eta=0.05):
    """
    GroupDRO: dynamically reweight groups based on current loss.
    q_i <- q_i * exp(η * ℓ_i)
    """
    rng = np.random.RandomState(seed)
    N = len(group_data)
    q = np.ones(N) / N
    w = np.zeros(group_data[0][0].shape[1])
    losses = []
    Xs, ys = [gd[0] for gd in group_data], [gd[1] for gd in group_data]

    for _ in range(rounds):
        groups = sample_groups(q, K, rng)
        local_models = []
        local_losses = np.zeros(N)
        for gid in groups:
            theta = local_train_lr(Xs[gid], ys[gid], w, epochs=epochs, lr=lr)
            local_models.append(theta)
            local_losses[gid] = mean_squared_error(ys[gid], Xs[gid] @ w)
        w = np.mean(local_models, axis=0)
        losses.append(global_weighted_loss(w, group_data, mu))

        # Update q toward high-loss groups
        q *= np.exp(eta * local_losses)
        q /= q.sum()
    return np.array(losses)
