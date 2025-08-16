# -----------------------------------------------------------
# Volatility Hedging under Stochastic Volatility: Heston & GARCH
# -----------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------
# 1. HESTON MODEL SIMULATION
# ---------------------

def simulate_heston(S0, v0, r, kappa, theta, sigma, rho, T, N, M):
    dt = T / N
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    for t in range(1, N + 1):
        z1 = np.random.normal(size=M)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=M)
        v[:, t] = np.maximum(v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt +
                             sigma * np.sqrt(np.maximum(v[:, t - 1], 0)) * np.sqrt(dt) * z1, 0)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt +
                                       np.sqrt(v[:, t - 1] * dt) * z2)
    return S, v

# ---------------------
# 2. GARCH(1,1) SIMULATION (placeholder)
# ---------------------

def simulate_garch(S0, omega, alpha, beta, r, T, N, M):
    dt = T / N
    S = np.zeros((M, N + 1))
    h = np.zeros((M, N + 1))
    z = np.random.normal(size=(M, N))
    S[:, 0] = S0
    h[:, 0] = omega / (1 - alpha - beta)
    for t in range(1, N + 1):
        h[:, t] = omega + alpha * (z[:, t - 1] ** 2) + beta * h[:, t - 1]
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * h[:, t]) * dt + np.sqrt(h[:, t] * dt) * z[:, t - 1])
    return S, h

# ---------------------
# 3. HEDGING STRATEGIES
# ---------------------

def delta_bs(S, K, T, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    return norm.cdf(d1)

def vega_bs(S, K, T, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    return S * norm.pdf(d1) * np.sqrt(T - t)

def hedging_loop(S_paths, K, T, r, sigma, dt, hedging_type="delta"):
    M, N = S_paths.shape[0], S_paths.shape[1] - 1
    pnl = np.zeros(M)
    for i in range(M):
        cash = 0
        delta = delta_bs(S_paths[i, 0], K, T, 0, r, sigma)
        cash -= delta * S_paths[i, 0]
        for t in range(1, N):
            t_prev = (t - 1) * dt
            delta_new = delta_bs(S_paths[i, t], K, T, t_prev, r, sigma)
            cash *= np.exp(r * dt)
            cash -= (delta_new - delta) * S_paths[i, t]
            delta = delta_new
        cash *= np.exp(r * dt)
        cash += np.maximum(S_paths[i, -1] - K, 0) - delta * S_paths[i, -1]
        pnl[i] = cash
    return pnl

# ---------------------
# 4. STATISTICS
# ---------------------

def compute_stats(pnl_array):
    return {
        'mean': np.mean(pnl_array),
        'std': np.std(pnl_array),
        'min': np.min(pnl_array),
        'max': np.max(pnl_array),
        'skew': pd.Series(pnl_array).skew(),
        'kurtosis': pd.Series(pnl_array).kurt()
    }

# ---------------------
# 5. PLOTTING RESULTS
# ---------------------

def plot_pnl_distribution(pnl_array, title):
    plt.figure(figsize=(8, 4))
    plt.hist(pnl_array, bins=50, density=True, alpha=0.7, color='steelblue')
    plt.title(title)
    plt.xlabel("P&L")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------
# 6. TABLE RESULTS (EXAMPLE)
# ---------------------

def build_result_table(results_dict):
    df = pd.DataFrame(results_dict).T
    df.columns = ['mean', 'std', 'min', 'max', 'skew', 'kurtosis']
    return df