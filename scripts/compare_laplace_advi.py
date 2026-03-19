"""
Compare Laplace approximation vs ADVI for posterior uncertainty estimation.

Produces three figures:
  1. Uncertainty comparison on a 1D density estimation task
  2. Uncertainty correlation scatter (Laplace vs ADVI)
  3. Runtime scaling comparison across different data sizes
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mellon

jax.config.update("jax_enable_x64", True)


def generate_1d_data(n, seed=42):
    """Mixture of two Gaussians in 1D."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    X1 = jax.random.normal(k1, (n // 2, 1)) * 0.3 + 2.0
    X2 = jax.random.normal(k2, (n // 2, 1)) * 0.5 - 1.0
    return jnp.concatenate([X1, X2], axis=0)


def generate_2d_data(n, seed=42):
    """Mixture of two Gaussians in 2D."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    X1 = jax.random.normal(k1, (n // 2, 2)) * 0.5 + jnp.array([1.0, 1.0])
    X2 = jax.random.normal(k2, (n // 2, 2)) * 0.5 + jnp.array([-1.0, -1.0])
    return jnp.concatenate([X1, X2], axis=0)


def fit_and_time(X, optimizer, n_landmarks=30, n_iter=200):
    """Fit a DensityEstimator and return predictor + elapsed time."""
    est = mellon.DensityEstimator(
        optimizer=optimizer,
        n_landmarks=n_landmarks,
        n_iter=n_iter,
        predictor_with_uncertainty=True,
    )
    t0 = time.perf_counter()
    est.fit(X)
    elapsed = time.perf_counter() - t0
    return est, elapsed


# ── Figure 1: 1D Uncertainty Comparison ──────────────────────────────────

print("Figure 1: 1D uncertainty comparison...")
X_1d = generate_1d_data(300)
X_grid = jnp.linspace(-3.0, 4.0, 200)[:, None]

est_laplace, _ = fit_and_time(X_1d, "L-BFGS-B", n_landmarks=30)
est_advi, _ = fit_and_time(X_1d, "advi", n_landmarks=30, n_iter=200)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for ax, est, label in [
    (axes[0], est_laplace, "L-BFGS-B + Laplace"),
    (axes[1], est_advi, "ADVI"),
]:
    pred = est.predict
    mean = pred(X_grid)
    cov = pred.covariance(X_grid)
    mean_cov = pred.mean_covariance(X_grid)
    total_unc = cov + mean_cov
    std = jnp.sqrt(total_unc)
    mean_std = jnp.sqrt(mean_cov)

    ax.plot(X_grid, mean, "k-", lw=1.5, label="Mean prediction")
    ax.fill_between(
        X_grid.ravel(),
        mean - 2 * std,
        mean + 2 * std,
        alpha=0.15,
        color="C0",
        label="Total uncertainty (2 std)",
    )
    ax.fill_between(
        X_grid.ravel(),
        mean - 2 * mean_std,
        mean + 2 * mean_std,
        alpha=0.3,
        color="C1",
        label="Mean uncertainty (2 std)",
    )
    ax.plot(X_1d, jnp.full(X_1d.shape[0], float(jnp.min(mean) - 0.5)), "|", color="gray", alpha=0.3, ms=8)
    ax.set_ylabel("Log density")
    ax.set_title(label)
    ax.legend(loc="upper right", fontsize=8)

axes[1].set_xlabel("x")
fig.suptitle("Posterior Uncertainty: Laplace vs ADVI", fontsize=14)
fig.tight_layout()
fig.savefig("laplace_vs_advi_1d.png", dpi=150)
print("  Saved laplace_vs_advi_1d.png")


# ── Figure 2: Uncertainty Correlation ────────────────────────────────────

print("Figure 2: Uncertainty correlation...")
X_2d = generate_2d_data(300)

est_lap2, _ = fit_and_time(X_2d, "L-BFGS-B", n_landmarks=30)
est_advi2, _ = fit_and_time(X_2d, "advi", n_landmarks=30, n_iter=200)

X_test = X_2d[:100]

mean_lap = est_lap2.predict(X_test)
mean_advi = est_advi2.predict(X_test)
unc_lap = est_lap2.predict.uncertainty(X_test)
unc_advi = est_advi2.predict.uncertainty(X_test)
mean_cov_lap = est_lap2.predict.mean_covariance(X_test)
mean_cov_advi = est_advi2.predict.mean_covariance(X_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.scatter(mean_advi, mean_lap, s=10, alpha=0.6)
lims = [min(float(mean_advi.min()), float(mean_lap.min())),
        max(float(mean_advi.max()), float(mean_lap.max()))]
ax.plot(lims, lims, "k--", alpha=0.4, lw=1)
corr = float(jnp.corrcoef(mean_lap, mean_advi)[0, 1])
ax.set_xlabel("ADVI mean prediction")
ax.set_ylabel("Laplace mean prediction")
ax.set_title(f"Mean Predictions (r={corr:.4f})")

ax = axes[1]
ax.scatter(jnp.sqrt(unc_advi), jnp.sqrt(unc_lap), s=10, alpha=0.6, color="C1")
ax.set_xlabel("ADVI total std")
ax.set_ylabel("Laplace total std")
corr_unc = float(jnp.corrcoef(unc_lap, unc_advi)[0, 1])
ax.set_title(f"Total Uncertainty (r={corr_unc:.4f})")
lims = [0, max(float(jnp.sqrt(unc_advi).max()), float(jnp.sqrt(unc_lap).max()))]
ax.plot(lims, lims, "k--", alpha=0.4, lw=1)

ax = axes[2]
ax.scatter(jnp.sqrt(mean_cov_advi), jnp.sqrt(mean_cov_lap), s=10, alpha=0.6, color="C2")
ax.set_xlabel("ADVI mean std")
ax.set_ylabel("Laplace mean std")
corr_mean = float(jnp.corrcoef(mean_cov_lap, mean_cov_advi)[0, 1])
ax.set_title(f"Mean Uncertainty (r={corr_mean:.4f})")
lims = [0, max(float(jnp.sqrt(mean_cov_advi).max()), float(jnp.sqrt(mean_cov_lap).max()))]
ax.plot(lims, lims, "k--", alpha=0.4, lw=1)

fig.suptitle("Laplace vs ADVI: Prediction & Uncertainty Correlation", fontsize=13)
fig.tight_layout()
fig.savefig("laplace_vs_advi_correlation.png", dpi=150)
print("  Saved laplace_vs_advi_correlation.png")


# ── Figure 3: Runtime Scaling ────────────────────────────────────────────

print("Figure 3: Runtime scaling...")
data_sizes = [50, 100, 200, 400, 800]
n_landmarks_list = [10, 15, 20, 30, 40]
times_lbfgs = []
times_lbfgs_laplace = []
times_advi = []

for n, nl in zip(data_sizes, n_landmarks_list):
    print(f"  n={n}, n_landmarks={nl}")
    X = generate_2d_data(n, seed=123)

    # L-BFGS-B without uncertainty (no Laplace)
    est_no_unc = mellon.DensityEstimator(
        optimizer="L-BFGS-B", n_landmarks=nl, predictor_with_uncertainty=False
    )
    t0 = time.perf_counter()
    est_no_unc.fit(X)
    times_lbfgs.append(time.perf_counter() - t0)

    # L-BFGS-B with Laplace
    _, t = fit_and_time(X, "L-BFGS-B", n_landmarks=nl)
    times_lbfgs_laplace.append(t)

    # ADVI
    _, t = fit_and_time(X, "advi", n_landmarks=nl, n_iter=200)
    times_advi.append(t)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(data_sizes, times_lbfgs, "o-", label="L-BFGS-B (no uncertainty)", color="C0")
ax.plot(data_sizes, times_lbfgs_laplace, "s-", label="L-BFGS-B + Laplace", color="C2")
ax.plot(data_sizes, times_advi, "^-", label="ADVI (200 iter)", color="C1")
ax.set_xlabel("Number of data points")
ax.set_ylabel("Wall time (seconds)")
ax.set_title("Runtime: Laplace vs ADVI")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("laplace_vs_advi_runtime.png", dpi=150)
print("  Saved laplace_vs_advi_runtime.png")

print("\nDone! All figures saved.")
