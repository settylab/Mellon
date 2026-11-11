"""Tests for the leverage diagonal helper and every site that uses it.

``_leverage_diag`` computes ``diag(B M^-1 B^T)`` with a Cholesky solve. The
reference here is ``diag(B @ solve(M, B.T))``, which forms the solve a different
way, so a wrong rearrangement in the helper does not cancel out.

``test_every_leverage_site_agrees_with_reference`` runs the public API twice --
once with the helper, once with the reference substituted for it -- and compares
the results, which keeps the comparison outside ``vmap``. It discovers the call
sites by reading the module source rather than hardcoding them, so a new call
site that the drivers below do not reach fails the test instead of being
silently untested.
"""

import re
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import mellon

# `mellon.conditional` is rebound to the `_conditional` re-export shim in
# `mellon/__init__.py`, so reach the implementation module explicitly.
conditional = import_module("mellon.conditional")


def _reference_leverage_diag(B, M):
    """diag(B M^-1 B^T), formed independently of the helper. vmap-safe."""
    return jnp.diag(B @ jax.scipy.linalg.solve(M, B.T, assume_a="pos"))


@pytest.fixture
def data():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (40, 2))
    y = jnp.sum(jnp.sin(X), axis=1) + 0.1 * jax.random.normal(k2, (40,))
    return X, y


@pytest.mark.parametrize("sigma", [1e-2, 1.0, 10.0])
def test_leverage_diag_matches_reference(sigma):
    """The helper agrees with an independently formed solve."""
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    B = jax.random.normal(k1, (25, 8))
    A = jax.random.normal(k2, (8, 8))
    K_uu = A @ A.T + jnp.eye(8)  # SPD, as the call sites build it
    M = conditional.stabilize(sigma**2 * K_uu + B.T @ B, conditional.DEFAULT_JITTER)

    got = conditional._leverage_diag(B, M)
    want = _reference_leverage_diag(B, M)

    assert got.shape == (B.shape[0],)
    assert jnp.allclose(got, want, rtol=1e-6, atol=1e-10), (
        f"max diff {jnp.max(jnp.abs(got - want))}"
    )


def _call_sites_in_source():
    """Line numbers of every ``_leverage_diag(...)`` call in the module."""
    src = Path(conditional.__file__).read_text().splitlines()
    return {
        i
        for i, line in enumerate(src, start=1)
        if re.search(r"(?<![\w.])_leverage_diag\s*\(", line) and "def " not in line
    }


def _drive_all_sites(X, y):
    """Exercise every leverage path reachable from the public API."""
    sigma_pf = jnp.array([0.5, 1.5])
    Y = jnp.stack([y, 2 * y], axis=1)
    out = {}

    # _LandmarksConditional._compute_obs_variance and ._leverage, scalar sigma
    est = mellon.FunctionEstimator(sigma=1.0, ls=1.0, n_landmarks=15, obs_variance=True)
    est.fit(X, y)
    out["obs_variance"] = est.predict.obs_variance(X)
    out["leverage"] = est.predict.leverage(X)

    # ... and per-feature sigma (the vmap arms)
    est_pf = mellon.FunctionEstimator(
        sigma=sigma_pf, ls=1.0, n_landmarks=15, obs_variance=True
    )
    est_pf.fit(X, Y)
    out["obs_variance_per_feature"] = est_pf.predict.obs_variance(X)
    out["leverage_per_feature"] = est_pf.predict.leverage(X)

    # _LandmarksConditionalCholesky._leverage. Called directly with an explicit
    # sigma: a density predictor fitted without `predictor_with_uncertainty`
    # carries `sigma=None`, so the public `leverage()` cannot reach this site.
    de = mellon.DensityEstimator(ls=1.0, n_landmarks=15)
    de.fit(X)
    out["leverage_cholesky"] = de.predict._leverage(X, 0.5)

    return out


def _run_with(helper, monkeypatch, X, y):
    """Drive every site with ``helper`` in place, recording the sites reached."""
    import inspect

    sites = set()

    def recording(B, M):
        sites.add(inspect.currentframe().f_back.f_lineno)
        return helper(B, M)

    monkeypatch.setattr(conditional, "_leverage_diag", recording)
    try:
        return _drive_all_sites(X, y), sites
    finally:
        monkeypatch.undo()


def test_every_leverage_site_agrees_with_reference(data, monkeypatch):
    """Every call site agrees with the reference, and none is left untested."""
    X, y = data

    got, sites_got = _run_with(conditional._leverage_diag, monkeypatch, X, y)
    want, sites_want = _run_with(_reference_leverage_diag, monkeypatch, X, y)

    expected = _call_sites_in_source()
    assert expected, "no _leverage_diag call sites found in mellon/conditional.py"
    assert sites_got == sites_want
    missing = expected - sites_got
    assert not missing, (
        f"call sites at lines {sorted(missing)} of mellon/conditional.py are not "
        f"reached by this test; reached {sorted(sites_got)}"
    )

    for name in got:
        assert got[name].shape == want[name].shape, name
        assert jnp.all(jnp.isfinite(got[name])), f"{name} is not finite"
        assert jnp.allclose(got[name], want[name], rtol=1e-5, atol=1e-8), (
            f"{name}: max diff {jnp.max(jnp.abs(got[name] - want[name]))}"
        )


def test_leverage_is_finite_and_in_unit_interval(data):
    """Sanity bound a wrong substitution would be likely to break."""
    X, y = data
    est = mellon.FunctionEstimator(sigma=1.0, ls=1.0, n_landmarks=15)
    est.fit(X, y)
    h = est.predict.leverage(X)

    assert jnp.all(jnp.isfinite(h))
    assert jnp.all(h >= -1e-8) and jnp.all(h <= 1 + 1e-6), (
        f"leverage outside [0, 1]: min {jnp.min(h)}, max {jnp.max(h)}"
    )


def test_sparse_gp_leverage_matches_explicit(data):
    """End-to-end: the landmark leverage matches the formula rebuilt from scratch.

    Unlike the reference substitution above, this also checks that the call site
    builds ``B`` and ``M`` as intended, not only that the helper is right.
    """
    X, y = data
    sigma = 1.0
    est = mellon.FunctionEstimator(sigma=sigma, ls=1.0, n_landmarks=15)
    est.fit(X, y)
    h = est.predict.leverage(X)

    xu = jnp.asarray(est.predict.landmarks)
    cov_func = est.predict.cov_func
    B = cov_func(X, xu)
    K_uu = cov_func(xu, xu)
    M = conditional.stabilize(sigma**2 * K_uu + B.T @ B, est.predict.jitter)
    h_explicit = _reference_leverage_diag(B, M)

    assert jnp.allclose(h, h_explicit, atol=1e-6), (
        f"leverage mismatch: max diff {jnp.max(jnp.abs(h - h_explicit))}"
    )
