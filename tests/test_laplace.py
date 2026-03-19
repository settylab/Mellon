"""Tests for Laplace approximation posterior uncertainty."""

import pytest
import jax
import jax.numpy as jnp
import mellon
from mellon.inference import compute_laplace_std


@pytest.fixture
def density_data():
    """Generate 2D data from a mixture of Gaussians."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    n = 200
    X1 = jax.random.normal(k1, (n // 2, 2)) * 0.5 + jnp.array([1.0, 1.0])
    X2 = jax.random.normal(k2, (n // 2, 2)) * 0.5 + jnp.array([-1.0, -1.0])
    X = jnp.concatenate([X1, X2], axis=0)
    return X


@pytest.fixture
def simple_loss():
    """A simple quadratic loss function for unit testing compute_laplace_std."""
    # loss(z) = 0.5 * z^T diag(precisions) z
    # Hessian diagonal = precisions
    # Expected std = 1/sqrt(precisions)
    precisions = jnp.array([1.0, 4.0, 9.0, 16.0])

    def loss_func(z):
        return 0.5 * jnp.sum(precisions * z**2)

    expected_std = 1.0 / jnp.sqrt(precisions)
    return loss_func, jnp.zeros(4), expected_std


class TestComputeLaplaceStd:
    """Unit tests for the compute_laplace_std function."""

    def test_quadratic_loss_exact(self, simple_loss):
        """Laplace std should be exact for a quadratic loss."""
        loss_func, z_map, expected_std = simple_loss
        std = compute_laplace_std(loss_func, z_map)
        assert jnp.allclose(std, expected_std, atol=1e-5), (
            f"Expected {expected_std}, got {std}"
        )

    def test_quadratic_loss_jit(self, simple_loss):
        """JIT compilation should produce the same result."""
        loss_func, z_map, expected_std = simple_loss
        std_nojit = compute_laplace_std(loss_func, z_map, jit=False)
        std_jit = compute_laplace_std(loss_func, z_map, jit=True)
        assert jnp.allclose(std_nojit, std_jit, atol=1e-6)

    def test_positive_output(self, simple_loss):
        """Output standard deviations should always be positive."""
        loss_func, z_map, _ = simple_loss
        std = compute_laplace_std(loss_func, z_map)
        assert jnp.all(std > 0)

    def test_nonquadratic_loss(self):
        """Laplace std should work for non-quadratic losses."""
        # loss(z) = sum(z^4 + z^2)
        # Hessian at z=0: diag of 12*z^2 + 2 = 2
        def loss_func(z):
            return jnp.sum(z**4 + z**2)

        z_map = jnp.zeros(5)
        std = compute_laplace_std(loss_func, z_map)
        expected = 1.0 / jnp.sqrt(2.0) * jnp.ones(5)
        assert jnp.allclose(std, expected, atol=1e-5)

    def test_clipping_near_zero_curvature(self):
        """Near-zero Hessian diagonal should be clipped, not produce inf."""
        # Flat loss: Hessian = 0
        def loss_func(z):
            return 0.0 * jnp.sum(z)

        z_map = jnp.zeros(3)
        std = compute_laplace_std(loss_func, z_map)
        assert jnp.all(jnp.isfinite(std))


class TestDensityEstimatorLaplace:
    """Integration tests: DensityEstimator with Laplace uncertainty."""

    def test_lbfgsb_with_uncertainty(self, density_data):
        """L-BFGS-B optimizer should produce uncertainty via Laplace."""
        X = density_data
        est = mellon.DensityEstimator(
            optimizer="L-BFGS-B",
            n_landmarks=20,
            predictor_with_uncertainty=True,
        )
        est.fit(X)
        assert est.pre_transformation_std is not None, (
            "Laplace should set pre_transformation_std"
        )
        assert jnp.all(est.pre_transformation_std > 0)

    def test_adam_with_uncertainty(self, density_data):
        """Adam optimizer should produce uncertainty via Laplace."""
        X = density_data
        est = mellon.DensityEstimator(
            optimizer="adam",
            n_landmarks=20,
            n_iter=50,
            predictor_with_uncertainty=True,
        )
        est.fit(X)
        assert est.pre_transformation_std is not None, (
            "Laplace should set pre_transformation_std for adam"
        )

    def test_advi_not_affected(self, density_data):
        """ADVI should still use its own std, not Laplace."""
        X = density_data
        est = mellon.DensityEstimator(
            optimizer="advi",
            n_landmarks=20,
            n_iter=50,
            predictor_with_uncertainty=True,
        )
        est.fit(X)
        # ADVI sets its own pre_transformation_std — just verify it exists
        assert est.pre_transformation_std is not None

    def test_no_uncertainty_no_laplace(self, density_data):
        """Without predictor_with_uncertainty, no Laplace should run."""
        X = density_data
        est = mellon.DensityEstimator(
            optimizer="L-BFGS-B",
            n_landmarks=20,
            predictor_with_uncertainty=False,
        )
        est.fit(X)
        assert est.pre_transformation_std is None

    def test_mean_covariance_available(self, density_data):
        """Laplace uncertainty should make mean_covariance available."""
        X = density_data
        est = mellon.DensityEstimator(
            optimizer="L-BFGS-B",
            n_landmarks=20,
            predictor_with_uncertainty=True,
        )
        est.fit(X)
        X_test = X[:10]
        mean_cov = est.predict.mean_covariance(X_test)
        assert mean_cov.shape == (10,)
        assert jnp.all(mean_cov >= 0)
        assert jnp.all(jnp.isfinite(mean_cov))

    def test_uncertainty_available(self, density_data):
        """Total uncertainty (covariance + mean_covariance) should work."""
        X = density_data
        est = mellon.DensityEstimator(
            optimizer="L-BFGS-B",
            n_landmarks=20,
            predictor_with_uncertainty=True,
        )
        est.fit(X)
        X_test = X[:10]
        unc = est.predict.uncertainty(X_test)
        cov = est.predict.covariance(X_test)
        mean_cov = est.predict.mean_covariance(X_test)
        # Total uncertainty should be sum of the two components
        assert jnp.allclose(unc, cov + mean_cov, atol=1e-6)

    def test_laplace_vs_advi_mean_agreement(self, density_data):
        """MAP (Laplace) and ADVI should produce similar mean predictions."""
        X = density_data
        est_map = mellon.DensityEstimator(
            optimizer="L-BFGS-B",
            n_landmarks=20,
            predictor_with_uncertainty=True,
        )
        est_map.fit(X)

        est_advi = mellon.DensityEstimator(
            optimizer="advi",
            n_landmarks=20,
            n_iter=200,
            predictor_with_uncertainty=True,
        )
        est_advi.fit(X)

        X_test = X[:20]
        pred_map = est_map.predict(X_test)
        pred_advi = est_advi.predict(X_test)
        # Mean predictions should be in the same ballpark
        corr = jnp.corrcoef(pred_map, pred_advi)[0, 1]
        assert corr > 0.8, f"Mean predictions poorly correlated: {corr:.3f}"

    def test_full_gp_laplace(self, density_data):
        """Laplace should work with full (non-sparse) GP."""
        X = density_data[:50]  # small for full GP
        est = mellon.DensityEstimator(
            optimizer="L-BFGS-B",
            n_landmarks=0,
            predictor_with_uncertainty=True,
        )
        est.fit(X)
        assert est.pre_transformation_std is not None
        X_test = X[:5]
        unc = est.predict.uncertainty(X_test)
        assert jnp.all(jnp.isfinite(unc))
        assert jnp.all(unc >= 0)
