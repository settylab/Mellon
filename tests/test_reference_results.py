"""Reference results test to ensure scalar sigma behavior is unchanged."""

import jax
import jax.numpy as jnp
import numpy as np
import mellon


def test_full_gp_reference_results():
    """Full GP: predictions, leverage, obs_variance match hardcoded values."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    n, d, p = 50, 2, 3
    X = jax.random.normal(k1, (n, d))
    y = jax.random.normal(k2, (n, p))
    X_test = jax.random.normal(k3, (10, d))
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0, obs_variance=True)
    est.fit(X, y)

    pred = est.predict(X_test)
    lev = est.predict.leverage(X, sigma=sigma)
    obsvar = est.predict.obs_variance(X_test)

    expected_pred = np.array([
        [0.1591912, -0.01633006, -0.09774735],
        [0.22242522, 0.18020723, -0.02099988],
        [0.19622299, 0.13606965, -0.1066963],
        [0.11826687, -0.1078843, -0.31056051],
        [0.14248863, -0.03011926, -0.29908757],
        [0.19947812, 0.11085447, -0.00750686],
        [0.12869758, -0.0557435, -0.31332486],
        [0.18549478, -0.04098856, 0.07950502],
        [0.29005287, 0.17010726, 0.36455042],
        [0.32726478, 0.31220231, 0.21231073],
    ])

    expected_lev = np.array([
        0.0372332, 0.07869925, 0.12117246, 0.05443739, 0.07560143,
        0.05055196, 0.05284116, 0.03140333, 0.04589148, 0.12702225,
        0.02890246, 0.08439047, 0.02921787, 0.07780366, 0.05287561,
        0.09885388, 0.09658274, 0.0378513, 0.0336515, 0.04042638,
        0.04148647, 0.04255076, 0.06422805, 0.05231018, 0.04072847,
        0.05364099, 0.04714973, 0.03281598, 0.12303139, 0.03775613,
        0.10646143, 0.09640494, 0.02881728, 0.03010999, 0.09627312,
        0.0325684, 0.06231224, 0.0371162, 0.03548587, 0.13666944,
        0.05732545, 0.03451524, 0.02859058, 0.07310316, 0.03799797,
        0.08597798, 0.03010433, 0.09246368, 0.09796963, 0.0286806,
    ])

    expected_obsvar = np.array([
        [0.95486132, 1.10382589, 1.09700611],
        [0.99352028, 1.09954301, 1.09154833],
        [1.07884384, 1.06994597, 1.12319011],
        [1.01419867, 0.87782108, 1.19101712],
        [1.18976692, 0.91071511, 1.20611143],
        [0.92173907, 1.14376553, 1.08436175],
        [1.14035324, 0.91377002, 1.20676145],
        [0.96502533, 1.00159358, 0.98472199],
        [0.48300975, 0.88916662, 0.78530785],
        [0.76511332, 0.98307023, 0.95662155],
    ])

    assert jnp.allclose(pred, expected_pred, atol=1e-5), (
        f"Full GP pred mismatch: max diff = {jnp.max(jnp.abs(pred - expected_pred))}"
    )
    assert jnp.allclose(lev, expected_lev, atol=1e-5), (
        f"Full GP lev mismatch: max diff = {jnp.max(jnp.abs(lev - expected_lev))}"
    )
    assert jnp.allclose(obsvar, expected_obsvar, atol=1e-5), (
        f"Full GP obsvar mismatch: max diff = {jnp.max(jnp.abs(obsvar - expected_obsvar))}"
    )


def test_sparse_gp_reference_results():
    """Sparse GP: predictions, leverage, obs_variance match hardcoded values."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    n, d, p = 50, 2, 3
    X = jax.random.normal(k1, (n, d))
    y = jax.random.normal(k2, (n, p))
    X_test = jax.random.normal(k3, (10, d))
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=15, obs_variance=True)
    est.fit(X, y)

    pred = est.predict(X_test)
    lev = est.predict.leverage(X, sigma=sigma)
    obsvar = est.predict.obs_variance(X_test)

    expected_pred = np.array([
        [0.15897022, -0.01638545, -0.09799344],
        [0.22247079, 0.17997088, -0.02106525],
        [0.19597164, 0.13587423, -0.10677722],
        [0.11883229, -0.10784295, -0.31086893],
        [0.14281352, -0.03052275, -0.29949588],
        [0.19941441, 0.11073389, -0.00750197],
        [0.12896235, -0.05618629, -0.3136724],
        [0.18774363, -0.04146911, 0.0794172],
        [0.28743858, 0.18400688, 0.37032512],
        [0.32729985, 0.31495833, 0.21358139],
    ])

    expected_lev = np.array([
        0.03717582, 0.07859248, 0.11760941, 0.05433303, 0.07468583,
        0.05050998, 0.052129, 0.03137777, 0.04579742, 0.12701128,
        0.02889238, 0.08426626, 0.0291823, 0.07777012, 0.05280633,
        0.0977312, 0.09354495, 0.03776445, 0.03362697, 0.04040875,
        0.04144854, 0.04250362, 0.06400144, 0.05229242, 0.04054663,
        0.05360086, 0.04700136, 0.03273106, 0.12270808, 0.03773976,
        0.10592451, 0.09559162, 0.02880928, 0.03009515, 0.09618252,
        0.03255463, 0.06229311, 0.03710011, 0.03544775, 0.13588841,
        0.05603113, 0.03446286, 0.02858433, 0.07291396, 0.03798352,
        0.08591082, 0.03008417, 0.09226401, 0.09565471, 0.02866667,
    ])

    expected_obsvar = np.array([
        [0.95491038, 1.10365859, 1.0955746],
        [0.9931193, 1.09942862, 1.09088032],
        [1.07885157, 1.07006534, 1.12270206],
        [1.01548025, 0.87862958, 1.18810439],
        [1.18968743, 0.91116762, 1.20737294],
        [0.92129621, 1.14325076, 1.08310076],
        [1.14049121, 0.91437944, 1.20744462],
        [0.96354644, 1.0002296, 0.97976098],
        [0.49732506, 0.88010271, 0.81728918],
        [0.76853996, 0.9817009, 0.96505346],
    ])

    assert jnp.allclose(pred, expected_pred, atol=1e-5), (
        f"Sparse GP pred mismatch: max diff = {jnp.max(jnp.abs(pred - expected_pred))}"
    )
    assert jnp.allclose(lev, expected_lev, atol=1e-5), (
        f"Sparse GP lev mismatch: max diff = {jnp.max(jnp.abs(lev - expected_lev))}"
    )
    assert jnp.allclose(obsvar, expected_obsvar, atol=1e-5), (
        f"Sparse GP obsvar mismatch: max diff = {jnp.max(jnp.abs(obsvar - expected_obsvar))}"
    )
