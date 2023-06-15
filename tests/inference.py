import jax.numpy as jnp
import mellon


def test_compute_transform():
    din = 2
    dout = 3
    z = jnp.ones(din)
    mu = jnp.ones(dout)
    L = jnp.ones((dout, din))
    transform = mellon.inference.compute_transform(mu, L)
    assert callable(transform), "compute_transform should returna function."
    y = transform(z)
    assert y.shape == (dout,), "The transformation shoudl produce the right shape."


def test_compute_loss_func():
    din = 2
    nn_distances = jnp.ones(din)
    loss_f = mellon.inference.compute_loss_func(nn_distances, 2, lambda x: x, 2)
    assert callable(loss_f), "compute_loss_func should returna function."
    z = jnp.ones(din)
    loss = loss_f(z)
    assert jnp.ndim(loss) == 0, "The loss should be a scalar value."


def test_minimize_adam():
    din = 2

    def loss_f(x):
        return jnp.sum(x**2)

    known_optimum = jnp.zeros(din)
    init = jnp.ones(din)
    result = mellon.inference.minimize_adam(loss_f, init, n_iter=2)
    assert hasattr(result, "pre_transformation")
    assert hasattr(result, "opt_state")
    assert hasattr(result, "losses")
    diff = jnp.abs(result.pre_transformation - known_optimum)
    assert jnp.max(diff) < 1e0, "Optimization result shoud be close to known minimum."


def test_run_advi():
    din = 2

    def loss_f(x):
        return jnp.sum(x**2)

    known_optimum = jnp.zeros(din)
    init = jnp.ones(din)
    result = mellon.inference.run_advi(loss_f, init, n_iter=2)
    assert hasattr(result, "pre_transformation")
    assert hasattr(result, "pre_transformation_std")
    assert hasattr(result, "losses")
    diff = jnp.abs(result.pre_transformation - known_optimum)
    assert jnp.max(diff) < 1e0, "Optimization result shoud be close to known minimum."


def test_minimize_lbfgsb():
    din = 2

    def loss_f(x):
        return jnp.sum(x**2)

    known_optimum = jnp.zeros(din)
    init = jnp.ones(din)
    result = mellon.inference.minimize_lbfgsb(loss_f, init)
    assert hasattr(result, "pre_transformation")
    assert hasattr(result, "opt_state")
    assert hasattr(result, "loss")
    diff = jnp.abs(result.pre_transformation - known_optimum)
    assert jnp.max(diff) < 1e-10, "Optimization result shoud be close to known minimum."


def test_compute_log_density_x():
    def test_trans(x):
        return hash(x)

    x = (1,)
    result = mellon.inference.compute_log_density_x(x, test_trans)
    assert result == test_trans(
        x
    ), "The latend representation of the density should be transformed correctly."
