"""Serialization must refuse to write state whose classes cannot be resolved.

Mellon serializes classes by name: the state carries ``classname`` and
``module_name``, and loading resolves the class with
``getattr(import_module(module_name), classname)``. A class that is not bound at
module level -- typically one defined inside a factory function -- still reports
a plausible ``__name__`` and ``__module__``, so the write used to succeed and
only the much later load failed. These tests pin the fail-fast behaviour and,
just as importantly, that nothing which used to round-trip stopped doing so.
"""

import copy
import pickle

import pytest
import numpy as np
import jax.numpy as jnp

import mellon
from mellon.base_cov import Covariance
from mellon.util import distance


class ModuleLevelCov(Covariance):
    """A user-defined kernel at module level: resolvable, must keep working."""

    def __init__(self, ls=1.0, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.ls = ls

    def k(self, x, y):
        return jnp.exp(-distance(x, y) / self.ls)


def build_closure_local_cov():
    """The factory pattern: the class exists only in this call's locals."""

    class ClosureLocalCov(Covariance):
        def __init__(self, ls=1.0, active_dims=None):
            super().__init__(active_dims=active_dims)
            self.ls = ls

        def k(self, x, y):
            return jnp.exp(-distance(x, y) / self.ls)

    return ClosureLocalCov


def _fit_estimator(cov_func_curry, n=60, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    est = mellon.FunctionEstimator(
        landmarks=X[:20],
        ls=1.0,
        sigma=1.0,
        mu=0.0,
        cov_func_curry=cov_func_curry,
    )
    est.fit(X, y)
    return est, X


# --- fail fast -------------------------------------------------------------


def test_closure_local_covariance_to_json_raises():
    cov = build_closure_local_cov()(ls=1.3)

    with pytest.raises(TypeError) as excinfo:
        cov.to_json()

    message = str(excinfo.value)
    assert (
        "build_closure_local_cov.<locals>.ClosureLocalCov" in message
    ), "The error must name the qualified name that shows where the class lives."
    assert (
        "top level" in message
    ), "The error must say how to fix it: define the class at module level."


def test_closure_local_covariance_to_dict_raises():
    cov = build_closure_local_cov()(ls=1.3)
    with pytest.raises(TypeError):
        cov.to_dict()


def test_closure_local_composite_covariance_raises():
    """Combining a closure-local kernel must not launder it past the check."""
    cov = build_closure_local_cov()(ls=1.3) + mellon.cov.Matern32(1.4)
    with pytest.raises(TypeError) as excinfo:
        cov.to_json()
    assert "ClosureLocalCov" in str(excinfo.value)


def test_predictor_with_closure_local_cov_func_raises(tmp_path):
    est, _ = _fit_estimator(build_closure_local_cov())

    with pytest.raises(TypeError) as excinfo:
        est.predict.to_json(str(tmp_path / "predictor.json.gz"), compress="gzip")

    assert "build_closure_local_cov.<locals>.ClosureLocalCov" in str(excinfo.value)
    assert not list(
        tmp_path.iterdir()
    ), "No unloadable file may be left behind after the refusal."


def test_unresolvable_module_level_class_raises():
    """Not only ``<locals>``: any class its module does not expose is caught."""
    Ghost = type(
        "Ghost",
        (ModuleLevelCov,),
        {"__module__": __name__},  # claims this module, but is never bound here
    )

    with pytest.raises(TypeError) as excinfo:
        Ghost(ls=1.0).to_json()

    message = str(excinfo.value)
    assert "Ghost" in message
    assert "<locals>" not in message, "This is the non-closure branch of the check."


def test_predictor_class_itself_is_checked():
    """The predictor's own class is resolved by name on load, too."""
    est, _ = _fit_estimator(ModuleLevelCov)
    predictor = est.predict

    LocalPredictor = type(
        "LocalPredictor",
        (type(predictor),),
        {"__qualname__": "somewhere.<locals>.LocalPredictor"},
    )
    predictor.__class__ = LocalPredictor

    with pytest.raises(TypeError) as excinfo:
        predictor.to_dict()
    assert "LocalPredictor" in str(excinfo.value)


# --- no regression ---------------------------------------------------------


@pytest.mark.parametrize(
    "cov",
    [
        mellon.cov.Matern32(1.4),
        mellon.cov.Matern52(0.8),
        mellon.cov.ExpQuad(1.1),
        ModuleLevelCov(ls=1.7),
        mellon.cov.Matern32(1.4) + mellon.cov.Exponential(3.4),
        mellon.cov.Matern32(1.4) * mellon.cov.Exponential(3.4),
        mellon.cov.Matern32(1.4) ** 2,
    ],
)
def test_resolvable_covariances_still_round_trip(cov):
    x = jnp.asarray(np.random.default_rng(1).normal(size=(4, 3)))
    values = cov(x, 2 * x)

    revalues = mellon.cov.Covariance.from_json(cov.to_json())(x, 2 * x)

    assert (
        jnp.max(jnp.abs(values - revalues)) == 0.0
    ), "Serialization of a module-level covariance must be exact."


def test_module_level_cov_func_predictor_round_trips_exactly(tmp_path):
    est, X = _fit_estimator(ModuleLevelCov)
    predictor = est.predict
    values = predictor(X)

    path = str(tmp_path / "predictor.json.gz")
    predictor.to_json(path, compress="gzip")
    reloaded = mellon.Predictor.from_json(path)
    revalues = reloaded(X)

    assert type(reloaded.cov_func) is ModuleLevelCov
    assert (
        jnp.max(jnp.abs(values - revalues)) == 0.0
    ), "A module-level covariance function must round-trip bit for bit."


def test_composite_covariance_records_its_full_module():
    cov = mellon.cov.Matern32(1.4) + mellon.cov.Exponential(3.4)

    module_name = cov.to_dict()["metadata"]["module_name"]

    assert module_name == type(cov).__module__, (
        "The recorded module must be the one the class actually lives in, "
        "otherwise a subclass in a package could not be resolved on load."
    )


def test_legacy_composite_state_still_loads():
    """Files written before the module name was recorded in full must load."""
    cov = mellon.cov.Matern32(1.4) + mellon.cov.Exponential(3.4)
    x = jnp.asarray(np.random.default_rng(2).normal(size=(4, 3)))
    values = cov(x, 2 * x)

    legacy_state = cov.to_dict()
    legacy_state["metadata"]["module_name"] = "mellon"  # the pre-fix value

    revalues = mellon.cov.Covariance.from_dict(legacy_state)(x, 2 * x)

    assert jnp.max(jnp.abs(values - revalues)) == 0.0


def test_deepcopy_and_pickle_of_resolvable_covariance_are_unaffected():
    cov = ModuleLevelCov(ls=1.7)
    x = jnp.asarray(np.random.default_rng(3).normal(size=(4, 3)))
    values = cov(x, 2 * x)

    for clone in (copy.deepcopy(cov), pickle.loads(pickle.dumps(cov))):
        assert jnp.max(jnp.abs(clone(x, 2 * x) - values)) == 0.0


def test_deepcopy_of_closure_local_covariance_now_raises():
    """``__getstate__`` is also the copy/pickle hook -- documented, not incidental.

    ``deepcopy`` used to succeed here because it carries the class by reference
    instead of resolving it by name; ``pickle`` already failed on its own. Both
    now raise the same ``TypeError``, and only for classes that were never
    serializable to begin with.
    """
    cov = build_closure_local_cov()(ls=1.3)

    with pytest.raises(TypeError):
        copy.deepcopy(cov)
    with pytest.raises(TypeError):
        pickle.dumps(cov)


def test_predictor_copy_with_closure_local_cov_func_raises_early():
    """It never worked; it now fails at copy() rather than inside from_dict."""
    est, _ = _fit_estimator(build_closure_local_cov())

    with pytest.raises(TypeError):
        est.predict.copy()


def test_copy_is_unaffected_for_resolvable_classes():
    est, X = _fit_estimator(ModuleLevelCov)
    predictor = est.predict

    duplicate = predictor.copy()

    assert jnp.max(jnp.abs(predictor(X) - duplicate(X))) == 0.0
