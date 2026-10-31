"""Narrow, offline scikit-learn compatibility for the unmodified course.

Legacy Boston data is shipped from scikit-learn 1.1.3, not fetched at runtime.
The original dataset was deprecated for well-documented ethical concerns;
this adapter exists only so the historical course notebook remains executable.
"""
from __future__ import annotations

from pathlib import Path


def install_boston():
    import numpy as np
    import sklearn.datasets as datasets
    from sklearn.utils import Bunch

    if "load_boston" in datasets.__dict__:
        return

    csv_file = Path(__file__).parent / "legacy_data" / "boston_house_prices.csv"
    features = np.array([
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
        "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ])
    def load_boston(*, return_X_y=False):
        values = np.loadtxt(csv_file, delimiter=",", skiprows=2, dtype=np.float64)
        if values.shape != (506, 14):
            raise RuntimeError(f"Invalid bundled Boston dataset: {values.shape}")
        data, target = values[:, :13].copy(), values[:, 13].copy()
        if return_X_y:
            return data, target
        return Bunch(
            data=data, target=target, feature_names=features.copy(),
            filename=str(csv_file),
            DESCR="Historical Boston house-prices dataset; kept solely for legacy coursework."
        )
    datasets.load_boston = load_boston


def install_linear_regression():
    import inspect
    import sklearn.linear_model as linear
    Original = linear.LinearRegression
    if getattr(Original, "_ml_legacy_normalize", False):
        return

    class LinearRegression(Original):
        """scikit-learn LinearRegression with deprecated normalize=False accepted."""

        _ml_legacy_normalize = True

        def __init__(self, *, fit_intercept=True, copy_X=True, tol=1e-6,
                     n_jobs=None, positive=False, normalize=False):
            if normalize not in (False, None):
                raise ValueError(
                    "normalize=True is not supported; use a preprocessing scaler"
                )
            self.normalize = normalize
            self.tol = tol
            kwargs = dict(fit_intercept=fit_intercept, copy_X=copy_X,
                          n_jobs=n_jobs, positive=positive)
            if "tol" in inspect.signature(Original.__init__).parameters:
                kwargs["tol"] = tol
            super().__init__(**kwargs)

    LinearRegression.__name__ = "LinearRegression"
    LinearRegression.__qualname__ = "LinearRegression"
    LinearRegression.__module__ = "sklearn.linear_model._base"
    import sklearn.linear_model._base as base
    base.LinearRegression = LinearRegression
    linear.LinearRegression = LinearRegression

def install_ridge():
    """Accept Ridge(normalize=True) with legacy feature L2 normalization.

    Old sklearn's normalize=True centered X and scaled each feature to unit
    L2 norm before solving Ridge, then translated coefficients back to the
    original coordinate system. Ignoring normalize would change the fitted
    regularized model; emulate the scale rather than dropping the argument.
    """
    import inspect
    import numpy as np
    import sklearn.linear_model as linear

    Original = linear.Ridge
    if getattr(Original, "_ml_legacy_normalize", False):
        return

    class Ridge(Original):
        _ml_legacy_normalize = True

        def __init__(self, alpha=1.0, *, fit_intercept=True, copy_X=True,
                     max_iter=None, tol=1e-4, solver="auto", positive=False,
                     random_state=None, normalize=False):
            self.normalize = normalize
            kwargs = dict(alpha=alpha, fit_intercept=fit_intercept,
                          copy_X=copy_X, max_iter=max_iter, tol=tol,
                          solver=solver, positive=positive,
                          random_state=random_state)
            supported = inspect.signature(Original.__init__).parameters
            super().__init__(**{key: value for key, value in kwargs.items()
                                if key in supported})

        def fit(self, X, y, sample_weight=None):
            if not self.normalize or not self.fit_intercept:
                return super().fit(X, y, sample_weight=sample_weight)
            if self.normalize is not True:
                raise ValueError("normalize must be True or False")
            values = np.asarray(X, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("Expected a two-dimensional feature matrix")
            if sample_weight is None:
                means = values.mean(axis=0)
                scale = np.linalg.norm(values - means, axis=0)
            else:
                weights = np.asarray(sample_weight, dtype=np.float64)
                if (weights.shape != (values.shape[0],) or
                        not np.all(np.isfinite(weights)) or
                        np.any(weights < 0) or weights.sum() <= 0):
                    raise ValueError("Invalid Ridge sample_weight")
                means = np.average(values, axis=0, weights=weights)
                scale = np.sqrt(np.sum((values - means) ** 2 *
                                       weights[:, None], axis=0))
            scale[scale == 0] = 1.0  # Old sklearn left constant columns alone.
            result = super().fit(values / scale, y, sample_weight=sample_weight)
            self.coef_ = np.asarray(self.coef_) / scale
            # The intercept learned in normalized coordinates already equals
            # y_offset - X_offset @ coef_ on the original feature scale.
            return result

    Ridge.__name__ = "Ridge"
    Ridge.__qualname__ = "Ridge"
    Ridge.__module__ = "sklearn.linear_model._ridge"
    import sklearn.linear_model._ridge as ridge_module
    ridge_module.Ridge = Ridge
    linear.Ridge = Ridge
