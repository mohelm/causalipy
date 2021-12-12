# Core Library
from functools import cached_property

# Third party
import numpy as np
import pandas as pd
from patsy import dmatrices, build_design_matrices
from scipy.optimize import minimize
from patsy.design_info import DesignInfo
from scipy.optimize.optimize import OptimizeResult

# First party
from causalipy.custom_types import NDArrayOfFloats


def _logistic(z: NDArrayOfFloats) -> NDArrayOfFloats:
    return 1 / (1 + np.exp(-z))


def _lr_objective(coeffs, y, X, return_hess=False):
    n, _ = X.shape
    inner = _logistic(X @ coeffs).reshape(-1, 1)
    jac = (X.T @ (y.squeeze().reshape(-1, 1) - inner)).squeeze()
    ll = (y * np.log(inner) + (1 - y) * np.log(1 - inner)).sum()

    if not return_hess:
        return -ll, -jac

    diag = inner * (1 - inner)
    W = np.zeros((n, n))
    np.fill_diagonal(W, diag)
    hess = -(X.T @ W @ X)
    return -ll, -jac, hess


def _lr(y: NDArrayOfFloats, X: NDArrayOfFloats, **kwargs) -> OptimizeResult:
    # Simple function to solve logistic regression - kinda slow.
    # TODO: Test this on simulated data - maybe data can be within the tol
    n, k = X.shape

    def _ll(coeffs: NDArrayOfFloats) -> float:
        inner = _logistic(X @ coeffs).reshape(-1, 1)
        ll = (y * np.log(inner) + (1 - y) * np.log(1 - inner)).sum()
        return -ll

    def _jacobian(coeffs: NDArrayOfFloats) -> NDArrayOfFloats:
        z = X @ coeffs
        jac = (X.T @ (y.squeeze() - _logistic(z))).squeeze()
        return -jac

    def _hessian(coeffs: NDArrayOfFloats) -> NDArrayOfFloats:
        z = X @ coeffs
        diag = _logistic(z) * (1 - _logistic(z))
        W = np.zeros((n, n))
        np.fill_diagonal(W, diag)

        return -(X.T @ W @ X)

    return minimize(
        _ll,
        np.zeros(k),
        jac=_jacobian,
        hess=_hessian,
        # method="trust-ncg",
        method="L-BFGS-B",
        # method="Newton-CG",
        options={"maxiter": 1000, "disp": False},
        **kwargs,
    )


class LogisticRegression:
    def __init__(self, formula: str, data: pd.DataFrame):
        y_dmat, X_dmat = dmatrices(formula, data=data)
        self._x_design_info: DesignInfo = X_dmat.design_info

        # Estimate the model
        res = _lr(y_dmat, X_dmat)

        # TODO: here you have some code duplication
        neg_ll, neg_jac, neg_hess = _lr_objective(res.x, y_dmat, X_dmat, return_hess=True)

        self.coefficients = res.x
        self.log_likelihood = -neg_ll
        self.jacobian = -neg_jac
        self.hessian = -neg_hess
        self.score = (y_dmat - self.predict_proba(data)) * X_dmat

        self.n_observations = X_dmat.shape[0]

    def predict_proba(self, X: pd.DataFrame) -> NDArrayOfFloats:
        return _logistic(self.get_design_matrix(X) @ self.coefficients).reshape(-1, 1)

    def predict_odds(self, X: pd.DataFrame) -> NDArrayOfFloats:
        probabilities = self.predict_proba(X)
        return (probabilities / (1 - probabilities)).reshape(-1, 1)

    @cached_property
    def vce(self) -> NDArrayOfFloats:
        # TODO: figure out if this is the only VCE
        return np.linalg.inv(self.hessian) * self.n_observations

    def get_design_matrix(self, X: pd.DataFrame) -> NDArrayOfFloats:
        return build_design_matrices([self._x_design_info], X)[0]
