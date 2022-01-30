from __future__ import annotations

# Core Library
from dataclasses import dataclass

# Third party
import numpy as np
import pandas as pd
from patsy import dmatrices, build_design_matrices
from patsy.design_info import DesignInfo

# First party
from causalipy.custom_types import NDArrayOfFloats


def isa(*types):
    """
    Creates a function checking if its argument
    is of any of given types.
    """
    return lambda x: isinstance(x, types)


def cr_vce(
    X: NDArrayOfFloats,
    eps: NDArrayOfFloats,
    cluster_indicator: NDArrayOfFloats,
    correction: int = 1,
) -> NDArrayOfFloats:
    # Cluster-robust variance estimate
    # Correction: 1 - Liang and Zeger (1986)
    _, k = X.shape
    sorter = cluster_indicator.argsort()
    X_ordered, eps_ordered = X[sorter, :], eps[sorter]
    _, clu_starts = np.unique(cluster_indicator[sorter], return_index=True)
    clu_ends = np.append(clu_starts[1:], len(X))
    b_clu = np.zeros((k, k))
    for clu_start, clu_end in zip(clu_starts, clu_ends):
        X_g = X_ordered[clu_start:clu_end, :]
        eps_g = eps_ordered[clu_start:clu_end] * np.sqrt(correction)
        b_clu += X_g.T @ np.kron(eps_g[:, None], eps_g) @ X_g

    bread = np.linalg.inv(X_ordered.T @ X_ordered)
    return bread @ (b_clu) @ bread


@dataclass
class ClusteredSeConfig:
    cluster: pd.Series
    cluster_correction: bool = True
    small_sample_correct: bool = True


is_clustered_se_config = isa(ClusteredSeConfig)


def hew_vce(X: NDArrayOfFloats, eps: NDArrayOfFloats) -> NDArrayOfFloats:
    # Huber-White Variance-Covariance Matrix
    # TODO: Consider implementing correction factors
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    vce_eps = np.zeros((n, n))
    np.fill_diagonal(vce_eps, eps ** 2)
    correction = n / (n - k)
    return correction * (bread @ (X.T @ vce_eps @ X) @ bread)


def hom_vce(X: NDArrayOfFloats, res: NDArrayOfFloats):
    n, k = X.shape
    inv = np.linalg.inv(X.T @ X)
    res_var = res @ res / (n - k)
    return inv * res_var


def _solve_ols(y: NDArrayOfFloats, X: NDArrayOfFloats) -> tuple[NDArrayOfFloats, NDArrayOfFloats]:
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    res = y - X @ coeffs
    return coeffs, res


class Ols:
    def __init__(self, formula: str, data: pd.DataFrame):
        y_dmat, X_dmat = dmatrices(formula, data=data)
        self._x_dmat, self._y_dmat = X_dmat, y_dmat
        self.n_obs, self.n_vars = X_dmat.shape

        self.coefficients, self._res = _solve_ols(y_dmat, X_dmat)
        self._x_design_info: DesignInfo = X_dmat.design_info

        self.residuals = X_dmat @ self.coefficients

        self.hessian = X_dmat.T @ X_dmat
        self.score = (y_dmat - self.predict(data)) * X_dmat

        self.hessian_inv = np.linalg.inv(self.hessian)

        self.asymptotic_linear_representation = self.score @ self.hessian_inv

    def get_design_matrix(self, X: pd.DataFrame) -> NDArrayOfFloats:
        return build_design_matrices([self._x_design_info], X)[0]

    def predict(self, data: pd.DataFrame) -> NDArrayOfFloats:
        return (self.get_design_matrix(data) @ self.coefficients).reshape(-1, 1)

    def _hom_vce(self):
        inv = np.linalg.inv(self._x_dmat.T @ self._x_dmat)
        res_var = self._res.T @ self._res / (self.n_obs - self.n_vars)
        return inv * res_var

    def _cr_vce(self, config: ClusteredSeConfig) -> NDArrayOfFloats:
        cluster_indicator = config.cluster
        correction_small_sample = (
            (self.n_obs - 1) / (self.n_obs - self.n_vars) if config.small_sample_correct else 1
        )
        # NUNIQUE will not work on NDArrayOfFloats
        correction_cluster = (
            (cluster_indicator.nunique() / (cluster_indicator.nunique() - 1))
            if config.cluster_correction
            else 1
        )
        # Cluster-robust variance estimate
        sorter = cluster_indicator.argsort()
        X_ordered, eps_ordered = self._x_dmat[sorter, :], self._res[sorter]
        _, clu_starts = np.unique(cluster_indicator[sorter], return_index=True)
        clu_ends = np.append(clu_starts[1:], len(self._x_dmat))
        b_clu = np.zeros((self.n_vars, self.n_vars))
        for clu_start, clu_end in zip(clu_starts, clu_ends):
            X_g = X_ordered[clu_start:clu_end, :]
            eps_g = eps_ordered[clu_start:clu_end]
            b_clu += X_g.T @ (eps_g @ eps_g.T) @ X_g

        bread = np.linalg.inv(X_ordered.T @ X_ordered)
        return (bread @ (b_clu) @ bread) * correction_cluster * correction_small_sample

    def get_standard_errors(self, config: None | ClusteredSeConfig = None) -> NDArrayOfFloats:

        if is_clustered_se_config(config):
            vcm = self._cr_vce(config)  # type: ignore
        else:
            vcm = self._hom_vce()

        return np.sqrt(np.diag(vcm)).reshape(-1, 1)
