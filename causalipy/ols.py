# Third party
import numpy as np
import pandas as pd
from patsy import dmatrices, build_design_matrices
from patsy.design_info import DesignInfo

# First party
from causalipy.custom_types import NDArrayOfFloats


def _solve_ols(y: NDArrayOfFloats, X: NDArrayOfFloats) -> tuple[NDArrayOfFloats, NDArrayOfFloats]:
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    res = y - X @ coeffs
    return coeffs, res


class Ols:
    def __init__(self, formula: str, data: pd.DataFrame):
        y_dmat, X_dmat = dmatrices(formula, data=data)

        self.coefficients, _ = _solve_ols(y_dmat, X_dmat)
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
