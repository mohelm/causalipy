# Third party
import numpy as np
import pandas as pd
from patsy import dmatrices, build_design_matrices
from numpy.typing import NDArray
from patsy.design_info import DesignInfo


class OlsResults:
    def __init__(
        self, coeffs: NDArray[np.float_], eps: NDArray[np.float_], x_design_info: DesignInfo
    ):
        self._coeffs = coeffs
        self._eps = eps
        self._x_design_info = x_design_info

    def design_matrix(self, X: pd.DataFrame) -> NDArray[np.float_]:
        return build_design_matrices([self._x_design_info], X)

    def predict(self, X: pd.DataFrame) -> NDArray[np.float_]:
        return (self.design_matrix(X) @ self._coeffs).reshape(-1, 1)


def _ols(
    y: NDArray[np.float_], X: NDArray[np.float_]
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    y = y[:, None] if y.ndim == 1 else y
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    res = y - X @ coeffs
    return coeffs, res


def ols(formula: str, data: pd.DataFrame) -> OlsResults:
    y_dmat, X_dmat = dmatrices(formula, data=data)
    return OlsResults(*_ols(y_dmat, X_dmat), X_dmat.design_info)  # type: ignore
