from __future__ import annotations

# Core Library
from enum import Enum
from typing import Union, cast

# Third party
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# First party
from metrics.ols import Ols
from metrics.custom_types import NDArrayOfFloats
from metrics.logistic_regression import LogisticRegression

MaybeNDArrayOfFloats = Union[NDArrayOfFloats, None]
MaybeInt = Union[int, None]


def _prepare_data(data: pd.DataFrame, late_period: int):
    return data.query("time_period == @late_period").assign(
        outcome=lambda df: df["outcome"].values
        - data.loc[data["time_period"] != late_period, "outcome"].values
    )


class Method(str, Enum):
    dr = "doubly_robust"
    ipw = "inverse_probability_weighting"
    oreg = "outcome_regression"


def _assign_estimation_method(formula_or: str | None, formula_ipw: str | None) -> Method:
    if formula_or and formula_ipw:
        return Method.dr
    elif formula_or:
        return Method.oreg
    else:
        return Method.ipw


class DoublyRobustDid:
    def _get_control_diff_in_diff(
        self, outcome: NDArray[np.float_], preds: NDArray[np.float_] | None
    ) -> NDArray[np.float_]:
        if self.method == Method.dr:
            preds = cast(NDArray[np.float_], preds)
            return outcome - preds
        if self.method == Method.oreg:
            preds = cast(NDArray[np.float_], preds)
            return preds
        return outcome

    def _get_treatment_diff_in_diff(
        self, outcome: NDArray[np.float_], preds: NDArray[np.float_] | None
    ) -> NDArray[np.float_]:
        if self.method == Method.dr:
            preds = cast(NDArray[np.float_], preds)
            return outcome - preds
        if self.method == Method.oreg:
            return outcome
        return outcome

    def _get_treatment_if(
        self, X: NDArray[np.float_] | None, n_treated: int | None
    ) -> NDArray[np.float_]:
        component1 = self._att_treated - self._weights_treated * self._eta_treated
        if self.method != Method.dr:
            return component1 / self._weights_treated.mean()

        X = cast(NDArray[np.float_], X)
        n_treated = cast(int, n_treated)
        or_model = cast(Ols, self.or_model)
        # TODO: careful here, you assume a certain structure of data.
        alr_or = (
            len(X)
            * np.r_[np.zeros((n_treated, X.shape[1])), or_model.asymptotic_linear_representation]
        )
        return (
            component1 - (alr_or @ (self._weights_treated * X).mean(axis=0)).reshape(-1, 1)
        ) / self._weights_treated.mean()

    def _get_control_if(
        self,
        outcome: NDArrayOfFloats,
        preds: MaybeNDArrayOfFloats,
        X: NDArrayOfFloats,
        n_treated: int,
    ) -> NDArray[np.float_]:
        component1 = self._att_control - self._weights_control * self._eta_control

        if self.method != Method.ipw:
            alr_or = (
                len(X)
                * np.r_[
                    np.zeros((n_treated, X.shape[1])),
                    self.or_model.asymptotic_linear_representation,
                ]
            )
            component_2 = alr_or @ ((self._weights_control * X).mean(axis=0)).reshape(-1, 1)

        if self.method != Method.oreg:
            alr = self.selection_model.score @ self.selection_model.vce
            inner = outcome - self._eta_control
            inner = inner if self.method == Method.ipw else inner - preds
            m2 = (self._weights_control * inner * X).mean(axis=0)
            component_3 = (alr @ m2).reshape(-1, 1)

        if self.method == Method.dr:
            return (component1 + component_3 - component_2) / self._weights_control.mean()

        if self.method == Method.oreg:
            return (component1 + component_2) / self._weights_control.mean()
        return (component1 + component_3) / self._weights_control.mean()

    def _get_if(
        self,
        outcome: NDArray[np.float_],
        preds: MaybeNDArrayOfFloats,
        X: NDArrayOfFloats,
        n_treated: int,
    ) -> NDArray[np.float_]:
        return self._get_treatment_if(X, n_treated) - self._get_control_if(
            outcome, preds, X, n_treated
        )

    def __init__(self, formula_or: str | None, formula_ipw: str | None, data: pd.DataFrame):

        self.method = _assign_estimation_method(formula_or, formula_ipw)

        # Prepare data
        late_period = data["time_period"].max()
        data = _prepare_data(data, late_period)
        n_treated = data.query("treatment_status==1").shape[0]
        outcome: NDArray[np.float_] = data["outcome"].values.reshape(-1, 1)
        self.n_units = data.shape[0]

        self.selection_model = LogisticRegression(formula_ipw, data) if formula_ipw else None
        self.or_model = Ols(formula_or, data.query("treatment_status==0")) if formula_or else None
        preds: MaybeNDArrayOfFloats = self.or_model.predict(data) if self.or_model else None

        # TODO: why is the design matrix identical?
        X = (
            self.or_model.get_design_matrix(data)
            if self.or_model
            else self.selection_model.get_design_matrix(data)
        )

        self._weights_treated = data["treatment_status"].values.reshape(-1, 1)
        self._weights_control = (
            self.selection_model.predict_odds(data) * (1 - self._weights_treated)
            if formula_ipw
            else self._weights_treated
        )
        self._att_treated = self._weights_treated * self._get_treatment_diff_in_diff(outcome, preds)
        self._att_control = self._weights_control * self._get_control_diff_in_diff(outcome, preds)
        self._eta_control = self._att_control.mean() / self._weights_control.mean()
        self._eta_treated = self._att_treated.mean() / self._weights_treated.mean()

        self._influence_function = self._get_if(outcome, preds, X, n_treated)

    @property
    def att(self) -> float:
        return self._eta_treated - self._eta_control

    def standard_errors(self) -> float:
        return np.std(self._influence_function) / np.sqrt(self.n_units)


if __name__ == "__main__":
    data = pd.read_feather("/Users/moritz.helm/.cache/py-did/sim-ds.feather").rename(
        columns={
            "X": "control",
            "treat": "treatment_status",
            "period": "time_period",
            "G": "group",
            "Y": "outcome",
        }
    )
    for col in ["treatment_status", "time_period", "group"]:
        data[col] = data[col].astype(int)

    data = data.query("time_period in (3, 4)")
    dr_did = DoublyRobustDid("outcome~control", "treatment_status ~ control", data)
    print(dr_did.att, dr_did.standard_errors())

    ipw_did = DoublyRobustDid(None, "treatment_status ~ control", data)
    print(ipw_did.att, ipw_did.standard_errors())

    or_did = DoublyRobustDid("outcome~control", None, data)
    print(or_did.att, or_did.standard_errors())
