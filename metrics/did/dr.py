from __future__ import annotations

# Core Library
from functools import cached_property

# Third party
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# First party
from metrics.ols import Ols
from metrics.logistic_regression import LogisticRegression


class DrDid:
    @property
    def _influence_function_treatment_component_1(self) -> NDArray[np.float_]:
        return self._att_treated - self._weights_treated * self._eta_treated

    def _influence_function_treatment_component_2(self, data: pd.DataFrame) -> NDArray[np.float_]:
        X = self.or_model.get_design_matrix(data)
        n_treatment = (data["treatment_status"] == 1).sum()
        # TODO: careful here, you assume a certain structure of data.
        alr_or = (
            len(data)
            * np.r_[
                np.zeros((n_treatment, X.shape[1])), self.or_model.asymptotic_linear_representation
            ]
        )
        return (alr_or @ (self._weights_treated * X).mean(axis=0)).reshape(-1, 1)

    def _influence_function_treatment_component(self, data: pd.DataFrame) -> NDArray[np.float_]:
        return (
            self._influence_function_treatment_component_1
            - self._influence_function_treatment_component_2(data)
        ) / self._weights_treated.mean()

    @property
    def _influence_function_control_component_1(self) -> NDArray[np.float_]:
        return self._att_control - self._weights_control * self._eta_control

    def _influence_function_control_component_2(self, data: pd.DataFrame) -> NDArray[np.float_]:
        alr = self.selection_model.score @ self.selection_model.vce
        X = self.or_model.get_design_matrix(data)
        m2 = (
            self._weights_control
            * (
                data["outcome"].values.reshape(-1, 1)
                - self.or_model.predict(data)
                - self._eta_control
            )
            * X
        ).mean(axis=0)
        return (alr @ m2).reshape(-1, 1)

    def _influence_function_control_component_3(self, data: pd.DataFrame) -> NDArray[np.float_]:
        X = self.or_model.get_design_matrix(data)
        n_treatment = (data["treatment_status"] == 1).sum()
        alr_or = (
            len(data)
            * np.r_[
                np.zeros((n_treatment, X.shape[1])), self.or_model.asymptotic_linear_representation
            ]
        )
        return (alr_or @ (self._weights_control * X).mean(axis=0)).reshape(-1, 1)

    def _influence_function_control_component(self, data: pd.DataFrame) -> NDArray[np.float_]:
        return (
            self._influence_function_control_component_1
            + self._influence_function_control_component_2(data)
            - self._influence_function_control_component_3(data)
        ) / self._weights_control.mean()

    def _compute_influence_function(self, data) -> NDArray[np.float_]:
        return self._influence_function_treatment_component(
            data
        ) - self._influence_function_control_component(data)

    def __init__(self, formula_or: str, formula_ipw: str, data: pd.DataFrame):
        # Remark: give in a dataframe with two time periods
        max_time_period = data["time_period"].max()  # noqa
        min_time_period = data["time_period"].min()  # noqa

        # TODO: solve this better
        data = data.set_index("id")

        # Estimate the selection model on the data of the later time period.
        data_selection_model = data.query("time_period == @max_time_period")
        self.selection_model = LogisticRegression(formula_ipw, data_selection_model)

        data = data.query("time_period == @max_time_period").assign(
            outcome=lambda df: df["outcome"].values
            - data.loc[data["time_period"] == min_time_period, "outcome"].values
        )
        self.n_units = data_selection_model.shape[0]
        self.or_model = Ols(formula_or, data.query("treatment_status==0"))

        self._weights_treated = data["treatment_status"].values.reshape(-1, 1)
        self._weights_control = self.selection_model.predict_odds(data) * (
            1 - self._weights_treated
        )

        # TODO: Naming?
        outcome = data["outcome"].values.reshape(-1, 1)
        diff_in_diff = outcome - self.or_model.predict(data)
        self._att_treated = self._weights_treated * diff_in_diff
        self._att_control = self._weights_control * diff_in_diff

        self._eta_control = self._att_control.mean() / self._weights_control.mean()
        self._eta_treated = self._att_treated.mean() / self._weights_treated.mean()
        self._influence_function = self._compute_influence_function(data)

        self._att = self._compute_att(data)

    def _compute_att(self, data: pd.DataFrame) -> float:
        treated_weights = data["treatment_status"].values.reshape(-1, 1)
        control_weights = self.selection_model.predict_odds(data) * (1 - treated_weights)
        treat = treated_weights * (
            data["outcome"].values.reshape(-1, 1) - self.or_model.predict(data)
        )
        control = control_weights * (
            data["outcome"].values.reshape(-1, 1) - self.or_model.predict(data)
        )
        return treat.mean() / treated_weights.mean() - control.mean() / control_weights.mean()

    @cached_property
    def standard_errors(self) -> float:
        return np.std(self._influence_function) / np.sqrt(self.n_units)

    @cached_property
    def att(self) -> float:
        return self._att


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
    dr_did = DrDid("outcome~control", "treatment_status ~ control", data)
    print(dr_did.att, dr_did.standard_errors)
