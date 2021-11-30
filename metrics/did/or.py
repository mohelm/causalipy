from __future__ import annotations

# Core Library
from functools import cached_property

# Third party
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# First party
from metrics.ols import Ols


class OrDid:
    def _compute_att(self, data_treated: pd.DataFrame) -> float:
        return (
            data_treated["outcome"].values.reshape(-1, 1) - self.or_model.predict(data_treated)
        ).mean()

    def _influence_function_treatment_component(self, data: pd.DataFrame) -> NDArray:
        mean = (data["outcome"] * data["treatment_status"]).mean() / data["treatment_status"].mean()
        return (
            data["treatment_status"] * (data["outcome"] - mean) / data["treatment_status"].mean()
        ).values.reshape(-1, 1)

    def _influence_function_control_component_1(self, data: pd.DataFrame) -> NDArray:
        preds = self.or_model.predict(data)
        mean = (preds * data["treatment_status"].values.reshape(-1, 1)).mean() / (
            data["treatment_status"]
        ).mean()
        return (data["treatment_status"].values.reshape(-1, 1)) * (preds - mean)

    def _influence_function_control_component_2(self, data: pd.DataFrame) -> NDArray:
        X = self.or_model.get_design_matrix(data)
        n_treatment = (data["treatment_status"] == 1).sum()
        # TODO: careful here: you are relying on a certain ordering of treatment and control obs in the dta
        # I thought there was also another issue but i forgot.
        alr_or = (
            len(data)
            * np.r_[
                np.zeros((n_treatment, X.shape[1])), self.or_model.asymptotic_linear_representation
            ]
        )
        return (alr_or @ (X * data["treatment_status"].values.reshape(-1, 1)).mean(axis=0)).reshape(
            -1, 1
        )

    def _influence_function_control_component(self, data: pd.DataFrame) -> NDArray:
        return (
            self._influence_function_control_component_1(data)
            + self._influence_function_control_component_2(data)
        ) / (data["treatment_status"]).mean()

    def _compute_influence_function(self, data) -> NDArray:
        return self._influence_function_treatment_component(
            data
        ) - self._influence_function_control_component(data)

    def __init__(self, formula: str, data: pd.DataFrame):

        # Remark: give in a dataframe with two time periods
        max_time_period = data["time_period"].max()  # noqa
        min_time_period = data["time_period"].min()  # noqa

        data = data.query("time_period == @max_time_period").assign(
            outcome=lambda df: df["outcome"].values
            - data.loc[data["time_period"] == min_time_period, "outcome"].values
        )
        self.n_obs = data.shape[0]

        data_control = data.query("treatment_status==0")
        self.or_model = Ols(formula, data_control)

        # Treatment effect
        data_treated = data.query("treatment_status==1")
        self._att = self._compute_att(data_treated)

        self._influence_function = self._compute_influence_function(data)

    @cached_property
    def standard_errors(self) -> float:
        return np.std(self._influence_function) / np.sqrt(self.n_obs)

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
    or_did = OrDid("outcome ~ control", data)
    print(or_did.att, or_did.standard_errors)
