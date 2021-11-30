from __future__ import annotations

# Core Library
from functools import cached_property

# Third party
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# First party
from metrics.logistic_regression import LogisticRegression


class IpwDid:
    def _influence_function_treatment_component(self, data: pd.DataFrame) -> NDArray:
        return (
            data["treatment_status"]
            * (
                data["outcome"]
                - (data["treatment_status"] * data["outcome"]).mean()
                / data["treatment_status"].mean()
            )
        ).values.reshape(-1, 1) / data["treatment_status"].mean()

    def _influence_function_control_component_1(self, data: pd.DataFrame, odds: NDArray) -> NDArray:
        control_group = (1 - data["treatment_status"].values.reshape(-1, 1)) * odds
        return control_group * (
            data["outcome"].values.reshape(-1, 1)
            - (
                (data["outcome"].values.reshape(-1, 1) * control_group).mean()
                / control_group.mean()
            )
        )

    def _influence_function_control_component_2(self, data: pd.DataFrame, odds: NDArray) -> NDArray:
        alr = self.selection_model.score @ self.selection_model.vce
        control_weights = (1 - data["treatment_status"].values.reshape(-1, 1)) * odds
        control_component = (
            data["outcome"].values.reshape(-1, 1) * control_weights
        ).mean() / control_weights.mean()
        m2 = (
            control_weights
            * (data["outcome"].values.reshape(-1, 1) - control_component)
            * self.selection_model.get_design_matrix(data)
        )
        return (alr @ m2.mean(axis=0)).reshape(-1, 1)

    def _influence_function_control_component(self, data: pd.DataFrame, odds: NDArray) -> NDArray:
        control_weights = (1 - data["treatment_status"].values.reshape(-1, 1)) * odds
        return (
            self._influence_function_control_component_1(data, odds)
            + self._influence_function_control_component_2(data, odds)
        ) / control_weights.mean()

    def _compute_influence_function(self, data, odds) -> NDArray:
        return self._influence_function_treatment_component(
            data
        ) - self._influence_function_control_component(data, odds)

    def __init__(self, formula: str, data: pd.DataFrame):
        # Remark: give in a dataframe with two time periods
        max_time_period = data["time_period"].max()  # noqa
        min_time_period = data["time_period"].min()  # noqa

        # TODO: solve this better
        data = data.set_index("id")

        # Estimate the selection model on the data of the later time period.
        data_selection_model = data.query("time_period == @max_time_period")
        self.selection_model = LogisticRegression(formula, data_selection_model)

        data = data.query("time_period == @max_time_period").assign(
            outcome=lambda df: df["outcome"].values
            - data.loc[data["time_period"] == min_time_period, "outcome"].values
        )
        self.n_units = data_selection_model.shape[0]

        self._att = self._compute_att(data)

        odds = self.selection_model.predict_odds(data)
        self._influence_function = self._compute_influence_function(data, odds)

    def _compute_att(self, data: pd.DataFrame) -> float:
        data_control = data.query("treatment_status==0")
        data_treated = data.query("treatment_status==1")
        odds = self.selection_model.predict_odds(data_control)
        return (
            data_treated["outcome"].values.reshape(-1, 1).mean()
            - ((data_control["outcome"].values.reshape(-1, 1) * odds).mean()) / odds.mean()
        )

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
    ipw_did = IpwDid("treatment_status ~ control", data)
    print(ipw_did.att, ipw_did.standard_errors)
