from __future__ import annotations

# Core Library
from functools import cached_property

# Third party
import numpy as np
import pandas as pd

# First party
from metrics.logistic_regression import LogisticRegression


class IpwDid:
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
        self.n_obs = data.shape[0]
        data_control = data.query("treatment_status==0")
        data_treated = data.query("treatment_status==1")
        self.n_units = data_selection_model.shape[0]

        self._att = self._compute_att(data_treated, data_control)

        # Variance
        alr_ps = self.selection_model.score @ self.selection_model.vce
        outcome_treated = data_treated["outcome"].values.reshape(-1, 1)
        outcome_control = data_control["outcome"].values.reshape(-1, 1)

        influence_function_treat_other = (outcome_treated - outcome_treated.mean()) / data[
            "treatment_status"
        ].mean()
        n_control = data_control.shape[0]
        odds = self.selection_model.predict_odds(data_control)

        self._influence_function = np.r_[
            influence_function_treat_other, np.zeros((n_control, 1))
        ] - (
            (
                np.r_[
                    np.zeros((data_treated.shape[0], 1)),
                    odds * outcome_control - odds * (odds * outcome_control).mean() / odds.mean(),
                ]
                + (
                    alr_ps
                    @ np.r_[
                        np.zeros((data_treated.shape[0], 2)),
                        (
                            (outcome_control - ((odds * outcome_control).mean() / odds.mean()))
                            * odds
                            * self.selection_model.get_design_matrix(data_control)
                        ),
                    ].mean(axis=0)
                ).reshape(-1, 1)
            )
            / np.r_[np.zeros((data_treated.shape[0], 1)), odds].mean()
        )

    def _compute_att(self, data_treated: pd.DataFrame, data_control: pd.DataFrame) -> float:
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
