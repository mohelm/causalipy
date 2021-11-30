from __future__ import annotations

# Core Library
from functools import cached_property

# Third party
import numpy as np
import pandas as pd

# First party
from metrics.ols import Ols


class OrDid:
    def _compute_att(self, data_treated: pd.DataFrame) -> float:
        return (
            data_treated["outcome"].values.reshape(-1, 1) - self.or_model.predict(data_treated)
        ).mean()

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
        n_control = data_control.shape[0]
        self.or_model = Ols(formula, data_control)

        treatment_prop = data["treatment_status"].mean()

        # Treatment effect
        data_treated = data.query("treatment_status==1")
        self._att = self._compute_att(data_treated)

        # Compute the influece functon
        alr_or = self.or_model.asymptotic_linear_representation

        n_treated = data_treated.shape[0]
        outcome_treated = data_treated["outcome"].values.reshape(-1, 1)
        self._influence_function_treat_other = (
            outcome_treated - outcome_treated.mean()
        ) / treatment_prop
        outcome_control = self.or_model.predict(data_treated)

        self._influence_function = (
            np.r_[self._influence_function_treat_other, np.zeros((n_control, 1))]
            - (
                np.r_[outcome_control - outcome_control.mean(), np.zeros((n_control, 1))]
                + np.r_[
                    np.zeros((n_treated, 1)),
                    (
                        len(data)
                        * alr_or
                        @ (
                            self.or_model.get_design_matrix(data_treated).mean(axis=0)
                            * treatment_prop
                        )
                    ).reshape(-1, 1),
                ]
            )
            / treatment_prop
        )

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
