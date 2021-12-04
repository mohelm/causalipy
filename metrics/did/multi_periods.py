from __future__ import annotations

# Core Library
from itertools import product

# Third party
import numpy as np
import pandas as pd

# First party
from metrics.did.doubly_robust import DoublyRobustDid


class MultiPeriodDid:
    def __init__(self, formula_or: str | None, formula_ipw: str | None, data: pd.DataFrame):
        groups = np.sort(data.group.unique())[1:]
        time_periods = np.sort(data.time_period.unique())

        combinations = ((g, t) for (g, t) in product(groups, time_periods) if g <= t)

        def _estimate_model(group: int, time: int) -> DoublyRobustDid:
            current_data = data.query("group in (0,@group)  and time_period in (@group-1,@time)")
            return DoublyRobustDid(formula_or, formula_ipw, current_data)

        self._estimates = {(g, t): _estimate_model(g, t) for g, t in combinations}

    def atts(self) -> pd.Series:
        return pd.Series(
            [est.att for est in self._estimates.values()],
            index=pd.MultiIndex.from_tuples(self._estimates.keys(), names=["group", "time_period"]),
            name="att",
        )

    def standard_errors(self) -> pd.Series:
        return pd.Series(
            [est.standard_errors() for est in self._estimates.values()],
            index=pd.MultiIndex.from_tuples(self._estimates.keys(), names=["group", "time_period"]),
            name="s.e.",
        )

    def _summary(self) -> pd.DataFrame:
        return pd.concat((self.atts(), self.standard_errors()), axis=1)

    def summary(self) -> pd.DataFrame:
        return self._summary()


if __name__ == "__main__":

    # data = pd.read_feather("/Users/moritz.helm/.cache/py-did/sim-ds.feather").rename(
    #     columns={
    #         "X": "control",
    #         "treat": "treatment_status",
    #         "period": "time_period",
    #         "G": "group",
    #         "Y": "outcome",
    #     }
    # )
    # for col in ["treatment_status", "time_period", "group"]:
    #     data[col] = data[col].astype(int)

    # mpd = MultiPeriodDid("outcome~control", "treatment_status ~ control", data)

    # MPDATA data
    data = pd.read_feather("/Users/moritz.helm/.cache/py-did/mpdta.feather").rename(
        columns={
            "treat": "treatment_status",
            "lemp": "outcome",
            "year": "time_period",
            "first.treat": "group",
            "lpop": "control",
            "countyreal": "id",
        }
    )
    for col in ["treatment_status", "time_period", "group", "id"]:
        data[col] = data[col].astype(int)

    mpd_minimum_wage = MultiPeriodDid("outcome ~ 1", "treatment_status ~ 1", data)
    print(mpd_minimum_wage.summary())
