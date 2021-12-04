from __future__ import annotations

# Core Library
from itertools import product

# Third party
import numpy as np
import pandas as pd
from scipy.stats import norm

# First party
from metrics.did.doubly_robust import DoublyRobustDid


class MultiPeriodDid:
    def __init__(self, formula_or: str | None, formula_ipw: str | None, data: pd.DataFrame):
        groups = np.sort(data.group.unique())[1:]
        time_periods = np.sort(data.time_period.unique())[1:]

        combinations = ((g, t) for (g, t) in product(groups, time_periods))

        def _estimate_model(group: int, time: int) -> DoublyRobustDid:
            early_tp = group - 1 if group <= time else time - 1  # noqa
            current_data = data.query("group in (0,@group)  and time_period in (@early_tp,@time)")
            return DoublyRobustDid(formula_or, formula_ipw, current_data)

        self._estimates = {(g, t): _estimate_model(g, t) for g, t in combinations}

    def atts(self) -> pd.Series:
        return pd.Series(
            [est.att for est in self._estimates.values()],
            index=pd.MultiIndex.from_tuples(self._estimates.keys(), names=["group", "time_period"]),
            name="att",
            dtype=np.float_,
        )

    @property
    def _index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(self._estimates.keys(), names=["group", "time_period"])

    def standard_errors(self) -> pd.Series:
        return pd.Series(
            [est.standard_errors() for est in self._estimates.values()],
            index=self._index,
            name="s.e.",
            dtype=np.float_,
        )

    def confidence_interval(self, alpha: float = 0.05) -> pd.DataFrame:
        upper_bound = (self.atts() + norm.ppf(1 - alpha / 2) * self.standard_errors()).rename(
            f"c.i. ub {alpha:.0%} "
        )
        lower_bound = (self.atts() + norm.ppf(alpha / 2) * self.standard_errors()).rename(
            f"c.i. lb {alpha:.0%}"
        )
        return pd.concat((lower_bound, upper_bound), axis=1)

    def _summary(self) -> pd.DataFrame:
        return pd.concat((self.atts(), self.standard_errors(), self.confidence_interval()), axis=1)

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
    summ = mpd_minimum_wage.summary()
    print(summ)
