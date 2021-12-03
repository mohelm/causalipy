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

        for g, t in combinations:
            current_data = data.query("group in (0,@g)  and time_period in (@g-1,@t)")
            dr = DoublyRobustDid(formula_or, formula_ipw, current_data)
            print(g, t, dr.att, dr.standard_errors())


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