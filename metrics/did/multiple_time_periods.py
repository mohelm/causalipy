from __future__ import annotations

# Core Library
from functools import cached_property

# Third party
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# First party
from metrics.ols import ols
from metrics.logistic_regression import LogisticRegression, lr


def simulate_data(
    n_treated_units: int = 4000,
    n_untreated_units: int = 4000,
    n_time_periods: int = 4,
    seed: int | None = None,
):
    """
    Simulate a dataset to demonstrate multi-period analysis.


    """
    time_periods = np.arange(n_time_periods) + 1
    n_total_units = n_treated_units + n_untreated_units
    n_total_obs_treated = n_time_periods * n_treated_units
    n_total_obs_untreated = n_time_periods * n_untreated_units

    rng = np.random.default_rng(seed)

    def _get_controls(mean: int, group_size: int) -> NDArray[np.float_]:
        return np.repeat(rng.normal(mean, 1, size=group_size), n_time_periods)

    controls = np.r_[_get_controls(0, n_untreated_units), _get_controls(1, n_treated_units)]

    error = rng.normal(0, 1, (n_total_units) * n_time_periods)

    theta = np.repeat(time_periods, n_total_units)

    group = np.append(
        np.repeat(
            rng.choice(time_periods, n_treated_units, True),
            n_time_periods,
        ),
        np.zeros(n_total_obs_untreated),
    )

    y0 = np.repeat(time_periods, n_total_units) * controls + theta + rng.normal(group, 1) + error

    time_period = np.tile(time_periods, n_total_units)
    treatment_status = np.where(
        (np.repeat(np.arange(n_total_units), n_time_periods) < n_treated_units),
        1,
        0,
    )
    y1 = np.r_[
        y0[:n_total_obs_treated]
        + np.where(
            (time_period[:n_total_obs_treated] >= group[:n_total_obs_treated]),
            time_period[:n_total_obs_treated] - group[:n_total_obs_treated] + 1,
            0,
        )
        + rng.normal(0, 1, n_total_obs_treated)
        - error[:n_total_obs_treated],
        np.empty(n_total_obs_untreated) * np.NaN,
    ]

    outcome_observed = np.where(treatment_status == 1, y1, y0)
    unit_id = np.r_[np.repeat(np.arange(n_total_units), n_time_periods)]

    # TODO: what is going on with this data+1
    return (
        pd.DataFrame(
            {
                "treatment_status": treatment_status,
                "outcome": outcome_observed,
                "control": controls,
                "time_period": time_period,
                "unit_id": unit_id,
                "group": group,
            }
        )
        .query("group!=1")
        .sort_values(["unit_id", "time_period"])
    )


def multi_period_did_or(outcome_regression: str, data: pd.DataFrame):
    # TODO: think about if you want to give the option to only compute the effects for the periods where they are (possibly) non-zero.
    # TODO: so there is also this factor Gg/E[Gg] - Do you need that here? Is everybody treated here?
    mask = data.groupby("unit_id")["treatment_status"].transform("sum") > 0
    never_treated = data[~mask]
    treated = data[mask]

    # OR
    for g in np.sort(data.group.unique()):
        for t in np.sort(data.time_period.unique()):
            print("time:", t, "group:", g)
            nt_data = never_treated.query(f"group=={g} and time_period==@t").assign(
                outcome=lambda df: df["outcome"].values
                - never_treated.query(f"time_period=={g-1} and group==@g")["outcome"].values
            )
            t_data = treated.query(f"group=={g} and time_period==@t").assign(
                outcome=lambda df: df["outcome"].values
                - treated.query(f"time_period=={g-1} and group==@g")["outcome"].values
            )
            or_result = ols("outcome ~ control", nt_data)
            # TODO: fix predict
            intermed = (
                t_data["outcome"].values[:, None] - or_result.predict(t_data).squeeze()[:, None]
            )
            print(intermed.mean())  # by what do you have to scale here - if at all


# IPW
def multi_period_did_ipw(data: pd.DataFrame):
    for g in np.sort(data.group.unique()):
        data_g = data.query("group==@g & time_period==@g")
        prop_mode = lr("treatment_status ~ control", data_g)
        preds = prop_mode.predict(data_g).T
        ipw_weights = preds / (1 - preds)
        ipw_weights = -ipw_weights / ipw_weights.mean()
        inv_treatment_prob = 1 / (data_g["treatment_status"].mean())

        for t in np.sort(data.time_period.unique()):
            data_tg = data.query(f"group=={g} and time_period==@t").assign(
                outcome=lambda df: df["outcome"].values
                - data.query(f"time_period=={g-1} and group==@g")["outcome"].values
            )
            ipw_weights[data_tg["treatment_status"] == 1] = inv_treatment_prob
            estimate = (ipw_weights * data_tg["outcome"].values.reshape(-1, 1)).mean()
            print("Time", t, "Group", g)
            print(estimate)


class IpwDid:
    def __init__(self, formula: str, data: pd.DataFrame):
        max_time_period = data["time_period"].max()
        min_time_period = data["time_period"].min()
        data.set_index("id", inplace=True)

        data_selection_model = data.query("time_period == @max_time_period")
        selection_model = LogisticRegression(formula, data_selection_model)
        difference = (
            data.query("time_period == @max_time_period")["outcome"].subtract(
                data.query("time_period==@min_time_period")["outcome"]
            )
        ).values.reshape(-1, 1)

        self.odds = selection_model.predict_odds(data_selection_model)
        self.treatment_status = data_selection_model["treatment_status"].values.reshape(-1, 1)
        self.control_weights = self.odds * (1 - self.treatment_status)
        self.treatment_status_mean = self.treatment_status.mean()
        self.control_weights_mean = self.control_weights.mean()

        # Treatment effect
        self._weights_difference_control = (
            difference * self.control_weights / self.control_weights_mean
        )
        self._weights_difference_treatment = (
            difference * self.treatment_status / self.treatment_status_mean
        )
        self._att = (self._weights_difference_treatment - self._weights_difference_control).mean()

        # Variance
        alr_ps = selection_model.score @ selection_model.vce

        self.influence_function = difference * (self.treatment_status - self.control_weights) - (
            (
                alr_ps
                @ (
                    self.control_weights
                    * difference
                    * selection_model.get_design_matrix(data_selection_model)
                ).mean(axis=0)
            ).reshape(-1, 1)
        )

        self.n_units = data_selection_model.shape[0]

    @cached_property
    def standard_errors(self) -> float:
        return np.std(self.influence_function) / np.sqrt(self.n_units)

    @cached_property
    def att(self) -> float:
        return self._att


class OrDid:
    def __init__(self, formula: str, data: pd.DataFrame):
        ...


class DrDid:
    def __init__(self, formula: str, data: pd.DataFrame):
        ...


def multi_period_did(
    outcome_regression: str | None, selection_regression: str | None, data: pd.DataFrame
):
    groups = np.sort(data.group.unique())[1:]  # Ignore the zero group
    time_periods = np.sort(data.group.unique())[1:]

    for g in groups:
        if selection_regression is not None:
            mask = (
                (data["treatment_status"] == 1) & (data["group"] == g) & (data["time_period"] == g)
            ) | ((data["treatment_status"] == 0) & (data["time_period"] == g))
            data_g = data[mask]
            selection_estimate = lr(selection_regression, data_g)
            preds = selection_estimate.predict(data_g).T
            ipw_weights = preds / (1 - preds)
            ipw_weights = -ipw_weights / ipw_weights.mean()

        for t in time_periods:
            # All data in t_g
            mask_1 = (
                (data["treatment_status"] == 1) & (data["group"] == g) & (data["time_period"] == t)
            ) | ((data["treatment_status"] == 0) & (data["time_period"] == t))

            # Only depends on G
            mask_2 = (
                (data["treatment_status"] == 1)
                & (data["group"] == g)
                & (data["time_period"] == g - 1)
            ) | ((data["treatment_status"] == 0) & (data["time_period"] == g - 1))

            data_tg = data[mask_1].assign(
                outcome=lambda df: df["outcome"].values - data[mask_2]["outcome"].values
            )
            treatment_prob = data_tg["treatment_status"].mean()
            if outcome_regression is not None:
                data_tg_nt = data_tg.query("treatment_status==0")
                or_result = ols(outcome_regression, data_tg_nt)
                preds = or_result.predict(data_tg)

            weights = (data_tg["treatment_status"].values / treatment_prob).reshape(-1, 1)
            if selection_regression is not None:
                weights[weights == 0] = ipw_weights[weights == 0]
            outcome = data_tg["outcome"].values.reshape(-1, 1)
            if outcome_regression is not None:
                outcome = outcome - preds

            # # Just follow here:
            X = or_result.design_matrix(data_tg)[0]
            treatment_status = data_tg["treatment_status"].values.reshape(-1, 1)
            weights_ols = (1 - data_tg["treatment_status"]).values.reshape(-1, 1)
            ex = weights_ols * outcome * X
            xpx = np.linalg.inv((weights_ols * X).T @ X) * len(X)
            alr_ols = ex @ xpx  # noqa

            # What design matrix here?
            score_ps = (
                data_tg["treatment_status"].values.reshape(-1, 1)
                - selection_estimate.predict(data_g).T
            ) * X
            hess = np.linalg.inv(selection_estimate._hess) * len(X)
            alr_ps = score_ps @ hess  # noqa

            M1 = (treatment_status * X).mean(axis=0)
            influence_fcn_treatment = (alr_ols @ M1).reshape(-1, 1)

            x = treatment_status * (
                outcome - ((treatment_status / treatment_status.mean()) * outcome).mean()
            )

            inf_treat = (x - influence_fcn_treatment) / treatment_status.mean()

            # CONTROLS
            preds = selection_estimate.predict(data_g).T
            ipw_weights = (preds / (1 - preds)) * (1 - treatment_status)
            x_cont = ipw_weights * (outcome - ((ipw_weights / ipw_weights.mean()) * outcome).mean())

            M2 = (
                ipw_weights * (outcome - ((ipw_weights / ipw_weights.mean()) * outcome).mean()) * X
            ).mean(axis=0)
            inf_cont_2 = (alr_ps @ M2).reshape(-1, 1)

            M3 = (ipw_weights * X).mean(axis=0)
            inf_cont_3 = (alr_ols @ M3).reshape(-1, 1)
            inf_cont = (x_cont + inf_cont_2 - inf_cont_3) / ipw_weights.mean()

            inf_ = inf_treat - inf_cont

            se = np.std(inf_) / np.sqrt(len(x))

            estimate = (weights * outcome).mean()
            print("Time period", t, "Group:", g, "Estimate", estimate, "SE", se)


if __name__ == "__main__":
    # data = simulate_data(4000, 4000, 4)
    # multi_period_did_or(outcome_regression="x", data=data)
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

    # multi_period_did_ipw(data=data)

    # multi_period_did("outcome ~ control", "treatment_status ~ control", data)

    data = data.query("time_period in (1, 2)")
    ipw_did = IpwDid("treatment_status ~ control", data)

    # First party
    from metrics.logistic_regression import LogisticRegression

    lr = LogisticRegression("treatment_status~control", data)

    # # Is group observed here?

    # def _demean_data(data, time: str, group: str):
    #     cols_to_deman = [col for col in data.columns if col not in (time, group)]
    #     time_averages = data.groupby(time)[cols_to_deman].mean()

    #     individual_averages = data.groupby(group)[cols_to_deman].mean()
    #     avg = data[cols_to_deman].mean(axis=0)
    #     return (
    #         data.set_index([time, group])[cols_to_deman]
    #         .subtract(time_averages[cols_to_deman])
    #         .subtract(individual_averages[cols_to_deman])
    #         .add(avg[cols_to_deman])
    #     )

    # demeaned = _demean_data(data, "time_period", "group")

    # res = ols("outcome ~ treatment_status+control ", demeaned)

    # data = pd.read_feather("/Users/moritz.helm/.cache/py-did/badcondecomp-castle.feather").assign(
    #     homicide_l=lambda df: np.log(df["homicide"])
    # )

    # demeaned = _demean_data(data, "year", "state")
    # res = ols("homicide_l ~ post - 1", demeaned)

    # data = pd.read_feather("/Users/moritz.helm/.cache/py-did/hdma.feather")
    # data["deny"] = data["deny"] - 1
    # res = lr("deny ~ pirat + hirat + lvrat ", data)

    # # Third party
    # from statsmodels.formula.api import logit

    # x = logit("deny ~ pirat+hirat+lvrat", data).fit(full_output=True)
    # coeffs = x.params.values
