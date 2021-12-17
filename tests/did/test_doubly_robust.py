# Core Library
from typing import NamedTuple

# Third party
import numpy as np
import pandas as pd

# First party
from causalipy.did.doubly_robust import DataHandler, DrEstimator, OrEstimator, IpwEstimator


class ExpectedResult(NamedTuple):
    att: float
    se: float


def test_ipw_estimator(simulate_data_multi_period_did):
    out = IpwEstimator(
        simulate_data_multi_period_did,
        outcome="outcome",
        treatment_indicator="treatment_status",
        formula="~control",
    )
    expected_result = ExpectedResult(1.2635, 0.0463)
    np.testing.assert_almost_equal(out.att, expected_result.att, 4)
    np.testing.assert_almost_equal(out.standard_errors(), expected_result.se, 4)


def test_or_estimator(simulate_data_multi_period_did):
    out = OrEstimator(
        simulate_data_multi_period_did,
        outcome="outcome",
        treatment_indicator="treatment_status",
        formula="~control",
    )
    expected_result = ExpectedResult(1.3139, 0.0412)
    np.testing.assert_almost_equal(out.att, expected_result.att, 4)
    np.testing.assert_almost_equal(out.standard_errors(), expected_result.se, 4)


def test_dr_estimator(simulate_data_multi_period_did):
    out = DrEstimator(
        simulate_data_multi_period_did,
        outcome="outcome",
        treatment_indicator="treatment_status",
        formula="~control",
    )
    expected_result = ExpectedResult(1.2776, 0.0438)
    np.testing.assert_almost_equal(out.att, expected_result.att, 4)
    np.testing.assert_almost_equal(out.standard_errors(), expected_result.se, 4)


def test_data_handler():

    # GIVEN a set of data with two time periods in which each unit is observed for two time periods
    data = pd.DataFrame(
        {
            "Y": [1, 2, 3, 4],
            "time": [1, 2, 1, 2],
            "D": [0, 0, 1, 1],
            "unit_id": [100, 100, 101, 101],
        }
    )

    # WHEN this set of data is passed to the DataHandler
    handled_data = DataHandler(data, "Y", "D", "time")

    # WHEN I ask for the untreated data then I expect to get back a a dataframe that contains only the data of the
    # untreated
    assert handled_data.untreated_data["D"].sum() == 0

    # AND I expect the number ob observations in the data of the untreated to correspond to the number of untreated
    # units in the late time period.
    n_untreated_obs_in_late_period = data.query("time==2 and D==0").shape[0]
    assert handled_data.untreated_data.shape == (n_untreated_obs_in_late_period, data.shape[1])

    # AND when I ask for the treatment status, I expect to get a one dimensional array that contains the correct
    # treatment status in the late period.
    treatment_status = handled_data.treatment_status
    n_obs_in_late_period = len(data) / 2
    assert treatment_status.shape == (n_obs_in_late_period, 1)
    np.testing.assert_almost_equal(
        treatment_status, data.query("time==2")["D"].values.reshape(-1, 1)
    )

    # AND when I ask for the outcome then I expect to get back an array of dimension 1 where the values are
    # the differences between the second period outcome and the first period outcome
    outcome = handled_data.outcome
    assert outcome.shape == (n_obs_in_late_period, 1)
    expected_outcome = (
        (data.loc[data.time == 2, "Y"].values - data.loc[data.time == 1, "Y"].values)
        .reshape(-1, 1)
        .astype(np.float_)
    )
    np.testing.assert_almost_equal(outcome.astype(np.float_), expected_outcome)

    # AND I expect the number of treated to correspond to the number of treated observations in the late period
    assert handled_data.n_treated == data.query("time==2 and D==1").shape[0]

    # AND I expect the number of units to correspond to the total number of observations in the late period
    assert handled_data.n_units == data.query("time==2").shape[0]

    # AND I expect the data to correspond to the input dataset in the late period except for the outcome variable
    assert handled_data.data.shape == data.query("time==2").shape
    for var in handled_data.data:
        if var != "Y":
            pd.testing.assert_series_equal(handled_data.data[var], data.query("time==2")[var])
        else:
            np.testing.assert_almost_equal(
                handled_data.data[var].values.reshape(-1, 1), expected_outcome
            )
