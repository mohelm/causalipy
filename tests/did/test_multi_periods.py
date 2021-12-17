# Core Library
from itertools import product

# Third party
import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng

# First party
from causalipy.did.multi_periods import DrDidMethod, MultiPeriodDid, simulate_data

rng = default_rng()


@pytest.fixture()
def dataset():
    n = 100
    time_periods = [2002, 2003, 2004, 2005]
    groups = np.arange(5)
    return pd.DataFrame(
        {
            "time_period": rng.choice(time_periods, n),
            "group": rng.choice(groups, n),
        }
    )


class FakeDoublyRobustDid:
    def __init__(self, *args):
        pass

    @property
    def att(self):
        return rng.random()

    def standard_errors(self):
        return rng.random()


def test_multi_period_did_summary(mocker, dataset):
    # GIVEN a dataset and a fake estimator
    mocker.patch("causalipy.did.multi_periods.dr_did_models", {DrDidMethod.dr: FakeDoublyRobustDid})

    all_groups = dataset.group.unique()
    groups = np.delete(all_groups, np.where(all_groups == 0))
    time_periods = np.sort(dataset.time_period.unique())[1:]

    combinations = ((g, t) for g, t in product(groups, time_periods))
    expected_index = (
        pd.MultiIndex.from_tuples(combinations).rename(["group", "time_period"]).sort_values()
    )

    # WHEN I call  the MultiPeriodDid class with the dataset
    dd = MultiPeriodDid(dataset)

    # THEN I expect for the att, the standard errors and the summary to get back the correct indices
    pd.testing.assert_index_equal(dd.get_summary().index, expected_index)
    pd.testing.assert_index_equal(dd.atts().index, expected_index)
    pd.testing.assert_index_equal(dd.standard_errors().index, expected_index)


def test_simulate_data_without_removel_of_initial_period():
    # GIVEN a set of function parameters
    n_treated_units = 100
    n_untreated_units = 150
    n_time_periods = 4

    # WHEN simualte data is called with said parameters (and I do not delete the initial period)
    data = simulate_data(
        n_treated_units, n_untreated_units, n_time_periods, remove_initial_period=False
    )

    # THEN I expect to get an instance of type dataframe back
    assert isinstance(data, pd.DataFrame)

    # AND I expect the number of treated units in the dataframe to equal the number of treated units supplied to the
    # function (and I expect the same for the untreated units).
    assert data.query("D==1")["unit_id"].nunique() == n_treated_units
    assert data.query("D==0")["unit_id"].nunique() == n_untreated_units

    # AND I expect there to be n_time_periods time periods
    assert data["time_period"].nunique() == n_time_periods

    # AND I also expect there to be n-time periods groups + 1 for the untreated group
    assert data["group"].nunique() == n_time_periods + 1


def test_simulate_data_with_removal_of_inital_period():
    # GIVEN a set of function parameters
    n_treated_units = 100
    n_untreated_units = 150
    n_time_periods = 5

    # WHEN simualte data is called with said parameters (and I do not delete the initial period)
    data = simulate_data(
        n_treated_units, n_untreated_units, n_time_periods, remove_initial_period=True
    )

    # THEN I expect to get an instance of type dataframe back
    assert isinstance(data, pd.DataFrame)

    # AND I expect there to be n_time_periods
    assert data["time_period"].nunique() == n_time_periods

    # AND I also expect there to be n-time groups (n_time_periods - 1 treated groups and one untreated group)
    assert data["group"].nunique() == n_time_periods


def test_simulate_data_with_no_variance_in_outcomes():
    # GIVEN a set of function parameters
    n_treated_units = 100
    n_untreated_units = 150
    n_time_periods = 5

    # WHEN simualte data is called with said parameters and all (relevant) sources of variation are set to zero
    data = simulate_data(
        n_treated_units,
        n_untreated_units,
        n_time_periods,
        y0_error_variance=0,
        y1_error_variance=0,
        group_effect_variance=0,
    )

    # THEN I expect the diff-in-diff between the treatment and the control outcome to equal exactly the difference
    # between the time period and the group (which identifies the initial treatment period plus 1 for all periods in
    # which the treatment could have an effect and 0 in the other periods.
    mean_outcomes = data.groupby(["D", "group", "time_period"])["Y"].mean()
    mean_outcome_control = mean_outcomes.loc[0]
    mean_outcome_treated = mean_outcomes.loc[1]

    for (group, time_period), outcome in mean_outcome_treated.items():
        if group - 1 > 0:
            diff_treat = outcome - mean_outcome_treated.loc[group, group - 1]
            diff_control = (
                mean_outcome_control[0, time_period] - mean_outcome_control.loc[0, group - 1]
            )
            diff_in_diff = diff_treat - diff_control
            expected_effect = 0 if time_period < group else time_period - group + 1
            np.testing.assert_array_almost_equal(diff_in_diff, expected_effect)
