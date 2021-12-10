# Core Library
from itertools import product

# Third party
import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng

# First party
from causalipy.did.multi_periods import DrDidMethod, MultiPeriodDid

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
