# Core Library
from typing import NamedTuple

# Third party
import numpy as np

# First party
from causalipy.did.doubly_robust import DrEstimator, OrEstimator, IpwEstimator


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
