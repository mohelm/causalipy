# Core Library
from typing import NamedTuple

# Third party
import numpy as np
import pytest

# First party
from metrics.did.doubly_robust import DoublyRobustDid


class ExpectedResult(NamedTuple):
    att: float
    se: float


@pytest.mark.parametrize(
    "or_model, ipw_model,expected_result",
    [
        ("outcome ~ control", "treatment_status ~ control", ExpectedResult(1.2776, 0.0438)),
        (None, "treatment_status ~ control", ExpectedResult(1.2635, 0.0463)),
        ("outcome ~ control", None, ExpectedResult(1.3139, 0.0412)),
    ],
)
def test_doubly_robust_results(
    or_model, ipw_model, expected_result, simulate_data_multi_period_did
):
    # GIVEN a specification for the two models and knowledge about the expected results

    # WHEN I call the DoublyRobustDid estimator
    out = DoublyRobustDid(or_model, ipw_model, simulate_data_multi_period_did)

    # THEN I expect to get back the expected results
    np.testing.assert_almost_equal(out.att, expected_result.att, 4)
    np.testing.assert_almost_equal(out.standard_errors(), expected_result.se, 4)
