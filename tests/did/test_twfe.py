# Third party
import numpy as np

# First party
from causalipy.ols import ClusteredSeConfig
from causalipy.did.twfe import Twfe


def test_iris_with_fixed_effects(iris_data):
    # GIVEN a the following fixed model
    twfe = Twfe(
        iris_data, " Sepal_Length ~  -1 + Sepal_Width + Petal_Length", fixed_effects=["Species"]
    )

    # WHEN I compute the parameter estimates and the standard errors
    estimates = twfe.estimates
    config = ClusteredSeConfig(iris_data["Species"], True, True)
    standard_errors = twfe.get_standard_errors(config)

    # THEN I expect that the estimates and the standard errors correspond to the expected results
    np.testing.assert_allclose(estimates, np.array([0.4322172, 0.7756295]).reshape(-1, 1), rtol=6)
    np.testing.assert_allclose(
        standard_errors, np.array([0.161308, 0.126546]).reshape(-1, 1), rtol=6
    )
