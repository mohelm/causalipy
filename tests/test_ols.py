# Third party
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

# First party
from causalipy.ols import Ols


@pytest.fixture
def iris_data():
    path = "/Users/moritz.helm/.cache/causalipy/iris.csv"  # TODO fix this pat
    return pd.read_csv(path).rename(columns=lambda col: col.replace(".", "_"))


@pytest.mark.parametrize(
    "type, config, expected_results",
    [
        ("cluster", ("Species", True, True), (0.162625624, 0.051733177, 0.006872555)),
        ("cluster", ("Species", True, False), (0.132783266, 0.042239962, 0.005611417)),
        ("cluster", ("Species", False, False), (0.13188909, 0.04195551, 0.00557363)),
        ("cluster", ("Species", False, True), (0.161530489, 0.051384801, 0.006826274)),
    ],
)
def test_iris_clustered_standard_errors(iris_data, type, config, expected_results):
    ols = Ols("Sepal_Length ~ Sepal_Width + Petal_Length", iris_data)
    se = ols.get_standard_errors(type, iris_data[config[0]], config[1], config[2])

    assert_array_almost_equal(se, expected_results)
