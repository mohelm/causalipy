# Core Library
from pathlib import Path

# Third party
import pandas as pd
import pytest

DATA_DIRECTORY = Path(__file__).parent / "data"


@pytest.fixture
def simulate_data_multi_period_did():
    data = pd.read_feather(f"{DATA_DIRECTORY}/sim-ds.feather").rename(
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
    return data.query("time_period in (3, 4)")


@pytest.fixture
def iris_data():
    path = "/Users/moritz.helm/.cache/causalipy/iris.csv"  # TODO fix this pat
    return pd.read_csv(path).rename(columns=lambda col: col.replace(".", "_"))
