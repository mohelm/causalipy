# Third party
import pandas as pd

# First party
from causalipy.ols import Ols
from causalipy.custom_types import NDArrayOfFloats


def _prepare_data(
    data: pd.DataFrame,
    outcome: str,
    time_period_indicator: str,
    late_period: int,
) -> pd.DataFrame:
    return data.query(f"{time_period_indicator} == @late_period").assign(
        outcome=lambda df: df[outcome].values
        - data.loc[data[time_period_indicator] != late_period, outcome].values
    )


class DataHandler:
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment_indicator: str,
        time_period_indicator: str,
    ):

        late_period = data[time_period_indicator].max()
        self._data = _prepare_data(data, outcome, time_period_indicator, late_period)
        self._treatment_indicator = treatment_indicator
        self._outcome = outcome
        self._time_period_indicator = time_period_indicator

    @property
    def untreated_data(self) -> pd.DataFrame:
        return self._data.query(f"{self._treatment_indicator==0}")

    @property
    def outcome(self) -> NDArrayOfFloats:
        return self._data[self._outcome].values.reshape(-1, 1)

    @property
    def treatment_status(self) -> NDArrayOfFloats:
        return self._data[self._treatment_indicator].values.reshape(-1, 1)

    @property
    def data(self) -> pd.DataFrame:
        return self.data


class OrEstimator:
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str = "Y",
        treatment_indicator: str = "D",
        time_period_indicator: str = "time_period",
        formula_or: str = "~ 1",
    ):

        self._data = DataHandler(data, outcome, treatment_indicator, time_period_indicator)
        self._outcome_model = Ols(formula_or, self._data.untreated_data)

        self._predictions = self._outcome_model.predict(data)
        self._weights_treated = self._data.treatment_status
        self._weights_control = self._data.treatment_status
        self._diff_in_diff = self._data.outcome - self._predictions
        self._att_treated = self._weights_treated * self._diff_in_diff
        self._att_control = self._weights_control * self._diff_in_diff
        self._eta_control = self._att_control.mean() / self._weights_control.mean()
        self._eta_treated = self._att_treated.mean() / self._weights_treated.mean()
