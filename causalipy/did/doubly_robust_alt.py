# Core Library
from abc import ABC, abstractmethod
from typing import ClassVar

# Third party
import numpy as np
import pandas as pd

# First party
from causalipy.ols import Ols
from causalipy.custom_types import MaybeString, NDArrayOfFloats
from causalipy.logistic_regression import LogisticRegression


class DataHandler:
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment_indicator: str,
        time_period_indicator: str,
    ):

        late_period = data[time_period_indicator].max()
        self._treatment_indicator = treatment_indicator
        self._outcome = outcome
        self._time_period_indicator = time_period_indicator
        self._data = self._prepare_data(data, late_period)

    def _prepare_data(self, data: pd.DataFrame, late_period: int) -> pd.DataFrame:
        return data.query(f"{self._time_period_indicator} == @late_period").assign(
            outcome=lambda df: df[self._outcome].values
            - data.loc[data[self._time_period_indicator] != late_period, self._outcome].values
        )

    @property
    def untreated_data(self) -> pd.DataFrame:
        return self._data.query(f"{self._treatment_indicator} == 0")

    @property
    def outcome(self) -> NDArrayOfFloats:
        return self._data[self._outcome].values.reshape(-1, 1)

    @property
    def treatment_status(self) -> NDArrayOfFloats:
        return self._data[self._treatment_indicator].values.reshape(-1, 1)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def n_treated(self) -> int:
        return self.treatment_status.sum()

    @property
    def n_units(self) -> int:
        return self._data.shape[0]


class BaseDrDid(ABC):

    _estimate_ipw: ClassVar[bool]
    _estimate_or: ClassVar[bool]

    @property
    def _predictions(self) -> NDArrayOfFloats:
        return self._outcome_model.predict(self._data.data)

    @property
    def _prop_odds(self) -> NDArrayOfFloats:
        return self._selection_model.predict_odds(self._data.data)

    @property
    def _weights_treated(self) -> NDArrayOfFloats:
        return self._data.treatment_status

    @property
    @abstractmethod
    def _weights_control(self) -> NDArrayOfFloats:
        ...

    @property
    @abstractmethod
    def _outcome_treated(self) -> NDArrayOfFloats:
        ...

    @property
    @abstractmethod
    def _outcome_control(self) -> NDArrayOfFloats:
        ...

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str = "Y",
        treatment_indicator: str = "D",
        time_period_indicator: str = "time_period",
        formula: MaybeString = "~ 1",
    ):
        self._data = DataHandler(data, outcome, treatment_indicator, time_period_indicator)

        if self._estimate_or:
            self._formula = f"{outcome} {formula}"
            self._outcome_model = Ols(self._formula, self._data.untreated_data)
        if self._estimate_ipw:
            self._formula = f"{treatment_indicator}  {formula}"
            self._selection_model = LogisticRegression(self._formula, self._data.data)

        self._att_treated = self._weights_treated * self._outcome_treated
        self._att_control = self._weights_control * self._outcome_control
        self._eta_control = self._att_control.mean() / self._weights_control.mean()
        self._eta_treated = self._att_treated.mean() / self._weights_treated.mean()

    @property
    def att(self) -> float:
        return self._eta_treated - self._eta_control


class OrEstimatorAlt(BaseDrDid):
    _estimate_ipw = False
    _estimate_or = True

    @property
    def _weights_control(self) -> NDArrayOfFloats:
        return self._data.treatment_status

    @property
    def _outcome_treated(self) -> NDArrayOfFloats:
        return self._data.outcome

    @property
    def _outcome_control(self) -> NDArrayOfFloats:
        return self._predictions


class IpwEstimatorAlt(BaseDrDid):
    _estimate_ipw = True
    _estimate_or = False

    @property
    def _weights_control(self) -> NDArrayOfFloats:
        return self._selection_model.predict_odds(self._data.data) * (1 - self._weights_treated)

    @property
    def _outcome_treated(self) -> NDArrayOfFloats:
        return self._data.outcome

    @property
    def _outcome_control(self) -> NDArrayOfFloats:
        return self._data.outcome


class DrEstimatorAlt(BaseDrDid):
    _estimate_ipw = True
    _estimate_or = True

    @property
    def _weights_control(self) -> NDArrayOfFloats:
        return self._selection_model.predict_odds(self._data.data) * (1 - self._weights_treated)

    @property
    def _outcome_treated(self) -> NDArrayOfFloats:
        return self._data.outcome - self._predictions

    @property
    def _outcome_control(self) -> NDArrayOfFloats:
        return self._data.outcome - self._predictions


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

        self._formula = f"{outcome} {formula_or}"
        self._outcome_model = Ols(self._formula, self._data.untreated_data)
        X = self._outcome_model.get_design_matrix(self._data.data)

        self._predictions = self._outcome_model.predict(self._data.data)
        self._weights_treated = self._data.treatment_status
        self._weights_control = self._data.treatment_status
        self._att_treated = self._weights_treated * self._data.outcome
        self._att_control = self._weights_control * self._predictions
        self._eta_control = self._att_control.mean() / self._weights_control.mean()
        self._eta_treated = self._att_treated.mean() / self._weights_treated.mean()

        self.influence_function = self._get_if(X)

    def _get_alr_or_model(self, X: NDArrayOfFloats) -> NDArrayOfFloats:
        # TODO: careful here, you assume a certain structure of data.
        return (
            len(X)
            * np.r_[
                np.zeros((self._data.n_treated, X.shape[1])),
                self._outcome_model.asymptotic_linear_representation,
            ]
        )

    def _get_treatment_if(self) -> NDArrayOfFloats:
        return (
            self._att_treated - self._weights_treated * self._eta_treated
        ) / self._weights_treated.mean()

    def _get_control_if(self, X) -> NDArrayOfFloats:
        component_1 = self._att_control - self._weights_control * self._eta_control
        alr_or = self._get_alr_or_model(X)
        component_2 = alr_or @ ((self._weights_control * X).mean(axis=0)).reshape(-1, 1)
        return (component_1 + component_2) / self._weights_control.mean()

    def _get_if(self, X: NDArrayOfFloats) -> NDArrayOfFloats:
        return self._get_treatment_if() - self._get_control_if(X)

    @property
    def att(self) -> float:
        return self._eta_treated - self._eta_control

    def standard_errors(self) -> float:
        return np.std(self.influence_function) / np.sqrt(self._data.n_units)


class IpwEstimator:
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str = "Y",
        treatment_indicator: str = "D",
        time_period_indicator: str = "time_period",
        formula_ipw: str = "~ 1",
    ):
        self._data = DataHandler(data, outcome, treatment_indicator, time_period_indicator)

        self._formula = f"{treatment_indicator}  {formula_ipw}"
        self._selection_model = LogisticRegression(self._formula, self._data.data)
        X = self._selection_model.get_design_matrix(self._data.data)

        self._weights_treated = self._data.treatment_status
        self._weights_control = self._selection_model.predict_odds(self._data.data) * (
            1 - self._weights_treated
        )
        self._att_treated = self._weights_treated * self._data.outcome
        self._att_control = self._weights_control * self._data.outcome
        self._eta_control = self._att_control.mean() / self._weights_control.mean()
        self._eta_treated = self._att_treated.mean() / self._weights_treated.mean()

        self.influence_function = self._get_if(X)

    def _get_treatment_if(self) -> NDArrayOfFloats:
        return (
            self._att_treated - self._weights_treated * self._eta_treated
        ) / self._weights_treated.mean()

    def _get_control_if(self, X) -> NDArrayOfFloats:
        component_1 = self._att_control - self._weights_control * self._eta_control
        alr = self._selection_model.score @ self._selection_model.vce
        inner = self._data.outcome - self._eta_control
        m2 = (self._weights_control * inner * X).mean(axis=0)
        component_2 = (alr @ m2).reshape(-1, 1)
        return (component_1 + component_2) / self._weights_control.mean()

    def _get_if(self, X: NDArrayOfFloats) -> NDArrayOfFloats:
        return self._get_treatment_if() - self._get_control_if(X)

    @property
    def att(self) -> float:
        return self._eta_treated - self._eta_control

    def standard_errors(self) -> float:
        return np.std(self.influence_function) / np.sqrt(self._data.n_units)
