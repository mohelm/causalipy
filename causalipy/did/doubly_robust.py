# Core Library
from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

# Third party
import numpy as np
import pandas as pd

# First party
from causalipy.ols import Ols
from causalipy.custom_types import NDArrayOfFloats
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
        dd = {
            f"{self._outcome}": lambda df: df[self._outcome].values
            - data.loc[data[self._time_period_indicator] != late_period, self._outcome].values
        }
        return data.query(f"`{self._time_period_indicator}` == @late_period").assign(**dd)

    @property
    def untreated_data(self) -> pd.DataFrame:
        return self._data.query(f"`{self._treatment_indicator}` == 0")

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
        return int(self.treatment_status.sum())

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
        formula: str = "~ 1",
    ):
        self._data = DataHandler(data, outcome, treatment_indicator, time_period_indicator)

        if self._estimate_or:
            self._formula_outcome_model = f"{outcome} {formula}"
            self._outcome_model = Ols(self._formula_outcome_model, self._data.untreated_data)
        if self._estimate_ipw:
            self._formula_treatment_status_model = f"{treatment_indicator}  {formula}"
            self._selection_model = LogisticRegression(
                self._formula_treatment_status_model, self._data.data
            )

        self._att_treated = self._weights_treated * self._outcome_treated
        self._att_control = self._weights_control * self._outcome_control
        self._eta_control = self._att_control.mean() / self._weights_control.mean()
        self._eta_treated = self._att_treated.mean() / self._weights_treated.mean()
        self.influence_function = self._get_if()

    @property
    def att(self) -> float:
        return self._eta_treated - self._eta_control

    def _get_alr_or_model(self, n: int, k: int) -> NDArrayOfFloats:
        # TODO: careful here, you assume a certain structure of data.
        # TODO: n and k should come from some place else
        return (
            n
            * np.r_[
                np.zeros((self._data.n_treated, k)),
                self._outcome_model.asymptotic_linear_representation,
            ]
        )

    def _get_if(self) -> NDArrayOfFloats:
        return self._get_treatment_if() - self._get_control_if()

    @property
    def _if_treated_component_1(self) -> NDArrayOfFloats:

        return self._att_treated - self._weights_treated * self._eta_treated

    @property
    def _if_control_component_1(self) -> NDArrayOfFloats:
        return self._att_control - self._weights_control * self._eta_control

    @property
    def _if_control_component_2(self) -> NDArrayOfFloats:
        alr = self._selection_model.score @ self._selection_model.vce
        inner = self._data.outcome - self._eta_control
        inner = inner if not self._estimate_or else inner - self._predictions
        X = self._selection_model.get_design_matrix(self._data.data)
        m2 = (self._weights_control * inner * X).mean(axis=0)
        return (alr @ m2).reshape(-1, 1)

    @property
    def _if_control_component_3(self) -> NDArrayOfFloats:
        X = self._outcome_model.get_design_matrix(self._data.data)
        alr_or = self._get_alr_or_model(*X.shape)
        return alr_or @ ((self._weights_control * X).mean(axis=0)).reshape(-1, 1)

    @abstractmethod
    def _get_treatment_if(self) -> NDArrayOfFloats:
        ...

    @abstractmethod
    def _get_control_if(self) -> NDArrayOfFloats:
        ...

    def standard_errors(self) -> float:
        return np.std(self.influence_function) / np.sqrt(self._data.n_units)


class OrEstimator(BaseDrDid):
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

    def _get_treatment_if(self) -> NDArrayOfFloats:
        return self._if_treated_component_1 / self._weights_treated.mean()

    def _get_control_if(self) -> NDArrayOfFloats:
        return (
            self._if_control_component_1 + self._if_control_component_3
        ) / self._weights_control.mean()


class IpwEstimator(BaseDrDid):
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

    def _get_treatment_if(self) -> NDArrayOfFloats:
        return self._if_treated_component_1 / self._weights_treated.mean()

    def _get_control_if(self) -> NDArrayOfFloats:
        return (
            self._if_control_component_1 + self._if_control_component_2
        ) / self._weights_control.mean()


class DrEstimator(BaseDrDid):
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

    def _get_treatment_if(self) -> NDArrayOfFloats:
        X = self._outcome_model.get_design_matrix(self._data.data)
        alr_or = self._get_alr_or_model(*X.shape)
        return (
            self._if_treated_component_1
            - (alr_or @ (self._weights_treated * X).mean(axis=0)).reshape(-1, 1)
        ) / self._weights_treated.mean()

    def _get_control_if(self) -> NDArrayOfFloats:
        return (
            self._if_control_component_1
            + self._if_control_component_2
            - self._if_control_component_3
        ) / self._weights_control.mean()


class DrDidMethod(str, Enum):
    dr = "dr"
    ipw = "ipw"
    oreg = "or"


dr_did_models = {
    DrDidMethod.dr: DrEstimator,
    DrDidMethod.ipw: IpwEstimator,
    DrDidMethod.oreg: OrEstimator,
}
