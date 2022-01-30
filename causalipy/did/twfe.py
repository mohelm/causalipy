from __future__ import annotations

# Core Library
from typing import Iterable

# Third party
import pandas as pd

# First party
from causalipy.ols import Ols, ClusteredSeConfig
from causalipy.custom_types import NDArrayOfFloats


class DataHandler:
    # TODO: is this a good idea. Are the fixed effects always separate from the equation
    def __init__(self, data: pd.DataFrame, fixed_effects: Iterable[str]):
        self._fixed_effects = fixed_effects
        self._data = self._demean_data(data) if fixed_effects else data

    def _demean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_deman = [col for col in data.columns if col not in self._fixed_effects]

        data_with_index = data.set_index(self._fixed_effects)

        for fe in self._fixed_effects:
            means = data.groupby(fe)[cols_to_deman].mean()
            data_with_index = data_with_index[cols_to_deman].subtract(means[cols_to_deman])
        return data_with_index

    @property
    def data(self) -> pd.DataFrame:
        return self._data


class Twfe:
    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        fixed_effects: Iterable[str],
    ):
        self._data = DataHandler(data, fixed_effects)

        self._res = Ols(formula, self._data.data)

    @property
    def estimates(self) -> NDArrayOfFloats:
        # Here its called estiamtes in ols it seems to be called coefficients...
        return self._res.coefficients

    def get_standard_errors(self, config: None | ClusteredSeConfig = None) -> NDArrayOfFloats:
        # TODO: are you specifying the clusters in a nice way here? Are you not repeating some stuff?
        return self._res.get_standard_errors(config)

    def summary(self) -> pd.DataFrame:
        ...
