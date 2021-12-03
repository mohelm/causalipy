from __future__ import annotations

# Core Library
from typing import Union

# Third party
import numpy as np
from numpy.typing import NDArray

NDArrayOfFloats = NDArray[np.float_]
# mypy complains if we use the NDArrayOfFloats here
MaybeNDArrayOfFloats = Union[NDArray[np.float_], None]
MaybeString = Union[str, None]
MaybeInt = Union[int, None]
