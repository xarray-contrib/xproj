from __future__ import annotations

import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, overload

import numpy as np
import pandas as pd
from pyproj import CRS
from xarray import Index, Variable

if TYPE_CHECKING:
    from xarray.core.indexing import IndexSelResult
    from xarray.core.types import JoinOptions

try:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
except ImportError:
    if TYPE_CHECKING:
        raise
    else:
        Self: Any = None


class CRSAwareIndex(Protocol):
    """Protocol class that defines a CRS-aware Xarray index."""

    @property
    def crs(self) -> CRS | None: ...

    # TODO: eventually we won't need to copy the xarray.Index interface here?
    # (https://github.com/python/typing/issues/213)

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> Index: ...

    @classmethod
    def concat(
        cls,
        indexes: Sequence[Self],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> Index: ...

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> dict[Hashable, Variable]: ...

    def to_pandas_index(self) -> pd.Index: ...

    def isel(self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]) -> Index | None: ...

    def sel(self, labels: dict[Any, Any]) -> IndexSelResult: ...

    def join(self, other: Self, how: JoinOptions = "inner") -> Self: ...

    def reindex_like(self, other: Self) -> dict[Hashable, Any]: ...

    @overload
    def equals(self, other: Index) -> bool: ...

    @overload
    def equals(self, other: Index, *, exclude: frozenset[Hashable] | None = None) -> bool: ...

    def equals(self, other: Index, **kwargs) -> bool: ...
