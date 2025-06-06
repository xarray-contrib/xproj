from __future__ import annotations

import abc
from collections.abc import Hashable
from typing import TYPE_CHECKING, Generic, TypeVar

import pyproj
from xarray import DataArray, Dataset

if TYPE_CHECKING:
    from xproj.typing import CRSAwareIndex, Self


T_Xarray_Object = TypeVar("T_Xarray_Object", Dataset, DataArray)


class ProjAccessorMixin(abc.ABC, Generic[T_Xarray_Object]):
    """Mixin class that marks XProj support for an Xarray accessor."""

    @abc.abstractmethod
    def _proj_set_crs(self, spatial_ref: Hashable, crs: pyproj.CRS) -> T_Xarray_Object:
        """Method called when setting a new CRS via
        :py:meth:`xarray.Dataset.proj.assign_crs()`.

        Parameters
        ----------
        spatial_ref : Hashable
            The name of the spatial reference (scalar) coordinate
            to which the CRS has been set.
        crs : pyproj.crs.CRS
            The new CRS attached to the spatial reference coordinate.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Either a new or an existing Dataset or DataArray.

        """
        ...


class ProjIndexMixin(abc.ABC):
    """Mixin class that marks XProj support for an Xarray index.

    An :py:class:`xarray.Index` that inherits from this mixin class is
    identified by XProj as a :term:`CRS-aware index` (note that an Xarray index
    that simply has a ``crs`` property may also be identified as such, although
    it may lack some XProj support).

    """

    @property
    @abc.abstractmethod
    def crs(self) -> pyproj.CRS | None:
        """Returns the coordinate reference system (CRS) of the index as a
        :class:`pyproj.crs.CRS` object, or ``None`` if CRS is undefined.
        """
        ...

    def _proj_crs_equals(self, other: CRSAwareIndex, allow_none: bool = False) -> bool:
        """Helper method to check if this CRS-aware index has the same CRS than the
        other given CRS-aware index.

        This method is usually called internally within the index's ``equals()``,
        ``join()`` and ``reindex_like()`` methods.

        Parameters
        ----------
        other : xarray.Index
            The other CRS-aware index to compare with this index.
        allow_none : bool, optional
            If True, any undefined CRS is treated as the same (default: False).

        """
        # code taken from geopandas (BSD-3 Licence)

        other_crs = other.crs

        if allow_none:
            if self.crs is None or other_crs is None:
                return True
        if not self.crs == other_crs:
            return False
        return True

    def _proj_set_crs(
        self: Self,
        spatial_ref: Hashable,
        crs: pyproj.CRS,
    ) -> Self:
        """Method called when mapping a CRS to index coordinate(s) via
        :py:meth:`xarray.Dataset.proj.map_crs`.

        Parameters
        ----------
        spatial_ref : Hashable
            The name of the spatial reference (scalar) coordinate.
        crs : pyproj.crs.CRS
            The new CRS attached to the spatial reference coordinate.

        Returns
        -------
        Index
            Either a new or an existing xarray Index.

        """
        raise NotImplementedError("This CRS-aware index does not support (re)setting the CRS.")

    def _proj_to_crs(
        self: Self,
        spatial_ref: Hashable,
        crs: pyproj.CRS,
    ) -> Self:
        """Method called when mapping a CRS to index coordinate(s) via
        :py:meth:`xarray.Dataset.proj.map_crs` with ``transform=True``.

        Parameters
        ----------
        spatial_ref : Hashable
            The name of the spatial reference (scalar) coordinate.
        crs : pyproj.crs.CRS
            The new CRS attached to the spatial reference coordinate.

        Returns
        -------
        Index
            Either a new or an existing xarray Index.

        """
        raise NotImplementedError(
            "This CRS-aware index does not support (re)setting the CRS "
            "with coordinate data transformation."
        )
