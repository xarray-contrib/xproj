import pyproj
import pytest
import xarray as xr
from xarray.indexes import PandasIndex

import xproj


class IndexWithCRS(PandasIndex, xproj.ProjIndexMixin):
    _crs: pyproj.CRS | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._crs = None
        self.transformed = False

    @property
    def crs(self) -> pyproj.CRS | None:
        return self._crs

    def _proj_set_crs(self, spatial_ref, crs):
        self._crs = crs
        return self

    def _proj_to_crs(self, spatial_ref, crs):
        new_index = self._copy()
        new_index._crs = crs
        new_index.transformed = True
        return new_index

    def equals(self, other, exclude=None) -> bool:
        if not isinstance(other, IndexWithCRS):
            return False
        if not self._proj_crs_equals(other, allow_none=True):
            return False
        return super().equals(other)

    def _copy(self, deep=True, memo=None):
        # bug in PandasIndex? subclass attribute not copied
        obj = super()._copy(deep=deep, memo=memo)
        obj._crs = self._crs
        obj.transformed = self.transformed
        return obj


def test_index_mixin_abstract() -> None:
    class Index(PandasIndex, xproj.ProjIndexMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    with pytest.raises(TypeError):
        Index([1, 2], "x")  # type: ignore


def test_map_crs() -> None:
    ds = (
        xr.Dataset(coords={"foo": ("x", [1, 2])})
        .set_xindex("foo", IndexWithCRS)
        .proj.assign_crs(spatial_ref=pyproj.CRS.from_epsg(4326))
    )

    ds_mapped = ds.proj.map_crs(spatial_ref=["foo"])
    assert ds_mapped.proj("foo").crs == pyproj.CRS.from_epsg(4326)

    # override + transform
    ds2 = ds_mapped.proj.assign_crs(spatial_ref=pyproj.CRS.from_epsg(4978), allow_override=True)

    with pytest.raises(ValueError, match="allow_override=False"):
        ds2.proj.map_crs(spatial_ref=["foo"])

    ds2_mapped = ds2.proj.map_crs(spatial_ref=["foo"], allow_override=True, transform=True)
    assert ds2_mapped.proj("foo").crs == pyproj.CRS.from_epsg(4978)
    assert ds2_mapped.xindexes["foo"].transformed is True


def test_map_crs_warns() -> None:
    # map CRS to a default PandasIndex -> not effect (warning)
    ds = xr.Dataset(coords={"x": [1, 2]})
    ds = ds.proj.assign_crs(spatial_ref=pyproj.CRS.from_epsg(4326))

    with pytest.warns(UserWarning, match=r"the index.*not recognized as CRS-aware.*"):
        ds.proj.map_crs(spatial_ref=["x"])


@pytest.mark.parametrize("epsg,crs_match", [(4326, True), (4978, False)])
def test_map_crs_read_only(epsg, crs_match) -> None:
    # try mapping the CRS a spatial ref coordinate to a CRS-aware index
    # that has read-only CRS access

    class IndexWithImmutableCRS(PandasIndex, xproj.ProjIndexMixin):
        @property
        def crs(self) -> pyproj.CRS | None:
            return pyproj.CRS.from_epsg(4326)

    ds = (
        xr.Dataset(coords={"foo": ("x", [1, 2])})
        .set_xindex("foo", IndexWithImmutableCRS)
        .proj.assign_crs(spatial_ref=pyproj.CRS.from_epsg(epsg))
    )

    # works if CRS match (map_crs has no effect)
    if crs_match:
        ds_mapped = ds.proj.map_crs(spatial_ref=["foo"])
        assert ds_mapped.proj("foo").crs == ds_mapped.proj("spatial_ref").crs

    # error if CRS mismatch
    else:
        with pytest.raises(NotImplementedError):
            ds.proj.map_crs(spatial_ref=["foo"], allow_override=True)

        with pytest.raises(NotImplementedError):
            ds.proj.map_crs(spatial_ref=["foo"], allow_override=True, transform=True)


def test_index_crs_equals() -> None:
    ds_base = xr.Dataset(coords={"foo": ("x", [1, 2])}).set_xindex("foo", IndexWithCRS)

    ds_crs_undef = ds_base.copy()
    ds_crs1 = ds_base.proj.assign_crs(spatial_ref=pyproj.CRS.from_epsg(4326)).proj.map_crs(
        spatial_ref=["foo"]
    )
    ds_crs2 = ds_base.proj.assign_crs(spatial_ref=pyproj.CRS.from_epsg(4978)).proj.map_crs(
        spatial_ref=["foo"]
    )

    assert ds_crs_undef.xindexes["foo"].equals(ds_crs1.xindexes["foo"])
    assert not ds_crs1.xindexes["foo"].equals(ds_crs2.xindexes["foo"])
