import pytest
from pyproj import CRS

from xproj import format_crs, get_common_crs


def test_format_crs() -> None:
    crs = CRS.from_epsg(4326)
    assert format_crs(crs) == "EPSG:4326"
    assert format_crs(crs, max_width=4) == "EPSG ..."


def test_get_common_crs() -> None:
    objs = [
        CRS.from_epsg(4326),
        None,
        CRS.from_epsg(4326),
    ]

    with pytest.warns(UserWarning, match="CRS is undefined for some of the inputs"):
        get_common_crs(objs)

    assert get_common_crs(objs, on_undefined_crs="ignore") == CRS.from_epsg(4326)

    with pytest.raises(ValueError, match="one or more inputs have undefined CRS"):
        get_common_crs(objs, on_undefined_crs="raise")

    assert get_common_crs([None, None]) is None

    with pytest.raises(ValueError, match="cannot determine common CRS"):
        get_common_crs([CRS.from_epsg(4326), CRS.from_epsg(4978)])
