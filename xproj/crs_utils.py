import warnings
from collections.abc import Sequence
from typing import Any, Literal

from pyproj import CRS


def format_crs(crs: CRS | None, max_width: int = 50) -> str:
    """Format CRS as a string.

    Parameters
    ----------
    crs : pyproj.crs.CRS
        The input CRS object to format.
    max_width : int, optional
        Maximum number of characters beyond which the formatted CRS
        will be truncated (default: 50).

    """
    if crs is not None:
        srs = crs.to_string()
    else:
        srs = "None"

    return srs if len(srs) <= max_width else " ".join([srs[:max_width], "..."])


def format_compact_cf(crs: CRS) -> dict[str, Any]:
    """Format CRS as a dictionary for minimal compatibility with
    CF conventions.

    More info:
    https://cfconventions.org/cf-conventions/cf-conventions.html

    Parameters
    ----------
    crs : pyproj.crs.CRS
        The input CRS object to format.

    Returns
    -------
    dict
        A dictionary with one ``crs_wkt`` item that contains
        the CRS information formatted as Well-Known Text (WKT).

    See Also
    --------
    xarray.Dataset.proj.write_crs_info
    format_full_cf_gdal

    """
    return {"crs_wkt": crs.to_wkt()}


def format_full_cf_gdal(crs: CRS) -> dict[str, Any]:
    """Format CRS as a dictionary for full compatibility with
    CF conventions and GDAL.

    More info:

    - https://cfconventions.org/cf-conventions/cf-conventions.html
    - https://gdal.org/en/stable/drivers/raster/netcdf.html

    Parameters
    ----------
    crs : pyproj.crs.CRS
        The input CRS object to format.

    Returns
    -------
    dict
        A dictionary with two ``crs_wkt`` and ``spatial_ref`` items
        that contains the CRS information formatted as Well-Known Text (WKT),
        as well as items representing all the CF grid mapping variable
        attributes exported via :py:meth:`pyproj.crs.CRS.to_cf`.

    See Also
    --------
    xarray.Dataset.proj.write_crs_info
    format_compact_cf

    """
    output = crs.to_cf()
    output["spatial_ref"] = crs.to_wkt()
    return output


def get_common_crs(
    crs_objs: Sequence[CRS | None] | set[CRS | None],
    on_undefined_crs: Literal["raise", "warn", "ignore"] = "warn",
    stacklevel: int = 3,
) -> CRS | None:
    """Try getting a common, unique CRS from an input sequence of (possibly
    undefined) CRS objects.

    Parameters
    ----------
    crs_objs : sequence or set
        Sequence of either :py:class:`pyproj.CRS` objects or ``None``
        (undefined CRS).
    on_undefined_crs : {"raise", "warn", "ignore"}, optional
        If "raise", raises a ValueError if a non-null CRS is found but
        one or more inputs have undefined CRS. If "warn" (default), emits a
        UserWarning instead. If "ignore", do nothing.
    stacklevel : int, optional
        Stack level value used for the emitted warning (default: 3).

    Returns
    -------
    pyproj.crs.CRS or None
        The common (possibly undefined) CRS.

    Raises
    ------
    ValueError
        If multiple conflicting CRS objects are found.

    Warns
    -----
    UserWarning
        If a common, unique CRS is found but one or more of the
        inputs have undefined CRS.

    """
    # code taken from geopandas (BSD-3 Licence)

    crs_objs = set(crs_objs)

    crs_not_none = [crs for crs in crs_objs if crs is not None]
    names = [crs.name for crs in crs_not_none]

    if len(crs_not_none) == 0:
        return None
    if len(crs_not_none) == 1:
        if len(crs_objs) != 1:
            if on_undefined_crs == "raise":
                raise ValueError("one or more inputs have undefined CRS.")
            elif on_undefined_crs == "warn":
                warnings.warn(  # noqa: B028
                    "CRS is undefined for some of the inputs. "
                    f"Setting output's CRS as {names[0]} "
                    "(the single non-null CRS provided).",
                    stacklevel=stacklevel,
                )
        return crs_not_none[0]

    raise ValueError(f"cannot determine common CRS from inputs CRSes {names}. ")
