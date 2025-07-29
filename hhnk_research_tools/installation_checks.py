# %%
import rasterio


def check_rasterio_pyproj_installation():
    """Had an issue with the proj installation that came with rasterio. Could be due
    to a pip install. Reinstall generally works.

    Otherwise check if the windows environment variable PROJ_DIR is set.
    """

    try:
        rasterio.crs.CRS.from_epsg(28992)
    except rasterio.errors.CRSError as e:
        print(
            "Fix you rasterio installation (pip uninstall rasterio, mamba install rasterio)"
        )
        raise (e)
