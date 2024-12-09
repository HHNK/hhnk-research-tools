# %%


import rasterio

# Had an issue with the proj installation that came with rasterio. Could be due
# to a pip install. This checks that.
try:
    rasterio.crs.CRS.from_epsg(28992)
except rasterio.errors.CRSError as e:
    print("Fix you rasterio installation (pip uninstall rasterio, mamba install rasterio)")
    raise (e)
