# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:08:39 2022

@author: chris.kerklaan
"""

# Third party imports
from osgeo import gdal, ogr

# Drivers
DRIVER_GDAL_MEM = gdal.GetDriverByName("MEM")


# TODO Deprecate?
def rasterize(
    vector_path,
    rows,
    columns,
    geotransform,
    spatial_reference_wkt,
    nodata=-9999,
    field=None,
    all_touches=False,
    options=None,
    return_ds=False,
    data_type=gdal.GDT_Float32,
):
    ds = ogr.Open(vector_path)
    layer = ds[0]
    target_ds = DRIVER_GDAL_MEM.Create("rasterize", columns, rows, 1, data_type)

    # set nodata
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.Fill(nodata)
    band.FlushCache()

    # set metadata
    target_ds.SetProjection(spatial_reference_wkt)
    target_ds.SetGeoTransform(geotransform)

    # set options
    gdal_options = []

    if field:
        gdal_options.append(f"ATTRIBUTE={field}")

    if all_touches:
        gdal_options.append("ALL_TOUCHES=TRUE")

    if options:
        gdal_options.extend(options)

    if len(gdal_options) == 0:
        gdal.RasterizeLayer(target_ds, (1,), layer, burn_values=(1,))
    else:
        gdal.RasterizeLayer(target_ds, [1], layer, options=gdal_options)

    if return_ds:
        return target_ds
    else:
        array = target_ds.ReadAsArray()
        target_ds = None
        return array
