# %%
import json
import types

import numpy as np
from osgeo import gdal, ogr

import hhnk_research_tools.logging as logging
from hhnk_research_tools.general_functions import (
    check_create_new_file,
    ensure_file_path,
)
from hhnk_research_tools.gis.raster import RasterOld
from hhnk_research_tools.rasters.raster_class import Raster
from hhnk_research_tools.variables import DEF_TRGT_CRS, GDAL_DATATYPE, GEOTIFF

DEFAULT_CREATE_OPTIONS = ["COMPRESS=ZSTD", "TILED=YES", "PREDICTOR=2", "ZSTD_LEVEL=1"]

logger = logging.get_logger(name=__name__)


# Conversion
def _gdf_to_json(gdf, epsg=DEF_TRGT_CRS):
    try:
        gdf_json = json.loads(gdf.to_json())
        gdf_json["crs"] = {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::{}".format(epsg)},
        }
        gdf_json_str = json.dumps(gdf_json)
        return gdf_json_str
    except Exception as e:
        raise e


def _gdf_to_ogr(gdf, epsg=DEF_TRGT_CRS):
    """Create ogr instance of gdf"""
    try:
        gdf_json = _gdf_to_json(gdf, epsg)
        ogr_ds = ogr.Open(gdf_json)
        polygon = ogr_ds.GetLayer()
        return ogr_ds, polygon
    except Exception as e:
        raise e


def gdf_to_raster(
    gdf,
    value_field,
    raster_out,
    nodata,
    metadata,
    epsg=DEF_TRGT_CRS,
    driver=GEOTIFF,
    datatype=GDAL_DATATYPE,
    create_options=DEFAULT_CREATE_OPTIONS,
    read_array=True,
    overwrite=False,
):
    """Dem is used as format raster. The new raster gets meta data from the DEM. A gdf is turned into ogr layer and is
    then rasterized.
    wsa.polygon_to_raster(polygon_gdf=mask_gdf[mask_type], valuefield='val', raster_output_path=mask_path[mask_type],
    nodata=0, meta=meta, epsg=28992, driver='GTiff')
    """
    try:
        if isinstance(raster_out, (Raster, RasterOld)):
            raster_out = raster_out.path

        gdf = gdf[[value_field, "geometry"]]  # filter unnecessary columns
        ogr_ds, polygon = _gdf_to_ogr(gdf, epsg)
        # make sure folders exist
        if raster_out != "":  # empty str when driver='MEM'
            ensure_file_path(raster_out)

        new_raster = create_new_raster_file(
            file_name=raster_out,
            nodata=nodata,
            meta=metadata,
            driver=driver,
            datatype=datatype,
            overwrite=overwrite,
        )

        if new_raster is not None:  # is None when raster already exists and was not overwritten.
            gdal.RasterizeLayer(
                new_raster,
                [1],
                polygon,
                options=[f"ATTRIBUTE={value_field}"] + create_options,
            )
            if read_array:
                raster_array = new_raster.ReadAsArray()
                return raster_array
            else:
                return None
    except Exception as e:
        raise e


# Saving
def _set_band_data(data_source, num_bands, nodata):
    try:
        for i in range(1, num_bands + 1):
            band = data_source.GetRasterBand(i)
            band.SetNoDataValue(nodata)
            band.Fill(nodata)
            band.FlushCache()  # close file after writing
            band = None
    except Exception as e:
        raise e


def create_new_raster_file(
    file_name,
    nodata,
    meta,
    driver=GEOTIFF,
    datatype=GDAL_DATATYPE,
    num_bands=1,
    create_options=None,
    overwrite=False,
):
    """
    ONLY FOR SINGLE BAND
    https://gdal.org/drivers/raster/gtiff.html#creation-options
    https://kokoalberti.com/articles/geotiff-compression-optimization-guide/
    Create new empty gdal raster using metadata from raster from sqlite (dem)
    driver='GTiff'
    driver='MEM'
    Compression:
    LZW - highest compression ratio, highest processing power
    DEFLATE
    PACKBITS - lowest compression ratio, lowest processing power

    Best compression options for Int16:
    [f"COMPRESS=ZSTD", f"TILED=True", "PREDICTOR=2", "ZSTD_LEVEL=1"]

    Best compression options for Float32:
    [f"COMPRESS=LERC_ZSTD", f"TILED=True", "PREDICTOR=2", "ZSTD_LEVEL=1", "MAX_Z_ERROR=0.001"]
    """
    try:
        if datatype is None:
            datatype = GDAL_DATATYPE
        if create_options is None:
            # if datatype==gdal.GDT_Float32:
            # options=[f"COMPRESS=LERC_DEFLATE", f"TILED=YES", "PREDICTOR=2", "ZSTD_LEVEL=1", "MAX_Z_ERROR=0.001"]
            # elif datatype==gdal.GDT_Int16:
            # options=[f"COMPRESS=LERC_ZSTD", f"TILED=YES", "PREDICTOR=2", "ZSTD_LEVEL=1", "MAX_Z_ERROR=0.001"]
            create_options = DEFAULT_CREATE_OPTIONS

            # else:
            #     options = [f"COMPRESS=DEFLATE", f"TILED=YES", "PREDICTOR=2", "ZSTD_LEVEL=1"]

        if driver == "MEM":
            check_is_file = False
        else:
            check_is_file = True
        if (
            check_create_new_file(output_file=file_name, overwrite=overwrite, check_is_file=check_is_file)
            or driver == "MEM"
        ):
            gdal_driver = gdal.GetDriverByName(driver)
            target_ds = gdal_driver.Create(
                str(file_name),
                meta.x_res,
                meta.y_res,
                num_bands,
                datatype,
                options=create_options,
            )

            target_ds.SetGeoTransform(meta.georef)
            _set_band_data(target_ds, num_bands, nodata)

            target_ds.SetProjection(meta.projection)
            return target_ds
    except Exception as e:
        raise e


def save_raster_array_to_tiff(
    output_file,
    raster_array,
    nodata,
    metadata,
    datatype=GDAL_DATATYPE,
    create_options=None,
    num_bands=1,
    overwrite=False,
):
    """
    ONLY FOR SINGLE BAND

    input:
    output_file (filepath)
    raster_array (values to be converted to tif)
    nodata (nodata value)
    metadata (dictionary)
    datatype -> gdal.GDT_Float32
    compression -> 'DEFLATE'
    num_bands -> 1
    """
    try:
        target_ds = create_new_raster_file(
            file_name=output_file,
            nodata=nodata,
            meta=metadata,
            datatype=datatype,
            create_options=create_options,
            overwrite=overwrite,
        )  # create new raster
        if target_ds is not None:  # is None when raster already exists and was not overwritten.
            for i in range(1, num_bands + 1):
                target_ds.GetRasterBand(i).WriteArray(raster_array)  # fill file with data
            target_ds = None
    except Exception as e:
        raise e


def build_vrt():
    raise DeprecationWarning("This function is depcrated in favour of hrt.Raster.build_vrt")


def create_meta_from_gdf(gdf, res):
    raise DeprecationWarning("This function is depcrated in favour of hrt.RasterMetadataV2.from_gdf.")


# def build_vrt(raster_folder, vrt_name="combined_rasters", bandlist=[1], bounds=None, overwrite=False):
#     """create vrt from all rasters in a folder.

#     raster_folder (str)
#     bounds (np.array): format should be; (xmin, ymin, xmax, ymax),
#         if None will use input files.
#     bandList doesnt work as expected, passing [1] works."""
#     raster_folder = Folder(raster_folder)
#     output_path = raster_folder.full_path(f"{vrt_name}.vrt")

#     if output_path.exists() and not overwrite:
#         print(f"vrt already exists: {output_path}")
#         return

#     tifs_list = [str(i) for i in raster_folder.find_ext(["tif", "tiff"])]

#     resolutions = []
#     for r in tifs_list:
#         r = Raster(r)
#         resolutions.append(r.metadata.pixel_width)
#     if len(np.unique(resolutions)) > 1:
#         raise Exception(f"Multiple resolutions ({resolutions}) found in folder. We cannot handle that yet.")

#     vrt_options = gdal.BuildVRTOptions(
#         resolution="highest",
#         separate=False,
#         resampleAlg="nearest",
#         addAlpha=False,
#         outputBounds=bounds,
#         bandList=bandlist,
#     )
#     ds = gdal.BuildVRT(destName=str(output_path), srcDSOrSrcDSTab=tifs_list, options=vrt_options)
#     ds.FlushCache()

#     if not output_path.exists():
#         print("Something went wrong, vrt not created.")


# TODO uitfaseren, dit kan door .vrt te maken dan zijn alle bounds direct goed.
def dx_dy_between_rasters(meta_big, meta_small):
    """create window to subset a large 2-d array with a smaller rectangle. Usage:
    shapes_array[dy_min:dy_max, dx_min:dx_max]
    window=create_array_window_from_meta(meta_big, meta_small)
    shapes_array[window]"""
    if meta_small.pixel_width != meta_big.pixel_width:
        raise Exception(
            f"""Input rasters dont have same resolution. 
                meta_big   = {meta_big.pixel_width}m
                meta_small = {meta_small.pixel_width}m"""
        )

    # FIXME waarom stond dit op max(0, x) en geeft dan geen verdere problemen?
    # dx_min = max(0, int((meta_small.x_min-meta_big.x_min)/meta_big.pixel_width))
    dx_min = int((meta_small.x_min - meta_big.x_min) / meta_big.pixel_width)
    dy_min = int((meta_big.y_max - meta_small.y_max) / meta_big.pixel_width)

    if dx_min < 0:
        raise Exception(f"dx_min smaller than 0 ({dx_min})")
    if dy_min < 0:
        raise Exception(f"dy_min smaller than 0 ({dy_min})")

    dx_max = int(min(dx_min + meta_small.x_res, meta_big.x_res))
    dy_max = int(min(dy_min + meta_small.y_res, meta_big.y_res))
    return dx_min, dy_min, dx_max, dy_max


class RasterCalculator:
    """Make a custom calculation between two rasters by
    reading the blocks and applying a calculation
    input raster should be of type hhnk_research_tools.gis.raster.Raster

    raster1: hrt.Raster -> big raster
    raster2: hrt.Raster -> smaller raster with full extent within big raster.
        Raster numbering is interchangeable as the scripts checks the bounds.
    raster_out: hrt.Raster -> output, doesnt need to exist. self.create also creates it.
    custom_run_window_function: function that takes window of small and big raster
        as input and does calculation with these arrays.
    customize below function for this, can take more inputs.

    def custom_run_window_function(self, raster1_window, raster2_window, band_out, **kwargs):
        #hrt.Raster_calculator custom_run_window_function
        #Customize this function with a calculation
        #Load windows
        block1 = self.raster1._read_array(window=raster1_window)
        block2 = self.raster2._read_array(window=raster2_window)

        #Calculate output
        block_out = None #replace with a calculation.

        # Write to file
        band_out.WriteArray(block_out, xoff=window_small[0], yoff=window_small[1])


    """

    def __init__(
        self,
        raster1: Raster,
        raster2: Raster,
        raster_out: Raster,
        custom_run_window_function,
        output_nodata,
        verbose=False,
    ):
        self.raster1 = raster1
        self.raster2 = raster2

        self.raster_big, self.raster_small, self.raster_mapping = self._checkbounds(raster1, raster2)
        self.raster_out = raster_out

        # dx dy between rasters.
        self.dx_min, self.dy_min, dx_max, dy_max = dx_dy_between_rasters(
            meta_big=self.raster_big.metadata, meta_small=self.raster_small.metadata
        )

        self.blocks_df = self.raster_small.generate_blocks()
        self.blocks_total = len(self.blocks_df)
        self.custom_run_window_function = types.MethodType(custom_run_window_function, self)
        self.output_nodata = output_nodata
        self.verbose = verbose

    def _checkbounds(self, raster1, raster2):
        x1, x2, y1, y2 = raster1.metadata.bounds
        xx1, xx2, yy1, yy2 = raster2.metadata.bounds
        bounds_diff = x1 - xx1, y1 - yy1, xx2 - x2, yy2 - y2  # subtract bounds
        check_arr = np.array([i <= 0 for i in bounds_diff])  # check if values <=0

        # If all are true (or all false) we know that the rasters fully overlap.
        if raster1.metadata.pixel_width != raster2.metadata.pixel_width:
            raise Exception("Rasters do not have equal resolution")

        if np.all(check_arr):
            # In this case raster1 is the bigger raster.
            return raster1, raster2, {"raster1": "big", "raster2": "small"}
        elif np.all(~check_arr):
            # In this case raster2 is the bigger raster
            return raster2, raster1, {"raster1": "small", "raster2": "big"}
        else:
            raise Exception("Raster bounds do not overlap. We cannot use this.")

    def create(self, overwrite=False) -> bool:
        """Create empty output raster
        returns bool wether the rest of the function should continue
        """
        # Check if function should continue.
        cont = True
        if not overwrite and self.raster_out.exists():
            cont = False

        if cont is True:
            if self.verbose:
                print(f"creating output raster: {self.raster_out.path}")
            target_ds = create_new_raster_file(
                file_name=self.raster_out.path,
                nodata=self.output_nodata,
                meta=self.raster_small.metadata,
            )
            target_ds = None
        else:
            if self.verbose:
                print(f"output raster already exists: {self.raster_out.path}")
        return cont

    def run(self, overwrite=False, **kwargs):
        """Loop over the small raster blocks, load both arrays and apply a custom function to it."""
        cont = self.create(overwrite=overwrite)

        if cont:
            target_ds = self.raster_out.open_gdal_source_write()
            band_out = target_ds.GetRasterBand(1)

            for idx, block_row in self.blocks_df.iterrows():
                # Load landuse
                window = {}
                window["small"] = block_row["window_readarray"]

                window["big"] = window["small"].copy()
                window["big"][0] += self.dx_min
                window["big"][1] += self.dy_min

                windows = {
                    "raster1": window[self.raster_mapping["raster1"]],
                    "raster2": window[self.raster_mapping["raster2"]],
                }

                self.custom_run_window_function(windows=windows, band_out=band_out, **kwargs)
                if self.verbose:
                    print(f"{idx} / {self.blocks_total}", end="\r")
                # break

            band_out.FlushCache()  # close file after writing
            band_out = None
            target_ds = None


def reproject(src: Raster, target_res: float, output_path: str):
    raise DeprecationWarning("This function is deprecated in favour of hrt.Raster.reproject")


def hist_stats(histogram: dict, stat_type: str, ignore_keys=[0]):
    """
    histogram (dict): histogram of raster built with np.unique(block_arr, return_counts=True)
    stat_type (str): statistics to calculate. Options are;
        ["median"]
    ignore_key (float/int/str): use this to remove the nodata value from hist

    calc median of a histogram. To create a hist per label, see example in
    nbs/sample_histogram_median.
    """

    total = 0
    for key in ignore_keys:
        histogram.pop(key, None)  # dont use 0 values in median calc

    # No values left, all values are nodata.
    if histogram == {}:
        return np.nan

    if stat_type == "median":
        median_index = (sum(histogram.values()) + 1) / 2
        for value in sorted(histogram.keys()):
            total += histogram[value]
            if total >= median_index:
                return value


# %%
