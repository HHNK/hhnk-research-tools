# %%
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as rxr
import shapely
import tqdm
import xarray as xr
from osgeo import gdal
from rasterio import features
from scipy import ndimage
from shapely import geometry

import hhnk_research_tools.logger as logging
from hhnk_research_tools.folder_file_classes.file_class import File
from hhnk_research_tools.general_functions import check_create_new_file
from hhnk_research_tools.rasters.raster_metadata import RasterMetadataV2

CHUNKSIZE = 4096

logger = logging.get_logger(name=__name__)


class Raster(File):
    def __init__(self, base, chunksize: Union[int, None] = CHUNKSIZE):
        super().__init__(base)

        if chunksize is None:
            self.chunksize = CHUNKSIZE
        else:
            self.chunksize = chunksize

        self._rxr = None
        self._profile = None  # rio profile
        self._metadata = None

    # DEPRECATED
    def deprecation_warn(self, txt: str = "") -> DeprecationWarning:
        return DeprecationWarning(f"Not available since 2024.2{txt}")

    @property
    def array(self):
        raise self.deprecation_warn()

    @property
    def _array(self):
        raise self.deprecation_warn()

    def _read_array(self, band=None, window: Optional[list] = None):
        """window=[x0, y0, x1, y1]--oud.
        window=[x0, y0, xsize, ysize]
        x0, y0 is left top corner!!
        """
        # TODO uitfaseren, niet met gdal maar rxr.read
        if band is None:
            gdal_src = self.open_gdal(mode="r")
            band = gdal_src.GetRasterBand(1)

        if window is not None:
            raster_array = band.ReadAsArray(
                xoff=int(window[0]),
                yoff=int(window[1]),
                win_xsize=int(window[2]),
                win_ysize=int(window[3]),
            )
        else:
            raster_array = band.ReadAsArray()

        band.FlushCache()  # close file after writing
        band = None
        return raster_array

    @property
    def get_array(self):
        raise self.deprecation_warn()

    @property
    def source(self):
        raise self.deprecation_warn(", use .open_gdal()")

    @property
    def source_set(self):
        raise self.deprecation_warn()

    @property
    def source_set(self):
        raise self.deprecation_warn()

    @property
    def band_count(self):
        raise self.deprecation_warn()

    @property
    def plot(self):
        # TODO rxr versie maken
        # plt.imshow(self._array)
        raise self.deprecation_warn()

    @property
    def profile(self):
        """Rio profile containing metadata. Can be used to create a new raster with
        the same attributes.
        """
        if self._profile is None:
            self._profile = self.open_rio().profile
        return self._profile

    @property
    def metadata(self) -> RasterMetadataV2:
        if self._metadata is None:
            self._metadata = RasterMetadataV2.from_rio_profile(rio_prof=self.open_rio().profile)
        return self._metadata

    @property
    def nodata(self) -> float:
        return self.profile["nodata"]

    @property
    def shape(self):
        return self.metadata.shape

    @property
    def pixelarea(self) -> float:
        return self.metadata.pixelarea

    def open_gdal_source_read(self):
        raise self.deprecation_warn(", use: .open_gdal()")

    def open_gdal_source_write(self):
        raise self.deprecation_warn(', use: .open_gdal(mode="r+")')

    def open_gdal(self, mode="r"):
        """usage;
        with self.open_gdal() as gdal_src: doesnt work.
        just dont write it to the class, and it should be fine..
        """
        if mode == "r":  # Read mode
            return gdal.Open(self.base, gdal.GA_ReadOnly)
        elif mode == "r+":  # Write mode
            return gdal.Open(self.base, gdal.GA_Update)
        else:
            raise ValueError(f"mode '{mode}' not in ['r','r+']")

    def open_rio(self, mode="r", profile=None, dtype=None):
        """Open raster with rasterio.
        Can be both read (r) and write (w)

        Parameters
        ----------
        mode : str, by default "r"
            use "w" for write access.
        profile :
            rio profile with raster metadata. Used to create new raster
        """
        if profile is None:
            return rio.open(self.base, mode)

        # Open (non-existing) raster with a profile
        else:
            profile_copy = profile.copy()  # Otherwise it will change the input profile
            dtype2 = profile_copy.pop("dtype")
            if dtype is None:
                dtype = dtype2
            return rio.open(self.base, mode, **profile_copy, dtype=dtype)

    def open_rxr(self, mask_and_scale=False, chunksize: Union[int, None] = None):
        """Open raster as rxr.DataArray

        Parameters
        ----------
        mask_and_scale : bool, default=False
            Lazily scale (using the scales and offsets from rasterio) and mask.
        chunksize: int | None, default=self.chunksize
            chuncksize for opening
        """

        if chunksize is None:
            chunksize = self.chunksize

        return rxr.open_rasterio(
            self.base,
            chunks={"x": chunksize, "y": chunksize},
            masked=True,
            mask_and_scale=mask_and_scale,
        )

    def overviews_build(self, factors: list = [10, 50], resampling="average") -> None:
        """Build overviews for faster rendering.
        documentation: https://gdal.org/programs/gdaladdo.html
        """
        with self.open_rio("r+") as dst:
            dst.build_overviews(factors, getattr(rio.enums.Resampling, resampling))
            dst.update_tags(ns="rio_overview", resampling=resampling)

        # TODO this seems to give a nice result, can we incorparte it here somewhre or can rio do the same?
        # !gdaladdo -ro -r nearest --config COMPRESS_OVERVIEW ZSTD --config PREDICTOR_OVERVIEW 2 --config ZSTD_LEVEL_OVERVIEW 1 "C:\Users\Wietse\Documents\HHNK\playground\dem_schemer_hoog_zuid_compressed_v4.tif" 8 32

    def overviews_available(self):
        """Display available overviews on raster."""
        with self.open_rio() as src:
            print([src.overviews(i) for i in src.indexes])

    def overviews_remove(self):
        """Remove overviews from raster
        Note that this only unlinks the overview. It is still in the file.
        """
        with self.open_gdal(mode="r+") as ds:
            ds.BuildOverviews("NONE")
            ds = None

        print(f"Removed overviews: {self.view_name_with_parents(2)}")

    def statistics(self, decimals=6) -> dict[str, float]:
        raster_src = self.open_rxr()
        return {
            "min": np.round(float(raster_src.min().values), decimals),
            "max": np.round(float(raster_src.max().values), decimals),
            "mean": np.round(float(raster_src.mean().values), decimals),
            "std": np.round(float(raster_src.std().values), decimals),
        }

    @classmethod
    def write(
        cls,
        raster_out: Union[str, Path],
        result: xr.DataArray,
        nodata: Optional[float] = None,
        dtype: str = "float32",
        scale_factor: Optional[float] = None,
        chunksize: int = CHUNKSIZE,  # TODO not sure if chunksize needed here
        crs=28992,
    ):
        """
        Write a rxr result to raster.
        works both for rasters and gdf.

        from geocube.api.core import make_geocube
        result = make_geocube(
            vector_data=grid_gpkg,
            resolution=(5, -5),
            fill=-9999
        )

        raster_out : str, Path, Raster
            output location
        result : xr.DataArray
            output array, either the result of an rxr calcultation or a dataframe turned into geocube
        dtype : str
            output raster dtype
        scale_factor: float
            When saving as int, use a scale_factor to get back to original scale.
        chunksize : int

        """
        raster_out = cls(raster_out, chunksize=chunksize)

        if "float" in dtype:
            compress = "LERC_DEFLATE"
        else:
            compress = "ZSTD"

        # Set nodata otherwise its not in raster
        if nodata is not None:
            result.rio.set_nodata(nodata)

        # Set crs
        if result.rio.crs is None:
            logger.warning(
                "write - Crs of output DataArray not provided. EPSG:28992 was assumed. set crs with e.g. da.rio.set_crs(28992)"
            )
        elif result.rio.crs != crs:
            raise ValueError("CRS on DataArray not equal to provided crs")
        result.rio.set_crs(crs)

        # Write to file
        result.rio.to_raster(
            raster_out.base,
            # chunks={"x": CHUNKSIZE, "y": CHUNKSIZE},
            chunks=True,
            # lock=threading.Lock(), #Use dask multithread
            lock=True,
            tiled=True,
            windowed=True,  # If lock is False, window doesnt work.
            COMPRESS=compress,  # gdal options
            PREDICTOR=2,  # gdal options
            ZSTD_LEVEL=1,  # gdal options
            MAX_Z_ERROR=0.001,  # gdal options
            NUM_THREADS="ALL_CPUS",  # gdal options
            BIGTIFF="YES",
            dtype=dtype,
        )

        # Settings the scale_factor does not work with rioxarray. Update the result
        if scale_factor:
            with raster_out.open_rio("r+") as src:
                src.scales = (scale_factor,)

        return raster_out

    @classmethod
    def build_vrt(
        cls,
        vrt_out,
        input_files: Union[str, list],
        bounds=None,
        overwrite: bool = False,
        resolution="highest",
        bandlist=[1],
    ):
        """Build vrt from input files.
        overwrite (bool)
        bounds (np.array): format should be; (xmin, ymin, xmax, ymax)
            if None will use input files. (Raster.metadata.bbox_gdal)
        input_files (list): list of paths to input rasters
        resolution: "highest"|"lowest"|"average"
            instead of "user" option, provide a float for manual target_resolution
        bandList: doesnt work as expected, passing [1] works.
        """
        # TODO add option for relative to VRT = 1

        vrt_out = cls(vrt_out)

        if vrt_out.suffix != ".vrt":
            raise ValueError(f"Provide .vrt file instead of {vrt_out.suffix}")

        if check_create_new_file(output_file=vrt_out, overwrite=overwrite):
            # Set inputfiles to list of strings.
            if not isinstance(input_files, list):
                input_files = [str(input_files)]
            else:
                input_files = [str(i) for i in input_files]

            if type(resolution) in (float, int):
                kwargs = {}
                xRes = resolution
                yRes = resolution
                resolution = "user"
            else:
                xRes = None
                yRes = None

            # Check resolution of input files
            input_resolutions = []
            for r in input_files:
                # break
                r = cls(r)
                input_resolutions.append(r.metadata.pixel_width)
            if len(np.unique(input_resolutions)) > 1:
                raise Exception(
                    f"Multiple resolutions ({input_resolutions}) found in input_files. We cannot handle that yet."
                )

            # Build vrt
            vrt_options = gdal.BuildVRTOptions(
                resolution=resolution,
                separate=False,
                resampleAlg="nearest",
                addAlpha=False,
                outputBounds=bounds,
                bandList=bandlist,
                xRes=xRes,
                yRes=yRes,
            )
            ds = gdal.BuildVRT(destName=str(vrt_out), srcDSOrSrcDSTab=input_files, options=vrt_options)
            ds.FlushCache()
        return vrt_out

    def sum_labels(
        self,
        label_raster,
        label_idx: np.array,
        count_values=False,
        decimals: Union[int, None] = None,
    ) -> dict:
        """Calculate the sum of the rastervalues per label.
        Used in statistiek stedelijk

        Parameters
        ----------
        label_idx : list or array
            The values to get the statistics for. e.g. label_gdf['id'].to_numpy()
        count_values : bool
            Instead of the sum it will count the occurences per label.
        decimals : int
            When provided, round the results to the specified decimals

        Returns
        -------
        label_dict : dict
            Each label_idx value is paired with the summed value in the raster.
            e.g. {1:132.94, 2:628.3}
        """
        if label_raster.metadata.georef != self.metadata.georef:
            raise ValueError(
                f"Label raster georef {label_raster.metadata.georef} does not match the raster georef {self.metadata.georef}"
            )

        blocks_gdf = RasterChunks.from_raster(self).to_gdf()

        r = self.open_rxr()  # Open rasters for reading
        r_label = label_raster.open_rxr()

        accum = np.zeros(label_idx.shape)
        for idx, block_row in tqdm.tqdm(blocks_gdf.iterrows(), total=len(blocks_gdf)):
            # Read the blocks / window
            minx, miny, maxx, maxy = block_row["window"]
            block = r.isel(x=slice(minx, maxx), y=slice(miny, maxy))
            block_label = r_label.isel(x=slice(minx, maxx), y=slice(miny, maxy))

            # Replace nodata values with 0
            block = xr.where(block == self.nodata, 0, block)
            block = xr.where(block.isnull(), 0, block)  # Make sure NaN values are also 0  # noqa: PD003

            # When counting, all other values become 1.
            if count_values:
                block = xr.where(block != 0, 1, block)

            # Calculate sum per label (region)
            result = ndimage.sum_labels(input=block.values, labels=block_label.values, index=label_idx)

            # Add result to total
            accum += result

        if count_values:
            decimals = 0
        if decimals is not None:
            logger.debug(f".sum_labels - Rounding to {decimals} decimals")
            if decimals == 0:
                accum = accum.astype(int)
            else:
                accum = np.round(accum, decimals)

        label_dict = dict(zip(label_idx, accum))
        return label_dict

    # %% TODO working

    def sum(self):
        """Calculate sum of raster"""
        da = self.open_rxr()
        return da.sum(skipna=True).values.item()

    def round_nearest(self, x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))

    def read_geometry(self, geometry: shapely.geometry.Polygon) -> np.array:
        """
        Read data within geometry. Outside geometry is set to NaN.

        Parameters
        ----------
        geometry : shapely.geometry
            Geometry in which the data is read.

        """
        resolution = self.metadata.pixel_width
        bounds = [self.round_nearest(i, resolution) for i in geometry.bounds]

        width = (bounds[2] - bounds[0]) * (1 / resolution)
        height = (bounds[3] - bounds[1]) * (1 / resolution)

        transform = rio.transform.from_bounds(*(bounds + [width, height]))
        bounds = rio.coords.BoundingBox(*bounds)

        if hasattr(geometry, "geoms"):
            geometry_list = list(geometry.geoms)
        else:
            geometry_list = [geometry]

        array = features.rasterize(
            geometry_list,
            out_shape=(int(height), int(width)),
            transform=transform,
        )

        raster = self.open_rio()
        window = raster.window(*bounds)
        data = raster.read(window=window)[0]
        data[array == 0] = raster.nodata # shape = 1, outside shape = 0 
        raster.close()
        return data

    def polygonize(self, array=None, field_name="field"):
        """TODO docstr and test"""
        raster = self.open_rio()
        if array is None:
            array = raster.read()

        mask = raster.dataset_mask()
        generator = features.shapes(array, mask=mask, transform=raster.transform)

        output = {field_name: [], "geometry": []}
        for i, (geom, value) in enumerate(generator):
            output[field_name].append(value)
            output["geometry"].append(shapely.geometry.shape(geom))

        return gpd.GeoDataFrame(output)

    @classmethod
    def reproject(cls, src, dst, target_res: float):
        """
        Parameters
        ----------
        src : hrt.Raster - can't be typehint due to circular import
        dst : hrt.Raster
        target_res : float
        """
        # https://svn.osgeo.org/gdal/trunk/autotest/alg/reproject.py

        meta = RasterMetadataV2.from_raster(src, res=target_res)
        rio_profile = meta.to_rio_profile(nodata=src.nodata, dtype=src.profile["dtype"])

        dst.open_rio(mode="w", profile=rio_profile)  # Create raster
        src_ds = src.open_gdal()
        dst_ds = dst.open_gdal(mode="r+")
        if dst_ds is not None:
            gdal.ReprojectImage(src_ds, dst_ds, src_wkt=src.metadata.projection)

        return cls(dst)


# %%


@dataclass
class RasterChunks:
    """Represents the chunks on a raster as dataframe or geodataframe.

    Call .to_df and .to_gdf respectively to get the dataframe.
    """

    metadata: RasterMetadataV2
    chunksize: int = CHUNKSIZE

    @classmethod
    def from_raster(cls, raster: Raster):
        return cls(metadata=raster.metadata, chunksize=raster.chunksize)

    def from_gdf(cls, gdf: gpd.GeoDataFrame, res: float, chunksize=CHUNKSIZE):
        metadata = RasterMetadataV2.from_gdf(gdf=gdf, res=res)
        return cls(metadata=metadata, chunksize=chunksize)

    def to_df(self) -> pd.DataFrame:
        """Generate blocks with the blocksize of the band.
        These blocks can be used as window to load the raster iteratively.
        """
        ncols = int(np.floor(self.metadata.x_res / self.chunksize))
        nrows = int(np.floor(self.metadata.y_res / self.chunksize))

        # Create arrays with index of where windows end. These are square blocks.
        xparts = np.linspace(0, self.chunksize * ncols, ncols + 1).astype(int)
        yparts = np.linspace(0, self.chunksize * nrows, nrows + 1).astype(int)

        # If raster has some extra data that didnt fall within a block it is added to the parts here.
        # These blocks are not square.
        if self.chunksize * ncols != self.metadata.x_res:
            xparts = np.append(xparts, self.metadata.x_res)
            ncols += 1
        if self.chunksize * nrows != self.metadata.y_res:
            yparts = np.append(yparts, self.metadata.y_res)
            nrows += 1

        blocks_df = pd.DataFrame(index=np.arange(nrows * ncols) + 1, columns=["ix", "iy", "window"])
        i = 0
        for ix in range(ncols):
            for iy in range(nrows):
                i += 1
                blocks_df.loc[i, :] = np.array(
                    (ix, iy, [xparts[ix], yparts[iy], xparts[ix + 1], yparts[iy + 1]]),
                    dtype=object,
                )

        blocks_df["window_readarray"] = blocks_df["window"].apply(
            lambda x: [int(x[0]), int(x[1]), int(x[2] - x[0]), int(x[3] - x[1])]
        )

        return blocks_df

    def _generate_window_geometry(self, window: list[int], geom_type="box"):
        """Create a geometry from a given window. Either return (box) polygons
        or create a point shape with only the centrepoint.

        Parameters
        ----------
        window : list[int]
            window in the raster has below format
            [
                xoff:int,  pixelcount offset from x_min
                yoff:int,  pixelcount offset from y_max
                xsize:int, pixels to right
                ysize:int, pixels down
            ]

        geom_type : str, options ["box","point"], by default "box"
            returntype of geometry. Box or centrepoint
        """
        minx = self.metadata.x_min
        maxy = self.metadata.y_max

        # find window bounds, account for pixel size
        minx += window[0] * self.metadata.pixel_width
        maxy += window[1] * self.metadata.pixel_height
        maxx = minx + window[2] * self.metadata.pixel_width
        miny = maxy + window[3] * self.metadata.pixel_height

        # rxr gebruikt bij xy het midden van de cel.
        rxr_minx = minx + self.metadata.pixel_width * 0.5
        rxr_maxy = maxy + self.metadata.pixel_height * 0.5

        if geom_type == "point":  # TODO do we need point?
            geom = geometry.Point(rxr_minx, rxr_maxy)
        elif geom_type == "box":
            geom = geometry.box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        return rxr_minx, rxr_maxy, geom

    def to_gdf(self, geom_type="box") -> gpd.GeoDataFrame:
        """Create blocks with shapely geometry"""
        blocks_df = self.to_df()

        blocks_df[["minx", "maxy", "geometry"]] = blocks_df.apply(
            lambda x: self._generate_window_geometry(
                window=x["window_readarray"],
                geom_type=geom_type,
            ),
            axis=1,
            result_type="expand",
        )

        blocks_df["xy"] = blocks_df.apply(lambda x: f"{x['minx']},{x['maxy']}", axis=1)

        blocks_df = gpd.GeoDataFrame(
            blocks_df,
            geometry="geometry",
            crs=self.metadata.projection,
        )

        return blocks_df


# %%

if __name__ == "__main__":
    from tests_hrt.config import TEST_DIRECTORY

    raster = Raster(TEST_DIRECTORY / r"depth_test.tif", chunksize=40)

    chunks = RasterChunks.from_raster(raster=raster)
    df = chunks.to_gdf()

    df.plot()
    # %%
    raster2 = Raster(TEST_DIRECTORY / r"depth_test2.tif", chunksize=40)

    with raster2.open_rio(mode="w", profile=raster.profile, dtype="float32") as dst:
        pass
