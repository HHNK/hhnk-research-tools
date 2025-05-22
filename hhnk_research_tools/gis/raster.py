# %%
import inspect
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy import ndimage
from shapely import geometry
from tqdm.notebook import tqdm

import hhnk_research_tools as hrt
from hhnk_research_tools.folder_file_classes.file_class import File
from hhnk_research_tools.general_functions import get_functions, get_variables

# If anything goes wrong in gdal, make sure we raise the errors instead
# of silenty ignoring the issues.
gdal.UseExceptions()


# %%
class RasterOld(File):
    def __init__(self, base, min_block_size=1024):
        super().__init__(base)

        self.source_set = False  # Tracks if the source exist on the system.
        self._array = None
        self.min_block_size = min_block_size

    @property
    def array(self):
        if self._array is not None:
            print("from memory")
            return self._array
        else:
            print("Array not loaded. Call Raster.get_array(window) first")
            return self._array

    @array.setter
    def array(self, raster_array, window=None, band_nr=1):
        self._array = raster_array

    def _read_array(self, band=None, window=None):
        # TODO hidden to public?
        """window=[x0, y0, x1, y1]--oud.
        window=[x0, y0, xsize, ysize]
        x0, y0 is left top corner!!
        """
        if band is None:
            gdal_src = self.open_gdal_source_read()
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

    def get_array(self, window=None, band_count=None):
        # TODO hoe deze gebruiken tov _read_array? is het nuttig om
        # array ook in cls weg te schrijven.
        try:
            if band_count is None:
                band_count = self.band_count

            gdal_src = self.open_gdal_source_read()
            if band_count == 1:
                raster_array = self._read_array(band=gdal_src.GetRasterBand(1), window=window)

            elif band_count == 3:
                red_array = self._read_array(band=gdal_src.GetRasterBand(1), window=window)
                green_array = self._read_array(band=gdal_src.GetRasterBand(2), window=window)
                blue_array = self._read_array(band=gdal_src.GetRasterBand(3), window=window)

                raster_array = np.dstack((red_array, green_array, blue_array))
            else:
                raise ValueError(
                    f"Unexpected number of bands in raster {self.base} (got {band_count}, expected 1 or 3)"
                )
            self._array = raster_array
            return raster_array

        except Exception as e:
            raise e from None

    @property
    def source(self):
        if super().exists():
            if not self.source_set:
                self.source = True  # call source.setter
            return self.open_gdal_source_read()
        else:
            return False

    @source.setter
    def source(self, value):
        """If source does not exist it will not be set.
        Bit clunky. But it should work that if it exists it will only be set once.
        Otherwise it will not set.
        """
        if super().exists():  # cannot use self.exists here.
            # Needs to be first otherwise we end in a loop when settings metadata/nodata/band_count
            self.source_set = True

            gdal_src = self.open_gdal_source_read()

            self._metadata = RasterMetadata(gdal_src=gdal_src)
            self._nodata = gdal_src.GetRasterBand(1).GetNoDataValue()
            self._band_count = gdal_src.RasterCount

    def open_gdal_source_read(self):
        """usage;
        with self.open_gdal_source_read() as gdal_src: doesnt work.
        just dont write it to the class, and it should be fine..
        """
        return gdal.Open(self.base, gdal.GA_ReadOnly)

    def open_gdal_source_write(self):
        """Open source with write access"""
        return gdal.Open(self.base, gdal.GA_Update)

    def unlink(self, missing_ok=True):
        """Remove raster if it exists, reset source."""
        self.path.unlink(missing_ok=missing_ok)
        if not self.exists():
            self.source_set = False

    def exists(self):
        file_exists = super().exists()

        if file_exists:
            if self.source_set:  # check this first for speed.
                return True
            else:
                self.source  # set source
                return True
        else:
            if self.source_set:
                self.source_set = False
            return False

    @property
    def nodata(self):
        if self.exists():
            return self._nodata

    @property
    def band_count(self):
        if self.exists():
            return self._band_count

    @property
    def metadata(self):
        if self.exists():
            return self._metadata

    def plot(self):
        plt.imshow(self._array)

    @property
    def shape(self):
        return self.metadata.shape

    @property
    def pixelarea(self):
        return self.metadata.pixelarea

    def statistics(self, approx_ok=False, force=True) -> dict:
        """
        Note that if approx_ok is first run with True and then False, and then True it will not
        return the same results.
        The False will create a more 'accurate' result, saves it in the xml. It will then use that.
        The force parameter doesnt change that.

        Parameters
        ----------
        approx_ok : bool
            If TRUE statistics may be computed based on overviews or a subset of all tiles.
        force : bool
            FALSE statistics will only be returned if it can be done without rescanning the image.
            If TRUE, statistics computation will be forced if pre-existing values are not quickly available.
        returns [min, max, mean, std]
        """
        raster_src = self.open_gdal_source_read()
        stats = raster_src.GetRasterBand(1).GetStatistics(approx_ok, force)  # [min, max, mean, std]
        d = 6  # decimals
        return {
            "min": np.round(stats[0], d),
            "max": np.round(stats[1], d),
            "mean": np.round(stats[2], d),
            "std": np.round(stats[3], d),
        }

    def generate_blocks(self, blocksize_from_source: bool = False) -> pd.DataFrame:
        """Generate blocks with the blocksize of the band.
        These blocks can be used as window to load the raster iteratively.
        blocksize_from_source (bool): read the blocksize from the source raster
            if its bigger than min_blocksize, use that.
        """

        if blocksize_from_source:
            gdal_src = self.open_gdal_source_read()
            band = gdal_src.GetRasterBand(1)
            block_height, block_width = band.GetBlockSize()
            band.FlushCache()  # close file after writing
            band = None
        else:
            block_height = 0
            block_width = 0

        if (block_height < self.min_block_size) or (block_width < self.min_block_size):
            block_height = self.min_block_size
            block_width = self.min_block_size

        ncols = int(np.floor(self.metadata.x_res / block_width))
        nrows = int(np.floor(self.metadata.y_res / block_height))

        # Create arrays with index of where windows end. These are square blocks.
        xparts = np.linspace(0, block_width * ncols, ncols + 1).astype(int)
        yparts = np.linspace(0, block_height * nrows, nrows + 1).astype(int)

        # If raster has some extra data that didnt fall within a block it is added to the parts here.
        # These blocks are not square.
        if block_width * ncols != self.shape[1]:
            xparts = np.append(xparts, self.shape[1])
            ncols += 1
        if block_height * nrows != self.shape[0]:
            yparts = np.append(yparts, self.shape[0])
            nrows += 1

        blocks_df = pd.DataFrame(index=np.arange(nrows * ncols) + 1, columns=["ix", "iy", "window"])
        i = 0
        for ix in range(ncols):
            for iy in range(nrows):
                i += 1
                blocks_df.loc[i, :] = np.array(
                    (ix, iy, [xparts[ix], yparts[iy], xparts[ix + 1], yparts[iy + 1]]), dtype=object
                )

        blocks_df["window_readarray"] = blocks_df["window"].apply(
            lambda x: [int(x[0]), int(x[1]), int(x[2] - x[0]), int(x[3] - x[1])]
        )

        self.blocks = blocks_df
        return blocks_df

    def _generate_blocks_geometry_row(self, window):
        minx = self.metadata.x_min
        maxy = self.metadata.y_max

        # account for pixel size
        minx += window[0] * self.metadata.pixel_width
        maxy += window[1] * self.metadata.pixel_height
        maxx = minx + window[2] * self.metadata.pixel_width
        miny = maxy + window[3] * self.metadata.pixel_height

        return geometry.box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    def generate_blocks_geometry(self) -> gpd.GeoDataFrame:
        """Create blocks with shapely geometry"""
        self.blocks = self.generate_blocks()
        self.blocks = gpd.GeoDataFrame(
            self.blocks,
            geometry=self.blocks["window_readarray"].apply(self._generate_blocks_geometry_row),
            crs=self.metadata.projection,
        )
        return self.blocks

    def sum_labels(self, labels_raster, labels_index):
        """Calculate the sum of the rastervalues per label."""
        if labels_raster.shape != self.shape:
            raise Exception(f"label raster shape {labels_raster.shape} does not match the raster shape {self.shape}")

        accum = None

        if not hasattr(self, "blocks"):
            _ = self.generate_blocks()

        for idx, block_row in tqdm(self.blocks.iterrows(), total=len(self.blocks)):
            window = block_row["window_readarray"]
            block = self._read_array(window=window)
            block[block == self.nodata] = 0
            block[pd.isna(block)] = 0

            block_label = labels_raster._read_array(window=window)

            # Calculate sum per label (region)
            result = ndimage.sum_labels(
                input=block, labels=block_label, index=labels_index
            )  # Which values in labels to take into account.

            if accum is None:
                accum = result
            else:
                accum += result
        return accum

    def iter_window(self, min_block_size=None):
        """Iterate of the raster using blocks, only returning the window, not the values."""
        if not hasattr(self, "blocks") or min_block_size is not None:
            if min_block_size is not None:
                self.min_block_size = min_block_size

            _ = self.generate_blocks_geometry()

        for idx, block_row in self.blocks.iterrows():
            window = block_row["window_readarray"]
            yield idx, window, block_row

    def to_file(self):
        pass

    def build_vrt(self, overwrite: bool, bounds, input_files: list, resolution="highest", bandlist=[1]):
        """Build vrt from input files.
        overwrite (bool)
        bounds (np.array): format should be; (xmin, ymin, xmax, ymax)
            if None will use input files. -> hrt.Raster.metadata.bbox_gdal
        input_files (list): list of paths to input rasters
        resolution: "highest"|"lowest"|"average"
            instead of "user" option, provide a float for manual target_resolution
        bandList: doesnt work as expected, passing [1] works.
        """
        if hrt.check_create_new_file(output_file=self.path, overwrite=overwrite):
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
                r = Raster(r)
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
            ds = gdal.BuildVRT(destName=str(self.path), srcDSOrSrcDSTab=input_files, options=vrt_options)
            ds.FlushCache()

    # FIXME hoe hier het beste omgaan met Folder zonder circular imports?
    # def build_vrt_from_folder(self, folder_path):
    #     """"""
    #     raster_folder = Folder(raster_folder)
    #     output_path = raster_folder.full_path(f'{vrt_name}.vrt')

    #     if output_path.exists() and not overwrite:
    #         print(f'vrt already exists: {output_path}')
    #         return

    #     tifs_list = [str(i) for i in raster_folder.find_ext(["tif", "tiff"])]

    def write_array(self, array, window: list, band=None):
        """Note that providing the band may be faster.

        array (np.array([])): block or raster array
        window (list): [x0, y0, xsize, ysize]
        x0, y0 is left top corner!!
        """
        flushband = False
        if band is None:
            gdal_src = self.open_gdal_source_write()
            band = gdal_src.GetRasterBand(1)
            flushband = True
        else:
            print("At this point just use band.WriteArray, no need for this function.")

        band.WriteArray(array, xoff=window[0], yoff=window[1])

        if flushband:
            # Only flush band if it was not provided
            # band.FlushCache()  # close file after writing
            band = None

    def __iter__(self):
        if not hasattr(self, "blocks"):
            _ = self.generate_blocks()

        for idx, block_row in self.blocks.iterrows():
            window = block_row["window_readarray"]
            block = self._read_array(window=window)
            yield window, block

    def __repr__(self):
        if self.exists():
            return f"""{self.path.name} @ {self.path}
exists: {self.exists()}
type: {type(self)}
shape: {self.metadata.shape}
pixelsize: {self.metadata.pixel_width}

functions: {get_functions(self)}
variables: {get_variables(self)}
"""

        else:
            return f"""{self.path.name} @ {self.path}
exists: {self.exists()}
type: {type(self)}

functions: {get_functions(self)}
variables: {get_variables(self)}
"""

    def create(
        self,
        metadata,
        nodata: float,
        datatype: Optional[int] = None,  # e.g. gdal.GDT_Float32
        create_options=None,
        verbose: bool = False,
        overwrite: bool = False,
    ):
        """Create empty raster

        metadata (RasterMetadata): metadata
        nodata (int): nodata value
        """
        # Check if function should continue.
        if verbose:
            print(f"creating output raster: {self.path}")
        target_ds = hrt.create_new_raster_file(
            file_name=self.path,
            nodata=nodata,
            meta=metadata,
            datatype=datatype,
            create_options=create_options,
            overwrite=overwrite,
        )
        target_ds = None

        # Reset source, if raster is deleted and recreated with different resolution
        # this would otherwise cause issues.
        self.source_set = False
        self.source = None  # Update raster now it exists

    def sum(self):
        """Calculate sum of raster"""
        raster_sum = 0
        for window, block in self:
            block[block == self.nodata] = 0
            raster_sum += np.nansum(block)
        return raster_sum


class RasterMetadata:
    """Metadata object of a raster. Resolution can be changed
    so that a new raster with another resolution can be created.

    Metadata can be created by supplying either:
    1. gdal_src
    2. res, bounds
    """

    def __init__(self, gdal_src=None, res: float = None, bounds_dict=None, proj="epsg:28992"):
        """
        gdal_src = gdal.Open(raster_source)
        bounds = {"minx":, "maxx":, "miny":, "maxy":}
        Projection only implemented for epsg:28992
        """

        if gdal_src is not None:
            self.proj = gdal_src.GetProjection()
            self.georef = gdal_src.GetGeoTransform()

            self.x_res = gdal_src.RasterXSize
            self.y_res = gdal_src.RasterYSize

        elif res is not None and bounds_dict is not None:
            projections = {
                "epsg:28992": 'PROJCS["Amersfoort / RD New",GEOGCS["Amersfoort",DATUM["Amersfoort",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],TOWGS84[565.2369,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812],AUTHORITY["EPSG","6289"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4289"]],PROJECTION["Oblique_Stereographic"],PARAMETER["latitude_of_origin",52.15616055555555],PARAMETER["central_meridian",5.38763888888889],PARAMETER["scale_factor",0.9999079],PARAMETER["false_easting",155000],PARAMETER["false_northing",463000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","28992"]]'
            }

            self.proj = projections[proj]
            self.georef = (int(np.floor(bounds_dict["minx"])), res, 0.0, int(np.ceil(bounds_dict["maxy"])), 0.0, -res)
            self.x_res = int((int(np.ceil(bounds_dict["maxx"])) - int(np.floor(bounds_dict["minx"]))) / res)
            self.y_res = int((int(np.ceil(bounds_dict["maxy"])) - int(np.floor(bounds_dict["miny"]))) / res)

        else:
            raise Exception("Metadata class called without proper input.")

    @property
    def pixel_width(self):
        return self.georef[1]

    @property
    def pixel_height(self):
        return self.georef[5]

    @property
    def x_min(self):
        return self.georef[0]

    @property
    def y_max(self):
        return self.georef[3]

    @property
    def x_max(self):
        return self.x_min + self.georef[1] * self.x_res

    @property
    def y_min(self):
        return self.y_max + self.georef[5] * self.y_res

    @property
    def bounds(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max]

    # TODO deprecated. Remove in future release.
    @property
    def bounds_dl(self):
        """Lizard v3 bounds"""
        raise Exception("use .bbox instead. lizard v4 api no longer supports bounds_dl")
        return {
            "west": self.x_min,
            "south": self.y_min,
            "east": self.x_max,
            "north": self.y_max,
        }

    @property
    def bbox(self):
        """Lizard v4 bbox; str(x1, y1, x2, y2)"""
        return f"{self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}"

    @property
    def bbox_gdal(self):
        """Gdal takes bbox as list, for instance in vrt creation."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @property
    def shape(self):
        return [self.y_res, self.x_res]

    @property
    def pixelarea(self):
        return abs(self.georef[1] * self.georef[5])

    @property
    def projection(self):
        try:
            proj_str = self.proj.split("AUTHORITY")[-1][2:-3].split('","')
            return f"{proj_str[0]}:{proj_str[1]}"
        except:
            return None

    def _update_georef(self, resolution):
        def res_str(georef_i):
            """Make sure negative values are kept."""
            if georef_i == self.pixel_width:
                return resolution
            if georef_i == -self.pixel_width:
                return -resolution

        georef_new = list(self.georef)
        georef_new[1] = res_str(georef_new[1])
        georef_new[5] = res_str(georef_new[5])
        return tuple(georef_new)

    def update_resolution(self, resolution_new: Union[float, int]):
        """Create new resolution metdata, only works for refining now."""
        resolution_current = self.pixel_width
        if (resolution_current / resolution_new).is_integer():
            self.x_res = int((resolution_current / resolution_new) * self.x_res)
            self.y_res = int((resolution_current / resolution_new) * self.y_res)
            self.georef = self._update_georef(resolution_new)
            print(f"updated metadata resolution from {resolution_current}m to {resolution_new}m")
        else:
            raise Exception(
                f"New resolution ({resolution_new}) can currently only be smaller than old resolution ({resolution_current})"
            )

    def __repr__(self):
        funcs = (
            "."
            + " .".join(
                [
                    i
                    for i in dir(self)
                    if not i.startswith("_") and hasattr(inspect.getattr_static(self, i), "__call__")
                ]
            )
        )  # getattr resulted in RecursionError. https://stackoverflow.com/questions/1091259/how-to-test-if-a-class-attribute-is-an-instance-method
        variables = "." + " .".join(
            [
                i
                for i in dir(self)
                if not i.startswith("__") and not hasattr(inspect.getattr_static(self, i), "__call__")
            ]
        )
        repr_str = f"""functions: {funcs}
variables: {variables}"""
        return f""".projection : {self.projection} 
.georef : {self.georef}
.bounds : {self.bounds}
.pixel_width : {self.pixel_width}
----
{repr_str}"""

    def __getitem__(self, item):
        """Metadata was a dict previously. This makes it that items from
        the class can be accessed like a dict.
        """
        return getattr(self, item)


if __name__ == "__main__":
    dem_path = Path(
        r"G:\02_Werkplaatsen\06_HYD\Projecten\HKC16015 Wateropgave 2.0\11. DCMB\hhnk-modelbuilder-master\data\fixed_data\DEM\DEM_AHN4_int.vrt"
    )
    if dem_path.exists():
        r = Raster(str(dem_path))
        print(r)
