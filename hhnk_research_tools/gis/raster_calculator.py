import datetime
import json
import types
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

import hhnk_research_tools as hrt


@dataclass
class RasterBlocks:
    """
    General function to load blocks of selected files with a given window.
    Also loads the masks and can check if a block is fully nodata, in which
    case it stopts loading.
    Input files should have the same extent, so make a vrt of them first if
    they are not. This is handled in RasterCalculator.

    For speed this class does not check if all inputs exist. This should still
    be the case.

    Parameters
    ----------
    window (list): [xmin, ymin, xsize, ysize]
        same as row['windows_readarray']
    raster_paths_dict (dict): {key:hrt.Raster}
        path items should be of type hrt.Raster.
    nodata_keys (list):
        the keys in raster_paths to check for all nodata values wont load other
        rasters if all values are nodata.
    yesdata_dict (dict): {key:list[float]}
        Inverse of nodata_keys. Checks if any of of the provided values in the
        list are available. Creates a mask of all values not equal.
    mask_keys (list[str]):
        Keys to add to nodatamask. Keys already listed in nodata_keys and yesdata_dict
        do not have to be defined here.
    """

    window: list
    raster_paths_dict: dict[str, hrt.Raster]
    nodata_keys: Optional[list[str]] = None
    yesdata_dict: Optional[dict[str, list[float]]] = None
    mask_keys: Optional[list[str]] = None

    def __post_init__(self):
        self.cont = True
        self.blocks = {}
        self.masks = {}

        try:
            # Creates a mask of all values equal to nodata
            if self.nodata_keys is not None:
                for key in self.nodata_keys:
                    self.blocks[key] = self.read_array_window(key)

                    if np.isnan(self.raster_paths_dict[key].nodata):
                        self.masks[key] = np.isnan(self.blocks[key])
                    else:
                        self.masks[key] = self.blocks[key] == self.raster_paths_dict[key].nodata

                    if np.all(self.masks[key]):
                        # if all values in masks are nodata then we can break loading
                        self.cont = False
                        break

            # Creates a mask of all values not equal to values in key list
            if self.yesdata_dict is not None:
                for key, val in self.yesdata_dict.items():
                    self.blocks[key] = self.read_array_window(key)
                    self.masks[key] = ~np.isin(self.blocks[key], [int(i) for i in val])  # inverse matches.

                    if np.all(self.masks[key]):
                        # if all values in masks are nodata then we can break loading
                        self.cont = False
                        break

            # Load other rasters if masks are not all True.
            if self.cont:
                for key in self.raster_paths_dict:
                    if key not in self.blocks:
                        self.blocks[key] = self.read_array_window(key)
                    if (key in self.mask_keys) and (key not in self.masks):
                        if np.isnan(self.raster_paths_dict[key].nodata):
                            self.masks[key] = np.isnan(self.blocks[key])
                        else:
                            self.masks[key] = self.blocks[key] == self.raster_paths_dict[key].nodata

        except Exception as e:
            raise e

    def read_array_window(self, key):
        """Read window from hrt.Raster"""
        return self.raster_paths_dict[key]._read_array(window=self.window)

    @property
    def masks_all(self):
        """Combine nodata masks"""
        return np.any([self.masks[i] for i in self.masks], 0)


class RasterCalculatorV2:
    """
    Base setup for raster calculations. The input rasters defined in raster_paths_dict
    are looped over per block. Note that all input rasters should have the same extent.
    This can be achieved with .vrt if the original rasters do not have the same extent.
    For each block the custom_run_window_function will be run. This always takes a
    block as input and also returns the block. For example:

    def run_dem_window(block):
        block_out = block.blocks['dem']

        #Watervlakken ophogen naar +10mNAP
        block_out[block.blocks['watervlakken']==1] = 10

        block_out[block.masks_all] = nodata
        return block_out

    Parameters
    ----------
    raster_out (hrt.Raster): output raster location
    raster_paths_dict (dict[str : hrt.Raster]): these rasters will have blocks loaded.
    nodata_keys (list [str]): keys to check if all values are nodata, if yes then skip
    mask_keys (list[str]):
        Keys to add to nodatamask. Keys already listed in nodata_keys and yesdata_dict
        do not have to be defined here.
    metadata_key (str): key in raster_paths_dict that will be used to
        create blocks and metadata
    custom_run_window_function: function that does calculation with blocks.
        function takes block (hrt.RasterBlocks) and kwargs as input and must return block
    yesdata_dict (dict): {key:list[float]}
        Inverse of nodata_keys. Checks if any of of the provided values in the
        list are available. Creates a mask of all values not equal.
    output_nodata (int): nodata of output raster
    min_block_size (int): min block size for generator blocks_df, higher is faster but
        uses more RAM.
    verbose (bool): print progress
    tempdir (hrt.Folder): pass if you want temp vrt's to be created in a specific tempdir
    """

    def __init__(
        self,
        raster_out: hrt.Raster,
        raster_paths_dict: dict[str : hrt.Raster],
        nodata_keys: list[str],
        mask_keys: list[str],
        metadata_key: str,
        custom_run_window_function: types.MethodType,
        yesdata_dict: dict[str : list[float]] = None,
        output_nodata: int = -9999,
        min_block_size: int = 4096,
        verbose: bool = False,
        tempdir: hrt.Folder = None,
    ):
        self.raster_out = raster_out
        self.raster_paths_dict = raster_paths_dict
        self.nodata_keys = nodata_keys
        self.mask_keys = mask_keys
        self.metadata_key = metadata_key
        self.custom_run_window_function = custom_run_window_function
        self.yesdata_dict = yesdata_dict
        self.output_nodata = output_nodata
        self.min_block_size = min_block_size
        self.verbose = verbose

        # Local vars
        if tempdir is None:
            self.tempdir = raster_out.parent.full_path(f"temp_{hrt.current_time(date=True)}")
        else:
            self.tempdir = tempdir

        # If bounds of input rasters are not the same a temp vrt is created
        # The path to these files are stored here.
        self.raster_paths_same_bounds = self.raster_paths_dict.copy()

        # Filled when running
        self.blocks_df: pd.DataFrame

    @property
    def metadata_raster(self) -> hrt.Raster:
        """Raster of which metadata is used to create output."""
        return self.raster_paths_dict[self.metadata_key]

    def verify(self, overwrite: bool = False) -> bool:
        """Verify if all inputs can be accessed and if they have the same bounds."""
        cont = True

        # Check if all input rasters have the same bounds
        bounds = {}
        for key, r in self.raster_paths_dict.items():
            if cont:
                if not isinstance(r, hrt.Raster):
                    raise TypeError(f"{key}:{r} in raster_paths_dict is not of type hrt.Raster")
                if not r.exists():
                    print(f"Missing input raster key: {key} @ {r}")
                    cont = False
                    continue
                bounds[key] = r.metadata.bounds

        # nodata_keys and yesdata_dict are mutually exclusive.
        if self.yesdata_dict is not None:
            for key in self.yesdata_dict:
                if key in self.nodata_keys:
                    raise ValueError(f"Key:'{key}' not allowed to be passed to both yesdata_dict and nodata_keys.")

        # Check resolution
        if cont:
            vrt_keys = []
            for key, r in self.raster_paths_dict.items():
                if r.metadata.pixelarea > self.metadata_raster.metadata.pixelarea:
                    print(f"Resolution of {key} is not the same as metadataraster {self.metadata_key}, creating vrt")
                    self.create_vrt(key)
                    vrt_keys.append(key)
                if r.metadata.pixelarea < self.metadata_raster.metadata.pixelarea:
                    cont = False
                    raise NotImplementedError(
                        f"Resolution of {key} is smaller than metadataraster {self.metadata_key}, \
this is not implemented or tested if it works."
                    )

        # Check bounds, if they are not the same as the metadata_raster, create a vrt
        if cont:
            for key, r in self.raster_paths_dict.items():
                if r.metadata.bounds != self.metadata_raster.metadata.bounds:
                    # Create vrt if it was not already created in resolution check
                    if key not in vrt_keys:
                        self.create_vrt(key)

                    if self.verbose:
                        print(f"{key} does not have same extent as {self.metadata_key}, creating vrt")

        # Check if we should create new file
        if cont:
            if self.raster_out is not None:
                cont = hrt.check_create_new_file(output_file=self.raster_out, overwrite=overwrite)
                if cont is False:
                    if self.verbose:
                        print(f"output raster already exists: {self.raster_out.name} @ {self.raster_out.path}")

        return cont

    def create(self):
        """Create empty output raster with metadata of metadata_raster"""
        if self.verbose:
            print(f"Creating output raster: {self.raster_out.name} @ {self.raster_out.path}")

        self.raster_out.create(metadata=self.metadata_raster.metadata, nodata=self.output_nodata)

    def create_vrt(self, raster_key: str):
        """Create vrt of input rasters with the extent of the metadata raster

        Parameters
        ----------
        raster_key (str) : key in self.raster_paths_dict to create vrt from.
        """
        input_raster = self.raster_paths_dict[raster_key]

        # Create temp output folder.
        self.tempdir.mkdir()
        output_raster = self.tempdir.full_path(f"{input_raster.stem}.vrt")
        print(f"Creating temporary vrt; {output_raster.name} @ {output_raster}")

        output_raster.build_vrt(
            vrt_out=output_raster,
            overwrite=True,
            bounds=self.metadata_raster.metadata.bbox_gdal,
            input_files=input_raster,
            resolution=self.metadata_raster.metadata.pixel_width,
        )

        self.raster_paths_same_bounds[raster_key] = output_raster

    def run(self, overwrite: bool = False, **kwargs):
        """Start raster calculation.

        Parameters
        ----------
        overwrite : bool, optional, by default False
            False -> if output already exists this will not run.
            True  -> remove existing output and continue
        **kwargs:
            extra arguments that can be passed to the custom_run_window_function
        """

        try:
            cont = self.verify(overwrite=overwrite)
            if cont:
                self.create()

            if cont:
                # Create blocks dataframe
                self.metadata_raster.min_block_size = self.min_block_size
                self.blocks_df = self.metadata_raster.generate_blocks()

                if self.verbose:
                    time_start = datetime.datetime.now()
                    blocks_total = len(self.blocks_df)

                # # Open output raster for writing
                gdal_src = self.raster_out.open_gdal_source_write()
                band_out = gdal_src.GetRasterBand(1)

                # Loop over generated blocks and do calculation per block
                for idx, block_row in self.blocks_df.iterrows():
                    window = block_row["window_readarray"]

                    # Load the blocks for the given window.
                    block = RasterBlocks(
                        window=window,
                        raster_paths_dict=self.raster_paths_same_bounds,
                        nodata_keys=self.nodata_keys,
                        yesdata_dict=self.yesdata_dict,
                        mask_keys=self.mask_keys,
                    )

                    # The blocks have an attribute that can prevent further calculation
                    # if certain conditions are met. It is False when a raster in the
                    # nodata keys has all value as nodata. Output should be nodata as well
                    if block.cont:
                        # Calculate output raster block with custom function.
                        block_out = self.custom_run_window_function(block=block, **kwargs)

                        band_out.WriteArray(block_out, xoff=window[0], yoff=window[1])

                        if self.verbose:
                            print(
                                f"{idx} / {blocks_total} ({hrt.time_delta(time_start)}s) - {self.raster_out.name}",
                                end="\r",
                            )

                # band_out.FlushCache()  # close file after writing, slow, needed?
                gdal_src = None  # Very important..
                band_out = None
                if self.verbose:
                    print("\nDone")
            else:
                if self.verbose:
                    print(f"{self.raster_out.name} not created, .verify was false.")
        except Exception as e:
            band_out.FlushCache()
            band_out = None
            self.raster_out.unlink()
            raise e

    def run_label_stats(
        self,
        label_gdf: gpd.GeoDataFrame,
        label_col: str,
        stats_json: hrt.File,
        decimals: int,
        **kwargs,
    ):
        """Create statistics per label (shape in a shapefile). The shapefile must be rasterized.
        This label_raster is passed by the metadata_key.

        Will create a histogram of all value counts per label. Output is a dictionary
        per label with the value counts. Takes into account decimals, as only integers
        are saved. self.outputdata is ignored.
        Example output:
        {
            'DECIMALS': 0,
            '0': {'2': 61, '6': 2358, '15': 267, '28': 1005, '29': 2262, '241': 279},
            '2': {'2': 2144, '6': 470, '15': 756, '28': 664, '29': 2880, '241': 302},
        }
        TODO issue #56 nagaan '241' staat onterecht hier.

        Parameters
        ----------
        label_gdf : gpd.GeoDataFrame
            Dataframe with the shapes to calculate statistics on. This dataframe must also
            have a raster which must be set as metadata_key.
        label_col : str
            Column of label_gdf that was used to create label_raster
        statistics_json : hrt.File
            .json file with statistics per label.
        decimals : int
            number of decimals to save. The result will be saves as integer.
            example with value 1.23;
                decimals=0 -> 1
                decimals=2 -> 123
        """
        try:
            cont = self.verify()

            if cont:
                if self.verbose:
                    print("Starting run_label_stats")
                    time_start = datetime.datetime.now()
                    blocks_total = len(label_gdf)

                if stats_json.exists():
                    stats_dict = json.loads(stats_json.path.read_text())
                else:
                    stats_dict = {"DECIMALS": decimals}

                # For each label calculate the statistics. This is done by creating metadata from the
                # label and then looping the blocks of this smaller part.
                calc_count = 0

                for index, (row_index, row_label) in enumerate(label_gdf.iterrows()):
                    key = f"{row_index}"

                    cont2 = True
                    if key in stats_dict:
                        if stats_dict[key] != {}:
                            cont2 = False

                    if cont2:
                        meta = hrt.RasterMetadataV2.from_gdf(
                            label_gdf.loc[[row_index]],
                            res=self.metadata_raster.metadata.pixel_width,
                        )

                        # Hack the metadata into a dummy raster file so we can create blocks
                        r = hrt.Raster("dummy")
                        r._metadata = meta

                        def exists_dummy():
                            return True

                        r.exists = exists_dummy

                        blocks_df = hrt.RasterChunks.to_df(r)

                        # Difference between single label and bigger raster
                        dx_dy_label = hrt.dx_dy_between_rasters(
                            meta_big=self.metadata_raster.metadata, meta_small=meta
                        )

                        # Calculate histogram, valuecounts per label
                        hist_label = {}

                        # Iterate over generated blocks and do calculation
                        for idx, block_row in blocks_df.iterrows():
                            window_label = block_row["window_readarray"]
                            window_big = window_label.copy()
                            window_big[0] += dx_dy_label[0]
                            window_big[1] += dx_dy_label[1]

                            # Load the blocks for the given window.
                            block = RasterBlocks(
                                window=window_big,
                                raster_paths_dict=self.raster_paths_same_bounds,
                                nodata_keys=self.nodata_keys,
                                yesdata_dict={self.metadata_key: [row_label[label_col]]},
                                mask_keys=self.mask_keys,
                            )

                            # The blocks have an attribute that can prevent further calculation
                            # if certain conditions are met. It is False when a raster in the
                            # nodata keys has all value as nodata. Output should be nodata as well
                            if block.cont:
                                block_out = self.custom_run_window_function(block=block, **kwargs)

                                # Create histogram of unique values of dem and count
                                val, count = np.unique(block_out, return_counts=True)
                                for v, c in zip(val, count):
                                    if np.isnan(v):
                                        v = "nodata"
                                    else:
                                        v = int(v * 10**decimals)
                                    if v not in hist_label.keys():
                                        hist_label[v] = int(c)
                                    else:
                                        hist_label[v] += int(c)

                        if self.output_nodata * 10**decimals in hist_label:
                            hist_label.pop(self.output_nodata * 10**decimals)

                        stats_dict[key] = hist_label.copy()
                        calc_count += 1

                        if self.verbose:
                            print(
                                f"{index + 1} / {blocks_total} ({hrt.time_delta(time_start)}s) - {stats_json.name}",
                                end="\r",
                            )

                    # Save intermediate results
                    if calc_count == 100:
                        calc_count = 0
                        stats_json.path.write_text(json.dumps(stats_dict))

                print("\n")
                stats_json.path.write_text(json.dumps(stats_dict))

        except Exception as e:
            raise e
