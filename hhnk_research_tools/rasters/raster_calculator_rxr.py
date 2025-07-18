import numpy as np
import pandas as pd
import xarray as xr

import hhnk_research_tools as hrt
import hhnk_research_tools.logging as logging

logger = logging.get_logger(__name__)


class RasterCalculatorRxr:
    """
    Third iteration, using rioxarray.
    Base setup for raster calculations. The input rasters in raster_paths_dict
    are converted to the same extent using vrt, if that is not the case yet. The
    metadata_key parameter determines the output profile and bounds.


    Parameters
    ----------
    raster_out (hrt.Raster): output raster location
    raster_paths_dict (dict[str : hrt.Raster]): these rasters will have blocks loaded.
    metadata_key (str): key in raster_paths_dict that will be used to
        create blocks and metadata
    tempdir (hrt.Folder): pass if you want temp vrt's to be created in a specific tempdir
    """

    def __init__(
        self,
        raster_out: hrt.Raster,
        raster_paths_dict: dict[str : hrt.Raster],
        metadata_key: str,
        tempdir: hrt.Folder = None,
    ):
        self.raster_out = raster_out
        self.raster_paths_dict = raster_paths_dict
        self.metadata_key = metadata_key

        # Local vars
        self.tempdir = tempdir
        if tempdir is None:
            self.tempdir = raster_out.parent.full_path(f"temp_{hrt.current_time(date=True)}")

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
        """Verify if all inputs can be accessed and if they have the same bounds.
        Create vrt based on metadata raster bounds and resolution
        """
        cont = True

        # Check if all input rasters have the same bounds
        bounds = {}
        error = None
        for key, r in self.raster_paths_dict.items():
            if cont:
                if not isinstance(r, hrt.Raster):
                    raise TypeError(f"{key}:{r} in raster_paths_dict is not of type hrt.Raster")
                if not r.exists():
                    logger.error(f"Missing input raster key: {key} @ {r}")
                    cont = False
                    error = f"{key}: {r} does not exist"
                    continue
                bounds[key] = r.metadata.bounds
        if error:
            raise FileNotFoundError(error)

        # Check resolution
        vrt_keys = []
        if cont:
            for key, r in self.raster_paths_dict.items():
                if r.metadata.pixelarea > self.metadata_raster.metadata.pixelarea:
                    logger.info(
                        f"Resolution of {key} is not the same as metadataraster {self.metadata_key}, creating temporary vrt"
                    )
                    self._create_vrt(key)
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
                        self._create_vrt(key)
                        vrt_keys.append(key)

                    logger.info(f"{key} does not have same extent as {self.metadata_key}, creating temporary vrt")

        # Check if we should create new file
        if cont:
            if self.raster_out is not None:
                cont = hrt.check_create_new_file(output_file=self.raster_out, overwrite=overwrite)
                if cont is False:
                    logger.info(f"Output raster already exists: {self.raster_out.name} @ {self.raster_out.path}")

        return cont

    def _create_vrt(self, raster_key: str):
        """Create vrt of input rasters with the extent of the metadata raster

        Parameters
        ----------
        raster_key (str) : key in self.raster_paths_dict to create vrt from.
        """
        input_raster = self.raster_paths_dict[raster_key]

        # Create temp output folder.
        self.tempdir.mkdir()
        output_raster = self.tempdir.full_path(f"{input_raster.stem}.vrt")

        output_raster = hrt.Raster.build_vrt(
            vrt_out=output_raster,
            input_files=input_raster,
            overwrite=True,
            bounds=self.metadata_raster.metadata.bbox_gdal,
            resolution=self.metadata_raster.metadata.pixel_width,
        )

        self.raster_paths_same_bounds[raster_key] = output_raster

    def get_nodatamasks(self, da_dict: dict, nodata_keys: list[str]):
        """Create nodata masks of selected nodata_keys
        Keys are passed with the nodata_keys variable.

        Parameters
        ----------
        da_dict : dict
            DataArrays stored in dictionary
        nodata_keys : list [str]
            Keys to check if all values are nodata, if yes then skip
        """
        nodatamasks = {}
        for key in nodata_keys:
            nodata = da_dict[key].rio.nodata
            if np.isnan(nodata):
                nodatamasks[key] = da_dict[key].isnull()  # noqa PD003, isna is not an option on xr.da
            else:
                nodatamasks[key] = da_dict[key] == nodata

        da_nodatamasks = self.concat_masks(nodatamasks)
        return da_nodatamasks

    def concat_masks(self, masks_dict: dict) -> xr.DataArray:
        """Concat DataArray from a dict of arrays.

        Returns
        -------
        Single DataArray where any of the dict arrays was True.
        """
        return xr.concat(list(masks_dict.values()), dim="condition").any(dim="condition")

    # def run(self, **kwargs):
    #     """Define your own run function when using this calcualtor."""
    #     raise NotImplementedError("""The raster calculator should have a run function. Example in;
    #     hhnk_threedi_tools\core\raster_creation\storage_raster.py""")

    # TODO used in statistiek stedelijk
    # def _run_label_stats(
    #     self,
    #     label_gdf: gpd.GeoDataFrame,
    #     label_col: str,
    #     stats_json: hrt.File,
    #     decimals: int,
    #     **kwargs,
    # ):
    #     """Create statistics per label (shape in a shapefile). The shapefile must be rasterized.
    #     This label_raster is passed by the metadata_key.

    #     Will create a histogram of all value counts per label. Output is a dictionary
    #     per label with the value counts. Takes into account decimals, as only integers
    #     are saved. self.outputdata is ignored.
    #     Example output:
    #     {
    #         'DECIMALS': 0,
    #         '0': {'2': 61, '6': 2358, '15': 267, '28': 1005, '29': 2262, '241': 279},
    #         '2': {'2': 2144, '6': 470, '15': 756, '28': 664, '29': 2880, '241': 302},
    #     }

    #     Parameters
    #     ----------
    #     label_gdf : gpd.GeoDataFrame
    #         Dataframe with the shapes to calculate statistics on. This dataframe must also
    #         have a raster which must be set as metadata_key.
    #     label_col : str
    #         Column of label_gdf that was used to create label_raster
    #     statistics_json : hrt.File
    #         .json file with statistics per label.
    #     decimals : int
    #         number of decimals to save. The result will be saves as integer.
    #         example with value 1.23;
    #             decimals=0 -> 1
    #             decimals=2 -> 123
    #     """
    #     try:
    #         cont = self.verify()

    #         if cont:
    #             if self.verbose:
    #                 print("Starting run_label_stats")
    #                 time_start = datetime.datetime.now()
    #                 blocks_total = len(label_gdf)

    #             if stats_json.exists():
    #                 stats_dict = json.loads(stats_json.path.read_text())
    #             else:
    #                 stats_dict = {"DECIMALS": decimals}

    #             # For each label calculate the statistics. This is done by creating metadata from the
    #             # label and then looping the blocks of this smaller part.
    #             calc_count = 0

    #             for index, (row_index, row_label) in enumerate(label_gdf.iterrows()):
    #                 key = f"{row_index}"

    #                 cont2 = True
    #                 if key in stats_dict:
    #                     if stats_dict[key] != {}:
    #                         cont2 = False

    #                 if cont2:
    #                     meta = hrt.RasterMetadataV2.from_gdf(
    #                         label_gdf.loc[[row_index]], res=self.metadata_raster.metadata.pixel_width
    #                     )

    #                     # Hack the metadata into a dummy raster file so we can create blocks
    #                     r = hrt.Raster("dummy")
    #                     r._metadata = meta

    #                     def exists_dummy():
    #                         return True

    #                     r.exists = exists_dummy
    #                     r.source_set = True
    #                     blocks_df = r.generate_blocks()

    #                     # Difference between single label and bigger raster
    #                     dx_dy_label = hrt.dx_dy_between_rasters(
    #                         meta_big=self.metadata_raster.metadata, meta_small=meta
    #                     )

    #                     # Calculate histogram, valuecounts per label
    #                     hist_label = {}

    #                     # Iterate over generated blocks and do calculation
    #                     for idx, block_row in blocks_df.iterrows():
    #                         window_label = block_row["window_readarray"]
    #                         window_big = window_label.copy()
    #                         window_big[0] += dx_dy_label[0]
    #                         window_big[1] += dx_dy_label[1]

    #                         # Load the blocks for the given window.
    #                         block = RasterBlocks(
    #                             window=window_big,
    #                             raster_paths_dict=self.raster_paths_same_bounds,
    #                             nodata_keys=self.nodata_keys,
    #                             yesdata_dict={self.metadata_key: [row_label[label_col]]},
    #                             mask_keys=self.mask_keys,
    #                         )

    #                         # The blocks have an attribute that can prevent further calculation
    #                         # if certain conditions are met. It is False when a raster in the
    #                         # nodata keys has all value as nodata. Output should be nodata as well
    #                         if block.cont:
    #                             block_out = self.custom_run_window_function(block=block, **kwargs)

    #                             # Create histogram of unique values of dem and count
    #                             val, count = np.unique(block_out, return_counts=True)
    #                             for v, c in zip(val, count):
    #                                 v = int(v * 10**decimals)
    #                                 if v not in hist_label.keys():
    #                                     hist_label[v] = int(c)
    #                                 else:
    #                                     hist_label[v] += int(c)

    #                     if self.output_nodata * 10**decimals in hist_label:
    #                         hist_label.pop(self.output_nodata * 10**decimals)

    #                     stats_dict[key] = hist_label.copy()
    #                     calc_count += 1

    #                     if self.verbose:
    #                         print(
    #                             f"{index+1} / {blocks_total} ({hrt.time_delta(time_start)}s) - {stats_json.name}",
    #                             end="\r",
    #                         )

    #                 # Save intermediate results
    #                 if calc_count == 100:
    #                     calc_count = 0
    #                     stats_json.path.write_text(json.dumps(stats_dict))

    #             print("\n")
    #             stats_json.path.write_text(json.dumps(stats_dict))

    #     except Exception as e:
    #         raise e
