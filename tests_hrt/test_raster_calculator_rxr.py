# %%

from typing import Union

import numpy as np
import xarray as xr

import hhnk_research_tools as hrt
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY


class RasterCalculatorRxrTest(hrt.RasterCalculatorRxr):
    def __init__(
        self,
        raster_out: hrt.Raster,
        raster_paths_dict: dict[str : hrt.Raster],
        metadata_key: str,
        nodata_keys: list[str],
        tempdir: hrt.Folder = None,
    ):
        super().__init__(
            raster_out=raster_out,
            raster_paths_dict=raster_paths_dict,
            metadata_key=metadata_key,
            tempdir=tempdir,
        )
        self.nodata_keys = nodata_keys
        # Local vars

    def run(self, chunksize: Union[int, None] = None, overwrite: bool = False):
        def calc_diff(da_out, da_depth, da_lu):
            lu_mask = xr.where(da_lu != 4, True, False)
            da_out = xr.where(da_depth, np.interp(x=da_depth.where(lu_mask), xp=[-1, 0, 1], fp=[-100, 0, 100]), 0)
            return da_out

        cont = self.verify(overwrite=overwrite)

        if cont:
            # Load dataarrays
            da_dict = {}
            da_depth = da_dict["depth"] = self.raster_paths_same_bounds["depth"].open_rxr(chunksize)
            da_lu = da_dict["lu"] = self.raster_paths_same_bounds["lu"].open_rxr(chunksize)

            nodata = da_depth.rio.nodata

            # Create global no data mask
            da_nodatamasks = self.get_nodatamasks(da_dict=da_dict, nodata_keys=self.nodata_keys)

            # Mask where out storage should be zero
            zeromasks = {}
            zeromasks["nodepth"] = da_depth == 0
            zeromasks["negative_dewa"] = da_depth < 0
            # Stack the conditions into a single DataArray
            da_zeromasks = self.concat_masks(zeromasks)
            da_out = xr.full_like(da_depth, da_depth.rio.nodata)

            # Create result array
            da_out = xr.map_blocks(
                calc_diff,
                obj=da_out,
                args=[da_depth, da_lu],
                template=da_out,
            )

            # Apply nodata and zero values
            da_out = xr.where(da_zeromasks, 0, da_out)
            da_out = xr.where(da_nodatamasks, nodata, da_out)

            self.raster_out = hrt.Raster.write(self.raster_out, result=da_out, nodata=nodata, chunksize=chunksize)
            return self.raster_out
        else:
            print("Cont is False")


def test_raster_calc_rxr():
    raster_depth = hrt.Raster(TEST_DIRECTORY / r"depth_test.tif")
    raster_lu = hrt.Raster(TEST_DIRECTORY / r"lu_small.tif")

    raster_out = hrt.Raster(TEMP_DIR / f"rastercalc_{hrt.get_uuid()}.tif")
    raster_paths_dict = {
        "depth": raster_depth,
        "lu": raster_lu,
    }

    nodata_keys = ["depth"]
    metadata_key = "depth"

    chunksize = 1024
    overwrite = True
    self = RasterCalculatorRxrTest(
        raster_out=raster_out,
        raster_paths_dict=raster_paths_dict,
        metadata_key=metadata_key,
        nodata_keys=nodata_keys,
        tempdir=None,
    )
    self.run(chunksize=chunksize, overwrite=overwrite)
    assert int(np.nansum(self.raster_out.open_rxr().values)) == 182843


# %%
