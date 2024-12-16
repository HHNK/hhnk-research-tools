# %%
# TODO remove/replace tests
from pathlib import Path

import numpy as np
import pytest

import hhnk_research_tools as hrt
from hhnk_research_tools.gis.raster import RasterOld
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY


class TestRaster:
    raster = RasterOld(TEST_DIRECTORY / r"depth_test.tif")
    gpkg_path = TEST_DIRECTORY / r"area_test.gpkg"

    def test_properties(self):
        assert self.raster.nodata == -9999
        assert self.raster.band_count == 1
        assert self.raster.pixelarea == 0.25
        assert self.raster.statistics() == {
            "min": -0.00988,
            "max": 0.484222,
            "mean": 0.133442,
            "std": 0.09345,
        }

        gdal_src = self.raster.open_gdal_source_read()
        band = gdal_src.GetRasterBand(1)

        block_height, block_width = band.GetBlockSize()
        assert block_height == 160
        assert block_width == 12

    def test_iter_window(self):
        for idx, window, block_row in self.raster.iter_window(min_block_size=40):
            break
        assert idx == 1
        assert window == [0, 0, 40, 40]
        assert block_row["window"] == [0, 0, 40, 40]

    def test_iter(self):
        # Make sure blocks are not initialized. Will otherwise be a bit broken
        self.raster.min_block_size = 1024
        if hasattr(self.raster, "blocks"):
            delattr(self.raster, "blocks")

        for window, block in self.raster:
            assert window == [0, 0, 160, 160]
            assert block.shape == (160, 160)

    def test__get_array(self):
        arr = self.raster.get_array()
        assert arr.shape == (160, 160)

        arr = self.raster.get_array(window=[0, 0, 80, 80])
        assert arr.sum() == np.float32(-46815150.0)

    def test_create(self):
        out_raster = RasterOld(TEMP_DIR / f"test_create_{hrt.get_uuid()}.tif")

        # remove raster
        out_raster.unlink()
        assert not out_raster.exists()

        # Create raster
        out_raster.create(metadata=self.raster.metadata, nodata=self.raster.nodata)
        assert out_raster.exists()


class TestRasterMetadata:
    def test_init_fail(self):
        # Test without input
        with pytest.raises(Exception):
            hrt.RasterMetadata()

    def test_update_resolution(self):
        meta = hrt.RasterMetadata(res=4, bounds_dict={"minx": 0, "maxx": 100, "miny": 0, "maxy": 100})
        assert meta.pixelarea == 16

        with pytest.raises(Exception):
            meta.update_resolution(resolution_new=8)

        meta.update_resolution(resolution_new=2)
        assert meta.x_res == 50


# %%
if __name__ == "__main__":
    import inspect

    selftest = TestRaster()
    self = selftest.raster
    self = selftest
    # Run all testfunctions
    for i in dir(selftest):
        if i.startswith("test_") and hasattr(inspect.getattr_static(selftest, i), "__call__"):
            print(i)
            getattr(selftest, i)()

    # %%
    selftest = TestRasterMetadata()
    self = selftest
    # Run all testfunctions
    for i in dir(selftest):
        if i.startswith("test_") and hasattr(inspect.getattr_static(selftest, i), "__call__"):
            print(i)
            getattr(selftest, i)()
# %%
