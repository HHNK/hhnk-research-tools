# %%
import importlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

import hhnk_research_tools as hrt
import hhnk_research_tools.rasters.raster_class as raster_class

importlib.reload(raster_class)
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY

logger = hrt.logging.get_logger("hhnk_research_tools", level="DEBUG")


class TestRaster:
    raster = raster_class.Raster(TEST_DIRECTORY / r"depth_test.tif")
    gpkg_path = TEST_DIRECTORY / r"area_test.gpkg"

    def test_properties(self):
        assert self.raster.nodata == -9999
        assert self.raster.profile["count"] == 1
        assert self.raster.pixelarea == 0.25
        assert self.raster.statistics() == {
            "min": -0.00988,
            "max": 0.484222,
            "mean": 0.133442,
            "std": 0.09345,
        }

        gdal_src = self.raster.open_gdal()
        band = gdal_src.GetRasterBand(1)

        block_height, block_width = band.GetBlockSize()
        assert block_height == 160
        assert block_width == 12

    def test_rasterchunks(self):
        chunks = raster_class.RasterChunks.from_raster(raster=self.raster)
        chunks.chunksize = 40
        gdf = chunks.to_gdf()
        assert gdf.loc[1, "window"] == [0, 0, 40, 40]

    def test_create(self):
        out_raster = raster_class.Raster(TEMP_DIR / f"test_create_{hrt.get_uuid()}.tif")

        # remove raster
        out_raster.unlink()
        assert not out_raster.exists()

        # Create raster
        with out_raster.open_rio(mode="w", profile=self.raster.profile, dtype="float32") as dst:
            pass
        assert out_raster.exists()
        assert out_raster.profile == self.raster.profile

    def test_sum_labels(self):
        label_raster = hrt.Raster(TEST_DIRECTORY / r"area_test_labels.tif", chunksize=40)

        label_gdf = gpd.read_file(TEST_DIRECTORY / r"area_test_labels.gpkg")
        label_idx = label_gdf["id"].to_numpy()

        # Sum labels
        label_sum = self.raster.sum_labels(label_raster=label_raster, label_idx=label_idx, decimals=2)
        assert round(sum(label_sum.values()), 2) == 1826.15

        # Count occurences per label.
        label_sum_count = self.raster.sum_labels(label_raster=label_raster, label_idx=label_idx, count_values=True)
        assert sum(label_sum_count.values()) == 13685


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
