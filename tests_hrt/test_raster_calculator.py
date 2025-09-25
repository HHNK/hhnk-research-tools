# %%
import json

import geopandas as gpd
import numpy as np
import pytest

import hhnk_research_tools as hrt
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY


def ztest_raster_blocks():
    """Test raster block loading"""
    # FIXME uitgezet, kan wss weg
    raster = hrt.Raster(TEST_DIRECTORY / r"depth_test.tif")

    gdf = hrt.RasterChunks.from_raster(raster).to_gdf()
    for idx, block_row in gdf.iterrows():
        break

    block = hrt.RasterBlocks(
        window=block_row["window_readarray"],
        raster_paths_dict={
            "raster1": raster,
            "raster2": raster,
        },
        nodata_keys=["raster1"],
        mask_keys=["raster2"],
    )

    assert block.cont is True
    assert block.masks_all.shape == (160, 160)

    block.blocks["raster1"][block.masks_all] = 0
    assert int(block.blocks["raster1"].sum()) == 1826


def ztest_raster_calculator():
    # FIXME uitgezet, kan wss weg
    """Test raster calculator"""
    raster_depth = hrt.Raster(TEST_DIRECTORY / r"depth_test.tif")
    raster_small = hrt.Raster(TEST_DIRECTORY / r"lu_small.tif")
    raster_out = hrt.Raster(TEMP_DIR / f"rastercalc_{hrt.get_uuid()}.tif")

    def run_window(block):
        block_out = block.blocks["small_raster"]

        # Nodatamasks toepassen
        block_out[block.masks_all] = 0
        return block_out

    # Check error in nodata_keys and yesdata_dict
    with pytest.raises(UnboundLocalError):
        calc = hrt.RasterCalculatorV2(
            raster_out=raster_out,
            raster_paths_dict={
                "depth": raster_depth,
                "small_raster": raster_small,
            },
            nodata_keys=["depth"],
            mask_keys=["depth", "small_raster"],
            metadata_key="depth",
            custom_run_window_function=run_window,
            yesdata_dict={"depth": [-99]},
            output_nodata=0,
            min_block_size=4096,
            verbose=True,
            tempdir=hrt.Folder(TEMP_DIR / "temprasters"),
        )

        calc.run(overwrite=False)

    # Working calculator
    calc = hrt.RasterCalculatorV2(
        raster_out=raster_out,
        raster_paths_dict={
            "depth": raster_depth,
            "small_raster": raster_small,
        },
        nodata_keys=["depth"],
        mask_keys=["depth", "small_raster"],
        metadata_key="depth",
        custom_run_window_function=run_window,
        yesdata_dict={"small_raster": [2, 28]},
        output_nodata=0,
        min_block_size=4096,
        verbose=True,
        tempdir=hrt.Folder(TEMP_DIR / "temprasters"),
    )

    calc.run(overwrite=False)

    assert raster_out.sum() == 19834


def test_raster_label_stats():
    """Test calculation of statistics per label"""

    def run_stats_window(block, output_nodata):
        """custom_run_window_function on blocks in hrt.RasterCalculator"""
        block_out = block.blocks["lu"].astype(np.int16)

        # Apply nodatamasks
        block_out[block.masks_all] = output_nodata
        return block_out

    label_shape = hrt.SpatialDatabase(TEST_DIRECTORY / r"area_test_labels.gpkg")
    label_raster = hrt.Raster(TEST_DIRECTORY / r"area_test_labels.tif")
    lu_raster = hrt.Raster(TEST_DIRECTORY / r"landuse_test.tif")
    stats_json = hrt.File(TEMP_DIR / f"rasterstats_{hrt.get_uuid()}.json")

    label_gdf = gpd.read_file(label_shape.path)

    if not label_raster.exists():
        hrt.gdf_to_raster(
            gdf=label_gdf,
            value_field="id",
            raster_out=label_raster,
            nodata=-9999,
            metadata=hrt.RasterMetadataV2.from_gdf(gdf=label_gdf, res=lu_raster.metadata.pixel_width),
        )

    calc = hrt.RasterCalculatorV2(
        raster_out=None,
        raster_paths_dict={
            "lu": lu_raster,
            "label": label_raster,
        },
        nodata_keys=None,
        mask_keys=["label"],
        metadata_key="label",
        custom_run_window_function=run_stats_window,
        output_nodata=-9999,
        min_block_size=4096,
        verbose=True,
        tempdir=hrt.Folder(TEMP_DIR / "temprasters"),
    )

    calc.run_label_stats(
        label_gdf=label_gdf,
        label_col="id",
        stats_json=stats_json,
        decimals=0,
        output_nodata=calc.output_nodata,
    )

    stats_dict = json.loads(stats_json.path.read_text())
    assert stats_dict["0"] == {"2": 61, "6": 2358, "15": 267, "28": 1005, "29": 2262}


# %%
if __name__ == "__main__":
    test_raster_blocks()
    test_raster_calculator()
    test_raster_label_stats()
