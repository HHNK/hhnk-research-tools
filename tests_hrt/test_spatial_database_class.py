# %%
import os

import geopandas as gpd
import pandas as pd

from hhnk_research_tools.folder_file_classes.spatial_database_class import SpatialDatabase
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY


def test_spatial_database_class():
    # load database
    model_fp = TEST_DIRECTORY / "mini_test_model.gpkg"
    spdb = SpatialDatabase(model_fp)

    # test available layers
    layers = spdb.available_layers()
    assert len(layers) == 3
    assert "channel" in layers

    # test loading layer with geometry
    gdf = spdb.load(layer="channel", index_column="id")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == 28992
    # assert "id" in gdf.index.names
    assert gdf["code"][514] == "29_1"

    # test loading layer without geometry
    df = spdb.load(layer="model_settings", columns=["dem_file", "friction_coefficient"])  # non-spatial table
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert df["dem_file"][0] == "dem_hoekje.tif"
    assert df["friction_coefficient"][0] == 666

    # test modifying using query
    # copy to temp dir
    temp_fp = TEMP_DIR / "temp_test_model.gpkg"
    os.system(f"cp {model_fp} {temp_fp}")
    spdb = SpatialDatabase(temp_fp)
    # modify friction coefficient
    query = "UPDATE model_settings SET friction_coefficient = 999"
    spdb.modify_gpkg_using_query(query=query)
    df_mod = spdb.load(layer="model_settings", columns=["dem_file", "friction_coefficient"])  # non-spatial table
    assert df_mod["friction_coefficient"][0] == 999


# %%
