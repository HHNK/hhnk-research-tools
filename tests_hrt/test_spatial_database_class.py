# %%
import shutil

import geopandas as gpd
import pandas as pd

from hhnk_research_tools.folder_file_classes.spatial_database_class import SpatialDatabase
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY


def test_spatial_database_class():
    # Load database
    model_filepath = TEST_DIRECTORY / "mini_test_model.gpkg"
    spatial_db = SpatialDatabase(base=model_filepath)

    # Test available layers
    layers = spatial_db.available_layers()
    assert len(layers) == 3
    assert "channel" in layers

    # Test loading layer with geometry
    gdf = spatial_db.load(layer="channel", index_column="id")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == 28992
    assert "id" in gdf.index.names
    assert gdf["code"][514] == "29_1"

    # Test loading layer without geometry
    df = spatial_db.load(layer="model_settings", columns=["dem_file", "friction_coefficient"])  # non-spatial table
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert df["dem_file"][0] == "dem_hoekje.tif"
    assert df["friction_coefficient"][0] == 666

    # Test modifying using query
    temp_filepath = TEMP_DIR / "temp_test_model.gpkg"
    shutil.copy(src=model_filepath, dst=temp_filepath)
    spatial_db = SpatialDatabase(temp_filepath)

    # Modify friction coefficient
    query = "UPDATE model_settings SET friction_coefficient = 999"
    spatial_db.modify_gpkg_using_query(query=query)
    df_modified = spatial_db.load(
        layer="model_settings", columns=["dem_file", "friction_coefficient"]
    )  # non-spatial table
    assert df_modified["friction_coefficient"][0] == 999


# %%
