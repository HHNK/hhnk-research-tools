# %%
import geopandas as gpd
import pandas as pd

from hhnk_research_tools.folder_file_classes.spatial_database_class import SpatialDatabase
from tests_hrt.config import TEST_DIRECTORY


def test_available_layers():
    model_fp = TEST_DIRECTORY / "mini_test_model.gpkg"
    spdb = SpatialDatabase(model_fp)


def test_spatial_database_class():
    # load database
    model_fp = TEST_DIRECTORY / "mini_test_model.gpkg"
    spdb = SpatialDatabase(model_fp)
    # test available layers
    layers = spdb.available_layers()
    assert len(layers) == 3
    assert "channel" in layers
    # test loading layer with geometry
    gdf = spdb.load(
        layer="channel",
        # id_col="id"
    )  # FIXME id_col not loaded, but we need it, how to fix?
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == 28992
    # assert "id" in gdf.index.names
    assert gdf["code"][0] == "29_1"
    # test loading layer without geometry
    df = spdb.load(layer="model_settings", columns=["dem_file", "friction_coefficient"])  # non-spatial table
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert df["dem_file"][0] == "dem_hoekje.tif"
    # test modifying using query
    query = "UPDATE model_settings SET friction_coefficient = 666"
    spdb.modify_gpkg_using_query(query=query)
    df_mod = spdb.load(layer="model_settings", columns=["dem_file", "friction_coefficient"])  # non-spatial table
    assert df_mod["friction_coefficient"][0] == 666
    # test predifine layers
    new_layers = ["aap", "noot", "mies"]
    spdb.add_layers(new_layers)

    layers = spdb.available_layers()  # TODO doesn't list new layers, how to fix?


# %%
