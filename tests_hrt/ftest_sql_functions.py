# %%

import geopandas as gpd
from local_settings import DATABASES

from hhnk_research_tools.sql_functions import (
    database_to_gdf,
    sql_builder_select_by_location,
)
from tests_hrt.config import TEST_DIRECTORY


def test_database_to_gdf():
    """test_database_to_gdf"""
    # %%
    gpkg_path = TEST_DIRECTORY / r"area_test_sql_helsdeur.gpkg"
    db_dicts = {
        "aquaprd": DATABASES.get("aquaprd_lezen", None),
        "bgt": DATABASES.get("bgt_lezen", None),
    }

    db_dict = db_dicts["bgt"]
    columns = None

    schema = "DAMO_W"
    table_name = "GEMAAL"

    # Load area for selection
    test_gdf = gpd.read_file(gpkg_path, engine="pyogrio")
    polygon_wkt = test_gdf.iloc[0]["geometry"]
    epsg_code = "28992"

    # Build sql to select by input polygon
    sql = sql_builder_select_by_location(
        schema=schema, table_name=table_name, epsg_code=epsg_code, polygon_wkt=polygon_wkt, simplify=True
    )
    db_dict = db_dicts["aquaprd"]
    columns = None

    gdf = database_to_gdf(db_dict=db_dict, sql=sql, columns=columns)
    assert gdf.loc[0, "code"] == "KGM-Q-29234"


# %%
