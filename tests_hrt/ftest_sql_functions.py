# %%

import geopandas as gpd
import oracledb
import pytest
from IPython.display import display
from local_settings import DATABASES

import hhnk_research_tools.logging as logging
from hhnk_research_tools.sql_functions import (
    _remove_blob_columns,
    database_to_gdf,
    get_table_columns,
    get_table_domains_from_oracle,
    sql_builder_select_by_location,
)
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY

logger = logging.get_logger(name=__name__)
# %%


def test_database_to_gdf_damo():
    """test_database_to_gdf for DAMO table"""
    # %%
    gpkg_path = TEST_DIRECTORY / r"area_test_sql_helsdeur.gpkg"

    db_dict = DATABASES.get("aquaprd_lezen", None)
    columns = None

    schema = "DAMO_W"
    table_name = "GEMAAL"
    # table_name = "PEILGEBIEDPRAKTIJK"

    # Load area for selection
    test_gdf = gpd.read_file(gpkg_path, engine="pyogrio")
    polygon_wkt = test_gdf.iloc[0]["geometry"]
    epsg_code = "28992"

    # Build sql to select by input polygon
    sql = sql_builder_select_by_location(
        schema=schema, table_name=table_name, epsg_code=epsg_code, polygon_wkt=polygon_wkt, simplify=True
    )

    gdf, sql2 = database_to_gdf(db_dict=db_dict, sql=sql, columns=columns, lower_cols=True, remove_blob_cols=True)
    assert gdf.loc[0, "code"] == "KGM-Q-29234"


def test_bgt_database_to_gdf_bgt():
    """test_database_to_gdf"""

    gpkg_path = TEST_DIRECTORY / r"area_test_sql_helsdeur.gpkg"

    db_dict = DATABASES.get("bgt_lezen", None)
    columns = None

    schema = "BGT"
    table_name = "HHNK_MV_WTD"
    # table_name = "PEILGEBIEDPRAKTIJK"

    # Load area for selection
    test_gdf = gpd.read_file(gpkg_path, engine="pyogrio")
    polygon_wkt = test_gdf.iloc[0]["geometry"]
    epsg_code = "28992"

    # Build sql to select by input polygon
    sql = sql_builder_select_by_location(
        schema=schema, table_name=table_name, epsg_code=epsg_code, polygon_wkt=polygon_wkt, simplify=True
    )

    gdf, sql2 = database_to_gdf(db_dict=db_dict, sql=sql, columns=columns, lower_cols=True, remove_blob_cols=True)
    assert gdf.loc[0, "id"] == 9728209


# %%


def test_database_to_gdf_no_cols():
    sql = """SELECT a.OBJECTID, a.CODE, a.NAAM,a.REGIEKAMER, 
                b.NAAM as ZUIVERINGSKRING, a.SHAPE
            FROM CS_OBJECTEN.RIOOLGEMAAL_EVW a
            left outer join CS_OBJECTEN.ZUIVERINGSKRING_EVW b
            on a.LOOST_OP_RWZI_CODE = b.RWZICODE
            FETCH FIRST 10 ROWS ONLY
            """
    db_dict = DATABASES.get("csoprd_lezen", None)
    columns = None
    gdf, sql2 = database_to_gdf(db_dict=db_dict, sql=sql, columns=columns)

    # This should fail
    sql = """SELECT a.OBJECTID, sdo_util.to_wktgeometry(a.SHAPE)
            FROM CS_OBJECTEN.RIOOLGEMAAL_EVW a
            """
    with pytest.raises(ValueError):
        gdf, sql2 = database_to_gdf(db_dict=db_dict, sql=sql, columns=columns)

    # %%


def test_remove_blob_cols():
    db_dict = DATABASES.get("aquaprd_lezen", None)
    columns = None
    # Find code by removing where statement and; gdf[gdf['se_anno_cad_data'].notna()]['code']
    sql = "SELECT * FROM DAMO_W.PEILGEBIEDPRAKTIJK WHERE CODE = '04851-23'"
    gdf, sql2 = database_to_gdf(db_dict=db_dict, sql=sql, columns=columns, lower_cols=True, remove_blob_cols=False)

    with pytest.raises(oracledb.InterfaceError):
        print(gdf)

    gdf2 = _remove_blob_columns(gdf)

    display(gdf)
    # %%


def test_get_table_columns():
    """
    Test with database connection.
    Difficult to test without this connection as sqlite and
    postges syntax for 'FETCH FIRST 0 ROWS ONLY' is different
    from Oracle.
    """
    # %%
    db_dict = DATABASES.get("aquaprd_lezen", None)

    columns = get_table_columns(
        db_dict=db_dict,
        schema="DAMO_W",
        table_name="HYDROOBJECT",
    )
    assert "CODE" in columns
    assert "NAAM" in columns
    assert "SHAPE" in columns

    # %%


def test_get_table_domains_from_oracle():
    """Test to get domaintable from DAMO_W."""
    db_dict = DATABASES.get("aquaprd_lezen", None)
    schema = "DAMO_W"
    table_name = "GEMAAL"

    domains = get_table_domains_from_oracle(
        db_dict=db_dict,
        schema=schema,
        table_name=table_name,
    )
    assert domains[domains["naamdomeinwaarde"] == "afvoeren"]["codedomeinwaarde"].values[0] == "2"

    # %%


# NOTE remove
