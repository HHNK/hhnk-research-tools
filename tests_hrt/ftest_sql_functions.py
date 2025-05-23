# %%

import geopandas as gpd
import oracledb
import pytest
from IPython.display import display
from local_settings import DATABASES

import hhnk_research_tools.logger as logging
from hhnk_research_tools.sql_functions import (
    _remove_blob_columns,
    database_to_gdf,
    execute_sql_selection,
    get_database_columns,
    sql_builder_select_by_location,
)
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY

logger = logging.get_logger(name=__name__)


def test_database_to_gdf():
    """test_database_to_gdf"""
    # %%
    gpkg_path = TEST_DIRECTORY / r"area_test_sql_helsdeur.gpkg"

    db_dict = DATABASES.get("bgt_lezen", None)
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

    db_dict = DATABASES.get("aquaprd_lezen", None)
    columns = None

    gdf, sql2 = database_to_gdf(db_dict=db_dict, sql=sql, columns=columns, lower_cols=True, remove_blob_cols=True)
    assert gdf.loc[0, "code"] == "KGM-Q-29234"

    # #Get all available tables in connection.
    # import oracledb
    # with oracledb.connect(**db_dict) as con:
    #     cur = oracledb.Cursor(con)
    #     a = execute_sql_selection("SELECT owner, table_name FROM all_tables", conn=con)


# %%


def test_database_to_gdf_no_cols():
    # %%
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
    # %%
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


def test_get_database_columns():
    """
    Test with database connection.
    Difficult to test without this connection as sqlite and
    postges syntax for 'FETCH FIRST 0 ROWS ONLY' is different
    from Oracle.
    """
    # %%
    db_dict = DATABASES.get("aquaprd_lezen", None)

    with pytest.raises(TypeError):
        list = get_database_columns(
            db_dict=db_dict,
            schema="DAMO_W",
            table_name="HYDROOBJECT",
        )
    assert "CODE" in list
    assert "NAAM" in list
    assert "SHAPE" in list

    # %%
