# %%
import os
import re
import sqlite3
from typing import Union

import geopandas as gpd
import pandas as pd
from osgeo import ogr
from shapely import Polygon, wkt

from hhnk_research_tools.dataframe_functions import df_convert_to_gdf
from hhnk_research_tools.variables import DEF_SRC_CRS, MOD_SPATIALITE_PATH

# %%


# TODO was: create_update_case_statement
def sql_create_update_case_statement(
    df,
    layer,
    df_id_col,
    db_id_col,
    new_val_col,
    excluded_ids=[],
    old_val_col=None,
    old_col_name=None,
    show_prev=False,
    show_proposed=False,
) -> str:
    """
    Create an sql statement with the following structure:
    UPDATE (table_name)
    SET (database_column_to_change) = CASE (database_id_col)
    WHEN (id) THEN (new value associated with id) OPTIONAL -- ['Previous' or 'Proposed'] previous or proposed value

    Initialization:

    """
    if show_proposed and show_prev:
        raise Exception("sql_create_update_case_statement: " "Only one of show_prev and show_proposed can be True")
    try:
        query = None
        if not show_prev and not show_proposed:
            vals_list = [(idx, val) for idx, val in zip(df[df_id_col], df[new_val_col]) if idx not in excluded_ids]
            statement_list = [f"WHEN {idx} THEN {val if not val is None else 'null'}" for idx, val in vals_list]
        else:
            comment = "Previous:" if show_prev else "Proposed"
            vals_list = [
                (old_val, new_val, cur_id)
                for old_val, new_val, cur_id in zip(df[old_val_col], df[new_val_col], df[df_id_col])
                if cur_id not in excluded_ids
            ]
            statement_list = [
                f"WHEN {cur_id} THEN {new_val if not new_val is None else 'null'} -- {comment} {old_val}"
                for old_val, new_val, cur_id in vals_list
            ]
        if statement_list:
            statement_string = "\n".join(statement_list)
            query = f"""
            UPDATE {layer}
            SET {old_col_name if old_col_name is not None else old_val_col} = CASE {db_id_col}
            {statement_string}
            ELSE {old_val_col}
            END
            """
        return query
    except Exception as e:
        raise e from None


def sql_construct_select_query(table_name, columns=None) -> str:
    """
    Construct sql queries that select either all
    or specified columns from a table.

    Columns has to be a list. If a list item is a tuple, it will be interpreted as:
    (column, alias). In other cases, we assume the item to be a valid column name.
    """
    base_query = "SELECT {columns} \nFROM {table_name}"
    try:
        if columns is not None:
            selection_lst = []
            if type(columns) == dict:
                for key, value in columns.items():
                    if value is not None:
                        selection_lst.append(f"{key} AS {value}")
                    else:
                        selection_lst.append(key)
            elif type(columns) == list:
                selection_lst = columns
            query = base_query.format(columns=",\n".join(selection_lst), table_name=table_name)
        else:
            query = base_query.format(columns="*", table_name=table_name)
        return query
    except Exception as e:
        raise e from None


# TODO REMOVE
def create_sqlite_connection(database_path):
    r"""Create connection to database. On windows with conda envs this requires the mod_spatialaite extension
    to be installed explicitly. The location of this extension is stored in
    hhnk_research_tools.variables.MOD_SPATIALITE_PATH (C:\ProgramData\Anaconda3\mod_spatialite-5.0.1-win-amd64)
    and can be downloaded from http://www.gaia-gis.it/gaia-sins/windows-bin-amd64/
    """
    try:
        conn = sqlite3.connect(database_path)
        conn.enable_load_extension(True)
        conn.execute("SELECT load_extension('mod_spatialite')")
        return conn
    except sqlite3.OperationalError as e:
        print("Error loading mod_spatialite")
        if e.args[0] == "The specified module could not be found.\r\n":
            if os.path.exists(MOD_SPATIALITE_PATH):
                os.environ["PATH"] = MOD_SPATIALITE_PATH + ";" + os.environ["PATH"]

                conn = sqlite3.connect(database_path)
                conn.enable_load_extension(True)
                conn.execute("SELECT load_extension('mod_spatialite')")
                return conn
            else:
                print(
                    r"""Download mod_spatialite extension from http://www.gaia-gis.it/gaia-sins/windows-bin-amd64/ 
                and place into anaconda installation C:\ProgramData\Anaconda3\mod_spatialite-5.0.1-win-amd64."""
                )
                raise e from None

    except Exception as e:
        raise e from None


# TODO REMOVE
def sql_table_exists(database_path, table_name):
    """Check if a table name exists in the specified database"""
    try:
        query = f"""PRAGMA table_info({table_name})"""
        df = execute_sql_selection(query=query, database_path=database_path)
        return not df.empty
    except Exception as e:
        raise e from None


# TODO REMOVE
def execute_sql_selection(query, conn=None, database_path=None, **kwargs) -> pd.DataFrame:
    """
    Execute sql query. Creates own connection if database path is given.
    Returns pandas dataframe
    """
    kill_connection = conn is None
    try:
        if database_path is None and conn is None:
            raise Exception("No connection or database path provided")
        if database_path is not None:
            conn = create_sqlite_connection(database_path=database_path)
        db = pd.read_sql(query, conn, **kwargs)
        return db
    except Exception as e:
        raise e from None
    finally:
        if kill_connection and conn is not None:
            conn.close()


# TODO REMOVE
def execute_sql_changes(query, database=None, conn=None):
    """
    Take a query that changes the database and tries
    to execute it. On success, changes are committed.
    On a failure, rolls back to the state before
    the query was executed that caused the error

    The explicit begin and commit statements are necessary
    to make sure we can roll back the transaction
    """
    conn_given = True
    try:
        if not conn:
            conn_given = False
            conn = create_sqlite_connection(database)
        try:
            conn.executescript(f"BEGIN; {query}; COMMIT")
        except Exception as e:
            conn.rollback()
            raise e from None
    except Exception as e:
        raise e from None
    finally:
        if not conn_given and conn:
            conn.close()


# TODO was: get_creation_statement_from_table
def _sql_get_creation_statement_from_table(src_table_name, dst_table_name, cursor):
    """Replace the original table name with the new name to make the creation statement"""
    try:
        creation_sql = f"""
                    SELECT sql
                    FROM sqlite_master
                    WHERE type = 'table'
                    AND name = '{src_table_name}'
                    """

        create_statement = cursor.execute(creation_sql).fetchone()[0]
        to_list = create_statement.split()
        all_but_name = [item if index != 2 else f'"{dst_table_name}"' for index, item in enumerate(to_list)]
        creation_statement = " ".join(all_but_name)
        return creation_statement
    except Exception as e:
        raise e from None


# TODO was: replace_or_add_table
def sqlite_replace_or_add_table(db, dst_table_name, src_table_name, select_statement=None):
    """
    Maintain the backup tables.
    Tables are created if they do not exist yet.
    After that, rows are replaced if their id is already
    in the backup, otherwise they are just inserted.

    columns HAS to be a list of tuples containing the name
    of the column and it's type
    """
    try:
        query_list = []
        conn = create_sqlite_connection(database_path=db)
        curr = conn.cursor()
        # Check if table exists
        exists = curr.execute(
            f"SELECT count() from sqlite_master " f"WHERE type='table' and name='{dst_table_name}'"
        ).fetchone()[0]
        if exists == 0:
            # Get the original creation statement from the table we are backing up if the new table doesn't exist
            if select_statement is None:
                select_statement = f"SELECT * from {src_table_name}"
            creation_statement = _sql_get_creation_statement_from_table(
                src_table_name=src_table_name,
                dst_table_name=dst_table_name,
                cursor=curr,
            )
            query_list.append(creation_statement)
            # Create the statement that copies the columns specified in select statement or copies the entire table
            copy_statement = f"INSERT INTO {dst_table_name} {select_statement}"
            query_list.append(copy_statement)
            query = ";\n".join(query_list)
        else:
            # If the backup table exists, we replace any rows that are changed since last backup
            query = f"REPLACE INTO {dst_table_name} " f"SELECT * from {src_table_name}"
        execute_sql_changes(query=query, conn=conn)
    except Exception as e:
        raise e from None
    finally:
        if conn:
            conn.close()


# TODO REMOVE
# TODO was: get_table_as_df
def sqlite_table_to_df(database_path, table_name, columns=None) -> pd.DataFrame:
    conn = None
    try:
        conn = create_sqlite_connection(database_path=database_path)
        query = sql_construct_select_query(table_name, columns)
        df = execute_sql_selection(query=query, conn=conn)
        return df
    except Exception as e:
        raise e from None
    finally:
        if conn:
            conn.close()


# TODO REMOVE
# TODO was: gdf_from_sql
def sqlite_table_to_gdf(query, id_col, to_gdf=True, conn=None, database_path=None) -> gpd.GeoDataFrame:
    """
    sqlite_table_to_gdf

    Returns DataFrame or GeoDataFrame from database query.

        sqlite_table_to_gdf(
                query (string)
                id_col (identifying column)
                to_gdf -> True (if False, DataFrame is returned)
                conn -> None (sqlite3 connection object)
                database_path -> None (path to database)

                Supply either conn or database path.
                )
    """
    if (conn is None and database_path is None) or (conn is not None and database_path is not None):
        raise Exception("Provide exactly one of conn or database_path")

    kill_conn = conn is None
    try:
        if conn is None:
            conn = create_sqlite_connection(database_path=database_path)
        df = execute_sql_selection(query=query, conn=conn)
        if to_gdf:
            df = df_convert_to_gdf(df=df, src_crs=DEF_SRC_CRS)
        df.set_index(id_col, drop=False, inplace=True)
        return df
    except Exception as e:
        raise e from None
    finally:
        if kill_conn and conn is not None:
            conn.close()


def sql_builder_select_by_location(
    schema: str,
    table_name: str,
    polygon_wkt: Polygon,
    geomcolumn: str = None,
    epsg_code="28992",
    simplify=False,
):
    """Create Oracle 12 SQL with intersection polygon.

    Parameters
    ----------
    schema : str
        database schema options are now ['DAMO_W', 'BGT']
    table_name : str
        table name in schema
    polygon_wkt : str
        Selection polygon. All data that intersects with this polygon will be selected
        Must be 1 geometry, so pass the geometry of a row, or gdf.dissolve() it first.
    simplify : bool
        Buffer by 2m
        Simplify the geometry with 1m tolerance
        Turn coordinates in ints to reduce sql size.
    """

    if geomcolumn is None:
        if schema == "DAMO_W":
            geomcolumn = "SHAPE"
        elif schema == "BGT":
            geomcolumn = "GEOMETRIE"
        else:
            raise ValueError("Provide geometry column")

    # Round coordinates to integers
    if simplify:
        polygon_wkt = polygon_wkt.buffer(2).simplify(tolerance=1)
        polygon_wkt = re.sub(r"\d*\.\d+", lambda m: format(float(m.group(0)), ".0f"), str(polygon_wkt))
    sql = f"""
        SELECT *
        FROM {schema}.{table_name}
        WHERE SDO_RELATE(
            {geomcolumn},
            SDO_GEOMETRY('{polygon_wkt}',{epsg_code}),
            'mask=ANYINTERACT'
        ) = 'TRUE'
        """

    return sql


def _oracle_curve_polygon_to_linear(blob_curvepolygon):
    """
    Turn curved polygon from oracle database into linear one
    (so it can be used in geopandas)
    Does no harm to linear geometries
    """

    # Import as an OGR curved geometry
    g1 = ogr.CreateGeometryFromWkt(str(blob_curvepolygon))
    if g1 is None:
        return None

    # Approximate as linear geometry, and export to GeoJSON
    g1l = g1.GetLinearGeometry()
    g2 = wkt.loads(g1l.ExportToWkt())

    return g2


def database_to_gdf(
    db_dict: dict, sql: str, columns: list[str] = None, lower_cols=True, crs="EPSG:28992"
) -> Union[gpd.GeoDataFrame, str]:
    """
    Connect to (oracle) database, create a cursor and execute sql

    Parameters
    ----------
    db_dict: dict
        connection dict. e.g.:
        {'service_name': 'ODSPRD',
        'user': '',
        'password': '',
        'host': 'srvxx.corp.hhnk.nl',
        'port': '1521'}
    sql: str
        oracledb 12 sql to execute
        Takes only one sql statement at a time, ';' is removed
    columns: list #TODO allow for dict input
        When not provided, get the column names from the external table
        geometry columns 'SHAPE' or 'GEOMETRIE' are renamed to 'geometry'
    lower_cols : bool
        return all output columns with no uppercase
    crs: str
        EPSG code, defaults to 28992.

    Returns
    -------
    gdf : Geodataframe with data and (linear) geometry, colum names in lowercase.
    sql : str with the used sql in the request.

    """
    # %%
    import oracledb  # Import here to prevent dependency

    if "sdo_util.to_wktgeometry" in sql.lower():
        raise ValueError("Dont pass sdo_util.to_wkt_geometry in the sql. It will be done here.")

    with oracledb.connect(**db_dict) as con:
        cur = oracledb.Cursor(con)

        # Modify sql to efficiently fetch description only
        sql = sql.replace(";", "")
        sql = sql.replace("select ", "SELECT ")  # Voor de mensen die geen caps gebruiken
        sql = sql.replace("where ", "WHERE ")  # Voor de mensen die geen caps gebruiken
        sql = sql.replace("from ", "FROM ")  # Voor de mensen die geen caps gebruiken
        pattern = r"FETCH FIRST \d+ ROWS ONLY"
        replacement = "FETCH FIRST 0 ROWS ONLY"
        matched_upper = re.search(pattern, sql)
        matched_lower = re.search(pattern.lower(), sql)
        if matched_upper:
            sql_desc = re.sub(pattern, replacement, sql)
        elif matched_lower:
            sql_desc = re.sub(pattern.lower(), replacement, sql)
        else:
            sql_desc = f"{sql} {replacement}"

        # Retrieve column names
        select_search_str = "SELECT *"
        if columns is None:
            cur.execute(sql_desc)  # TODO hier kan nog een WHERE staan met spatial select
            columns_out = [i[0] for i in cur.description]

            if "SELECT *" in sql:
                cols_dict = {c: c for c in columns_out}
            else:
                # When columns are passed, use those for the sql
                select_search_str = sql.split("FROM")[0]

                cols_sql = select_search_str.split("SELECT")[1].replace("\n", "").split(",")
                cols_sql = [c.lstrip().rstrip() for c in cols_sql]
                cols_dict = dict(zip(columns_out, cols_sql))

        elif isinstance(columns, list):
            cols_dict = {c: c for c in columns}
            columns_out = cols_dict.keys()
        elif isinstance(columns, dict):
            # van name : name_out naar
            # name_out : name as name_out
            cols_dict = {v: f"{k} as {v}" for k, v in columns.items()}
            for geomcol in ["shape", "geometrie", "geometry"]:
                if geomcol in cols_dict.keys():
                    cols_dict[geomcol] = geomcol
                if geomcol.upper() in cols_dict.keys():
                    cols_dict[geomcol.upper()] = geomcol
            columns_out = columns.values()

        # Modify geometry column name to get WKT geometry
        for key, col in cols_dict.items():
            for geomcol in ["shape", "geometrie", "geometry"]:
                if col.lower() == geomcol:
                    cols_dict[key] = f"sdo_util.to_wktgeometry({col}) as geometry"
                # Find pattern e.g.: a.shape
                if re.search(pattern=rf"(^|\w+\.){geomcol.lower()}$", string=col.lower()):
                    cols_dict[key] = f"sdo_util.to_wktgeometry({col}) as geometry"

        col_select = ", ".join(cols_dict.values())
        sql2 = sql.replace(select_search_str, f"SELECT {col_select} ")

        # Execute modified sql request
        try:
            cur.execute(sql2)
        except Exception as e:
            print("fail")
            print(sql2)
            raise e

        # load cursor to dataframe
        df = pd.DataFrame(cur.fetchall(), columns=columns_out)

        # Take column names from cursor and replace exotic geometry column names
        for i in df.columns:
            name = i
            if lower_cols:
                name = i.lower()
            if i.lower() in ("shape", "geometrie"):
                name = "geometry"

            df.rename(columns={i: name}, inplace=True)

        # make geodataframe and convert curve geometry to linear
        if "geometry" in df.columns:
            df = df.set_geometry(gpd.GeoSeries(df["geometry"].apply(_oracle_curve_polygon_to_linear)), crs=crs)
        return df, sql2


# %%
if __name__ == "__main__":
    sql = """SELECT a.OBJECTID, a.CODE, a.NAAM, a.OBSCODE, a.SPOC_CODE, a.SOORTRIOOLGEMAALCODE, 
                a.LOOST_OP_RWZI_CODE, a.LOOST_OP_RIOOLGEMAAL_CODE, a.SOORTRIOOLGEMAAL, a.REGIEKAMER, 
                b.NAAM as ZUIVERINGSKRING, a.SHAPE
            FROM CS_OBJECTEN.RIOOLGEMAAL_EVW a
            left outer join CS_OBJECTEN.ZUIVERINGSKRING_EVW b
            on a.LOOST_OP_RWZI_CODE = b.RWZICODE
            FETCH FIRST 10 ROWS ONLY
            """
    from tests_hrt.local_settings import DATABASES

    db_dicts = {
        "aquaprd": DATABASES.get("aquaprd_lezen", None),
        "bgt": DATABASES.get("bgt_lezen", None),
        "csoprd": DATABASES.get("csoprd", None),
    }
    db_dict = db_dicts["csoprd"]
    columns = None
