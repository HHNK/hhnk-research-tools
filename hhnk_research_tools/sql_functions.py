# %%
import os
import re
import sqlite3
from typing import Optional, Tuple

import geopandas as gpd
import oracledb
import pandas as pd
from osgeo import ogr
from shapely import Polygon, wkt

import hhnk_research_tools.logging as logging
from hhnk_research_tools.dataframe_functions import df_convert_to_gdf
from hhnk_research_tools.variables import DEF_SRC_CRS, MOD_SPATIALITE_PATH

logger = logging.get_logger(name=__name__)

# %%


# TODO was: create_update_case_statement
def sql_create_update_case_statement(
    df: pd.DataFrame,
    layer: str,
    df_id_col: str,
    db_id_col: str,
    new_val_col: str,
    excluded_ids: Optional[list] = None,
    old_val_col: Optional[str] = None,
    old_col_name: Optional[str] = None,
    show_prev: bool = False,
    show_proposed: bool = False,
) -> str:
    """
    Create an sql statement with the following structure:
    UPDATE (table_name)
    SET (database_column_to_change) = CASE (database_id_col)
    WHEN (id) THEN (new value associated with id) OPTIONAL -- ['Previous' or 'Proposed'] previous or proposed value

    Initialization:

    Returns
    -------
    query : str
    """
    if excluded_ids is None:
        excluded_ids = []
    if show_proposed and show_prev:
        raise Exception("sql_create_update_case_statement: Only one of show_prev and show_proposed can be True")
    try:
        query = None
        if not show_prev and not show_proposed:
            vals_list = [(idx, val) for idx, val in zip(df[df_id_col], df[new_val_col]) if idx not in excluded_ids]
            statement_list = [f"WHEN {idx} THEN {val if val is not None else 'null'}" for idx, val in vals_list]
        else:
            comment = "Previous:" if show_prev else "Proposed"
            vals_list = [
                (old_val, new_val, cur_id)
                for old_val, new_val, cur_id in zip(df[old_val_col], df[new_val_col], df[df_id_col])
                if cur_id not in excluded_ids
            ]
            statement_list = [
                f"WHEN {cur_id} THEN {new_val if new_val is not None else 'null'} -- {comment} {old_val}"
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
            if isinstance(columns, dict):
                for key, value in columns.items():
                    if value is not None:
                        selection_lst.append(f"{key} AS {value}")
                    else:
                        selection_lst.append(key)
            elif isinstance(columns, list):
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
        logger.error("Error loading mod_spatialite")
        if e.args[0] == "The specified module could not be found.\r\n":
            if os.path.exists(MOD_SPATIALITE_PATH):
                os.environ["PATH"] = MOD_SPATIALITE_PATH + ";" + os.environ["PATH"]

                conn = sqlite3.connect(database_path)
                conn.enable_load_extension(True)
                conn.execute("SELECT load_extension('mod_spatialite')")
                return conn
            else:
                logger.error(
                    rf"""Download mod_spatialite extension from http://www.gaia-gis.it/gaia-sins/windows-bin-amd64/ 
                and place into anaconda installation {MOD_SPATIALITE_PATH}."""
                )
                raise e from None

    except Exception as e:
        raise e from None


# TODO REMOVE
def sql_table_exists(database_path, table_name: str):
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
def _sql_get_creation_statement_from_table(src_table_name: str, dst_table_name: str, cursor) -> str:
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
            f"SELECT count() from sqlite_master WHERE type='table' and name='{dst_table_name}'"
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
            query = f"REPLACE INTO {dst_table_name} SELECT * from {src_table_name}"
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
    include_todays_mutations=False,
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
    include_todays_mutations : bool
        Choose whether to use todays mutations in data, normally mutations are available
        overnight.
        Not sure if this works for BGT or OGS
    """
    # Set custom geometry columns
    if geomcolumn is None:
        if schema == "DAMO_W":
            geomcolumn = "SHAPE"
        elif schema == "BGT":
            geomcolumn = "GEOMETRIE"
        elif schema == "CS_OBJECTEN":
            geomcolumn = "SHAPE"
        else:
            raise ValueError("Provide geometry column")

    # modify table_name to include today's mutations
    if include_todays_mutations and "_EVW" not in table_name:
        table_name = f"{table_name}_EVW"

    # TODO use convex hull and clip to avoid too long sql
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


def sql_builder_select_by_id_list_statement(
    sub_id_list_sql: str,
    schema: str,
    sub_table: str,
    sub_id_column: str,
    include_todays_mutations=False,
):
    """Create Oracle 12 SQL using extra statement that list id's from another table.
    Created for 'Profielen' and other table from DAMO_W database.

    Parameters
    ----------
    sub_id_list_sql: str
        sql to extract id list
    schema : str
        database schema options are now ['DAMO_W', 'BGT']
    sub_table : str
        table name in schema
    sub_id_column : str
        Name of id column in target table
    include_todays_mutations : bool
        Choose whether to use todays mutations in data, normally mutations are available
        overnight.
        Not sure if this works for BGT or OGS
    """

    # modify table_name to include today's mutations
    if include_todays_mutations and "_EVW" not in sub_table:
        sub_table = f"{sub_table}_EVW"

    sql = f"""SELECT *
FROM {schema}.{sub_table}
WHERE {sub_id_column} IN (
    {sub_id_list_sql} 
)"""

    return sql


def _oracle_curve_polygon_to_linear(blob_curvepolygon):
    """
    Turn curved polygon from oracle database into linear one
    (so it can be used in geopandas)
    Does no harm to linear geometries, polygon or other
    """
    try:
        # Import as an OGR curved geometry
        g1 = ogr.CreateGeometryFromWkt(str(blob_curvepolygon))

        # Check if geometry is valid
        if g1 is None or not g1.IsValid():
            logger.warning(f"Invalid geometry found: {blob_curvepolygon}, returning None")
            return None

        # Approximate as linear geometry, and export to GeoJSON
        g1l = g1.GetLinearGeometry()
        g2 = wkt.loads(g1l.ExportToWkt())
    except Exception as e:
        logger.warning(f"Unable to convert curve polygon {str(blob_curvepolygon)} to linear: {e}, returning None")
        return None

    return g2


def _remove_blob_columns(df):
    """
    Remove columns that stay in blob from oracle database.
    Blob columns prohibit further processing of the data
    since they require an open connection to the Oralce
    database.

    Known blob column in DAMO: se_anno_cad_data (DAMO_W.PEILGEBIEDPRAKTIJK)

    """
    # Loop columns and check all rows for blob data
    blob_columns = set()
    for c in df.keys():
        # Check only when column has type object for efficiency
        if df[c].dtypes == "object":
            # Loop through unique values since some are None
            for row in df[c].unique():
                if isinstance(row, oracledb.LOB):
                    blob_columns.add(c)
                    break

    if blob_columns:
        logger.warning(f"Columns {blob_columns} contain BLOB data and are removed")

    # Remove blob data from geodataframe
    df.drop(columns=blob_columns, inplace=True)

    return df


def database_to_gdf(
    db_dict: dict,
    sql: str,
    columns: list[str] = None,
    lower_cols: bool = True,
    remove_blob_cols: bool = True,
    crs="EPSG:28992",
) -> Tuple[gpd.GeoDataFrame, str]:
    """
    Connect to (oracle) database, create a cursor and execute sql

    Parameters
    ----------
    db_dict: dict
        connection dict. e.g.:
        {'service_name': '',
        'user': '',
        'password': '',
        'host': '',
        'port': ''}
    sql: str
        oracledb 12 sql to execute
        Takes only one sql statement at a time, ';' is removed
    columns: list
        When not provided, get the column names from the external table
        geometry columns 'SHAPE' or 'GEOMETRIE' are renamed to 'geometry'
    lower_cols : bool
        return all output columns with no uppercase
    remove_blob_cols: bool
        remove columns that contain oracle blob data
    crs: str
        EPSG code, defaults to 28992.

    Returns
    -------
    gdf : gpd.GeoDataFrame
        gdf with data and (linear) geometry, colum names in lowercase.
    sql : str
        The used sql in the request.

    """

    if "sdo_util.to_wktgeometry" in sql.lower():
        raise ValueError(
            "Dont pass sdo_util.to_wkt_geometry in the sql. It will be added here. Just use e.g. SHAPE as column."
        )

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
        else:
            raise ValueError("Columns must be a list {columns}")

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
            logger.error(f"""Failed request. Here is the sql:
{sql}""")
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
            df = df.set_geometry(
                gpd.GeoSeries(df["geometry"].apply(_oracle_curve_polygon_to_linear)),
                crs=crs,
            )
            # Check if geometries in dataframe are valid
            invalid_geoms = ~df.geometry.is_valid
            if invalid_geoms.any():
                logger.warning(f"{invalid_geoms.sum()} invalid geometries found in dataframe.")
                df = df[~invalid_geoms]

        # remove blob columns from oracle
        if remove_blob_cols:
            df = _remove_blob_columns(df)

        return df, sql2


def get_table_columns(
    db_dict: dict,
    schema: str,
    table_name: str,
):
    """
    Connect to (oracle) databaseand retrieve table columns

    Parameters
    ----------
    db_dict: dict
        connection dict. e.g.:
        {'service_name': '',
        'user': '',
        'password': '',
        'host': '',
        'port': ''}
    schema: str
        schema name
    table_name: str
        table name

    Returns
    -------
    columns_out : List of column names
    """
    columns_out = None
    sql = f"""
        SELECT *
        FROM {schema}.{table_name}
        FETCH FIRST 0 ROWS ONLY
        """

    with oracledb.connect(**db_dict) as con:
        cur = oracledb.Cursor(con)
        cur.execute(sql)
        columns_out = [i[0] for i in cur.description]

    return columns_out


def get_tables_from_oracle_db(db_dict: dict):
    """
    Get list of all tables in database.

    Outputs OWNER (SCHEMA), TABLE_NAME

    Parameters
    ----------
    db_dict : dict
        connection dict. e.g.:
        {'service_name': '',
        'user': '',
        'password': '',
        'host': '',
        'port': ''}
    """
    with oracledb.connect(**db_dict) as con:
        cur = oracledb.Cursor(con)
        tables_df = execute_sql_selection("SELECT owner, table_name FROM all_tables", conn=con)

    return tables_df


def get_table_domains_from_oracle(
    db_dict: dict,
    schema: str,
    table_name: str,
    column_list: list[str] = None,
):
    """
    Get domain for DAMO_W tables.

    Parameters
    ----------
    db_dict : dict
        connection dict. e.g.:
        {'service_name': '',
        'user': '',
        'password': '',
        'host': '',
        'port': ''}
    schema : str
        schema name
    table_name : str
        Table name
    column_list : str
        List of column names for which to retrieve the domains.
    """

    if schema == "DAMO_W":
        if column_list is None:
            column_list = get_table_columns(db_dict=db_dict, schema=schema, table_name=table_name)

        # make all items list columnlist lowercase
        column_list = [c.lower() for c in column_list]

        # standard queries
        map_query = f"""
                SELECT *
                FROM DAMO_W.DAMOKOLOM
                WHERE LOWER(DAMOTABELNAAM) = '{table_name.lower()}'
                AND DAMODOMEINNAAM IS NOT NULL
                """
        domain_query = """
                SELECT *
                FROM DAMO_W.DAMODOMEINWAARDE
                """
        # Query database
        with oracledb.connect(**db_dict) as con:
            cur = oracledb.Cursor(con)
            map_df = execute_sql_selection(map_query, conn=con)
            map_df.columns = map_df.columns.str.lower()
            map_df = map_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
            # select relevant domains for columns
            map_df = map_df[map_df["damokolomnaam"].isin(column_list)]

            # List domains from DAMO
            domain_df = execute_sql_selection(domain_query, conn=con)
            domain_df.columns = domain_df.columns.str.lower()
            domain_df = domain_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        domains = pd.DataFrame()
        for i in map_df["damodomeinnaam"].unique():
            # Select relevant domains
            domain_rules = domain_df[domain_df["damodomeinnaam"] == i]
            domain_rules = domain_rules[
                [
                    "damodomeinnaam",
                    "codedomeinwaarde",
                    "naamdomeinwaarde",
                    "fieldtype",
                ]
            ]
            # select relevant mapping columns
            mapping = map_df[map_df["damodomeinnaam"] == i]
            mapping = mapping[
                [
                    "damotabelnaam",
                    "damokolomnaam",
                    "damodomeinnaam",
                    "definitie",
                ]
            ]

            # join mapping and domain
            df = mapping.merge(domain_rules, on="damodomeinnaam", how="left")

            domains = pd.concat([domains, df], ignore_index=True)

        return domains

    else:
        logger.warning("Schema not supported, only DAMO_W contains domains.")
        return None
