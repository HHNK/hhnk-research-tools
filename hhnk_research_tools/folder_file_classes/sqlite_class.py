# %% #TODO REMOVE
import os
import sqlite3
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

import hhnk_research_tools as hrt
from hhnk_research_tools.folder_file_classes.file_class import File
from hhnk_research_tools.variables import MOD_SPATIALITE_PATH

logger = hrt.logging.get_logger(name=__name__)


class Sqlite(File):
    def __init__(self, base):
        super().__init__(base)
        logger.warning(
            "DeprecationWarning: Stop using Sqlite class from hhnk_research_tools. Use SpatialDatabase class instead."
        )

    def connect(self) -> Optional[sqlite3.Connection]:
        if self.exists():
            return self.create_sqlite_connection()
        else:
            return None

    def create_sqlite_connection(self):
        r"""Create connection to database. On windows with conda envs this requires the mod_spatialaite extension
        to be installed explicitly. The location of this extension is stored in
        hhnk_research_tools.variables.MOD_SPATIALITE_PATH (C:\ProgramData\Anaconda3\mod_spatialite-5.0.1-win-amd64)
        and can be downloaded from http://www.gaia-gis.it/gaia-sins/windows-bin-amd64/
        """
        try:
            conn = sqlite3.connect(self.path)
            conn.enable_load_extension(True)
            conn.execute("SELECT load_extension('mod_spatialite')")
            return conn
        except sqlite3.OperationalError as e:
            if e.args[0] == "The specified module could not be found.\r\n":
                if os.path.exists(MOD_SPATIALITE_PATH):
                    os.environ["PATH"] = MOD_SPATIALITE_PATH + ";" + os.environ["PATH"]

                    conn = sqlite3.connect(self.path)
                    conn.enable_load_extension(True)
                    conn.execute("SELECT load_extension('mod_spatialite')")
            else:
                print(
                    r"""Download mod_spatialite extension from http://www.gaia-gis.it/gaia-sins/windows-bin-amd64/ 
                and place into anaconda installation C:\ProgramData\Anaconda3\mod_spatialite-5.0.1-win-amd64."""
                )
                raise e from None

        except Exception as e:
            raise e from None

        return conn

    def read_table(
        self,
        table_name: str,
        id_col: Optional[str] = None,
        columns: Optional[list] = None,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Read table as (geo)dataframe. If there is a geometry column
        then it will load as a gdf in epsg 28992.
        Run .list_tables to get an overview over available (v2) tables

        Parameters
        ----------
        table_name : str
            Name of the SQLite table to read.
        id_col : str, optional
            Column name to set as index.
        columns : list[str], optional
            List of column names to include in the DataFrame.

        Returns
        -------
        df : Union[pd.DataFrame, gpd.GeoDataFrame]
            Loaded table as a (Geo)DataFrame.
        """
        conn = None
        try:
            conn = self.connect()

            table_meta = self.sql_table_info(table_name=table_name, conn=conn)
            if "the_geom" in table_meta["name"].values:
                query = f"SELECT *, AsWKT(the_geom) as 'geometry' \nFROM {table_name}"

                df = pd.read_sql(query, conn)
                df.drop("the_geom", axis=1, inplace=True)

                df = hrt.df_convert_to_gdf(df=df, src_crs="4326")

            else:
                query = f"SELECT * \nFROM {table_name}"
                df = pd.read_sql(query, conn)

            if columns:
                df = df[columns]
            if id_col:
                df.set_index(id_col, drop=True, inplace=True)

            return df
        except KeyError as e:
            raise Exception(e, f"available columns are: {table_meta['name'].values}")
        except Exception as e:
            raise e from None
        finally:
            if conn:
                conn.close()

    def execute_sql_selection(self, query, conn=None, **kwargs) -> pd.DataFrame:
        """
        Execute sql query. Creates own connection if database path is given.
        Returns pandas dataframe
        """
        kill_connection = conn is None  # Only kill connection when it was not provided as input
        try:
            if conn is None:
                conn = self.connect()
            db = pd.read_sql(query, conn, **kwargs)
            if "geometry" in db.keys():
                db = hrt.df_convert_to_gdf(db)
            return db
        except Exception as e:
            raise e from None
        finally:
            if kill_connection and conn is not None:
                conn.close()

    def execute_sql_changes(self, query: str, conn=None):
        """
        Take a query that changes the database and try
        to execute it. On success, changes are committed.
        On a failure, roll back to the state before
        execution.

        The explicit begin and commit statements are necessary
        to make sure we can roll back the transaction
        """

        # Dont execute empty str.
        if query in [None, ""]:
            return

        kill_connection = conn is None  # Only kill connection when it was not provided as input
        try:
            if conn is None:
                conn = self.connect()
            try:
                conn.executescript(f"BEGIN; {query}; COMMIT")
            except Exception as e:
                conn.rollback()
                raise e from None
        except Exception as e:
            raise e from None
        finally:
            if kill_connection and conn is not None:
                conn.close()

    # TODO was sql_table_exists
    def sql_table_info(self, table_name, conn=None):
        """Return table info if it exists"""
        query = f"""PRAGMA table_info({table_name})"""
        df = self.execute_sql_selection(query=query, conn=conn)
        return df

    def list_tables(self):
        """Get a list of all v2 tables."""
        query = """SELECT name
                    FROM sqlite_schema 
                    WHERE type='table' 
                    AND name LIKE 'v2_%'
                    ORDER BY name
                """
        return self.execute_sql_selection(query=query)


# %%

if __name__ == "__main__":
    self = Sqlite(r"E:\02.modellen\model_test_v2\02_schematisation\00_basis\bwn_test.sqlite")

    table_name = "v2_channel"

    df = self.read_table(table_name=table_name, columns=["id", "geometry"])
    display(df)
