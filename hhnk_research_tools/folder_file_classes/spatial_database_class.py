# %%
import sqlite3
import types
import warnings
from typing import Optional, Union

import fiona
import geopandas as gpd
import pandas as pd

from hhnk_research_tools import logging
from hhnk_research_tools.folder_file_classes.file_class import File
from hhnk_research_tools.general_functions import get_functions, get_variables

logger = logging.get_logger(name=__name__)


class SpatialDatabase(File):
    def __init__(self, base):
        super().__init__(base)

        self.layerlist = []
        self.layers = types.SimpleNamespace()  # empty class

    def load(
        self,
        layer: Optional[str] = None,
        index_column: Optional[str] = None,
        columns: Optional[list] = None,
        epsg_code: int = 28992,
        engine: str = "pyogrio",
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Load layer as geodataframe. When no layer provided, use the default / first one.

        Parameters
        ----------
        layer : str, optional, by default None
            Name of the layer to load. If None, and only one layer is available, that one is used.
            If multiple layers are available, user input is requested to select one.
        index_column : str, optional, by default None
            Column to set as index, if the column is not loaded by geopandas, it will be try to
            load it using a SQL query.
        columns : list, optional, by default None (all columns)
            List of columns to load, if None all columns are loaded. Index column is always loaded.
        epsg_code : int, optional, by default 28992 (Amersfoort / RD New)
            EPSG code to reproject geometries to.
        engine : str, optional, by default "pyogrio"
            Engine to use for reading the file, . Other options are "fiona" and "pyogrio"

        """
        avail_layers = self.available_layers()
        if layer is None:
            if len(avail_layers) == 1:
                logger.info(f"Only one layer found in {self.path}, using that one: {avail_layers[0]}")
                layer = avail_layers[0]
            else:
                layer = input(f"Select layer [{avail_layers}]:")
        if layer not in avail_layers:
            raise ValueError(
                f"Layer {layer} not available in {self.view_name_with_parents(2)}. Options are: {avail_layers}"
            )

        df = gpd.read_file(self.path, layer=layer, engine=engine)

        if type(df) is pd.DataFrame:
            logger.info(f"Layer {layer} is not a spatial table, returning pandas DataFrame.")
        else:
            df = df.to_crs(epsg_code)

        if index_column:
            if index_column not in df.columns:
                # The id_col may not be in the columns. This is the case for
                # 3Di models where it is a primary_key but not also stored as column.
                # Pyogrio does not support loading primary keys as index directly.
                logger.info(f"index_column `{index_column}` not in columns, trying to load it using SQL query.")
                # Connect to the GPKG file like a normal SQLite database
                with sqlite3.connect(self.path) as conn:
                    # List available tables/layers (optional)
                    id_column = pd.read_sql(f"SELECT {index_column} FROM {layer};", conn)

                df[index_column] = id_column

            df.set_index(index_column, drop=True, inplace=True)

        if columns:
            df = df[columns]

        return df

    def modify_gpkg_using_query(self, query: str):  # TODO test
        """
        Modify a GeoPackage using a SQL query. This function connects to the GeoPackage,
        executes the provided SQL query, and commits the changes.

        The explicit begin and commit statements are necessary
        to make sure we can roll back the transaction

        Parameters
        ----------
        query : str
            The SQL query to execute on the GeoPackage.

        Example
        -------
        # Example usage:
        db = SpatialDatabase("path/to/your.gpkg")
        db.modify_gpkg_using_query("ALTER TABLE your_table ADD COLUMN new_column TEXT")
        """

        # Dont execute empty str.
        if query in [None, ""]:
            logger.warning("Empty query provided, no changes made to the GeoPackage.")
            return

        try:
            # Connect to the GeoPackage
            conn = sqlite3.connect(self.path)
            cursor = conn.cursor()

            # Execute the provided SQL query
            cursor.executescript(f"BEGIN; {query}; COMMIT")

            # Commit changes and close the connection
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"An error occurred while modifying the GeoPackage: {e}")
        finally:
            if conn:
                conn.close()

    # TODO remove in future releases
    def add_layer(self, name: str):
        """Predefine layers so we can write output to that layer."""
        logger.warning("Function add_layer(s) will be deprecated in future releases. Use gpd.to_file instead.")
        if name not in self.layerlist:
            new_layer = SpatialDatabaseLayer(name, parent=self)
            self.layerlist.append(name)
            setattr(self.layers, name, new_layer)

    # TODO remove in future releases
    def add_layers(self, names: list):
        """Add multiple layers"""
        for name in names:
            self.add_layer(name)

    def available_layers(self):
        """Return available layers in file gdb"""
        return fiona.listlayers(self.base)

    def __repr__(self):
        repr_str = f"""{self.path.name} @ {self.path}
exists: {self.exists()}
type: {type(self)}

functions: {get_functions(self)}
variables: {get_variables(self)}
layers (access through .layers): {self.layerlist}"""
        return repr_str


class SpatialDatabaseLayer:
    def __init__(self, name: str, parent: SpatialDatabase):
        self.name = name
        self.parent = parent

    def load(self) -> gpd.GeoDataFrame:
        return gpd.read_file(self.parent.path, layer=self.name, engine="pyogrio")


class FileGDB(SpatialDatabase):
    def __init__(self, base):
        super().__init__(base)
        warnings.warn(
            "FileGDB is deprecated and will be removed in a future release. Please use SpatialDatabase instead",
            DeprecationWarning,
            stacklevel=2,
        )


class FileGDBLayer(SpatialDatabaseLayer):
    def __init__(self, name: str, parent):
        warnings.warn(
            "FileGDBLayer is deprecated and will be removed in a future release. Please use SpatialDatabaseLayer instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(name, parent)


# %%
