# %%
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
DEFAULT_SRID = 28992


class SpatialDatabase(File):
    def __init__(self, base):
        super().__init__(base)

        self.layerlist = []
        self.layers = types.SimpleNamespace()  # empty class

    def load(
        self,
        layer: Optional[str] = None,
        id_col: Optional[str] = None,
        columns: Optional[list] = None,
        epsg_code: int = DEFAULT_SRID,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Load layer as geodataframe. When no layer provided, use the default / first one."""
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

        df = gpd.read_file(self.path, layer=layer, engine="pyogrio")

        if type(df) is pd.DataFrame:
            logger.info(f"Layer {layer} is not a spatial table, returning pandas DataFrame.")
        else:
            df = df.to_crs(epsg_code)

        if columns:
            df = df[columns]
        if id_col:
            df.set_index(id_col, drop=True, inplace=True)

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
        import sqlite3

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

    def add_layer(self, name: str):
        """Predefine layers so we can write output to that layer."""
        if name not in self.layerlist:
            new_layer = SpatialDatabaseLayer(name, parent=self)
            self.layerlist.append(name)
            setattr(self.layers, name, new_layer)

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
