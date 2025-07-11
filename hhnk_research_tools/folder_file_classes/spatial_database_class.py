# %%
import types
import warnings
from typing import Optional

import fiona
import geopandas as gpd

from hhnk_research_tools.folder_file_classes.file_class import File
from hhnk_research_tools.general_functions import get_functions, get_variables


class SpatialDatabase(File):
    def __init__(self, base):
        super().__init__(base)

        self.layerlist = []
        self.layers = types.SimpleNamespace()  # empty class

    def load(self, layer: Optional[str] = None) -> gpd.GeoDataFrame:
        """Load layer from geodataframe. When no layer provided, use the default / first one."""
        avail_layers = self.available_layers()
        if layer is None:
            if len(avail_layers) == 1:
                layer = avail_layers[0]
            else:
                layer = input(f"Select layer [{avail_layers}]:")
        if layer not in avail_layers:
            raise ValueError(
                f"Layer {layer} not available in {self.view_name_with_parents(2)}. Options are: {avail_layers}"
            )
        gdf = gpd.read_file(self.path, layer=layer, engine="pyogrio")
        return gdf

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
        return gpd.read_file(self.parent.base, layer=self.name, engine="pyogrio")


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
