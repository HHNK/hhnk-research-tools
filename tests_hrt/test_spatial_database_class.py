# %%
import pandas as pd

from hhnk_research_tools.folder_file_classes.spatial_database_class import SpatialDatabase, SpatialDatabaseLayer
from tests_hrt.config import TEST_DIRECTORY


def test_spatial_database_class():
    """Test the SpatialDatabase and SpatialDatabaseLayer"""
    # %%
    base = TEST_DIRECTORY.joinpath("area_test.gpkg")
    layer_name = "area_test"
    db = SpatialDatabase(base=base)

    # Check loading layer directly
    assert isinstance(db.load(layer=layer_name), pd.DataFrame)

    # Check initial state
    assert db.layerlist == []

    available_layers = db.available_layers()
    assert available_layers == [layer_name]

    # Add a layer
    db.add_layer(layer_name)
    assert layer_name in db.layerlist
    assert hasattr(db.layers, layer_name)

    # Load the layer with SpatialDatabaseLayer
    layer: SpatialDatabaseLayer = getattr(db.layers, layer_name)
    assert isinstance(layer.load(), pd.DataFrame)


# %% Run test
if __name__ == "__main__":
    test_spatial_database_class()
# %%
