import lazy_loader as _lazy
import time

t0 = time.time()

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)


print(f"Load in {(time.time() - t0):.3f}s threedi.__init__")


# TODO old, remove when all works.
# from hhnk_research_tools.threedi.geometry_functions import (
#     coordinates_to_points,
#     grid_nodes_to_gdf,
#     line_geometries_to_coords,
# )
