# hhnk_research_tools/__init__.py
import time

# t0 = time.time()
from typing import TYPE_CHECKING
import lazy_loader as _lazy
from importlib.metadata import version


if TYPE_CHECKING:
    # hhnk_research_tools/__init__.pyi
    from . import (
        folder_file_classes,
        gis,
        installation_checks,
        logging,
        rasters,
        threedi,
        variables,
        waterschadeschatter,
    )

    # Exposing functions to top level
    from .dataframe_functions import (
        df_add_geometry_to_gdf,
        df_convert_to_gdf,
        gdf_write_to_csv,
        gdf_write_to_geopackage,
    )
    from .folder_file_classes.file_class import File
    from .folder_file_classes.folder_file_classes import (
        Folder,
    )
    from .folder_file_classes.spatial_database_class import (
        FileGDB,  # TODO pending deprecation
        FileGDBLayer,  # TODO pending deprecation
        SpatialDatabase,
        SpatialDatabaseLayer,
    )
    from .folder_file_classes.sqlite_class import (
        Sqlite,
    )
    from .folder_file_classes.threedi_schematisation import (
        RevisionsDir,
        ThreediResult,
        ThreediSchematisation,
    )
    from .general_functions import (
        check_create_new_file,
        convert_gdb_to_gpkg,
        current_time,
        dict_to_class,
        ensure_file_path,
        get_functions,
        get_pkg_resource_path,
        get_uuid,
        get_variables,
        load_source,
        time_delta,
    )
    from .gis.interactive_map import create_interactive_map
    from .gis.raster import RasterMetadata, RasterOld  # noqa: F401
    from .gis.raster_calculator import RasterBlocks, RasterCalculatorV2
    from .raster_functions import (
        RasterCalculator,
        build_vrt,
        create_meta_from_gdf,
        create_new_raster_file,
        dx_dy_between_rasters,
        gdf_to_raster,
        hist_stats,
        reproject,
        save_raster_array_to_tiff,
    )
    from .rasters.raster_calculator_rxr import RasterCalculatorRxr
    from .rasters.raster_class import Raster, RasterChunks
    from .rasters.raster_metadata import RasterMetadataV2
    from .sql_functions import (
        create_sqlite_connection,
        database_to_gdf,
        execute_sql_changes,
        execute_sql_selection,
        sql_builder_select_by_location,
        sql_construct_select_query,
        sql_create_update_case_statement,
        sql_table_exists,
        sqlite_replace_or_add_table,
        sqlite_table_to_df,
        sqlite_table_to_gdf,
    )
    from .threedi.call_api import call_threedi_api
    from .threedi.read_api_file import read_api_file

    # curves
    from .waterschadeschatter.wss_curves_areas import AreaDamageCurves
    from .waterschadeschatter.wss_curves_areas_post import AreaDamageCurvesAggregation

    # waterschadeschatter
    from .waterschadeschatter.wss_main import Waterschadeschatter

    # ... all other symbols you expose for typing
else:
    __version__ = version("hhnk_research_tools")
    __getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

    # print(f"Load in {(time.time() - t0):.3f}s main.__init__ ")

    # Set default logg ing to console
    from .logging import set_default_logconfig

    # FIXME not ideal to do here. can we do this on the level of caller packages?
    set_default_logconfig(
        level_root="WARNING",
        level_dict={
            "DEBUG": ["__main__"],
            "INFO": ["hrt", "htt"],
            "ERROR": ["fiona", "rasterio"],
        },
    )


__doc__ = """
General toolbox for loading, converting and saving serval datatypes.
"""
# print(f"Load2 in {(time.time() - t0):.3f}s main.__init__")
