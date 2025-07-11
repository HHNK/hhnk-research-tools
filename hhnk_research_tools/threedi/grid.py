# First party imports
import tempfile

import geopandas as gpd
import pandas as pd
import shapely.wkt as wkt

import hhnk_research_tools as hrt
from hhnk_research_tools.threedi.geometry_functions import line_geometries_to_coords, point_geometries_to_wkt
from hhnk_research_tools.threedi.variables.results_mapping import one_d_two_d

DEF_TRGT_CRS = 28992
one_d_node_id_col = "1dnode_id"
node_id_col = "node_id"
node_geometry_col = "node_geometry"
node_type_col = "node_type"
init_wlevel_col = "initial_waterlevel"
storage_area_col = "storage_area"

connection_val = "connection"
added_calc_val = "added_calculation"
one_d_two_d_crosses_levee_val = "1d2d_crosses_levee"
one_d_two_d_crosses_fixed = "1d2d_crosses_fixeddrainage"
levee_id_col = "levee_id"
levee_height_col = "levee_height"

type_col = "type"


# Third-party imports


class Grid:
    def __init__(self, grid_folder=None, sqlite_path=None, dem_path=None):
        # moved imports here because gridbuilder has h5py issues
        # TODO gridadmin should be loaded from ThreediResult
        from threedigrid.admin.gridadmin import GridH5Admin
        from threedigrid.admin.gridresultadmin import GridH5ResultAdmin
        from threedigrid_builder import make_gridadmin

        self.folder = hrt.Folder(grid_folder)
        self.sqlite_path = sqlite_path
        self.dem_path = dem_path

        if sqlite_path and dem_path:
            # using output here results in error, so we use the returned dict
            grid = make_gridadmin(sqlite_path, dem_path)
            self.nodes = hrt.df_convert_to_gdf(pd.DataFrame(grid["nodes"]), geom_col_type="wkb", src_crs="28992")
            self.lines = hrt.df_convert_to_gdf(pd.DataFrame(grid["lines"]), geom_col_type="wkb", src_crs="28992")
            self.cells = hrt.df_convert_to_gdf(pd.DataFrame(grid["cells"]), geom_col_type="wkb", src_crs="28992")

        if not grid_folder and (sqlite_path and dem_path):
            self.grid_dir = tempfile.TemporaryDirectory()
            self.folder = hrt.Folder(self.grid_dir.name)
            make_gridadmin(sqlite_path, dem_path, self.folder.path / "gridadmin.h5")

        if self.folder.exists():
            self.admin_path = self.folder.path / "gridadmin.h5"
            self.grid_path = self.folder.path / "results_3di.nc"

            if self.grid_path.exists():
                self.grid = GridH5ResultAdmin(str(self.admin_path), str(self.grid_path))

            if self.admin_path.exists():
                self.admin = GridH5Admin(str(self.admin_path))

    def read_1d2d_lines(self):
        return read_1d2d_lines(self.admin)

    def import_levees(self):
        return import_levees(self.admin)


def read_1d2d_lines(results):
    """Uitlezen 1d2d lijnelementen
    Alle 1d2d lijnelementen in het model.
    """
    try:
        # Creates geodataframe with geometries of 1d2d subset of nodes in 3di results
        coords = line_geometries_to_coords(results.lines.subset(one_d_two_d).line_geometries)
        one_d_two_d_lines_gdf = gpd.GeoDataFrame(geometry=coords, crs=f"EPSG:{DEF_TRGT_CRS}")

        # 1d nodes om te bepalen bij welk kunstwerk het hoort
        one_d_two_d_lines_gdf[one_d_node_id_col] = [a[1] for a in results.lines.subset(one_d_two_d).line_nodes]
        one_d_two_d_lines_gdf[node_id_col] = results.nodes.filter(
            id__in=one_d_two_d_lines_gdf[one_d_node_id_col].tolist()
        ).content_pk
        one_d_two_d_lines_gdf.index = one_d_two_d_lines_gdf[one_d_node_id_col]

        # Get values corresponding to id's in onetwo_line_geo from results and add to dataframe
        oned_nodes_list = results.nodes.filter(id__in=one_d_two_d_lines_gdf[one_d_node_id_col].tolist())
        oned_conn_nodes_id_list = oned_nodes_list.connectionnodes.id.tolist()
        oned_conn_nodes_init_wlvl_list = oned_nodes_list.connectionnodes.initial_waterlevel.tolist()
        oned_conn_nodes_storage_area_list = oned_nodes_list.connectionnodes.storage_area
        oned_added_calculation_nodes_list = oned_nodes_list.added_calculationnodes.id.tolist()

        # Add node geometries
        one_d_two_d_lines_gdf[node_geometry_col] = point_geometries_to_wkt(oned_nodes_list.coordinates)

        # Add information about node type
        one_d_two_d_lines_gdf[node_type_col] = None  # if frame is empty this prevents .loc from producing an error.
        one_d_two_d_lines_gdf.loc[
            one_d_two_d_lines_gdf[one_d_node_id_col].isin(oned_conn_nodes_id_list),
            node_type_col,
        ] = connection_val
        one_d_two_d_lines_gdf.loc[
            one_d_two_d_lines_gdf[one_d_node_id_col].isin(oned_added_calculation_nodes_list),
            node_type_col,
        ] = added_calc_val

        # Add initial waterlevel to nodes
        one_d_two_d_lines_gdf.loc[
            one_d_two_d_lines_gdf[one_d_node_id_col].isin(oned_conn_nodes_id_list),
            init_wlevel_col,
        ] = oned_conn_nodes_init_wlvl_list

        # Add storage area from connection nodes to the table
        storage_area_lst = oned_conn_nodes_storage_area_list
        # storage_area_lst = [a.decode(UTF8) for a in oned_conn_nodes_storage_area_list]
        one_d_two_d_lines_gdf.loc[
            one_d_two_d_lines_gdf[one_d_node_id_col].isin(oned_conn_nodes_id_list),
            storage_area_col,
        ] = storage_area_lst
        one_d_two_d_lines_gdf[storage_area_col] = pd.to_numeric(one_d_two_d_lines_gdf[storage_area_col])
        return one_d_two_d_lines_gdf
    except Exception as e:
        raise e from None


def import_levees(results):
    def levees_to_linestring(levee_geom):
        levee_linestr = []
        for line in levee_geom:
            line.FlattenTo2D()  # Er staat nog een hoogte opgeslagen in de levee van 0. Deze wordt verwijderd.
            levee_linestr.append(wkt.loads(line.ExportToWkt()))
        return levee_linestr

    levee_line = levees_to_linestring(results.levees.geoms)
    levee_line_geo = gpd.GeoDataFrame(geometry=levee_line, crs=f"EPSG:{DEF_TRGT_CRS}")
    levee_line_geo[levee_id_col] = results.levees.id
    levee_line_geo[levee_height_col] = results.levees.crest_level
    levee_line_geo.index = levee_line_geo[levee_id_col]
    return levee_line_geo
