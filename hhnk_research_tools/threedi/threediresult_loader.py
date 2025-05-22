# %%
# TODO unused
levee_id_col = "levee_id"
levee_height_col = "levee_height"


class ThreediResultLoader:
    def __init__(self, grid):
        self.grid = grid

    def levee():
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


# %%
