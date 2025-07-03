import pathlib
import numpy as np
from tqdm import tqdm
from rasterio import features
from shapely.geometry import box
import geopandas as gp
from typing import Optional, Union

import hhnk_research_tools as hrt
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import split_geometry_in_tiles
from hhnk_research_tools.variables import DEFAULT_NODATA_VALUES

# globals
PREFIX = "damage_curve"

class DCCustomLanduse:
    """ 
    Create a custom landuse based on the input landuse.
    A custom landuse needed due to differences in the buildings used in height models and creation of landuse maps.

    DCCUSTOMLanduse:
        - Masks all bag data and replaces it with water.
        - Pushed buildings onto the landuse as number 2.
        - Does it tiled.

    """
    def __init__(self, panden_path:str, landuse_raster_path:str, tile_size:int=1000):
        
        self.landuse = hrt.Raster(landuse_raster_path)
        self.panden = gp.read_file(panden_path)
        self.tile_size = tile_size

        bbox_gdal = self.landuse.metadata.bbox_gdal
        self.extent = gp.GeoDataFrame(
        geometry=[box(*bbox_gdal)], 
        crs=self.landuse.metadata.projection
        )

    def tiles(self):
        return split_geometry_in_tiles(
        self.extent.geometry.iloc[0],
        envelope_tile_size=self.tile_size
    )

    def run(self, output_dir):
        tiles = self.tiles()
        src = self.landuse.open_rio()

        for i in tqdm(range(len(tiles)), "Creating custom landuse"):
            tile = tiles.iloc[i]

            metadata = hrt.RasterMetadataV2.from_gdf(gdf=gp.GeoDataFrame(tile).set_geometry(tile.name),
                                           res=self.landuse.metadata.pixel_width)

            lu_array = self.landuse.read_geometry(tile.geometry,set_nan=False)

            mask = (lu_array >= 2) & (lu_array <= 14)
            lu_array[mask] = 254

            # Get panden that intersect with tile
            tile_panden = self.panden[self.panden.intersects(tile.geometry)]
            
            if len(tile_panden) > 0:

                # Rasterize panden for this tile
                panden_array = features.rasterize(
                    [(geom, 1) for geom in tile_panden.geometry],
                    out_shape=metadata.shape,
                    transform=metadata.affine,
                    dtype=np.uint8
                )

                lu_array[panden_array == 1] = 2

            output_path = pathlib.Path(output_dir) / f"{PREFIX}_lu_tile_{i}.tif"
            hrt.save_raster_array_to_tiff(
                output_file=output_path,
                raster_array=lu_array,
                metadata=metadata,
                nodata=DEFAULT_NODATA_VALUES['uint8'],
                overwrite=True
            )
