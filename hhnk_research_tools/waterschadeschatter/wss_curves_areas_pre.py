from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
from rasterio import features
from shapely.geometry import box
from tqdm import tqdm

import hhnk_research_tools as hrt
from hhnk_research_tools.variables import DEFAULT_NODATA_VALUES
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import split_geometry_in_tiles, pad_zeros

# globals
PREFIX = "damage_curve"
BAG_FUNCTIES = {
    "woonfunctie": 2,
    "celfunctie": 3,
    "industriefunctie": 4,
    "kantoorfunctie": 5,
    "winkelfunctie": 5,
    "kas": 7,
    "logiesfunctie": 8,
    "bijeenkomstfunctie": 9,
    "sportfunctie": 10,
    "onderwijsfunctie": 11,
    "gezondheidszorgfunctie": 12,
    "overige gebruiksfunctie": 13,
}


class CustomLanduse:
    """
    Create a custom landuse based on the input landuse.
    A custom landuse needed due to differences in the buildings used in height models and creation of landuse maps.

    DCCUSTOMLanduse:
        - Masks all bag data and replaces it with water.
        - Pushed buildings onto the landuse as number 2.
        - Does it tiled.

    """

    def __init__(
        self, panden_path: Union[str, Path], landuse_raster_path: Union[str, Path], tile_size: int = 1000
    ) -> None:
        self.landuse = hrt.Raster(landuse_raster_path)
        self.panden = gpd.read_file(panden_path)
        self.tile_size = tile_size

        bbox_gdal = self.landuse.metadata.bbox_gdal
        self.extent = gpd.GeoDataFrame(geometry=[box(*bbox_gdal)], crs=self.landuse.metadata.projection)

    def tiles(self) -> gpd.GeoDataFrame:
        """Generate tiles for processing the landuse raster in chunks."""
        return split_geometry_in_tiles(self.extent.geometry.iloc[0], envelope_tile_size=self.tile_size)

    def run(self, output_dir: Union[str, Path]) -> None:
        """Create custom landuse raster tiles by replacing areas with building footprints."""
        tiles = self.tiles()
        src = self.landuse.open_rio()

        for i in tqdm(range(len(tiles)), "Creating custom landuse"):
            tile = tiles.iloc[i]

            metadata = hrt.RasterMetadataV2.from_gdf(
                gdf=gpd.GeoDataFrame(tile).set_geometry(tile.name), res=self.landuse.metadata.pixel_width
            )

            lu_array = self.landuse.read_geometry(tile.geometry, set_nan=False)

            mask = (lu_array >= 2) & (lu_array <= 14)
            lu_array[mask] = 254

            # Get panden that intersect with tile
            tile_panden = self.panden[self.panden.intersects(tile.geometry)]

            if len(tile_panden) > 0:
                # check functie per pand en vul dat in
                for functie in tile_panden["gebruiksdoel"].unique():
                    if functie in BAG_FUNCTIES:
                        tile_panden_function = tile_panden[tile_panden["gebruiksdoel"] == functie]
                        BAG_nummer = BAG_FUNCTIES[functie]
                    elif functie is None:
                        tile_panden_function = tile_panden[tile_panden["gebruiksdoel"].isnull()]
                        BAG_nummer = 2
                    elif "," in functie:
                        functie_split = functie.split(",")[-1]
                        tile_panden_function = tile_panden[tile_panden["gebruiksdoel"] == functie]
                        BAG_nummer = BAG_FUNCTIES[functie_split]
                    else:
                        tile_panden_function = tile_panden[tile_panden["gebruiksdoel"] == functie]
                        BAG_nummer = 2

                    # Rasterize panden for this tile
                    panden_array = features.rasterize(
                        [(geom, 1) for geom in tile_panden_function.geometry],
                        out_shape=lu_array.shape,
                        transform=metadata.affine,
                        dtype=np.uint8,
                    )

                    lu_array[panden_array == 1] = BAG_nummer

            output_path = Path(output_dir) / f"{PREFIX}_lu_tile_{i}.tif"
            hrt.save_raster_array_to_tiff(
                output_file=output_path,
                raster_array=lu_array,
                metadata=metadata,
                nodata=DEFAULT_NODATA_VALUES["uint8"],
                overwrite=True,
            )
