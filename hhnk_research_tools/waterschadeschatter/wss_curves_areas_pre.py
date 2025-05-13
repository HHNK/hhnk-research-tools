import rioxarray
import geopandas as gp
import numpy as np
import xarray as xr
from rasterio import features
import hhnk_research_tools as hrt

def classify_panden(landuse_raster_path, output_path):

    # Load raster data with rioxarray
    raster = rioxarray.open_rasterio(landuse_raster_path, chunks={"x": 1024, "y": 1024})

    # Apply the classification function
    classified_raster = raster.map_blocks(classify_pixels)

    # Save the result to a new GeoTIFF file
    classified_raster.rio.to_raster(output_path, chunks={"x": 1024, "y": 1024}, compress='zstd'})

def rasterize_panden(panden_path, example_raster_path, output_path):

    panden = gp.read_file(panden_path,layer = 'BAG_panden_20241001_HHNK' )
    example_raster = hrt.Raster(example_raster_path)

    geometry = panden.geometry
    if hasattr(geometry, "geoms"):
        geometry_list = list(geometry.geoms)
    else:
        geometry_list = [geometry]

    array = features.rasterize(
        geometry_list,
        out_shape=example_raster.metadata.shape,
        transform=example_raster.metadata.georef,
    )

    hrt.save_raster_array_to_tiff(
        output_file=output_path,
        raster_array=array,
        nodata=example_raster.nodata,
        overwrite=True,
)


# Define classification function
def classify_pixels(data_array):
    """Classifies pixels based on value ranges."""

    data = data_array.values

    classified = np.select(
        condlist=[(data >= 2) & (data <= 14)], # in essentie all bag klasses
        choicelist=[254],  # Water
    )

    classified_da = xr.DataArray(
    classified,
    dims=data_array.dims,  # Use the same dimensions as the input
    coords=data_array.coords,  # Use the same coordinates as the input
    attrs=data_array.attrs  # Preserve attributes
)

    return classified_da

if __name__ == "__main__":
    redo_panden(, )
    landuse_raster_path =r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\data\rasters\landgebruik\landuse2023_tiles\all.vrt" 
    classified_landuse_path = r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\landgebruik_panden_vervangen/all.tif"
     classify_panden

    panden_path = r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\data\vectors\BAG_panden_20241001_HHNK_RUW.gpkg"
    example_raster_path = r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\data\rasters\landgebruik\landuse2023_tiles\all.vrt"
    output_path = r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\landgebruik_panden_vervangen/panden_rasterized.tif"
    rasterize_panden(panden__path, example_raster_path, output_path)