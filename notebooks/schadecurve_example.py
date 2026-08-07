# %%
"""Tested on 20260807 by WVG"""

# Imports
from pathlib import Path

import geopandas as gpd
from osgeo import gdal
from pandas.api.types import is_integer_dtype, is_string_dtype

import hhnk_research_tools as hrt
from hhnk_research_tools.waterschadeschatter.wss_curves_areas_post import DEFAULT_RAIN

gdal.UseExceptions()
MAIN_DIR = Path(__file__).parent
INPUT_DIR = MAIN_DIR / "raw_input"

OUTPUT_DIR = MAIN_DIR / "wss_curve_berekening"
# NOTE achteraf aanpassen van mappenstructuur kan problemen geven bij het vinden van de juiste bestanden
# in de output worden op verschillende plekken namelijk paden opgeslagen.
# %%

# Pre validate input (AreaDamageCurves will fail with a less clear error)
landuse_path_dir = INPUT_DIR / "landuse.tif"
dem_path_dir = INPUT_DIR / "dem.tif"
buildings_path = INPUT_DIR / "buildings.gpkg"
buildings_id_field = "idx"
area_path = INPUT_DIR / "peilgebieden.gpkg"
area_start_level_field = "drainage_level"
area_id = "pid"
aggregate_vector_path = INPUT_DIR / "polders.gpkg"
aggregate_vector_id_field = "NAAM"


def validate_id_field(gdf: gpd.GeoDataFrame, id_field: str, name: str, expected_type: str) -> None:
    if id_field not in gdf.columns:
        raise ValueError(f"'{name}': ID field '{id_field}' not found in GeoDataFrame columns.")
    if gdf[id_field].duplicated().any():
        raise ValueError(f"'{name}': ID field '{id_field}' contains duplicate values.")
    dtype = gdf[id_field].dtype
    if expected_type == "int" and not is_integer_dtype(dtype):
        raise ValueError(f"'{name}': ID field '{id_field}' ({dtype}) is not an integer type.")
    if expected_type == "str" and not is_string_dtype(dtype):
        raise ValueError(f"'{name}': ID field '{id_field}' ({dtype}) is not a string type.")


area = gpd.read_file(area_path)
validate_id_field(area, area_id, "area", expected_type="int")

buildings = gpd.read_file(buildings_path)
validate_id_field(buildings, buildings_id_field, "buildings", expected_type="int")

polders = gpd.read_file(aggregate_vector_path)
validate_id_field(polders, aggregate_vector_id_field, "polders", expected_type="str")

dem = hrt.Raster(dem_path_dir)
landuse = hrt.Raster(landuse_path_dir)


def repair_unique_id_field(gdf: gpd.GeoDataFrame, id_field: str) -> gpd.GeoDataFrame:
    """Reset index and drop original id field. Use manually and write afterwards to file."""
    gdf = gdf.drop(id_field, axis=1)
    gdf = gdf.reset_index(drop=False, names=[id_field])
    return gdf


# b = repair_unique_id_field(buildings, buildings_id_field)
# b.to_file(buildings_path)

# %%
# Aanmaken AreaDamageCurves object
adc = self = hrt.AreaDamageCurves(
    output_dir=OUTPUT_DIR,
    landuse_path_dir=landuse_path_dir,
    # Dit is de DEM in mNAP. Tijdens berekening rekenen we het om naar cmNAP als integers. (DAMAGE_DECIMALS)
    dem_path_dir=dem_path_dir,
    buildings_path=buildings_path,
    buildings_id_field=buildings_id_field,
    wss_settings_file="wss_settings.json",
    area_path=area_path,
    area_id=area_id,
    area_start_level_field=area_start_level_field,
    curve_step=0.1,
    curve_max=2.5,
    resolution=0.5,
    area_layer_name=None,  # Als meerdere layers in area_path
    nodamage_file=None,  # Als gebruikt, dan ook nodamage_filter = True zetten bij adc.run()
    wss_curves_filter_settings_file="settings_curves_filter.json",
    wss_config_file=hrt.get_pkg_resource_path(
        package_resource=hrt.waterschadeschatter.resources, name="cfg_lizard.cfg"
    ),
    settings_json_file=None,  # Werkt niet lekker. Schrijft de gebruikte instellingen niet weg.
    database_file=None,  # Niet gebruiken, kijkt naar oracle db als alternatief van area_path.
    overwrite=False,
    area_ids=None,  # Filtering binnen area_path
)
# %%
# Uitvoeren hoofdberekening
area_ids = None
run_1d = True
run_2d = False
multiprocessing = False
processes = -1
nodamage_filter = False
depth_filter = True
write = True
area_data_kwargs = {}
adc.run(
    area_ids=None,  # Als je bij hrt.AreaDamageCurves al area_ids invult, dan kan hier None staan.
    run_1d=True,  # Sneller dan 2d
    run_2d=False,
    multiprocessing=False,  # True werkte bij mij niet.
    processes=-1,
    nodamage_filter=False,  # Dit is een extra filter naast de depth_filter. Gebruikt de nodamage_file
    depth_filter=True,  # doet berekening op diepte in settings_curves_filter.json en filtert schades.
    write=True,
)

# %%
# Post-processing
# Aggregatie object aanmaken
agg = self = hrt.AreaDamageCurvesAggregation(
    result_path=OUTPUT_DIR,
    aggregate_vector_path=aggregate_vector_path,
    aggregate_vector_id_field=aggregate_vector_id_field,
    aggregate_vector_layer_name=None,
    # landuse_conversion_path="landuse_conversion.csv", # Staat default goed
)

# Aggregatie uitvoeren
aggregation = True
mm_rain = DEFAULT_RAIN
create_html = True
agg.run(aggregation=aggregation, mm_rain=mm_rain, create_html=create_html)

# %%
