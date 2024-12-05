# %%
import importlib
import os

import hhnk_threedi_tools as htt

import hhnk_research_tools as hrt

htt = importlib.reload(htt)
hrt = importlib.reload(hrt)

import geopandas as gpd
import hhnk_threedi_tools as htt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hhnk_research_tools as hrt
import hhnk_research_tools.waterschadeschatter.wss_calculations as wss_calculations
import hhnk_research_tools.waterschadeschatter.wss_loading as wss_loading
import hhnk_research_tools.waterschadeschatter.wss_main as wss_main

importlib.reload(wss_main)
importlib.reload(wss_loading)
importlib.reload(wss_calculations)

from osgeo import gdal

gdal.UseExceptions()


# %%
from shapely.geometry import box

# pgb_all_path = r'E:\github\wvangerwen\hhnk_schadeschatter\01_data\peilgebieden_indeling.shp'
pgb_all_path = r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\01_peilgebieden\ad_peilgebieden_raw.shp"
# dem_path = r"G:\02_Werkplaatsen\06_HYD\Projecten\HKC16015 Wateropgave 2.0\11. DCMB\hhnk-modelbuilder-master\data\fixed_data\DEM\DEM_AHN4_int.vrt"
dem_path = r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\03_downloads\ahn3_tiles\combined_rasters.vrt"
output_dir = r"E:\github\wvangerwen\hhnk_schadeschatter\01_data\drainage_areas"
dem_raster = hrt.Raster(dem_path)
gdf = gpd.read_file(pgb_all_path)

gdf["dirname"] = gdf.apply(lambda x: f"pgb_{x['id']}_{x['code']}", axis=1)
gdf["dirname"] = gdf["dirname"].apply(lambda x: x.replace(":", "_"))
gdf["boxarea"] = gdf.bounds.apply(lambda x: box(*x).area, axis=1)


ws_range = range(0, 260, 10)
ws_range = range(100, 110, 10)


class PgbFolder(htt.core.folders.Folder):
    def __init__(self, base):
        super().__init__(base)

        for ws in ws_range:
            self.add_file(objectname=f"ws_{ws}", filename=f"ws_{ws}.tif")
            self.add_file(objectname=f"depth_{ws}", filename=f"depth_{ws}.tif")
            self.add_file(objectname=f"dmg_{ws}", filename=f"dmg_{ws}.tif")


# %%


# %%
# ~~ bereken waterstand ~~ #
OVERWRITE = False

for index, row in gdf.iterrows():
    if row["boxarea"] > 50000000:
        # TODO find solution for bigger areas
        continue
    else:
        pgb_folder = PgbFolder(base=os.path.join(output_dir, row["dirname"]))
        pgb_gdf = gpd.GeoDataFrame(row).T

        meta = hrt.RasterMetadataV2.from_gdf(gdf=pgb_gdf, res=dem_raster.metadata["pixel_width"])
        for ws in ws_range:
            pgb_gdf["wlvl"] = pgb_gdf["streefpeil"] + ws / 100

            out_path = getattr(pgb_folder, f"ws_{ws}").path
            if not os.path.exists(out_path):
                wlvl_array = hrt.gdf_to_raster(
                    gdf=pgb_gdf, value_field="wlvl", raster_out=out_path, nodata=-9999, metadata=meta, read_array=False
                )


# %%


# ~~ bereken waterdiepte~~ #
def calculate_depth(self, window_small, window_big, band_out):
    """calculate depth for pgb
    raster_big=dem
    raster_small=pgb. Should be added to Raster_calculator.custom_run
    then use Raster_calculator.run to run this function.
    """
    # Load windows
    block_big = self.raster_big._read_array(window=window_big)
    block_small = self.raster_small._read_array(window=window_small)

    # Calculate output
    block_out = block_small - block_big
    block_out[block_out <= -0.01] = self.raster_out.nodata  # depth array should run from -1cm to max depth.
    # At -1 the dmg table starts.

    # Mask output
    block_out[block_big == self.raster_big.nodata] = self.raster_out.nodata
    block_out[block_small == self.raster_small.nodata] = self.raster_out.nodata

    # Write to file
    band_out.WriteArray(block_out, xoff=window_small[0], yoff=window_small[1])


# TODO WSS rond diepteraster af op 2 decimalen.
for index, row in gdf.iterrows():
    if row["boxarea"] > 50000000:
        # TODO find solution for bigger areas
        continue
    else:
        pgb_folder = PgbFolder(base=os.path.join(output_dir, row["dirname"]))
        for ws in ws_range:
            # if ws==250:
            ws_raster = hrt.Raster(getattr(pgb_folder, f"ws_{ws}").path)
            depth_raster = hrt.Raster(getattr(pgb_folder, f"depth_{ws}").path)

            if ws_raster.exists():
                depth_calc = hrt.Raster_calculator(
                    raster1=dem_raster,
                    raster2=ws_raster,
                    raster_out=depth_raster,
                    custom_run_window_function=calculate_depth,
                    verbose=False,
                )
                depth_calc.create(nodata=-9999, overwrite=False)
                depth_calc.verbose = True  # Dont see raster creation but do show progress.
                depth_calc.run()

    break


# %%

cfg_file = r"../01_data/cfg/cfg_hhnk_2020.cfg"
cfg_file = r"../01_data/cfg/cfg_lizard.cfg"

landuse_file = r"../01_data/landuse2019_tiles/waterland_landuse2019.vrt"
landuse_file = r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\03_downloads\landuse2019_tiles\ad_landuse2019.vrt"

wss_settings = {
    "inundation_period": 24,  # uren
    "herstelperiode": "10 dagen",
    "maand": "sep",
    "cfg_file": cfg_file,
    "dmg_type": "gem",
}

importlib.reload(wss_main)
importlib.reload(wss_loading)
importlib.reload(wss_calculations)


for index, row in gdf.iterrows():
    if row["boxarea"] > 50000000:
        # TODO find solution for bigger areas
        continue
    else:
        if row["code"] == "04751-04":
            pgb_folder = PgbFolder(base=os.path.join(output_dir, row["dirname"]))
            for ws in ws_range:
                # if ws==250:

                wss_local = wss_main.Waterschadeschatter(
                    depth_file=getattr(pgb_folder, f"depth_{ws}").path,
                    landuse_file=landuse_file,
                    wss_settings=wss_settings,
                )

                wss_local.run(
                    output_raster=hrt.Raster(getattr(pgb_folder, f"dmg_{ws}").path),
                    calculation_type="sum",
                    verbose=False,
                    overwrite=False,
                )
                # break


# %% TESTING
self = wss_local
caller = wss_local
indices = self.indices
dmg_table_landuse = self.dmg_table_landuse
dmg_table_general = self.dmg_table_general

# Difference between landuse and depth raster.
dx_min, dy_min = hrt.dx_dy_between_rasters(meta_big=self.lu_raster.metadata, meta_small=self.depth_raster.metadata)

pixel_factor = self.depth_raster.pixelarea
blocks_df = self.depth_raster.generate_blocks()
DMG_NODATA = 0  # let op staat dubbel, ook in wss_main.

len_total = len(blocks_df)

damage_block = {}
damage_arr = np.zeros(self.depth_raster.shape)

for idx, block_row in blocks_df.iterrows():
    # if idx==0:
    # Load landuse

    window_depth = block_row["window_readarray"]

    window_lu = window_depth.copy()
    window_lu[0] += dx_min
    window_lu[1] += dy_min

    lu_block = self.lu_raster._read_array(window=window_lu)
    lu_block = lu_block.astype(int)
    lu_block[lu_block == self.lu_raster.nodata] = 0
    if lu_block.mean() != 0:
        # Load depth
        depth_block = self.depth_raster._read_array(window=window_depth)
        depth_mask = depth_block == self.depth_raster.nodata

        depth_block[depth_block == self.depth_raster.nodata] = np.nan  # Schadetabel loopt vanaf -0.01cm
    else:
        print("no lu plz fix")
    # Interpoleren van het diepteraster en de gamma array.
    ysize, xsize = depth_block.shape
    k, j = np.meshgrid(np.arange(xsize), np.arange(ysize))

    xp = np.array(dmg_table_general["inundatiediepte"])

    # Zoek voor de diepte_array de dichtst bijliggende index wanneer vergeleken met xp.
    # Ookal gaat dit 'links', we willen de index hebben daarom krijg index_onder -1
    depth_block = np.round(depth_block, 2)  # FIXME Stowa WSS rond af op 2 decimalen.
    depth_mask = depth_block == caller.depth_raster.nodata

    index_boven = np.searchsorted(xp, depth_block, side="left")
    index_boven[depth_mask] = (
        1  # Tijdelijke oplossing om de berekening wel te kunnen maken. Gamma_inundatiediepte worden later voor dit masker ook op 0 gezet.
    )

    index_onder = index_boven.copy() - 1

    # Stacked array zijn de gamma waarden van de inundatiediepte. Heeft dezelfde vorm als depth_block.
    # En dan x lagen gelijk aan len(xp)
    stacked_array = []
    for i_depth in range(len(xp)):
        lookup = np.asarray([dmg_table_landuse[i].gamma_inundatiediepte[i_depth] for i in range(0, 255)])
        stacked_array.append(lookup[lu_block])  # Max directe schade.
    # stacked_array.append(np.take(lookup, lu_block))
    stacked_array = np.stack(stacked_array)

    # Met de index arrays kunnen we nu de onder en bovenwaarde ophalen om vervolgens linear te interpoleren
    # Bijkomend mag de geinterpoleerde waarde niet groter of kleiner zijn dan xp[-1]
    # Dat wordt bereikt door in wss_loading de cfg waarden voor en achter aan te vullen.

    # gamma_inundatiediepte = (y2-y1)/(x2-x1) * (x-x1) + y1
    np.seterr(invalid="ignore")  # Delen door 0 is niet te voorkomen. Met het mask berekenen we de waarden hier.
    mask = index_onder == index_boven  # dy/dx is hier 0, we nemen dan de gamma_inundatiediepte met de onderste index .
    y1 = stacked_array[index_onder, j, k]
    y2 = stacked_array[index_boven, j, k]
    gamma_inundatiediepte = (np.divide((y2 - y1), (xp[index_boven] - xp[index_onder]))) * (
        depth_block - xp[index_onder]
    ) + y1

    gamma_inundatiediepte[mask] = y1[mask]
    gamma_inundatiediepte[depth_mask] = np.nan

    # Indirecte schade telt alleen bij inundatiediepte >0
    mask_indirect = depth_block <= 0

    # For debugging.
    caller.gamma_inundatiediepte = gamma_inundatiediepte

    # DAMAGE CALCULATION

    # Apply lookup table. Change the landuse with direct damage.
    def calculate_damage_direct(i):
        """schade = max. directe schade · γdiepte · γduur · γseizoen + indirecte schade per dag · hersteltijd"""
        lu = dmg_table_landuse[i]
        return lu.direct * lu.gamma_inundatieduur_interp * lu.gamma_maand[indices["maand"]] * lu.direct_eenheid_factor

    def calculate_damage_indirect(i):
        # inundatiediepte > 0 heeft altijd indirecte schade
        """schade = max. directe schade · γdiepte · γduur · γseizoen + indirecte schade per dag · hersteltijd"""
        lu = dmg_table_landuse[i]
        return lu.indirect * lu.gamma_herstelperiode[indices["herstelperiode"]] * lu.indirect_eenheid_factor

    # Create lookuptable. Landuse value will be replaced by calculated damage.
    lookup_direct = np.asarray([calculate_damage_direct(i) for i in range(0, 255)])
    lookup_indirect = np.asarray([calculate_damage_indirect(i) for i in range(0, 255)])

    # Max directe schade.
    damage_direct = lookup_direct[lu_block]  # same as np.take(lookup_direct, lu_block), .take seems slower.
    damage_indirect = lookup_indirect[lu_block]
    damage_indirect[mask_indirect] = DMG_NODATA

    damage_block = (damage_direct * gamma_inundatiediepte + damage_indirect) * pixel_factor
    damage_arr[block_row["window"][1] : block_row["window"][3], block_row["window"][0] : block_row["window"][2]] = (
        damage_block
    )
# %%

for idx in damage_block:
    fig = plt.imshow(damage_block[idx])
    plt.show()
# %%

plt.imshow(damage_arr)
# %%

# %%

ws = 30
window_old = [27563, 26512, 1, 1]
window_new = [1501, 908, 1, 1]
indices = {"herstelperiode": 5, "maand": 8}

paths = {}
paths["dem"] = (
    r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\03_downloads\ahn3_tiles\combined_rasters.vrt"
)
paths["landuse"] = (
    r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\03_downloads\landuse2019_tiles\ad_landuse2019.vrt"
)
paths["dmg_old"] = (
    rf"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\06_schade\tif\schade_ad_{ws}.tif"
)
paths["dmg_new"] = rf"E:\github\wvangerwen\hhnk_schadeschatter\01_data\drainage_areas\pgb_403_GPG-CB-50\dmg_{ws}.tif"
paths["ws_old"] = (
    rf"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\02_input_wss\ad_{ws}.tif"
)
paths["ws_new"] = rf"E:\github\wvangerwen\hhnk_schadeschatter\01_data\drainage_areas\pgb_403_GPG-CB-50\ws_{ws}.tif"
paths["depth_new"] = (
    rf"E:\github\wvangerwen\hhnk_schadeschatter\01_data\drainage_areas\pgb_403_GPG-CB-50\depth_{ws}.tif"
)

windows = {}
windows["dem"] = window_old
windows["landuse"] = window_old
windows["dmg_old"] = window_old
windows["dmg_new"] = window_new
windows["ws_old"] = window_old
windows["ws_new"] = window_new
windows["depth_new"] = window_new

rasters = {}
for r in paths:
    rasters[r] = hrt.Raster(paths[r])

values = {}
for r in rasters:
    values[r] = rasters[r]._read_array(window=windows[r])[0][0]


# lu.direct_gem * lu.gamma_inundatieduur_interp * lu.gamma_maand[indices['maand']] * lu.direct_eenheid_factor

print(pd.Series(values))

# Bereken schade

x = values["depth_new"]
xp = [-0.01, 0.01, 0.05, 0.15, 0.3, 2.5]
fp = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

gamma_diepte = np.interp(x=x, xp=xp, fp=fp)


1209 * 0.2 * 1 * 1 / 10000 * gamma_diepte * 0.25
