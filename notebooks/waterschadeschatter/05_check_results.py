# %%
"""
Controle van de resultaten. Vergeleken met de resultaten uit de WSS van STOWA en de WSS uit Lizard

"""

# import threedi_raster_edits as tre
import numpy as np
from hhnk_threedi_tools import Folders

import hhnk_research_tools.waterschadeschatter.wss_main as wss_main

# User input
folder_path = r"E:\02.modellen\23_Katvoed"
scenario = "katvoed #1 piek_ghg_T1000"  # mapnaam


folder = Folders(folder_path)

dem_path = folder.model.schema_base.rasters.dem.path
dem_path = r"E:\\02.modellen\\23_Katvoed\\02_schematisation\\00_basis\\rasters/dem_katvoed_ahn3.tif"

threedi_result = folder.threedi_results.one_d_two_d[scenario]


import hhnk_research_tools as hrt

depth_out = threedi_result.full_path("depth_local_min_lizard.tif")

depth_local = threedi_result.full_path("wdepth_orig_ahn3.tif")
depth_lizard = threedi_result.full_path("depth_for_lizard_dmg_res0_5m.tif")

# %%


def difference_between_rasters(self, window_small, window_big, band_out):
    # Load windows
    block_big = self.raster_big._read_array(window=window_big)
    block_small = self.raster_small._read_array(window=window_small)

    # Calculate output
    block_out = block_small - block_big
    block_out[np.logical_and(block_out < 0.001, block_out > -0.001)] = (
        self.raster_out.nodata
    )  # depth array should run from -1cm to max depth.
    # At -1 the dmg table starts.

    # Mask output
    block_out[block_big == self.raster_big.nodata] = self.raster_out.nodata
    block_out[block_small == self.raster_small.nodata] = self.raster_out.nodata

    # Write to file
    band_out.WriteArray(block_out, xoff=window_small[0], yoff=window_small[1])


# %% Verschil diepte Lizard en lokaal
depth_calc = hrt.Raster_calculator(
    raster1=depth_local,
    raster2=depth_lizard,
    raster_out=depth_out,
    custom_run_window_function=difference_between_rasters,
    verbose=False,
)
depth_calc.create(nodata=-9999, overwrite=False)
depth_calc.verbose = True  # Dont see raster creation but do show progress.
depth_calc.run()


# %%
# Schadeschatter heeft wat extra voorbereiding nodig.
from pathlib import Path

schadeschatter_path = Path(r"E:\01.basisgegevens\hhnk_schadeschatter")

import sys

if str(schadeschatter_path) not in sys.path:
    sys.path.append(str(schadeschatter_path))


# Variables
# cfg_file = schadeschatter_path/'01_data/cfg/cfg_hhnk_2020.cfg'
cfg_file = schadeschatter_path / "01_data/cfg/cfg_lizard.cfg"
landuse_file = schadeschatter_path / "01_data/landuse2019_3di_tiles/combined_rasters.vrt"

depth_file = depth_lizard.path
output_raster = threedi_result.full_path("damage_local_lizard_settings.tif")

wss_settings = {
    "inundation_period": 48,  # uren
    "herstelperiode": "10 dagen",
    "maand": "sep",
    "cfg_file": cfg_file,
    "dmg_type": "gem",
}

# Calculation
self = wss_main.Waterschadeschatter(depth_file=depth_file, landuse_file=landuse_file, wss_settings=wss_settings)


# Berekenen schaderaster
self.run(
    output_raster=output_raster,
    calculation_type="sum",
    verbose=True,
    overwrite=False,
)


# %% Verschil schade Lizard en lokaal

damage_local = output_raster
damage_lizard = threedi_result.full_path("total_damage_res0_5m.tif")
diff_out = threedi_result.full_path("diff_damage_local_min_lizard_noround.tif")

diff_calc = hrt.Raster_calculator(
    raster1=damage_local,
    raster2=damage_lizard,
    raster_out=diff_out,
    custom_run_window_function=difference_between_rasters,
    verbose=False,
)
diff_calc.create(nodata=-9999, overwrite=False)
diff_calc.verbose = True  # Dont see raster creation but do show progress.
diff_calc.run()


# %%

# narekenen enkele pixel

row = 4902
col = 7072
lu_id = 6
depth = 0.025765955448150635

dmg_lizard = 391.222412109375
dmg_local = 391.2530212402344


depth = np.round(depth, 2)

lu = self.dmg_table_landuse[lu_id]
damage_direct = (
    lu.direct * lu.gamma_inundatieduur_interp * lu.gamma_maand[self.indices["maand"]] * lu.direct_eenheid_factor
)


damage_indirect = lu.indirect * lu.gamma_herstelperiode[self.indices["herstelperiode"]] * lu.indirect_eenheid_factor


# gamma_inundatiediepte
lu.gamma_inundatiediepte

xp = np.array(self.dmg_table_general["inundatiediepte"])

index_boven = np.searchsorted(xp, depth, side="left")
index_onder = index_boven.copy() - 1

gamma_inundatiediepte = (
    np.divide(
        (lu.gamma_inundatiediepte[index_boven] - lu.gamma_inundatiediepte[index_onder]),
        (xp[index_boven] - xp[index_onder]),
    )
) * (depth - xp[index_onder]) + lu.gamma_inundatiediepte[index_onder]

pixel_factor = self.depth_raster.pixelarea
(damage_direct * gamma_inundatiediepte + damage_indirect) * pixel_factor


# %%
YES = True
# damage_local_lizard_settings
STATISTICS_APPROXIMATE = YES
STATISTICS_MAXIMUM = 78.963439941406
STATISTICS_MEAN = 0.010128559109839
STATISTICS_MINIMUM = 3.1851233456281e-11
STATISTICS_STDDEV = 0.31189795248779
STATISTICS_VALID_PERCENT = 14.68


# damage_local_lizard_settings_noround
STATISTICS_APPROXIMATE = YES
STATISTICS_MAXIMUM = 78.897483825684
STATISTICS_MEAN = 0.010578963108369
STATISTICS_MINIMUM = 2.278506769926e-07
STATISTICS_STDDEV = 0.32678209648164
STATISTICS_VALID_PERCENT = 14.31


# total_damage_res0_5m
STATISTICS_APPROXIMATE = YES
STATISTICS_MAXIMUM = 78.895858764648
STATISTICS_MEAN = 0.0015141723263587
STATISTICS_MINIMUM = 0
STATISTICS_STDDEV = 0.12367166627342
STATISTICS_VALID_PERCENT = 100


# %%
def get_raster_total(r: hrt.Raster):
    total = 0
    for window, block in r:
        block[block == r.nodata] = 0
        total += np.nansum(block)
    return total


damage_raster = {}
damage_raster["local"] = threedi_result.full_path("damage_local_lizard_settings.tif")
damage_raster["local_noround"] = threedi_result.full_path("damage_local_lizard_settings_noround.tif")
damage_raster["lizard"] = threedi_result.full_path("total_damage_res0_5m.tif")

total = {}
for key in damage_raster:
    total[key] = get_raster_total(damage_raster[key])
    print(f"€{round(total[key],2)} - {key}")
