# %%
"""
TODO

"""



# %%
import functools
import sys
sys.path.append('../..')
import hhnk_schadeschatter.local_settings as local_settings
if local_settings.DEBUG:
    sys.path.insert(0, local_settings.hhnk_threedi_tools_path)
    sys.path.insert(0, local_settings.hhnk_research_tools_path)

    import importlib
    import hhnk_threedi_tools as htt
    import hhnk_research_tools as hrt
    htt=importlib.reload(htt)
    hrt=importlib.reload(hrt)
    importlib.reload(htt.core.folders)


import importlib
import hhnk_research_tools as hrt

import geopandas as gpd
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import hhnk_schadeschatter.functions.wss_loading as wss_loading
import hhnk_schadeschatter.functions.wss_calculations as wss_calculations
import hhnk_schadeschatter.functions.wss_main as wss_main
importlib.reload(wss_main)
importlib.reload(wss_loading)
importlib.reload(wss_calculations)

from osgeo import gdal
gdal.UseExceptions()

from pathlib import Path


# hrt.build_vrt(r'E:\github\wvangerwen\hhnk_schadeschatter\01_data\landuse2020_tiles', bandlist=None, overwrite=True)


# %% create initial wl_raster.
# ws=0

# pgb_file = r'../01_data/texel_peilgebieden_raw.shp'


# output_ws = rf'../01_data/ws_{ws}.tif'

# gdf = gpd.read_file(pgb_file)
# gdf['streefpeil_ws'] = gdf['streefpeil']+ws/100


# meta=hrt.RasterMetadataV2.from_gdf(gdf, res=0.5)


# _ = hrt.gdf_to_raster(gdf=gdf, 
#             value_field='streefpeil_ws',
#             raster_out=output_ws,
#             nodata=-9999,
#             metadata=meta,
#             read_array=False,)


# %%
# Input
cfg_file = r'../01_data/schadetabel_hhnk_2020.cfg'
landuse_file = r'../01_data/landuse2019_tiles/waterland_landuse2019.vrt'

depth_file = r'../01_data/marken_rev23_max_depth_blok_GGG_T10.tif'
output_file = r'../01_data/marken_rev23_damage_blok_GGG_T10.tif'

wss_settings = {'inundation_period': 48, #uren
                'herstelperiode':'10 dagen',
                'maand':'sep',
                'cfg_file':cfg_file}


# %%
# Calculation
wss_local = wss_main.Waterschadeschatter(depth_file=depth_file, 
                            landuse_file=landuse_file, 
                            wss_settings=wss_settings)


wss_local.run(output_raster=hrt.Raster(output_file), 
    calculation_type="sum", 
    verbose=True, 
    overwrite=False,
    )



# %%
lu_block=lu_block
depth_block=depth_block
indices=self.indices
dmg_table_landuse=self.dmg_table_landuse
dmg_table_general=self.dmg_table_general
pixel_factor=pixel_factor




# %%
depth = 0.0142824
#dmg_lizard = 0.00752583
# dmg_wss = 0.006045
# landuse=157 weidehooi
direct_gem = 1209/10000 #1209/ha
inderect_gem=0

gamma_inundatiediepte = 1
gamma_inundatieduur = 0.2
gamma_maand = 1
pixel_factor=0.25

direct_gem*gamma_inundatiediepte*gamma_inundatieduur*gamma_maand*pixel_factor











# %%
plt.imshow(depth_block)
plt.imshow(lu_block)







# %%

def landuse_not_in_cfg():
    """Check if any landuse is not in the cfg table"""
    no_landuse = np.isin(lu_block,[i for i in dmg_table_landuse.keys()])
    return np.unique(lu_block[~no_landuse])


# %%

# %load_ext line_profiler
# %lprun -f calculate_damage calculate_damage(lu_block, depth_block, indices, dmg_table_landuse, dmg_table_general)


#%%






# %%
def calculate_depth(dem, peilgebied_df, ws):
    """Bereken de inundatiediepte
    dem -> raster
    peilgebied_df -> shapefile met kolom 'streefpeil'. 
    ws -> int ws stijging tov streefpeil"""



def calculate_damage(depth_array, landuse_array, damage_table):
    """Bereken de schade
    depth -> raster
    landuse -> raster
    damage_table -> df of dict van df

    schade = max. directe schade · γdiepte · γduur · γseizoen + indirecte schade per dag · hersteltijd
    TODO indirecte schade nemen we nu niet mee.
    """

    for pixel in depth_array:
        'bereken schade per pixel.'