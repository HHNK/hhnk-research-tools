# %%
#
#
#
"""Compare the local WSS to the lizard WSS.
# Use: zwartedijkspolder_sted_DPRA160"""

import os

# from functions.create_folders_dict import create_folders_dict_wss
# import functions.wsa_tools as wsa #general tools used across all scripts
import sys

import geopandas as gpd
import hhnk_threedi_tools as htt
import numpy as np
import threedi_scenario_downloader.downloader as dl
from osgeo import gdal

# sys.path.remove( 'C:\\Users\\wvangerwen\\AppData\\Roaming\\Python\\Python37\\site-packages',)
import hhnk_research_tools as hrt

# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()


api_keys_path = rf"{os.getenv('APPDATA')}\3Di\QGIS3\profiles\default\python\plugins\hhnk_threedi_plugin\api_key.txt"
api_keys = hrt.read_api_file(api_keys_path)
dl.set_api_key(api_key=api_keys["lizard"])

test_name = "test_zwartedijkspolder"
test_name = "test_bwn_test"
test_path = rf"E:\github\wvangerwen\hhnk_schadeschatter\01_data\{test_name}"
# https://hhnk.lizard.net/api/v4/scenarios/95534846-af1f-47d2-a474-c12137ab08be/
# https://hhnk.lizard.net/api/v4/rasters/6cd63807-f635-4e56-9ae0-56dbf592ce7e/


class Test_Folder(htt.core.folders.Folder):
    def __init__(self, base):
        super().__init__(base)

        self.add_file("depth_lizard", "depth_dmg.tif")
        self.add_file("dmg_lizard", "dmg_lizard.tif")
        self.add_file("dmg_local", "dmg_local.tif")
        self.add_file("cfg_lizard", "cfg_lizard.cfg")


test_folder = Test_Folder(test_path)


# %%


from shapely.geometry import box

# scenario_uuid = '95534846-af1f-47d2-a474-c12137ab08be' #zwartedijkspolder_sted_DPRA160
scenario_uuid = "f742fc33-cc02-41de-950c-502f9ac7c0a5"  # bwn_test #5 1d2d_ggg

dem = hrt.Raster(
    r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC16015 Wateropgave 2.0\07. Poldermodellen\99.Zwartedijkspolder\02. Sqlite modeldatabase\rasters\dem_alkmaardermeer.tif"
)

# dmge-depth
# total-damage
r = dl.get_raster(scenario_uuid=scenario_uuid, raster_code="dmge-depth")

bounds = {}
bounds["minx"] = r["spatial_bounds"]["west"]
bounds["miny"] = r["spatial_bounds"]["south"]
bounds["maxx"] = r["spatial_bounds"]["east"]
bounds["maxy"] = r["spatial_bounds"]["north"]

a = gpd.GeoDataFrame(geometry=[box(**bounds)], crs="EPSG:4326")
bounds = a.to_crs("EPSG:28992").bounds
bounds = dict(bounds.iloc[0])

bounds["minx"] = np.floor(bounds["minx"] * 2) / 2
bounds["miny"] = np.floor(bounds["miny"] * 2) / 2
bounds["maxx"] = np.ceil(bounds["maxx"] * 2) / 2
bounds["maxy"] = np.ceil(bounds["maxy"] * 2) / 2

bounds_new = {}
bounds_new["west"] = bounds["minx"]
bounds_new["south"] = bounds["miny"]
bounds_new["east"] = bounds["maxx"]
bounds_new["north"] = bounds["maxy"]


# dl.download_raster(scenario=[scenario_uuid, scenario_uuid],
#     raster_code=['dmge-depth', 'total-damage'],
#     target_srs=None,
#     resolution=0.5,
#     bounds=bounds_new,
#     bounds_srs= 'EPSG:28992',
#     time=None,
#     pathname=[test_folder.depth_lizard.path, test_folder.dmg_lizard.path],
#     is_threedi_scenario=True,  # For lizard rasters that are not a Threedi result.
#     export_task_csv='dl.csv',)

# %%


# %% Round depth raster
# arr, nodata,metadata = test_folder.depth_lizard.load()
# mask= arr==nodata
# arr= np.round(arr,3)
# arr[mask]=nodata

# hrt.save_raster_array_to_tiff(
#     output_file=test_folder.depth_lizard_round.path,
#     raster_array=arr,
#     nodata=nodata,
#     metadata=metadata)

# %%
import sys

sys.path.append("../..")
import hhnk_schadeschatter.functions.wss_main as wss_main

# Input
cfg_file = test_folder.cfg_lizard.path
landuse_file = r"E:\github\wvangerwen\hhnk_schadeschatter\01_data\landuse2020_tiles\combined_rasters.vrt"
landuse_file = r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\04_kop\03_downloads\landuse2019_tiles\kop_landuse2019.vrt"
depth_file = test_folder.depth_lizard.path
output_file = test_folder.dmg_local.path

# %%
wss_settings = {
    "inundation_period": 48,  # uren
    "herstelperiode": "10 dagen",
    "maand": "sep",
    "cfg_file": cfg_file,
    "dmg_type": "gem",
}


# Calculation
wss_local = wss_main.Waterschadeschatter(depth_file=depth_file, landuse_file=landuse_file, wss_settings=wss_settings)

wss_local.run(
    output_raster=hrt.Raster(output_file),
    calculation_type="sum",
    verbose=True,
    overwrite=False,
)


# %%
import pandas as pd

ws_range = range(100, 110, 10)


class PgbFolder(htt.core.folders.Folder):
    def __init__(self, base):
        super().__init__(base)

        for ws in ws_range:
            self.add_file(objectname=f"ws_{ws}", filename=f"ws_{ws}.tif")
            self.add_file(objectname=f"depth_{ws}", filename=f"depth_{ws}.tif")
            self.add_file(objectname=f"dmg_{ws}", filename=f"dmg_{ws}.tif")


output_dir = r"E:\github\wvangerwen\hhnk_schadeschatter\01_data\drainage_areas"
pgb_folder = PgbFolder(base=os.path.join(output_dir, "pgb_4770_04751-04"))
landuse_file = r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\03_downloads\landuse2019_tiles\ad_landuse2019.vrt"
dmg_path = r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\06_schade\tif\schade_ad_100.tif"


ws = 100

window_local = [5144, 2362, 1, 1]

indices = {"herstelperiode": 5, "maand": 8}

paths = {}
# paths['dem'] = r"\\srv57d1\geo_info\02_Werkplaatsen\06_HYD\Projecten\HKC21002 Schadecurven\07. SchadecurvenPy\03_gebieden\06_ad\03_downloads\ahn3_tiles\combined_rasters.vrt"
paths["landuse"] = landuse_file
paths["dmg_wss"] = dmg_path
paths["dmg_local"] = getattr(pgb_folder, f"dmg_{ws}").path
paths["depth_lizard"] = getattr(pgb_folder, f"depth_{ws}").path


windows = {}
# windows['dem'] = window_old
windows["landuse"] = window_local.copy()
windows["dmg_wss"] = window_local.copy()
windows["dmg_local"] = window_local.copy()
windows["depth_lizard"] = window_local.copy()


rasters = {}
for r in paths:
    rasters[r] = hrt.Raster(paths[r])

# change landuse window
for r in ["landuse", "dmg_wss"]:
    dx_min, dy_min = hrt.dx_dy_between_rasters(meta_big=rasters[r].metadata, meta_small=rasters["dmg_local"].metadata)

    windows[r][0] += dx_min
    windows[r][1] += dy_min

values = {}
for r in rasters:
    values[r] = rasters[r]._read_array(window=windows[r])[0][0]


# lu.direct_gem * lu.gamma_inundatieduur_interp * lu.gamma_maand[indices['maand']] * lu.direct_eenheid_factor
r = pd.Series(values)
print(r)

# Bereken schade
x = values["depth_lizard"]

xp = [-0.01, 0.01, 0.05, 0.15, 0.3, 2.5]
fp = [0.0, 0.0, 1.0, 2.0, 3.0, 50.0]
direct_gem = 800 / 10000
gamma_duur = 1
indirect_gem = 0
gamma_herstelperiode = 10


# x= np.round(x,2)

# xp=[-0.01, 0.01, 0.05, 0.15, 0.3, 0.5, 1.5]
# fp=[0, 0.02, 0.08, 0.16, 0.16, 0.76, 1] #gammainundatiediepte
# direct_gem = 1587.16
# gamma_duur = 1
# indirect_gem = 10
# gamma_herstelperiode = 1


gamma_diepte = np.interp(x=x, xp=xp, fp=fp)

gamma_diepte = 67.30454545454546
pixel_factor = 0.25
gamma_maand = 1


(
    direct_gem * gamma_duur * gamma_maand * gamma_diepte * pixel_factor
    + gamma_herstelperiode * indirect_gem * pixel_factor
)


# %%
import numpy as np

x = 4
index_boven = np.searchsorted(xp, [[4]], side="left")

# index_boven[index_boven==len(xp)]-=1

index_onder = index_boven.copy() - 1
index_onder[index_onder == -1] = 0

index_onder = 5
index_boven = 5
np.seterr(invalid="ignore")
y1 = 50
y2 = 50

(np.divide((y2 - y1), (xp[index_boven] - xp[index_onder]))) * (4 - xp[index_onder]) + y1


# %%
arr = {}
for rastr in ["dmg_local", "dmg_wss"]:
    arr[rastr] = rasters[rastr].get_array()
    arr[rastr][arr[rastr] == rasters[rastr].nodata] = 0

    # print(f"{rastr}: {np.nanpercentile(arr[rastr], q=95)}")
    print(f"{rastr}: {np.nanmax(arr[rastr])}")

# %%

a = arr["dmg_local"]
np.where(a > 149)
# np.nansum(arr['dmg_local']- arr['dmg_lizard'])
