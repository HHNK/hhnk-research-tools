# %%
"""
Klaarzetten van de landgebruiksrasters.
"""

# import threedi_scenario_downloader.downloader as dl
import importlib

import hhnk_threedi_tools.external.downloader as dl

importlib.reload(dl)
import logging

dl.set_logging_level(logging.DEBUG)
dl.LIZARD_URL = "https://hhnk.lizard.net/api/v4/"
import os
from pathlib import Path

import numpy as np
from osgeo import gdal

# from functions.create_folders_dict import create_folders_dict_wss
# import functions.wsa_tools as wsa #general tools used across all scripts
import hhnk_research_tools as hrt

# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()


api_keys_path = rf"{os.getenv('APPDATA')}\3Di\QGIS3\profiles\default\python\plugins\hhnk_threedi_plugin\api_key.txt"
api_keys = hrt.read_api_file(api_keys_path)
dl.set_api_key(api_key=api_keys["lizard"])


# uuid of raster
uuids = {}
# uuids['landgebruik'] = 'b73189fc-058d-4351-9b20-2538248fae4f'
# uuids['landgebruik2019_3di'] = 'a80a6d31-f539-4037-9765-4fb880654424' #Deze wordt gebruikt in de 3Di nabewerking schadeberekening. Dat is hier terug te vinden; https://api.3di.live/v3.0/simulations/113368/results/post_processing/lizard/overview/
# uuids['landuse2018'] = '7a93a6de-680d-41ca-8e7a-4b356baa18c7'
# uuids['landuse2019'] = '169853de-8319-4a7f-b076-f2ee0ae5b8c4'
# uuids['landuse2020'] = 'afedc34b-427e-4fd1-ac19-ba83032e2407'
# uuids['landuse2021'] = '59064cc0-a0a2-4509-b790-e112cb718100'
uuids["hittestress"] = "aac48ef4-247d-4dca-a878-35b2b6a8b460"


# uuids['ahn3'] = '36588275-f3e3-4120-8c1e-602f7ae85386' #Deze is niet meer te vinden maar wel gebruikt in de schadeberekening van 2020.
# uuids['ahn3'] = '81ef31b1-a65c-4071-acce-c77a172ba8a9'

output_dir = {}
output_dir["landgebruik2019_3di"] = r"E:\01.basisgegevens\hhnk_schadeschatter\01_data\landuse2019_3di_tiles"
output_dir["landuse2019"] = r"E:\01.basisgegevens\hhnk_schadeschatter\01_data\landuse2019_tiles"
output_dir["landuse2020"] = r"E:\01.basisgegevens\hhnk_schadeschatter\01_data\landuse2020_tiles"
output_dir["landuse2021"] = r"E:\01.basisgegevens\hhnk_schadeschatter\01_data\landuse2021_tiles"
output_dir["hittestress"] = r"E:\01.basisgegevens\rasters\lizard_hittestress_v2"

# resolution of output raster.
RESOLUTION = 0.5  # m
CHUNKSIZE = 4096

# outputloc
outDirs = {}
scenario_uuid = []
resolution = RESOLUTION
bbox = []
pathname = []
raster_code = []
is_threedi_scenario = []
cont = False

# Boundaries for download
bounds_hhnk = {"west": 100500.0, "south": 486900.0, "east": 150150.0, "north": 577550.0}


xres = (bounds_hhnk["east"] - bounds_hhnk["west"]) / RESOLUTION
yres = (bounds_hhnk["north"] - bounds_hhnk["south"]) / RESOLUTION

for key, uuid in uuids.items():
    # Check if tiles are on system.
    for i in range(0, int(np.ceil(xres / CHUNKSIZE))):
        for j in range(0, int(np.ceil(yres / CHUNKSIZE))):
            # Split raster into smaller chuncks.
            b = {
                "west": bounds_hhnk["west"] + i * CHUNKSIZE * RESOLUTION,
                "south": bounds_hhnk["south"] + j * CHUNKSIZE * RESOLUTION,
                "east": min(bounds_hhnk["west"] + (i + 1) * CHUNKSIZE * RESOLUTION, bounds_hhnk["east"]),
                "north": min(bounds_hhnk["south"] + (j + 1) * CHUNKSIZE * RESOLUTION, bounds_hhnk["north"]),
            }

            bbox_v4 = f"{b['west']},{b['south']},{b['east']},{b['north']}"
            output_file = os.path.join(output_dir[key], f"{key}_x{str(i).zfill(2)}_y{str(j).zfill(2)}.tif")
            # Opsplitsen in kleinere gebieden
            if not os.path.exists(output_file):
                bbox.append(bbox_v4)
                scenario_uuid.append(uuid)
                pathname.append(output_file)
                raster_code.append("")
                is_threedi_scenario.append(False)
            else:
                print("{} already on system".format(output_file.split(os.sep)[-1]))

    if not os.path.exists(output_dir[key]):
        os.mkdir(output_dir[key])
    # #Create task and download rasters
    if len(pathname) != 0:
        # cont='y'
        cont = input(f"Start downloading {len(pathname)} tiles. \nContinue? [y/n]")


# %% Download

blocks = [slice(i, min(i + 25, len(pathname))) for i in range(0, len(pathname), 25)]

for block in blocks:
    print(block)
    for key, uuid in uuids.items():
        if cont == "y":
            print("Start download")
            dl.download_raster(
                scenario=scenario_uuid[block],
                raster_code=raster_code[block],
                projection="EPSG:28992",
                resolution=resolution,
                bbox=bbox[block],
                pathname=pathname[block],
                is_threedi_scenario=is_threedi_scenario[block],
                export_task_csv=Path(pathname[block][0]).with_suffix(".txt"),
            )
# %% Create VRT
for key, uuid in uuids.items():
    hrt.build_vrt(output_dir[key], bandlist=None, overwrite=True)

# %% omzetten van nodata values van inf naar -9999
import geopandas as gpd

output_dir["hittestress"] = r"E:\01.basisgegevens\rasters\lizard_hittestress\combined_rasters.vrt"
out_dir_v2 = r"E:\01.basisgegevens\rasters\lizard_hittestress_v2"
r = hrt.Raster(r"E:\01.basisgegevens\rasters\lizard_hittestress\combined_rasters.vrt")

r.min_block_size = 2**14
blocks = r.generate_blocks_geometry()

for index, block_row in blocks.iterrows():
    out_block_r = hrt.Raster(os.path.join(out_dir_v2, f"hittestress_{block_row.ix}_{block_row.iy}.tif"))

    if not out_block_r.exists():
        window = block_row["window_readarray"]
        block = r._read_array(window=window)
        block[block > 1.79769313e100] = -9999

        hrt.save_raster_array_to_tiff(
            output_file=out_block_r,
            raster_array=block,
            nodata=-9999,
            metadata=hrt.RasterMetadataV2.from_gdf(gpd.GeoDataFrame(block_row).T, res=0.5),
            create_options=None,
            num_bands=1,
            overwrite=False,
        )
