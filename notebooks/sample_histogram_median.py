# %%
# 20241227 WVG: Dont know where this is used.
import numpy as np

import hhnk_research_tools as hrt

dem = hrt.Raster(r"..\tests_hrt\data\depth_test.tif")
blocks_df = hrt.RasterChunks.from_raster(dem).to_gdf()
hist = {}

for idx, block_row in blocks_df.iterrows():
    block_arr = dem._read_array(window=block_row["window_readarray"])
    # block_arr[nodatamask] = 0

    val, count = np.unique(block_arr.round(2), return_counts=True)
    for v, c in zip(val, count):
        if v not in hist.keys():
            hist[v] = c
        else:
            hist[v] += c
    # else:
    #     hist = False

hrt.hist_stats(histogram=hist, stat_type="median", ignore_keys=[-9999])
