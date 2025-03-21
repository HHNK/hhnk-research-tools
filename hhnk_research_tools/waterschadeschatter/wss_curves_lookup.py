# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:29:47 2024

@author: kerklaac5395
"""
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import namedtuple

from hhnk_research_tools.waterschadeschatter import wss_calculations
from hhnk_research_tools.waterschadeschatter import wss_loading

from hhnk_research_tools.waterschadeschatter.wss_curves_utils import WSSTimelog, write_dict

DMG_NODATA = 0  # let op staat dubbel, ook in wss_main.
WORK_DIR = Path(
    r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External"
)
DATA_PATH = WORK_DIR / "data"
NODATA_UINT16 = 65535
NAME = "WSS LookupTable"


class DummyCaller:
    def __init__(self, nodata=-9999):
        self.depth_raster = namedtuple("Raster", "nodata")(-9999)
        self.gamma_inundatiediepte = 0


class WaterSchadeSchatterLookUp:
    """
    Is a lookup table which retrieves from a combinatie of depth and landuse,

    bases on a configuration.

    Use as follows:

        wsslookup = WaterSchadeSchatterLookUp(wss_settings)
        damage = wsslookup[depth, lu]

    """

    def __init__(
        self,
        wss_settings,
        depth_steps=[0.1, 0.2, 0.3],
        pixel_factor=0.5 * 0.5,
        nodata=-9999,
        quiet=False,
    ):
        self.settings = wss_settings

        self.dmg_table_landuse, self.dmg_table_general = (
            wss_loading.read_dmg_table_config(wss_settings)
        )
        self.indices = {}
        self.indices["herstelperiode"] = self.dmg_table_general["herstelperiode"].index(
            wss_settings["herstelperiode"]
        )
        self.indices["maand"] = self.dmg_table_general["maand"].index(
            wss_settings["maand"]
        )
        self.pixel_factor = pixel_factor
        self.caller = DummyCaller(nodata)
        self.depth_steps = depth_steps
        self.time = WSSTimelog(NAME, quiet)
        self.quiet = quiet
        self.generate_table()

    @property
    def mapping_arrays(self):
        if not hasattr(self, "_mapping_arrays"):
            self._mapping_arrays = {}
            for ds, luses in self.output.items():

                k = np.array(list(luses.keys()))
                v = np.array(list(luses.values()))

                mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)
                mapping_ar[k] = v
                self._mapping_arrays[ds] = mapping_ar

        return self._mapping_arrays

    def __getitem__(self, lu_depth):
        return self.output[lu_depth[0]][lu_depth[1]]

    def generate_table(self):
        self.time._message("Start generating table")
        self.output = {}
        for depth in tqdm(self.depth_steps, NAME):

            depth = round(depth, 2)
            self.output[depth] = {}

            for lu_num in self.dmg_table_landuse:
                damage = wss_calculations.calculate_damage(
                    caller=self.caller,  # wss_main.Waterschadeschatter,
                    lu_block=np.array([[lu_num], [lu_num]]),
                    depth_block=np.array([[depth], [depth]]),
                    indices=self.indices,
                    dmg_table_landuse=self.dmg_table_landuse,
                    dmg_table_general=self.dmg_table_general,
                    pixel_factor=self.pixel_factor,
                    calculation_type="sum",
                )
                self.output[depth][lu_num] = damage[0][0]

            self.output[depth][NODATA_UINT16] = 0
            self.output[depth][255] = 0  # not in cfg

        self.time._message("Ended generating table")
        
    def write(self, path):
        write_dict(self.output, path)
        

if __name__ == "__main__":
    lu_array = np.array([[2], [2]])
    depth_array = np.array([[0.1], [0.1]])

    cfg_file = DATA_PATH / "cfg" / "cfg_hhnk_2020.cfg"

    wss_settings = {
        "inundation_period": 48,  # uren
        "herstelperiode": "10 dagen",
        "maand": "sep",
        "cfg_file": cfg_file,
        "dmg_type": "min",
    }

    lookup = WaterSchadeSchatterLookUp(wss_settings, depth_steps=[0.1, 2.5])
    lookup.generate_table()

    # gem = 10