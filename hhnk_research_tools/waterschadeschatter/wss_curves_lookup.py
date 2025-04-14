# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:29:47 2024

@author: kerklaac5395
"""

import functools
from collections import namedtuple

import numpy as np
from tqdm import tqdm

from hhnk_research_tools.variables import DEFAULT_NODATA_VALUES
from hhnk_research_tools.waterschadeschatter import wss_calculations, wss_loading
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import WSSTimelog, write_dict

DMG_NODATA = 0  # let op staat dubbel, ook in wss_main.
NODATA_UINT16 = 65535
NAME = "WSS LookupTable"


class DummyCaller:
    def __init__(self, nodata=-9999):
        self.depth_raster = namedtuple("Raster", "nodata")(nodata)
        self.gamma_inundatiediepte = 0


class WaterSchadeSchatterLookUp:
    """
    Is a lookup table which retrieves from a combinatie of depth and landuse,

    bases on a configuration.

    Use as follows:

        wsslookup = WaterSchadeSchatterLookUp(wss_settings)
        damage = wsslookup[depth, lu]

        of
        table = wsslookup.output


    Params:
        wss_settings:str, Pad naar een config file van de waterschadeschatter.
        depth_steps:list, Lijst met peilstijgingen
        pixel_factor:float, m2 per pixel
        nodata: int, nodata waarde
        quiet: bool, Wel of geen console output.
    """

    def __init__(
        self,
        wss_settings: str,
        depth_steps=[0.1, 0.2, 0.3],
        pixel_factor=0.5 * 0.5,
        nodata=DEFAULT_NODATA_VALUES["float32"],
        quiet=False,
    ):
        self.settings = wss_settings

        self.dmg_table_landuse, self.dmg_table_general = wss_loading.read_dmg_table_config(wss_settings)
        self.indices = {}
        self.indices["herstelperiode"] = self.dmg_table_general["herstelperiode"].index(wss_settings["herstelperiode"])
        self.indices["maand"] = self.dmg_table_general["maand"].index(wss_settings["maand"])
        self.pixel_factor = pixel_factor
        self.caller = DummyCaller(nodata)
        self.depth_steps = depth_steps
        self.time = WSSTimelog(NAME, quiet)
        self.quiet = quiet
        self.output = {}
        self.run()

    # @functools.cached_property
    # def mapping_arrays(self):

    #     data = {}
    #     for ds, luses in self.output.items():

    #         k = np.array(list(luses.keys()))
    #         v = np.array(list(luses.values()))

    #         mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)
    #         mapping_ar[k] = v
    #         data[ds] = mapping_ar
    #     return data

    def __getitem__(self, lu_depth):
        return self.output[lu_depth[0]][lu_depth[1]]

    def run(self):
        self.time._message("Start generating table")

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

            self.output[depth][DEFAULT_NODATA_VALUES["uint16"]] = 0
            self.output[depth][255] = 0  # not in cfg

        self.time._message("Ended generating table")

    def write_dict(self, path):
        write_dict(self.output, path)
