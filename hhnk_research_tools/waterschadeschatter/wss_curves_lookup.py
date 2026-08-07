# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:29:47 2024

@author: kerklaac5395
"""

from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

import hhnk_research_tools.logging as logging
from hhnk_research_tools.variables import DEFAULT_NODATA_VALUES
from hhnk_research_tools.waterschadeschatter import wss_calculations, wss_loading
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import WSSTimelog, write_dict

# Logger
logger = logging.get_logger(__name__)

# Globals
NAME = "WSS LookupTable"
LU_LOOKUP_FACTOR = 100


class _DummyCaller:
    """A dummy class to mock waterschadeschatter."""

    def __init__(self, nodata: Union[int, float]) -> None:
        self.depth_raster = namedtuple("Raster", "nodata")(nodata)
        self.gamma_inundatiediepte = 0


class WaterSchadeSchatterLookUp:
    """
    Is a lookup table which retrieves from a combination of depth and landuse,

    bases on a configuration.

    Example
    -------
    >>> wsslookup = WaterSchadeSchatterLookUp(wss_settings)
    >>> damage = wsslookup[depth, lu]
    >>> table = wsslookup.output

    Flattening is used unnest the dictionary.
    This makes sure that vectorized operations using pandas can be used in wss_curves_areas:AreaDamageCurveMethods:run.
    For example, the key for landuse 2 with a depth step of 0.1 is 200.1, with a LU_LOOKUP_FACTOR of 100.
    The dict should then be {200.1: damage_value}.

    Parameters
    ----------
    wss_settings : dict
        Dictionary met de instellingen van de waterschadeschatter.
    depth_steps : list[float], default is [0.1, 0.2, 0.3]
        Lijst met peilstijgingen
    pixel_factor : float, default is 0.25
        m2 per pixel
    nodata : Union[int, float]
        nodata waarde
    """

    def __init__(
        self,
        wss_settings: Dict[str, Any],
        depth_steps: List[float] = [0.1, 0.2, 0.3],
        pixel_factor: float = 0.5 * 0.5,
        nodata: Union[int, float] = DEFAULT_NODATA_VALUES["float32"],
    ) -> None:
        self.settings = wss_settings

        self.dmg_table_landuse, self.dmg_table_general = wss_loading.read_dmg_table_config(wss_settings)
        self.indices = {
            "herstelperiode": self.dmg_table_general["herstelperiode"].index(wss_settings["herstelperiode"]),
            "maand": self.dmg_table_general["maand"].index(wss_settings["maand"]),
        }

        self.pixel_factor = pixel_factor
        self.caller = _DummyCaller(nodata=nodata)
        self.depth_steps = depth_steps
        self.time = WSSTimelog(name=NAME)
        self.output = {}

    def __getitem__(self, lu_depth: Tuple[float, int]) -> float:
        """Get damage value for a specific landuse-depth combination."""
        return self.output[lu_depth[0]][lu_depth[1]]

    def run(self, flatten: bool = True) -> None:
        """Generate lookup table for all landuse-depth combinations.

        Parameters
        ----------
        Flatten : bool
            Unnests the dictionary and replaces it by a code (lu*LU_LOOKUP_FACTOR)+depth_step.
        """
        self.time.log("Start generating table")

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

        self.time.log("Ended generating table")

        if flatten:
            self.time.log("Flatten lookup to increase speed using (lu*LU_LOOKUP_FACTOR)+depth_step.")
            flattened = {}
            for depth_step, lu_lookup in self.output.items():
                for lu, damage in lu_lookup.items():
                    flattened[(lu * LU_LOOKUP_FACTOR) + depth_step] = damage
            self.output = flattened

    def write_dict(self, path: Union[str, Path]) -> None:
        """Write lookup table to JSON file."""
        write_dict(self.output, path)
