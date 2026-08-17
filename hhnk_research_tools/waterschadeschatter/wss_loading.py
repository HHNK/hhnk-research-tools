import configparser
import os
from pathlib import Path

import numpy as np


def read_dmg_table_config(wss_settings) -> dict:
    """Read damagetable. Returns dictionary with keys corresponding to the
    landuse index."""

    class Landuse_damagetable:
        """Damagetable for one land use"""

        def __init__(self, section):
            """Section is part of the configparser. [i.items(sect) for i in parser.sections()]"""
            self.section = dict(section)
            self.lizard_exception = False  # Track if we used exception

            for key in self.section:
                if "[" in self.section[key] or "." in self.section[key]:
                    setattr(self, key, eval(self.section[key]))
                else:
                    try:
                        setattr(self, key, int(self.section[key]))
                    except:
                        setattr(self, key, self.section[key])

            # Omrekenen van eenheden.
            if self.direct_eenheid == "/ha":
                self.direct_eenheid_factor = 1 / 10000
            elif self.direct_eenheid == "/m2":
                self.direct_eenheid_factor = 1
            else:
                raise Exception(f"unit: {self.direct_eenheid} unkown")
            if self.indirect_eenheid == "/m2/dag":
                self.indirect_eenheid_factor = 1
            elif self.indirect_eenheid == "/wegvak/dag":  # Uitzetten van de indirecte schaden wegen.
                self.indirect_eenheid_factor = 0
            else:
                raise Exception(f"unit: {self.indirect_eenheid} unkown")

            self.direct = getattr(self, f"direct_{wss_settings['dmg_type']}")
            self.indirect = getattr(self, f"indirect_{wss_settings['dmg_type']}")

            # Store gamma_inundatieduur curve for later interpolation
            # If inundation_period is a raster, interpolation happens per-pixel in calculate_damage
            # If it's a scalar value, we interpolate once here
            if "inundation_period" in wss_settings and wss_settings["inundation_period"] is not None:
                x = wss_settings["inundation_period"]
                xp = dmg_table_general["inundatieduur"]
                fp = self.gamma_inundatieduur
                self.gamma_inundatieduur_interp = np.interp(x=x, xp=xp, fp=fp)
            else:
                # No scalar inundation_period provided, will use raster
                self.gamma_inundatieduur_interp = None

            if "lizard" in Path(wss_settings["cfg_file"]).name:
                self.lizard_exception = True
                self.gamma_herstelperiode = [
                    i * j / 24 for i, j in zip(self.gamma_herstelperiode, dmg_table_general["herstelperiode_int"])
                ]

            # Pad gamma for interpolation
            self.gamma_inundatiediepte.insert(0, self.gamma_inundatiediepte[0])
            self.gamma_inundatiediepte.append(self.gamma_inundatiediepte[-1])

        def __repr__(self):
            variables = "." + " .".join(
                [i for i in dir(self) if not i.startswith("__") and not hasattr(getattr(self, i), "__call__")]
            )
            return f"""{self.omschrijving} - {variables}"""

    if wss_settings["cfg_file"] is None or not os.path.exists(wss_settings["cfg_file"]):
        raise Exception(f"could not find config file in: {wss_settings['cfg_file']}")
    parser = configparser.ConfigParser()
    parser.read(filenames=wss_settings["cfg_file"])

    dmg_table_landuse = {}

    # Read file
    # Create general table
    for sect in parser.sections():
        if sect == "algemeen":
            dmg_table_general = dict(parser.items(sect))
            for key in dmg_table_general:  # List in string to list.
                dmg_table_general[key] = eval(dmg_table_general[key])

            dmg_table_general["inundatiediepte"].insert(0, -np.inf)
            dmg_table_general["inundatiediepte"].append(np.inf)
            break

    # create table per landuse
    for sect in parser.sections():
        if sect != "algemeen":
            dmg_table_landuse[int(sect)] = Landuse_damagetable(parser.items(sect))

    # Fill missing values with dummy values. Dummy is defined as landuse=0
    for i in range(0, 255):  # Careful. Landuse value should be max 254
        if i not in dmg_table_landuse:
            dmg_table_landuse[i] = dmg_table_landuse[0]
    if dmg_table_landuse[i].lizard_exception:
        print("Uitzondering - Gamma herstelperiode voor lizard config is vermenigvuldigd met herstelperiode.")
    return dmg_table_landuse, dmg_table_general
