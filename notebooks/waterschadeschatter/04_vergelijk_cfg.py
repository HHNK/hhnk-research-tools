# %%
"""
Vergelijking tussen de lizard, stowa en hhnk wss cfg.
Hhnk wss is de cfg zoals gebruikt in 2020 bij opnieuw afleiden van schadecurven.

"""

import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hhnk_research_tools.waterschadeschatter.wss_calculations as wss_calculations
import hhnk_research_tools.waterschadeschatter.wss_loading as wss_loading
import hhnk_research_tools.waterschadeschatter.wss_main as wss_main

importlib.reload(wss_main)
importlib.reload(wss_loading)
importlib.reload(wss_calculations)

from osgeo import gdal

gdal.UseExceptions()

# %%

cfg_file = {}
cfg_file["stowa"] = r"../01_data/cfg/cfg_stowa_standaard.cfg"
cfg_file["hhnk"] = r"../01_data/cfg/cfg_hhnk_2020.cfg"
cfg_file["lizard"] = r"../01_data/cfg/cfg_lizard.cfg"
output_folder = r"../02_output/vergelijking_cfg/"

dmg_table_landuse = {}
dmg_table_general = {}


class Cfg:
    def __init__(self, srs, wss_settings):
        self.srs = srs
        self.wss_settings = wss_settings
        self.dmg_table_landuse, self.dmg_table_general = wss_loading.read_dmg_table_config(
            wss_settings=self.wss_settings
        )

    def info(self):
        print(f"""
            srs - {srs}
            landuse - {dmg_table_landuse[srs][i].omschrijving}
            xp_new = {xp_new}
            fp - {fp}
            xp_old - {xp_old}
            gamma_diepte_new - {gamma_diepte}
            """)


srs_types = ["stowa", "hhnk", "lizard"]
cfg = {}
# Load config files
for srs in srs_types:
    wss_settings = {
        "inundation_period": 24,  # uren
        "herstelperiode": "10 dagen",
        "maand": "sep",
        "cfg_file": cfg_file[srs],
        "dmg_type": "gem",
    }

    cfg[srs] = Cfg(srs, wss_settings)


# Filter landuse ids that dont have 0 direct_dmg
lu_ids = [
    i
    for i in cfg[srs].dmg_table_landuse
    if (cfg[srs_types[0]].dmg_table_landuse[i].direct != 0) or (cfg[srs_types[1]].dmg_table_landuse[i].direct != 0)
]

# Initialize subplots
tot = len(lu_ids)  # Number of subplots
cols = 5  # Number of columns
rows = tot // cols
if tot % cols != 0:
    rows += 1
position = range(1, tot + 1)

# lu_ids = lu_ids[0:10] #faster for testing

# %%

linestyle = {}
linestyle["stowa"] = "solid"
linestyle["hhnk"] = "dashdot"
linestyle["lizard"] = "dashed"


class LuFig:
    """Landuse figure with custom x and y calculation."""

    def __init__(self, output_file, xlabel, ylabel, get_x_y):
        self.get_x_y = get_x_y

        # self.plot(output_file, xlabel, ylabel)

    def get_max_dmg(self):
        self.max_dmg = {}
        self.max_dmg["lu"] = {}
        for srs in srs_types:
            self.max_dmg[srs] = {}
        for i in lu_ids:
            self.max_dmg["lu"][i] = cfg[srs].dmg_table_landuse[i].omschrijving
            for srs in srs_types:
                lu = cfg[srs].dmg_table_landuse[i]
                x, y = self.get_x_y(cfg, srs, lu)
                self.max_dmg[srs][i] = y[-1]

    def plot(self, output_file, xlabel, ylabel):
        fig = plt.figure(figsize=[20, 60])

        pos = 1
        for i in lu_ids:
            ax = fig.add_subplot(rows, cols, pos)

            if i == lu_ids[0]:
                ax1 = ax

            for srs in srs_types:
                lu = cfg[srs].dmg_table_landuse[i]
                x, y = self.get_x_y(cfg, srs, lu)
                ax.plot(x, y, linestyle=linestyle[srs])

            ax.set_title(f"{i} - {lu.omschrijving}")
            ax.grid()

            pos += 1

        # Ax en legend voor eerste figuur. rest is hetzelfde.
        ax1.legend(srs_types)
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel(xlabel)

        # plt.suptitle('Gamma_inundatiediepte')
        fig.tight_layout()
        plt.show()
        fig.savefig(output_file)


# %% Directe schade
def get_x_y_schade_direct(cfg, srs, lu):
    fp = lu.gamma_inundatiediepte
    xp_new = [-0.01, 0.01, 0.05, 0.15, 0.3, 0.5, 1.5, 2.5]
    xp_old = cfg[srs].dmg_table_general["inundatiediepte"]
    y = [np.round(j * lu.direct * lu.direct_eenheid_factor, 2) for j in np.interp(x=xp_new, xp=xp_old, fp=fp)]
    return xp_new, y


lufig = LuFig(
    output_file=os.path.join(output_folder, "vergelijk_schade_direct.png"),
    ylabel="Schade [€ /m2/dag]",
    xlabel="Inundatiediepte [m]",
    get_x_y=get_x_y_schade_direct,
)

lufig.get_max_dmg()
pd.DataFrame(lufig.max_dmg).to_csv(os.path.join(output_folder, "dmg_direct_250cm.csv"))


# %% Indirecte schade
def get_x_y_schade_indirect(cfg, srs, lu):
    x = cfg[srs].dmg_table_general["herstelperiode_int"]
    y = [np.round(lu.indirect * j * lu.indirect_eenheid_factor, 2) for j in lu.gamma_herstelperiode]
    return x, y


LuFig(
    output_file=os.path.join(output_folder, "vergelijk_schade_indirect.png"),
    ylabel="Schade [€ /m2/dag]",
    xlabel="Inundatieduur [uren]",
    get_x_y=get_x_y_schade_indirect,
)


# %% Inundatieduur
def get_x_y_inundatieduur(cfg, srs, lu):
    x = cfg[srs].dmg_table_general["inundatieduur"]
    y = lu.gamma_inundatieduur
    return x, y


LuFig(
    output_file=os.path.join(output_folder, "vergelijk_gamma_inundatieduur.png"),
    xlabel="Inundatieduur [uren]",
    ylabel="gamma_inundatieduur",
    get_x_y=get_x_y_inundatieduur,
)


# %% Gamma maand
def get_x_y_gamma_maand(cfg, srs, lu):
    x = cfg[srs].dmg_table_general["maand"]
    y = lu.gamma_maand
    return x, y


LuFig(
    output_file=os.path.join(output_folder, "vergelijk_gamma_maand.png"),
    xlabel="maand",
    ylabel="gamma_maand",
    get_x_y=get_x_y_gamma_maand,
)


# %%
