# %%
import os
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hhnk_research_tools.waterschadeschatter.wss_curves_utils import AreaDamageCurveFolders

# Globals
STANDAARD_FIGUUR = (10, 10)
LANDGEBRUIK_FIGUUR = (20, 10)
DPI = 300
MAX_PEILVERHOGING = 2.5


class Figuur:
    def __init__(self, figsize=STANDAARD_FIGUUR):
        self.figsize = figsize

    def create(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

    def plot(self, dataframe):
        self.ax.plot(dataframe)

    def xlabel(self, naam):
        self.ax.set(xlabel=naam)

    def ylabel(self, naam):
        self.ax.set(ylabel=naam)

    def title(self, naam):
        self.ax.set(title=naam)

    def xlim(self, xmin, xmax):
        self.ax.set(xlim=(xmin, xmax))

    def ylim(self, ymin, ymax):
        self.ax.set(ylim=(ymin, ymax))

    def xticks(self, ticks, *labels):
        self.ax.set_xticks(ticks, *labels)

    def yticks(self, ticks, labels=[]):
        self.ax.set_yticks(ticks, labels)

    def set_x_y_label(self):
        self.xlabel(self.xlabel_dsc)
        self.ylabel(self.ylabel_dsc)

    def set_x_y_lim(self):
        self.xlim(self.xlim_min, self.xlim_max)
        self.ylim(self.ylim_min, self.ylim_max)

    def write(self, path, dpi=DPI):
        plt.savefig(path, dpi=dpi)
        plt.close()

class CurveFiguur(Figuur):
    def __init__(self, damage_df):
        super().__init__()
        self.df_damages = damage_df
        self.xlabel_dsc = "Peilverhoging boven streefpeil (m)"
        self.ylabel_dsc = "Schadebedrag (Euro's)"

    def run(self, output_path, name, title, dpi=DPI):
        self.create()
        self.plot(self.df_damages)
        self.set_x_y_label()
        self.title(f"{title} voor {name}")
        self.write(output_path, dpi=dpi)

class BergingsCurveFiguur(Figuur):
    def __init__(self, path, feature):
        super().__init__()
        self.df_vol_level = pd.read_csv(path, index_col=0)

        self.xlabel_dsc = "Waterstand (m+NAP)"
        self.ylabel_dsc = "Volume (m3)"
        self.ylabel_mm = "Berging (mm)"
        self.feature = feature

    def volume2mm(self, y):
        area = self.feature.geometry.area
        return y / area * 1000

    def convert_V_to_mm(self, ax):
        y1, y2 = self.ax.get_ylim()
        self.ax_mm.set_ylim(self.volume2mm(y1), self.volume2mm(y2))
        self.ax_mm.figure.canvas.draw()

    def run(self, output_path, name, dpi=DPI):
        for col in self.df_vol_level.columns:
            valid_data = self.df_vol_level[col].dropna()
            self.create()
            self.ax_mm = self.ax.twinx()
            self.ax.callbacks.connect("ylim_changed", self.convert_V_to_mm)

            self.plot(valid_data)
            self.set_x_y_label()
            self.ax_mm.set_ylabel(self.ylabel_mm)
            self.title(f"bergingscurve voor {name}")
            self.write(output_path, dpi=dpi)


class PercentageFiguur(Figuur):
    def __init__(self, path, agg_dir):
        super().__init__(figsize=LANDGEBRUIK_FIGUUR)
        self.df_lu_opp_schade = pd.read_csv(path, index_col=0)
        self.df_lu_opp_schade.columns = pd.Series(self.df_lu_opp_schade.columns).str.split(" ").str[0]
        self.df_sum_damages = pd.read_csv(agg_dir.agg_damage.path, index_col=0)
        self.agg_dir = agg_dir

        self.xlabel_dsc = "Peilverhoging boven streefpeil (m)"
        self.ylabel_dsc = "Percentage t.o.v. totaal"
        self.xlim_min = 0.1
        self.xlim_max = MAX_PEILVERHOGING
        self.ylim_min = 0
        self.ylim_max = 1
        self.x_ticks_list = np.arange(0.1, MAX_PEILVERHOGING + 0.1, 0.1)
        self.y_ticks_list = np.arange(0, 1.1, 0.1)
        self.ylabels = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]

    def set_x_y_ticks(self):
        self.xticks(self.x_ticks_list)
        self.yticks(ticks=self.y_ticks_list, labels=self.ylabels)

    def handles_legend(self, lu_omzetting):
        nieuwe_klasses = np.array(lu_omzetting["nieuwe_klasse"].unique())
        self.color_dict = {}
        colors = []
        self.handels = []
        self.labels = []
        for klasse in nieuwe_klasses:
            label = lu_omzetting.where(lu_omzetting["nieuwe_klasse"] == klasse)["beschrijving"].dropna().unique()[0]
            color = lu_omzetting.where(lu_omzetting["nieuwe_klasse"] == klasse)["kleur"].dropna().unique()[0]
            self.labels.append(label)
            colors.append(color)
        for i in range(0, len(colors)):
            handle = mpatches.Patch(color=colors[i])
            self.handels.append(handle)
            self.color_dict[str(nieuwe_klasses[i])] = colors[i]

    def lu_verdeling_peilgebied(self, id):
        self.df_peilgebied = self.df_lu_opp_schade.loc[self.df_lu_opp_schade["fid"] == id].dropna(axis=1)
        self.df_peilgebied = self.df_peilgebied.drop("fid", axis=1)
        self.df_peilgebied_perc = self.df_peilgebied.divide(self.df_peilgebied.sum(axis=1), axis=0)
        self.df_peilgebied_perc.columns = [str(int(x)) for x in self.df_peilgebied_perc]

        self.lu = []
        self.lu_ids = []
        for col in self.df_peilgebied_perc.columns:
            self.lu.append(self.df_peilgebied_perc[col])
            self.lu_ids.append(col)

    # def sum_damages(self):
    #     self.df_sum_damages = self.df_lu_opp_schade.copy()
    #     self.df_sum_damages['totaal'] = self.df_sum_damages.drop('fid', axis=1).sum(axis=1)

    # .index.where(self.df_lu_opp_schade['fid'] == id).dropna(), self.df_sum_damages['totaal'].where(self.df_sum_damages['fid'] == id).dropna()

    def plot_schadecurve_totaal(self, id):
        self.ax2 = self.ax.twinx()
        self.ax2.plot(self.df_sum_damages, color="black", linewidth=2)
        self.ax2.set_ylabel("Schadebedrag (Euro's)")
        self.ax2.set_ylim(bottom=0)

    def combine_classes(self, lu_omzetting, output_path):
        nieuwe_klasses = np.array(lu_omzetting["nieuwe_klasse"].unique())
        samenvoeging_klasses = {
            "fid": self.df_lu_opp_schade["fid"]
        }  # waar komt self.df vandaan? En misschien een iets duidelijkere naam geven hiervoor.
        for klasse in nieuwe_klasses:
            oude_lu_per_nieuwe_klasse = (
                lu_omzetting.where(lu_omzetting["nieuwe_klasse"] == klasse)["LU_class"].dropna().values.tolist()
            )
            oude_lu_per_nieuwe_klasse = [str(int(x)) for x in oude_lu_per_nieuwe_klasse]
            samenvoeging_klasses[klasse] = self.df_lu_opp_schade.filter(items=oude_lu_per_nieuwe_klasse).sum(axis=1)
        self.df_combined_classes = pd.DataFrame(data=samenvoeging_klasses)
        self.df_combined_classes.to_csv(output_path, sep=",")
        self.df_lu_opp_schade = self.df_combined_classes.copy()


class LandgebruikCurveFiguur(PercentageFiguur):
    def __init__(self, path, agg_dir):
        super().__init__(path, agg_dir)

    def run(self, lu_omzetting, output_path, name, schadecurve_totaal=False, dpi=DPI):
        ids = np.array(self.df_lu_opp_schade["fid"].unique())
        for id in ids:
            self.lu_verdeling_peilgebied(id)
            self.create()
            self.handles_legend(lu_omzetting)
            self.ax.stackplot(
                self.df_peilgebied_perc.index,
                self.lu,
                colors=[self.color_dict.get(x, "black") for x in self.df_peilgebied_perc.columns],
            )
            if schadecurve_totaal:
                # self.sum_damages()
                self.plot_schadecurve_totaal(id)
            self.set_x_y_label()
            self.set_x_y_lim()
            self.set_x_y_ticks()
            self.title(f"landgebruikverdeling voor {name}")

            self.ax.legend(
                handles=self.handels, labels=self.labels, bbox_to_anchor=(0.05, -0.05), loc="upper left", ncols=8
            )

            self.write(output_path, dpi=dpi)


class DamagesLuCurveFiguur(PercentageFiguur):
    def __init__(self, path, agg_dir):
        super().__init__(path, agg_dir)

    def run(self, lu_omzetting, output_path, name, schadecurve_totaal=False, dpi=DPI):
        ids = np.array(self.df_lu_opp_schade["fid"].unique())
        for id in ids:
            self.lu_verdeling_peilgebied(id)

            self.create()
            self.handles_legend(lu_omzetting)
            self.ax.stackplot(
                self.df_peilgebied_perc.index,
                self.lu,
                colors=[self.color_dict.get(x, "gray") for x in self.df_peilgebied_perc.columns],
            )
            if schadecurve_totaal:
                # self.sum_damages()
                self.plot_schadecurve_totaal(id)
            self.set_x_y_label()
            self.set_x_y_lim()
            self.set_x_y_ticks()
            self.title(f"schadeverdeling voor {name}")

            self.ax.legend(
                handles=self.handels, labels=self.labels, bbox_to_anchor=(0.05, -0.05), loc="upper left", ncols=8
            )

            self.write(output_path, dpi=dpi)


if __name__ == "__main__":
    lu_conversion_table = pd.read_csv(
        r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\Project\05 Project execution\Omzettingstabel_landgebruik.csv"
    )
    # damages = DamagesLuCurveFiguur(agg_dir.agg_landuse_damages.path)
    # damages.combine_classes(lu_conversion_table, agg_dir.path/"result_lu_damages_classes.csv")
    # (agg_dir.path/"schade_percentagescurve").mkdir(exist_ok = True)
    # damages.run(self.lu_conversion_table, agg_dir.path/"schade_percentagescurve")
