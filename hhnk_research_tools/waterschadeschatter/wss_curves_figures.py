# %%
from pathlib import Path
from typing import Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Globals
STANDAARD_FIGUUR = (10, 10)  # figsize
LANDGEBRUIK_FIGUUR = (20, 10)  # figsize
DPI = 300
MAX_PEILVERHOGING = 2.5  # meter


class Figuur:
    def __init__(self, figsize: Tuple[int, int] = STANDAARD_FIGUUR):
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


class BergingsCurveFiguur(Figuur):
    def __init__(self, path: Path):
        super().__init__()
        self.df = pd.read_csv(path, index_col=0)

        self.xlabel_dsc = "Waterstand (m+NAP)"
        self.ylabel_dsc = "Volume (m3)"

    def run(self, output_dir: Path, dpi=DPI):
        for col in self.df.columns:
            valid_data = self.df[col].dropna()
            self.create()
            self.set_x_y_label()
            self.plot(valid_data)
            self.title(f"bergingscurve voor {col}")
            plotpng = output_dir.joinpath(f"bergingscurve_{col}.png")
            self.write(plotpng, dpi=dpi)


class PercentageFiguur(Figuur):
    def __init__(self):
        super().__init__(figsize=LANDGEBRUIK_FIGUUR)

        self.xlabel_dsc = "Peilverhoging boven streefpeil (m)"
        self.ylabel_dsc = "Percentage t.o.v. totaal"
        self.xlim_min = 0.1
        self.xlim_max = MAX_PEILVERHOGING
        self.ylim_min = 0
        self.ylim_max = 1
        self.x_ticks_list = np.arange(0.1, MAX_PEILVERHOGING + 0.1, 0.1)
        self.y_ticks_list = np.arange(0, 1.1, 0.1)
        self.ylabels = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]

    def set_x_y_ticks(self) -> None:
        self.xticks(self.x_ticks_list)
        self.yticks(ticks=self.y_ticks_list, labels=self.ylabels)

    def handles_legend(self, lu_omzetting) -> None:
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

    def lu_verdeling_peilgebied(self, id) -> None:
        self.df_peilgebied = self.df.loc[self.df["fid"] == id].dropna(axis=1)
        self.df_peilgebied = self.df_peilgebied.drop("fid", axis=1)
        self.df_peilgebied_perc = self.df_peilgebied.divide(self.df_peilgebied.sum(axis=1), axis=0)
        self.df_peilgebied_perc.columns = [str(int(x)) for x in self.df_peilgebied_perc]

        self.lu = []
        self.lu_ids = []
        for col in self.df_peilgebied_perc.columns:
            self.lu.append(self.df_peilgebied_perc[col])
            self.lu_ids.append(col)

    def sum_damages(self) -> None:
        self.df_sum_damages = self.df.copy()
        self.df_sum_damages["totaal"] = self.df_sum_damages.drop("fid", axis=1).sum(axis=1)

    def plot_schadecurve_totaal(self, id) -> None:
        self.ax2 = self.ax.twinx()
        self.ax2.plot(
            self.df_sum_damages.index.where(self.df["fid"] == id).dropna(),
            self.df_sum_damages["totaal"].where(self.df_sum_damages["fid"] == id).dropna(),
            color="black",
            linewidth=2,
        )
        self.ax2.set_ylabel("Schadebedrag (Euro's)")
        self.ax2.set_ylim(bottom=0)

    def combine_classes(self, lu_omzetting: pd.DataFrame, output_path: Path) -> None:
        nieuwe_klasses = np.array(lu_omzetting["nieuwe_klasse"].unique())
        samenvoeging_klasses = {
            "fid": self.df["fid"]
        }  # TODO waar komt self.df vandaan? En misschien een iets duidelijkere naam geven hiervoor.
        for klasse in nieuwe_klasses:
            oude_lu_per_nieuwe_klasse = (
                lu_omzetting.where(lu_omzetting["nieuwe_klasse"] == klasse)["LU_class"].dropna().values.tolist()
            )
            oude_lu_per_nieuwe_klasse = [str(int(x)) for x in oude_lu_per_nieuwe_klasse]
            samenvoeging_klasses[klasse] = self.df.filter(items=oude_lu_per_nieuwe_klasse).sum(axis=1)
        self.df_combined_classes = pd.DataFrame(data=samenvoeging_klasses)
        self.df_combined_classes.to_csv(output_path, sep=",")
        self.df = self.df_combined_classes.copy()


class LandgebruikCurveFiguur(PercentageFiguur):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path, index_col=0)

    def run(self, lu_omzetting, output_dir: Path, schadecurve_totaal=False, dpi=DPI):
        ids = np.array(self.df["fid"].unique())
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
                self.sum_damages()
                self.plot_schadecurve_totaal(id)
            self.set_x_y_label()
            self.set_x_y_lim()
            self.set_x_y_ticks()
            self.title(f"landgebruikverdeling voor {id}")

            self.ax.legend(
                handles=self.handels, labels=self.labels, bbox_to_anchor=(0.05, -0.05), loc="upper left", ncols=8
            )

            plotpng = output_dir.joinpath(f"landgebruikcurve_{id}.png")
            self.write(plotpng, dpi=dpi)


class DamagesLuCurveFiguur(PercentageFiguur):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path, index_col=0)

    def run(self, lu_omzetting, output_dir: Path, schadecurve_totaal: bool = False, dpi: int = DPI):
        ids = np.array(self.df["fid"].unique())
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
                self.sum_damages()
                self.plot_schadecurve_totaal(id)
            self.set_x_y_label()
            self.set_x_y_lim()
            self.set_x_y_ticks()
            self.title(f"schadeverdeling voor {id}")

            self.ax.legend(
                handles=self.handels, labels=self.labels, bbox_to_anchor=(0.05, -0.05), loc="upper left", ncols=8
            )

            plotpng = output_dir.joinpath(f"schadecurve_{id}.png")
            self.write(plotpng, dpi=dpi)


# %% figuur voor bergingscurve
def bergingscurve(input: pd.DataFrame, output_directory: Path):
    output_dir = output_directory.joinpath("Bergingscurve")
    output_dir.mkdir(exist_ok=True)

    for col in input.columns:
        valid = input[col].dropna()
        # Prettier plotting with seaborn
        sns.set_theme(font_scale=1.5)
        sns.set_style("whitegrid")

        # Format histograms
        plt.rcParams["figure.figsize"] = (8, 8)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax = sns.lineplot(valid)
        ax.set(
            xlabel="Waterstand (m+NAP)",
            ylabel="Volume (m3)",
            title=f"bergingscurve voor {col}",
        )

        plotpng = output_dir.joinpath(f"bergingscurve_{col}.png")
        plt.savefig(plotpng, dpi=300)
        plt.close()
