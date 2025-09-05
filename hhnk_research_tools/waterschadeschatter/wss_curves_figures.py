# %%
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, Union

import geopandas as gp
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pathlib import Path, WindowsPath

# Globals
STANDAARD_FIGUUR = (10, 10)
LANDGEBRUIK_FIGUUR = (20, 10)
DPI = 300
MAX_PEILVERHOGING = 2.5
COLORS = np.random.rand(50, 3)


@dataclass
class Figuur:
    xlabel_description: str
    ylabel_description: str
    figsize: Tuple[int, int] = STANDAARD_FIGUUR

    def __post_init__(self) -> None:
        self.fig: Figure = None
        self.ax: Axes = None

    def create(self) -> None:
        """Create a new figure and axis with the specified figure size."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

    def plot(self, dataframe: Union[pd.DataFrame, pd.Series]) -> None:
        """Plot the given dataframe or series on the current axis."""
        if self.ax:
            self.ax.plot(dataframe)

    def xlabel(self, naam: str) -> None:
        """Set the x-axis label."""
        if self.ax:
            self.ax.set(xlabel=naam)

    def ylabel(self, naam: str) -> None:
        """Set the y-axis label."""
        if self.ax:
            self.ax.set(ylabel=naam)

    def title(self, naam: str) -> None:
        """Set the plot title."""
        if self.ax:
            self.ax.set(title=naam)

    def xlim(self, xmin: float, xmax: float) -> None:
        if self.ax:
            self.ax.set(xlim=(xmin, xmax))

    def ylim(self, ymin: float, ymax: float) -> None:
        if self.ax:
            self.ax.set(ylim=(ymin, ymax))

    def grid(self) -> None:
        if self.ax:
            self.ax.grid(axis="both", color="lightgray")

    def xticks(self, ticks: List[float], labels: Optional[List[str]] = None) -> None:
        if self.ax:
            self.ax.set_xticks(ticks, labels)

    def yticks(self, ticks: List[float], labels: Optional[List[str]] = None) -> None:
        if self.ax:
            self.ax.set_yticks(ticks, labels=labels)

    def set_x_y_label(self) -> None:
        self.xlabel(self.xlabel_description)
        self.ylabel(self.ylabel_description)

    def set_x_y_lim(self) -> None:
        self.xlim(self.xlim_min, self.xlim_max)
        self.ylim(self.ylim_min, self.ylim_max)

    def write(self, path: str, dpi: int = DPI) -> None:
        """Save the figure to file and close the plot."""
        plt.savefig(path, dpi=dpi)
        plt.close()


class CurveFiguur(Figuur):
    def __init__(self, damage_df: pd.DataFrame) -> None:
        super().__init__(
            xlabel_description="Peilverhoging boven streefpeil (m)",
            ylabel_description = "Volume (m3)"
        )
        self.df_damages = damage_df

    def run(self, output_path: str, name: str, title: str, dpi: int = DPI) -> None:
        """Generate and save a damage curve figure."""
        self.create()
        self.plot(self.df_damages)
        if ("Berging" or "berging") in title:
            self.ylabel_description = "Volume (m3)"
        if ("Schade" or "schade") in title:
            self.ylabel_description = "Schadebedrag (Euro's)"
        self.set_x_y_label()
        self.title(f"{title} voor {name}")
        self.grid()
        self.write(output_path, dpi=dpi)


class BergingsCurveFiguur(Figuur):
    def __init__(self, volume_level_path: Union[WindowsPath, pd.DataFrame], vector_area: gp.GeoSeries) -> None:
        super().__init__(
            xlabel_description="Waterstand (m+NAP)",
            ylabel_description="Volume (m3)",
        )
        if type(volume_level_path) == WindowsPath:
            self.df_vol_level = pd.read_csv(volume_level_path, index_col=0)
        if type(volume_level_path) == pd.DataFrame:
            self.df_vol_level = volume_level_path.copy()
        self.ylabel_mm = "Berging (mm)"
        self.vector_area = vector_area
        self.ax_mm: Optional[Axes] = None

    def volume2mm(self, y: float) -> float:
        area = self.vector_area.geometry.area
        return y / area * 1000

    def convert_V_to_mm(self, ax: Axes) -> None:
        if self.ax and self.ax_mm:
            y1, y2 = self.ax.get_ylim()
            self.ax_mm.set_ylim(self.volume2mm(y1), self.volume2mm(y2))
            self.ax_mm.figure.canvas.draw()

    def run(self, output_path: str, name: str, dpi: int = DPI) -> None:
        for col in self.df_vol_level.columns:
            valid_data = self.df_vol_level[col].dropna()
            self.create()
            if self.ax:
                self.ax_mm = self.ax.twinx()
                self.ax.callbacks.connect("ylim_changed", self.convert_V_to_mm)

                self.plot(valid_data)
                self.set_x_y_label()
                self.grid()
                self.ax_mm.set_ylabel(self.ylabel_mm)
                self.title(f"bergingscurve voor {name}")
                self.write(output_path, dpi=dpi)


class PercentageFiguur(Figuur):
    def __init__(self, path: str, agg_dir: Any) -> None:
        super().__init__(
            xlabel_description="Peilverhoging boven streefpeil (m)",
            ylabel_description="Percentage t.o.v. totaal",
            figsize=LANDGEBRUIK_FIGUUR,
        )
        self.df_lu_opp_schade: pd.DataFrame = pd.read_csv(path, index_col=0)
        self.df_lu_opp_schade.columns = pd.Series(self.df_lu_opp_schade.columns).str.split(" ").str[0]
        self.df_sum_damages: pd.DataFrame = pd.read_csv(agg_dir.agg_damage.path, index_col=0)
        self.df_building_damage: pd.DataFrame = pd.read_csv(agg_dir.agg_building_dmg.path, index_col=0)
        self.df_building_damage.columns = pd.Series(self.df_building_damage.columns).str.split(" ").str[0]
        self.agg_dir = agg_dir

        self.xlim_min: float = 0.1
        self.xlim_max: float = MAX_PEILVERHOGING
        self.ylim_min: float = 0
        self.ylim_max: float = 1
        self.x_ticks_list: NDArray = np.arange(0.1, MAX_PEILVERHOGING + 0.1, 0.1)
        self.y_ticks_list: NDArray = np.arange(0, 1.1, 0.1)
        self.ylabels: List[str] = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]

    def set_x_y_ticks(self) -> None:
        self.xticks(self.x_ticks_list)
        self.yticks(ticks=self.y_ticks_list, labels=self.ylabels)

    def handles_legend(self, lu_omzetting: pd.DataFrame) -> None:
        nieuwe_klasses = np.array(lu_omzetting["nieuwe_klasse"].unique())
        self.color_dict = {}
        colors = []
        self.handles = []
        self.labels = []
        for klasse in nieuwe_klasses:
            label = lu_omzetting.where(lu_omzetting["nieuwe_klasse"] == klasse)["beschrijving"].dropna().unique()[0]
            color = lu_omzetting.where(lu_omzetting["nieuwe_klasse"] == klasse)["kleur"].dropna().unique()[0]
            self.labels.append(label)
            colors.append(color)
        for i in range(0, len(colors)):
            handle = mpatches.Patch(color=colors[i])
            self.handles.append(handle)
            self.color_dict[str(nieuwe_klasses[i])] = colors[i]

    def handles_legend_buildings(self) -> None:
        colors_b = COLORS[0 : len(self.df_building_dmg_perc_filtered.columns)]
        self.colors_b = [tuple(color) for color in colors_b]
        self.labels_b = list(self.df_building_dmg_perc_filtered.columns)
        self.handles_b = []
        for i in range(0, len(self.colors_b)):
            self.handles_b.append(mpatches.Patch(color=self.colors_b[i]))

    def lu_verdeling_peilgebied(self, id: int) -> None:
        self.df_peilgebied = self.df_lu_opp_schade.loc[self.df_lu_opp_schade["fid"] == id].dropna(axis=1)
        self.df_peilgebied = self.df_peilgebied.drop("fid", axis=1)
        self.df_peilgebied_perc = self.df_peilgebied.divide(self.df_peilgebied.sum(axis=1), axis=0)
        self.df_peilgebied_perc.columns = [str(int(x)) for x in self.df_peilgebied_perc]

        self.lu = []
        self.lu_ids = []
        for col in self.df_peilgebied_perc.columns:
            self.lu.append(self.df_peilgebied_perc[col])
            self.lu_ids.append(col)

    def schade_buildings_verdeling_peilgebied(self, id: int) -> None:
        self.df_building_damage_2 = self.df_building_damage.loc[self.df_building_damage["fid"] == id].dropna(axis=1)
        self.df_building_damage_2 = self.df_building_damage_2.drop("fid", axis=1)
        self.df_building_damage_perc = self.df_building_damage_2.divide(self.df_building_damage_2.sum(axis=1), axis=0)
        self.df_building_dmg_perc_filtered = self.df_building_damage_perc.loc[
            :, (self.df_building_damage_perc > 0.01).any(axis=0)
        ]
        self.df_building_dmg_perc_filtered.columns = [str(int(x)) for x in self.df_building_dmg_perc_filtered]

        self.schade_building = []
        self.building_nr = []
        for col in self.df_building_dmg_perc_filtered.columns:
            self.schade_building.append(self.df_building_dmg_perc_filtered[col])
            self.building_nr.append(col)

    # def sum_damages(self):
    #     self.df_sum_damages = self.df_lu_opp_schade.copy()
    #     self.df_sum_damages['totaal'] = self.df_sum_damages.drop('fid', axis=1).sum(axis=1)

    # .index.where(self.df_lu_opp_schade['fid'] == id).dropna(), self.df_sum_damages['totaal'].where(self.df_sum_damages['fid'] == id).dropna()

    def plot_schadecurve_totaal(self) -> None:
        self.ax2 = self.ax.twinx()
        self.ax2.plot(self.df_sum_damages, color="black", linewidth=2)
        self.ax2.set_ylabel("Schadebedrag (Euro's)")
        self.ax2.set_ylim(bottom=0)

    def plot_schade_buildings_totaal(self) -> None:
        self.df_tot_buildings = self.df_building_damage_2.sum(axis=1)
        self.ax2.plot(self.df_tot_buildings, color="red", linewidth=2)

    def combine_classes(self, lu_omzetting: pd.DataFrame, output_path: str) -> None:
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
    def __init__(self, path: str, agg_dir: Any) -> None:
        super().__init__(path, agg_dir)

    def run(
        self, lu_omzetting: pd.DataFrame, output_path: str, name: str, schadecurve_totaal: bool = False, dpi: int = DPI
    ) -> None:
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
                self.plot_schadecurve_totaal()
                self.handles.append(mlines.Line2D([], [], color="black", linewidth=2))
                self.labels.append("Totale schade")
            self.ylabel_description = self.ylabel_description + " landgebruik"
            self.set_x_y_label()
            self.set_x_y_lim()
            self.set_x_y_ticks()
            self.grid()
            self.title(f"landgebruikverdeling voor {name}")

            self.ax.legend(
                handles=self.handles, labels=self.labels, bbox_to_anchor=(0.05, -0.05), loc="upper left", ncols=8
            )

            self.write(output_path, dpi=dpi)


class DamagesLuCurveFiguur(PercentageFiguur):
    def __init__(self, path: str, agg_dir: Any) -> None:
        super().__init__(path, agg_dir)

    def run(
        self, lu_omzetting: pd.DataFrame, output_path: str, name: str, schadecurve_totaal: bool = False, dpi: int = DPI
    ) -> None:
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
                self.plot_schadecurve_totaal()
                self.handles.append(mlines.Line2D([], [], color="black", linewidth=2))
                self.labels.append("Totale schade")
            self.ylabel_description = self.ylabel_description + " schade"
            self.set_x_y_label()
            self.set_x_y_lim()
            self.set_x_y_ticks()
            self.grid()
            self.title(f"schadeverdeling voor {name}")

            self.ax.legend(
                handles=self.handles, labels=self.labels, bbox_to_anchor=(0.05, -0.05), loc="upper left", ncols=8
            )

            self.write(output_path, dpi=dpi)


class BuildingsSchadeFiguur(PercentageFiguur):
    def __init__(self, path: str, agg_dir: Any) -> None:
        super().__init__(path, agg_dir)

    def run(
        self,
        output_path: str,
        name: str,
        schadecurve_totaal: bool = False,
        schadebuildings_totaal: bool = False,
        dpi: int = DPI,
    ) -> None:
        ids = np.array(self.df_lu_opp_schade["fid"].unique())
        for id in ids:
            # self.lu_verdeling_peilgebied(id)
            self.schade_buildings_verdeling_peilgebied(id)

            self.handles_legend_buildings()
            self.create()
            # self.handles_legend(lu_omzetting)
            self.ax.stackplot(
                self.df_building_dmg_perc_filtered.index,
                self.schade_building,
                colors=self.colors_b,
            )
            if schadecurve_totaal:
                # self.sum_damages()
                self.plot_schadecurve_totaal()
                self.handles_b.append(mlines.Line2D([], [], color="black", linewidth=2))
                self.labels_b.append("Totale schade")
            if schadebuildings_totaal:  # werkt alleen als schadecurve_totaal ook aan staat ivm met ax2
                self.plot_schade_buildings_totaal()
                self.handles_b.append(mlines.Line2D([], [], color="red", linewidth=2))
                self.labels_b.append("Totale schade panden")
            self.ylabel_description = self.ylabel_description + " panden schade \n (pand meegenomen vanaf 1% schade)"
            self.set_x_y_label()
            self.set_x_y_lim()
            self.set_x_y_ticks()
            self.grid()
            self.title(f"schadeverdeling panden voor {name}")

            self.ax.legend(
                handles=self.handles_b, labels=self.labels_b, bbox_to_anchor=(0.05, -0.05), loc="upper left", ncols=8
            )
            # self.ax.legend(
            #     handles=self.handels, labels=self.labels, bbox_to_anchor=(0.05, -0.05), loc="upper left", ncols=8
            # )

            self.write(output_path, dpi=dpi)


# %%
class DamagesAggFiguur(Figuur):
    def __init__(self, agg_csv_dir: str) -> None:
        super().__init__(
            xlabel_description="Volume (m3)",
            ylabel_description="Schadebedrag (Euro's)",
        )
        self.df_agg_csv = pd.read_csv(agg_csv_dir)
        self.xlim_min = 0
        self.xlim_max = self.df_agg_csv["Volume [m3]"].max() * 1.1
        self.ylim_min = 0
        self.ylim_max = self.df_agg_csv.iloc[:, 1:].max().max() * 1.1
        self.x_ticks_list = np.arange(0, self.xlim_max, 500000)
        self.y_ticks_list = np.arange(0, self.ylim_max, 50000)

    def run(self, output_path: str, name: str, dpi: int = DPI) -> None:
        self.create()
        for i in range(1, 4):
            col_name = self.df_agg_csv.columns[i]
            self.ax.plot(self.df_agg_csv["Volume [m3]"], self.df_agg_csv[col_name], label=col_name, linewidth=2)
        self.ax.legend(
            labels=[
                "Neerslag bergen vanaf het laagste peilvak",
                "Neerslag verdelen zodat in elk peilvak een gelijke waterdiepte is",
                "In het peilvak gevallen neerslag bergen in hetzelfde peilvak",
            ],
            loc="upper left",
        )
        self.set_x_y_label()
        self.set_x_y_lim()
        self.xticks(self.x_ticks_list)
        self.yticks(self.y_ticks_list)
        self.grid()
        self.title(f"Schadebedragen voor drie type aggregaties voor {name}")
        self.write(output_path, dpi=dpi)
