import argparse
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.features import rasterize
from shapely.geometry import box, shape

import hhnk_research_tools as hrt
import hhnk_research_tools.waterschadeschatter.resources as wss_resources
from hhnk_research_tools.variables import DEFAULT_NODATA_VALUES

LANDUSE_CONVERSION_TABLE = hrt.get_pkg_resource_path(wss_resources, "landuse_conversion_table.csv")
BUILDING_CONVERSION_TABLE = hrt.get_pkg_resource_path(wss_resources, "building_conversion_table.csv")

BUILDING_DTYPE = "uint32"
BUILDINGS_ID_FIELD = "feature_id"
BUILDING_DAMAGE_THRESHOLD = 1000


@dataclass
class WSSPost:
    """Post-processing class for WSS damage analysis with visualization and statistics generation."""

    damage_path: Union[str, Path]
    landuse_path: Union[str, Path]
    buildings_path: Union[str, Path]
    output_dir: Union[str, Path]

    def __post_init__(self) -> None:
        """Initialize raster objects and spatial extent after dataclass creation."""
        self.damage = hrt.Raster(self.damage_path)
        self.bounds = self.damage.metadata.bbox_gdal  # [minx, miny, maxx, maxy]
        self.extent_geom = box(self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])
        self.output_dir = Path(self.output_dir)

    @cached_property
    def damage_array(self) -> np.ndarray:
        """Load and return the damage raster array."""
        return self.damage._read_array()

    @cached_property
    def has_damage_array(self) -> np.ndarray:
        """Return boolean array indicating pixels with damage > 0."""
        return self.damage_array > 0

    @cached_property
    def landuse_array(self) -> np.ndarray:
        """Create VRT for landuse raster clipped to damage extent and return array."""
        vrt_path = self.output_dir / "landuse.vrt"
        self.landuse = hrt.Raster.build_vrt(
            vrt_out=vrt_path,
            input_files=self.landuse_path,
            bounds=self.damage.metadata.bbox_gdal,
            overwrite=True,
        )
        _vrt_array = self.landuse._read_array()
        return _vrt_array

    @cached_property
    def buildings_array(self) -> np.ndarray:
        """Rasterize buildings to match damage raster resolution and extent."""
        buildings = gpd.read_file(self.buildings_path)
        self.buildings = buildings[buildings.intersects(self.extent_geom)]
        _buildings_array = np.full(
            self.damage_array.shape, DEFAULT_NODATA_VALUES[BUILDING_DTYPE], dtype=BUILDING_DTYPE
        )

        if len(self.buildings) == 0:
            return _buildings_array

        buildings_meta = hrt.RasterMetadataV2.from_gdf(gdf=self.buildings, res=self.damage.metadata.pixel_width)
        shapes = [(building.geometry, int(building[BUILDINGS_ID_FIELD])) for _, building in self.buildings.iterrows()]

        rasterize(
            shapes=shapes,
            out=_buildings_array,
            transform=buildings_meta.to_rio_profile(0)["transform"],
            dtype=BUILDING_DTYPE,
            fill=DEFAULT_NODATA_VALUES[BUILDING_DTYPE],
            merge_alg=rasterio.enums.MergeAlg.replace,
            all_touched=False,
        )
        return _buildings_array

    @cached_property
    def lu_convert_table(self) -> pd.DataFrame:
        """Load landuse conversion table with codes, descriptions, and colors."""
        return pd.read_csv(LANDUSE_CONVERSION_TABLE)

    @cached_property
    def building_convert_table(self) -> pd.DataFrame:
        """Load building conversion table with functions, descriptions, and colors."""
        return pd.read_csv(BUILDING_CONVERSION_TABLE)

    @cached_property
    def lu_color_mapping(self) -> Dict[str, str]:
        """Create mapping from landuse descriptions to colors."""
        return dict(zip(self.lu_convert_table["beschrijving"], self.lu_convert_table["kleur"]))

    @cached_property
    def lu_label_mapping(self) -> Dict[int, str]:
        """Create mapping from landuse codes to descriptions."""
        return dict(zip(self.lu_convert_table["LU_class"], self.lu_convert_table["beschrijving"]))

    @cached_property
    def building_color_mapping(self) -> Dict[str, str]:
        """Create mapping from building functions to colors."""
        return dict(zip(self.building_convert_table["functie"], self.building_convert_table["kleur"]))

    def create_bar_chart(
        self,
        data: pd.DataFrame,
        field: str,
        field_label: str,
        ylabel: str,
        title: str,
        color: Union[str, list] = "random",
        output_path: Optional[Union[str, Path]] = None,
        secondary_data: Optional[pd.DataFrame] = None,
        secondary_field: Optional[str] = None,
        secondary_field_label: Optional[str] = None,
        secondary_ylabel: Optional[str] = None,
        secondary_color: Union[str, list] = "orange",
        chart_type: str = "bar",
        legend: bool = False,
        labels: bool = True,
    ) -> None:
        """
        Create bar chart with optional secondary y-axis for dual-metric visualization.

        Parameters
        ----------
        data : pd.DataFrame
            Primary data with index as categories
        field : str
            Column name for primary bar heights
        field_label : str
            Column name for primary value labels
        ylabel : str
            Primary y-axis label
        title : str
            Chart title
        color : str or array-like
            Primary bar colors
        output_path : str, optional
            Path to save chart
        secondary_data : pd.DataFrame, optional
            Secondary data for second y-axis (must have same index as primary data)
        secondary_field : str, optional
            Column name for secondary bar heights
        secondary_field_label : str, optional
            Column name for secondary value labels
        secondary_ylabel : str, optional
            Secondary y-axis label
        secondary_color : str or array-like
            Secondary bar colors (default: orange)
        chart_type : str
            Type of secondary chart: 'bar' or 'line' (default: 'bar')
        legend : bool
            Whether to display legend (default: False)
        labels : bool
            Whether to display value labels on bars (default: True)
        """
        # Create chart
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Primary bars
        bars1 = ax1.bar(range(len(data.index)), data[field], color=color, alpha=0.7, width=0.35, label="Primary")
        ax1.set_xticks(range(len(data.index)))
        ax1.set_xticklabels(data.index, rotation=45, ha="right")
        ax1.set_ylabel(ylabel, color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(alpha=0.3)

        # Add value labels for primary bars
        if labels:
            for bar, value in zip(bars1, data[field_label]):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    str(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Secondary axis if data provided
        if secondary_data is not None and secondary_field is not None:
            ax2 = ax1.twinx()

            if chart_type == "line":
                # Line plot on secondary axis
                line = ax2.plot(
                    range(len(secondary_data.index)),
                    secondary_data[secondary_field],
                    color=secondary_color,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label="Secondary",
                )
                ax2.set_ylabel(secondary_ylabel, color=secondary_color)
                ax2.tick_params(axis="y", labelcolor=secondary_color)

                # Add value labels for line points
                if secondary_field_label and labels:
                    for i, value in enumerate(secondary_data[secondary_field_label]):
                        ax2.text(
                            i,
                            secondary_data[secondary_field].iloc[i] * 1.05,
                            str(value),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color=secondary_color,
                        )
            else:
                # Bar plot on secondary axis with offset - made darker
                bar_width = 0.35
                x_pos = [x + bar_width for x in range(len(secondary_data.index))]
                bars2 = ax2.bar(
                    x_pos,
                    secondary_data[secondary_field],
                    color=secondary_color,
                    alpha=0.9,
                    width=bar_width,
                    label="Secondary",
                    edgecolor="black",
                    linewidth=0.8,
                )  # Dark edges for contrast
                ax2.set_ylabel(secondary_ylabel)
                ax2.tick_params(axis="y")

                # Add value labels for secondary bars
                if secondary_field_label and labels:
                    for bar, value in zip(bars2, secondary_data[secondary_field_label]):
                        ax2.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.01,
                            str(value),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            if legend:
                # Add legend for dual-axis chart
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, [ylabel, secondary_ylabel], loc="upper right")

        ax1.set_title(title)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Bar chart saved: {output_path}")
        else:
            plt.show()

    def create_boxplot(
        self,
        damage_per_building: pd.DataFrame,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        title: str = "Schade per Gebouwtype",
        minimum_n: int = 5,
        damage_threshold: float = 1000,
    ) -> None:
        """
        Create a nice boxplot for damage per building type
        Uses the 'label' column and color mapping already prepared in the DataFrame

        Parameters:
        -----------
        damage_per_building : pd.DataFrame
            DataFrame with 'dmg' column and 'label' column for building types
        output_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size (width, height)
        title : str
            Plot title
        """

        # Use the prepared data with labels already mapped
        damage_df = damage_per_building.copy()

        # Filters of dmg and mininmum of N
        damage_df = damage_df[damage_df["dmg"] > damage_threshold]

        # Filter out labels that appear only once (insufficient for boxplot statistics)
        label_counts = damage_df["label"].value_counts()
        valid_labels = label_counts[label_counts > minimum_n].index
        damage_df = damage_df[damage_df["label"].isin(valid_labels)]
        if damage_df.empty:
            return

        # Convert damage to thousands for better readability
        damage_df["damage_k"] = damage_df["dmg"] / 1000

        # Get unique building types and their colors
        building_types = damage_df["label"].unique()
        colors = damage_df["color"].unique()

        # Create figure
        plt.figure(figsize=figsize)

        # Create boxplot with custom styling
        box_plot = plt.boxplot(
            [damage_df[damage_df["label"] == bt]["damage_k"].values for bt in building_types],
            labels=building_types,
            patch_artist=True,  # Enable color filling
            notch=True,  # Add notches for median confidence interval
            showmeans=True,  # Show mean as well as median
            meanline=True,  # Show mean as line instead of point
        )

        # Color the boxes
        for patch, color in zip(box_plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)

        # Style the other elements
        for element in ["whiskers", "fliers", "medians", "caps"]:
            plt.setp(box_plot[element], color="black", linewidth=1.5)

        # Style means
        plt.setp(box_plot["means"], color="red", linewidth=2)

        # Customize plot
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.ylabel("Schade (k€)", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        # Set logarithmic y-scale
        # plt.yscale('log')

        # Add statistics text box
        stats_text = []
        for bt in building_types:
            bt_data = damage_df[damage_df["label"] == bt]["damage_k"]
            median_val = bt_data.median()
            mean_val = bt_data.mean()
            count = len(bt_data)
            stats_text.append(f"{bt}: n={count}, μ={mean_val:.1f}k€, m={median_val:.1f}k€")

        # Add legend/stats box in top-right corner
        plt.figtext(
            0.98,
            0.98,
            "\n".join(stats_text),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            verticalalignment="top",
            horizontalalignment="right",
        )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Damage boxplot saved: {output_path}")
        else:
            plt.show()

    def run_barchart_total_landuse_area(self, return_data_only: bool = False) -> Optional[pd.DataFrame]:
        """Generate bar chart showing total area by landuse type."""

        # Count all unique landuse-use pixels
        lu_counts = fast_unique_counts(self.landuse_array, self.landuse.nodata)

        # Calculate area in Hectares
        lu_counts = pd.DataFrame(data={"lu_pixels": lu_counts[1]}, index=lu_counts[0])
        lu_counts["area"] = lu_counts["lu_pixels"] * (self.damage.metadata.pixel_width**2)
        lu_counts["area_ha"] = (lu_counts["area"] / 10000).astype(int)

        # Convert land-use codes' to description and combine similar descriptions
        lu_counts["label"] = lu_counts.index.map(self.lu_label_mapping)
        lu_counts = lu_counts.groupby("label").sum()

        # color
        lu_counts["color"] = lu_counts.index.map(self.lu_color_mapping)

        if return_data_only:
            return lu_counts

        lu_counts.to_csv(self.output_dir / "total_landuse_area.csv")
        self.create_bar_chart(
            lu_counts,
            "area_ha",
            "area_ha",
            "Oppervlakte [Ha]",
            "Totaal landgebruik",
            color=lu_counts["color"],
            output_path=self.output_dir / "total_landuse_area.png",
        )

    def run_barchart_damage_landuse_area(self, return_data_only: bool = False) -> Optional[pd.DataFrame]:
        """Generate bar chart showing total area by landuse type."""

        # select only landuse array's that ahve damage
        lu_array = self.landuse_array.copy()
        lu_array[self.damage_array <= 0] = self.landuse.nodata

        # Count all unique landuse-use pixels
        lu_counts = fast_unique_counts(lu_array, self.landuse.nodata)

        # Calculate area in Hectares
        lu_counts = pd.DataFrame(data={"lu_pixels": lu_counts[1]}, index=lu_counts[0])
        lu_counts["area"] = lu_counts["lu_pixels"] * (self.damage.metadata.pixel_width**2)
        lu_counts["area_ha"] = (lu_counts["area"] / 10000).astype(int)

        # Convert land-use codes' to description and combine similar descriptions
        lu_counts["label"] = lu_counts.index.map(self.lu_label_mapping)
        lu_counts = lu_counts.groupby("label").sum()

        # color
        lu_counts["color"] = lu_counts.index.map(self.lu_color_mapping)

        if return_data_only:
            return lu_counts

        lu_counts.to_csv(self.output_dir / "damage_landuse_area.csv")
        self.create_bar_chart(
            lu_counts,
            "area_ha",
            "area_ha",
            "Oppervlakte [Ha]",
            "Landgebruik met schade",
            color=lu_counts["color"],
            output_path=self.output_dir / "damage_landuse_area.png",
        )

    def run_barchart_damage_per_landuse(self, return_data_only: bool = False) -> Optional[pd.DataFrame]:
        """Generate bar chart showing total damage amount by landuse type."""

        landuse_array = self.landuse_array[self.has_damage_array]
        damage_array = self.damage_array[self.has_damage_array]

        damage_per_lu = pd.DataFrame({"dmg": damage_array.astype(int), "lu": landuse_array})
        damage_per_lu = damage_per_lu.groupby("lu").sum()

        # Convert land-use codes' to description and combine similar descriptions
        damage_per_lu["label"] = damage_per_lu.index.map(self.lu_label_mapping)
        damage_per_lu = damage_per_lu.groupby("label").sum()

        # color
        damage_per_lu["color"] = damage_per_lu.index.map(self.lu_color_mapping)

        if return_data_only:
            return damage_per_lu

        damage_per_lu.to_csv(self.output_dir / "damage_per_landuse.csv")

        self.create_bar_chart(
            damage_per_lu,
            "dmg",
            "dmg",
            "Schade [euro]",
            "Schade per type landgebruik",
            color=damage_per_lu["color"],
            output_path=self.output_dir / "damage_per_landuse.png",
        )

    def run_barchart_landuse(self) -> None:
        """Generate dual-axis bar chart comparing landuse area vs damage."""

        landuse_area = self.run_barchart_damage_landuse_area(True)
        landuse_damage = self.run_barchart_damage_per_landuse(True)

        common_index = landuse_area.index.intersection(landuse_damage.index)
        lu_area_aligned = landuse_area.loc[common_index]
        lu_damage_aligned = landuse_damage.loc[common_index]

        self.create_bar_chart(
            data=lu_area_aligned,
            field="area_ha",
            field_label="area_ha",
            ylabel="Oppervlakte [ha]",
            title="Landgebruik: Oppervlakte vs Schade",
            color=lu_area_aligned["color"],
            secondary_data=lu_damage_aligned,
            secondary_field="dmg",
            secondary_field_label="dmg",
            secondary_ylabel="Schade [euro]",
            secondary_color=lu_damage_aligned["color"],
            chart_type="bar",  # or "bar" for dual bars
            output_path=self.output_dir / "landuse_area_vs_damage.png",
            legend=False,
            labels=False,
        )

    def run_barchart_damage_per_building(
        self,
        damage_threshold: float = BUILDING_DAMAGE_THRESHOLD,
        use_label: str = "gebruiksdoel",
        return_data_only: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Generate bar chart showing damage distribution by building type."""

        buildings_array = self.buildings_array[self.has_damage_array]
        damage_array = self.damage_array[self.has_damage_array]

        damage_per_building = pd.DataFrame(
            {"dmg": damage_array.flatten().astype(int), "building": buildings_array.flatten()}
        )

        damage_per_building = damage_per_building.groupby("building").sum()
        # remove
        damage_per_building = damage_per_building[damage_per_building.index != DEFAULT_NODATA_VALUES[BUILDING_DTYPE]]

        if not return_data_only:
            damage_per_building = damage_per_building[damage_per_building["dmg"] > damage_threshold]

        mapping = dict(zip(self.buildings[BUILDINGS_ID_FIELD], self.buildings[use_label]))

        damage_per_building["label"] = damage_per_building.index.map(mapping)

        # Er zijn soms meerdere functies boven op elkaar, hier wordt alleen de eerste genomen
        damage_per_building["label"] = damage_per_building["label"].str.split(",").str[0]

        # soms zijn ze ook niet gedefinieerd
        damage_per_building["label"] = damage_per_building["label"].fillna("Onbekend")

        # kleur
        damage_per_building["color"] = damage_per_building["label"].map(self.building_color_mapping)
        damage_per_building["color"] = damage_per_building["color"].fillna("gray")

        if return_data_only:
            return damage_per_building

        damage_per_building.to_csv(self.output_dir / "damage_per_building.csv")
        self.create_bar_chart(
            damage_per_building,
            "dmg",
            "label",
            "Schade [euro]",
            f"Schade per gebouw (boven {damage_threshold} euro)",
            color=damage_per_building["color"],
            output_path=self.output_dir / "damage_per_building.png",
        )

    def run_boxplot_damage_per_building(self, use_label: str = "gebruiksdoel") -> None:
        """Generate boxplot showing damage distribution by building type with statistical summaries."""
        damage_per_building = self.run_barchart_damage_per_building(use_label, return_data_only=True)
        self.create_boxplot(damage_per_building, output_path=self.output_dir / "damage_per_building_type_boxplot.png")


def fast_unique_counts(array: np.ndarray, nodata_value: Union[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized unique counts for building data with known nodata value.

    Parameters
    ----------
    array : np.ndarray
        Input array to count unique values
    nodata_value : int or float
        Value to exclude from counting

    Returns
    -------
    tuple
        Tuple of (unique_values, counts)
    """
    # Remove nodata values first (they're typically the majority)
    valid_data = array[array != nodata_value]

    if len(valid_data) == 0:
        return np.array([]), np.array([])

    # Use bincount if values are in reasonable range
    if valid_data.min() >= 0:  # Reasonable range
        counts = np.bincount(valid_data)
        unique_vals = np.nonzero(counts)[0]
        counts = counts[unique_vals]
        return unique_vals, counts
    else:
        # Fallback to standard unique for large ranges
        return np.unique(valid_data, return_counts=True)


def parse_run_cmd() -> None:
    """
    Parse command line arguments for running WSS post-processing analysis.

    Returns
    -------
    None
        This function parses arguments and executes the post-processing analysis.
    """
    parser = argparse.ArgumentParser(description="WSS Post-processing: Generate damage analysis charts and statistics")

    # Required arguments
    parser.add_argument("-damage_raster", type=str, required=True, help="Path to damage raster file")
    parser.add_argument("-landuse_raster", type=str, required=True, help="Path to landuse raster file")
    parser.add_argument("-output_dir", type=str, required=True, help="Output directory for results")

    parser.add_argument("-buildings_vector", type=str, required=True, help="Path to buildings vector file")
    parser.add_argument(
        "--damage_threshold",
        type=float,
        required=False,
        default=1000.0,
        help="Minimum damage threshold for building analysis. Default: 1000",
    )

    # Analysis options
    parser.add_argument(
        "--run_landuse_analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run landuse damage analysis. Default: True",
    )
    parser.add_argument(
        "--run_building_analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run building damage analysis. Default: True",
    )
    parser.add_argument(
        "--create_boxplots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create boxplot visualizations. Default: True",
    )
    parser.add_argument(
        "--use_label",
        type=str,
        required=False,
        default="gebruiksdoel",
        help="Building label field to use. Default: 'gebruiksdoel'",
    )

    args = parser.parse_args()
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    try:
        # Initialize WSSPost object
        wss_post = WSSPost(
            damage_path=hrt.Raster(args.damage_raster),
            landuse_path=hrt.Raster(args.landuse_raster),
            buildings_path=hrt.Raster(args.buildings_vector) if args.buildings_vector else None,
            output_dir=Path(args.output_dir),
        )

        # Create output directory
        wss_post.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting WSS post-processing analysis...")
        print(f"Damage raster: {args.damage_raster}")
        print(f"Landuse raster: {args.landuse_raster}")
        print(f"Buildings raster: {args.buildings_vector or 'None'}")
        print(f"Output directory: {args.output_dir}")

        # Run landuse analysis if requested
        if args.run_landuse_analysis:
            print("\n=== Running Landuse Analysis ===")

            # Generate landuse area chart
            print("Creating landuse area chart...")
            wss_post.run_barchart_total_landuse_area()

            # Generate landuse area damage chart 
            print("Creating landuse area damage chart...")
            wss_post.run_barchart_damage_landuse_area()

            # Generate landuse damage chart
            print("Creating landuse damage chart...")
            wss_post.run_barchart_damage_per_landuse()

            # Generate combined landuse chart
            print("Creating combined landuse chart...")
            wss_post.run_barchart_landuse()

        # Run building analysis if requested and data available
        if args.run_building_analysis:
            print("\n=== Running Building Analysis ===")

            # Generate building damage chart
            print("Creating building damage chart...")
            wss_post.run_barchart_damage_per_building(damage_threshold=args.damage_threshold, use_label=args.use_label)

            # Generate building boxplot if requested
            if args.create_boxplots:
                print("Creating building damage boxplot...")
                wss_post.run_boxplot_damage_per_building(use_label=args.use_label)

        elif args.run_building_analysis:
            print("WARNING: Building analysis requested but no building data provided")

        print("\n=== Analysis Complete ===")
        print(f"Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    parse_run_cmd()
