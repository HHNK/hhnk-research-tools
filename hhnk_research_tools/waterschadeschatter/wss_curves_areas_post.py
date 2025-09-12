# %%
"""
Created on Thu Oct 10 15:27:18 2024

@author: kerklaac5395
"""

import datetime
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator, Optional, Tuple, TypeVar, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm

import hhnk_research_tools as hrt
import hhnk_research_tools.logging as logging
import hhnk_research_tools.waterschadeschatter.resources as wss_resources
from hhnk_research_tools.variables import DEFAULT_NODATA_GENERAL
from hhnk_research_tools.waterschadeschatter.wss_curves_figures import (
    BergingsCurveFiguur,
    BuildingsSchadeFiguur,
    CurveFiguur,
    DamagesAggFiguur,
    DamagesLuCurveFiguur,
    LandgebruikCurveFiguur,
)
from hhnk_research_tools.waterschadeschatter.wss_curves_folium import WSSCurvesFolium
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import (
    DRAINAGE_LEVEL_FIELD,
    ID_FIELD,
    AreaDamageCurveFolders,
    WSSTimelog,
)

# Type variables for generics
T = TypeVar("T")
D = TypeVar("D", bound=pd.DataFrame)

# Logger
logger = logging.get_logger(__name__)

# Globals
NAME = "AreaDamageCurves Aggregation"

LANDUSE_CONVERSION_TABLE = hrt.get_pkg_resource_path(wss_resources, "landuse_conversion_table.csv")

# Defaults
DEFAULT_RAIN = 200  # mm
DEFAULT_BUFFER = 100  # m
DEFAULT_RESOLUTION = 0.01  # m
DEFAULT_ROUND = 2
DEFAULT_AGG_METHODS = ["lowest_area", "equal_depth", "equal_rain"]
DEFAULT_PREDICATE = "_within"
INT = np.int64

@dataclass
class AreaDamageCurvesAggregation:
    """
    Aggregate the output of wss_curves_areas.py.

    Parameters
    ----------
    result_path : str
        Result path for the output directory.
    aggregate_vector_path : str
        Aggregation vector
    vector_field : str
        Field which is used to acces the aggregation vector.
    landuse_conversion_path : str
        Path to the landuse conversion table
    """

    result_path: Union[str, Path]
    aggregate_vector_path: Union[str, Path, None] = None
    aggregate_vector_id_field: Optional[str] = None
    aggregate_vector_layer_name: Optional[str] = None
    landuse_conversion_path: Union[str, Path, None] = LANDUSE_CONVERSION_TABLE

    def __post_init__(
        self,
    ):
        self.dir = AreaDamageCurveFolders(self.result_path, create=True)

        time_now = datetime.datetime.now().strftime("%Y%m%d%H%M")
        log_file = self.dir.work.log.joinpath(f"log_{time_now}.log")
        self.time = WSSTimelog(name=NAME, log_file=log_file)

        self.time.log("Initializing!")
        self.time.log("Reading drainage areas")
        self.drainage_areas = self.dir.input.area.load()
        self.drainage_areas[ID_FIELD] = self.drainage_areas[ID_FIELD].astype(str)

        self.time.log("Reading aggregate vector")
        if self.aggregate_vector_path:
            self.vector = gpd.read_file(
                self.aggregate_vector_path,
                layer=self.aggregate_vector_layer_name,
                use_arrow=True,
            )
            self.field = self.aggregate_vector_id_field

        self.time.log("Reading landuse conversion table")
        self.lu_conversion_table = pd.read_csv(self.landuse_conversion_path)

        self.predicate = DEFAULT_PREDICATE
        self.time.log("Finished initializing!")

    @classmethod
    def from_settings_json(cls, settings_json_file: Union[str, Path]) -> "AreaDamageCurvesAggregation":
        """Create aggregation instance from settings JSON file."""
        with open(str(settings_json_file)) as json_file:
            settings = json.load(json_file)
        return cls(**settings)

    def __iter__(self) -> Iterator[Tuple[int, pd.Series, gpd.GeoDataFrame]]:
        """Iterate over vector features and corresponding drainage areas."""
        for idx, feature in tqdm(self.vector.iterrows(), NAME, total=len(self.vector)):
            predicate_func = getattr(self, self.predicate)  # default = self._within
            areas = predicate_func(feature.geometry)
            if len(areas) > 0:
                yield idx, feature, areas

    def _fid_columns(self, fids: list, df: pd.DataFrame) -> pd.Series:
        """retrieve column in df where fid is present and second element of underscore"""
        return df.columns[df.columns.str.split("_").str[1].isin(fids)]
    
    @cached_property
    def lu_area_data(self):
        if self.dir.output.result_lu_areas.exists():
            self.time.log("Reading landuse-area data")
            data = pd.read_csv(self.dir.output.result_lu_areas.path, index_col=0)
            data.index = self.damage_curve.index
            return data
    
    @cached_property
    def lu_dmg_data(self):
        if self.dir.output.result_lu_damage.exists():
            self.time.log("Reading landuse-damage data")
            data = pd.read_csv(self.dir.output.result_lu_damage.path, index_col=0)
            data.index = self.damage_curve.index    
            return data
        
    @cached_property
    def bu_dmg_data(self):
        if self.dir.output.result_bu_damage.exists():
            self.time.log("Reading building-damage data")
            data = pd.read_csv(self.dir.output.result_bu_damage.path, index_col=0)
            data.index = self.damage_curve.index
            return data
        
    @cached_property
    def damage_curve(self) -> pd.DataFrame:
        if self.dir.output.result.exists():
            damage = pd.read_csv(self.dir.output.result.path, index_col=0)
            damage.columns = damage.columns.astype(str)
            return damage
        return pd.DataFrame()

    @cached_property
    def vol_curve(self) -> pd.DataFrame:
        if self.dir.output.result_vol.exists():
            vol = pd.read_csv(self.dir.output.result_vol.path, index_col=0)
            vol.columns = vol.columns.astype(str)
            return vol
        return pd.DataFrame()

    @cached_property
    def damage_interpolated_curve(self) -> pd.DataFrame:
        if self.dir.post_processing.damage_interpolated_curve.path.exists():
            self.time.log("Reading local damage interpolated curve!")
            return pd.read_csv(self.dir.post_processing.damage_interpolated_curve.path, index_col=0)
        
        self.time.log("Creating damage interpolated curve!")
        return self._curve_linear_interpolate(curve=self.damage_curve, resolution=DEFAULT_RESOLUTION)

    @cached_property
    def vol_interpolated_curve(self) -> pd.DataFrame:
        if self.dir.post_processing.volume_interpolated_curve.path.exists():
            self.time.log("Reading local volume interpolated curve!")
            return pd.read_csv(self.dir.post_processing.volume_interpolated_curve.path, index_col=0)
        
        self.time.log("Creating volume interpolated curve!")
        return self._curve_linear_interpolate(curve=self.vol_curve, resolution=DEFAULT_RESOLUTION)

    @cached_property
    def damage_level_curve(self) -> pd.DataFrame:
        if self.dir.post_processing.damage_level_curve.path.exists():
            self.time.log("Reading local damage level curve!")
            return pd.read_csv(self.dir.post_processing.damage_level_curve.path, index_col=0)
        
        self.time.log("Creating damage level curve!")
        return self._curves_to_level(curves=self.damage_curve)

    @cached_property
    def vol_level_curve(self) -> pd.DataFrame:
        if self.dir.post_processing.vol_level_curve.path.exists():
            self.time.log("Reading local volume level curve!")
            return pd.read_csv(self.dir.post_processing.vol_level_curve.path, index_col=0)
        self.time.log("Creating volume level curve!")
        return self._curves_to_level(curves=self.vol_curve)

    @cached_property
    def damage_level_per_ha(self) -> pd.DataFrame:
        damage_per_ha = {}
        for idx, area in self.drainage_areas.iterrows():
            level_data = self.damage_level_curve[area[ID_FIELD]]
            level_damage_m2 = level_data / area.geometry.area  # eur/m2
            level_damage_ha = level_damage_m2 / 10000  # eur/ha
            damage_per_ha[area[ID_FIELD]] = level_damage_ha
        return pd.DataFrame(damage_per_ha)

    @cached_property
    def damage_per_m3(self) -> pd.DataFrame:
        """Damage per m3 at a certain waterlevel"""
        data = {}
        for idx, area in self.drainage_areas.iterrows():
            vol_curve = self.vol_curve[area[ID_FIELD]]
            dmg_curve = self.damage_curve[area[ID_FIELD]]

            damage_shift = dmg_curve - dmg_curve.shift(+1)
            volume_shift = vol_curve - vol_curve.shift(+1)
            damage_per_m3 = damage_shift / volume_shift
            damage_per_m3.loc[0.1] = 0
            data[area[ID_FIELD]] = damage_per_m3
        return pd.DataFrame(data)

    @cached_property
    def fdla_geometry(self):
        da = self.drainage_areas.copy()
        da.index = da[ID_FIELD]
        return da.geometry

    @property
    def selection(self) -> dict[str, pd.Series]:
        selection = {}
        for idx, feature, areas in self:
            selection[feature[self.field]] = areas
        return selection

    def _within(self, geometry: shapely.geometry.base.BaseGeometry, buffer: int = DEFAULT_BUFFER) -> gpd.GeoDataFrame:
        buffered = geometry.buffer(buffer)
        area_within = self.drainage_areas[self.drainage_areas.geometry.within(buffered)]
        return area_within

    def _curve_linear_interpolate(self, curve: pd.Series, resolution: float, round: int = DEFAULT_ROUND) -> pd.Series:
        index = np.arange(0, curve.index.values[-1] + resolution, resolution).round(round)
        new_index = list(set(list(index) + list(curve.index)))
        new_index.sort()
        interpolated = curve.reindex(new_index)
        interpolated.loc[0.0] = 0
        return interpolated.interpolate("linear")

    def _curve_depth_to_level(
        self, curve: pd.Series, drainage_area: pd.Series, round: int = DEFAULT_ROUND
    ) -> pd.Series:
        curve.index = np.round(curve.index + drainage_area[DRAINAGE_LEVEL_FIELD], round)
        return curve

    def _curves_to_level(
        self,
        curves: pd.DataFrame,
        resolution=DEFAULT_RESOLUTION,
        round: int = DEFAULT_ROUND,
    ) -> pd.DataFrame:
        """Curves of drainage areas from depth to level, also interpolated."""
        d_sorted = self.drainage_areas.sort_values(by=DRAINAGE_LEVEL_FIELD, ascending=True)

        min_level = d_sorted.iloc[0][DRAINAGE_LEVEL_FIELD]
        max_level = d_sorted.iloc[-1][DRAINAGE_LEVEL_FIELD] + curves.index[-1]

        index = np.arange(min_level, max_level, resolution).round(round)
        level_curve = pd.DataFrame(index=index)

        for idx, drainage_area in d_sorted.iterrows():
            curve = curves[str(drainage_area[ID_FIELD])]
            if pd.isna(curve).all():
                curve = curve.fillna(DEFAULT_NODATA_GENERAL)
            curve = curve.astype(INT)
            interpolated_curve = self._curve_linear_interpolate(curve=curve, resolution=resolution)
            level = self._curve_depth_to_level(curve=interpolated_curve, drainage_area=drainage_area)
            level_curve[level.name] = level
            level_curve = level_curve.copy()  # to supress defragment warning.

        return level_curve

    def agg_damage(self, feature, areas_within) -> dict[str, pd.Series]:
        """Sum damage curves within the given areas."""
        damage_curves = self.damage_curve[areas_within[ID_FIELD]]
        return pd.Series(damage_curves.sum(axis=1), name=feature[self.field])

    def agg_volume(self, feature, areas_within) -> dict[str, pd.Series]:
        """Sum volume curves within the given areas."""
        vol_curves = self.vol_curve[areas_within[ID_FIELD]]
        return pd.Series(vol_curves.sum(axis=1), name=feature[self.field])

    def agg_landuse(self, feature, areas_within) -> dict[str, pd.Series]:
        """Sum land use areas data within the given areas."""
        lu_data = self.lu_area_data[self._fid_columns(areas_within[ID_FIELD], self.lu_area_data)]
        lu_data.columns = lu_data.columns.str.split("_").str[0]
        return lu_data.groupby(lu_data.columns, axis=1).sum()

    def agg_landuse_dmg(self, feature, areas_within) -> dict[str, pd.Series]:
        """Sum land use damage data within the given areas."""
        lu_data = self.lu_dmg_data[self._fid_columns(areas_within[ID_FIELD], self.lu_dmg_data)]
        lu_data.columns = lu_data.columns.str.split("_").str[0]
        return lu_data.groupby(lu_data.columns, axis=1).sum()

    def agg_buildings_dmg(self, feature, areas_within) -> dict[str, pd.Series]:
        """Sum land use damage data within the given areas."""
        bu_data = self.bu_dmg_data[self._fid_columns(areas_within[ID_FIELD], self.bu_dmg_data)]
        bu_data.columns = bu_data.columns.str.split("_").str[0]
        return bu_data.groupby(bu_data.columns, axis=1).sum()

    def agg_damage_level_per_ha(self, feature, areas_within) -> dict[str, pd.DataFrame]:
        """Damage level per ha within the given areas."""
        return self.damage_level_per_ha[areas_within[ID_FIELD]]

    def agg_damage_level_per_m3(self, feature, areas_within) -> dict[str, pd.DataFrame]:
        """Damage level per m3 within the given areas."""
        return self.damage_per_m3[areas_within[ID_FIELD]]

    def aggregate_rain_curve(
        self, feature, areas_within, method="lowest_area", mm_rain=DEFAULT_RAIN
    ) -> dict[str, pd.Series]:
        """Different for distribution of rain in the drainage area"""
        self.time.log(f"Creating aggregation with method: {method}.")
        if method == "lowest_area":
            output = self.agg_rain_lowest_area(feature, areas_within, mm_rain)
        elif method == "equal_depth":
            output = self.agg_rain_equal_depth(feature, areas_within, mm_rain)
        elif method == "equal_rain":
            output = self.agg_rain_own_area_retention(areas_within, mm_rain)
        self.time.log(f"Creating aggregation with method: {method} finished!")
        return output

    def agg_rain_lowest_area(
        self,
        feature,
        areas_within,
        mm_rain=DEFAULT_RAIN,
    ) -> pd.Series:
        """
        Create a new damage curve starting at the area with the lowest drainage level.
        1. Rain falls in the lowest areas, so damage curve is taken from that area.
        2. If the drainage level of the second area is reached, the damagecurves of the first and second area summed.
        3. This happens until total volume of the rain is stored.

        Returns a curve based on volume
        """
        total_volume_area = feature.geometry.area * (mm_rain / 1000)  # m3 rain in area.
        level_damage_curve = self.damage_level_curve[areas_within[ID_FIELD]]
        level_vol_curve = self.vol_level_curve[areas_within[ID_FIELD]]
        level_damage_curve = level_damage_curve.ffill()
        level_vol_curve = level_vol_curve.ffill()

        area_curve = pd.DataFrame(data={"volume": level_vol_curve.sum(axis=1)})
        area_curve["damage"] = level_damage_curve.sum(axis=1)
        agg_curve = area_curve[area_curve.volume <= total_volume_area]
        agg_curve.index = agg_curve.volume
        agg_curve = agg_curve.drop(columns=["volume"])

        agg_series = agg_curve["damage"]
        agg_series.name = "damage_lowest_area"

        return agg_series

    def agg_rain_equal_depth(self, feature, areas_within, mm_rain=DEFAULT_RAIN) -> pd.Series:
        """
        Create a new damage curve based on equal depth at every area.
        1. Essentially a sum of all damagecurves.
        2. The curve stops when a total volume is reached.
        """

        total_volume_area = feature.geometry.area * (mm_rain / 1000)  # m3 rain in area.
        damage_curves = self.damage_curve[areas_within[ID_FIELD]]
        volume_curves = self.vol_curve[areas_within[ID_FIELD]]

        area_curve = pd.DataFrame(data={"volume": volume_curves.sum(axis=1)})
        area_curve["damage"] = damage_curves.sum(axis=1)

        area_curve_int = self._curve_linear_interpolate(area_curve, 0.01)
        agg_curve = area_curve_int[area_curve_int.volume <= total_volume_area]
        agg_curve.index = agg_curve.volume
        agg_curve = agg_curve.drop(columns=["volume"])

        agg_series = agg_curve["damage"]
        agg_series.name = "damage_equal_depth"

        return agg_series

    def agg_rain_own_area_retention(self, areas_within, mm_rain=DEFAULT_RAIN, int_round=1000) -> pd.Series:
        """Compute the rain per drainage level area, retains it in its own
        place.
        1. Get volume damage curves per area
        2. Compute rain volume per area.
        3. Retrieve damage per volume in certain area
        """

        # interpolated volume damage curves
        area_curve_lst = []
        for idx, area in tqdm(areas_within.iterrows(), total=len(areas_within)):
            vol_curve = self.vol_interpolated_curve[area[ID_FIELD]].values.astype(INT)
            dam_curve = self.damage_interpolated_curve[area[ID_FIELD]].values.astype(INT)
            curve = pd.DataFrame(index=vol_curve, data={area[ID_FIELD]: dam_curve})
            curve = curve[~curve.index.duplicated(keep="first")]
            curve_interpolated = self._curve_linear_interpolate(curve, int_round).astype(INT)
            area_curve_lst.append(curve_interpolated)

        area_curve = pd.concat(area_curve_lst, axis=1, sort=True)
        area_curve = area_curve.ffill()  # all last values need to be nan

        area_curve.index = area_curve.index.astype(INT)
        area_curve = area_curve[~area_curve.index.duplicated(keep="first")]
        area_curve = area_curve.loc[:, ~area_curve.columns.duplicated()].copy()

        aggregate_curve = {}
        for i in range(0, mm_rain):
            total_damage = 0
            total_volume = 0
            for idx, area in areas_within.iterrows():
                volume = round(area.geometry.area * (i / 1000))
                area_vol_dam = area_curve[area[ID_FIELD]]
                if volume > area_vol_dam.index[-1]:
                    damage = area_vol_dam.values[-1]
                else:
                    damage = area_vol_dam[round(volume, -3)]  # depends on the int_round value (-3 = 1000, -1= 10)

                if np.isnan(damage):
                    damage = 0

                total_volume += volume
                total_damage += damage

            aggregate_curve[total_volume] = total_damage

        agg_series = pd.Series(aggregate_curve, name="damage_own_area_retention")
        return agg_series

    def create_folium_html(self, feature, areas_within, depth_steps=[0.5, 1, 1.5], damage_steps=[100, 1000, 10000, 100000]):
        """Create interactive Folium map showing damage curves and drainage areas."""
        self.time.log("Creating folium html.")
        
        path = str(self.dir.post_processing.path / feature[self.field])+ ".html" 
        if Path(path).exists():
            self.time.log(f"Folium html exists: {path}, skipping!")
            return 
        
        fol = WSSCurvesFolium()
        fol.add_water_layer()

        fdla_schade = gpd.read_file(self.dir.post_processing.peilgebieden.path, layer="schadecurve")
        fdla_schade = fdla_schade[fdla_schade[ID_FIELD].isin(areas_within[ID_FIELD])]
        
        fdla_geometry = fdla_schade.geometry
        fdla_geometry.index = fdla_schade[ID_FIELD]
        
        agg_schade = gpd.read_file(self.dir.post_processing.aggregatie.path, layer="aggregatie")
        agg_schade[self.field]= ["_".join(i) for i in agg_schade['index'].str.split("_").str[:-1]]
        agg_schade = agg_schade[agg_schade[self.field].isin([feature[self.field]])]

        # Split aggregation data by curve type
        agg_landgebruik = agg_schade[agg_schade["index"].str.contains("landgebruikcurve")]
        agg_schade_curves = agg_schade[agg_schade["index"].str.contains("schadecurve")]
        agg_berging = agg_schade[agg_schade["index"].str.contains("bergingscurve")]
        agg_aggregatie = agg_schade[agg_schade["index"].str.contains("aggregate")]

        # grafieken
        fol.add_graphs(
            "Peilgebieden: Schadecurve (grafiek)",
            fdla_schade,
            "png_path",
            width=600,
            height=600,
        )
        fol.add_graphs(
            "Aggregatie: Landgebruikcurve (grafiek)",
            agg_landgebruik,
            "png_path",
            width=1200,
            height=600,
        )
        fol.add_graphs(
            "Aggregatie: Schadecurve (grafiek)",
            agg_schade_curves,
            "png_path",
            width=1200,
            height=600,
        )
        fol.add_graphs(
            "Aggregatie: Bergingscurve (grafiek)",
            agg_berging,
            "png_path",
            width=800,
            height=800,
        )
        fol.add_graphs(
            "Aggregatie: Aggregatie (grafiek)",
            agg_aggregatie,
            "png_path",
            width=800,
            height=800,
        )

        # per peilgebied
        fol.add_border_layer("Peilgebieden: Peilvakken", fdla_schade, tooltip_fields=["pid"])

        # schade per peilstijging
        for idx, depth_step_data in self.damage_curve.iterrows():
            if idx not in depth_steps:
                continue
            ds_df = gpd.GeoDataFrame(depth_step_data, geometry=fdla_geometry, crs="EPSG:28992")
            fol.add_layer(
                name=f"Peilgebieden: Schadecurve {idx}m peilstijging.",
                gdf=ds_df,
                datacolumn=str(idx),
                tooltip_fields=[str(idx)],
                data_min=int(ds_df[idx].quantile(0.1)),
                data_max=int(ds_df[idx].quantile(0.9)),
                colormap_name="plasma",
                colormap_type="continuous",
                show=False,
                show_colormap=True,
            )

        # peilstijging per schade
        for damage_step in damage_steps:
            mask = self.damage_curve > damage_step
            first_occurrence = mask.apply(lambda x: x[x].index[0] if x.any() else -9999)
            ds_df = gpd.GeoDataFrame(
                {"Peilstijging": first_occurrence},
                geometry=fdla_geometry,
                crs="EPSG:28992",
            )

            # Add to map
            fol.add_layer(
                name=f"Peilgebieden: Peilstijging schade boven €{damage_step:,}",
                gdf=ds_df,
                datacolumn="Peilstijging",  # the first occurrence values
                tooltip_fields=["Peilstijging"],
                data_min=0,
                data_max=ds_df["Peilstijging"].max(),
                colormap_name="plasma",
                colormap_type="continuous",
                show=False,
                show_colormap=True,
            )




        for depth_step in depth_steps:
            series = {}
            for fid in fdla_geometry.index:
                lu_data = self.lu_area_data[self._fid_columns([fid], self.lu_area_data)]
                max_lu = lu_data.loc[depth_step].idxmax().split("_")[0]
                series[fid] = max_lu
            
            lu_max_dmg_per_step = pd.Series(data=series, index=list(series.keys()))

            max_lu = gpd.GeoDataFrame(
                {"Landgebruik (max)": lu_max_dmg_per_step.astype(int)},
                geometry=fdla_geometry,
                crs="EPSG:28992",
            )

            # Create a mapping of landuse IDs to their colors from the conversion table
            lu_colors = self.lu_conversion_table.set_index(self.lu_conversion_table.columns[0])["kleur"].to_dict()
            lu_functie = self.lu_conversion_table.set_index(self.lu_conversion_table.columns[0])[
                "beschrijving"
            ].to_dict()
            max_lu["colors"] = max_lu["Landgebruik (max)"].map(lu_colors)
            max_lu["Functie"] = max_lu["Landgebruik (max)"].map(lu_functie)

            # Add to map with custom colors from landuse conversion table
            fol.add_layer(
                name=f"Peilgebieden: Landgebruik met de meeste schade  {depth_step}m peilstijging",
                gdf=max_lu,
                datacolumn="colors",  # Use the colors field for custom colors
                tooltip_fields=["Landgebruik (max)", "Functie"],
                data_min=None,
                data_max=None,
                colormap_name=None,  # No colormap needed since we use custom colors
                colormap_type="categorical",
                show=False,
                show_colormap=False,
            )

        # per aggregatie gebied
        fol.add_border_layer("Aggregatie: Aggregatiegrenzen", agg_schade, tooltip_fields=["index"])
        fol.add_title("Schadecurves en aggregaties")
        
        path = str(self.dir.post_processing.path / feature[self.field])+ ".html" 
        fol.save(path)
        self.time.log("Creating folium html finished!")

    def create_figures(self):
        """Generate damage curve figures for all drainage areas."""
        self.time.log("Creating basic figures.")
        self.dir.post_processing.create_figure_dir(self.drainage_areas[ID_FIELD].tolist())
        for area_id in tqdm(self.drainage_areas[ID_FIELD], "Create basic figures"):
            path = self.dir.post_processing.figures[f"schadecurve_{area_id}"].path
            if path.exists():
                continue

            data = self.damage_interpolated_curve[area_id]
            figure = CurveFiguur(pd.DataFrame(data))
            figure.run(name=area_id, output_path=path, title="Schadecurve")

            path = self.dir.post_processing.figures[f"bergingscurve_{area_id}"].path
            if path.exists():
                continue

            data = self.vol_interpolated_curve[area_id]
            figure = BergingsCurveFiguur(pd.DataFrame(data), self.drainage_areas[self.drainage_areas[ID_FIELD] == area_id])
            figure.run(name=area_id, output_path=path)

            path = self.dir.post_processing.figures[f"panden_{area_id}"].path
            if path.exists():
                continue

            bu_area_data = self.bu_dmg_data[self._fid_columns([area_id], self.bu_dmg_data)]
            bu_area_data.columns = bu_area_data.columns.str.split("_").str[0]
            bu_area_data = bu_area_data.dropna(axis=1)
            figure = CurveFiguur(pd.DataFrame(bu_area_data))
            figure.run(name=area_id, output_path=path, title="Schade aan panden")
        self.time.log("Creating basic figures finished.")

    def create_fdla_gpkg(self) -> None:
        """Create GeoPackage file containing drainage area geometries with damage curve data."""
        self.time.log("Creating basic FDLA geopackage.")
        local_file = self.dir.post_processing.peilgebieden.path
        if local_file.exists():
            return

        fig_dir = self.dir.post_processing.figures
        ids = self.drainage_areas[ID_FIELD]

        schadecurves = self.drainage_areas.copy()
        schadecurves["image_path"] = ["file:///" + str(fig_dir[f"schadecurve_{i}"].path) for i in ids]
        schadecurves["png_path"] = [str(fig_dir[f"schadecurve_{i}"].path) for i in ids]
        schadecurves.index = schadecurves.pid
        transposed = self.damage_curve.T
        transposed.columns = transposed.columns.astype(str)
        schadecurves = pd.concat([schadecurves, transposed], axis=1)

        local_file = self.dir.post_processing.peilgebieden.path
        schadecurves.to_file(local_file, layer="schadecurve", driver="GPKG")

        # add styling
        styling = self.create_styling(table_names=["schadecurve"])
        gpd.GeoDataFrame(styling).to_file(local_file, layer="layer_styles", driver="GPKG")
        self.time.log("Creating basic FDLA geopackage finished.")

    def create_aggregate_gpkg(self) -> None:
        """Create GeoPackage file containing aggregated damage curves by area."""
        self.time.log("Creating aggregated geopackage.")

        output_file = self.dir.post_processing.aggregatie.path
        if output_file.exists():
            self.time.log("Aggregated geopackage exists! Skipping!")
            return

        ids = self.vector[self.field]
        figure_type = ["aggregate", "bergingscurve", "landgebruikcurve", "schadecurve"]

        data: dict[str, list] = {"image_path": [], "png_path": [], "index": [], "geometry": []}
        for i in ids:
            if not hasattr(self.dir.post_processing, i):
                continue
            geometry = self.vector[self.vector[self.field] == i].geometry.iloc[0]
            if not hasattr(self.dir.post_processing[i], "figures"):
                continue
            agg_figure_dir = self.dir.post_processing[i].figures
            for ftype in figure_type:
                path = getattr(agg_figure_dir, ftype).path
                data["image_path"].append("file:///" + str(path))
                data["png_path"].append(str(path))
                data["index"].append(i + "_" + ftype)
                data["geometry"].append(geometry)

        aggregate_gdf = gpd.GeoDataFrame(
            data,
            geometry=data["geometry"],
            crs=self.vector.crs,
            dtype=str,
            index=data["index"],
        )

        
        aggregate_gdf.to_file(output_file, layer="aggregatie", driver="GPKG")

        # add styling
        styling_df = self.create_styling(table_names=["aggregatie"])
        gpd.GeoDataFrame(styling_df).to_file(output_file, layer="layer_styles", driver="GPKG")
        self.time.log("Creating aggregated geopackage finished!")

    def create_styling(self, table_names) -> pd.DataFrame:
        qmls = [self._get_qml(name) for name in table_names]
        empty_str_list = [""] * len(table_names)

        data = {
            "f_table_catalog": empty_str_list,
            "f_table_schema": empty_str_list,
            "f_table_name": table_names,
            "f_geometry_column": ["geom"] * len(table_names),
            "styleQML": qmls,
            "styleSLD": empty_str_list,
            "useAsDefault": [True] * len(table_names),
            "description": empty_str_list,
            "owner": empty_str_list,
            "ui": empty_str_list,
            "update_time": empty_str_list,
        }
        return pd.DataFrame(data)

    def _get_qml(self, layer_name: str):
        return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.3-Białowieża" styleCategories="MapTips">
  <mapTip>&lt;img src="[% {layer_name} %]" width=600 height=600 ></mapTip>
  <layerGeometryType>2</layerGeometryType>
</qgis>"""

    def create_aggregated_figures(self, agg_dir, name, feature) -> None:
        """

        Parameters
        ----------
        agg_dir : AggregateDir(Folder)
            Aggregate directory from AggregateDir in AreaDamageCurveFolders
        """
        self.time.log("Creating aggregated figures.")
        if not agg_dir.figures.bergingscurve.path.exists():
            bc = BergingsCurveFiguur(agg_dir.agg_volume.path, feature)
            bc.run(output_path=agg_dir.figures.bergingscurve.path, name=name)

        if not agg_dir.figures.landgebruikcurve.path.exists():
            lu_curve = LandgebruikCurveFiguur(agg_dir.agg_landuse.path, agg_dir)
            lu_curve.combine_classes(
                lu_omzetting=self.lu_conversion_table,
                output_path=agg_dir.result_lu_areas_classes.path,
            )

            lu_curve.run(
                lu_omzetting=self.lu_conversion_table,
                name=name,
                output_path=agg_dir.figures.landgebruikcurve.path,
                schadecurve_totaal=True,
            )

        if not agg_dir.figures.schadecurve.path.exists():
            damages = DamagesLuCurveFiguur(agg_dir.agg_landuse_dmg.path, agg_dir)
            damages.combine_classes(
                lu_omzetting=self.lu_conversion_table,
                output_path=agg_dir.result_lu_damages_classes.path,
            )
            damages.run(
                lu_omzetting=self.lu_conversion_table,
                output_path=agg_dir.figures.schadecurve.path,
                name=name,
                schadecurve_totaal=True,
            )
        
        if not agg_dir.figures.aggregate.path.exists():
            damages_agg = DamagesAggFiguur(agg_dir.aggregate.path)
            damages_agg.run(
                output_path=agg_dir.figures.aggregate.path,
                name=name,
            )
        if not agg_dir.figures.panden.path.exists():
            building_damages = BuildingsSchadeFiguur(agg_dir.agg_building_dmg.path, agg_dir)
            building_damages.run(
                output_path=agg_dir.figures.panden.path,
                name=name,
                schadecurve_totaal=True,
                schadebuildings_totaal=True,
            )
        self.time.log("Creating aggregated figures finished!")

    def agg_run(self, feature, areas_within, mm_rain: int = DEFAULT_RAIN) -> dict:
        """Create a dataframe in which methods can be compared"""
        v_lowest = self.aggregate_rain_curve(feature, areas_within, DEFAULT_AGG_METHODS[0], mm_rain)
        v_equal_depth = self.aggregate_rain_curve(feature, areas_within, DEFAULT_AGG_METHODS[1], mm_rain)
        v_equal_rain = self.aggregate_rain_curve(feature, areas_within, DEFAULT_AGG_METHODS[2], mm_rain)

        v_equal_depth = v_equal_depth[~v_equal_depth.index.duplicated()]
        v_lowest = v_lowest[~v_lowest.index.duplicated()]
        v_equal_rain = v_equal_rain[~v_equal_rain.index.duplicated()]

        combined = pd.concat([v_lowest, v_equal_depth, v_equal_rain], axis=1, sort=True)
        combined = combined.interpolate("linear").astype(INT)
        combined.index = combined.index.astype(INT)
        return combined
    
    def run_aggregate_feature(self, feature, areas_within, mm_rain):
        name = feature[self.field]
        agg_dir = self.dir.post_processing.create_aggregate_dir(name)

        if not agg_dir.selection.path.exists():
            self.selection[name].to_file(agg_dir.selection.path)

        if not agg_dir.aggregate.path.exists():
            aggregate = self.agg_run(feature, areas_within, mm_rain)
            aggregate.index.name = "Volume [m3]"
            aggregate = aggregate.add_suffix(" [eur]")
            aggregate.to_csv(agg_dir.aggregate.path)

        if not agg_dir.agg_damage.path.exists():
            agg_damage = self.agg_damage(feature, areas_within)
            agg_damage.index.name = "Peilstijging [m]"
            agg_damage.name = agg_damage.name + " [euro]"
            agg_damage.to_csv(agg_dir.agg_damage.path)

        if not agg_dir.agg_volume.path.exists():
            agg_volume = self.agg_volume(feature, areas_within)
            agg_volume.index.name = "Peilstijging [m]"
            agg_volume.name = agg_volume.name + " [m3]"
            agg_volume.to_csv(agg_dir.agg_volume.path)

        if not agg_dir.agg_landuse.path.exists():
            agg_landuse = self.agg_landuse(feature, areas_within)
            agg_landuse.index.name = "Peilstijging [m]"
            agg_landuse = agg_landuse.add_suffix(" [m2]")
            agg_landuse.to_csv(agg_dir.agg_landuse.path)

        if not agg_dir.agg_landuse_dmg.path.exists():
            agg_landuse_dmg = self.agg_landuse_dmg(feature, areas_within)
            agg_landuse_dmg.index.name = "Peilstijging [m]"
            agg_landuse_dmg = agg_landuse_dmg.add_suffix(" [euro]")
            agg_landuse_dmg.to_csv(agg_dir.agg_landuse_dmg.path)

        if not agg_dir.agg_building_dmg.path.exists():
            agg_building_dmg = self.agg_buildings_dmg(feature, areas_within)
            agg_building_dmg.index.name = "Peilstijging [m]"
            agg_building_dmg = agg_building_dmg.add_suffix(" [euro]")
            agg_building_dmg.to_csv(agg_dir.agg_building_dmg.path)

        if not agg_dir.agg_damage_ha.path.exists():
            agg_damage_ha = self.agg_damage_level_per_ha(feature, areas_within)
            agg_damage_ha.index.name = "Peilstijging [m]"
            agg_damage_ha = agg_damage_ha.add_suffix(" [euro/ha]")
            agg_damage_ha.to_csv(agg_dir.agg_damage_ha.path)

        if not agg_dir.agg_damage_m3.path.exists():
            agg_damage_m3 = self.agg_damage_level_per_m3(feature, areas_within)
            agg_damage_m3.index.name = "Peilstijging [m]"
            agg_damage_m3 = agg_damage_m3.add_suffix(" [euro/m3]")
            agg_damage_m3.to_csv(agg_dir.agg_damage_m3.path)

        agg_dir.create_figure_dir(name)
        self.create_aggregated_figures(agg_dir, name, feature)


    def run(self, aggregation: bool = True, mm_rain: int = DEFAULT_RAIN, create_html: bool = True) -> None:
        #try:
            self.time.log(f"Started run with {mm_rain} mm rain.")
            if not self.dir.post_processing.damage_interpolated_curve.exists():
                self.damage_interpolated_curve.to_csv(self.dir.post_processing.damage_interpolated_curve.path)
            if not self.dir.post_processing.volume_interpolated_curve.exists():
                self.vol_interpolated_curve.to_csv(self.dir.post_processing.volume_interpolated_curve.path)
            if not self.dir.post_processing.damage_level_curve.exists():
                self.damage_level_curve.to_csv(self.dir.post_processing.damage_level_curve.path)
            if not self.dir.post_processing.vol_level_curve.exists():
                self.vol_level_curve.to_csv(self.dir.post_processing.vol_level_curve.path)
            if not self.dir.post_processing.damage_per_m3.exists():
                self.damage_per_m3.to_csv(self.dir.post_processing.damage_per_m3.path)
            if not self.dir.post_processing.damage_level_per_ha.exists():
                self.damage_level_per_ha.to_csv(self.dir.post_processing.damage_level_per_ha.path)

            self.create_figures()
            self.create_fdla_gpkg()

            if aggregation:
                self.time.log("Starting aggregation calculation.")
                for _, feature, areas_within in self:
                    self.time.log(f"Starting aggregation for {feature[self.field]}.")
                    
                    #try:
                    self.run_aggregate_feature(feature, areas_within, mm_rain)
                    #except Exception as e:
                    #    self.time.log(f"Failure for {feature[self.field]}")
                    #    self.time.log(f"Traceback: {e}")
                    
                self.time.log("Aggregation calculations finished!")

            self.create_aggregate_gpkg()
            if create_html:
                for _, feature, areas_within in self:
                    try:
                        self.create_folium_html(feature, areas_within)
                    except Exception as e:
                        self.time.log("Folium html failure")
                        self.time.log(f"Traceback: {e}")  

        #except Exception as e:
        #    self.time.log("Aggregation run failure")
        #    self.time.log(f"Traceback: {e}")           


# %%
if __name__ == "__main__":
    import sys

    adca = AreaDamageCurvesAggregation.from_settings_json(str(sys.argv[1]))
    adca.run(aggregation=True)

# %%
