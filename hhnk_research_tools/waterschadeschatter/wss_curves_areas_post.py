# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:27:18 2024

@author: kerklaac5395
"""

# First-party imports
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd

# Third-party imports
import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm

import hhnk_research_tools.logger as logging

# Local imports
from hhnk_research_tools.variables import DEFAULT_NODATA_GENERAL
from hhnk_research_tools.waterschadeschatter.wss_curves_figures import (
    BergingsCurveFiguur,
    DamagesLuCurveFiguur,
    LandgebruikCurveFiguur,
    CurveFiguur,
    DamagesAggFiguur
)
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import (
    DRAINAGE_LEVEL_FIELD,
    ID_FIELD,
    AreaDamageCurveFolders,
    # WSSTimelog,
)
from hhnk_research_tools.waterschadeschatter.wss_curves_folium import WSSCurvesFolium

# Logger
logger = logging.get_logger(__name__)

# Globals
NAME = "AreaDamageCurves Aggregation"

# Defaults
DEFAULT_RAIN = 200  # mm
DEFAULT_BUFFER = 100  # m
DEFAULT_RESOLUTION = 0.01  # m
DEFAULT_ROUND = 2
DEFAULT_AGG_METHODS = ["lowest_area", "equal_depth", "equal_rain"]
DEFAULT_PREDICATE = "_within"


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
    landuse_conversion_path: Union[str, Path, None] = None

    def __post_init__(
        self,
    ):
        self.dir = AreaDamageCurveFolders(self.result_path, create=True)

        if self.dir.output.result_lu_areas.exists():
            self.lu_area_data = pd.read_csv(self.dir.output.result_lu_areas.path, index_col=0)

        if self.dir.output.result_lu_damage.exists():
            self.lu_dmg_data = pd.read_csv(self.dir.output.result_lu_damage.path, index_col=0)

        self.drainage_areas = self.dir.input.area.load()

        if self.aggregate_vector_path:
            self.vector = gpd.read_file(
                self.aggregate_vector_path, layer=self.aggregate_vector_layer_name, use_arrow=True
            )
            self.field = self.aggregate_vector_id_field

        if self.landuse_conversion_path:
            self.lu_conversion_table = pd.read_csv(self.landuse_conversion_path)

        self.predicate = DEFAULT_PREDICATE
        # self.time = WSSTimelog(subject=NAME, output_dir=self.dir.post_processing.path)

    @classmethod
    def from_settings_json(cls, settings_json_file):
        with open(str(settings_json_file)) as json_file:
            settings = json.load(json_file)

        return cls(**settings)

    def __iter__(self):
        for idx, feature in tqdm(self.vector.iterrows(), NAME, total=len(self.vector)):
            predicate_func = getattr(self, self.predicate)  # default = self._within
            areas = predicate_func(feature.geometry)
            if len(areas) > 0:
                yield idx, feature, areas

    @cached_property
    def damage_curve(self) -> pd.DataFrame:
        if self.dir.output.result.exists():
            damage = pd.read_csv(self.dir.output.result.path, index_col=0)
            damage.columns = damage.columns
            return damage

    @cached_property
    def vol_curve(self) -> pd.DataFrame:
        if self.dir.output.result_vol.exists():
            vol = pd.read_csv(self.dir.output.result_vol.path, index_col=0)
            vol.columns = vol.columns
            return vol

    @cached_property
    def damage_interpolated_curve(self):
        return self._curve_linear_interpolate(curve=self.damage_curve, resolution=DEFAULT_RESOLUTION)

    @cached_property
    def vol_interpolated_curve(self):
        return self._curve_linear_interpolate(curve=self.vol_curve, resolution=DEFAULT_RESOLUTION)

    @cached_property
    def damage_level_curve(self) -> pd.DataFrame:
        return self._curves_to_level(curves=self.damage_curve)

    @cached_property
    def vol_level_curve(self) -> pd.DataFrame:
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
    
    def _within(self, geometry: shapely.geometry, buffer: int = DEFAULT_BUFFER):
        buffered = geometry.buffer(buffer)
        area_within = self.drainage_areas[self.drainage_areas.geometry.within(buffered)]
        return area_within

    def _curve_linear_interpolate(self, curve: pd.Series, resolution, round: int = DEFAULT_ROUND):
        index = np.arange(0, curve.index.values[-1] + resolution, resolution).round(round)
        new_index = list(set(list(index) + list(curve.index)))
        new_index.sort()
        interpolated = curve.reindex(new_index)
        interpolated.loc[0.0] = 0
        return interpolated.interpolate("linear")

    def _curve_depth_to_level(self, curve: pd.Series, drainage_area: pd.Series, round: int = DEFAULT_ROUND):
        curve.index = np.round(curve.index + drainage_area[DRAINAGE_LEVEL_FIELD], round)
        return curve

    def _curves_to_level(
        self, curves: pd.DataFrame, resolution=DEFAULT_RESOLUTION, round: int = DEFAULT_ROUND
    ) -> pd.DataFrame:
        """Curves of drainage areas from depth to level, also interpolated."""
        d_sorted = self.drainage_areas.sort_values(by=DRAINAGE_LEVEL_FIELD, ascending=True)

        min_level = d_sorted.iloc[0][DRAINAGE_LEVEL_FIELD]
        max_level = d_sorted.iloc[-1][DRAINAGE_LEVEL_FIELD] + curves.index[-1]

        index = np.arange(min_level, max_level, resolution).round(round)
        level_curve = pd.DataFrame(index=index)

        for idx, drainage_area in d_sorted.iterrows():
            curve = curves[drainage_area[ID_FIELD]]
            if pd.isna(curve).all():
                curve = curve.fillna(DEFAULT_NODATA_GENERAL)
            curve = curve.astype(int)
            interpolated_curve = self._curve_linear_interpolate(curve=curve, resolution=resolution)
            level = self._curve_depth_to_level(curve=interpolated_curve, drainage_area=drainage_area)
            level_curve[level.name] = level
            level_curve = level_curve.copy()  # to supress defragment warning.

        return level_curve

    def agg_damage(self) -> dict[str, pd.Series]:
        """Sum damage curves within the given areas."""

        self.agg_sum_curves_output = {}
        for idx, feature, areas_within in self:
            damage_curves = self.damage_curve[areas_within[ID_FIELD]]
            d = pd.Series(damage_curves.sum(axis=1), name=feature[self.field])
            self.agg_sum_curves_output[feature[self.field]] = d

        return self.agg_sum_curves_output

    def agg_volume(self) -> dict[str, pd.Series]:
        """Sum damage curves within the given areas."""

        agg_volume = {}
        for idx, feature, areas_within in self:
            vol_curves = self.vol_curve[areas_within[ID_FIELD]]
            d = pd.Series(vol_curves.sum(axis=1), name=feature[self.field])
            agg_volume[feature[self.field]] = d

        return agg_volume

    def agg_landuse(self) -> dict[str, pd.Series]:
        """Sum land use areas data within the given areas."""
        self.agg_lu = {}
        for idx, feature, areas_within in self:
            lu_data = self.lu_area_data[self.lu_area_data.fid.isin(areas_within[ID_FIELD])]
            lu_areas_summed = lu_data.groupby(lu_data.index).sum()
            self.agg_lu[feature[self.field]] = lu_areas_summed

        return self.agg_lu

    def agg_landuse_dmg(self) -> dict[str, pd.Series]:
        """Sum land use damage data within the given areas."""
        self.agg_lu = {}
        for idx, feature, areas_within in self:
            lu_data = self.lu_dmg_data[self.lu_dmg_data.fid.isin(areas_within[ID_FIELD])]
            lu_areas_summed = lu_data.groupby(lu_data.index).sum()
            self.agg_lu[feature[self.field]] = lu_areas_summed

        return self.agg_lu

    def agg_damage_level_per_ha(self) -> dict[str, pd.DataFrame]:
        """Damage level per ha within the given areas."""	
        self.agg_ha = {}
        for idx, feature, areas_within in self:
            ha_data = self.damage_level_per_ha[areas_within[ID_FIELD]]
            self.agg_ha[feature[self.field]] = ha_data
        return self.agg_ha
    
    def agg_damage_level_per_m3(self) -> dict[str, pd.DataFrame]:
        """Damage level per m3 within the given areas."""
        self.agg_m3 = {}
        for idx, feature, areas_within in self:
            m3_data = self.damage_level_per_ha[areas_within[ID_FIELD]]
            self.agg_m3[feature[self.field]] = m3_data
        return self.agg_m3
    
    def aggregate_rain_curve(self, method="lowest_area", mm_rain=DEFAULT_RAIN) -> dict[str, pd.Series]:
        """Different for distribution of rain in the drainage area"""

        output = {}

        for idx, feature, areas_within in self:
            if method == "lowest_area":
                output[feature[self.field]] = self.agg_rain_lowest_area(feature, areas_within, mm_rain)
            elif method == "equal_depth":
                output[feature[self.field]] = self.agg_rain_equal_depth(feature, areas_within, mm_rain)
            elif method == "equal_rain":
                output[feature[self.field]] = self.agg_rain_own_area_retention(areas_within, mm_rain)
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
        area_curve = []
        for idx, area in tqdm(areas_within.iterrows(), total=len(areas_within)):
            vol_curve = self.vol_interpolated_curve[area[ID_FIELD]].values.astype(int)
            dam_curve = self.damage_interpolated_curve[area[ID_FIELD]].values.astype(int)
            curve = pd.DataFrame(index=vol_curve, data={area[ID_FIELD]: dam_curve})
            curve = curve[~curve.index.duplicated(keep="first")]
            curve_interpolated = self._curve_linear_interpolate(curve, int_round).astype(int)
            area_curve.append(curve_interpolated)

        area_curve = pd.concat(area_curve, axis=1, sort=True)
        area_curve = area_curve.ffill()  # all last values need to be nan

        area_curve.index = area_curve.index.astype(int)
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
    
    def create_folium_html(self, depth_steps=[0.5, 1, 1.5], damage_steps=[100, 1000, 10000, 100000]):

        fol = WSSCurvesFolium()
        fol.add_water_layer()

        fdla_schade = gpd.read_file(self.dir.post_processing.peilgebieden.path, layer="schadecurve")
        agg_schade = gpd.read_file(self.dir.post_processing.aggregatie.path, layer="aggregatie")

        # Split aggregation data by curve type
        agg_landgebruik = agg_schade[agg_schade['index'].str.contains('landgebruikcurve')]
        agg_schade_curves = agg_schade[agg_schade['index'].str.contains('schadecurve')]
        agg_berging = agg_schade[agg_schade['index'].str.contains('bergingscurve')]
        agg_aggregatie = agg_schade[agg_schade['index'].str.contains('aggregatie')]

        # grafieken
        fol.add_graphs("Peilgebieden: Schadecurve (grafiek)", fdla_schade, 'png_path', width=600, height=600)
        fol.add_graphs("Aggregatie: Landgebruikcurve (grafiek)", agg_landgebruik, 'png_path', width=1200, height=600)
        fol.add_graphs("Aggregatie: Schadecurve (grafiek)", agg_schade_curves, 'png_path', width=1200, height=600)
        fol.add_graphs("Aggregatie: Bergingscurve (grafiek)", agg_berging, 'png_path', width=800, height=800)
        fol.add_graphs("Aggregatie: Aggregatie (grafiek)", agg_aggregatie, 'png_path', width=800, height=800)

        # per peilgebied 
        fol.add_border_layer("Peilgebieden: Peilvakken", fdla_schade, tooltip_fields=['pid'])
        
        # schade per peilstijging
        for idx, depth_step_data in self.damage_curve.iterrows():
            if idx not in depth_steps:
                continue
            ds_df = gpd.GeoDataFrame(depth_step_data, geometry = self.fdla_geometry, crs="EPSG:28992")
            fol.add_layer(name=f"Peilgebieden: Schadecurve {idx}m peilstijging.",
                          gdf=ds_df,
                          datacolumn=str(idx),
                          tooltip_fields=[str(idx)],
                          data_min=int(ds_df[idx].quantile(0.1)),
                          data_max=int(ds_df[idx].quantile(0.9)),
                          colormap_name="plasma",
                          colormap_type="continuous",
                          show=False
                          )
            
        # peilstijging per schade
        for damage_step in damage_steps:
            mask = self.damage_curve > damage_step
            first_occurrence = mask.apply(lambda x: x[x].index[0] if x.any() else -9999)
            ds_df = gpd.GeoDataFrame(
                {"Peilstijging":first_occurrence},
                geometry=self.fdla_geometry,
                crs="EPSG:28992"
            )
            
            # Add to map
            fol.add_layer(
                name=f"Peilgebieden: Peilstijging schade boven €{damage_step:,}",
                gdf=ds_df,
                datacolumn='Peilstijging',  # the first occurrence values
                tooltip_fields=['Peilstijging'],
                data_min=0,
                data_max=ds_df['Peilstijging'].max(),
                colormap_name="plasma",
                colormap_type="continuous",
                show=False,
            )

        for depth_step in depth_steps:  
            depth_data = self.lu_dmg_data.loc[depth_step] 
            depth_data.index = depth_data.fid
            depth_data.drop(columns=['fid'], inplace=True)
            depth_data.fillna(DEFAULT_NODATA_GENERAL, inplace=True)
            lu_max_dmg_per_step = depth_data.idxmax(axis=1).astype(float)
            max_lu = gpd.GeoDataFrame({"Landgebruik (max)":lu_max_dmg_per_step},geometry= self.fdla_geometry, crs="EPSG:28992")
            
            # Create a mapping of landuse IDs to their colors from the conversion table
            lu_colors = self.lu_conversion_table.set_index(self.lu_conversion_table.columns[0])['kleur'].to_dict()
            lu_functie = self.lu_conversion_table.set_index(self.lu_conversion_table.columns[0])['beschrijving'].to_dict()
            max_lu['colors'] = max_lu["Landgebruik (max)"].map(lu_colors)
            max_lu['Functie'] = max_lu["Landgebruik (max)"].map(lu_functie)

            # Add to map with custom colors from landuse conversion table
            fol.add_layer(
                name=f"Peilgebieden: Landgebruik met de meeste schade  {depth_step}m peilstijging",
                gdf=max_lu,
                datacolumn='colors',  # Use the colors field for custom colors
                tooltip_fields=['Landgebruik (max)', 'Functie'],
                data_min=None,
                data_max=None,
                colormap_name=None,  # No colormap needed since we use custom colors
                colormap_type="categorical",
                show=False,
                show_colormap=False
            )

        # per aggregatie gebied
        fol.add_border_layer("Aggregatie: Aggregatiegrenzen", agg_schade, tooltip_fields=['index'])
        fol.add_title("Schadecurves en aggregaties")
        fol.save(self.dir.post_processing.schadecurves_html.path)
    
    def create_figures(self):
        """Create figures for the damage and volume c        fol.add_title("Schadecurves en aggregaties")
        fol.save(self.dir.post_processing.schadecurves_html.path)urves per drainage area."""

        self.dir.post_processing.create_figure_dir(self.drainage_areas[ID_FIELD].tolist())
        for area_id in self.drainage_areas[ID_FIELD]:

            data = self.damage_interpolated_curve[area_id]
            figure = CurveFiguur(pd.DataFrame(data))
            path = self.dir.post_processing.figures[f"schadecurve_{area_id}"].path 
            figure.run(name=area_id, output_path=path, title="Schadecurve")
            
            data = self.vol_interpolated_curve[area_id]
            figure = CurveFiguur(pd.DataFrame(data))
            path = self.dir.post_processing.figures[f"bergingscurve_{area_id}"].path 
            figure.run(name=area_id, output_path=path, title="Bergingscurve")

    def create_fdla_gpkg(self) -> None:

        fig_dir = self.dir.post_processing.figures
        ids = self.drainage_areas[ID_FIELD]

        schadecurves = self.drainage_areas.copy()
        schadecurves['image_path'] = ["file:///"+str(fig_dir[f"schadecurve_{i}"].path) for i in ids]
        schadecurves['png_path'] = [str(fig_dir[f"schadecurve_{i}"].path) for i in ids]
        schadecurves.index = schadecurves.pid
        transposed = self.damage_curve.T
        transposed.columns =transposed.columns.astype(str) 
        schadecurves = pd.concat([schadecurves,transposed],axis=1)

        local_file = self.dir.post_processing.peilgebieden.path
        schadecurves.to_file(local_file, layer='schadecurve', driver='GPKG')

        # add styling
        styling = self.create_styling(table_names=["schadecurve"])
        gpd.GeoDataFrame(styling).to_file(local_file, layer="layer_styles", driver='GPKG')

    def create_aggregate_gpkg(self) -> None:
        
        ids = self.vector[self.field]
        figure_type = ['aggregate', 'bergingscurve', 'landgebruikcurve', 'schadecurve']

        data = {"image_path": [], "png_path": [], "index": [], "geometry": []}
        for i in ids:
            geometry = self.vector[self.vector[self.field] == i].geometry.iloc[0]
            agg_figure_dir = self.dir.post_processing[i].figures
            for ftype in figure_type:
                path = getattr(agg_figure_dir, ftype).path
                data['image_path'].append("file:///"+str(path))
                data['png_path'].append(str(path)) 
                data['index'].append(i+"_"+ftype)
                data['geometry'].append(geometry)

        output = gpd.GeoDataFrame(data, geometry=data['geometry'], crs=self.vector.crs, dtype=str, index=data['index'])

        local_file = self.dir.post_processing.aggregatie.path
        output.to_file(local_file, layer='aggregatie', driver='GPKG')

        # add styling
        styling = self.create_styling(table_names=["aggregatie"])
        gpd.GeoDataFrame(styling).to_file(local_file, layer="layer_styles", driver='GPKG')

    def create_styling(self, table_names):
        qmls = [self.get_qml(name)  for name in table_names]
        empty_str_list = [""]*len(table_names)

        data = {"f_table_catalog":empty_str_list,
         "f_table_schema":empty_str_list,
         "f_table_name":table_names,
         "f_geometry_column":["geom"]*len(table_names),   	
         "styleQML":qmls,
         "styleSLD":empty_str_list,
         "useAsDefault":[True]*len(table_names),
         "description":empty_str_list,
         "owner":empty_str_list,
         "ui":empty_str_list,
         "update_time":empty_str_list,
         }
        return pd.DataFrame(data)
        
    def get_qml(self,layer_name):
        QML_STYLE = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.3-Białowieża" styleCategories="MapTips">
  <mapTip>&lt;img src="[% {layer_name} %]" width=600 height=600 ></mapTip>
  <layerGeometryType>2</layerGeometryType>
</qgis>"""
        return QML_STYLE

    def create_aggregated_figures(self, agg_dir, name, feature) -> None:
        """

        Parameters
        ----------
        agg_dir : AggregateDir(Folder)
            Aggregate directory from AggregateDir in AreaDamageCurveFolders
        """

        bc = BergingsCurveFiguur(path=agg_dir.agg_volume.path, feature=feature)
        bc.run(output_path=agg_dir.figures.bergingscurve.path, name=name)

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

        damages = DamagesLuCurveFiguur(agg_dir.agg_landuse_dmg.path, agg_dir)
        damages.combine_classes(
            lu_omzetting=self.lu_conversion_table,
            output_path=agg_dir.result_lu_damages_classes.path,
        )
        damages.run(lu_omzetting=self.lu_conversion_table,
            output_path=agg_dir.figures.schadecurve.path,
            name=name, schadecurve_totaal=True)
        
        damages_agg = DamagesAggFiguur(agg_dir.aggregate.path)
        damages_agg.run(output_path=agg_dir.figures.aggregate.path, 
            name=name,
        )

    def agg_run(self, mm_rain=DEFAULT_RAIN) -> dict:
        """Create a dataframe in which methods can be compared"""
        lowest = self.aggregate_rain_curve(DEFAULT_AGG_METHODS[0], mm_rain)
        equal_depth = self.aggregate_rain_curve(DEFAULT_AGG_METHODS[1], mm_rain)
        equal_rain = self.aggregate_rain_curve(DEFAULT_AGG_METHODS[2], mm_rain)
        output = {}
        for k, v_lowest in lowest.items():
            v_equal_depth = equal_depth[k]
            v_equal_rain = equal_rain[k]

            v_equal_depth = v_equal_depth[~v_equal_depth.index.duplicated()]
            v_lowest = v_lowest[~v_lowest.index.duplicated()]
            v_equal_rain = v_equal_rain[~v_equal_rain.index.duplicated()]

            combined = pd.concat([v_lowest, v_equal_depth, v_equal_rain], axis=1, sort=True)
            combined = combined.interpolate("linear").astype(int)
            combined.index = combined.index.astype(int)
            output[k] = combined

        return output

    def run(self, aggregation=True, mm_rain=DEFAULT_RAIN) -> None:
        # general data
        self.damage_interpolated_curve.to_csv(self.dir.post_processing.damage_interpolated_curve.path)
        self.vol_interpolated_curve.to_csv(self.dir.post_processing.volume_interpolated_curve.path)
        self.damage_level_curve.to_csv(self.dir.post_processing.damage_level_curve.path)
        self.vol_level_curve.to_csv(self.dir.post_processing.vol_level_curve.path)
        self.damage_per_m3.to_csv(self.dir.post_processing.damage_per_m3.path)
        self.damage_level_per_ha.to_csv(self.dir.post_processing.damage_level_per_ha.path)

        self.create_figures()
        self.create_fdla_gpkg()

        if aggregation:
            aggregations = self.agg_run(mm_rain)
            agg_damage = self.agg_damage()
            agg_volume = self.agg_volume()
            agg_landuse = self.agg_landuse()
            agg_landuse_dmg = self.agg_landuse_dmg()
            agg_damage_ha = self.agg_damage_level_per_ha()
            agg_damage_m3 = self.agg_damage_level_per_m3()

            for _, feature, _ in self:
                name = feature[self.field]
                agg_dir = self.dir.post_processing.create_aggregate_dir(name)

                # aggregations
                aggregate = aggregations[name]
                aggregate.index.name = "Volume [m3]"
                aggregate = aggregate.add_suffix(" [eur]")
                aggregate.to_csv(agg_dir.aggregate.path)

                # selected geometries
                self.selection[name].to_file(agg_dir.selection.path)

                agg_d = agg_damage[name]
                agg_d.index.name = "Peilstijging [m]"
                agg_d.name = agg_d.name + " [euro]"
                agg_d.to_csv(agg_dir.agg_damage.path)

                agg_v = agg_volume[name]
                agg_v.index.name = "Peilstijging [m]"
                agg_v.name = agg_v.name + " [m3]"
                agg_v.to_csv(agg_dir.agg_volume.path)

                agg_l = agg_landuse[name]
                agg_l.index.name = "Peilstijging [m]"
                agg_l = agg_l.add_suffix(" [m2]")
                agg_l.to_csv(agg_dir.agg_landuse.path)

                agg_ld = agg_landuse_dmg[name]
                agg_ld.index.name = "Peilstijging [m]"
                agg_ld = agg_ld.add_suffix(" [euro]")
                agg_ld.to_csv(agg_dir.agg_landuse_dmg.path)

                agg_d_ha = agg_damage_ha[name]
                agg_d_ha.index.name = "Peilstijging [m]"
                agg_d_ha = agg_d_ha.add_suffix(" [euro/ha]")
                agg_d_ha.to_csv(agg_dir.agg_damage_ha.path)
                
                agg_d_m3 = agg_damage_m3[name]
                agg_d_m3.index.name = "Peilstijging [m]"
                agg_d_m3 = agg_d_m3.add_suffix(" [euro/m3]")
                agg_d_m3.to_csv(agg_dir.agg_damage_m3.path)
                
                agg_dir.create_figure_dir(name)
                self.create_aggregated_figures(agg_dir, name, feature)

        self.create_aggregate_gpkg()


        self.create_folium_html()


if __name__ == "__main__":
    import sys

    adca = AreaDamageCurvesAggregation.from_settings_json(str(sys.argv[1]))
    adca.run(aggregation=True)
