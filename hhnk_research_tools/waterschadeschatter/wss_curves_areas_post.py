# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:27:18 2024

@author: kerklaac5395

TODO:
    4. Log bestand netjes maken

"""
# TODO remove!
import sys

sys.path.append(
    r"C:/Users/kerklaac5395/OneDrive - ARCADIS/Documents/GitHub/hhnk-research-tools"
)


from tqdm import tqdm
import numpy as np
import json
import geopandas as gp
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import (
    WSSTimelog,
    AreaDamageCurveFolders,
    ID_FIELD,
    DRAINAGE_LEVEL_FIELD,
)


BUFFER = 100
NAME = "AreaDamageCurves Aggregation"
NODATA = -9999
RES = 0.01
ROUND = 2
AGG_METHODS = ["lowest_area", "equal_depth", "equal_rain"]
PREDICATE = "_within"


class AreaDamageCurvesAggregation:

    def __init__(
        self, result_path, aggregate_vector_path=None, vector_field=None, quiet=False
    ):

        self.dir = AreaDamageCurveFolders(result_path, create=True)

        if self.dir.output.result.exists():
            self.damage = pd.read_csv(self.dir.output.result.path, index_col=0)
            self.damage.columns = self.damage.columns.astype(int)

        if self.dir.output.result_lu_areas.exists():
            self.lu_area_data = pd.read_csv(
                self.dir.output.result_lu_areas.path, index_col=0
            )

        if self.dir.output.result_vol.exists():
            self.vol = pd.read_csv(self.dir.output.result_vol.path, index_col=0)
            self.vol.columns = self.vol.columns.astype(int)

        self.drainage_areas = self.dir.input.area.load()

        if aggregate_vector_path:
            self.vector = gp.read_file(aggregate_vector_path)
            self.field = vector_field

        self.predicate = PREDICATE
        self.time = WSSTimelog(NAME, quiet, self.dir.post.path)

    def __iter__(self):
        for idx, feature in tqdm(
            self.vector.iterrows(), "WSS Aggregation", total=len(self.vector)
        ):
            predicate_func = getattr(self, self.predicate)
            areas = predicate_func(feature.geometry)
            if len(areas) > 0:
                yield idx, feature, areas

    @property
    def damage_curve(self):
        return self.damage

    @property
    def vol_curve(self):
        return self.vol

    @property
    def damage_interpolated_curve(self):
        if not hasattr(self, "_damage_interpolated_curve"):
            self._damage_interpolated_curve = self._curve_linear_interpolate(
                self.damage, RES
            )
        return self._damage_interpolated_curve

    @property
    def vol_interpolated_curve(self):
        if not hasattr(self, "_vol_interpolated_curve"):
            self._vol_interpolated_curve = self._curve_linear_interpolate(self.vol, RES)
        return self._vol_interpolated_curve

    @property
    def damage_level_curve(self):
        if not hasattr(self, "_damage_level_curve"):
            self._damage_level_curve = self._curves_to_level(self.damage)
        return self._damage_level_curve

    @property
    def vol_level_curve(self):
        if not hasattr(self, "_vol_level_curve"):
            self._vol_level_curve = self._curves_to_level(self.vol)
        return self._vol_level_curve

    @property
    def damage_per_m3(self):
        """Damage per m3 at a certain waterlevel"""
        if not hasattr(self, "_damage_per_m3"):

            data = {}
            for idx, area in self.drainage_areas.iterrows():
                vol_curve = self.vol[area[ID_FIELD]]
                dam_curve = self.damage[area[ID_FIELD]]

                damage_shift = dam_curve - dam_curve.shift(+1)
                volume_shift = vol_curve - vol_curve.shift(+1)
                damage_per_m3 = damage_shift / volume_shift
                damage_per_m3.loc[0.1] = 0
                data[area[ID_FIELD]] = damage_per_m3
            self._damage_per_m3 = pd.DataFrame(data)

        return self._damage_per_m3

    @property
    def selection(self):
        selection = {}
        for idx, feature, areas in self:
            selection[feature[self.field]] = areas
        return selection

    @classmethod
    def from_settings_json(cls, file):
        with open(str(file)) as json_file:
            settings = json.load(json_file)

        return cls(**settings)

    def _within(self, geometry, buffer=BUFFER):
        buffered = geometry.buffer(buffer)
        area_within = self.drainage_areas[self.drainage_areas.geometry.within(buffered)]
        return area_within

    def _curve_linear_interpolate(self, curve, resolution):
        index = np.arange(0, curve.index.values[-1] + resolution, resolution).round(
            ROUND
        )
        new_index = list(set(list(index) + list(curve.index)))
        new_index.sort()
        interpolated = curve.reindex(new_index)
        interpolated.loc[0.0] = 0
        return interpolated.interpolate("linear")

    def _curve_depth_to_level(self, curve, drainage_area):
        curve.index = np.round(curve.index + drainage_area[DRAINAGE_LEVEL_FIELD], ROUND)
        return curve

    def _curves_to_level(self, curves, resolution=RES):
        """Curves of drainage areas from depth to level, also interpolated."""
        d_sorted = self.drainage_areas.sort_values(
            by=DRAINAGE_LEVEL_FIELD, ascending=True
        )

        min_level = d_sorted.iloc[0][DRAINAGE_LEVEL_FIELD]
        max_level = d_sorted.iloc[-1][DRAINAGE_LEVEL_FIELD] + curves.index[-1]

        index = np.arange(min_level, max_level, resolution).round(ROUND)
        level_curve = pd.DataFrame(index=index)

        for idx, d_area in d_sorted.iterrows():
            curve = curves[d_area[ID_FIELD]]
            if pd.isna(curve).all():
                curve = curve.fillna(NODATA)
            curve = curve.astype(int)
            interpolated_curve = self._curve_linear_interpolate(curve, resolution)
            level = self._curve_depth_to_level(interpolated_curve, d_area)
            level_curve[level.name] = level
            level_curve = level_curve.copy()  # to supress defragment warning.

        return level_curve

    def agg_damage(self):
        """Sums damage curves within the given areas."""

        self.agg_sum_curves_output = {}
        for idx, feature, areas_within in self:
            damage_curves = self.damage[areas_within[ID_FIELD]]
            d = pd.Series(damage_curves.sum(axis=1), name=feature[self.field])
            self.agg_sum_curves_output[feature[self.field]] = d

        return self.agg_sum_curves_output

    def agg_volume(self):
        """Sums damage curves within the given areas."""

        self.agg_volume = {}
        for idx, feature, areas_within in self:
            vol_curves = self.vol[areas_within[ID_FIELD]]
            d = pd.Series(vol_curves.sum(axis=1), name=feature[self.field])
            self.agg_volume[feature[self.field]] = d

        return self.agg_volume

    def agg_landuse(self):
        """Sums land use areas data within the given areas."""
        self.agg_lu = {}
        for idx, feature, areas_within in self:
            lu_data = self.lu_area_data[
                self.lu_area_data.fid.isin(areas_within[ID_FIELD])
            ]
            lu_areas_summed = lu_data.groupby(lu_data.index).sum()
            self.agg_lu[feature[self.field]] = lu_areas_summed

        return self.agg_lu

    # def agg_vol_damage_level_curve(self):
    #     level_damage_curve = self.damage_level_curve.ffill()
    #     level_vol_curve = self.vol_level_curve.ffill()

    #     curve = pd.DataFrame(data={"volume":level_vol_curve.sum(axis=1)})
    #     curve['damage']  = level_damage_curve.sum(axis=1)
    #     return curve

    # def agg_vol_damage_curve_per_area(self):

    #     # interpolated volume damage curves
    #     data = []
    #     for idx, area in self.drainage_areas.iterrows():
    #         vol_curve = self.vol[area[ID_FIELD]]
    #         dam_curve = self.damage[area[ID_FIELD]]

    #         vol_curve = self._curve_linear_interpolate(vol_curve, 0.01)
    #         dam_curve = self._curve_linear_interpolate(dam_curve, 0.01)

    #         curve = pd.DataFrame(index=vol_curve.values, data={area[ID_FIELD]: dam_curve.values})
    #         curve = curve[~curve.index.duplicated(keep='first')]
    #         curve_interpolated = self._curve_linear_interpolate(curve, 1000)
    #         data.append(curve_interpolated.astype(int))
    #     curves = pd.concat(data,axis=1,sort=True)
    #     curves = curves.ffill() # all last values need to be nan

    #     curves.index = curves.index.astype(int)
    #     curves = curves[~curves.index.duplicated(keep='first')]

    #     return curves

    def aggregate_rain_curve(self, method="lowest_area"):
        """Methods for distribution of rain in the drainage area"""

        output = {}

        for idx, feature, areas_within in self:

            if method == "lowest_area":
                output[feature[self.field]] = self.agg_rain_lowest_area(
                    feature, areas_within
                )
            elif method == "equal_depth":
                output[feature[self.field]] = self.agg_rain_equal_depth(
                    feature, areas_within
                )
            elif method == "equal_rain":
                output[feature[self.field]] = self.agg_rain_own_area_retention(
                    feature, areas_within
                )
        return output

    def agg_rain_lowest_area(
        self,
        feature,
        areas_within,
        mm_rain=150,
        step_size=0.01,
        field_name="streefpeil",
    ):
        """
        Creates a new damage curve starting at the area with the lowest drainage level.
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

    def agg_rain_equal_depth(self, feature, areas_within, mm_rain=150):
        """
        Creates a new damage curve based on equal depth at every area.
        1. Essentially a sum of all damagecurves.
        2. The curve stops when a total volume is reached.

        """

        total_volume_area = feature.geometry.area * (mm_rain / 1000)  # m3 rain in area.
        damage_curves = self.damage[areas_within[ID_FIELD]]
        volume_curves = self.vol[areas_within[ID_FIELD]]

        area_curve = pd.DataFrame(data={"volume": volume_curves.sum(axis=1)})
        area_curve["damage"] = damage_curves.sum(axis=1)

        area_curve_int = self._curve_linear_interpolate(area_curve, 0.01)
        agg_curve = area_curve_int[area_curve_int.volume <= total_volume_area]
        agg_curve.index = agg_curve.volume
        agg_curve = agg_curve.drop(columns=["volume"])

        agg_series = agg_curve["damage"]
        agg_series.name = "damage_equal_depth"

        return agg_series

    def agg_rain_own_area_retention(self, feature, areas_within, mm_rain=150):
        """Computes the rain per drainage level area, retains it in its own
        place.
        1. Get volume damage curves per area
        2. Compute rain volume per area.
        3. Retrieve damage per volume in certain area

        """

        # interpolated volume damage curves
        data = []
        for idx, area in areas_within.iterrows():
            vol_curve = self.vol[area[ID_FIELD]]
            dam_curve = self.damage[area[ID_FIELD]]

            vol_curve = self._curve_linear_interpolate(vol_curve, 0.01)
            dam_curve = self._curve_linear_interpolate(dam_curve, 0.01)

            curve = pd.DataFrame(
                index=vol_curve.values, data={area[ID_FIELD]: dam_curve.values}
            )
            curve = curve[~curve.index.duplicated(keep="first")]
            curve_interpolated = self._curve_linear_interpolate(curve, 10)
            data.append(curve_interpolated.astype(int))

        area_curve = pd.concat(data, axis=1, sort=True)
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
                    damage = area_vol_dam[round(volume, -1)]

                if np.isnan(damage):
                    damage = 0

                total_volume += volume
                total_damage += damage

            aggregate_curve[total_volume] = total_damage

        agg_series = pd.Series(aggregate_curve, name="damage_own_area_retention")
        return agg_series

    def agg_run(self):
        """Creates a dataframe in which methods can be compared"""
        lowest = self.aggregate_rain_curve(AGG_METHODS[0])
        equal_depth = self.aggregate_rain_curve(AGG_METHODS[1])
        equal_rain = self.aggregate_rain_curve(AGG_METHODS[2])
        output = {}
        for k, v_lowest in lowest.items():
            v_equal_depth = equal_depth[k]
            v_equal_rain = equal_rain[k]

            v_equal_depth = v_equal_depth[~v_equal_depth.index.duplicated()]
            v_lowest = v_lowest[~v_lowest.index.duplicated()]
            v_equal_rain = v_equal_rain[~v_equal_rain.index.duplicated()]

            combined = pd.concat(
                [v_lowest, v_equal_depth, v_equal_rain], axis=1, sort=True
            )
            combined = combined.interpolate("linear").astype(int)
            combined.index = combined.index.astype(int)
            output[k] = combined

        return output

    def run(self, aggregation=True):

        # general data
        self.damage_interpolated_curve.to_csv(
            self.dir.post.damage_interpolated_curve.path
        )
        self.vol_interpolated_curve.to_csv(self.dir.post.volume_interpolated_curve.path)
        self.damage_level_curve.to_csv(self.dir.post.damage_level_curve.path)
        self.vol_level_curve.to_csv(self.dir.post.vol_level_curve.path)
        self.damage_per_m3.to_csv(self.dir.post.damage_per_m3.path)

        if aggregation:
            aggregations = self.agg_run()
            agg_damage = self.agg_damage()
            agg_volume = self.agg_volume()
            agg_landuse = self.agg_landuse()

            for _, feature, _ in self:
                name = feature[self.field]

                path = self.dir.post.path / name
                path.mkdir(exist_ok=True)

                # aggregations
                aggregate = aggregations[name]
                aggregate.index.name = "Volume [m3]"
                aggregate = aggregate.add_suffix(" [eur]")
                aggregate.to_csv(path / "aggregate.csv")

                # selected geometries
                self.selection[name].to_file(path / "selection.gpkg")

                agg_d = agg_damage[name]
                agg_d.index.name = "Waterdepth [m]"
                agg_d.name = agg_d.name + " [euro]"
                agg_d.to_csv(path / "agg_damage.csv")

                agg_v = agg_volume[name]
                agg_v.index.name = "Waterdepth [m]"
                agg_v.name = agg_v.name + " [m3]"
                agg_v.to_csv(path / "agg_volume.csv")

                agg_l = agg_landuse[name]
                agg_l.index.name = "Waterdepth [m]"
                agg_l = agg_l.add_suffix(" [m2]")
                agg_l.to_csv(path / "agg_landuse.csv")

    # def write_figures(
    #     self,
    # ):

    #     self.export = JsonToFigure(self.output_json, self.path)
    #     for peil_id in self:
    #         self.export.run(peil_id)


class JsonToFigure:
    def __init__(self, json_path, output_dir):
        with open(json_path) as json_file:
            self.data = json.load(json_file)

        self.fig_dir = pathlib.Path(output_dir) / "figures"
        self.fig_dir.mkdir(exist_ok=True)

    def run(self, peilgebied_id):
        curve = self.data[str(peilgebied_id)]

        dieptes = [float(i) for i in curve.keys()]
        schades = [float(i) for i in curve.values()]

        plt.figure(figsize=(16, 9))
        plt.plot(np.array(schades) / 1000000, np.array(dieptes) * 100, label="Schade")
        plt.xlabel("Schade [Euro's] (miljoen)")
        plt.ylabel("Waterdiepte boven streefpeil [cm]")
        plt.title(f"Schadecurve voor peilgebied {peilgebied_id}")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.savefig(self.fig_dir / f"{peilgebied_id}.png")


if __name__ == "__main__":
    import sys

    adca = AreaDamageCurvesAggregation.from_settings_json(str(sys.argv[1]))
    adca.run(aggregation=True)

    # adca = AreaDamageCurvesAggregation.from_settings_json(r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\run_settings\run_wss_test_westzaan_aggregate.json")
