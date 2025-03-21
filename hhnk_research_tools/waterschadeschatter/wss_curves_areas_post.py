# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:27:18 2024

@author: kerklaac5395

Post-processing:
    - Vertaald de data naar geografische subgebieden.
    - Doet ook een korte analyse van het landgebruik.
    
    
"""
import sys

sys.path.append(
    r"C:\Users\kerklaac5395\OneDrive - ARCADIS\Documents\GitHub\hydrologen-projecten\schadeberekeningen"
)
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

from hhnk_research_tools.waterschadeschatter.wss_curves_utils import WSSTimelog

OUTPUT_DIR = pathlib.Path(
    r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\output"
)
BUFFER = 100
NAME = "AreaDamageCurves Aggregation"
NODATA = -9999

class AreaDamageCurvesAggregation:

    def __init__(self, result_path, aggregate_vector_path=None, vector_field=None, quiet=False):
        self.path = pathlib.Path(result_path)

        if (self.path / "run_1d_mp.csv").exists():
            self.damage = pd.read_csv(self.path / "run_1d_mp.csv", index_col=0)
            self.damage.columns = self.damage.columns.astype(int)
            self.json = self.path / "run_1d_mp.json"

        if (self.path / "run_1d_mp_lu_areas.csv").exists():
            self.lu_area_data = pd.read_csv(self.path / "run_1d_mp_lu_areas.csv", index_col=0)
            
        if (self.path / "run_1d_mp_vol.csv").exists():
            self.vol = pd.read_csv(self.path / "run_1d_mp_vol.csv", index_col=0)
            self.vol.columns = self.vol.columns.astype(int)

        self.drainage_areas = gp.read_file(result_path / "areas.gpkg")
        self.field_id = "peil_id"
        self.field_name = "streefpeil"
        
        self.output_dir = self.path / "postprocessing"
        self.output_dir.mkdir(exist_ok=True)
        if aggregate_vector_path:
            self.vector = gp.read_file(aggregate_vector_path)
            self.field = vector_field
        
        self.predicate = "within"

        self.time = WSSTimelog(NAME, quiet, self.output_dir)

    
    def __iter__(self):
        for idx, feature in tqdm(self.vector.iterrows(), "WSS Aggregation", total=len(self.vector)):
            predicate_func = getattr(self, self.predicate)
            areas = predicate_func(feature.geometry)
            yield idx, feature, areas
               
    @property
    def damage_curve(self):
        return self.damage
    
    @property
    def vol_curve(self):
        return self.vol
    
    @property
    def damage_level_curve(self):
        return self._curves_to_level("damage")
    
    @property
    def vol_level_curve(self):
        return self._curves_to_level("vol")

        
    
    def within(self, geometry, buffer=BUFFER):
        buffered = geometry.buffer(buffer)
        area_within = self.drainage_areas[self.drainage_areas.geometry.within(buffered)]
        return area_within

    def within_areas(self):
        self.selection = {}
        for idx, feature, areas in self:
            self.selection[feature[self.field]] = areas
        return self.selection
    
    def _curve_linear_interpolate(self, curve, resolution):
        index=np.arange(0, curve.index.values[-1]+resolution,resolution).round(2)
        new_index = list(set(list(index) + list(curve.index)))
        new_index.sort()
        interpolated = curve.reindex(new_index)
        interpolated.loc[0.0] = 0
        interpolated = interpolated.interpolate("linear")
            
        return interpolated
    
    def _curve_depth_to_level(self, curve, drainage_area):
        curve.index = np.round(curve.index + drainage_area[self.field_name],2)
        return curve
    
    def _curves_to_level(self, curve_type="vol", resolution=0.01):
        """ Curves of drainage areas from depth to level, also interpolated."""
        
        if curve_type == "vol":
            curves = self.vol
        elif curve_type == "damage":
            curves = self.damage
            
        d_sorted = self.drainage_areas.sort_values(by=self.field_name, ascending=True)
    
        min_level = d_sorted.iloc[0][self.field_name]
        max_level = d_sorted.iloc[-1][self.field_name] + curves.index[-1]
        
        index = np.arange(min_level, max_level,resolution).round(2)
        level_curve = pd.DataFrame(index=index)
        data = {}
        
        for idx, d_area in d_sorted.iterrows():
            curve = curves[d_area[self.field_id]]
            if pd.isna(curve).all():
                curve = curve.fillna(NODATA)
            curve = curve.astype(int)
            interpolated_curve = self._curve_linear_interpolate(curve, resolution)
            level = self._curve_depth_to_level(interpolated_curve, d_area)
            data[d_area.peil_id] = level
            
        level_curve = pd.DataFrame(index=index, data=data)
            # level_curve[d_area.peil_id] = level
        return level_curve
    
    def agg_select(self):
        """ selects features within the given areas"""
        self.predicate = "within"
        self.agg_select_output = {}
        for idx, feature, area_within in self:
            damage = self.damage[area_within[self.field_id]]
            self.agg_select_output[feature[self.field_id]] = damage
        return self.agg_select_output
        
    def agg_damage(self):
        """Sums damage curves within the given areas."""
        
        self.agg_sum_curves_output = {}
        for idx, feature, areas_within in self:
            damage_curves = self.damage[areas_within[self.field_id]]
            self.agg_sum_curves_output[idx] = damage_curves.sum(axis=1)
        
        return self.agg_sum_curves_output
    
    def agg_volume(self):
        """Sums damage curves within the given areas."""
        
        self.agg_volume = {}
        for idx, feature, areas_within in self:
            vol_curves = self.vol[areas_within[self.field_id]]
            self.agg_volume[idx] = vol_curves.sum(axis=1)
        
        return self.agg_volume
    
    def agg_landuse(self):
        """  Sums land use areas data within the given areas."""
        self.agg_lu = {}
        for idx, feature, areas_within in self:
            lu_data = self.lu_area_data[self.lu_area_data.fid.isin(areas_within.peil_id)]
            lu_areas_summed = lu_data.groupby(lu_data.index).sum()
            self.agg_lu[feature[self.field]] = lu_areas_summed
            
        return self.agg_lu
    
                
    def agg_vol_damage_level_curve(self):
        level_damage_curve = self.damage_level_curve.ffill()
        level_vol_curve = self.vol_level_curve.ffill()
        
        curve = pd.DataFrame(data={"volume":level_vol_curve.sum(axis=1)})
        curve['damage']  = level_damage_curve.sum(axis=1)
        return curve
    
    def agg_vol_damage_level_curve_per_area(self):
        
        # interpolated volume damage curves 
        data = []
        for idx, area in self.drainage_areas.iterrows():
            vol_curve = self.vol[area[self.field_id]]
            dam_curve = self.damage[area[self.field_id]]
            
            vol_curve = self._curve_linear_interpolate(vol_curve, 0.01)
            dam_curve = self._curve_linear_interpolate(dam_curve, 0.01)

            curve = pd.DataFrame(index=vol_curve.values, data={area.peil_id: dam_curve.values})
            curve = curve[~curve.index.duplicated(keep='first')]
            curve_interpolated = self._curve_linear_interpolate(curve, 1000)
            data.append(curve_interpolated.astype(int))
        curves = pd.concat(data,axis=1,sort=True)
        curves = curves.ffill() # all last values need to be nan
        

        curves.index = curves.index.astype(int)        
        curves = curves[~curves.index.duplicated(keep='first')]
        
        return curves
    
    def aggregate_rain_curve(self, method="lowest_area"):
        """Methods for distribution of rain in the drainage area"""
        
        output = {}
        
        for idx, feature, areas_within in self:
    
            if method == "lowest_area":
                output[feature[self.field]] = self.agg_rain_lowest_area(feature, areas_within)
            elif method == "equal_depth":
                output[feature[self.field]] = self.agg_rain_equal_depth(feature, areas_within)
            elif method == "equal_rain":
                output[feature[self.field]] = self.agg_rain_own_area_retention(feature, areas_within)
        return output
    
    def agg_rain_lowest_area(self, feature, areas_within, mm_rain=150, step_size=0.01, field_name="streefpeil"):
        """ 
        Creates a new damage curve starting at the area with the lowest drainage level.
        1. Rain falls in the lowest areas, so damage curve is taken from that area.
        2. If the drainage level of the second area is reached, the damagecurves of the first and second area summed.
        3. This happens until total volume of the rain is stored.
        
        Returns a curve based on volume
        """
        total_volume_area = feature.geometry.area*(mm_rain/1000) # m3 rain in area.
        level_damage_curve= self.damage_level_curve[areas_within[self.field_id]]
        level_vol_curve= self.vol_level_curve[areas_within[self.field_id]]
        level_damage_curve = level_damage_curve.ffill()
        level_vol_curve = level_vol_curve.ffill()
        
        area_curve = pd.DataFrame(data={"volume":level_vol_curve.sum(axis=1)})
        area_curve['damage']  = level_damage_curve.sum(axis=1)
        agg_curve = area_curve[area_curve.volume<=total_volume_area]
        agg_curve.index  = agg_curve.volume
        agg_curve = agg_curve.drop(columns=['volume'])        
        
        agg_series = agg_curve['damage']
        agg_series.name = "damage_lowest_area"
        
        return agg_series
    
    def agg_rain_equal_depth(self, feature, areas_within, mm_rain=150):
        """  
        Creates a new damage curve based on equal depth at every area.
        1. Essentially a sum of all damagecurves.
        2. The curve stops when a total volume is reached.
        
        """

        total_volume_area = feature.geometry.area*(mm_rain/1000) # m3 rain in area.
        damage_curves = self.damage[areas_within[self.field_id]]
        volume_curves = self.vol[areas_within[self.field_id]]
        
        area_curve = pd.DataFrame(data={"volume":volume_curves.sum(axis=1)})
        area_curve['damage'] = damage_curves.sum(axis=1)

        area_curve_int = self._curve_linear_interpolate(area_curve,0.01)
        agg_curve = area_curve_int[area_curve_int.volume<=total_volume_area]
        agg_curve.index  = agg_curve.volume
        agg_curve = agg_curve.drop(columns=['volume'])
        
        agg_series = agg_curve['damage']
        agg_series.name = "damage_equal_depth"
        
        
        return agg_series
    
    def agg_rain_own_area_retention(self, feature, areas_within, mm_rain=150):
        """ Computes the rain per drainage level area, retains it in its own 
            place.
            1. Get volume damage curves per area
            2. Compute rain volume per area.
            3. Retrieve damage per volume in certain area
            
        """    
        
        # interpolated volume damage curves 
        data = []
        for idx, area in areas_within.iterrows():
            vol_curve = self.vol[area[self.field_id]]
            dam_curve = self.damage[area[self.field_id]]
            
            vol_curve = self._curve_linear_interpolate(vol_curve, 0.01)
            dam_curve = self._curve_linear_interpolate(dam_curve, 0.01)

            curve = pd.DataFrame(index=vol_curve.values, data={area.peil_id: dam_curve.values})
            curve = curve[~curve.index.duplicated(keep='first')]
            curve_interpolated = self._curve_linear_interpolate(curve, 10)
            data.append(curve_interpolated.astype(int))
            
        area_curve = pd.concat(data,axis=1,sort=True)
        area_curve = area_curve.ffill() # all last values need to be nan

        area_curve.index = area_curve.index.astype(int)        
        area_curve = area_curve[~area_curve.index.duplicated(keep='first')]
        area_curve = area_curve.loc[:,~area_curve.columns.duplicated()].copy()

        aggregate_curve  = {}
        for i in range(0,mm_rain):
            total_damage = 0
            total_volume = 0
            for idx, area in areas_within.iterrows():
                volume = round(area.geometry.area*(i/1000))
                area_vol_dam = area_curve[area.peil_id]
                if volume > area_vol_dam.index[-1]:
                    damage = area_vol_dam.values[-1]
                else:    
                    damage = area_vol_dam[round(volume,-1)]
                
                if np.isnan(damage):
                    damage = 0
                
                total_volume += volume
                total_damage += damage
                
            aggregate_curve[total_volume] = total_damage
        
        agg_series = pd.Series(aggregate_curve, name= "damage_own_area_retention")
        return agg_series
    
    def agg_run(self):
        methods=  ["lowest_area","equal_depth", "equal_rain"]
        lowest = self.aggregate_rain_curve(methods[0])
        equal_depth = self.aggregate_rain_curve(methods[1])
        equal_rain = self.aggregate_rain_curve(methods[2])
        output = {}
        for k, v_lowest in lowest.items():
            v_equal_depth = equal_depth[k]
            v_equal_rain = equal_rain[k]

            v_equal_depth = v_equal_depth[~v_equal_depth.index.duplicated()]
            v_lowest = v_lowest[~v_lowest.index.duplicated()]
            v_equal_rain = v_equal_rain[~v_equal_rain.index.duplicated()]
            
            combined = pd.concat([v_lowest, v_equal_depth, v_equal_rain],axis=1,sort=True)
            combined = combined.interpolate("linear").astype(int)
            combined.index = combined.index.astype(int)
            output[k] = combined
            
        return output       
        
    def write_select(self, name="1"):
        sel_path =  self.output_dir / ("select_" + name)
        sel_path.mkdir(exist_ok=True)
        
        with pd.ExcelWriter(sel_path / f"select_{name}.xlsx") as writer:
            for k, v in self.select_output.items():    
                v.to_excel(writer, sheet_name=k)
        self.write_selected_geometries(sel_path, name)
        
    def write_aggregate(self, output, name="1"):
        agg_path =  self.output_dir / ("agg_" + name)
        agg_path.mkdir(exist_ok=True)
        
        # with pd.ExcelWriter(agg_path / f"aggregate_{name}.xlsx") as writer:
        #     self.output_sum_per_step.to_excel(writer, sheet_name="output_sum_per_step")
        #     self.found_ids.to_excel(writer, sheet_name="id_mapping")
        
        with pd.ExcelWriter(agg_path / f"aggregate_areas_{name}.xlsx") as writer:
            for k, v in output.items():
                v.to_excel(writer, sheet_name=k)
            
        # self.write_selected_geometries(agg_path, name)

    def write_selected_geometries(self, path,name):
        self.selection = self.within_areas()
        first_key = list(self.selection)[0]      
        first_gdf = self.selection[first_key]
        file_path = path /f"areas_{name}.gpkg"
        file_path.unlink(missing_ok=True)
        first_gdf.to_file(file_path, layer=first_key, driver='GPKG')
        del self.selection[first_key]
        for layer_name, data in self.selection.items():
            data.to_file(file_path, layer=layer_name, driver='GPKG')
            

    def write_figures(
        self,
    ):

        self.export = JsonToFigure(self.output_json, self.path)
        for peil_id in self:
            self.export.run(peil_id)

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
    polders_path = r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\data\vectors/polder_heiloo.shp"

    t = AreaDamageCurvesAggregation(OUTPUT_DIR / "test5", polders_path, "name")

    self = t
    mm=150
    
    for idx, feature, areas_within in self:
        break
    
        total_volume_area = feature.geometry.area*(mm/1000) # m3 rain in area.
        level_damage_curve= self.damage_level_curve
        level_vol_curve= self.vol_level_curve
        
    # test.aggregate(polders_path)
    