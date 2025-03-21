# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:50:56 2024

@author: kerklaac5395

#TODO:
    - Geef input settings terug bij wegschrijven.
"""
import json
import pathlib
import traceback
import pandas as pd
import numpy as np
import multiprocessing as mp

import geopandas as gp
from tqdm import tqdm

import hhnk_research_tools as hrt
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import WSSTimelog, pad_zeros, write_dict
from hhnk_research_tools.waterschadeschatter.wss_curves_lookup import WaterSchadeSchatterLookUp

# Globals
NODATA_UINT8 = 255
DAMAGE_DECIMALS = 2
MAX_PROCESSES = (
    mp.cpu_count() - 2
)  # still wanna do something on the computa use minus 2
SHOW_LOG = 30  # seconds
NAME = "WSS AreaDamageCurve"


class AreaDamageCurveMethods:
    def __init__(self, peilgebied_id, data, nodamage_filter=True):
        self.peilgebied_id = peilgebied_id
        self.lu = hrt.Raster(data["lu_vrt_path"])
        self.dem = hrt.Raster(data["dem_vrt_path"])
        self.area_start_level = data["area_start_level"]
        self.area_vector = gp.read_file(
            data["area_path"], layer=data["area_layer_name"]
        )
        self.area_id = data["area_id"]
        self.lookup_table = data["lookup_table"]
        self.metadata = data["metadata"]
        self.depth_steps = data["depth_steps"]
        self.output_dir = pathlib.Path(data["output_dir"])
        self.nodata = data["nodata"]
        self.filter_settings = data["filter_settings"]
        self.log_file = data["log_file"]

        self.area_dir = self.output_dir / str(peilgebied_id)
        self.area_dir.mkdir(exist_ok=True)
        self.area_gdf, self.area_meta, self.area_start_level = self.get_area_meta(
            self.peilgebied_id
        )
        self.pixel_width = self.metadata.pixel_width
        self.time = WSSTimelog(None, True, None, log_file=self.log_file)
        if nodamage_filter:
            self.damage_filter(self.filter_settings)

    @property
    def geometry(self):
        return list(self.area_gdf.geometry)[0]

    def get_area_meta(self, peilgebied_id):
        area = self.area_vector.loc[self.area_vector[self.area_id] == peilgebied_id]
        area_meta = hrt.RasterMetadataV2.from_gdf(
            gdf=area, res=self.metadata.pixel_width
        )
        start_level = area[self.area_start_level].iloc[0]

        return area, area_meta, start_level

    def damage_filter(self, settings):

        lu_array = self.lu.read(self.geometry)

        # filter specifc landuse
        lu_select = np.isin(lu_array, settings["landuse"])

        # calculate damage at specific depth and filter above zero
        self.run(run_1d=False, depth_steps=[settings["depth"]])
        damage = hrt.Raster(self.area_dir / f"damage_{settings['depth']}.tif")
        damage_array = damage.read(self.geometry)

        # select only places with damage
        damage_select = damage_array > settings["damage_threshold"]

        # nodamage spaces
        nodamage_array = lu_select & damage_select

        # boolean, padded zeros towards shape.
        padded = pad_zeros(nodamage_array * 1, damage.shape)

        # polygonize
        no_damage = damage.polygonize(array=padded.astype(int))
        no_damage = no_damage[no_damage.field != 0.0]
        no_damage.crs = self.dem.metadata.projection

        # select area size and do buffer
        no_damage = no_damage[no_damage.area > settings["area_size"]]
        no_damage.geometry = no_damage.buffer(settings["buffer"])

        # extract from input
        gdf = gp.overlay(self.area_gdf, no_damage, how="difference")
        gdf = gdf.drop(columns=[i for i in gdf.columns if i not in ["geometry"]])
        gdf.to_file(self.area_dir / "nodamage_filtered.gpkg")

        self.area_gdf = gdf
        damage = None # close raster

    def run(self, run_1d=True, depth_steps=None):
        """
        Actually two functions in one, which is quite ugly.
        However, We try to keep 2d calculation as close
        as the 1d calculation to ensure the possibility of validating 1D
        calculations with 2D calculations. Hence two functions in one.
        """

        if depth_steps is None:
            depth_steps = self.depth_steps

        run_2d = not run_1d

        lu_array = self.lu.read(self.geometry)
        dem_array = self.dem.read(self.geometry).astype(float)

        if run_1d:
            lu_array = lu_array.flatten()
            dem_array = dem_array.flatten()

        nodata = dem_array == self.dem.nodata

        if run_1d:
            lu_array = lu_array[~nodata]
            dem_array = dem_array[~nodata]

        if run_2d:
            lu_array[nodata] = NODATA_UINT8
            dem_array[nodata] = np.nan

        dem_array[dem_array < self.area_start_level] = self.area_start_level

        curve = {}
        curve_vol = {}
        counts_lu = {}
        damage_lu = {}

        for ds in depth_steps:
            depth = ((self.area_start_level + ds) - dem_array).round(DAMAGE_DECIMALS)
            zero_depth_mask = depth <= 0
            lu = np.copy(lu_array)  # TODO is this really needed?

            if run_1d:
                depth = depth[~zero_depth_mask]
                lu = lu[~zero_depth_mask]

                data = pd.DataFrame(data={"depth": depth, "lu": lu})
                unique_counts = (
                    data.groupby(["depth", "lu"])
                    .size()
                    .reset_index()
                    .rename(columns={0: "count"})
                )

                volume = 0
                damage = 0
                damage_per_lu = {lu_id: 0 for lu_id in set(unique_counts.lu)}
                for idx, row in unique_counts.iterrows():
                    row_damage = self.lookup_table[row.depth][row.lu] * row["count"]
                    damage += row_damage
                    volume += (
                        self.pixel_width * self.pixel_width * row.depth * row["count"]
                    )
                    damage_per_lu[row.lu] += row_damage

            if run_2d:
                depth[zero_depth_mask] = np.nan
                lu[zero_depth_mask] = NODATA_UINT8

                # actual slow calc, but best to vis output.
                stacked = np.vstack((depth.flatten(), lu.flatten()))
                damage_1d = np.zeros(stacked.shape[1])
                volume_1d = np.zeros(stacked.shape[1])
                for i in range(stacked.shape[1]):
                    d = stacked[0][i]
                    l = stacked[1][i]
                    if np.isnan(d):
                        continue

                    damage_1d[i] = self.lookup_table[d][l]
                    volume_1d[i] = d * self.pixel_width * self.pixel_width

                damage_2d = damage_1d.reshape(depth.shape)
                damage = damage_2d.sum()

                volume_2d = volume_1d.reshape(depth.shape)
                volume = volume_2d.sum()

                damage_per_lu = {}
                self.write_tif(self.area_dir / f"depth_{ds}.tif", depth)
                self.write_tif(self.area_dir / f"lu_{ds}.tif", lu)
                self.write_tif(self.area_dir / f"damage_{ds}.tif", damage_2d)
                self.write_tif(self.area_dir / f"level_{ds}.tif", depth + dem_array)

            curve[ds] = damage
            curve_vol[ds] = volume
            un, c = np.unique(lu, return_counts=True)
            counts_lu[ds] = dict(zip(un, c))
            damage_lu[ds] = damage_per_lu

        timedelta = self.time.time_since_start
        if timedelta.total_seconds() > SHOW_LOG:
            self.time._message(
                f"Area {self.peilgebied_id} calulation time is >30s: {str(timedelta)[:7]}"
            )

        curve_df = pd.DataFrame(curve,index=range(0, len(curve)))
        curve_df.to_csv(self.area_dir / "curve.csv")

        curve_vol_df = pd.DataFrame(curve_vol,index=range(0, len(curve_vol)))
        curve_vol_df.to_csv(self.area_dir / "curve_vol.csv")

        counts_lu = pd.DataFrame(counts_lu).T
        counts_lu.to_csv(self.area_dir / "counts_lu.csv")

        damage_lu = pd.DataFrame(damage_lu).T
        damage_lu.to_csv(self.area_dir / "damage_lu.csv")

        return curve, curve_vol

    def write_tif(self, path, array):
        hrt.save_raster_array_to_tiff(
            path, array, self.nodata, self.area_meta, overwrite=True
        )


class AreaDamageCurves:
    """
    This object creates damage curves for given areas.
    
    Params:
        area_path:str, Vectorfile for areas to be processed.
        landuse_path_dir:str, Directory or path to landuse file(s).
        dem_path_dir:str, Directory or path to dem file(s).
        output_dir:str, Output directory.
        area_id="id", Unique id field from area_path.
        area_start_level:str = None, Start level field (streefpeil) from area_path.
        curve_step:float=0.1, Stepsize of damage curves.
        curve_max:float=3, Maximum depth of damage curves.
        res:float=0.5, Resolution of rasters default 0.5.
        nodata:int=-9999, Nodata value of rasters default -9999.
        quiet:bool=False, Show progress if false.
        area_layer_name:str=None, if type is geopackage, a layer name can be given.
        
    """

    def __init__(
        self,
        area_path:str,
        landuse_path_dir:str,
        dem_path_dir:str,
        output_dir:str,
        area_id="id",
        area_start_level:str = None,
        curve_step:float=0.1,
        curve_max:float=3,
        res:float=0.5,
        nodata:int=-9999,
        quiet:bool=False,
        area_layer_name:str=None,
        wss_filter_settings=None,
        wss_config=None,
        wss_settings=None
    ):
        
        
        self._wss_settings = {}
        self._wss_curves_filter_settings = None
        self._wss_config = None
        
        self.quiet = quiet
        self.area_path = area_path
        self.landuse_path_dir = landuse_path_dir
        self.dem_path_dir = dem_path_dir      
        self.area_layer_name = area_layer_name
        self.area_id = area_id
        self.area_start_level = area_start_level
        self.res = res
        self.nodata = nodata
        self.curve_step = curve_step
        self.curve_max = curve_max
        self.wss_filter_settings = wss_filter_settings
        self.wss_config = wss_config
        self.wss_settings = wss_settings
        
        self.output_dir = pathlib.Path(output_dir)
        
        self.time = WSSTimelog(NAME, self.quiet, self.output_dir)
        
        self.area_vector = gp.read_file(self.area_path, layer=area_layer_name)
        self.metadata = hrt.RasterMetadataV2.from_gdf(gdf=self.area_vector, res=res)
        self.bbox_gdal = self.metadata.bbox_gdal

        self.depth_steps = np.arange(curve_step, curve_max + curve_step, curve_step)
        self.depth_steps = [round(i, 2) for i in self.depth_steps]

        self.vrt_dir = self.output_dir / "vrt"
        self.vrt_dir.mkdir(exist_ok=True, parents=True)
        self.lu_vrt_path = self.vrt_dir / "lu.vrt"
        self.dem_vrt_path = self.vrt_dir / "dem.vrt"
        self._inputs_to_vrt()

    def __iter__(self):
        for id in self.area_vector[self.area_id]:
            yield id

    def __len__(self):
        return len(self.area_vector)

    def __repr__(self):
        return f"""AreaDamageCures\n\n\tWSS settings: {self.wss_settings}\n\n\tFilter settings: {self.wss_curves_filter_settings}"""        

    @classmethod
    def from_settings_json(cls, file):
        with open(str(file)) as json_file:
            settings = json.load(json_file)
            
        return cls(**settings)

    @property
    def wss_settings(self):
        return {**self._wss_settings, **{'cfg_file':self.wss_config}}

    @wss_settings.setter
    def wss_settings(self, value):
        if value is None:
            self._wss_settings = None
        else:
            with open(str(value)) as json_file:
                self._wss_settings = json.load(json_file)

        if hasattr(self, "_lookup"):
            del self._lookup  # reset the lookup table

    @property
    def wss_config(self):
        return self._wss_config

    @wss_config.setter
    def wss_config(self, value):
        self._wss_config = value
        
    @property
    def wss_curves_filter_settings(self):
        return self._wss_curves_filter_settings

    @wss_curves_filter_settings.setter
    def wss_curves_filter_settings(self, value):
        if value is None:
            self._wss_curves_filter_settings = None
        else:
            with open(str(value)) as json_file:
                self._wss_curves_filter_settings = json.load(json_file)

    @property
    def lookup(self):  # lookup is always based on 2 decimals
        if not hasattr(self, "_lookup"):
            step = 1 / 10**DAMAGE_DECIMALS
            depth_steps = np.arange(step, self.curve_max + step, step)
            depth_steps = [round(i, 2) for i in depth_steps]
            self._lookup = WaterSchadeSchatterLookUp(self.wss_settings, depth_steps)
        return self._lookup

    @property
    def area_stats_data(self):
        data = {}
        data["lu_vrt_path"] = self.lu_vrt_path
        data["dem_vrt_path"] = self.dem_vrt_path
        data["area_start_level"] = self.area_start_level
        data["area_path"] = self.area_path
        data["area_layer_name"] = self.area_layer_name
        data["area_id"] = self.area_id
        data["lookup_table"] = self.lookup.output
        data["metadata"] = self.metadata
        data["depth_steps"] = self.depth_steps
        data["output_dir"] = str(self.output_dir_area)
        data["nodata"] = self.nodata
        data["filter_settings"] = self.wss_curves_filter_settings
        data["log_file"] = self.time.log_file
        return data
    

    def _inputs_to_vrt(self):
        """creates vrt rasters out of all input"""
        self.time._message("Start vrt conversion land use and dem")
        self.lu = self._input_to_vrt(self.landuse_path_dir, self.lu_vrt_path)
        self.dem = self._input_to_vrt(self.dem_path_dir, self.dem_vrt_path)
        self.time._message("Ended vrt conversion land use and dem")

    def _input_to_vrt(self, path_or_dir, vrt_path):
        _input = pathlib.Path(path_or_dir)
        if _input.is_dir():
            vrt_input = list(_input.rglob("*.tif"))
        elif _input.is_file():
            vrt_input = [str(_input)]
        else:
            print("Unrecognized inputs")

        return hrt.Raster.build_vrt(
            vrt_path, vrt_input, bounds=self.bbox_gdal, overwrite=True
        )

    def get_area_meta(self, peilgebied_id):
        area = self.area_vector.loc[self.area_vector[self.area_id] == peilgebied_id]
        area_meta = hrt.RasterMetadataV2.from_gdf(
            gdf=area, res=self.metadata.pixel_width
        )
        start_level = area[self.area_start_level].iloc[0]

        return area, area_meta, start_level

    def write(self):

        self.time._message("Start writing")

        output_csv = str(self.output_dir_area) + ".csv"
        output_vol_csv = str(self.output_dir_area) + "_vol.csv"
        output_lu_areas_csv = str(self.output_dir_area) + "_lu_areas.csv"
        output_lu_damage_csv = str(self.output_dir_area) + "_lu_damage.csv"
        output_json = str(self.output_dir_area / self.output_dir_area) + ".json"

        write_dict(self.curve, output_json)

        self.time._message("Writing curves")
        self.curve_df = pd.DataFrame.from_dict(self.curve)
        self.curve_df.to_csv(output_csv)
        self.curve_vol_df = pd.DataFrame.from_dict(self.curve_vol)
        self.curve_vol_df.to_csv(output_vol_csv)

        self.time._message("Writing combined land-use files")
    
        lu_counts = []
        for fid in tqdm(self, "Land-use counts"):
            counts_lu_file = self.output_dir_area / str(fid) / "counts_lu.csv"
            if not counts_lu_file.exists():
                continue
            curve_lu = pd.read_csv(counts_lu_file, index_col=0)
            curve_lu = curve_lu.fillna(0) * self.res * self.res
            curve_lu["fid"] = str(fid)
            lu_counts.append(curve_lu)
        lu_areas = pd.concat(lu_counts)
        lu_areas.to_csv(output_lu_areas_csv)
        
        lu_damage = []
        for fid in tqdm(self, "Land-use damage"):
            damage_lu_file = self.output_dir_area / str(fid) / "damage_lu.csv"
            if not damage_lu_file.exists():
                continue
            damage_lu = pd.read_csv(damage_lu_file, index_col=0)
            damage_lu = damage_lu.fillna(0)
            damage_lu["fid"] = str(fid)
            lu_damage.append(damage_lu)
        lu_damage = pd.concat(lu_damage)
        lu_damage.to_csv(output_lu_damage_csv)
        
        drop_col = [
            i for i in self.area_vector.columns if i not in [self.area_id, 
                                                             "geometry",
                                                             self.area_start_level]
        ]
        vector = self.area_vector.drop(columns=drop_col)
        vector = vector.rename({self.area_id: "pid"})
        vector = vector.rename({self.area_start_level: "drainage_level"})
        
        vector.to_file(self.output_dir / "areas.gpkg")

        self.time._message("End writing")

    def run(
        self,
        run_1d=False,
        run_2d=False,
        multiprocessing=True,
        processes=MAX_PROCESSES,
        nodamage_filter=True,
    ):
        self.quiet = True

        if run_1d:
            run_name = "run_1d"
        if run_2d:
            run_name = "run_2d"
        if multiprocessing:
            run_name += "_mp"

        if processes == "max":
            processes = MAX_PROCESSES

        self.time._message(f"Starting {run_name}!")

        self.output_dir_area = self.output_dir / run_name
        self.output_dir_area.mkdir(exist_ok=True, parents=True)
        self.curve = {}
        self.curve_vol = {}

        if multiprocessing:
            data = self.area_stats_data
            args = [[pid, data, run_1d, nodamage_filter] for pid in self]

            with mp.Pool(processes=processes) as pool:
                output = list(
                    tqdm(
                        pool.imap(area_method_mp, args),
                        total=len(args),
                        desc=f"{NAME}: (MP{processes}) {run_name}",
                    )
                )

            for run in output:
                run = list(run)
                if len(run[1]) == 0:            
                    self.time._message(f"{run[0]} failure! Traceback {run[2]}")
                    run[2] = {}
                    
                self.curve[run[0]] = run[1]
                self.curve_vol[run[0]] = run[2]

        if not multiprocessing:
            for peil_id in tqdm(self, f"{NAME}: Damage {run_name}"):
                area_stats = AreaDamageCurveMethods(peil_id, self.area_stats_data, nodamage_filter)
                curve, curve_vol = area_stats.run(run_1d)
                self.curve[peil_id] = curve
                self.curve_vol[peil_id] = curve_vol

        self.write()
        self.quiet = False
        self.time._message(f"Ended {run_name}")



def area_method_mp(args):
    peilgebied_id = args[0]
    data = args[1]
    run_1d = args[2]
    nodamage_filter = args[3]

    try:
        area_method_1d = AreaDamageCurveMethods(peilgebied_id, data, nodamage_filter)
        curve, curve_vol = area_method_1d.run(run_1d)
        return (peilgebied_id, curve, curve_vol)
    except Exception:
        # print("Multiprocessing failure for area:", peilgebied_id)
        
        return (peilgebied_id, {}, str(traceback.format_exc()))

