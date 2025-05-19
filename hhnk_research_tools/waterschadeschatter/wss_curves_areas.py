# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:50:56 2024

@author: kerklaac5395

Methodiek schade, volume en landgebruik
1. Per waterlaag (bv 10 cm) wordt de waterdiepte uitgerekend obv hoogtemodel.
2. Alle unieke waterdiepte en landgebruik pixels worden geteld.
3. Schade = Combinatie opgezocht in de schadetabel en vermenigdvuldigd het aantal pixels.
4. Volume = Oppervlak pixel vermenigvuldigd met de diepte en aantal pixels.

"""
import json
import argparse
import datetime
import multiprocessing as mp
import shutil
import traceback
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

import hhnk_research_tools as hrt
from hhnk_research_tools.rasters.raster_class import Raster
from hhnk_research_tools.variables import DEFAULT_NODATA_VALUES
from hhnk_research_tools.waterschadeschatter.wss_curves_lookup import (
    WaterSchadeSchatterLookUp,
)
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import (
    DRAINAGE_LEVEL_FIELD,
    ID_FIELD,
    AreaDamageCurveFolders,
    WSSTimelog,
    get_drainage_areas,
    pad_zeros,
    write_dict,
    fdla_performance
)

# Globals
DAMAGE_DECIMALS = 2
MAX_PROCESSES = mp.cpu_count() - 1  # still wanna do something on the computa use minus 1
MP_AREA_LIMIT = 1000000  # m2 #10000000
NAME = "WSS AreaDamageCurve"


class AreaDamageCurveMethods:
    def __init__(self, peilgebied_id: int, data: dict, nodamage_filter: bool = True, depth_filter: bool = True):
        self.peilgebied_id = peilgebied_id
        self.dir = AreaDamageCurveFolders(data["output_dir"], create=True)

        self.area_vector = gpd.read_file(self.dir.input.area.path)
        self.lu = hrt.Raster(self.dir.input.lu.path)
        self.dem = hrt.Raster(self.dir.input.dem.path)

        self.lookup_table = data["lookup_table"]
        self.filter_settings = data["filter_settings"]
        self.log_file = data["log_file"]
        self.metadata = data["metadata"]
        self.depth_steps = data["depth_steps"]
        self.nodata = data["nodata"]
        self.run_type = data["run_type"]
        self.nodamage_file = data["nodamage_file"]

        self.dir.work[self.run_type].add_fdla_dirs(self.depth_steps + [self.filter_settings["depth"]])
        self.fdla_dir = self.dir.work[self.run_type][f"fdla_{peilgebied_id}"]

        self.area_gdf = self.area_vector.loc[self.area_vector[ID_FIELD] == peilgebied_id]
        self.area_start_level = self.area_gdf[DRAINAGE_LEVEL_FIELD].iloc[0]
        self.area_meta = hrt.RasterMetadataV2.from_gdf(gdf=self.area_gdf, res=self.metadata.pixel_width)

        self.pixel_width = self.metadata.pixel_width
        time_file = self.dir.work.log.fdla.joinpath(f"time_{self.peilgebied_id}.csv")
        self.time = WSSTimelog(name=self.peilgebied_id, log_file=self.log_file, time_file=time_file)
        
        if depth_filter:
            self.depth_damage_filter(self.filter_settings)
        if nodamage_filter:
            self.nodamage_filter()

    @property
    def geometry(self):
        return list(self.area_gdf.geometry)[0]

    @cached_property
    def nodamage_gdf(self):
        return gpd.read_file(self.nodamage_file)

    @property
    def lu_array(self) -> np.ndarray:
        return self.lu.read_geometry(geometry=self.geometry)

    @property
    def dem_array(self) -> np.ndarray:
        return self.dem.read_geometry(geometry=self.geometry)

    def nodamage_filter(self) -> None:
        """Filter the input polygon based on a nodamage gdf"""
        self.time.log("start nodamage_filter")
        dif = gpd.overlay(self.area_gdf, self.nodamage_gdf, how="difference")
        dif = dif.drop(columns=[i for i in dif.columns if i not in ["geometry"]])
        dif.to_file(self.fdla_dir.nodamage_filtered.path)
        self.area_gdf = dif
        self.time.log("end nodamage_filter")
        
    def depth_damage_filter(self, settings: dict) -> None:
        """Filter the input polygon based on damage at a certain depth."""

        self.time.log("start depth_damage_filter (ddf)")

        lu_select = np.isin(self.lu_array, settings["landuse"])
        self.time.log("ddf: filter specifc landuse finished")

        self.run(run_1d=False, depth_steps=[settings["depth"]])
        filter_path = self.fdla_dir.path / f"damage_{settings['depth']}.tif"
        damage = hrt.Raster(filter_path)
        damage_array = damage.read_geometry(self.geometry)
        self.time.log("ddf: calculate damage at specific depth and filter above zero finished")

        damage_select = damage_array > settings["damage_threshold"]
        self.time.log("ddf: select only places with damage finished")
        
        nodamage_array = lu_select & damage_select
        self.time.log("ddf: nodamage spaces finished")
        
        padded = pad_zeros(a=nodamage_array * 1, shape=damage.shape)
        self.time.log("ddf: boolean, padded zeros towards shape. finished")
        
        no_damage = damage.polygonize(array=padded.astype(int))
        no_damage = no_damage[no_damage.field != 0.0]
        no_damage.crs = self.dem.metadata.projection
        self.time.log("ddf: polygonize (gdf) finished")

        no_damage = no_damage[no_damage.area > settings["area_size"]]
        no_damage.geometry = no_damage.buffer(settings["buffer"])
        self.time.log("ddf: select area size and do buffer finished")

        dif = gpd.overlay(self.area_gdf, no_damage, how="difference")
        dif = dif.drop(columns=[i for i in dif.columns if i not in ["geometry"]])
        dif.to_file(self.fdla_dir.depth_filter.path)
        self.time.log("ddf: extract from input finished")

        self.area_gdf = dif
        damage = None  # close raster
        self.time.log("end depth_damage_filter (ddf)")
        
    def run(self, run_1d: bool = True, depth_steps: Optional[list[float]] = None, write_2d: bool = False) -> Tuple[dict, dict]:
        """
        Actually two functions in one, which is quite ugly.
        However, We try to keep 2d calculation as close
        as the 1d calculation to ensure the possibility of validating 1D
        calculations with 2D calculations. Hence two functions in one.

        Parameters
        ----------
        run_1d: bool, default is True
            True -> Runs the curves in 1d. All spatial sense is lost, but it is
            quicker and less costly then 2D.
            False -> Runs the curves in 2d, which retains all spatial info.
        depth_steps: list[float]
           List of depth steps on which the damage is calculated.
        """
        
        self.time.log("Start run")

        if depth_steps is None:
            depth_steps = self.depth_steps

        run_2d = not run_1d

        lu_array = self.lu_array
        dem_array = self.dem_array.astype(float)
        self.time.log("run: reading rasters finished.")

        if run_1d:
            lu_array = lu_array.flatten()
            dem_array = dem_array.flatten()

        nodata = dem_array == self.dem.nodata

        if run_1d:
            lu_array = lu_array[~nodata]
            dem_array = dem_array[~nodata]

        if run_2d:
            lu_array[nodata] = DEFAULT_NODATA_VALUES["uint8"]
            dem_array[nodata] = np.nan

        dem_array[dem_array < self.area_start_level] = self.area_start_level
        self.time.log("run: pre-processing rasters finished.")

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
                data = data.groupby(["depth", "lu"]).size().reset_index().rename(columns={0: "count"})
                data["lookup"] = data["lu"].astype(str) + "_"+data["depth"].astype(str)
                data['damage'] = data['lookup'].map(self.lookup_table) * data['count']
                data['volume'] = data['depth'] * self.pixel_width**2 * data['count']
                damage_per_lu = dict(data.groupby("lu").damage.sum())
                self.time.log("Finished depth step 1d")

            if run_2d:
                depth[zero_depth_mask] = np.nan
                lu[zero_depth_mask] = DEFAULT_NODATA_VALUES["uint8"]

                data = pd.DataFrame(data={"depth": depth.flatten(), "lu": lu.flatten()})
                data["lookup"] = data["lu"].astype(str) + "_"+data["depth"].astype(str)
                data['damage'] = data['lookup'].map(self.lookup_table)
                data['volume'] = data['depth'] * self.pixel_width**2
                damage_2d = data.damage.values.reshape(depth.shape)
                self.time.log("run: 2D calculations finished")
                
                self.write_tif(path=self.fdla_dir["damage_" + str(ds)].path, array=damage_2d)
                if write_2d:
                    volume_2d = data.volume.values.reshape(depth.shape)
                    self.write_tif(path=self.fdla_dir["volume_" + str(ds)].path, array=volume_2d)
                    self.write_tif(path=self.fdla_dir["depth_" + str(ds)].path, array=depth)
                    self.write_tif(path=self.fdla_dir["lu_" + str(ds)].path, array=lu)
                    self.write_tif(path=self.fdla_dir["level_" + str(ds)].path, array=depth + dem_array)
                    
                damage_per_lu = {}
                self.time.log("run: 2D writing finished")

            curve[ds] = data.damage.sum()
            curve_vol[ds] = data.volume.sum()
            un, c = np.unique(lu, return_counts=True)
            counts_lu[ds] = dict(zip(un, c))
            damage_lu[ds] = damage_per_lu

        timedelta = self.time.time_since_start
        self.time.log(f"Area {self.peilgebied_id} calulation time: {str(timedelta)[:7]}")
        
        curve_df = pd.DataFrame(curve, index=range(0, len(curve)))
        curve_df.to_csv(self.fdla_dir.curve.path)

        curve_vol_df = pd.DataFrame(curve_vol, index=range(0, len(curve_vol)))
        curve_vol_df.to_csv(self.fdla_dir.curve_vol.path)

        counts_lu_df = pd.DataFrame(counts_lu).T
        counts_lu_df.to_csv(self.fdla_dir.counts_lu.path)

        damage_lu_df = pd.DataFrame(damage_lu).T
        damage_lu_df.to_csv(self.fdla_dir.damage_lu.path)

        self.time.log("end run")
        self.time.write()

        return curve, curve_vol

    def write_tif(self, path, array):
        hrt.save_raster_array_to_tiff(
            output_file=path,
            raster_array=array,
            nodata=self.nodata,
            metadata=self.area_meta,
            overwrite=True,
        )


@dataclass
class AreaDamageCurves:
    """
    Creates damage curves for given areas.
    Class works based on cached_properties delete variable to reset.

    Parameters
    ----------
    output_dir : path
        Output directory.
    area_path : str
        Vectorfile for areas to be processed.
    landuse_path_dir : str
        Directory or path to landuse file(s).
    dem_path_dir : str
        Directory or path to dem file(s).
    area_id : str, default is "id"
        Unique id field from area_path.
    area_start_level : str
        Start level field (streefpeil) from area_path.
    curve_step : float, default is 0.1
        Stepsize of damage curves.
    curve_max : float, default is 3
        Maximum depth of damage curves.
    res : float, default is 0.5
        Resolution of rasters default 0.5.
    nodata : int, default is -9999
        Nodata value of rasters
    area_layer_name : str, default is None
        If type is geopackage, a layer name can be given.

    """

    output_dir: Union[str, Path]
    landuse_path_dir: Union[str, Path]
    dem_path_dir: Union[str, Path]
    wss_settings_file: Union[str, Path]
    area_path: Union[str, Path] = None
    area_id: str = "code"
    area_start_level_field: str = "streefpeil_winter"
    curve_step: float = 0.1
    curve_max: float = 3
    resolution: float = 0.5
    nodata: int = -9999  # TODO @Ckerklaan1 gebruik DEFAULT_NODATA_VALUES
    area_layer_name: Optional[str] = None
    nodamage_file: Union[str, Path] = None
    wss_curves_filter_settings_file: Optional[str] = None
    wss_config_file: Optional[str] = None
    settings_json_file: Optional[Path] = None
    database_file: Optional[Path] = None

    def __post_init__(self):
        """Initialise needed things"""
        self.dir = AreaDamageCurveFolders(base=self.output_dir, create=True)

        time_now = datetime.datetime.now().strftime("%Y%m%d%H%M")
        log_file = self.dir.work.log.joinpath(f"log_{time_now}.log")
        time_file = self.dir.work.log.joinpath(f"time_{time_now}.csv")

        self.time = WSSTimelog(name=NAME,log_file=log_file,time_file=time_file)

        # Write settings json
        if self.settings_json_file:
            shutil.copy(self.settings_json_file, self.dir.input.joinpath(f"settings_{time_now}.json"))

        # Create vrts
        self.time.log("Start vrt conversion dem")
        self.dem = self._input_to_vrt(self.dem_path_dir, self.dir.input.dem.path)
        self.time.log("Start vrt conversion lu")
        self.lu = self._input_to_vrt(self.landuse_path_dir, self.dir.input.lu.path)

        # Copy used config to workdir
        if self.wss_config_file:
            shutil.copy(self.wss_config_file, self.dir.input.wss_cfg_settings.path)
        
        self.time.log("Initizalized AreaDamageCurves!")

    def __iter__(self):
        for id in self.area_vector[ID_FIELD]:
            yield id

    def __len__(self):
        return len(self.area_vector)

    @classmethod
    def from_settings_json(cls, settings_json_file: Path):
        """Initialise class from a settings.json, returns an AreaDamageCurves class."""

        with open(str(settings_json_file)) as json_file:
            settings = json.load(json_file)
        return cls(**settings, settings_json_file=settings_json_file)

    @cached_property
    def area_vector(self) -> gpd.GeoDataFrame:
        if self.area_path is not None:
            self.time.log(f"Reading vector from file: {self.area_path}.")
            vector = gpd.read_file(self.area_path, layer=self.area_layer_name, engine="pyogrio")
        else:
            self.time.log(f"Reading vector from database: {self.database_file}")
            vector = get_drainage_areas(self.database_file)
        
        self.time.log("Processing drainage levels")
        keep_col = [self.area_id, "geometry", self.area_start_level_field]
        drop_col = [i for i in vector.columns if i not in keep_col]
        vector = vector.drop(columns=drop_col)
        vector.rename(
            columns={
                self.area_id: ID_FIELD,
                self.area_start_level_field: DRAINAGE_LEVEL_FIELD,
            },
            inplace=True,
        )
        vector = self._check_nan(vector)
        vector.to_file(self.dir.input.area.path)
        self.time.log("Processing drainage levels Finished")
        return vector

    @cached_property
    def wss_settings(self) -> dict:
        with open(str(self.wss_settings_file)) as json_file:
            settings = json.load(json_file)
        write_dict(dictionary=settings, path=self.dir.input.wss_settings.path)

        return {**settings, **{"cfg_file": self.wss_config_file}}

    @cached_property
    def wss_curves_filter_settings(self):
        with open(str(self.wss_curves_filter_settings_file)) as json_file:
            settings = json.load(json_file)

        write_dict(dictionary=settings, path=self.dir.input.wss_curves_filter_settings.path)
        return settings

    @cached_property
    def lookup(self):
        self.time.log("Processing lookup table")
        step = 1 / 10**DAMAGE_DECIMALS
        depth_steps = np.arange(step, self.curve_max + step, step)
        depth_steps = [round(i, 2) for i in depth_steps]
        lookup = WaterSchadeSchatterLookUp(wss_settings=self.wss_settings, depth_steps=depth_steps)
        lookup.run(flatten=True)
        lookup.write_dict(path=self.dir.input.wss_lookup.path)
        self.time.log("Processing lookup table finished!")
        return lookup

    @cached_property
    def metadata(self) -> hrt.RasterMetadataV2:
        return hrt.RasterMetadataV2.from_gdf(gdf=self.area_vector, res=self.resolution)

    @cached_property
    def depth_steps(self) -> list[float]:
        steps = np.arange(self.curve_step, self.curve_max + self.curve_step, self.curve_step)
        return [round(i, 2) for i in steps]

    @property
    def area_data(self) -> dict:
        """This is needed for multiprocessing."""
        data = {}
        data["lu_vrt_path"] = self.dir.input.lu.path
        data["dem_vrt_path"] = self.dir.input.dem.path
        data["run_type"] = self.run_type
        data["lookup_table"] = self.lookup.output
        data["metadata"] = self.metadata
        data["depth_steps"] = self.depth_steps
        data["output_dir"] = str(self.dir.path)
        data["nodata"] = self.nodata
        data["filter_settings"] = self.wss_curves_filter_settings
        data["log_file"] = self.time.log_file
        data["nodamage_file"] = str(self.nodamage_file)
        return data

    def areas_divide(self):
        """Divide areas by size"""
        self.time.log(f"Divide areas by surface area {MP_AREA_LIMIT} squared meters.")
        areas = {"small": [], "large": []}

        vector = self.area_vector.copy()
        vector['area'] = vector.geometry.area
        
        self.time.log(f"Sort by area.")
        vector = vector.sort_values(by=['area'])

        areas['large'] = list(vector.loc[vector.area > MP_AREA_LIMIT][ID_FIELD])
        areas['small'] = list(vector.loc[vector.area <= MP_AREA_LIMIT][ID_FIELD])
        self.time.log(f"Found {len(areas['large'])} large areas over limit!")
        return areas

    def _check_nan(self, gdf) -> gpd.GeoDataFrame:
        """Check for NaN"""
        if gdf[DRAINAGE_LEVEL_FIELD].isna().sum() > 0:
            self.time.log("Found drainage level NAN's, deleting from input.")
            gdf = gdf[~gdf[DRAINAGE_LEVEL_FIELD].isna()]
        return gdf

    def _input_to_vrt(self, path_or_dir, vrt_path) -> Raster:
        _input = Path(path_or_dir)
        if _input.is_dir():
            vrt_input = list(_input.rglob("*.tif"))
        elif _input.is_file():
            vrt_input = [_input]
        else:
            self.time.log("Unrecognized inputs")

        vrt = Raster.build_vrt(vrt_out=vrt_path, input_files=vrt_input, bounds=self.metadata.bbox_gdal, overwrite=True)

        return vrt

    def write(self) -> None:
        self.time.log("Start writing output")
        self.time.log("Writing damage and volume curves")
        self.curve_df = pd.DataFrame.from_dict(self.curve)
        self.curve_df.to_csv(self.dir.output.result.path)

        self.curve_vol_df = pd.DataFrame.from_dict(self.curve_vol)
        self.curve_vol_df.to_csv(self.dir.output.result_vol.path)

        self.time.log("Writing combined land-use files")
        lu_counts = []
        for fid in tqdm(self, "Land-use counts"):
            counts_lu_file = self.dir.work[self.run_type][f"fdla_{fid}"].counts_lu.path
            if not counts_lu_file.exists():
                continue
            curve_lu = pd.read_csv(counts_lu_file, index_col=0)
            curve_lu = curve_lu.fillna(0) * self.resolution**2
            curve_lu["fid"] = str(fid)
            lu_counts.append(curve_lu)
        lu_areas = pd.concat(lu_counts)
        lu_areas.to_csv(self.dir.output.result_lu_areas.path)

        lu_damage: list[pd.DataFrame] = []
        for fid in tqdm(self, "Land-use damage"):
            damage_lu_file = self.dir.work[self.run_type][f"fdla_{fid}"].damage_lu.path
            if not damage_lu_file.exists():
                continue
            damage_lu = pd.read_csv(damage_lu_file, index_col=0)
            damage_lu = damage_lu.fillna(0)
            damage_lu["fid"] = str(fid)
            lu_damage.append(damage_lu)
        lu_damage_df = pd.concat(lu_damage)
        lu_damage_df.to_csv(self.dir.output.result_lu_damage.path)

        mask = self.area_vector[ID_FIELD].isin(self.failures)
        failures = self.area_vector[mask]
        failures.to_file(self.dir.output.failures.path)

        fdla_performance(self.dir.input.area.path, self.dir.work.log.fdla.path)

        self.time.log("End writing")
        self.time.write()

    def run_mp_ordered(self, **kwargs):
        """Optimized multiprocessing run: divides larger and smaller areas."""
        self.time.log("Start optimized multiprocessing run")
        area_ids = self.areas_divide()

        self.time.log("Running small areas with mp.")
        self.run(area_ids["small"], run_1d=True, multiprocessing=True, write=False, **kwargs)

        self.time.log("Running large areas alone.")
        self.run(area_ids["large"], run_1d=True, multiprocessing=False, write=True, **kwargs)

    def run(
        self,
        area_ids: list = None,
        run_1d: bool = False,
        run_2d: bool = False,
        multiprocessing: bool = True,
        processes: Union[int, Literal["max"]] = MAX_PROCESSES,
        nodamage_filter: bool = True,
        depth_filter: bool = True,
        write: bool = True,
    ):
        if area_ids is None:
            area_ids = list(self)

        if run_1d:
            self.run_type = "run_1d"
        elif run_2d:
            self.run_type = "run_2d"
        else:
            raise ValueError("Expected one of [run_1d, run_2d] to be True")

        for pid in area_ids:
            self.dir.work[self.run_type].create_fdla_dir(str(pid), self.depth_steps)

        if processes == "max" or processes == -1:
            processes = MAX_PROCESSES

        self.time.log(f"Starting {self.run_type}!")

        self.curve = {}
        self.curve_vol = {}
        self.failures = []

        if multiprocessing:
            data = self.area_data
            args = [[pid, data, run_1d, nodamage_filter, depth_filter] for pid in area_ids]

            with mp.Pool(processes=processes) as pool:
                output = list(
                    tqdm(
                        pool.imap(area_method_mp, args),
                        total=len(args),
                        desc=f"{NAME}: (MP{processes}) {self.run_type}",
                    )
                )

            for run in output:
                run = list(run)
                if len(run[1]) == 0:
                    self.time.log(f"{run[0]} failure! Traceback {run[2]}")
                    self.failures.append(run[0])
                    run[2] = {}

                self.curve[run[0]] = run[1]
                self.curve_vol[run[0]] = run[2]

        if not multiprocessing:
            for peil_id in tqdm(area_ids, f"{NAME}: Damage {self.run_type}"):
                area_stats = AreaDamageCurveMethods(
                    peilgebied_id=peil_id,
                    data=self.area_data,
                    nodamage_filter=nodamage_filter,
                    depth_filter=depth_filter,
                )
                curve, curve_vol = area_stats.run(run_1d)
                self.curve[peil_id] = curve
                self.curve_vol[peil_id] = curve_vol
        if write:
            self.write()

        self.time.log(f"Ended {self.run_type}")


def area_method_mp(args):
    peilgebied_id = args[0]
    data = args[1]
    run_1d = args[2]
    nodamage_filter = args[3]

    try:
        area_method_1d = AreaDamageCurveMethods(
            peilgebied_id=peilgebied_id,
            data=data,
            nodamage_filter=nodamage_filter,
        )
        curve, curve_vol = area_method_1d.run(run_1d)
        return (peilgebied_id, curve, curve_vol)
    except Exception:
        return (peilgebied_id, {}, str(traceback.format_exc()))

def parse_run_cmd():

    parser = argparse.ArgumentParser(description="Command-line tool met subcommando's.")
    parser.add_argument("-settings", type=str, required=True, help="Settings json")

    parser.add_argument("--run_1d", action=argparse.BooleanOptionalAction, help="Run 1D. Default:True")
    parser.add_argument("--run_2d", action=argparse.BooleanOptionalAction, help="Run 2D. Default:False")
    parser.add_argument("--multiprocessing", action=argparse.BooleanOptionalAction, help="Do multiprocessing. Default: True")
    parser.add_argument("--ordered_mp_run", action=argparse.BooleanOptionalAction, help="Do optimized multiprocessing run (areas are sorted). Default: True")
    parser.add_argument("--dd_filter", action=argparse.BooleanOptionalAction, help="Do depth damage filter. Default: True")
    parser.add_argument("--nd_filter", action=argparse.BooleanOptionalAction, help="Do no damage filter. Default: True")
    parser.add_argument("--processes", type=int, required=False, default=-1, help="Amount of processes. Default: -1 (max)")
    parser.add_argument("--area_ids", type=list, required=False, default=None, help="Subset of area ids. Default: All")

    parser.set_defaults(run_1d=True, run_2d=False, area_ids=None, multiprocessing=True, processes=-1, ordered_mp_run=False, dd_filter=True, nd_filter=True)

    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    adc = hrt.AreaDamageCurves.from_settings_json(Path(str(args.settings)))
    if args.ordered_mp_run:
        adc.run_mp_ordered(depth_filter=args.dd_filter, nodamage_filter=args.nd_filter, processes=args.processes)
    else:
        adc.run(area_ids=args.area_ids,
                run_1d=args.run_1d,
                run_2d=args.run_2d,
                multiprocessing=args.multiprocessing,
                processes=args.processes,
                depth_filter=args.dd_filter,
                nodamage_filter=args.nd_filter)
        
if __name__ == "__main__":
    parse_run_cmd()
