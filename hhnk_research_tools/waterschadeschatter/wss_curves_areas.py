# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:50:56 2024

@author: kerklaac5395

#TODO:
    - Geef input settings terug bij wegschrijven.

Methodiek schade, volume en landgebruik
1. Per waterlaag (bv 10 cm) wordt de waterdiepte uitgerekend obv hoogtemodel.
2. Alle unieke waterdiepte en landgebruik pixels worden geteld.
3. Schade = Combinatie opgezocht in de schadetabel en vermenigdvuldigd het aantal pixels.
4. Volume = Oppervlak pixel vermenigvuldigd met de diepte en aantal pixels.
"""

import functools
import json
import multiprocessing as mp
import pathlib
import shutil
import traceback
from dataclasses import dataclass
from typing import Union

import geopandas as gp
import numpy as np
import pandas as pd
from tqdm import tqdm

import hhnk_research_tools as hrt

# logging
import hhnk_research_tools.logger as logging
from hhnk_research_tools.variables import DEFAULT_NODATA_VALUES
from hhnk_research_tools.waterschadeschatter.wss_curves_lookup import (
    WaterSchadeSchatterLookUp,
)
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import (
    DRAINAGE_LEVEL_FIELD,
    ID_FIELD,
    AreaDamageCurveFolders,
    WSSTimelog,
    pad_zeros,
    write_dict,
)

# Logger
logger = logging.get_logger(__name__)

# Globals
DAMAGE_DECIMALS = 2
MAX_PROCESSES = mp.cpu_count() - 1  # still wanna do something on the computa use minus 2
SHOW_LOG = 30  # seconds
NAME = "WSS AreaDamageCurve"

# Defaults
DEFAULT_AREA_ID = "id"
DEFAULT_AREA_START_LEVEL_FIELD = "streefpeil"
DEFAULT_QUIET = False


class AreaDamageCurveMethods:
    def __init__(self, peilgebied_id: int, data: dict, nodamage_filter: bool = True):
        self.peilgebied_id = peilgebied_id
        self.dir = AreaDamageCurveFolders(data["output_dir"])

        self.area_vector = gp.read_file(self.dir.input.area.path)
        self.lu = hrt.Raster(self.dir.input.lu.path)
        self.dem = hrt.Raster(self.dir.input.dem.path)

        self.lookup_table = data["lookup_table"]
        self.filter_settings = data["filter_settings"]
        self.log_file = data["log_file"]
        self.metadata = data["metadata"]
        self.depth_steps = data["depth_steps"]
        self.nodata = data["nodata"]
        self.run_type = data["run_type"]

        self.dir.work[self.run_type].add_fdla_dirs(self.depth_steps + [self.filter_settings["depth"]])
        self.fdla_dir = self.dir.work[self.run_type][f"fdla_{peilgebied_id}"]

        self.area_gdf = self.area_vector.loc[self.area_vector[ID_FIELD] == peilgebied_id]
        self.area_start_level = self.area_gdf[DRAINAGE_LEVEL_FIELD].iloc[0]
        self.area_meta = hrt.RasterMetadataV2.from_gdf(gdf=self.area_gdf, res=self.metadata.pixel_width)

        self.pixel_width = self.metadata.pixel_width
        self.time = WSSTimelog("Multi", True, None, log_file=self.log_file)
        if nodamage_filter:
            self.damage_filter(self.filter_settings)

    @property
    def geometry(self):
        return list(self.area_gdf.geometry)[0]

    @property
    def lu_array(self):
        return self.lu.read_geometry(self.geometry)

    @property
    def dem_array(self):
        return self.dem.read_geometry(self.geometry)

    def damage_filter(self, settings: dict):
        # filter specifc landuse
        lu_select = np.isin(self.lu_array, settings["landuse"])

        # calculate damage at specific depth and filter above zero
        self.run(run_1d=False, depth_steps=[settings["depth"]])
        filter_path = self.fdla_dir.path / f"damage_{settings['depth']}.tif"
        damage = hrt.Raster(filter_path)
        damage_array = damage.read_geometry(self.geometry)

        # select only places with damage
        damage_select = damage_array > settings["damage_threshold"]

        # nodamage spaces
        nodamage_array = lu_select & damage_select

        # boolean, padded zeros towards shape.
        padded = pad_zeros(nodamage_array * 1, damage.shape)

        # polygonize (gdf)
        no_damage = damage.polygonize(array=padded.astype(int))
        no_damage = no_damage[no_damage.field != 0.0]
        no_damage.crs = self.dem.metadata.projection

        # select area size and do buffer
        no_damage = no_damage[no_damage.area > settings["area_size"]]
        no_damage.geometry = no_damage.buffer(settings["buffer"])

        # extract from input
        dif = gp.overlay(self.area_gdf, no_damage, how="difference")
        dif = dif.drop(columns=[i for i in dif.columns if i not in ["geometry"]])
        dif.to_file(self.fdla_dir.nodamage_filtered.path)

        self.area_gdf = dif
        damage = None  # close raster

    def run(self, run_1d: bool = True, depth_steps: list = None):
        """
        run_1d: True: Runs the curves in 1d. All spatial sense is lost, but it is
        quicker then 2D.
        run_1d: False: Runs the curves in 2d, which retains all spatial info.
        depth_steps: List of floats on which the damage is calculated.

        Actually two functions in one, which is quite ugly.
        However, We try to keep 2d calculation as close
        as the 1d calculation to ensure the possibility of validating 1D
        calculations with 2D calculations. Hence two functions in one.
        """

        if depth_steps is None:
            depth_steps = self.depth_steps

        run_2d = not run_1d

        lu_array = self.lu_array
        dem_array = self.dem_array.astype(float)

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
                unique_counts = data.groupby(["depth", "lu"]).size().reset_index().rename(columns={0: "count"})

                volume = 0
                damage = 0
                damage_per_lu = {lu_id: 0 for lu_id in set(unique_counts.lu)}
                for idx, row in unique_counts.iterrows():
                    row_damage = self.lookup_table[row.depth][row.lu] * row["count"]
                    damage += row_damage
                    volume += self.pixel_width * self.pixel_width * row.depth * row["count"]
                    damage_per_lu[row.lu] += row_damage

            if run_2d:
                depth[zero_depth_mask] = np.nan
                lu[zero_depth_mask] = DEFAULT_NODATA_VALUES["uint8"]

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
                    volume_1d[i] = d * self.pixel_width**2

                damage_2d = damage_1d.reshape(depth.shape)
                damage = damage_2d.sum()

                volume_2d = volume_1d.reshape(depth.shape)
                volume = volume_2d.sum()

                damage_per_lu = {}
                self.write_tif(self.fdla_dir["depth_" + str(ds)].path, depth)
                self.write_tif(self.fdla_dir["lu_" + str(ds)].path, lu)
                self.write_tif(self.fdla_dir["damage_" + str(ds)].path, damage_2d)
                self.write_tif(self.fdla_dir["level_" + str(ds)].path, depth + dem_array)

            curve[ds] = damage
            curve_vol[ds] = volume
            un, c = np.unique(lu, return_counts=True)
            counts_lu[ds] = dict(zip(un, c))
            damage_lu[ds] = damage_per_lu

        timedelta = self.time.time_since_start
        if timedelta.total_seconds() > SHOW_LOG:
            logger.info(f"Area {self.peilgebied_id} calulation time is >30s: {str(timedelta)[:7]}")

        curve_df = pd.DataFrame(curve, index=range(0, len(curve)))
        curve_df.to_csv(self.fdla_dir.curve.path)

        curve_vol_df = pd.DataFrame(curve_vol, index=range(0, len(curve_vol)))
        curve_vol_df.to_csv(self.fdla_dir.curve_vol.path)

        counts_lu = pd.DataFrame(counts_lu).T
        counts_lu.to_csv(self.fdla_dir.counts_lu.path)

        damage_lu = pd.DataFrame(damage_lu).T
        damage_lu.to_csv(self.fdla_dir.damage_lu.path)

        return curve, curve_vol

    def write_tif(self, path, array):
        hrt.save_raster_array_to_tiff(path, array, self.nodata, self.area_meta, overwrite=True)


@dataclass
class AreaDamageCurves:
    """
    Creates damage curves for given areas.
    Class works based on cached_properties delete variable to reset.

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

    area_path: Union[str, pathlib.Path]
    landuse_path_dir: Union[str, pathlib.Path]
    dem_path_dir: Union[str, pathlib.Path]
    output_path: Union[str, pathlib.Path]
    area_id: str = DEFAULT_AREA_ID
    area_start_level_field: str = DEFAULT_AREA_START_LEVEL_FIELD
    curve_step: float = 0.1
    curve_max: float = 3
    resolution: float = 0.5
    nodata: int = -9999  # TODO @Ckerklaan1 gebruik DEFAULT_NODATA_VALUES
    quiet: bool = DEFAULT_QUIET
    area_layer_name: str = None
    wss_curves_filter_settings_file: str = None
    wss_config_file: str = None
    wss_settings_file: str = None
    settings_json_file: str = None

    def __post_init__(self):
        """initializes needed things"""
        self.dir = AreaDamageCurveFolders(self.output_path, create=True)
        self.time = WSSTimelog(NAME, self.quiet, self.dir.work.path)
        self._write_settings_json()
        self._create_input_vrt()

    def __iter__(self):
        for id in self.area_vector[ID_FIELD]:
            yield id

    def __len__(self):
        return len(self.area_vector)

    @classmethod
    def from_settings_json(cls, file: Union[str, pathlib.Path]):
        """Initialise class from a settings.json, returns an AreaDamageCurves class."""

        with open(str(file)) as json_file:
            settings = json.load(json_file)
        return cls(**settings, settings_json_file=file)

    @functools.cached_property
    def output_dir(self):
        return self.output_path

    @functools.cached_property
    def area_vector(self):
        vector = gp.read_file(self.area_path, layer=self.area_layer_name, engine="pyogrio")
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
        return vector

    @functools.cached_property
    def wss_settings(self):
        with open(str(self.wss_settings_file)) as json_file:
            settings = json.load(json_file)
        write_dict(settings, self.dir.input.wss_settings.path)

        return {**settings, **{"cfg_file": self.wss_config}}

    @functools.cached_property
    def wss_config(self):
        shutil.copy(self.wss_config_file, self.dir.input.wss_cfg_settings.path)
        return self.wss_config_file

    @functools.cached_property
    def wss_curves_filter_settings(self):
        with open(str(self.wss_curves_filter_settings_file)) as json_file:
            settings = json.load(json_file)

        write_dict(settings, self.dir.input.wss_curves_filter_settings.path)
        return settings

    @functools.cached_property
    def lookup(self):
        step = 1 / 10**DAMAGE_DECIMALS
        depth_steps = np.arange(step, self.curve_max + step, step)
        depth_steps = [round(i, 2) for i in depth_steps]
        _lookup = WaterSchadeSchatterLookUp(self.wss_settings, depth_steps)
        _lookup.run()
        _lookup.write_dict(self.dir.input.wss_lookup.path)
        return _lookup

    @functools.cached_property
    def lu(self):
        logger.info("Start vrt conversion lu")
        return self._input_to_vrt(self.landuse_path_dir, self.dir.input.lu.path)

    @functools.cached_property
    def dem(self):
        logger.info("Start vrt conversion dem")
        return self._input_to_vrt(self.dem_path_dir, self.dir.input.dem.path)

    @functools.cached_property
    def metadata(self):
        return hrt.RasterMetadataV2.from_gdf(gdf=self.area_vector, res=self.resolution)

    @functools.cached_property
    def depth_steps(self):
        steps = np.arange(self.curve_step, self.curve_max + self.curve_step, self.curve_step)
        return [round(i, 2) for i in steps]

    @property
    def area_data(self):
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
        return data

    def _check_nan(self, gdf):
        """Checks for NAN's"""
        if gdf[DRAINAGE_LEVEL_FIELD].isna().sum() > 0:
            logger.info("Found drainage level NAN's, deleting from input.")
            gdf = gdf[~gdf[DRAINAGE_LEVEL_FIELD].isna()]
        return gdf

    def _input_to_vrt(self, path_or_dir, vrt_path):
        _input = pathlib.Path(path_or_dir)
        if _input.is_dir():
            vrt_input = list(_input.rglob("*.tif"))
        elif _input.is_file():
            vrt_input = [str(_input)]
        else:
            logger.info("Unrecognized inputs")

        return hrt.Raster.build_vrt(vrt_path, vrt_input, bounds=self.metadata.bbox_gdal, overwrite=True)

    def _create_input_vrt(self):
        self.dem
        self.lu

    def _write_settings_json(self):
        if self.settings_json_file:
            date = self.time.start_time.strftime("%Y%m%d%H%M")
            shutil.copy(self.settings_json_file, self.dir.input.path / f"settings_{date}.json")

    def write(self):
        logger.info("Start writing output")
        logger.info("Writing damage and volume curves")
        self.curve_df = pd.DataFrame.from_dict(self.curve)
        self.curve_df.to_csv(self.dir.output.result.path)

        self.curve_vol_df = pd.DataFrame.from_dict(self.curve_vol)
        self.curve_vol_df.to_csv(self.dir.output.result_vol.path)

        logger.info("Writing combined land-use files")
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

        lu_damage = []
        for fid in tqdm(self, "Land-use damage"):
            damage_lu_file = self.dir.work[self.run_type][f"fdla_{fid}"].damage_lu.path
            if not damage_lu_file.exists():
                continue
            damage_lu = pd.read_csv(damage_lu_file, index_col=0)
            damage_lu = damage_lu.fillna(0)
            damage_lu["fid"] = str(fid)
            lu_damage.append(damage_lu)
        lu_damage = pd.concat(lu_damage)
        lu_damage.to_csv(self.dir.output.result_lu_damage.path)

        mask = self.area_vector[ID_FIELD].isin(self.failures)
        failures = self.area_vector[mask]
        failures.to_file(self.dir.output.failures.path)

        logger.info("End writing")

    def run(
        self,
        run_1d=False,
        run_2d=False,
        multiprocessing=True,
        processes=MAX_PROCESSES,
        nodamage_filter=True,
    ):
        self.run_type = "run_1d"
        if run_2d:
            self.run_type = "run_2d"

        for pid in self:
            self.dir.work[self.run_type].create_fdla_dir(str(pid), self.depth_steps)

        if processes == "max":
            processes = MAX_PROCESSES

        logger.info(f"Starting {self.run_type}!")

        self.curve = {}
        self.curve_vol = {}
        self.failures = []

        if multiprocessing:
            data = self.area_data
            args = [[pid, data, run_1d, nodamage_filter] for pid in self]

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
                    logger.info(f"{run[0]} failure! Traceback {run[2]}")
                    self.failures.append(run[0])
                    run[2] = {}

                self.curve[run[0]] = run[1]
                self.curve_vol[run[0]] = run[2]

        if not multiprocessing:
            for peil_id in tqdm(self, f"{NAME}: Damage {self.run_type}"):
                area_stats = AreaDamageCurveMethods(
                    peilgebied_id=peil_id, data=self.area_data, nodamage_filter=nodamage_filter
                )
                curve, curve_vol = area_stats.run(run_1d)
                self.curve[peil_id] = curve
                self.curve_vol[peil_id] = curve_vol

        self.write()
        self.quiet = False
        logger.info(f"Ended {self.run_type}")
        self.time.close()


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
        return (peilgebied_id, {}, str(traceback.format_exc()))


if __name__ == "__main__":
    import sys

    adc = hrt.AreaDamageCurves.from_settings_json(str(sys.argv[1]))
    adc.run(run_1d=True, multiprocessing=True, processes="max", nodamage_filter=True)
