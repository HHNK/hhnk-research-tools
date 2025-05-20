# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:07:02 2024

@author: kerklaac5395
"""

import pathlib
import datetime
import json
import json.tool
import os

import numpy as np
import geopandas as gp
import pandas as pd
from hhnk_research_tools import Folder
from hhnk_research_tools.logger import add_file_handler, get_logger
from hhnk_research_tools.sql_functions import database_to_gdf

# GLOBALS
ID_FIELD = "pid"
DRAINAGE_LEVEL_FIELD = "drainage_level"


class AreaDamageCurveFolders(Folder):
    def __init__(self, base, create=False):
        super().__init__(base, create=create)

        self.input = Input(self.base, create=create)

        self.work = Work(self.base, create=create)

        self.output = Output(self.base, create=create)

        self.post_processing = PostProcessing(self.base, create=create)

        if self.exists():
            self.create_readme()

    def create_readme(self):
        readme_txt = """ 
Deze tool wordt gebruikt om de schadecurve per peilgebied te berekenen.
De resultaten worden ondergebracht in een aantal mappen en bestanden:

    Input
    area.gpkg: Peilgebieden
    dem.vrt: Hoogtemodel
    lu.vrt: Landgebruik
    wss_config_settings.json: Schadetabel waterschadeschatter
    wss_curves_filter_settings.json: Filter settings voor schadecurves
    wss_lookup: Tabel voor schade per combinatie landgebruik en diepte
    wss_settings: Instellingen voor de waterschadeschatter (duur, hersteltijd etc.)
    
    Work
    log: Logging van processen
    run_1d: Resultaten per peilgebied
    run_2d: Resultaten per peilgebied
    
    Output
    result.csv: Schadecurve per peilgebied
    result_lu_areas.csv: Oppervlak landgebruiks(curve) per peilgebied
    result_lu_damage.csv: Schade landgebruiks(curve) per peilgebied
    result_vol.csv: Volume(curve) per peilgebied
    
    Post
    damage_interpolated_curve.csv: Schadecurve per cm
    damage_level_curve.csv: Schadecurve op basis van waterstand
    damage_per_m3.csv: Schade per m3 per waterstand
    
        Folder per aggregatiegebied
        agg_damage.csv: Sommering van schade
        agg_landuse.csv: Sommering van landgebruiksoppervlak
        agg_volume.csv: Sommering van volume
        
        Aggregatiemethodieken (150 mm neerslag)
        1. agg_rain_lowest_area 
        Een schadecurve die start vanaf het laatste peilgebied en wordt gesommeerd wanneer het volgende peilgebied wordt bereikt.
            
        2. agg_rain_equal_depth
        In elke peilgebied wordt waterdiepte behouden.
        De schadecurves worden gesommeerd.
        
        3. agg_rain_own_area_retention
        De neerslag die valt wordt hier ook vastgehouden in hetzelfde peilgebied.
        
        aggregate.csv: Bovenstaande methodiek zijn omgezet van schadecurves naar volumes en in een bestand gezet. 
"""
        self.joinpath("read_me.txt").write_text(readme_txt)

    @property
    def structure(self):
        return f"""  
               {self.space}Folders
               {self.space}├── Input (.input)
               {self.space}├── Work (.work)
               {self.space}├── Output (.output)
               {self.space}└── Post (.post)
               """


class Input(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "input"), create)
        self.add_file("dem", "dem.vrt")
        self.add_file("lu", "lu.vrt")
        self.add_file("area", "area.gpkg")
        self.add_file("wss_settings", "wss_settings.json")
        self.add_file("wss_cfg_settings", "wss_config_settings.json")
        self.add_file("wss_curves_filter_settings", "wss_curves_filter_settings.json")
        self.add_file("settings_json_file", "settings_json_file.json")
        self.add_file("wss_lookup", "wss_lookup.json")


class Work(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "work"), create)

        self.run_1d = Run1D(self.base, create)
        self.run_2d = Run2D(self.base, create)
        self.log = Log(self.base, create)


class Run1D(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "run_1d"), create)

    def add_fdla_dirs(self, depth_steps):
        """Add directory fixed drainage level areas"""
        for i in self.path.glob("*"):
            setattr(self, f"fdla_{i.stem}", FDLADir(self.base, False, i.stem, depth_steps))

    def create_fdla_dir(self, name, depth_steps):
        """Create fixed drainage level areas"""
        setattr(self, f"fdla_{name}", FDLADir(self.base, True, name, depth_steps))


class Run2D(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "run_2d"), create)

    def add_fdla_dirs(self, depth_steps):
        for i in self.path.glob("*"):
            setattr(self, f"fdla_{i.stem}", FDLADir(self.base, False, i.stem, depth_steps))

    def create_fdla_dir(self, name, depth_steps):
        setattr(self, f"fdla_{name}", FDLADir(self.base, True, name, depth_steps))


class Log(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "log"), create)
        self.fdla = FDLATime(self.base, create)

class FDLATime(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "fdla_performance"), create)

class Output(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "output"), create)

        self.add_file("result", "result.csv")
        self.add_file("result_vol", "result_vol.csv")
        self.add_file("result_lu_areas", "result_lu_areas.csv")
        self.add_file("result_lu_damage", "result_lu_damage.csv")
        self.add_file("failures", "failures.gpkg")


class PostProcessing(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "post_processing"), create)

        self.add_file("damage_interpolated_curve", "damage_interpolated_curve.csv")
        self.add_file("volume_interpolated_curve", "volume_interpolated_curve.csv")
        self.add_file("damage_level_curve", "damage_level_curve.csv")
        self.add_file("vol_level_curve", "vol_level_curve.csv")
        self.add_file("damage_per_m3", "damage_per_m3.csv")
        self.add_file("damage_level_per_ha", "damage_level_per_ha.csv")

    def add_aggregate_dirs(self):
        for i in self.path.glob("*"):
            setattr(self, i.stem, AggregateDir(self.base, create=True, name=i.stem))

    def create_aggregate_dir(self, name):
        directory = AggregateDir(self.base, create=True, name=name)
        setattr(self, name, directory)
        return directory


class FDLADir(Folder):
    """Folder/directory for fixed drainage level areas."""

    def __init__(self, base, create, name, depth_steps):
        super().__init__(os.path.join(base, name), create)

        self.add_file("curve", "curve.csv")
        self.add_file("curve_vol", "curve_vol.csv")
        self.add_file("counts_lu", "counts_lu.csv")
        self.add_file("damage_lu", "damage_lu.csv")
        self.add_file("depth_filter", "depth_filtered.gpkg")
        self.add_file("nodamage_filtered", "nodamage_filtered.gpkg")
        self.add_file("time", "time.csv")

        for ds in depth_steps:
            self.add_file(f"depth_{ds}", f"depth_{ds}.tif")
            self.add_file(f"level_{ds}", f"level_{ds}.tif")
            self.add_file(f"lu_{ds}", f"lu_{ds}.tif")
            self.add_file(f"damage_{ds}", f"damage_{ds}.tif")


class AggregateDir(Folder):
    def __init__(self, base, create, name):
        super().__init__(os.path.join(base, name), create)

        self.add_file("agg_damage", "agg_damage.csv")
        self.add_file("agg_landuse", "agg_landuse_area.csv")
        self.add_file("agg_landuse_dmg", "agg_landuse_damage.csv")
        self.add_file("agg_volume", "agg_volume.csv")
        self.add_file("aggregate", "aggregate.csv")
        self.add_file("selection", "selection.gpkg")


class WSSTimelog:
    """
    WSSTimeLog keep track of timings of functions.

    """

    def __init__(self, name, log_file=None, time_file=None):
        self.name = str(name)
        self.time_file = time_file
        self.log_file = log_file
        self.start_time = datetime.datetime.now()
        self.data = {"time": [self.start_time], "message":["WSSTimelog initialized"]}
        self.logger = get_logger(self.name, level="INFO", filepath=log_file, filemode="a")

    @property
    def time_since_start(self):
        return datetime.datetime.now() - self.start_time

    def log(self, message):
        self.data['time'].append(datetime.datetime.now())
        self.data['message'].append(message)
        self.logger.info(message)
    
    def write(self):
        df = pd.DataFrame(index=self.data['message'])
        df['time'] = pd.to_datetime(self.data['time'])
        df = df.sort_values(by="time")

        df['duration'] = df['time'].diff().dt.total_seconds()
        df.to_csv(self.time_file)
        
def write_dict(dictionary, path, overwrite=True):
    exists = os.path.exists(path)
    if not exists or overwrite:
        with open(str(path), "w") as fp:
            json.dump(dictionary, fp)


def pad_zeros(a: np.array, shape: tuple):
    """paddes an array with zero's to shape"""
    z = np.zeros(shape)
    z[: a.shape[0], : a.shape[1]] = a
    return z


def get_drainage_areas(settings_path):
    with open(settings_path) as f:
        data = json.load(f)

    db_dict = data["csoprd_lezen"]
    sql = "SELECT * FROM CS_OBJECTEN.COMBINATIEPEILGEBIED"
    gdf, sql2 = database_to_gdf(db_dict=db_dict, sql=sql, columns=None)
    return gdf

def fdla_performance(fdla_file, fdla_time_dir):
    
    folder = pathlib.Path(fdla_time_dir)
    fdla = gp.read_file(fdla_file)
    fdla['area']= fdla.geometry.area 
    areas = fdla[['area']]
    areas.index = areas.index.astype(str)

    duration = None
    for i, f in enumerate(folder.glob("*.csv")):
        pid = f.stem 

        time = pd.read_csv(f,index_col=0)
        if duration is None:
            duration = time[['duration']]
            duration = duration.rename(columns={"duration":pid})
        else:
            try:
                duration[pid] = time.duration.values
            except Exception as e:
                print("Skipping", e)

    total_duration = duration.sum().sort_values(ascending=False)
    average_duration_per_message = (duration.sum(axis=1) / len(total_duration)).sort_values(ascending=False)
    percentage = (duration / duration.sum()).mean(axis=1)*100
    percentage = percentage.sort_values(ascending=False)

    with pd.ExcelWriter(folder/'fdla_performance_analysis.xlsx') as writer:  
        total_duration.to_excel(writer, sheet_name='total_duration')
        average_duration_per_message.to_excel(writer, sheet_name='average_duration_per_message')
        percentage.to_excel(writer, sheet_name='percentage')