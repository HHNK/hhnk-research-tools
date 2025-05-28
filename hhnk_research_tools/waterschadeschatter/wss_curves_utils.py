# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:07:02 2024

@author: kerklaac5395
"""

import datetime
import json
import logging
import math
import os
import pathlib

import geopandas as gp
import numpy as np
import pandas as pd
import shapely

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
        self.add_file("lu_input", "lu_input.vrt")
        self.add_file("area", "area.gpkg")
        self.add_file("wss_settings", "wss_settings.json")
        self.add_file("wss_cfg_settings", "wss_config_settings.json")
        self.add_file("wss_curves_filter_settings", "wss_curves_filter_settings.json")
        self.add_file("settings_json_file", "settings_json_file.json")
        self.add_file("wss_lookup", "wss_lookup.json")
        self.tiles = Tiles(self.base, create)
        self.custom_landuse_tiles = CustomLandUseTiles(self.base, create)

class Tiles(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "tiles"), create)

class CustomLandUseTiles(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "custom_landuse_tiles"), create)

class Work(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "work"), create)

        self.run_1d = Run1D(self.base, create)
        self.run_2d = Run2D(self.base, create)
        self.log = Log(self.base, create)


class Run1D(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "run_1d"), create)

    def add_fdla_dir(self, depth_steps, name):
        """Add directory fixed drainage level areas"""
        setattr(self, f"fdla_{name}", FDLADir(self.base, False, name, depth_steps))

    def add_fdla_dirs(self, depth_steps):
        """Add directory fixed drainage level areas"""
        for i in self.path.glob("*"):
            self.add_fdla_dir(depth_steps, i.stem)

    def create_fdla_dir(self, name, depth_steps, overwrite, tiled=False):
        """Create fixed drainage level areas"""
        if (pathlib.Path(self.base) / f"fdla_{name}").exists() and not overwrite:
            self.add_fdla_dir(depth_steps, name)
        else:
            setattr(self, f"fdla_{name}", FDLADir(self.base, True, name, depth_steps))


class Run2D(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "run_2d"), create)

    def add_fdla_dir(self, depth_steps, name):
        """Add directory fixed drainage level areas"""
        setattr(self, f"fdla_{name}", FDLADir(self.base, False, name, depth_steps))

    def add_fdla_dirs(self, depth_steps):
        """Add directory fixed drainage level areas"""
        for i in self.path.glob("*"):
            self.add_fdla_dir(depth_steps, i.stem)

    def create_fdla_dir(self, name, depth_steps, overwrite):
        """Create fixed drainage level areas"""
        if (pathlib.Path(self.base) / f"fdla_{name}").exists() and not overwrite:
            self.add_fdla_dir(depth_steps, name)
        else:
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
        
        self.add_file("fdla_figures", "figures.gpkg")
        self.add_file("schadecurves_html", "Schadecurves.html")

        self.create_readme()

    def add_aggregate_dirs(self):
        for i in self.path.glob("*"):
            setattr(self, i.stem, AggregateDir(self.base, create=True, name=i.stem))

    def create_aggregate_dir(self, name):
        directory = AggregateDir(self.base, create=True, name=name)
        setattr(self, name, directory)
        return directory
    
    def create_figure_dir(self, fdla_ids):
        """Create a directory for figures for fdla's."""
        setattr(self, "figures", Figures(self.base, create=True, fdla_ids=fdla_ids))
        
    def create_readme(self):
        readme_txt = """ 
De figures.gpkg kan gebruikt worden om grafieken te tonen in qgis.
De volgende stappen moeten daarvoor worden uitgevoerd:
1. Voeg de figures.gpkg toe aan QGIS.
2. Ga naar eigenschappen de laag schadecurve, voeg hieraan het volgende toe:
    <img src="[% schadecurve %]" width=600 height=600 >
3. Herhaal dit voor de andere lagen.
4. Zet onder view je maptips aan.
5. Hoover over de geomtriee om de grafieken te bekijken.
"""
        self.joinpath("read_me_qgis_grafiek_lezen.txt").write_text(readme_txt)


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
            self.add_file(f"damage_{ds}", f"damage_{ds}.tif")
        #    self.add_file(f"depth_{ds}", f"depth_{ds}.tif")
        #    self.add_file(f"level_{ds}", f"level_{ds}.tif")
        #    self.add_file(f"lu_{ds}", f"lu_{ds}.tif")


class AggregateDir(Folder):
    def __init__(self, base, create, name):
        super().__init__(os.path.join(base, name), create)

        self.add_file("agg_damage", "agg_damage.csv")
        self.add_file("agg_landuse", "agg_landuse_area.csv")
        self.add_file("agg_landuse_dmg", "agg_landuse_damage.csv")
        self.add_file("agg_volume", "agg_volume.csv")
        self.add_file("agg_damage_ha", "agg_damage_ha.csv")
        self.add_file("agg_damage_m3", "agg_damage_m3.csv")
        self.add_file("aggregate", "aggregate.csv")
        self.add_file("result_lu_areas_classes", "result_lu_areas_classes.csv")
        self.add_file("result_lu_damages_classes", "result_lu_damages_classes.csv")
        self.add_file("selection", "selection.gpkg")

    def create_figure_dir(self, name):
        directory = AggregationFigures(self.base, create=True, name=name)
        setattr(self, "figures", directory)
        return directory
    
class AggregationFigures(Folder):
    def __init__(self, base, create, name):
        super().__init__(os.path.join(base, "figures"), create)

        self.add_file("landgebruikcurve", f"landgebruikcurve_{name}.png")
        self.add_file("bergingscurve", f"bergingscurve_{name}.png")
        self.add_file("schadecurve", f"schadecurve_{name}.png")

class Figures(Folder):
    def __init__(self, base, create, fdla_ids):
        super().__init__(os.path.join(base, "figures"), create)
        for name in fdla_ids:
            self.add_file(f"landgebruikcurve_{name}", f"landgebruikcurve_{name}.png")
            self.add_file(f"bergingscurve_{name}", f"bergingscurve_{name}.png")
            self.add_file(f"schadecurve_{name}", f"schadecurve_{name}.png")

class WSSTimelog:
    """
    WSSTimeLog keep track of timings of functions.

    """

    def __init__(self, name, log_file=None, time_file=None, quiet=False):
        self.name = str(name)
        self.time_file = time_file
        self.log_file = log_file
        self.start_time = datetime.datetime.now()
        self.data = {"time": [self.start_time], "message": ["WSSTimelog initialized"]}
        self.logger = get_logger(self.name, level=logging.DEBUG, filepath=log_file, filemode="a")
        self.quiet = quiet

    @property
    def time_since_start(self):
        return datetime.datetime.now() - self.start_time

    def log(self, message):
        self.data["time"].append(datetime.datetime.now())
        self.data["message"].append(message)
        if not self.quiet:
            self.logger.info(message)

    def write(self):
        df = pd.DataFrame(index=self.data["message"])
        df["time"] = pd.to_datetime(self.data["time"])
        df = df.sort_values(by="time")

        df["duration"] = df["time"].diff().dt.total_seconds()
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


def fdla_performance(fdla_gdf, tile_gdf, fdla_time_dir, folder):
    """
    Analyze the performance of fixed drainage level areas (FDLA) by calculating the duration of processing for each area.
    """
    # Combine the tile_gdf with fdla_gdf, ensuring no duplicates based on 'pid'
    if len(tile_gdf) > 0:
        fdla_gdf = fdla_gdf[~fdla_gdf["pid"].isin(tile_gdf["ori_pid"])]
        fdla_gdf = gp.GeoDataFrame(
            pd.concat([fdla_gdf, tile_gdf]).reset_index(), geometry="geometry", crs=fdla_gdf.crs
        )

    fdla_gdf["area"] = fdla_gdf.geometry.area
    fdla_gdf.index = fdla_gdf.pid

    for i, (_, data) in enumerate(fdla_gdf.iterrows()):
        path = fdla_time_dir / f"time_{data.pid}.csv"
        if not path.exists():
            print(f"File {path} does not exist, skipping.")
            continue

        time = pd.read_csv(path, index_col=0)
        if i == 0:
            duration = time[["duration"]]
            duration = duration.rename(columns={"duration": data.pid})
            areas = [data.area]
        else:
            values = time.duration.values
            if len(values) == len(duration):
                duration[data.pid] = time.duration.values
                areas.append(data.area)

    total_duration = duration.sum().sort_values(ascending=False)
    average_duration_per_message = (duration.sum(axis=1) / len(total_duration)).sort_values(ascending=False)
    percentage = (duration / duration.sum()).mean(axis=1) * 100
    percentage = percentage.sort_values(ascending=False)

    with pd.ExcelWriter(folder / "fdla_performance_analysis.xlsx") as writer:
        pd.DataFrame(duration).to_excel(writer, sheet_name="duration_per_message")
        total_duration.to_excel(writer, sheet_name="total_duration")
        average_duration_per_message.to_excel(writer, sheet_name="average_duration_per_message")
        percentage.to_excel(writer, sheet_name="percentage")
        # duration_per_area = pd.DataFrame(data={"duration": duration.sum(), "area": areas})
        # duration_per_area.to_excel(writer, sheet_name="duration_per_area")


def split_geometry_in_tiles(geometry: shapely.geometry, envelope_tile_size=250) -> gp.GeoDataFrame:
    """Split geometry into squares of maximum area size.
    This function takes a geometry and splits it into smaller square tiles of a specified size.
        params:
            geometry (shapely.geometry): The geometry to be split into tiles.
            envelope_tile_size (float): The size one length of each square tile in the same units as the geometry's CRS.
    """
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    # Calculate square size (use sqrt since we want squares)
    square_size = envelope_tile_size

    # Calculate number of squares needed in each direction
    nx = math.ceil(width / square_size)
    ny = math.ceil(height / square_size)

    squares = []
    for i in range(nx):
        for j in range(ny):
            # Create square bounds
            x0 = bounds[0] + i * square_size
            y0 = bounds[1] + j * square_size
            x1 = min(x0 + square_size, bounds[2])
            y1 = min(y0 + square_size, bounds[3])

            # Create square and intersect with original geometry
            square = shapely.geometry.box(x0, y0, x1, y1)
            intersection = geometry.intersection(square)

            if not intersection.is_empty:
                squares.append(intersection)

    return gp.GeoDataFrame(geometry=squares, crs="EPSG:28992")
