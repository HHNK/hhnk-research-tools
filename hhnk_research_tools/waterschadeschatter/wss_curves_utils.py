# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:07:02 2024

@author: kerklaac5395
"""

import datetime
import json
import os

import numpy as np

from hhnk_research_tools import Folder
from hhnk_research_tools.logging import add_file_handler, get_logger

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
        self.add_file("nodamage_filtered", "nodamage_filtered.gpkg")

        for ds in depth_steps:
            self.add_file(f"depth_{ds}", f"depth_{ds}.tif")
            self.add_file(f"level_{ds}", f"level_{ds}.tif")
            self.add_file(f"lu_{ds}", f"lu_{ds}.tif")
            self.add_file(f"damage_{ds}", f"damage_{ds}.tif")


class AggregateDir(Folder):
    def __init__(self, base, create, name):
        super().__init__(os.path.join(base, name), create)

        self.add_file("agg_damage", "agg_damage.csv")
        self.add_file("agg_landuse", "agg_landuse.csv")
        self.add_file("agg_volume", "agg_volume.csv")
        self.add_file("aggregate", "aggregate.csv")
        self.add_file("selection", "selection.gpkg")


class WSSTimelog:
    """
    WSSTimeLog logs all message and writes them to a logfile.
    It also print messages to your consolse and keeps track of time.

    """

    def __init__(self, subject, output_dir=None, log_file=None):
        self.subject = subject
        self.start_time = datetime.datetime.now()
        self.output_dir = output_dir
        self.use_logging = output_dir is not None or log_file is not None

        if self.use_logging:
            self.log_file = log_file

            if log_file is None:
                now = datetime.datetime.now().strftime("%Y%m%d%H%M")
                log_dir = self.output_dir / "log"
                log_dir.mkdir(exist_ok=True, parents=True)
                self.log_file = log_dir / f"{now} - {subject}.log"

            self.logger = get_logger(self.subject)
            add_file_handler(logger=self.logger, filepath=self.log_file)

    @property
    def time_since_start(self):
        return datetime.datetime.now() - self.start_time

    def close(self):
        """Loggers have to be removed in handler"""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()


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
