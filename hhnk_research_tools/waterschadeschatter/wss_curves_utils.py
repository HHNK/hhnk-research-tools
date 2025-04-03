# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:07:02 2024

@author: kerklaac5395
"""
import os
import json
import datetime
import numpy as np
from hhnk_research_tools import Folder


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
        with open(os.path.join(self.base, "read_me.txt"), mode="w") as f:
            f.write(readme_txt)

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
    
    def create_fdla_dir(self, name, depth_steps):
        setattr(self, f"flda_{name}", FDLADir(self.base, True, name, depth_steps))
        
    
class Run2D(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "run_2d"), create)
        
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
        
    def create_aggregate_dir(self, name):
        return AggregateDir(self.base, True, name)
    
class FDLADir(Folder):
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
            self.add_file(f"lu_{ds}", f"lu_{ds}.gpkg")

class AggregateDir:
    def __init__(self, base, create, name):
        super().__init__(os.path.join(base, name), create)
        
        
class WSSTimelog:
    """
    WSSTimeLog logs all message and writes them to a logfile.
    It also print messages to your consolse and keeps track of time.
    
    """
    def __init__(self, subject, quiet, output_dir=None, log_file=None):
        self.s = subject
        self.quiet = quiet
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

            self.logger = create_logger(self.log_file)

    @property
    def time_since_start(self):
        delta = datetime.datetime.now() - self.start_time
        return delta

    def _message(self, msg):
        now = str(datetime.datetime.now())[:19]
        if not self.quiet:
            print(self.s, f"{now} [since start: {str(self.time_since_start)[:7]}]", msg)

        if self.use_logging:
            self.logger.info(msg)


def create_logger(filename):
    import multiprocessing, logging

    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have
    # duplicated messages in the output
    if not len(logger.handlers):
        logger.addHandler(handler)
    return logger


def write_dict(dictionary, path, overwrite=True):
    exists = os.path.exists(path)
    if not exists or overwrite:
        with open(str(path), "w") as fp:
            json.dump(dictionary, fp)


def pad_zeros(a, shape):
    z = np.zeros(shape)
    z[: a.shape[0], : a.shape[1]] = a
    return z


if __name__ == "__main__":
    test = AreaDamageCurveFolders(
        r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\output\damage_curves_2024_test",
        create=True,
    )
