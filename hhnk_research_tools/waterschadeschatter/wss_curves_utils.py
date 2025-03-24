# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:07:02 2024

@author: kerklaac5395
"""
import json
import datetime
import numpy as np

# First-party imports
import os
from pathlib import Path

import hhnk_research_tools as hrt
import pandas as pd
from hhnk_research_tools import Folder
from hhnk_research_tools.variables import (
    GDB,
    SHAPE,
    file_types_dict,
)


FOLDER_STRUCTURE = """
    Main Folders object
        ├── Input
        │ ├── dem.vrt
        │ ├── lu.vrt
        │ ├── lookup.json
        │ ├── area.gpkg
        ├── Work
        │ ├── run_1d_mp
        │ ├── run_2d_mp
        ├── Output
        │ ├── run_1d_mp.csv
        │ └── run_1d_mp.json

    """


class AreaDamageCurveFolders(Folder):
    def __init__(self, base, create=False):
        super().__init__(base, create=create)

        self.input = Input(self.base, create=create)

        self.work = Work(self.base, create=create)

        self.output = Output(self.base, create=create)

        self.post = Post(self.base, create=create)
        
    @property
    def structure(self):
        return f"""  
               {self.space}Folders
               {self.space}├── Input (.input)
               {self.space}├── Work (.work)
               {self.space}├── Output (.output)
               {self.space}└── Post (.post)
               """

    @property
    def full_structure(self):
        return print(FOLDER_STRUCTURE)

class Input(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "input"), create)
        self.add_file("dem", "dem.vrt")
        self.add_file("lu", "lu.vrt")
        self.add_file("area", "area.gpkg")
        self.add_file("wss_settings", "wss_settings.json")
        self.add_file("wss_cfg_settings", "wss_config_settings.json")
        self.add_file("wss_curves_filter_settings", "wss_curves_filter_settings.json")
        self.add_file("wss_lookup", "wss_lookup.json")
        
class Work(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "work"), create)

        self.run_1d = Run1D(self.base, create)
        self.run_2d = Run2D(self.base, create)

class Run1D(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "run_1d"), create)
    
class Run2D(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "run_2d"), create)
    

class Output(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "output"), create)
        
        self.add_file("result", "result.csv")
        self.add_file("result_vol", "result_vol.csv")
        self.add_file("result_lu_areas", "result_lu_areas.csv")
        self.add_file("result_lu_damage", "result_lu_damage.csv")


class Post(Folder):
    def __init__(self, base, create):
        super().__init__(os.path.join(base, "post"), create)
        
        


class WSSTimelog:
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
    test = AreaDamageCurveFolders(r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\output\damage_curves_2024_test", create=True)

