# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:44:05 2024

@author: kerklaac5395
"""

import pytest
import sys

sys.path.append(
    r"C:\Users\kerklaac5395\OneDrive - ARCADIS\Documents\GitHub\hydrologen-projecten\schadeberekeningen"
)
sys.path.append(
    r"C:/Users/kerklaac5395/OneDrive - ARCADIS/Documents/GitHub/hhnk-research-tools"
)

from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY

from hhnk_research_tools.waterschadeschatter.wss_curves_areas import AreaDamageCurves
# from hrt.wss_areas_curves_post import AreaDamageCurvesAggregation

import pandas as pd


WSS_DATA = TEST_DIRECTORY / "wss_curves"

CFG_FILE = WSS_DATA / "cfg_hhnk_2020.cfg"
SETTINGS_FILE = WSS_DATA /  "settings_hhnk_2020.json"
FILTER_SETTINGS_FILE = WSS_DATA /  "filter_settings_hhnk_2020.json"


AREA_PATH = WSS_DATA / "wss_curve_area.shp"
DEM_PATH = WSS_DATA / "wss_curve_area_dem.tif"
LU_PATH = WSS_DATA / "wss_curve_area_lu.tif"
AREA_ID = "peil_id"
OUTPUT_PATH = TEMP_DIR
CURVE_STEP = 0.5
CURVE_MAX = 2.5
AREA_START_LEVEL = "streefpeil"
RESULT = WSS_DATA / "result.csv"


class TestWSSCurves:
    # TODO toevoegen pyfixture
    # TODO sneller maken naar 1 seconde, heel klein peilgebied.
    @pytest.fixture(scope="class")
    def schadecurves(self):

        schadecurves = AreaDamageCurves(
            AREA_PATH,
            LU_PATH,
            DEM_PATH,
            OUTPUT_PATH,
            area_id=AREA_ID,
            curve_max=CURVE_MAX,
            curve_step=CURVE_STEP,
            area_start_level=AREA_START_LEVEL,
        )

        schadecurves.wss_config = CFG_FILE
        schadecurves.wss_settings = SETTINGS_FILE
        schadecurves.wss_filter_settings = FILTER_SETTINGS_FILE

        return schadecurves

    @pytest.fixture(scope="class")
    def output(self):
        output = pd.read_csv(RESULT)
        return output

    def test_integrated_1d_mp(self, schadecurves, output):
        schadecurves.run(run_1d=True, multiprocessing=True, processes=4)
        test_output = pd.read_csv(OUTPUT_PATH / "run_1d_mp.csv")
        assert (output == test_output).all()[0]

    def test_integrated_1d(self, schadecurves, output):
        schadecurves.run(run_1d=True, multiprocessing=False)

        test_output = pd.read_csv(OUTPUT_PATH / "run_1d.csv")
        assert (output == test_output).all()[0]

    def test_integrated_2d_mp(self, schadecurves, output):
        schadecurves.run(run_2d=True, multiprocessing=True, processes=4, nodamage_filter=True)

        test_output = pd.read_csv(OUTPUT_PATH / "run_2d_mp.csv")
        assert (output == test_output).all()[0]

    def test_integrated_2d(self, schadecurves, output):
        schadecurves.run(run_2d=True, multiprocessing=False, processes=4, nodamage_filter=True)

        test_output = pd.read_csv(OUTPUT_PATH / "run_2d.csv")
        assert (output == test_output).all()[0]

# OUDDORP_PATH = DATA_PATH / "agg_ouderdorperpolder"
# RESULT_AGG = DATA_PATH / "agg_ouderdorperpolder" / "aggregaties.xlsx"
# POLDER_PATH = DATA_PATH / "polder_ouddorp.shp"

# class TestWSSAggregation:
    
#     def test_agg_methods(self):
        

#         agg = AreaDamageCurvesAggregation(OUDDORP_PATH, POLDER_PATH, "polder_id")
#         output = agg.agg_run()[18]
        
#         data = pd.read_excel(RESULT_AGG,index_col=0)
        
#         assert (data == output).all()[0]
        
        