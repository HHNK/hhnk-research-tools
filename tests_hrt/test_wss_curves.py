# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:44:05 2024

@author: kerklaac5395
"""

import pytest

from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY

from hhnk_research_tools.waterschadeschatter.wss_curves_areas import AreaDamageCurves
from hhnk_research_tools.waterschadeschatter.wss_curves_areas_post import AreaDamageCurvesAggregation
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import AreaDamageCurveFolders

import pandas as pd


WSS_DATA = TEST_DIRECTORY / "wss_curves"

WSS_CFG_FILE = WSS_DATA / "wss_cfg_hhnk_2020.cfg"
WSS_SETTINGS_FILE = WSS_DATA /  "wss_settings_hhnk_2020.json"
WSS_CURVE_FILTER_SETTINGS_FILE = WSS_DATA /  "wss_curve_filter_settings_hhnk_2020.json"
RUN_CURVES_FILE = WSS_DATA / "run_wss_curves_2024.json"

AREA_PATH = WSS_DATA / "wss_curve_area.gpkg"
AREA_AGGREGATE_PATH = WSS_DATA / "wss_curve_area_aggregate.gpkg"

DEM_PATH = WSS_DATA / "wss_curve_area_dem.tif"
LU_PATH = WSS_DATA / "wss_curve_area_lu.tif"
AREA_ID = "peil_id"
VECTOR_FIELD = "name"
OUTPUT_PATH = TEMP_DIR
CURVE_STEP = 0.5
CURVE_MAX = 2.5
AREA_START_LEVEL = "streefpeil"
RESULT = WSS_DATA / "result.csv"
RESULT_AGGREGATE = WSS_DATA / "result_aggregate.csv"


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

        schadecurves.wss_config = WSS_CFG_FILE
        schadecurves.wss_settings = WSS_SETTINGS_FILE
        schadecurves.wss_curves_filter_settings = WSS_CURVE_FILTER_SETTINGS_FILE

        return schadecurves

    @pytest.fixture(scope="class")
    def output(self):
        output = pd.read_csv(RESULT)
        return output

    def test_integrated_1d_mp(self, schadecurves, output):
        schadecurves.run(run_1d=True, multiprocessing=True, processes=4)
        test_output = pd.read_csv(OUTPUT_PATH / "output"/"result.csv")
        assert (output == test_output).all()[0]

    def test_integrated_1d(self, schadecurves, output):
        schadecurves.run(run_1d=True, multiprocessing=False)

        test_output = pd.read_csv(OUTPUT_PATH / "output"/"result.csv")
        assert (output == test_output).all()[0]

    def test_integrated_2d_mp(self, schadecurves, output):
        schadecurves.run(run_2d=True, multiprocessing=True, processes=4, nodamage_filter=True)

        test_output = pd.read_csv(OUTPUT_PATH / "output"/"result.csv")
        assert (output == test_output).all()[0]

    def test_integrated_2d(self, schadecurves, output):
        schadecurves.run(run_2d=True, multiprocessing=False, processes=4, nodamage_filter=True)

        test_output = pd.read_csv(OUTPUT_PATH / "output"/"result.csv")
        assert (output == test_output).all()[0]


class TestWSSAggregation:
    
    @pytest.fixture(scope="class")
    def aggregatie(self):

        adca= AreaDamageCurvesAggregation(
            result_path=OUTPUT_PATH,
            aggregate_vector_path=AREA_AGGREGATE_PATH, 
            vector_field=VECTOR_FIELD, 
            quiet=False
        )

        return adca
    
    @pytest.fixture(scope="class")
    def output(self):
        output = pd.read_csv(RESULT)
        return output
    
    def test_agg_methods(self,aggregatie,output):
        aggregatie.run()
        # folders = AreaDamageCurveFolders(OUTPUT_PATH)
        
        test_output = pd.read_csv(OUTPUT_PATH / "post" / "Wieringermeer" / "aggregate.csv")   
        assert (test_output == output).all()[0]
        
class TestWSSAggregationFolders:
    def test_generating_folders(self):
        
        folders = AreaDamageCurveFolders(TEMP_DIR)
        
        assert (TEMP_DIR / "work").exists()
        