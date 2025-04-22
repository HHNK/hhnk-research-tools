# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:44:05 2024

@author: kerklaac5395
"""

import pandas as pd
import pytest

import hhnk_research_tools as hrt
import hhnk_research_tools.waterschadeschatter.resources as wss_resources
from hhnk_research_tools.waterschadeschatter.wss_curves_areas import AreaDamageCurves
from hhnk_research_tools.waterschadeschatter.wss_curves_areas_post import AreaDamageCurvesAggregation
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import AreaDamageCurveFolders
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY

WSS_DATA = TEST_DIRECTORY / "wss_curves"

WSS_CFG_FILE = WSS_DATA / "wss_cfg_hhnk_2020.cfg"
WSS_SETTINGS_FILE = WSS_DATA / "wss_settings_hhnk_2020.json"
WSS_CURVE_FILTER_SETTINGS_FILE = WSS_DATA / "wss_curve_filter_settings_hhnk_2020.json"
RUN_CURVES_FILE = WSS_DATA / "run_wss_curves_2024.json"

AREA_PATH = WSS_DATA / "wss_curve_area.gpkg"
AREA_AGGREGATE_PATH = WSS_DATA / "wss_curve_area_aggregate.gpkg"

DEM_PATH = WSS_DATA / "wss_curve_area_dem.tif"
LU_PATH = WSS_DATA / "wss_curve_area_lu.tif"
AREA_ID = "peil_id"
VECTOR_FIELD = "name"
OUTPUT_DIR = TEMP_DIR.joinpath(f"wss_curves_{hrt.current_time(date=True)}")
CURVE_STEP = 0.5

CURVE_MAX = 2.5
AREA_START_LEVEL = "streefpeil"
EXPECTED_RESULT = WSS_DATA / "expected_result.csv"
EXPECTED_RESULT_AGGREGATE = WSS_DATA / "expected_result_aggregate.csv"
LANDUSE_CONVERSION_TABLE = hrt.get_pkg_resource_path(wss_resources, "landuse_conversion_table.csv")
# %%


class TestWSSAggregationFolders:
    """Generieke test, in onderstaande test worden ook output bestande getest"""

    def test_generating_folders(self):
        self = AreaDamageCurveFolders(OUTPUT_DIR, create=True)

        assert (OUTPUT_DIR / "work").exists()
        assert (OUTPUT_DIR / "post_processing").exists()
        assert (OUTPUT_DIR / "input").exists()
        assert (OUTPUT_DIR / "output").exists()


class TestWSSCurves:
    @pytest.fixture(scope="class")
    def schadecurves(self):
        schadecurves = self = AreaDamageCurves(
            output_dir=OUTPUT_DIR,
            area_path=AREA_PATH,
            landuse_path_dir=LU_PATH,
            dem_path_dir=DEM_PATH,
            wss_settings_file=WSS_SETTINGS_FILE,
            area_id=AREA_ID,
            curve_max=CURVE_MAX,
            curve_step=CURVE_STEP,
            area_start_level_field=AREA_START_LEVEL,
            wss_config_file=WSS_CFG_FILE,
            wss_curves_filter_settings_file=WSS_CURVE_FILTER_SETTINGS_FILE,
        )

        return schadecurves

    @pytest.fixture(scope="class")
    def output(self):
        output = pd.read_csv(EXPECTED_RESULT)
        return output

    def test_integrated_1d_mp(self, schadecurves: AreaDamageCurves, output: pd.DataFrame):
        schadecurves.run(run_1d=True, multiprocessing=True, processes=4)
        test_output = pd.read_csv(schadecurves.dir.output.result.path)
        pd.testing.assert_frame_equal(output, test_output)

    def test_integrated_1d(self, schadecurves: AreaDamageCurves, output: pd.DataFrame):
        schadecurves.run(run_1d=True, multiprocessing=False)
        test_output = pd.read_csv(schadecurves.dir.output.result.path)
        pd.testing.assert_frame_equal(output, test_output)

    def test_integrated_2d_mp(self, schadecurves: AreaDamageCurves, output: pd.DataFrame):
        schadecurves.run(run_2d=True, multiprocessing=True, processes=4, nodamage_filter=True)

        test_output = pd.read_csv(schadecurves.dir.output.result.path)
        pd.testing.assert_frame_equal(output, test_output)

    def test_integrated_2d(self, schadecurves: AreaDamageCurves, output: pd.DataFrame):
        schadecurves.run(run_2d=True, multiprocessing=False, processes=4, nodamage_filter=True)

        test_output = pd.read_csv(schadecurves.dir.output.result.path)
        pd.testing.assert_frame_equal(output, test_output)


class TestWSSAggregation:
    @pytest.fixture(scope="class")
    @pytest.mark.dependency(name="TestWSSCurves")
    def aggregatie(self):
        aggregatie = AreaDamageCurvesAggregation(
            result_path=OUTPUT_DIR,
            aggregate_vector_path=AREA_AGGREGATE_PATH,
            vector_field=VECTOR_FIELD,
            landuse_conversion_path=LANDUSE_CONVERSION_TABLE,
        )

        return aggregatie

    @pytest.fixture(scope="class")
    def output(self):
        output = pd.read_csv(EXPECTED_RESULT_AGGREGATE)
        return output

    def test_agg_methods(
        self,
        aggregatie: AreaDamageCurvesAggregation,
        output: pd.DataFrame,
    ):
        aggregatie.run()
        test_output = pd.read_csv(aggregatie.dir.post_processing["Wieringermeer"].aggregate.path)
        pd.testing.assert_frame_equal(output, test_output)


# %%
if __name__ == "__main__":
    selftest = TestWSSCurves()
    output = selftest.schadecurves()
    output = selftest.output()

    selftest = TestWSSAggregation()
    aggregatie = selftest.aggregatie()
    output = selftest.output()

    selftest.test_agg_methods(aggregatie=aggregatie, output=output)
