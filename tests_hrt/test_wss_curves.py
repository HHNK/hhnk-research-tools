# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:44:05 2024

@author: kerklaac5395
"""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import hhnk_research_tools as hrt
import hhnk_research_tools.waterschadeschatter.resources as wss_resources
from hhnk_research_tools.waterschadeschatter.wss_curves_areas import AreaDamageCurves
from hhnk_research_tools.waterschadeschatter.wss_curves_areas_post import AreaDamageCurvesAggregation
from hhnk_research_tools.waterschadeschatter.wss_curves_areas_pre import CustomLanduse
from hhnk_research_tools.waterschadeschatter.wss_curves_lookup import WaterSchadeSchatterLookUp
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import AreaDamageCurveFolders
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY

WSS_DATA = TEST_DIRECTORY / "wss_curves"

WSS_CFG_FILE = WSS_DATA / "wss_cfg_hhnk_2020.cfg"
WSS_SETTINGS_FILE = WSS_DATA / "wss_settings_hhnk_2020.json"
WSS_CURVE_FILTER_SETTINGS_FILE = WSS_DATA / "wss_curve_filter_settings_hhnk_2020.json"
RUN_CURVES_FILE = WSS_DATA / "run_wss_curves_2024.json"

AREA_PATH = WSS_DATA / "wss_curve_area.gpkg"
AREA_AGGREGATE_PATH = WSS_DATA / "wss_curve_area_aggregate.gpkg"
NODAMAGE_PATH = WSS_DATA / "nodamage.gpkg"
PANDEN_PATH = WSS_DATA / "wss_curve_panden.gpkg"
LU_CUSTOM_PATH = WSS_DATA / "damage_curve_lu_customized.tif"
LOOKUP_PATH = WSS_DATA / "wss_lookup.json"

DEM_PATH = WSS_DATA / "wss_curve_area_dem.tif"
LU_PATH = WSS_DATA / "wss_curve_area_lu.tif"
AREA_ID = "peil_id"
VECTOR_FIELD = "name"
OUTPUT_DIR = TEMP_DIR.joinpath(f"wss_curves_{hrt.current_time(date=True)}")
OUTPUT_DIR.mkdir(exist_ok=True)
CURVE_STEP = 0.5

CURVE_MAX = 2.5
AREA_START_LEVEL = "streefpeil"
EXPECTED_RESULT = WSS_DATA / "expected_result.csv"
EXPECTED_RESULT_OPTIMIZED = WSS_DATA / "expected_result_optimized.csv"
EXPECTED_RESULT_AGGREGATE = WSS_DATA / "expected_result_aggregate.csv"
LANDUSE_CONVERSION_TABLE = hrt.get_pkg_resource_path(wss_resources, "landuse_conversion_table.csv")
EXPECTED_LANDGEBRUIKCURVE = WSS_DATA / "landgebruikcurve_Wieringermeer.png"
EXPECTED_BERGINGSCURVE = WSS_DATA / "bergingscurve_Wieringermeer.png"
EXPECTED_SCHADECURVE = WSS_DATA / "schadecurve_Wieringermeer.png"
EXPECTED_AGGREGATE = WSS_DATA / "schade_aggregate_Wieringermeer.png"
EXPECTED_LOOKUP = WSS_DATA / "wss_lookup_test.json"

# %%


class TestCustomLanduse:
    def test_custom_landuse(self):
        cl = CustomLanduse(PANDEN_PATH, LU_PATH)
        cl.run(OUTPUT_DIR)

        o = hrt.Raster(OUTPUT_DIR / "damage_curve_lu_tile_0.tif")
        r = hrt.Raster(LU_CUSTOM_PATH)
        assert (r._read_array() == o._read_array()).all()


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
            nodamage_file=NODAMAGE_PATH,
            area_id=AREA_ID,
            curve_max=CURVE_MAX,
            curve_step=CURVE_STEP,
            area_start_level_field=AREA_START_LEVEL,
            wss_config_file=WSS_CFG_FILE,
            wss_curves_filter_settings_file=WSS_CURVE_FILTER_SETTINGS_FILE,
            panden_path=PANDEN_PATH,
        )

        shutil.copy(LOOKUP_PATH, self.dir.input.wss_lookup.path)

        return schadecurves

    @pytest.fixture(scope="class")
    def output(self):
        output = pd.read_csv(EXPECTED_RESULT)
        return output

    # Note: 2D wordt eerst getest omdat result_lu_damage.csv wordt overschreven en
    # zodoende niet goed getest wordt in de validatie.
    def test_integrated_2d_mp(self, schadecurves: AreaDamageCurves, output: pd.DataFrame):
        schadecurves.run(run_2d=True, multiprocessing=True, processes=4, nodamage_filter=True)
        test_output = pd.read_csv(schadecurves.dir.output.result.path)
        pd.testing.assert_frame_equal(output, test_output)

    def test_integrated_2d(self, schadecurves: AreaDamageCurves, output: pd.DataFrame):
        schadecurves.run(run_2d=True, multiprocessing=False, processes=4, nodamage_filter=True)
        test_output = pd.read_csv(schadecurves.dir.output.result.path)
        pd.testing.assert_frame_equal(output, test_output)

    def test_integrated_1d_mp(self, schadecurves: AreaDamageCurves, output: pd.DataFrame):
        schadecurves.run(run_1d=True, multiprocessing=True, processes=4)
        test_output = pd.read_csv(schadecurves.dir.output.result.path)
        pd.testing.assert_frame_equal(output, test_output)

    def test_integrated_1d(self, schadecurves: AreaDamageCurves, output: pd.DataFrame):
        schadecurves.run(run_1d=True, multiprocessing=False)
        test_output = pd.read_csv(schadecurves.dir.output.result.path)
        pd.testing.assert_frame_equal(output, test_output)

    def test_integrated_1d_mp_optimized(self, schadecurves: AreaDamageCurves):
        schadecurves.run_mp_optimized(limit=500, tile_size=100)
        test_output = pd.read_csv(schadecurves.dir.output.result.path)
        output_optimized = pd.read_csv(EXPECTED_RESULT_OPTIMIZED)
        pd.testing.assert_frame_equal(output_optimized, test_output)

    def test_lookup(self, schadecurves: AreaDamageCurves):
        lookup = WaterSchadeSchatterLookUp(wss_settings=schadecurves.wss_settings, depth_steps=[0, 1])
        lookup.run(flatten=True)
        lookup.write_dict(path=OUTPUT_DIR / "wss_lookup_test.json")

        with open(str(OUTPUT_DIR / "wss_lookup_test.json")) as json_file:
            lookup_output = json.load(json_file)

        with open(str(EXPECTED_LOOKUP)) as json_file:
            lookup_expected = json.load(json_file)
        assert lookup_expected == lookup_output


class TestWSSAggregation:
    @pytest.fixture(scope="class")
    @pytest.mark.dependency(name="TestWSSCurves")
    def aggregatie(self):
        aggregatie = AreaDamageCurvesAggregation(
            result_path=OUTPUT_DIR,
            aggregate_vector_path=AREA_AGGREGATE_PATH,
            aggregate_vector_id_field=VECTOR_FIELD,
            landuse_conversion_path=LANDUSE_CONVERSION_TABLE,
        )

        return aggregatie

    @pytest.fixture(scope="class")
    def output(self):
        output = pd.read_csv(EXPECTED_RESULT_AGGREGATE)
        return output

    def compare_images(self, img1_path, img2_path):
        """Compare images using Mean Squared Error"""
        img1 = np.array(Image.open(img1_path).convert("RGB"))
        img2 = np.array(Image.open(img2_path).convert("RGB"))
        return (img1 == img2).all()

    def test_agg_methods(
        self,
        aggregatie: AreaDamageCurvesAggregation,
        output: pd.DataFrame,
    ):
        aggregatie.run(create_html=False)
        test_output = pd.read_csv(aggregatie.dir.post_processing["Wieringermeer"].aggregate.path)
        pd.testing.assert_frame_equal(output, test_output)

    def test_images(self, aggregatie: AreaDamageCurvesAggregation):
        path_landgebruikcurve = aggregatie.dir.post_processing["Wieringermeer"].figures.landgebruikcurve.path
        path_bergingscurve = aggregatie.dir.post_processing["Wieringermeer"].figures.bergingscurve.path
        path_schadecurve = aggregatie.dir.post_processing["Wieringermeer"].figures.schadecurve.path
        path_aggregate = aggregatie.dir.post_processing["Wieringermeer"].figures.aggregate.path
        assert self.compare_images(path_landgebruikcurve, EXPECTED_LANDGEBRUIKCURVE)
        assert self.compare_images(path_bergingscurve, EXPECTED_BERGINGSCURVE)
        assert self.compare_images(path_schadecurve, EXPECTED_SCHADECURVE)
        assert self.compare_images(path_aggregate, EXPECTED_AGGREGATE)


class TestWSSLookup:
    @pytest.fixture(scope="class")
    def lookup(self):
        lookup = WaterSchadeSchatterLookUp(
            result_path=OUTPUT_DIR,
            aggregate_vector_path=AREA_AGGREGATE_PATH,
            aggregate_vector_id_field=VECTOR_FIELD,
            landuse_conversion_path=LANDUSE_CONVERSION_TABLE,
        )

        return lookup


# %%
if __name__ == "__main__":
    selftest = TestWSSCurves()
    output = selftest.schadecurves()
    output = selftest.output()

    selftest = TestWSSAggregation()
    aggregatie = selftest.aggregatie()
    output = selftest.output()

    selftest.test_agg_methods(aggregatie=aggregatie, output=output)
