# %%
from pathlib import Path

import hhnk_research_tools as hrt
from hhnk_research_tools.waterschadeschatter import wss_main
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY


def test_wss():
    cfg_file = hrt.get_pkg_resource_path(package_resource=hrt.waterschadeschatter.resources, name="cfg_lizard.cfg")
    landuse_file = TEST_DIRECTORY / "landuse_test.tif"
    depth_file = TEST_DIRECTORY / "depth_test.tif"
    output_file = hrt.Raster(TEMP_DIR / rf"schade_test_{hrt.get_uuid()}.tif")

    wss_settings = {
        "inundation_period": 48,  # uren
        "herstelperiode": "10 dagen",
        "maand": "sep",
        "cfg_file": cfg_file,
        "dmg_type": "gem",
    }

    # Calculation
    self = wss_main.Waterschadeschatter(
        depth_file=depth_file,
        landuse_file=landuse_file,
        wss_settings=wss_settings,
    )

    # Berkenen schaderaster
    output_raster = hrt.Raster(output_file)
    verbose = True
    overwrite = True
    self.run(
        output_raster=output_raster,
        calculation_type="sum",
        verbose=verbose,
        overwrite=overwrite,
    )

    assert output_file.exists()
    assert output_file.statistics() == {"min": 3.6e-05, "max": 88.486397, "mean": 19.272263, "std": 31.117451}


# %%
if __name__ == "__main__":
    # test_wss()
    # %%
    tempdir = Path(r"D:\github\wvangerwen\temp_demo_schadeschatter")
    landuse_file = tempdir / "landuse_castricum.tif"
    depth_file = tempdir / "depth_max_blok_glg_T10.tif"
    output_file = hrt.Raster(tempdir / rf"schade_test_{hrt.get_uuid()}.tif")
