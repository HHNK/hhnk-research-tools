# %%
from pathlib import Path

import numpy as np
import rasterio.transform

import hhnk_research_tools as hrt
from hhnk_research_tools.waterschadeschatter import wss_main
from tests_hrt.config import TEMP_DIR


def create_test_raster(path: Path, data: np.ndarray):
    """Create a simple test raster file."""
    raster = hrt.Raster(path)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999,
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": "EPSG:28992",
        "transform": rasterio.transform.from_origin(0, 0, 1.0, 1.0),
    }
    with raster.open_rio(mode="w", profile=profile) as dst:
        dst.write(data, 1)
    return raster


def test_wss_inundation_period_raster():
    """Test that inundation_period raster produces spatially variable damage."""
    cfg_file = hrt.get_pkg_resource_path(package_resource=hrt.waterschadeschatter.resources, name="cfg_lizard.cfg")

    # Create test data: 1h left, 4h middle, 72h right
    landuse_data = np.full((10, 15), 15, dtype=np.float32)
    # info: gamma_inundatieduur for landuse 15
    # = [0.5, 1, 1, 1, 1] for [1, 12, 24, 72, 480] hours
    depth_data = np.full((10, 15), 0.5, dtype=np.float32)
    inundation_period_data = np.full((10, 15), 1, dtype=np.float32)  # Left: 1h
    inundation_period_data[:, 5:10] = 4  # Middle: interpolation between 1h and 12h
    inundation_period_data[:, 10:] = 72  # Right: 72h
    landuse_file = TEMP_DIR / f"test_landuse_{hrt.get_uuid()}.tif"
    depth_file = TEMP_DIR / f"test_depth_{hrt.get_uuid()}.tif"
    inundation_period_file = TEMP_DIR / f"test_inundation_{hrt.get_uuid()}.tif"
    output_file = TEMP_DIR / f"test_output_{hrt.get_uuid()}.tif"

    create_test_raster(landuse_file, landuse_data)
    create_test_raster(depth_file, depth_data)
    create_test_raster(inundation_period_file, inundation_period_data)

    wss = wss_main.Waterschadeschatter(
        depth_file=depth_file,
        landuse_file=landuse_file,
        inundation_period_file=inundation_period_file,
        wss_settings={"herstelperiode": "10 dagen", "maand": "sep", "cfg_file": cfg_file, "dmg_type": "gem"},
    )

    wss.run(output_raster=hrt.Raster(output_file), calculation_type="sum", verbose=False, overwrite=True)

    # Verify spatially variable damage
    with hrt.Raster(output_file).open_rio() as src:
        result = src.read(1)

    left_mean = np.mean(result[:, :5])  # 1h zone
    middle_mean = np.mean(result[:, 5:10])  # 4h zone (interpolated)
    right_mean = np.mean(result[:, 10:])  # 72h zone

    # Calculate expected gamma values:
    # 1h: gamma = 0.5
    # 4h: gamma = 0.5 + (1-0.5)/(12-1)*(4-1) = 0.5 + 0.5/11*3 = 0.636
    # 72h: gamma = 1.0
    expected_gamma_1h = 0.5
    expected_gamma_4h = 0.5 + (1.0 - 0.5) / (12 - 1) * (4 - 1)
    expected_gamma_72h = 1.0

    assert left_mean > 0 and middle_mean > 0 and right_mean > 0, "Damage should be positive"
    assert left_mean < middle_mean < right_mean, "Damage should increase with inundation period"

    # Verify interpolation: damage ratios should match gamma ratios
    ratio_middle_to_left = middle_mean / left_mean
    expected_ratio = expected_gamma_4h / expected_gamma_1h
    assert abs(ratio_middle_to_left - expected_ratio) < 0.01, (
        f"Interpolation incorrect: {ratio_middle_to_left:.3f} vs {expected_ratio:.3f}"
    )

    ratio_right_to_left = right_mean / left_mean
    expected_ratio_72h = expected_gamma_72h / expected_gamma_1h
    assert abs(ratio_right_to_left - expected_ratio_72h) < 0.01, (
        f"Ratio incorrect: {ratio_right_to_left:.3f} vs {expected_ratio_72h:.3f}"
    )

    print("✓ Inundation period raster test passed:")
    print(f"  Left (1h)={left_mean:.2f}, gamma={expected_gamma_1h}")
    print(f"  Middle (4h)={middle_mean:.2f}, gamma={expected_gamma_4h:.3f} (interpolated)")
    print(f"  Right (72h)={right_mean:.2f}, gamma={expected_gamma_72h}")


# %%
if __name__ == "__main__":
    test_wss_inundation_period_raster()
# %%
