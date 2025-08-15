import math

import geopandas as gpd
import shapely


def split_geometry_in_squares(geometry: shapely.geometry, max_area: float = 10000) -> gpd.GeoDataFrame:
    """Split geometry into squares of maximum area size."""
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    # Calculate square size (use sqrt since we want squares)
    square_size = max_area

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

    return gpd.GeoDatFrame(geomery=squares, crs=geometry.crs)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    adc = hrt.AreaDamageCurves.from_settings_json(
        r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\run_settings\run_wss_test_heiloo.json"
    )
    # adc.run_mp_optimized(processes=1)
    adc.run(run_1d=True, multiprocessing=True, processes=15)

    profiler.disable()
    profiler.dump_stats(
        r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\profile/output.prof"
    )
