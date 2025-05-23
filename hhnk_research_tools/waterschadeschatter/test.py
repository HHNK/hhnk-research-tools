import hhnk_research_tools as hrt
import cProfile
import hhnk_research_tools as hrt
from hhnk_research_tools.variables import DEFAULT_NODATA_VALUES

raster = hrt.Raster(r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\output\damage_curves_2024_full\input\dem.vrt")
da = raster.open_rxr(chunksize=256)
output_path = r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\output\damage_curves_2024_full\input\dem_100.tif"   
da_100 = da *100
hrt.Raster.write(
    raster_out=output_path,
    result=da_100,
    dtype='int16',
    nodata=DEFAULT_NODATA_VALUES['int16'],
    crs=da_100.rio.crs,
)
def process_raster(input_raster: Raster, output_path: Path, chunksize: int = 4096):
    # Open input raster with chunking
    da = input_raster.open_rxr(chunksize=chunksize)
    
    # Apply your modification (example: multiply all values by 2)
    modified_da = da * 2
    
    # Write modified data to new raster
    Raster.write(
        raster_out=output_path,
        result=modified_da,
        dtype='float32',
        nodata=input_raster.nodata,
        chunksize=chunksize
    )

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    adc = hrt.AreaDamageCurves.from_settings_json(r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\run_settings\run_wss_test_heiloo.json")
    #adc.run_mp_optimized(processes=1)
    adc.run(run_1d=True, multiprocessing=True, processes=15)

    profiler.disable()
    profiler.dump_stats(r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\profile/output.prof")