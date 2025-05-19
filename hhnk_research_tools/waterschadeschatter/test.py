import hhnk_research_tools as hrt
import cProfile

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    adc = hrt.AreaDamageCurves.from_settings_json(r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\run_settings\run_wss_test_heiloo.json")
    #adc.run_mp_optimized(processes=1)
    adc.run(run_1d=True, multiprocessing=True, processes=15)

    profiler.disable()
    profiler.dump_stats(r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\profile/output.prof")