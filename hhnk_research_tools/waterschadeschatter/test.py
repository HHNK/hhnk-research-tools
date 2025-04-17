import hhnk_research_tools as hrt

adc = hrt.AreaDamageCurves.from_settings_json(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\run_settings\run_wss_filter_2024_Jasmijn.json")
adc.run(
        run_1d=True, multiprocessing=True, processes="max", nodamage_filter=True
    )