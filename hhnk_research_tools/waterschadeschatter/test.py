import hhnk_research_tools as hrt

adc = hrt.AreaDamageCurves.from_settings_json(r"E:\05.schadecurven\settings\run_wss_filter_2024_texel.json")
adc.run_mp_optimized(processes="max", nodamage_filter=True)
