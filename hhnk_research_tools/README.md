# Module WaterSchadeSchatterCurves

De module ‘Waterschadeschatter Curves’ is een uitgebreide toolset voor het berekenen en naverwerken van schadecurves per peilgebied. De module berekent per peilgebied per waterstandsstijging (0.1 m, 0.2 m, etc.): 
* de totale schade, 
* de schade per volume, 
* schade per type en oppervlak landgebruik. 

De output bestaat uit ruwe data in de vorm van csv’s, maar ook visualisaties in de vorm van interactieve folium kaarten, GIS-bestanden en figuren.

# Quick-start
Er zijn 2 manieren voor het aansturen van de scripting:
-	Via python
-	Via de command line tool

**python**

Binnen python kun je de volgende code volgen om de module aan te zetten:

```python 
# Imports
import hhnk_research_tools as hrt
from pathlib import Path

# Aanmaken AreaDamageCurves object
adc = hrt.AreaDamageCurves(
    output_dir=Path("output"),
    area_path="peilgebieden.gpkg",
    landuse_path_dir="landgebruik.tif", 
    dem_path_dir="hoogtemodel.tif",
    wss_settings_file="wss_settings.json",
    wss_config_file="wss_cfg_hhnk_2020.cfg",
    area_id="peil_id",
    curve_step=0.1,
    curve_max=3.0,
    resolution=0.5
)

# Uitvoeren hoofdberekening
adc.run(run_1d=True, multiprocessing=True, processes=4)

# Post-processing
Voor post-processing kun je de volgende code uitvoeren:

# Post-processing settings
post_settings = {
    "result_path": "output",
    "aggregate_vector_path": "aggregatie_gebieden.gpkg",
    "aggregate_vector_id_field": "gebied_id", 
    "landuse_conversion_path": "landuse_conversion.csv"
}

# Aggregatie object aanmaken
agg = hrt.AreaDamageCurvesAggregatio(
	result_path=Path("output"), 
	"aggregate_vector_path": "aggregatie_gebieden.gpkg",
    "aggregate_vector_id_field": "gebied_id", 
    "landuse_conversion_path": "landuse_conversion.csv")

# Aggregatie uitvoeren
agg.run()
```
Je kunt ook gebruik maken van een ‘master settings json’, op deze manier kun je je instellingen makkelijk opslaan en later hergebruiken.
Voorbeeld voor de hoofdmodule:

```
# Aanmaken master settings JSON
settings = {
    "output_dir": "output",
    "area_path": "peilgebieden.gpkg",
    "landuse_path_dir": "landgebruik.tif",
    "dem_path_dir": "hoogtemodel.tif", 
    "wss_settings_file": "wss_settings.json",
    "wss_config_file": "wss_cfg_hhnk_2020.cfg",
    "curve_step": 0.1,
    "curve_max": 3.0
}

# Opslaan als JSON
import json
with open("run_settings.json", "w") as f:
    json.dump(settings, f)

# Laden en uitvoeren
adc = hrt.AreaDamageCurves.from_settings_json("run_settings.json")
adc.run(run_1d=True, multiprocessing=True)

# Een voorbeeld voor HHNK
\\corp.hhnk.nl\data\Hydrologen_data\05.Schadecurves\settings\run_wss_filter_2024.json

```

**command line tool**

Draai de volgende regel in je command line tool:
```
python -m hhnk_research_tools.waterschadeschatter.wss_curves_areas run_settings.json 

# Overzicht van de opties:
python -m hhnk_research_tools.waterschadeschatter.wss_curves_areas -h 

```

**output**

Door de hoofdmodule ‘wss_curves_areas’ worden 4 bestanden gegenereerd:
1.	result.csv: Schadecurves per peilgebied 
2.	result_lu_areas.csv: Oppervlak per landgebruik per dieptestap
3.	result_lu_damage.csv: Schade per landgebruik per dieptestap
4.	result_vol.csv: Volume per dieptestap

In de post-processing stap wordt de output gebruikt, i.c.m. een vectorbestand voor aggregaties (bijvoorbeeld polders of polderclusters). De standaard outputbestanden worden geïnterpoleerd (damage_interpolated_curve.csv), omgerekend naar waterstand (damage_level_curve.csv), schade per hectare (damage_level_per_ha.csv). Ook de bergingscurve wordt omgerekend naar waterstand (vol_level_curve.csv) en geinterpoleerd (volume_interpolated_curve.csv).  het volume wordt ook gecombineerd met de schade (damage_per_m3.csv). Deze bestanden vormen de basis om de aggregatie per aggregatievector uit te voeren.

 Naast aggregaties in de vorm van een csv, zijn ze ook beschikbaar in een GeoPackage (peilgebied.gpkg) en als figuren (in de map 'figuren').
 Hiernaast vinden er ook aggregaties plaats per polygoon in de opgegeven vector. Dit zijn vaak polders of polderclusters. Binnen een polder worden alle peilvakken geselecteerd met een WITHIN statement. Vervolgens worden de schadecurves/landgebruikscurves/schade aan gebouwen van de peilvakken bij elkaar opgeteld en verwerkt in curves specifiek van die subselectie. Dit zijn bestanden zoals agg_building_damage.csv, agg_damage.csv, agg_damage_ha.csv, agg_damage_m3.csv, agg_landuse_area.csv, agg_landuse_damage.csv en agg_volume.csv. Deze worden vervolgens omgezet naar figuren welke weer zichtbaar zijn in de html. Voor informatie kun je doorscrollen naar 'post-processing'.

# Workflow
De workflow volgt een aantal stappen om naar het eindresultaat te komen.

---
**pre-processing**

Als de hoofdmodule (wss_curves_areas.py) wordt aangeroepen wordt gestart met een aantal pre-processing stappen (wss_curves_areas_preprocess.py en __init__ in wss_curves_areas.py).

1.	Als de bewerkte landgebruikskaart nog niet bestaat, wordt deze aangemaakt. Hierbij wordt het landgebruik met panden op nodata gezet en worden panden vanuit het pandenbestand opgedrukt op de landgebruikskaart.
2.	De input landgebruikskaart en het hoogtemodel worden verwerkt tot VRT’s.
3.	De lookup tables worden aangemaakt. Een lookup table is een tussenproduct, waarmee de schade van een bepaalde combinatie tussen waterdiepte en landgebruik kan worden opgezocht. Als deze nog niet bestaat dan wordt deze aangemaakt. Als hij wel bestaat wordt hij ingeladen. De lookup table zorgt voor extra snelheid in het rekenproces.
4.	De logging wordt geinitialiseerd in de map ‘work/log’.
5.	Alle input wordt gekopieerd naar de folder ‘input’, zodat teruggekeken kan worden wat voor settings en input er gebruikt zijn.

**hoofdberekening**

In de hoofdberekening wordt de schade per peilgebied berekend. Per peilgebied wordt de benodigde data ‘uitgeknipt’ vanuit de landgebruikskaart en hoogtekaart. Vervolgens worden de kaarten teruggeschaald naar ‘int16’ om het datagebruik te verminderen. Er worden ook filters toegepast welke je kunt selecteren in de command line tool. Er zijn hier 2 soorten filters:
1.	‘Geen schade filter’: Bij de input wordt een vector (nodamage_file) opgegeven, welke geen schade in het gebied bevatten. Binnen deze vector wordt geen schade berekend. Dit filter wordt gebruikt om bekende foutlocaties buiten de schadeberekening te houden. 
2.	‘Diepte schade filter’: Filtert de schade bij een bepaalde combinatie tussen diepte en landgebruik. In de input wordt een json opgegeven (wss_curves_filter_settings_file) waarin deze combinatie staat beschreven.

Na de filters wordt de schade per waterdiepte/peilstijging berekend, er zijn hierin 2 opties:
1.	1D, deze slaat de numpy array plat naar een lijst, waardoor er geen ruimtelijke informatie meer in zit. Dit is de snelste manier om het door te rekenen. De tussenresultaten komen terecht in de folder work/run_1d.
2.	2D, hierbij blijft de ruimtelijk informatie behouden. De tussenresultaten komen terecht in de folder work/run_2d.

Om de snelheid van de module te bevorderen is gekozen voor het gevectorizeerd doorrekenen. De lookup tabel wordt gegenereerd als (lu x LU_LOOKUP_FACTOR) + depth_step. Bij een landgebruik van 2, een peilstijging van 0.1 en een lookup_factor van 100 (standaard) krijg je dus de code 200.1. Omdat je gebruik maakt 'unnested' dictionaires kunnen combinaties tussen landgebruik en waterdiepte gevectorizeerd (dus snel) worden opgezocht in de tabel.

In de hoofdberekening zijn er meerdere opties toegevoegd om het proces in snelheid en geheugengebruik te optimaliseren.
1.	Multiprocessing: Elk peilgebied kan doorgerekend worden met een ander proces. Het aantal mogelijke processen is afhankelijk van het aantal beschikbare processen op je computer.
2.	Optimized multiprocessing (-run_mp_optimized): De peilgebieden worden opgedeeld in 2 klasses op basis van het oppervlak van het extent van een peilgebied (-mp_envelope_area_limit). De peilgebieden met het oppervlak kleiner dan de limiet worden in meerdere processen berekend. De peilgebieden groter dan de limiet worden getegeld berekend, de tegelgrootte kun je opgeven (-tile_size). De tegels zonder data, worden verwijderd. Dunne gebieden met een groot extent, zoals de boezem van HHNK kunnen daardoor efficiënt worden doorgerekend. 

Tijdens het hoofdproces kan de voorgang bekeken worden in de command prompt of python en in het logging bestand (work/log). Na de hoofdberekening wordt de data weggeschreven in de eerder genoemde bestanden. De peilgebieden die mislukt zijn, zijn te vinden onder output/failures.gpkg.

**post-processing**

Als de post processing module (wss_curves_areas_post.py) wordt aangeroepen wordt de data van het hoofdproces eerst ingelezen. Vervolgens worden ook de peilgebieden, landgebruiksconversie tabel en aggregatie vector ingelezen.

Vervolgens worden de output bestanden van het hoofdproces eerst nabewerkt:
-	Schades en volumes worden lineair geïnterpoleerd (meestal van 10 cm naar 1 cm).
-	Schades en volumes worden omgerekend van peilstijging naar waterstanden
-	Schade per ha wordt berekend.
-	Schade per m3 wordt berekend.

Hierna worden de figuren per peilgebied aangemaakt zoals de schadecurve en de bergingscurve. De linkjes van deze figuren worden toegevoegd aan geopackages. De GeoPackage (peilgebieden.gpkg) kunnen ingeladen worden in QGIS om de figuren direct te bekijken (read_me_qgis_grafiek_lezen.txt) en de schadebedragen te kunnen zien.

Als er een aggregatie wordt gedaan, is dit de volgende stap. Het script loopt over de aggregatievector heen en maakt een mapje aan op basis van het opgegeven veld (aggregate_vector_layer_name). Per aggregatievlak worden peilgebieden geselecteerd op basis van een ‘within’ gis operatie. Er zit een kleine buffer om het aggregatievlak heen om ervoor te zorgen dat alle peilgebieden goed worden meegenomen. De selectie wordt weggeschreven als ‘selection.gpkg’. Het belangrijkste wat wordt weggeschreven is de schade per volume volgens 3 verschillende methodieken en de figuren.
1.	Afvoeren naar laagste peilvak (damage_lowest_area): Hierin gaan we ervan uit dat de neerslag in een gebied vrij kan afvoeren naar het laagst gelegen (bemalen) peilvak. Er geldt voor de hoger gelegen peilvakken dus geen afvoerbeperking of het is technisch niet mogelijk om de afvoer te beperken. De enige feedback vindt plaats als het peil in het laagste peilvak dusdanig stijgt tot boven het peil van hoger gelegen peilvakken, dat deze ook beginnen te bergen en er dus schade kan gaan optreden.
2.	Gelijke peilstijging per peilvak (damage_equal_depth): hierin stijgt het peil even snel in alle peilvakken van een polder. De schade van alle schadecurves per peilgebied worden opgeteld. 
3.	Vasthouden per peilvak (damage_own_retention): Hierin wordt neerslag niet afgevoerd maar vastgehouden per peilgebied. De peilstijging volgt uit de maaiveldcurve per peilvak en verschilt daardoor met de vorige variant.

Met de figuren wordt inzicht gegeven in waar de schade door ontstaat. Er wordt een viertal figuren gedefinieerd:
1.	Bergingscurve
2.	Landgebruikscurve: geeft inzicht in het landgebruiksoppervlak bij een bepaalde maaiveldhoogte
3.	Schade aggregatie: Schadecurves voor de 3 types aggregaties.
4.	Schadeverdeling: Een grafiek waarin duidelijk wordt welk landgebruik zorgt voor de grootste schade.

Hierna wordt de aggregatie GeoPackage aangemaakt, waarin de figuren kunnen worden gevisualizeerd in QGIS (read_me_qgis_grafiek_lezen.txt).

Als laatste wordt er een interactieve html kaart gemaakt met Folium (Schadecurves.html). Hierin zijn de kaarten interactief te bekijken inclusief extra kaarten zoals: Peilstijging schade boven 1000 euro, landgebruik met de meeste schade bij 0.5 meter peilstijging.

