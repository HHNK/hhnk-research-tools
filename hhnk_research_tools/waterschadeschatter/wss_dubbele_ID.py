import pandas as pd
import geopandas as gpd

shape = gpd.read_file(r"C:\temp\HHNK\test_dubble_IDs.shp")
def check_dubble_ID(shape):
    dubbele_ID = shape["Id"].where(shape.duplicated('Id')).dropna().values
    if len(dubbele_ID) == 0:
        print("Geen dubbele ID's in shapefile")
    else:
        print(f"{len(dubbele_ID)} dubbele ID's in shapefile: {dubbele_ID}")

