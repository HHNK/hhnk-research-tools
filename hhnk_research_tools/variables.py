import numpy as np
import rasterio as rio
from osgeo import gdal

# default_variables
DEF_GEOMETRY_COL = "geometry"
DEF_TRGT_CRS = 28992
DEF_SRC_CRS = 4326
DEF_DELIMITER = ","
DEF_ENCODING = "utf-8"

# Download from http://www.gaia-gis.it/gaia-sins/windows-bin-amd64/ into anaconda installation
# Needs to be added to path before using.
MOD_SPATIALITE_PATH = r"C:\ProgramData\Anaconda3\mod_spatialite-5.0.1-win-amd64"


# definitions
WKT = "wkt"
GPKG_DRIVER = "GPKG"
ESRI_DRIVER = "ESRI Shapefile"
OPEN_FILE_GDB_DRIVER = "OpenFileGDB"

# types
#   Output file types: to prevent typo's and in case of remapping
TIF = "tif"
CSV = "csv"
QML = "qml"
TEXT = "txt"
SHAPE = "shp"
SQL = "sql"
GEOTIFF = "GTiff"
GPKG = "gpkg"
H5 = "h5"
NC = "nc"
SQLITE = "sqlite"
GDB = "gdb"
SHX = "shx"
DBF = "dbf"
PRJ = "prj"
GDAL_DATATYPE = gdal.GDT_Float32
file_types_dict = {
    "csv": ".csv",
    "txt": ".txt",
    "shp": ".shp",
    "sql": ".sql",
    "sqlite": ".sqlite",
    "tif": ".tif",
    "gdb": ".gdb",
    "gpkg": ".gpkg",
    "qml": ".qml",
    "h5": ".h5",
    "nc": ".nc",
    "shx": ".shx",
    "dbf": ".dbf",
    "prj": ".prj",
}
UTF8 = "utf-8"

# Gridadmin variable names
all_1d = "1D_ALL"
all_2d = "2D_ALL"

# rain
t_0_col = "t_0"
t_start_rain_col = "t_start_rain"
t_end_rain_min_one_col = "t_end_rain_min_one"
t_end_rain_col = "t_end_rain"
t_end_sum_col = "t_end_sum"
t_index_col = "value"

# Results
one_d_two_d = "1d2d"


DEFAULT_NODATA_VALUES = {
    "int8": np.iinfo(np.int8).min,
    "int16": np.iinfo(np.int16).min,
    "int32": np.iinfo(np.int32).min,
    "uint8": np.iinfo(np.uint8).max,
    "uint16": np.iinfo(np.uint16).max,
    "uint32": np.iinfo(np.uint32).max,
    "float32": np.nan,
    "float64": np.nan,  # Or np.finfo(np.float64).min
    "bool": False,
}  # TODO the gdal datatypes to numpy name conversion is found here; rasterio.dtypes.dtype_fwd
GDAL_DTYPES = rio.dtypes.dtype_fwd  # dict with keys=gdal int, value=str dtype

DEFAULT_NODATA_GENERAL = -9999
