# %%
# if __name__ == "__main__":
#     import set_local_paths  # add local git repos.

import importlib
from dataclasses import dataclass

import pytest

import hhnk_research_tools as hrt
import hhnk_research_tools.folder_file_classes.file_class as fcl
import hhnk_research_tools.folder_file_classes.folder_file_classes as ffcl
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY

importlib.reload(fcl)
importlib.reload(ffcl)
if __name__ == "__main__":  # check if python is configed correctly
    print(hrt.__file__)


# TODO tests uitbreiden
def test_file():
    file_path = TEMP_DIR / f"testfile_{hrt.get_uuid()}.txt"
    file_path.write_text("test")

    file = fcl.File(file_path)
    assert file.exists() is True


def test_verify_exists():
    """"""
    # %%
    label_shape = hrt.FileGDB(TEST_DIRECTORY / r"area_test_labels.gpkg")
    label_raster = hrt.Raster(TEST_DIRECTORY / r"area_test_labels.tif")
    nonexisting_raster = hrt.Raster(TEST_DIRECTORY / r"thisrasterdoesntexist.tif")

    @dataclass
    class TstCls:
        label_shape: hrt.File
        label_raster: hrt.File

    tstcls = TstCls(label_shape=label_shape, label_raster=label_raster)

    assert hrt.Folder.verify_exists(cls=tstcls, keys=["label_shape", "label_raster"]) == True
    assert hrt.Folder.verify_exists(files=[label_shape, label_raster]) == True

    with pytest.raises(FileNotFoundError):
        hrt.Folder.verify_exists(files=[label_shape, label_raster, nonexisting_raster])


# %%
def test_folder():
    folder = ffcl.Folder(TEMP_DIR)

    assert folder.exists() is True


# %%
if __name__ == "__main__":
    test_file()
    test_folder()
