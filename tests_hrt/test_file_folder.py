# %%
# if __name__ == "__main__":
#     import set_local_paths  # add local git repos.

import importlib

import hhnk_research_tools as hrt
import hhnk_research_tools.folder_file_classes.file_class as fcl
import hhnk_research_tools.folder_file_classes.folder_file_classes as ffcl
from tests_hrt.config import TEMP_DIR

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


def test_folder():
    folder = ffcl.Folder(TEMP_DIR)

    assert folder.exists() is True


# %%
if __name__ == "__main__":
    test_file()
    test_folder()
