# %%
import shutil
from pathlib import Path

import hhnk_research_tools as hrt

TEST_DIRECTORY = Path(__file__).parent.absolute() / "data"


# TEMP_DIR = TEST_DIRECTORY/r"temp"
# if TEMP_DIR.exists():
#     shutil.rmtree(TEMP_DIR)
TEMP_DIR = hrt.Folder(TEST_DIRECTORY / r"temp", create=True)

TEMP_DIR.unlink_contents()
TEMP_DIR = TEMP_DIR.path
cont = False
for i in TEMP_DIR.iterdir():
    if i.is_dir:
        cont = True
        # for dirname in ["batch_test", "test_project_", "vrt_test", "temp_", "wss_curves"]:
        #     if dirname in str(i):
        #         cont = True

        if cont:
            try:
                shutil.rmtree(i)
            except:
                pass
