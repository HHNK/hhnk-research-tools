# %% Code test

import hhnk_research_tools.logging as logging
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY


def test_logger():
    """Test if logger works in shell / file"""
    # %%
    logfile = TEMP_DIR.joinpath("testlog.log")

    logger = logging.get_logger("test", level="DEBUG", filepath=logfile, filemode="w")
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")

    # A sublogger should also write to the same file handler
    logger2 = logging.get_logger("test.sublogger", level="ERROR")
    logger2.warning("info_sublogger")  # shouldnt log because of level
    logger2.error("error_sublogger")

    # Now a completely different logger shouldnt write to the same file handler
    logger3 = logging.get_logger("other.sublogger", level="WARNING")
    logger3.error("error_other")

    logtxt = logfile.read_text()
    assert "warning_sublogger" not in logtxt
    assert "error_sublogger" in logtxt
    assert "error_other" not in logtxt
