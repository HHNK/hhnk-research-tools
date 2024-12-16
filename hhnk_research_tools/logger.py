# %%
"""
These functions allow for adding logging to the console.
This is applied by default in the __init__ of hhnk_research_tools. So when it is imported
in a project, the logging will be set according to these rules.
"""

import logging
import sys
from logging import *  # noqa: F401,F403 # type: ignore


def get_logconfig_dict(level_root="WARNING", level_dict=None, log_filepath=None):
    """Make a dict for the logging.

    Parameters
    ----------
    level_root : str
        Default log level, warnings are printed to console.
    level_dict : dict[level:list]
        e.g. {"INFO" : ['hhnk_research_tools','hhnk_threedi_tools']}
        Apply a different loglevel for these packages.
    log_filepath : str
        Option to write a log_filepath.
    """
    logconfig_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {
            "": {  # root logger
                "level": level_root,
                "handlers": ["debug_console_handler", "stderr"],  # , 'info_rotating_file_handler'],
            },
        },
        "handlers": {
            "null": {
                "class": "logging.NullHandler",
            },
            "debug_console_handler": {
                "level": "NOTSET",
                "formatter": "time_level_name",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "stderr": {
                "level": "ERROR",
                "formatter": "time_level_name",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "formatters": {
            "time_level_name": {
                "format": "%(asctime)s|%(levelname)-8s| %(name)s:%(lineno)-4s| %(message)s",
                "datefmt": "%H:%M:%S",
            },
            # "error": {"format": "%(asctime)s-%(levelname)s-%(name)s-%(process)d::%(module)s|%(lineno)s:: %(message)s"},
        },
    }

    # Apply a different loglevel for these packages.
    if level_dict:
        for loglevel, level_list in level_dict.items():
            if not isinstance(level_list, list):
                raise TypeError("Level_dict should provide lists.")

            for pkg in level_list:
                logconfig_dict["loggers"][pkg] = {
                    "level": loglevel,
                    "propagate": False,
                    "handlers": ["debug_console_handler"],
                }

    if log_filepath:
        # Not possible to add a default filepath because it would always create this file,
        # even when nothing is being written to it.
        logconfig_dict["handlers"]["info_rotating_file_handler"] = {
            "level": "INFO",
            "formatter": "time_level_name",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "when": "D",
            "backupCount": 7,
            "filename": log_filepath,
        }
    return logconfig_dict


def set_default_logconfig(level_root="WARNING", level_dict=None, log_filepath=None):
    """Use this to set the default config, which will log to the console.

    In the __init__.py of hrt the hrt logger is initiated. We only need logging.GetLogger to add
    loggers to functions and classes. Same can be done for other packages.
    Use this in functions:

    import hhnk_research_tools as hrt
    logger = hrt.logging.get_logger(name=__name__, level='INFO')

    Example changing the default behaviour:
    hrt.logging.set_default_logconfig(
        level_root="WARNING",
        level_dict={
            "DEBUG": ["__main__"],
            "INFO": ["hhnk_research_tools", "hhnk_threedi_tools"],
            "WARNING": ['fiona', 'rasterio']
        },
    )
    """
    log_config = get_logconfig_dict(level_root=level_root, level_dict=level_dict, log_filepath=log_filepath)

    logging.config.dictConfig(log_config)


def get_logger(name: str, level=None):
    """
    Name should default to __name__, so the logger is linked to the correct file

    When using in a (sub)class, dont use this function. The logger will inherit the settings.
    Use:
        self.logger = hrt.logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    Othwerise:
        logger = hrt.logging.get_logger(name=__name__, level='INFO')

    Parameters
    ----------
    name : str
        Default use
        name = __name__
    level : str
        Only use this when debugging. Otherwise make the logger inherit the level from the config.
        When None it will use the default from get_logconfig_dict.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
