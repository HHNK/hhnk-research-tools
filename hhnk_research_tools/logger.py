# %%

import inspect
import logging
import sys
from logging import *  # noqa: F401,F403 # type: ignore
from pathlib import Path

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "loggers": {
        "": {  # root logger
            "level": "NOTSET",
            "handlers": ["debug_console_handler"],  # , 'info_rotating_file_handler'],
        },
        "hhnk_research_tools": {
            "level": "INFO",
            "propagate": False,
            "handlers": ["debug_console_handler"],
        },
        "fiona": {  # Quiet by default!
            "handlers": ["null"],
            "propagate": False,
            "level": "ERROR",
        },
        "rasterio": {  # Quiet by default!
            "handlers": ["null"],
            "propagate": False,
            "level": "ERROR",
        },
    },
    "handlers": {
        "null": {
            "level": "INFO",
            "class": "logging.NullHandler",
        },
        "debug_console_handler": {
            "level": "DEBUG",
            "formatter": "info",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        # 'info_rotating_file_handler': {
        #     'level': 'INFO',
        #     'formatter': 'info',
        #     'class': 'logging.handlers.RotatingFileHandler',
        #     'filename': 'info.log',
        #     'mode': 'a',
        #     'maxBytes': 1048576,
        #     'backupCount': 10
        # },
    },
    "formatters": {
        "info": {
            "format": "%(asctime)s|%(levelname)-7s| %(name)s:%(lineno)-4s| %(message)s",
            "datefmt": "%H:%M:%S",
        },
        # "error": {"format": "%(asctime)s-%(levelname)s-%(name)s-%(process)d::%(module)s|%(lineno)s:: %(message)s"},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str, level=logging.INFO, format_short=False):
    """
    Name should default to __name__, so the logger is linked to the correct file

    When using in a (sub)class, dont use this function. The logger will inherit the settings.
    Use:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    In the __init__.py of hrt the hrt logger is initiated. We only need logging.GetLogger to add
    loggers to functions and classes. Same can be done for other packages.

    Parameters
    ----------
    name : str
        Default use
        name = __name__
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    date_format = "%Y-%m-%d %H:%M:%S"

    # Log console entry format:
    # [<level> ][ <logger> ] <message>
    date_format_short = "%H:%M:%S"

    stdout_log_format = "[%(levelname)-8s][ %(name)s :%(lineno)s ] %(message)s"
    stdout_log_format = "[%(asctime)s][%(levelname)-8s][ %(name)s:%(lineno)s ] %(message)s"

    if format_short is True:
        stdout_log_format = "%(levelname)-5s | %(message)s"
    else:
        stdout_log_format = "%(asctime)s|%(levelname)-7s| %(name)s:%(lineno)-4s| %(message)s"

    stdout_log_formatter = logging.Formatter(stdout_log_format, date_format_short, style="%")

    # With this formatting we can search the code in vs-code. Use ctrl+P -> copy filename:lineno -> enter
    # e.g. ctrl+P -> logger.py:51 -> will get you to this file
    file_log_format = "[{asctime}][%(levelname)-8s][ %(filename)s:%(lineno)s ] %(message)s"
    file_log_formatter = logging.Formatter(file_log_format, date_format)

    # file_log_format = "[%(levelname)-8s]" "[ %(name)s :%(lineno)s ] " "%(message)s"
    # file_log_formatter = logging.Formatter(file_log_format)

    # def add_stdout_handler(loglevel: int = logging.INFO):
    #   handler = logging.StreamHandler()
    #   handler.setLevel(loglevel)
    #   handler.setFormatter(stdout_log_formatter)
    #   root.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(stdout_log_formatter)

    # print(handler)
    # print(logger.handlers)

    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            # make sure no duplicate handlers are added
            if not isinstance(handler, logging.FileHandler) and not isinstance(handler, logging.StreamHandler):
                logger.addHandler(handler)
                # logger.info("added logger hashandlers")

    else:
        logger.addHandler(handler)
        # logger.info("added logger no handlers")

    return logger


# File logger example, unused.
# def file_log_filter(record):
#     if record.name == "root":
#         record.qualifiedFuncName = "." + record.funcName
#     else:
#         record.qualifiedFuncName = record.name + "." + record.funcName
#     return True


# # Log file entry format:
# # [<time>][<level> ][ <file>:<line> ][ <class>.<function>() ] <message>
# file_log_format = (
#     "[%(asctime)s]" "[%(levelname)-8s]" "[ %(filename)s:%(lineno)s ]" "[ %(qualifiedFuncName)s() ] " "%(message)s"
# )
# file_log_formatter = logging.Formatter(file_log_format, date_format)


# def add_file_handler(logfile: Path, loglevel: int = logging.DEBUG):
#     logfile.parent.mkdir(parents=True, exist_ok=True)
#     logfile.unlink(missing_ok=True)
#     handler = logging.FileHandler(str(logfile))
#     handler.setLevel(loglevel)
#     handler.addFilter(file_log_filter)
#     handler.setFormatter(file_log_formatter)
#     root.addHandler(handler)


# %%
