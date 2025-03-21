# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:07:02 2024

@author: kerklaac5395
"""
import json
import datetime
import numpy as np


class WSSTimelog:
    def __init__(self, subject, quiet, output_dir=None, log_file=None):
        self.s = subject
        self.quiet = quiet
        self.start_time = datetime.datetime.now()
        self.output_dir = output_dir
        self.use_logging = output_dir is not None or log_file is not None

        if self.use_logging:
            self.log_file = log_file

            if log_file is None:
                now = datetime.datetime.now().strftime("%Y%m%d%H%M")
                log_dir = self.output_dir / "log"
                log_dir.mkdir(exist_ok=True, parents=True)
                self.log_file = log_dir / f"{now} - {subject}.log"

            self.logger = create_logger(self.log_file)

    @property
    def time_since_start(self):
        delta = datetime.datetime.now() - self.start_time
        return delta

    def _message(self, msg):
        now = str(datetime.datetime.now())[:19]
        if not self.quiet:
            print(self.s, f"{now} [since start: {str(self.time_since_start)[:7]}]", msg)

        if self.use_logging:
            self.logger.info(msg)


def create_logger(filename):
    import multiprocessing, logging

    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have
    # duplicated messages in the output
    if not len(logger.handlers):
        logger.addHandler(handler)
    return logger


def write_dict(dictionary, path):
    with open(str(path), "w") as fp:
        json.dump(dictionary, fp)

def pad_zeros(a, shape):
    z = np.zeros(shape)
    z[: a.shape[0], : a.shape[1]] = a
    return z
