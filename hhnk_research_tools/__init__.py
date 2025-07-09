# hhnk_research_tools/__init__.py
import time

t0 = time.time()

import lazy_loader as _lazy
from importlib.metadata import version

# Get version from the pyproject.toml
__version__ = version("hhnk_research_tools")
__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)


print(f"Load1 in {(time.time() - t0):.3f} s")

# Set default logging to console
from .logger import set_default_logconfig

# FIXME not ideal to do here. can we do this on the level of caller packages?
set_default_logconfig(
    level_root="WARNING",
    level_dict={
        "DEBUG": ["__main__"],
        "INFO": ["hrt", "htt"],
        "ERROR": ["fiona", "rasterio"],
    },
)

# FIXME not ideal to do here. can we do this lazily?
import hhnk_research_tools.installation_checks


__doc__ = """
General toolbox for loading, converting and saving serval datatypes.
"""
print(f"Load2 in {(time.time() - t0):.3f} s")
