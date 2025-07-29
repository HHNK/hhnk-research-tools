# hhnk_research_tools/__init__.py
import lazy_loader as _lazy
from importlib.metadata import version


# hhnk_research_tools/__init__.pyi
# ... all other symbols you expose for typing
__version__ = version("hhnk_research_tools")
__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

# Set default logg ing to console
from .logging import set_default_logconfig

# FIXME not ideal to do here. can we do this on the level of caller packages?
set_default_logconfig(
    level_root="WARNING",
    level_dict={
        "DEBUG": ["__main__"],
        "INFO": ["hrt", "htt"],
        "ERROR": ["fiona", "rasterio"],
    },
)


__doc__ = """
General toolbox for loading, converting and saving serval datatypes.
"""
