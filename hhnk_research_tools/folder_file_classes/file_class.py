import json
from pathlib import Path
from typing import Optional

from hhnk_research_tools.general_functions import (
    ensure_file_path,
    get_functions,
    get_variables,
)


class BasePath:
    """pathlib.path like object that is used as base in File and Folder classes"""

    def __init__(self, base=None):
        self._base = base
        self.path = Path(str(base)).absolute().resolve()

    # decorated properties
    @property
    def base(self) -> str:
        """Path as posix string (foreward slashes)"""
        return self.path.as_posix()

    @property
    def name(self) -> str:
        """Name with suffix"""
        return self.path.name

    # TODO remove in future release
    @property
    def pl(self):
        import warnings

        warnings.warn(
            ".pl is deprecated since v2023.4 and will be removed in a future release. Please use .path instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.path

    @property
    def path_if_exists(self) -> Optional[str]:
        """Return filepath if the file exists otherwise return None"""
        if self.exists():
            return str(self.path)
        return None

    # def is_file(self):
    #     return self.path.suffix != ""

    def exists(self) -> bool:
        """Dont return true on empty path."""
        if not self._base:
            return False
        return self.path.exists()

    def __str__(self):
        return self.base


class File(BasePath):
    """pathlib.Path like file object"""

    def __init__(self, base):
        super().__init__(base)

    # Path properties
    @property
    def stem(self) -> str:  # stem (without suffix)
        return self.path.stem

    @property
    def suffix(self) -> str:
        return self.path.suffix

    def unlink(self, missing_ok: bool = True) -> None:
        self.path.unlink(missing_ok=missing_ok)

    def read_json(self) -> str:
        if self.path.suffix == ".json":
            return json.loads(self.path.read_text())
        raise TypeError(f"{self.name} is not a json.")

    def ensure_file_path(self):
        ensure_file_path(self.path)

    @property
    def parent(self):
        """Return hrt.Folder instance. Import needs to happen here
        to prevent circular imports.
        """
        from hhnk_research_tools.folder_file_classes.folder_file_classes import Folder

        return Folder(self.path.parent)

    def __repr__(self) -> str:
        repr_str = f"""{self.path.name} @ {self.path}
exists: {self.exists()}
type: {type(self)}
functions: {get_functions(self)}
variables: {get_variables(self)}
"""
        return repr_str

    def view_name_with_parents(self, parents: int = 0) -> str:
        """Display name of file with number of parents

        parents (int): defaults to 0
            number of parents to show
        """
        parents = min(len(self.path.parts) - 2, parents)  # avoids index-error
        return self.base.split(self.path.parents[parents].as_posix(), maxsplit=1)[-1]
