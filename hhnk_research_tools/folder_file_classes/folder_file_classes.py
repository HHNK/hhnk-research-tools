# %%
import warnings
from pathlib import Path
from typing import Optional, Union

from hhnk_research_tools import Raster
from hhnk_research_tools.folder_file_classes.file_class import BasePath, File
from hhnk_research_tools.folder_file_classes.spatial_database_class import SpatialDatabase
from hhnk_research_tools.folder_file_classes.sqlite_class import Sqlite
from hhnk_research_tools.general_functions import get_functions, get_variables

# %%


class Folder(BasePath):
    """Base folder class for creating, deleting and see if folder exists"""

    def __init__(self, base, create=False):
        super().__init__(base)

        self.files = {}
        self.olayers = {}
        self.space = "\t\t\t\t"
        if create:
            self.mkdir(parents=False)

    # TODO is deze nog nodig??
    @property
    def structure(self):
        return ""

    @property
    def content(self) -> list:
        return [i for i in self.path.glob("*")]

    # @property
    # def paths(self):
    #     """Get all properties that are a subclass of BasePath."""
    #     return [
    #         i for i in get_variables(self, stringify=False) if issubclass(type(getattr(self, i)), BasePath)
    #         ]

    # @property
    # def files_list(self):
    #     """If a path is not a folder, it is a file :-)."""
    #     return [i for i in self.paths if i not in self.folders]

    # @property
    # def folders_list(self):
    #     """Check if paths are instance of Folder."""
    #     return [i for i in self.paths if not isinstance(getattr(self, i), Folder)]

    @property
    def parent(self):
        """Return hrt.Folder instance. Import needs to happen here
        to prevent circular imports.
        """
        return Folder(self.path.parent)

    def __getitem__(self, attribute):
        return getattr(self, attribute)

    #Gives an recursion error
    #def __setattr__(self, name, value):
    #    return setattr(self, name, value)

    def create(self, parents=False, verbose=False):
        """Create folder, if parents==False path wont be
        created if parent doesnt exist.
        """
        warnings.warn(
            ".create is deprecated v2024.1 and will be removed in a future release. Please use .mkdir instead",
            DeprecationWarning,
            stacklevel=2,
        )

        self.mkdir(parents=parents, verbose=verbose)

    def mkdir(self, parents: bool = False, verbose: bool = False, exist_ok: bool = True):
        """Create folder and parents

        Parameters
        ----------
        parents : bool, default is True
            False -> dont create if parent don't exist
            True -> also create parent dirs
        verbose : bool, default is True
            True -> Print output
        exist_ok : bool, default is True
            False -> raises error when folder folder already exists
        """
        if not parents:
            if not self.parent.exists():
                if verbose:
                    print(f"'{self.path}' not created, parent does not exist.")
                return
        self.path.mkdir(parents=parents, exist_ok=exist_ok)
        return self.full_path(parents)

    def find_ext(self, ext: list):
        """Find files with a certain extension"""
        if isinstance(ext, str):
            ext = [ext]
        file_list = []
        for e in ext:
            file_list += [i for i in self.path.glob(f"*.{e.replace('.', '')}")]
        return file_list

    def joinpath(self, *args) -> Path:
        return self.path.joinpath(*args)

    def full_path(
        self, name, return_only_file_class: bool = False
    ):  # -> Union[File, Folder, SpatialDatabase, Raster, Sqlite]:
        """
        Return the full path of a file or a folder when only a name is known.
        Will return the object based on suffix

        Parameters
        ----------
        return_only_file_class : bool
            Only return file class, can speed up some functions because hrt.Raster
            initialization takes some time.

        Returns
        -------
        Union[File, Folder, SpatialDatabase, Raster, Sqlite]
        """
        name = str(name)
        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        filepath = self.joinpath(name)

        if name in [None, ""]:
            new_file = self
        elif return_only_file_class:
            new_file = File(filepath)
        else:
            if filepath.suffix == "":
                new_file = Folder(filepath)
            elif filepath.suffix in [".gdb", ".gpkg", ".shp"]:
                new_file = SpatialDatabase(filepath)
            elif filepath.suffix in [".tif", ".tiff", ".vrt"]:
                new_file = Raster(filepath)
            elif filepath.suffix in [".sqlite"]:
                new_file = Sqlite(filepath)
            else:
                new_file = File(filepath)
        return new_file

    def add_file(self, objectname, filename):
        """Add file as attribute. type is determined by filename extension."""
        new_file = self.full_path(filename)

        self.files[objectname] = new_file
        setattr(self, objectname, new_file)
        return new_file

    def unlink_contents(
        self,
        names: Optional[list[str]] = None,
        rmfiles: bool = True,
        rmdirs: bool = False,
    ) -> None:
        """Unlink all content when names is an empty list.
        Otherwise just remove the names.

        Parameters
        ----------
        names : list[str], optional, default: None
            Names of files or directories to unlink
        rmfiles : bool, optional, default: True
            Whether to remove regular files
        rmdirs : bool, optional, default: False
            Whether to remove empty directories

        """
        if names is None:
            names = self.content
        for name in names:
            pathname = self.path / name
            try:
                if pathname.exists():
                    # FIXME rmdir is only allowed for empty dirs
                    # can use shutil.rmtree, but this can be dangerous,
                    # not sure if we should support that here.
                    if pathname.is_dir():
                        if rmdirs:
                            pathname.rmdir()
                    else:
                        if rmfiles:
                            pathname.unlink()
            except Exception as e:
                print(pathname, e)

    def __repr__(self):
        # FIXME dit gaat mis bij @properties. hrt.ThreediResult aggregate_grid toegevoegd en die crashed de kernel
        # Het lijkt me dat de paths de property opent, zonder dat we dit willen.
        paths = [i for i in get_variables(self, stringify=False) if issubclass(type(getattr(self, i)), BasePath)]
        folders = [i for i in paths if isinstance(getattr(self, i), Folder)]
        files = [i for i in paths if i not in folders]
        repr_str = f"""{self.path.name} @ {self.path}
Exists: {self.exists()}
type: {type(self)}
    Folders:\t{folders}
    Files:\t{files}
functions: {get_functions(self)}
variables: {get_variables(self)}"""
        return repr_str
