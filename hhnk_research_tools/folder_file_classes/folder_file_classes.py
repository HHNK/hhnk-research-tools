# %%
import types
import warnings
from pathlib import Path

import fiona
import geopandas as gpd

from hhnk_research_tools import Raster
from hhnk_research_tools.folder_file_classes.file_class import BasePath, File
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
    def content(self):
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
            False -> dont create if parent don't exist.
            True -> also create parents.
        verbose : bool, default is True
            False
            True -> Print output
        """
        if not parents:
            if not self.parent.exists():
                if verbose:
                    print(f"'{self.path}' not created, parent does not exist.")
                return
        self.path.mkdir(parents=parents, exist_ok=exist_ok)

    def find_ext(self, ext: list):
        """Find files with a certain extension"""
        if type(ext) == str:
            ext = [ext]
        file_list = []
        for e in ext:
            file_list += [i for i in self.path.glob(f"*.{e.replace('.', '')}")]
        return file_list

    def full_path(self, name, return_only_file_class=False):
        """
        Return the full path of a file or a folder when only a name is known.
        Will return the object based on suffix

        return_only_file_class (bool): only return file class, can speed up
            some functions because hrt.Raster initialization takes some time.
        """
        name = str(name)
        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        filepath = self.path.joinpath(name)

        if name in [None, ""]:
            new_file = self  # TODO was, Path(""), gaat dit goed?
        elif return_only_file_class:
            new_file = File(filepath)
        else:
            if filepath.suffix == "":
                new_file = Folder(filepath)
            elif filepath.suffix in [".gdb", ".gpkg", ".shp"]:
                new_file = FileGDB(filepath)
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

    def unlink_contents(self, names=[], rmfiles=True, rmdirs=False):
        """Unlink all content when names is an empty list.
        Otherwise just remove the names.
        """
        if not names:
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


class FileGDB(File):
    def __init__(self, base):
        super().__init__(base)

        self.layerlist = []
        self.layers = types.SimpleNamespace()  # empty class

    def load(self, layer=None):
        if layer is None:
            avail_layers = self.available_layers()
            if len(avail_layers) == 1:
                layer = avail_layers[0]
            else:
                layer = input(f"Select layer [{avail_layers}]:")
        return gpd.read_file(self.path, layer=layer, engine="pyogrio")

    def add_layer(self, name: str):
        """Predefine layers so we can write output to that layer."""
        if name not in self.layerlist:
            new_layer = FileGDBLayer(name, parent=self)
            self.layerlist.append(name)
            setattr(self.layers, name, new_layer)

    def add_layers(self, names: list):
        """Add multiple layers"""
        for name in names:
            self.add_layer(name)

    def available_layers(self):
        """Return available layers in file gdb"""
        return fiona.listlayers(self.base)

    def __repr__(self):
        repr_str = f"""{self.path.name} @ {self.path}
exists: {self.exists()}
type: {type(self)}

functions: {get_functions(self)}
variables: {get_variables(self)}
layers (access through .layers): {self.layerlist}"""
        return repr_str


class FileGDBLayer:
    def __init__(self, name: str, parent: FileGDB):
        self.name = name
        self.parent = parent

    def load(self):
        return gpd.read_file(self.parent.base, layer=self.name)
