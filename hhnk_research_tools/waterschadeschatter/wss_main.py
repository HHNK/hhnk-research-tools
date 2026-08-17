# %%
from pathlib import Path

from osgeo import gdal
from tqdm import tqdm

import hhnk_research_tools as hrt
import hhnk_research_tools.waterschadeschatter.wss_calculations as wss_calculations
import hhnk_research_tools.waterschadeschatter.wss_loading as wss_loading
from hhnk_research_tools.gis.raster import RasterOld as Raster

# from hhnk_research_tools.rasters.raster_class import Raster #FIXME new import

gdal.UseExceptions()

DMG_NODATA = 0


class Waterschadeschatter:
    """Waterschadeschatter berekening van de schade bij een bepaalde inundatiediepte per
    landgebruiksfunctie.
    wss_settings heeft onderstaand format. De naamgeving van herstelperiode en maand moet
    overeenkomen met de .cfg. De duur is een integer in uren. In de tabel vindt daarbij
    lineare interpolatie plaats.
    wss_settings = {'inundation_period': 48, #uren (optioneel als inundation_period_file wordt gegeven)
                'herstelperiode':'10 dagen',
                'maand':'sep',
                'cfg_file':cfg_file,
                'dmg_type':'gem'}

    dmg_type in ["min", "gem", "max"]

    LET OP: depth_file heeft dieptes nodig vanaf -0.01cm, zoals in .cfg ook staat. Dit moet
    dus ook meegenomen bij de vertaling van waterstand naar waterdiepte.

    inundation_period kan zowel als raster (inundation_period_file parameter) of als globale
    waarde (wss_settings['inundation_period']) worden meegegeven. Als inundation_period_file
    wordt gebruikt, wordt de waarde in wss_settings genegeerd.
    """

    def __init__(
        self,
        depth_file,
        landuse_file,
        wss_settings=None,
        inundation_period_file=None,
        min_block_size=2048,
        resample_landuse=False,
    ):
        self.wss_settings = wss_settings
        self.min_block_size = min_block_size
        self.resample_landuse = resample_landuse

        self.lu_raster = landuse_file
        if not isinstance(landuse_file, Raster):
            self.lu_raster = Raster(landuse_file)

        if isinstance(depth_file, Raster):
            self.depth_raster = depth_file
            self.depth_raster.min_block_size = min_block_size
        else:
            self.depth_raster = Raster(depth_file, self.min_block_size)

        # Handle inundation_period_file (can be Raster object or file path)
        self.inundation_period_raster = None
        if inundation_period_file is not None:
            if isinstance(inundation_period_file, Raster):
                self.inundation_period_raster = inundation_period_file
                self.inundation_period_raster.min_block_size = min_block_size
            else:
                self.inundation_period_raster = Raster(inundation_period_file, self.min_block_size)

        self.gamma_inundatiediepte = None

        self.validate()

        # Resample landuse to match depth resolution if requested
        if self.resample_landuse:
            self._resample_landuse_to_depth()

        # Inladen configuratie
        if self.wss_settings is not None:
            self.load_damage_table()

    def validate(self):
        """Check if input exists"""
        for r in [self.lu_raster, self.depth_raster]:
            if not r.exists():
                raise Exception(f"could not find input file in: {r}")

        if self.inundation_period_raster is not None:
            if not self.inundation_period_raster.exists():
                raise Exception(f"could not find inundation_period file in: {self.inundation_period_raster}")

    def _resample_landuse_to_depth(self):
        """Resample landuse raster to match depth raster resolution using nearest neighbor."""
        import tempfile

        lu_pixelwidth = self.lu_raster.metadata["pixel_width"]
        depth_pixelwidth = self.depth_raster.metadata["pixel_width"]

        print(f"Resampling landuse from {lu_pixelwidth}m to {depth_pixelwidth}m resolution...")

        # Create temporary resampled file
        temp_dir = Path(tempfile.gettempdir())
        temp_resampled = temp_dir / f"landuse_resampled_{depth_pixelwidth}m.tif"

        # Use GDAL to resample (nearest neighbor for landuse classification)
        from osgeo import gdal

        # Get depth raster metadata
        depth_meta = self.depth_raster.metadata

        # Resample using gdalwarp
        options = gdal.WarpOptions(
            format="GTiff",
            xRes=depth_meta["pixel_width"],
            yRes=depth_meta["pixel_height"],
            resampleAlg="near",  # nearest neighbor for categorical data
            outputBounds=(
                depth_meta["x_min"],
                depth_meta["y_min"],
                depth_meta["x_max"],
                depth_meta["y_max"],
            ),
            dstSRS=depth_meta["proj"],
        )

        gdal.Warp(str(temp_resampled), str(self.lu_raster.path), options=options)

        # Replace lu_raster with resampled version
        self.lu_raster = Raster(temp_resampled)
        print(f"Landuse resampled to {temp_resampled}")

    def load_damage_table(self):
        # Inladen configuratie
        self.dmg_table_landuse, self.dmg_table_general = wss_loading.read_dmg_table_config(self.wss_settings)

        # Get indices
        self.indices = self.get_dmg_table_indices()

    def get_dmg_table_indices(self):
        """Check the index in the table using the input values for herstelperiode and maand."""
        indices = {}
        indices["herstelperiode"] = self.dmg_table_general["herstelperiode"].index(self.wss_settings["herstelperiode"])
        indices["maand"] = self.dmg_table_general["maand"].index(self.wss_settings["maand"])
        return indices

    def run(
        self,
        output_raster: Raster,
        calculation_type="sum",
        verbose=False,
        overwrite=False,
    ):
        """Calculation type options: 'sum','direct','indirect'"""

        if output_raster.exists():
            if overwrite is False:
                return
            else:
                output_raster.path.unlink()

        output_raster = hrt.RasterOld(output_raster)  # TODO wss omzetten naar rxr

        # Create output raster
        output_raster.create(
            metadata=self.depth_raster.metadata, nodata=DMG_NODATA, verbose=verbose, overwrite=overwrite
        )

        # Load raster so we can edit it.
        target_ds = output_raster.open_gdal_source_write()
        dmg_band = target_ds.GetRasterBand(1)

        # Difference between landuse and depth raster.
        dx_min, dy_min, dx_max, dy_max = hrt.dx_dy_between_rasters(
            meta_big=self.lu_raster.metadata, meta_small=self.depth_raster.metadata
        )

        pixel_factor = self.depth_raster.pixelarea
        blocks_df = self.depth_raster.generate_blocks()

        len_total = len(blocks_df)
        for idx, block_row in tqdm(blocks_df.iterrows(), total=len(blocks_df)):
            # Load landuse
            window_depth = block_row["window_readarray"]

            window_lu = window_depth.copy()
            window_lu[0] += dx_min
            window_lu[1] += dy_min

            lu_block = self.lu_raster._read_array(window=window_lu)
            lu_block = lu_block.astype(int)
            lu_block[lu_block == self.lu_raster.nodata] = 0
            # TODO np.all(self.polder==folder.dst.tmp.polder.nodata) is mogelijk net iets sneller.
            if lu_block.mean() != 0:
                # Load depth
                depth_block = self.depth_raster._read_array(window=window_depth)
                # depth_mask = depth_block==self.depth_raster.nodata
                # depth_block[depth_mask] = np.nan #Schadetabel loopt vanaf -0.01cm

                # Load inundation_period if raster is provided
                inundation_period_block = None
                if self.inundation_period_raster is not None:
                    inundation_period_block = self.inundation_period_raster._read_array(window=window_depth)

                # Calculate damage
                damage_block = wss_calculations.calculate_damage(
                    caller=self,
                    lu_block=lu_block,
                    depth_block=depth_block,
                    inundation_period_block=inundation_period_block,
                    indices=self.indices,
                    dmg_table_landuse=self.dmg_table_landuse,
                    dmg_table_general=self.dmg_table_general,
                    pixel_factor=pixel_factor,
                    calculation_type=calculation_type,
                )
                # Write to file
                dmg_band.WriteArray(damage_block, xoff=window_depth[0], yoff=window_depth[1])

            # print(f"{idx} / {len_total}", end="\r")
            # break

        dmg_band.FlushCache()  # close file after writing
        dmg_band = None
        target_ds = None

    def __repr__(self):
        """List available objects, distinction between functions and variables"""
        funcs = "." + " .".join(
            [i for i in dir(self) if not i.startswith("__") and hasattr(getattr(self, i), "__call__")]
        )
        variables = "." + " .".join(
            [i for i in dir(self) if not i.startswith("__") and not hasattr(getattr(self, i), "__call__")]
        )
        repr_str = f"""functions: {funcs}
variables: {variables}"""
        return repr_str


if __name__ == "__main__":
    if True:
        # Main folder containing scenario subfolders
        main_folder = Path(r"G:\02_Werkplaatsen\04_GIS\WSS regelmacht Schermer\naar mezelf sturen")

        # Scenario names (subfolders)
        scenarios = [
            "basis_afdeling_1",
            "basis_afdeling_2",
            "basis_gebiedsbreed",
            "maatregel_afdeling_1",
            "maatregel_afdeling_2",
            "maatregel_gebiedsbreed",
        ]

        # Global configuration files
        cfg_file = r"E:\01.basisgegevens\hhnk_schadeschatter\01_data/cfg/cfg_hhnk_2020.cfg"
        landuse_file = r"E:\01.basisgegevens\rasters\landgebruik\landuse2023_tiles/combined_rasters.vrt"

        # File names (same for all scenarios, located in scenario subfolders)
        depth_filename = "max_wd.tif"
        output_filename = "wss_schades.tif"
        inundation_period_filename = "inundation_duration_1.0cm.tif"

        # WSS settings
        wss_settings = {
            # "inundation_period": 48,  # uren
            "herstelperiode": "10 dagen",
            "maand": "sep",
            "cfg_file": cfg_file,
            "dmg_type": "gem",
        }

        # Validate global files exist
        assert Path(cfg_file).exists(), f"cfg_file not found: {cfg_file}"
        assert Path(landuse_file).exists(), f"landuse_file not found: {landuse_file}"

        # First, validate all scenario files exist before processing
        print("Validating all scenario files...")
        scenario_files = {}
        for scenario in scenarios:
            scenario_folder = main_folder / scenario

            depth_file = scenario_folder / depth_filename
            output_file = scenario_folder / output_filename
            inundation_period_file = scenario_folder / inundation_period_filename

            # Check if all required files exist
            assert scenario_folder.exists(), f"Scenario folder not found: {scenario_folder}"
            assert depth_file.exists(), f"depth_file not found for scenario '{scenario}': {depth_file}"
            assert inundation_period_file.exists(), (
                f"inundation_period_file not found for scenario '{scenario}': {inundation_period_file}"
            )

            # Store file paths for this scenario
            scenario_files[scenario] = {
                "depth_file": depth_file,
                "output_file": output_file,
                "inundation_period_file": inundation_period_file,
            }

        print(f"All files validated successfully for {len(scenarios)} scenarios.\n")

        # Process each scenario
        for scenario in scenarios:
            print(f"Processing scenario: {scenario}")

            files = scenario_files[scenario]

            # Calculation
            wss = Waterschadeschatter(
                depth_file=files["depth_file"],
                landuse_file=landuse_file,
                wss_settings=wss_settings,
                inundation_period_file=files["inundation_period_file"],
                resample_landuse=True,  # Resample landuse to match depth resolution
            )

            # Berekenen schaderaster
            wss.run(
                output_raster=Raster(files["output_file"]),
                calculation_type="sum",
                verbose=True,
                overwrite=False,
            )

            print(f"Completed scenario: {scenario}\n")
