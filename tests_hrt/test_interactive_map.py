# %% Code test
import geopandas as gpd

import hhnk_research_tools.logger as logging
from hhnk_research_tools.gis.interactive_map import create_interactive_map
from tests_hrt.config import TEMP_DIR, TEST_DIRECTORY

logger = logging.get_logger("hrt.gis.interactive_map", level="DEBUG")


def test_interactive_map():
    gdf = gpd.read_file(TEST_DIRECTORY.joinpath(r"area_test_labels.gpkg"))
    gdf["label"] = "label"
    datacolumn = "id"
    colormap_name = "viridis"
    colormap_steps = None
    output_path = TEMP_DIR.joinpath("interactive_map.html")
    title = "Interactive map test"
    legend_label = "Testlabels"
    tooltip_columns = ["id", "label"]
    tooltip_aliases = ["id", "Label"]
    quantiles = [0, 0.5, 0.8, 1]

    v = create_interactive_map(
        gdf=gdf,
        datacolumn=datacolumn,
        output_path=output_path,
        title=title,
        legend_label=legend_label,
        tooltip_columns=tooltip_columns,
        tooltip_aliases=tooltip_aliases,
        colormap_name=colormap_name,
        colormap_steps=colormap_steps,
        quantiles=quantiles,
    )

    assert output_path.exists()


# %%
if __name__ == "__main__":
    test_interactive_map()
