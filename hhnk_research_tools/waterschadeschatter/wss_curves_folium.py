# %%
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import branca
import branca.colormap as cm
import folium
import geopandas as gpd
from folium.plugins import MarkerCluster
from jinja2 import Template
from matplotlib import colors as mcolors

import hhnk_research_tools.logging as logging

logger = logging.get_logger(__name__, level="DEBUG")


class DummyColormap:
    """
    Dummy class to represent a colormap when no colormap is needed.
    This is used to avoid errors when no colormap is specified.
    """

    def __init__(self, colormap_dict: Dict[Any, str], name: str) -> None:
        self.colormap_dict = colormap_dict
        self.name = name

    def get_name(self) -> str:
        return self.name

    def __call__(self, value: Any) -> str:
        return self.colormap_dict[value]


class WSSCurvesFolium:
    def __init__(self) -> None:
        """Initialize WSSCurvesFolium with empty map and layer collections."""
        self.create_map()
        self.layers: List[folium.GeoJson] = []
        self.color_maps: List[Union[DummyColormap, cm.LinearColormap]] = []
        self.color_map_binds: List[BindColormap] = []
        self.m: folium.Map

    def create_map(self) -> None:
        """Create base Folium map with Dutch aerial imagery background."""

        self.m = folium.Map(
            location=[52.8, 4.9],
            tiles="nlmaps.luchtfoto",
            zoom_start=10,
            attr="<a href=https://nlmaps.nl/>NL Maps luchtfoto</a>",
        )

    def get_colormap(
        self,
        label: str,
        data_min: float,
        data_max: float,
        gdf: gpd.GeoDataFrame,
        colormap_field: Optional[str] = None,
        colormap_name: str = "plasma",
        nr_steps: int = 5,
        colormap_type: str = "categorical",
    ) -> Union[DummyColormap, cm.LinearColormap]:
        """
        Get a colormap that either creates a gradient for continuous data or unique colors for categorical data.

        Args:
            label (str): Label for the colormap
            data_min: Minimum value for numeric data
            data_max: Maximum value for numeric data
            gdf: gpd.GeoDataFrame containing the data
            colormap_field: Field to use for determining unique values
            colormap_name (str): Name of matplotlib colormap or "custom" for custom colors
            nr_steps (int): Number of steps for continuous data
            colormap_type (str): "continuous" or "categorical"
            custom_colors (dict): Dictionary mapping values to matplotlib color names or hex codes
        """
        if colormap_type == "categorical":
            # If colormap_field is provided, use it to determine the unique values
            colormap = {}
            for color in gdf[colormap_field]:
                if color in mcolors.CSS4_COLORS or color in mcolors.TABLEAU_COLORS:
                    # Convert named color to hex
                    hex_color = mcolors.to_hex(color)
                else:
                    # Use the color as is (assuming it's already hex)
                    hex_color = color if color else "#808080"  # fallback to gray
                    print("Fallback to gray for color:", color)

                colormap[color] = hex_color  # RRGGBBAA
            colormap = DummyColormap(colormap, name=label)

        elif colormap_type == "continuous":
            # For continuous data, use branca's built-in colormaps
            colormap = getattr(cm.linear, colormap_name)
            colormap = colormap.scale(data_min, data_max)
            colormap = colormap.to_step(nr_steps)
        else:
            raise ValueError(f"Unknown colormap type: {colormap_type}. Use 'categorical' or 'continuous'.")

        colormap.caption = label
        return colormap

    def add_water_layer(self) -> None:
        """Add Dutch water layer as tile overlay to the map."""
        folium.TileLayer("nlmaps.water", attr="<a href=https://nlmaps.nl/>NL Maps water</a>").add_to(self.m)

    def add_border_layer(self, name: str, gdf: gpd.GeoDataFrame, tooltip_fields: List[str], show: bool = True) -> None:
        """Add border layer with transparent fill and black outline to the map."""
        border_style = {"color": "#000000", "weight": "1.5", "fillColor": "#58b5d1", "fillOpacity": 0.08}

        layer = folium.GeoJson(
            gdf,
            style_function=lambda x: border_style,
            name=name,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_fields,
            ),
            control=True,
            show=show,
            overlay=True,
        )
        # Store the show parameter on the layer object
        layer.show = show
        self.layers.append(layer)

    def add_layer(
        self,
        name: str,
        gdf: gpd.GeoDataFrame,
        datacolumn: str,
        tooltip_fields: List[str],
        data_min: float,
        data_max: float,
        colormap_name: str,
        colormap_type: str = "categorical",
        show: bool = False,
        show_colormap: bool = True,
    ) -> None:
        """Add a styled layer to the map with colormap based on a data column."""

        colormap = self.get_colormap(
            label=f"Legend {name}",
            data_min=data_min,
            data_max=data_max,
            colormap_name=colormap_name,
            colormap_type=colormap_type,
            colormap_field=datacolumn,
            gdf=gdf,
        )

        layer = folium.GeoJson(
            gdf,
            style_function=lambda feature: {
                "fillColor": colormap(feature["properties"][datacolumn]),
                "color": "white",
                "fillOpacity": 1,
                "weight": 1,
            },
            name=name,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_fields,
            ),
            show=show,
            control=True,
            overlay=True,
        )
        # Store the show parameter on the layer object
        layer.show = show
        self.layers.append(layer)

        if show_colormap:
            self.color_maps.append(colormap)
            self.color_map_binds.append(BindColormap(layer, colormap))

    def add_graphs(
        self, name: str, gdf: gpd.GeoDataFrame, image_field: str, width: int = 520, height: int = 520
    ) -> None:
        """Add interactive image popups to the map as marker clusters."""

        marker_cluster = MarkerCluster(name=name, overlay=True, control=True, icon_create_function=None)

        for idx, data in gdf.iterrows():
            centroid = data.geometry.centroid
            point_gdf = gpd.GeoDataFrame(geometry=[centroid], crs=gdf.crs)

            # Reproject to WGS84
            point_wgs84 = point_gdf.to_crs(epsg=4326)

            # Get lat/lon coordinates
            lon, lat = point_wgs84.geometry.x.iloc[0], point_wgs84.geometry.y.iloc[0]

            encoded = base64.b64encode(open(data[image_field], "rb").read()).decode()
            html = f'<img src="data:image/png;base64,{encoded}" width="{width}" height="{height}">'
            iframe = folium.IFrame(html, width=f"{width}px", height=f"{height}px")
            popup = folium.Popup(iframe, max_width=width)

            icon = folium.Icon(color="red", icon="ok")
            marker = folium.Marker(location=[lat, lon], popup=popup, icon=icon)
            marker.add_to(marker_cluster)

        marker_cluster.add_to(self.m)

    def add_legend(self) -> None:
        """Add colormap legend to the map."""
        self.m.add_child(self.colormap)

    def add_title(self, title: str) -> None:
        """Add a title overlay to the map."""
        # Add title to map
        title_html = f'<h1 style="position:absolute;z-index:100000;bottom:1vw;background-color:rgba(255, 255, 255, 0.8);padding:10px;border-radius:5px;" >{title}</h1>'
        self.m.get_root().html.add_child(folium.Element(title_html))

    def save(self, output_path: Union[str, Path]) -> None:
        """Save the interactive map to a file with all layers and controls."""
        # Create layer control first
        layer_control = folium.LayerControl(collapsed=False, sort_layers=True, position="topright")

        # Add all layers to the control but only show them if show=True
        for l in self.layers:
            if hasattr(l, "show") and l.show:
                self.m.add_child(l)
            else:
                # Add to map but hide it
                l.add_to(self.m)
                self.m.get_root().html.add_child(
                    folium.Element(f"""
                    <script>
                        document.addEventListener("DOMContentLoaded", function() {{
                            map.removeLayer({l.get_name()});
                        }});
                    </script>
                """)
                )

        # Add the layer control after all layers
        self.m.add_child(layer_control)

        # Add colormaps and bindings
        for c in self.color_maps:
            self.m.add_child(c)

        for b in self.color_map_binds:
            self.m.add_child(b)

        logger.debug(f"Saving interactive map to: {output_path}")
        self.m.save(output_path)


class BindColormap(branca.element.MacroElement):  # from https://github.com/python-visualization/folium/issues/450
    """Binds a colormap to a given layer.

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    """

    def __init__(self, layer: folium.GeoJson, colormap: Union[DummyColormap, cm.LinearColormap]) -> None:
        super().__init__()
        self.layer = layer
        self.colormap = colormap
        self._template = Template("""
        {% macro script(this, kwargs) %}
            {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
            {{this._parent.get_name()}}.on('overlayadd', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
                }});
            {{this._parent.get_name()}}.on('overlayremove', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'none';
                }});
        {% endmacro %}
        """)  # noqa
