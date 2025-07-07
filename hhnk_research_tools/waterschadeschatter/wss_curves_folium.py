# %%
import base64
import colorsys
import numpy as np

# Third party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import folium
from folium import IFrame
from folium.plugins import MarkerCluster
import branca.colormap as cm
from branca.element import MacroElement
from jinja2 import Template
import geopandas as gp

# Local imports
import hhnk_research_tools.logger as logging

logger = logging.get_logger(__name__, level="DEBUG")

class DummyColormap:
    """
    Dummy class to represent a colormap when no colormap is needed.
    This is used to avoid errors when no colormap is specified.
    """
    def __init__(self, colormap_dict, name):
        self.colormap_dict = colormap_dict
        self.name = name

    def get_name(self):
        return self.name
    
    def __call__(self, value):
        return self.colormap_dict[value]
    
class WSSCurvesFolium:
    def __init__(self):
        self.create_map()
        self.layers = []
        self.color_maps = []
        self.color_map_binds = []

    def create_map(self):
        # Create the map
        # background maps at: https://leaflet-extras.github.io/leaflet-providers/preview/
        self.m = folium.Map(
            location=[52.8, 4.9],
            tiles="nlmaps.luchtfoto",
            zoom_start=10,
            attr="<a href=https://nlmaps.nl/>NL Maps luchtfoto</a>",
        )

    def get_colormap(self, label, data_min, data_max, gdf, colormap_field=None, colormap_name="plasma", nr_steps=5, colormap_type="categorical"):
        """
        Get a colormap that either creates a gradient for continuous data or unique colors for categorical data.
        
        Args:
            label (str): Label for the colormap
            data_min: Minimum value for numeric data
            data_max: Maximum value for numeric data
            gdf: GeoDataFrame containing the data
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
                    hex_color = color if color else '#808080'  # fallback to gray
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

    def add_water_layer(self):
        folium.TileLayer("nlmaps.water", attr="<a href=https://nlmaps.nl/>NL Maps water</a>").add_to(self.m)

    def add_border_layer(self, name, gdf, tooltip_fields, show=True):
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


    def add_layer(self, name, gdf, datacolumn, tooltip_fields, data_min, data_max, colormap_name, colormap_type="categorical", show=False, show_colormap=True):

        colormap = self.get_colormap(label=f"Legend {name}",
                                      data_min=data_min,
                                      data_max=data_max,
                                      colormap_name=colormap_name,
                                      colormap_type=colormap_type,
                                      colormap_field=datacolumn,
                                      gdf=gdf)

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
            
    def add_graphs(self, name, gdf, image_field, width=520, height=520):

        marker_cluster = MarkerCluster(
                name=name,
                overlay=True,
                control=True,
                icon_create_function=None
            )


        for idx, data in gdf.iterrows():
            centroid = data.geometry.centroid
            point_gdf = gp.GeoDataFrame(geometry=[centroid], crs=gdf.crs)

            # Reproject to WGS84
            point_wgs84 = point_gdf.to_crs(epsg=4326)

            # Get lat/lon coordinates
            lon, lat = point_wgs84.geometry.x.iloc[0], point_wgs84.geometry.y.iloc[0]

            encoded = base64.b64encode(open(data[image_field], 'rb').read()).decode()
            html = f'<img src="data:image/png;base64,{encoded}" width="{width}" height="{height}">'
            iframe = IFrame(html, width=width, height=height)
            popup = folium.Popup(iframe, max_width=width)

            icon = folium.Icon(color="red", icon="ok")
            marker = folium.Marker(location=[lat, lon], popup=popup, icon=icon)
            marker.add_to(marker_cluster)

        marker_cluster.add_to(self.m)

    def add_legend(self):
        self.m.add_child(self.colormap)

    def add_title(self, title):
        # Add title to map
        title_html = f'<h1 style="position:absolute;z-index:100000;bottom:1vw;background-color:rgba(255, 255, 255, 0.8);padding:10px;border-radius:5px;" >{title}</h1>'
        self.m.get_root().html.add_child(folium.Element(title_html))

    def save(self, output_path):
        # Create layer control first
        layer_control = folium.LayerControl(collapsed=False, sort_layers=True, position='topright')
        
        # Add all layers to the control but only show them if show=True
        for l in self.layers:
            if hasattr(l, 'show') and l.show:
                self.m.add_child(l)
            else:
                # Add to map but hide it
                l.add_to(self.m)
                self.m.get_root().html.add_child(folium.Element(f'''
                    <script>
                        document.addEventListener("DOMContentLoaded", function() {{
                            map.removeLayer({l.get_name()});
                        }});
                    </script>
                '''))
        
        # Add the layer control after all layers
        self.m.add_child(layer_control)

        # Add colormaps and bindings
        for c in self.color_maps:
            self.m.add_child(c)

        for b in self.color_map_binds:
            self.m.add_child(b)
    
        logger.debug(f"Saving interactive map to: {output_path}")
        self.m.save(output_path)


class BindColormap(MacroElement):  # from https://github.com/python-visualization/folium/issues/450
    """Binds a colormap to a given layer.

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    """

    def __init__(self, layer, colormap):
        super(BindColormap, self).__init__()
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


if __name__ == "__main__":
    figures = gp.read_file(
        r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\output\heiloo_optimized\post_processing/figures.gpkg",
        layer="bergingscurve",
    )
    fol = WSSCurvesFolium()
    fol.add_colormap(0, 5)
    fol.add_water_layer()
    fol.add_layer("Schades", figures, datacolumn="drainage_level", tooltip_fields=["pid"])
    fol.add_graphs("Schadegrafieken", figures, "png_path")
    fol.add_title("Schadecurves")
    fol.add_legend("Legenda")
    fol.save(
        r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\output\heiloo_optimized\post_processing/test.html"
    )
