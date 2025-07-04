# %%
        
from branca.element import MacroElement

from jinja2 import Template
import base64
import geopandas as gp
from folium import IFrame
import branca.colormap as cm
import folium
import geopandas as gp
from folium.plugins import MarkerCluster
import hhnk_research_tools.logger as logging

logger = logging.get_logger(__name__, level="DEBUG")


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

    def get_colormap(self, label, data_min, data_max, colormap_name= "plasma", nr_steps=5):
        # Use getattr to dynamically access the colormap
        colormap = getattr(cm.linear, colormap_name)
        colormap = colormap.scale(data_min, data_max)
     #   steps = [i for i in range(int(data_min), int(data_max), int((data_max-data_min)/nr_steps))]
        colormap = colormap.to_step(nr_steps)
        colormap.caption = label
        return colormap

    def add_water_layer(self):
        folium.TileLayer("nlmaps.water", attr="<a href=https://nlmaps.nl/>NL Maps water</a>"
                         ).add_to(self.m)
        
    def add_border_layer(self, name, gdf, tooltip_fields, show=True):
        border_style = {'color': '#000000', 'weight': '1.5', 'fillColor': '#58b5d1', 'fillOpacity': 0.08}

        layer = folium.GeoJson(
            gdf,
            style_function=lambda x:border_style,
            name=name,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_fields,
            ),
            control=True,
            show=show,
            overlay=True,
        )
        
        self.layers.append(layer)
        
    
    def add_layer(self, name, gdf, datacolumn, tooltip_fields, data_min, data_max, colormap_name, show):

        colormap = self.get_colormap(f"Legend {name}", data_min, data_max, colormap_name )

        layer = folium.GeoJson(
            gdf,
            style_function=lambda feature: {
                "fillColor": colormap(feature["properties"][datacolumn]),
                "color": "white",
                "fillOpacity": 0.8,
                "weight": 1,
            },
            name=name,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_fields,
            show=False,
            control=True,
            overlay=True,

            ),
        )
        self.layers.append(layer)
        self.color_maps.append(colormap)
        self.color_map_binds.append(BindColormap(layer, colormap))
        
    def add_graphs(self, name, gdf, image_field):

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
            html = '<img src="data:image/png;base64,{}" width="520" height="520">'.format
            iframe = IFrame(html(encoded), width=500+20, height=500+20)
            popup = folium.Popup(iframe, max_width=500+20)

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
        
        for l in self.layers:
            self.m.add_child(l)
        
        self.m.add_child(folium.LayerControl(collapsed=False))

        for c in self.color_maps:
            self.m.add_child(c)
        
        for b in self.color_map_binds:
            self.m.add_child(b)
        
        
        logger.debug(f"Saving interactive map to: {output_path}")
        self.m.save(output_path)

class BindColormap(MacroElement): # from https://github.com/python-visualization/folium/issues/450
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
        self._template = Template(u"""
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
    figures = gp.read_file(r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\output\heiloo_optimized\post_processing/figures.gpkg", layer="bergingscurve")
    fol = WSSCurvesFolium()
    fol.add_colormap(0, 5)
    fol.add_water_layer()
    fol.add_layer("Schades", figures, datacolumn="drainage_level", tooltip_fields=['pid'])
    fol.add_graphs("Schadegrafieken",figures, 'png_path')
    fol.add_title("Schadecurves")
    fol.add_legend("Legenda")
    fol.save(r"C:\Users\kerklaac5395\ARCADIS\30225745 - Schadeberekening HHNK - Documents\External\output\heiloo_optimized\post_processing/test.html")

