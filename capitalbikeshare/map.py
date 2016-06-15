import pandas as pd
from datetime import timedelta
import xml.etree.ElementTree as et
from bokeh.models.glyphs import Circle
from bokeh.models import (
    GMapPlot, ColumnDataSource,
    HoverTool, PanTool, WheelZoomTool, PreviewSaveTool,
    GMapOptions, DataRange1d, MultiLine, ResetTool, UndoTool, RedoTool,
    BoxZoomTool)
from bokeh.io import output_file, show
from collections import Counter


def xml_to_pandas(xml_file):
    """Convert the station status xml file to a pandas dataframe"""
    tree = et.parse(xml_file)
    root = tree.getroot()

    l = []
    for station in root:
        d = {}
        for attrib in station:
            d[str(attrib.tag)] = str(attrib.text)
        l.append(d)

    df = pd.DataFrame.from_dict(l)
    return df

# read the data
data = pd.read_csv('data/2016-Q1-Trips-History-Data.csv',
                   parse_dates=['Start date', 'End date'],
                   infer_datetime_format=True)
data['Duration'] = data['Duration (ms)'].apply(lambda x:
                                               timedelta(milliseconds=int(x)))

data['End station number'] = data['End station number'].astype(int)
data['Start station number'] = data['Start station number'].astype(int)

data = data.dropna()
# after dropping some columns we can convert from float to int
bike_stations = xml_to_pandas('data/bike_stations.xml')
bike_stations['terminalName'] = bike_stations['terminalName'].astype(int)
bike_stations['lat'] = bike_stations['lat'].astype(float)
bike_stations['long'] = bike_stations['long'].astype(float)
station_locations = bike_stations[['terminalName']]
station_locations['location'] = list(zip(bike_stations['lat'],
                                         bike_stations['long']))

# merge location and usage info
data = data.merge(station_locations, how='left',
                  left_on='Start station number', right_on='terminalName')
data.columns = [w if w != 'location' else
                'start location' for w in data.columns]
data.drop('terminalName', axis=1, inplace=True)

data = data.merge(station_locations,
                  left_on='End station number', right_on='terminalName')
data.columns = [w if w != 'location' else
                'end location' for w in data.columns]
data.drop('terminalName', axis=1, inplace=True)
data = data[data['end location'].map(lambda x: isinstance(x, tuple))]
data = data[data['start location'].map(lambda x: isinstance(x, tuple))]

pairs = []

for index, row in data.iterrows():
    pair = tuple(sorted([row['start location'], row['end location']]))
    pairs.append(pair)

pair_dict = Counter(pairs)
most_common, ncm = pair_dict.most_common(1)[0]

# Get all pairwise lats, longs, scales, numbers in bokeh plotting format
lats = []
lons = []
size_scale = []
num = []
for k, v in pair_dict.items():
    lats.append([k[0][0], k[1][0]])
    lons.append([k[0][1], k[1][1]])
    transformed = (v-1)/(ncm-1)
    size_scale.append(transformed)
    num.append(v)

# Create data sources and hover tools
source = ColumnDataSource(
    data=dict(
        lat=bike_stations['lat'].values,
        lon=bike_stations['long'].values,
        name=bike_stations['name'].values,
    )
)

line_source = ColumnDataSource(
    data=dict(
        lats=lats,
        lons=lons,
        line_alpha=[max(0.1, scale) for scale in size_scale],
        line_width=[scale*20 for scale in size_scale],
        num=num,
    )
)

hover = HoverTool(
    tooltips=[
        ("Station Name: ", "@name"),
        ("Trips: ", "@num"),
    ]
)

map_options = GMapOptions(lat=38.889490, lng=-77.035180,
                          map_type="terrain", zoom=13)

plot = GMapPlot(
    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options,
    title="Washington, DC",
    plot_width=1280, plot_height=1280, responsive=True
)

# Finally plot it
lines = MultiLine(xs="lons", ys="lats", line_alpha="line_alpha",
                  line_width="line_width", line_color="red", line_cap="round")
circle = Circle(x="lon", y="lat", size=10, fill_color="blue",
                fill_alpha=0.8, line_color=None)
plot.add_glyph(source, circle)
plot.add_glyph(line_source, lines)

plot.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(), hover,
               ResetTool(), UndoTool(), RedoTool(), PreviewSaveTool())
output_file("gmap_plot.html")
show(plot)
