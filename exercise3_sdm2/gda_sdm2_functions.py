# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:21:14 2019

@author: martinhe
"""

import datetime
import random

import geopandas as gpd
import ipyleaflet as ipy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import shapely.wkt as wkt
from IPython.display import HTML
from ipywidgets import HTML as ipy_HTML
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.geometry import Point
from sklearn.cluster import DBSCAN


def read_romataxidata(input_file, nrows=None):
    """Function to read the The roma/taxi dataset.
    (https://crawdad.org/roma/taxi/20140717/)
    
    nrows [int]: Number of rows that are read from the .txt file
    
    returns an numpy array with [id, taxi-id, date, lat, lon]:
    id [int]: Unique id of observation
    taxi-id [int]: Unique id of taxi
    date [str]: Timestamp in datetime format
    lat [float]: Coordinates in wgs84
    lon [float]: Coordinates in wgs84
    
    Args:
        input_file (TYPE): Description
        nrows (None, optional): Description
    
    Returns:
        TYPE: Description
    """

    data = pd.read_csv(input_file, nrows=nrows, sep=";",
                       names=["id", "Taxi-id", "time", "geometry"])

    data["time"] = pd.to_datetime(data["time"])
    data["time"] = data["time"].dt.tz_convert(None)  # to utc, remove tz
    # data["time"] = data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    geometry = data["geometry"].apply(wkt.loads)
    data.drop(["geometry", "id"], axis=1, inplace=True)

    data["lat"] = [geom.x for geom in geometry]
    data["lon"] = [geom.y for geom in geometry]

    data = data[["lon", "lat", "time"]]

    return data.values.tolist()


def transform(data_in, timescale=60, input_crs="EPSG:4326", output_crs="EPSG:25833"):
    """Transform timestamped data (x,y,t) into a 2D coordinate system with relative timestamps.

    This is the vectorized (=faster) version of the function from exercise 2.
    
    
    Args:
        data_in (TYPE): Description
        timescale (int, optional): Scaling factor that the timestamp is devided by. Timestamps are in seconds
            therefore a scaling factor of 60 transforms them into minutes.
        input_crs (str, optional): Coordinate system of the data,
        output_crs (str, optional): Output coordinate system (should be projected)
    
    Returns:
        TYPE: numpy array with (x, y, t)
    """

    transformer = Transformer.from_crs(crs_from=input_crs, crs_to=output_crs, always_xy=True)
    data_out = []

    t_reference = datetime.datetime(2000, 1, 1)

    data_in = np.asarray(data_in)

    x = data_in[:, 0]
    y = data_in[:, 1]
    ts = data_in[:, 2]
    x, y = transformer.transform(x, y)

    ts = np.asarray([((ts_this - t_reference).total_seconds()) / timescale for ts_this in ts])
    data_out = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1)), ts.reshape((-1, 1))), axis=1)

    return data_out


def apply_dbscan(X, eps=15, min_samples=5, metric='chebyshev'):
    """ Function derived from scipy dbscan example
    http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py
    """

    X = np.array(X)

    ##########################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X)

    labels = db.labels_
    core_samples_indices = db.core_sample_indices_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    return labels, core_samples_indices


def filter_by_label(df, min_label=-1, max_label=1, label_col_name='label', slack=0):
    """Filter a (quasi-sorted) pandas dataframe by a column and min/max values.
    This function returns all values in between the first min_label and the first appearance
    of (max_label+1), to provide some slack. 
    
    Args:
        df (TYPE): Dataframe with data. Index has to be an enumeration of the dataframe
        min_label (int, optional): Lowest label value to include
        max_label (int, optional): Largest label value to include
        label_col_name (str, optional): Name of the column for filtering. Has to exist in df
        slack (int, optional): Number of datapoints that will be returned around the limits. 
    
    Returns:
        TYPE: Description
    """

    idx_min = df[label_col_name].eq(min_label).idxmax()
    idx_max = df[label_col_name].eq(max_label + 1).idxmax()

    if idx_max == 0:
        idx_max = df.shape[0]

    # make sure that you are not generatuing invalid boundaries
    idx_min = max(idx_min - slack, 0)
    idx_max = min(idx_max + slack, df.shape[0])

    df_filtered = df.iloc[idx_min:idx_max, :].copy()

    return df_filtered


def get_tripleg_geometry_from_points(list_of_points):
    # if tripleg invalid, leave the loop. Remember, a linestring has at least two points
    if len(list_of_points) < 2:
        return None
    coords = [(point.x, point.y) for point in list_of_points]

    return LineString(coords)


def get_tripleg(cluster_start, cluster_end, start_time, end_time, geometry):
    tripleg_dict = {'cluster_start': cluster_start,
                    'cluster_end': cluster_end,
                    't_start': start_time,
                    't_end': end_time,
                    'geometry': geometry}

    return tripleg_dict


def get_ipyleaflet_trackpoint_layer(gdf, min_label=0, max_label=1, slack=0, filter_gdf=True):
    circlelist_noise = list()
    circlelist_sp = list()

    if filter_gdf:
        gdf = filter_by_label(gdf, min_label=min_label, max_label=max_label, slack=slack)
    else:
        pass

    # create colormap
    unique_labels = gdf['label'].unique().tolist()
    nb_labels = len(unique_labels)
    try:
        unique_labels.remove(-1)
    except ValueError:
        pass

    true_min_label = min(unique_labels)

    colors = plt.cm.Spectral(np.linspace(0, 1, nb_labels))
    colors = [matplotlib.colors.to_hex(color_this) for color_this in colors]

    for ix, row in gdf.iterrows():

        message = ipy_HTML()
        t_string = row['timestamp'].strftime("%Y-%m-%d %H-%M-%S")

        message.value = """<table>
                            <tr> <td>label:</td>      <td>&emsp;</td> <td>{}</td> </tr>
                            <tr> <td>timestamp:</td>  <td>&emsp;</td> <td>{}</td> </tr>
                            </table>""".format(row.label, t_string)

        if int(row.label) == -1:
            fillcolor_this = "Gray"
            opacity_this = 0.6
            stroke = True
            radius_this = 5

            circle = ipy.CircleMarker()
            circle.location = (row.geometry.y, row.geometry.x)
            circle.radius = radius_this
            circle.fill_opacity = opacity_this
            circle.fill_color = fillcolor_this
            circle.stroke = stroke
            circle.color = 'Black'
            circle.weight = 1
            circle.opacity = 0.3
            circle.popup = message
            circlelist_noise.append(circle)

        else:
            fillcolor_this = colors[int(row.label) - true_min_label]
            opacity_this = 0.5
            stroke = True
            radius_this = 7
            circle = ipy.CircleMarker()
            circle.location = (row.geometry.y, row.geometry.x)
            circle.radius = radius_this
            circle.fill_opacity = opacity_this
            circle.fill_color = fillcolor_this
            circle.stroke = stroke
            circle.color = 'Black'
            circle.weight = 1
            circle.opacity = 0.3
            circle.popup = message

            circlelist_sp.append(circle)

    layer_group_noise = ipy.LayerGroup(layers=circlelist_noise)
    layer_group_noise.name = 'trackpoints noise'

    layer_group_sp = ipy.LayerGroup(layers=circlelist_sp)
    layer_group_sp.name = 'trackpoints staypoints'

    return layer_group_noise, layer_group_sp


def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1, 2 ** 64)))

    html_str = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current,
        toggle_text=toggle_text
    )

    return HTML(html_str)
