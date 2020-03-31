# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:21:14 2019

@author: martinhe
"""


import numpy as np
import pandas as pd
import datetime

from sklearn.cluster import DBSCAN

from shapely.geometry import Point
import shapely.wkt as wkt
import geopandas as gpd
import pyproj
from pyproj import Transformer




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

    data = pd.read_csv(input_file,  nrows=nrows, sep=";", 
                           names=["id", "Taxi-id", "time", "geometry"])
    
    data["time"] = pd.to_datetime(data["time"])
    data["time"] = data["time"].dt.tz_convert(None) #to utc, remove tz
    #data["time"] = data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    geometry = data["geometry"].apply(wkt.loads)
    data.drop(["geometry", "id"], axis=1, inplace=True)
    
    data["lat"] = [geom.x for geom in geometry]
    data["lon"] = [geom.y for geom in geometry]
    
    data = data[["lon","lat","time"]]
    
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
    
    t_reference = datetime.datetime(2000,1,1)
    
    data_in = np.asarray(data_in)
    
    x = data_in[:,0]
    y = data_in[:,1]
    ts = data_in[:,2]
    x, y = transformer.transform(x, y)

    ts = np.asarray([((ts_this-t_reference).total_seconds())/timescale for ts_this in ts])
    data_out = np.concatenate((x.reshape((-1,1)),y.reshape((-1,1)),ts.reshape((-1,1))), axis=1)
    
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
    idx_max = df[label_col_name].eq(max_label+1).idxmax()

    # make sure that you are not generatuing invalid boundaries
    idx_min = max(idx_min-slack, 0)
    idx_max = min(idx_max + slack, df.shape[0])

    df_filtered = df.iloc[idx_min:idx_max,:].copy()
    
    return df_filtered