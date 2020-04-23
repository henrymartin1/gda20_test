# -*- coding: utf-8 -*-
"""
This script reads in the geolife data (as it can be downloaded from
https://www.microsoft.com/en-us/download/details.aspx?id=52367) and loads it
in a postgis database
"""

import os
import time
import json
import ntpath
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
import psycopg2
import trackintel as ti
from shapely.geometry import Point
import matplotlib.pyplot as plt

FEET2METER = 0.3048

CRS_WGS84 = {'init' :'epsg:4326'}

data_folder = os.path.join(".", "data_geolife", "*")
user_folder = glob.glob(data_folder)


# In the geolife data, every user has a folder with a file with tracking data
# for every day. We iterate every folder concatenate all files of 1 user into
# a single pandas dataframe and send it to the postgres database.

for user_folder_this in user_folder:
    t_start = time.time()

    # extract user id from path
    _, tail = ntpath.split(user_folder_this)
    user_id = int(tail)
    print("start user_id: ", user_id)


    input_files = glob.glob(os.path.join(
                user_folder_this, "Trajectory", "*.plt"))
    df_list = []

    # read every day of every user
    for input_file_this in input_files:
        
        data_this = pd.read_csv(input_file_this, skiprows=6, header=None,
                                names=['lat', 'lon', 'zeros', 'elevation', 
                                       'date days', 'date', 'time'])

        data_this['tracked_at'] = pd.to_datetime(data_this['date']
                                                 + ' ' + data_this['time'])

        data_this.drop(['zeros', 'date days', 'date', 'time'], axis=1,
                       inplace=True)
        data_this['user_id'] = user_id
        data_this['elevation'] = data_this['elevation'] * FEET2METER
        
        data_this['geom'] = list(zip(data_this.lon, data_this.lat))
        data_this['geom'] = data_this['geom'].apply(Point)

        df_list.append(data_this)

    pfs = pd.concat(df_list, axis=0, ignore_index=True)
    pfs = gpd.GeoDataFrame(pfs, geometry="geom", crs=CRS_WGS84)
    pfs["accuracy"] = None

    t_end = time.time()
    print("finished user_id: ", user_id, "Duration: ", "{:.0f}"
          .format(t_end-t_start))
    
#pfs.index = np.random.randint(low=0, high=10e7, size=len(pfs))
#pfs = pfs[pfs["lon"] > 116.327]
#pfs = pfs[pfs["lat"] > 39.9675]

spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=100, time_threshold=5*60)
plcs = spts.as_staypoints.extract_places()
tpls = pfs.as_positionfixes.extract_triplegs(spts)
    
#fig, ax = plt.subplots()
#
#tpls.plot(ax=ax)
#spts.plot(ax=ax)
#pfs.plot(marker="o",markersize=1, color="r", ax=ax)

pfs['tracked_at'] = pfs['tracked_at'].astype(np.int64) // 10**9
spts['started_at'] = spts['started_at'].astype(np.int64) // 10**9
spts['finished_at'] = spts['finished_at'].astype(np.int64) // 10**9
tpls['started_at'] = tpls['started_at'].astype(np.int64) // 10**9
tpls['finished_at'] = tpls['finished_at'].astype(np.int64) // 10**9


pfs.to_file("shp\\geolife_positionfixes.shp")
spts.to_file("shp\\geolife_staypoints.shp")
#plcs.to_file("shp\\geolife_places.shp")
tpls.to_file("shp\\geolife_triplegs.shp")

spts2 = gpd.read_file("shp\\geolife_staypoints.shp")


#spoint = spts.loc[3,'geom']
