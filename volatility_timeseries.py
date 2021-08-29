#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 22:33:27 2021

@author: vittoriavolta
"""

import os, sys, cmath, math, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize, linalg, special 
from scipy.interpolate import splrep, splev
from copy import deepcopy
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from numpy import ma, atleast_2d, pi, sqrt, sum, transpose
from scipy.special import gammaln, logsumexp
from scipy.stats.mstats import mquantiles


from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm   
from pyhht.utils import extr, boundary_conditions
warnings.filterwarnings("ignore")

directory = '/Users/vittoriavolta/Desktop/article'

announc = pd.read_excel(directory + "/" + "announcements.xlsx")

for c in announc.columns:
    announc[c] = pd.to_datetime(announc[c]).dt.date 
    
ecb = announc["ECB"].dropna()
boe = announc["BoE"].dropna()

data_ecb = pd.DataFrame()
data_boe = pd.DataFrame()
for filename in os.listdir(directory):
    if filename.startswith("EST") and filename.endswith(".txt"):
        dataset = pd.read_csv(directory + "/" + filename, header = None)
        dataset.columns = ["datetime", "open", "high", "low", "close"]
        dataset[['date','time']] = dataset.datetime.str.split(" ", expand=True) 
        dataset.drop("datetime", axis = 1, inplace = True)
        dataset["date"] = pd.to_datetime(dataset["date"]).dt.date 
        dataset["time"] = pd.to_datetime(dataset["time"]).dt.time 
        data_ecb = pd.concat([data_ecb, dataset])
    elif filename.startswith("UKX") and filename.endswith(".txt"):
        dataset = pd.read_csv(directory + "/" + filename, header = None)
        dataset.columns = ["datetime", "open", "high", "low", "close"]
        dataset[['date','time']] = dataset.datetime.str.split(" ", expand=True) 
        dataset.drop("datetime", axis = 1, inplace = True)
        dataset["date"] = pd.to_datetime(dataset["date"]).dt.date 
        dataset["time"] = pd.to_datetime(dataset["time"]).dt.time 
        data_boe = pd.concat([data_boe, dataset])
        
# ECB

startTime_ecb = datetime.time(9,1)
endTime_ecb = datetime.time(17,29)
strDate_ecb = datetime.date(2015,12,10)
endDate_ecb = data_ecb["date"].max()


businessDates_ecb = pd.date_range(strDate_ecb, endDate_ecb, freq = "B").date
businessHours_ecb = pd.date_range(str(startTime_ecb), str(endTime_ecb), freq = "1min").time

format_data_ecb = data_ecb.pivot_table(index = "time", columns = "date", values = "close")
format_data_ecb = format_data_ecb.loc[:, format_data_ecb.columns.isin(businessDates_ecb)]
format_data_ecb = format_data_ecb.loc[format_data_ecb.index.isin(businessHours_ecb),:]
format_data_ecb = format_data_ecb.loc[:, format_data_ecb.columns[format_data_ecb.isnull().mean() < 0.3]]

sqr_rets_ecb = (np.log(format_data_ecb) - np.log(format_data_ecb.shift(1)))**2
ann = (format_data_ecb.shape[0]-1)*252
vol_ecb = np.sqrt(sqr_rets_ecb.rolling(window = 5).mean())*np.sqrt(ann)
vol_est_ecb = vol_ecb.loc[:, vol_ecb.columns.isin(ecb)]
vol_est_boe = vol_ecb.loc[:, vol_ecb.columns.isin(boe)]

# BoE

startTime_boe = datetime.time(3,0)
endTime_boe = datetime.time(11,29)
strDate_boe = datetime.date(2015,1,1)
endDate_boe = data_boe["date"].max()


businessDates_boe = pd.date_range(strDate_boe, endDate_boe, freq = "B").date
businessHours_boe = pd.date_range(str(startTime_boe), str(endTime_boe), freq = "1min").time

format_data_boe = data_boe.pivot_table(index = "time", columns = "date", values = "close")
format_data_boe = format_data_boe.loc[:, format_data_boe.columns.isin(businessDates_boe)]
format_data_boe = format_data_boe.loc[format_data_boe.index.isin(businessHours_boe),:]
format_data_boe = format_data_boe.loc[:, format_data_boe.columns[format_data_boe.isnull().mean() < 0.3]]

sqr_rets_boe = (np.log(format_data_boe) - np.log(format_data_boe.shift(1)))**2
ann = (format_data_boe.shape[0]-1)*252
vol_boe = np.sqrt(sqr_rets_boe.rolling(window = 5).mean())*np.sqrt(ann)
vol_ukx_boe = vol_boe.loc[:, vol_boe.columns.isin(boe)]
vol_ukx_ecb = vol_boe.loc[:, vol_boe.columns.isin(ecb)]

