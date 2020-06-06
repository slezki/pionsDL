#!/usr/bin/env python
# coding: utf-8

import sys
import uproot
import os
import numpy as np
import pandas as pd


root_file = '/lustre/cms/store/user/slezki/Filtered/histo_y2s2y1spp_2018_MC_v1.root'
dataset_dir = '../dataset/'

os.makedirs(dataset__dir, exist_ok=True)

t_tree = 'tTrue'
f_tree = 'tFake'

pP_variables = ['pionP_charge','pionP_dxy', 'pionP_dz', 'pionP_dRwithDimuon'] # base features for pi+
pM_variables = ['pionM_charge','pionM_dxy', 'pionM_dz', 'pionM_dRwithDimuon'] # base features for pi-


df_t = uproot.open(root_file)[t_tree] # read tTrue tree
df_f = uproot.open(root_file)[f_tree] # read tFake tree


new_cols = ['charge', 'dxy', 'dz', 'dRwithDimuon'] # easy rename, since datasets will be separated

dfP_t = df_t.pandas.df(pP_variables,flatten=False) # base dataframe for True pi+
dfP_t.columns = new_cols

dfM_t = df_t.pandas.df(pM_variables,flatten=False) # base dataframe for True pi-
dfM_t.columns = new_cols

dfP_f = df_f.pandas.df(pP_variables,flatten=False) # base dataframe for Fake pi+
dfP_f.columns = new_cols

dfM_f = df_f.pandas.df(pM_variables,flatten=False) # base dataframe for Fake pi-
dfM_f.columns = new_cols


# Now read additional features from TLorentzVector for True pi+ and pi-, and for Fake pi+ and pi-
lorentz_var = ['E', 'pt', 'eta', 'phi', 'px', 'py', 'pz']

lP_t = np.array([df_t.array('pionP_p4').E, df_t.array('pionP_p4').pt, df_t.array('pionP_p4').eta, df_t.array('pionP_p4').phi, df_t.array('pionP_p4').x, df_t.array('pionP_p4').y, df_t.array('pionP_p4').z]).T
lM_t = np.array([df_t.array('pionM_p4').E, df_t.array('pionM_p4').pt, df_t.array('pionM_p4').eta, df_t.array('pionM_p4').phi, df_t.array('pionM_p4').x, df_t.array('pionM_p4').y, df_t.array('pionM_p4').z]).T

lP_f = np.array([df_f.array('pionP_p4').E, df_f.array('pionP_p4').pt, df_f.array('pionP_p4').eta, df_f.array('pionP_p4').phi, df_f.array('pionP_p4').x, df_f.array('pionP_p4').y, df_f.array('pionP_p4').z]).T
lM_f = np.array([df_f.array('pionM_p4').E, df_f.array('pionM_p4').pt, df_f.array('pionM_p4').eta, df_f.array('pionM_p4').phi, df_f.array('pionM_p4').x, df_f.array('pionM_p4').y, df_f.array('pionM_p4').z]).T


# Add the new features to the previous dataframes

dfP_t = pd.concat([pd.DataFrame(lP_t, columns=lorentz_var), dfP_t], axis=1)
dfM_t = pd.concat([pd.DataFrame(lM_t, columns=lorentz_var), dfM_t], axis=1)

dfP_f = pd.concat([pd.DataFrame(lP_f, columns=lorentz_var), dfP_f], axis=1)
dfM_f = pd.concat([pd.DataFrame(lM_f, columns=lorentz_var), dfM_f], axis=1)


# Check that the datasets are correct (use only one print)
print(dfP_t.head())
print('Len: {}'.format(len(dfP_t)))

print(dfM_t.head())
print('Len: {}'.format(len(dfM_t)))

print(dfP_f.head())
print('Len: {}'.format(len(dfP_f)))

print(dfM_f.head())
print('Len: {}'.format(len(dfM_f)))


# Now save datasets in a single hdf5 file

dfP_t.to_hdf(dataset_dir + 'pions.h5', "true_pionP")
dfM_t.to_hdf(dataset_dir + 'pions.h5', "true_pionM")
dfP_f.to_hdf(dataset_dir + 'pions.h5', "fake_pionP")
dfM_f.to_hdf(dataset_dir + 'pions.h5', "fake_pionM")


print('Datasets correctly saved!')



