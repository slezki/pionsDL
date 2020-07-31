#!/usr/bin/env python
# coding: utf-8

import sys
import uproot
import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type',  type=str, default='RS')
args = parser.parse_args()

filetype = args.type


def signal_background_datasetMaker(p1, p2):

    p1_variables = [p1 + var for var in pion_variables]
    p2_variables = [p2 + var for var in pion_variables]

    df_t = uproot.open(root_file)[t_tree] # read tTrue tree
    df_f = uproot.open(root_file)[f_tree] # read tFake tree

    df1_t = df_t.pandas.df(p1_variables,flatten=False) # base dataframe for True pi1
    df1_t.columns = new_cols

    df2_t = df_t.pandas.df(p2_variables,flatten=False) # base dataframe for True pi2
    df2_t.columns = new_cols

    df1_f = df_f.pandas.df(p1_variables,flatten=False) # base dataframe for Fake pi1
    df1_f.columns = new_cols

    df2_f = df_f.pandas.df(p2_variables,flatten=False) # base dataframe for Fake pi2
    df2_f.columns = new_cols

    # Now read additional features from TLorentzVector for True pi1 and pi2, and for Fake pi1 and pi2
    p1p4 = p1+'_p4'
    p2p4 = p2+'_p4'

    l1_t = np.array([df_t.array(p1p4).E, df_t.array(p1p4).pt, df_t.array(p1p4).eta, df_t.array(p1p4).phi, df_t.array(p1p4).x, df_t.array(p1p4).y, df_t.array(p1p4).z]).T
    l2_t = np.array([df_t.array(p2p4).E, df_t.array(p2p4).pt, df_t.array(p2p4).eta, df_t.array(p2p4).phi, df_t.array(p2p4).x, df_t.array(p2p4).y, df_t.array(p2p4).z]).T

    l1_f = np.array([df_f.array(p1p4).E, df_f.array(p1p4).pt, df_f.array(p1p4).eta, df_f.array(p1p4).phi, df_f.array(p1p4).x, df_f.array(p1p4).y, df_f.array(p1p4).z]).T
    l2_f = np.array([df_f.array(p2p4).E, df_f.array(p2p4).pt, df_f.array(p2p4).eta, df_f.array(p2p4).phi, df_f.array(p2p4).x, df_f.array(p2p4).y, df_f.array(p2p4).z]).T

    # Add the new features to the previous dataframes
    df1_t = pd.concat([pd.DataFrame(l1_t, columns=lorentz_var), df1_t], axis=1)
    df2_t = pd.concat([pd.DataFrame(l2_t, columns=lorentz_var), df2_t], axis=1)

    df1_f = pd.concat([pd.DataFrame(l1_f, columns=lorentz_var), df1_f], axis=1)
    df2_f = pd.concat([pd.DataFrame(l2_f, columns=lorentz_var), df2_f], axis=1)

    # Read Y1Spp mass
    dfy1s_t = df_t.pandas.df('Y1Spipi_M',flatten=False)
    dfy1s_f = df_f.pandas.df('Y1Spipi_M',flatten=False)

    # Read event number
    dfevt_t = df_t.pandas.df('event',flatten=False)
    dfevt_f = df_f.pandas.df('event',flatten=False)

    # Create ground truths
    gt_t = np.ones(len(df1_t))
    gt_f = np.zeros(len(df1_f))

    # Concatenate the event information, the ground truth and Y1S dataset
    df3_t = pd.concat([dfevt_t, dfy1s_t, pd.DataFrame(gt_t, columns=['label'])], axis=1)
    df3_f = pd.concat([dfevt_f, dfy1s_f, pd.DataFrame(gt_f, columns=['label'])], axis=1)

    # Concatenate the datasets for pi1, pi2 and additional info separately
    df1_all = pd.concat([df1_t, df1_f], axis=0)
    df2_all = pd.concat([df2_t, df2_f], axis=0)
    df3_all = pd.concat([df3_t, df3_f], axis=0)

    # Check that the datasets are correct (use only one print)
    print(df1_all.head())
    print('Len: {}'.format(len(df1_all)))

    print(df2_all.head())
    print('Len: {}'.format(len(df2_all)))

    print(df3_all.head())
    print('Len: {}'.format(len(df3_all)))

    # Now save datasets in a single hdf5 file
    df1_all.to_hdf(dataset_dir + filetype + '.h5', "pion1")
    df2_all.to_hdf(dataset_dir + filetype + '.h5', "pion2")
    df3_all.to_hdf(dataset_dir + filetype + '.h5', "add_info")

    print('Datasets for signal + background correctly saved!')


def additional_background_datasetMaker(p1, p2):

    p1_variables = [p1 + var for var in pion_variables]
    p2_variables = [p2 + var for var in pion_variables]

    df_all = uproot.open(root_file)[a_tree] # read tAll tree

    df1_all = df_all.pandas.df(p1_variables,flatten=False) # base dataframe for pi1
    df1_all.columns = new_cols

    df2_all = df_all.pandas.df(p2_variables,flatten=False) # base dataframe for pi2
    df2_all.columns = new_cols

    # Now read additional features from TLorentzVector for pi1 and pi2
    p1p4 = p1+'_p4'
    p2p4 = p2+'_p4'
    l1_all = np.array([df_all.array(p1p4).E, df_all.array(p1p4).pt, df_all.array(p1p4).eta, df_all.array(p1p4).phi, df_all.array(p1p4).x, df_all.array(p1p4).y, df_all.array(p1p4).z]).T
    l2_all = np.array([df_all.array(p2p4).E, df_all.array(p2p4).pt, df_all.array(p2p4).eta, df_all.array(p2p4).phi, df_all.array(p2p4).x, df_all.array(p2p4).y, df_all.array(p2p4).z]).T

    # Add the new features to the previous dataframes
    df1_all = pd.concat([pd.DataFrame(l1_all, columns=lorentz_var), df1_all], axis=1)
    df2_all = pd.concat([pd.DataFrame(l2_all, columns=lorentz_var), df2_all], axis=1)

    # Read Y1Spp mass
    dfy1s_all = df_all.pandas.df('Y1Spipi_M',flatten=False)

    # Read event number
    dfevt_all = df_all.pandas.df('event',flatten=False)

    # Create ground truths
    gt_all = np.zeros(len(df1_all))

    # Concatenate the event information, the ground truth and Y1S dataset
    df3_all = pd.concat([dfevt_all, dfy1s_all, pd.DataFrame(gt_all, columns=['label'])], axis=1)

    # Check that the datasets are correct (use only one print)
    print(df1_all.head())
    print('Len: {}'.format(len(df1_all)))

    print(df2_all.head())
    print('Len: {}'.format(len(df2_all)))

    print(df3_all.head())
    print('Len: {}'.format(len(df3_all)))

    # Now save datasets in a single hdf5 file
    df1_all.to_hdf(dataset_dir + filetype + '.h5', "pion1")
    df2_all.to_hdf(dataset_dir + filetype + '.h5', "pion2")
    df3_all.to_hdf(dataset_dir + filetype + '.h5', "add_info")

    print('Datasets for background-only correctly saved!')


### Here starts the code
dataset_dir = '../dataset/'
os.makedirs(dataset_dir, exist_ok=True)

t_tree = 'tTrue'
f_tree = 'tFake'
a_tree = 'tAll'

pion_variables = ['_charge','_dxy', '_dz', '_dRwithDimuon', '_fromPV'] # base features for pions
new_cols = ['charge', 'dxy', 'dz', 'dRwithDimuon', 'fromPV'] # easy rename, since datasets will be separated
lorentz_var = ['E', 'pt', 'eta', 'phi', 'px', 'py', 'pz']

if(filetype == 'RS'):
    root_file = '/lustre/cms/store/user/slezki/Filtered/forDNN/histo_y2s2y1spp_RS_2018MC_v2.root'
    p1 = 'pion1'
    p2 = 'pion2'
    signal_background_datasetMaker(p1, p2)
elif(filetype == 'sidebands'):
    root_file = '/lustre/cms/store/user/slezki/Filtered/forDNN/histo_xb2y1spp_RS_2018DataRunII_BKGsidebands_v2.root'
    p1 = 'pionP'
    p2 = 'pionM'
    additional_background_datasetMaker(p1,p2)
elif(filetype == 'WS'):
    root_file = '/lustre/cms/store/user/slezki/Filtered/forDNN/histo_xb2y1spp_WS_2018DataRunII_underY2S_v2.root'
    p1 = 'pion1'
    p2 = 'pion2'
    additional_background_datasetMaker(p1,p2)
else:
    print('ERROR: filetype not understood')
    exit()
