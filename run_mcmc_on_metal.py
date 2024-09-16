import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import pandas as pd
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as poly
import emcee
import corner
from scipy.optimize import minimize
#from IPython.display import display, Math
import matplotlib as mpl
from multiprocessing.pool import Pool
from mcmc_functions import *
import pickle
import h5py
import glob
import os

SAVE_DIR = '/DATA/pnr5sh/'
CHAINS_SAVE_DIR = '/DATA/pnr5sh/mcmc_chains/'

# read in the data (df created in fp jupyter notebook)
snztf18 = pd.read_csv('cleaning/outputs/snztf18_cleaned_binned_cut.csv')
sn19rwd = pd.read_csv('cleaning/outputs/sn19rwd_cleaned_binned_cut.csv')
sn20bio = pd.read_csv('cleaning/outputs/sn20bio_cleaned_binned_cut.csv')
sn21gno = pd.read_csv('cleaning/outputs/sn21gno_cleaned_binned_cut.csv')
sn21heh = pd.read_csv('cleaning/outputs/sn21heh_cleaned_binned_cut.csv')
sn21vgn = pd.read_csv('cleaning/outputs/sn21vgn_cleaned_binned_cut.csv')
sn22hnt = pd.read_csv('cleaning/outputs/sn22hnt_cleaned_binned_cut.csv')
sn22jpx = pd.read_csv('cleaning/outputs/sn22jpx_cleaned_binned_cut.csv')
sn22qzr = pd.read_csv('cleaning/outputs/sn22qzr_cleaned_binned_cut.csv')
# # added later
sn20ano = pd.read_csv('cleaning/outputs/sn20ano_cleaned_binned_cut.csv')
sn20ikq = pd.read_csv('cleaning/outputs/sn20ikq_cleaned_binned_cut.csv')
sn20rsc = pd.read_csv('cleaning/outputs/sn20rsc_cleaned_binned_cut.csv')
sn21pb =  pd.read_csv('cleaning/outputs/sn21pb_cleaned_binned_cut.csv')

df_list = [snztf18,sn19rwd,sn20ano,sn20bio,sn20ikq,sn20rsc,sn21gno,sn21heh,
           sn21pb, sn21vgn,sn22hnt,sn22jpx,sn22qzr]
df_str_list = ['snztf18','sn19rwd','sn20ano','sn20bio','sn20ikq','sn20rsc','sn21gno','sn21heh',
               'sn21pb', 'sn21vgn','sn22hnt','sn22jpx','sn22qzr']

# converting columns
def match_col_names(df):
    df['JD'] = df['mjd']+2400000.5
    df['mag_err_all'] = df['err']
    return

for i,df in enumerate(df_list):
    match_col_names(df)

# cut down the data
xlims_jd_ls = [[25+2.4582e6, 55+2.4582e6], [760+2.458e6, 800+2.458e6], [870+2.458e6,910+2.458e6],[2458875.5, 2458920.5],[-35+2.459e6,20+2.459e6], [79.5+2.459e6,110+2.459e6], [290+2.459e6, 320+2.459e6],[2459295.5, 2459340.5], [18+2.4592e6, 55+2.4592e6], [2459434.0, 2459480.5], [2459680.5, 2459730.5],[2459709.5, 2459740.5], [799+2.459e6, 875+2.459e6]]

snztf18_cut = slice_df(snztf18,xlims_jd_ls[0])
sn19rwd_cut = slice_df(sn19rwd,xlims_jd_ls[1])
sn20ano_cut = slice_df(sn20ano,xlims_jd_ls[2])
sn20bio_cut = slice_df(sn20bio,xlims_jd_ls[3])
sn20ikq_cut = slice_df(sn20ikq,xlims_jd_ls[4])
sn20rsc_cut = slice_df(sn20rsc,xlims_jd_ls[5])
sn21gno_cut = slice_df(sn21gno,xlims_jd_ls[6])
sn21heh_cut = slice_df(sn21heh,xlims_jd_ls[7])
sn21pb_cut = slice_df(sn21pb, xlims_jd_ls[8])
sn21vgn_cut = slice_df(sn21vgn,xlims_jd_ls[9])
sn22hnt_cut = slice_df(sn22hnt,xlims_jd_ls[10])
sn22jpx_cut = slice_df(sn22jpx,xlims_jd_ls[11])
sn22qzr_cut = slice_df(sn22qzr,xlims_jd_ls[12])

df_cut_list = [snztf18_cut,sn19rwd_cut,sn20ano_cut,sn20bio_cut,sn20ikq_cut,sn20rsc_cut,sn21gno_cut,
               sn21heh_cut,sn21pb_cut,sn21vgn_cut,sn22hnt_cut,sn22jpx_cut,sn22qzr_cut]

a2_inds = [14,22,38,1,1,32,14,27,24,10,42,10,49]

#actually normalizing the LCs to a2
for i,df in enumerate(df_cut_list):
    norm_LC_a2(df, a2_inds[i])

# run subselect func
snztf18_sub,a0 = subselect(df_cut_list[0])
sn19rwd_sub,a1 = subselect(df_cut_list[1])
sn20ano_sub,a2 = subselect(df_cut_list[2])
sn20bio_sub,a3 = subselect(df_cut_list[3])
sn20ikq_sub,a4 = subselect(df_cut_list[4])
sn20rsc_sub,a5 = subselect(df_cut_list[5])
sn21gno_sub,a6 = subselect(df_cut_list[6])
sn21heh_sub,a7 = subselect(df_cut_list[7])
sn21pb_sub,a8 = subselect(df_cut_list[8])
sn21vgn_sub,a9 = subselect(df_cut_list[9])
sn22hnt_sub,a10 = subselect(df_cut_list[10])
sn22jpx_sub,a11 = subselect(df_cut_list[11])
sn22qzr_sub,a12 = subselect(df_cut_list[12])

df_sub_ls = [snztf18_sub,sn19rwd_sub,sn20ano_sub,sn20bio_sub,sn20ikq_sub,sn20rsc_sub,sn21gno_sub,sn21heh_sub,
             sn21pb_sub,sn21vgn_sub,sn22hnt_sub,sn22jpx_sub,sn22qzr_sub]
new_a2_inds = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12]

# def boolean list of rise1
r1_bool_dict = {
                "g": {"snztf18":True,
                                  "sn19rwd":True,
                                                'sn20ano':False,
                                                              "sn20bio":True,
                                                                            'sn20ikq':True,
                                                                                          'sn20rsc':True,
                                                                                                        "sn21gno":True,
                                                                                                                      "sn21heh":False,
                                                                                                                                    'sn21pb':False,
                                                                                                                                                  "sn21vgn":True,
                                                                                                                                                                'sn22hnt':False,
                                                                                                                                                                              'sn22jpx':False,
                                                                                                                                                                                            'sn22qzr':False,
                                                                                                                                                                                                          },
                        "r": {"snztf18":True,
                                          "sn19rwd":True,
                                                        'sn20ano':True,
                                                                      "sn20bio":False,
                                                                                    'sn20ikq':True,
                                                                                                  'sn20rsc':True,
                                                                                                                "sn21gno":True,
                                                                                                                              "sn21heh":False,
                                                                                                                                            'sn21pb':True,
                                                                                                                                                          "sn21vgn":True,
                                                                                                                                                                        'sn22hnt':True,
                                                                                                                                                                                      'sn22jpx':True,
                                                                                                                                                                                                    'sn22qzr':True,
                                                                                                                                                                                                                  }
                            }
r1_g_bools = r1_bool_dict['g'].values()
r1_r_bools = r1_bool_dict['r'].values()

# actually creating list of prior space limits
snztf18_pdict,sn19rwd_pdict,sn20ano_pdict,sn20bio_pdict,sn20ikq_pdict,sn20rsc_pdict,sn21gno_pdict,sn21pb_pdict,sn21heh_pdict,sn21vgn_pdict,sn22hnt_pdict,sn22jpx_pdict,sn22qzr_pdict=[],[],[],[],[],[],[],[],[],[],[],[],[]
temp_pdict = [snztf18_pdict,sn19rwd_pdict,sn20ano_pdict,sn20bio_pdict,sn20ikq_pdict,sn20rsc_pdict,sn21gno_pdict,sn21heh_pdict,sn21pb_pdict,sn21vgn_pdict,sn22hnt_pdict,sn22jpx_pdict,sn22qzr_pdict]

for i,sub in enumerate(df_sub_ls):
    print('ON SNE', df_str_list[i])
    df_str = df_str_list[i]
    p = calc_prior(sub, r1_g=r1_bool_dict['g'][df_str], r1_r=r1_bool_dict['r'][df_str])
    temp_pdict[i].append(p) #applies extra 1st dimension

pdict_ls = [snztf18_pdict[0],sn19rwd_pdict[0],sn20ano_pdict[0],sn20bio_pdict[0],sn20ikq_pdict[0],sn20rsc_pdict[0],
            sn21gno_pdict[0],sn21heh_pdict[0],sn21pb_pdict[0], sn21vgn_pdict[0],sn22hnt_pdict[0],sn22jpx_pdict[0],sn22qzr_pdict[0]]

# run p0_calc
snztf18_p0,sn19rwd_p0,sn20ano_p0,sn20bio_p0,sn20ikq_p0,sn20rsc_p0,sn21gno_p0,sn21heh_p0,sn21pb_p0,sn21vgn_p0,sn22hnt_p0,sn22jpx_p0,sn22qzr_p0=[],[],[],[],[],[],[],[],[],[],[],[],[]
temp_p0s = [snztf18_p0,sn19rwd_p0,sn20ano_p0,sn20bio_p0,sn20ikq_p0,sn20rsc_p0,sn21gno_p0,sn21heh_p0,sn21pb_p0,sn21vgn_p0,sn22hnt_p0,sn22jpx_p0,sn22qzr_p0]

for i,pdict in enumerate(pdict_ls):
    p = p0_calc(pdict)
    temp_p0s[i].append(p) #applies extra 1st dimension

p0s = [snztf18_p0[0],sn19rwd_p0[0],sn20ano_p0[0],sn20bio_p0[0],sn20ikq_p0[0],sn20rsc_p0[0],sn21gno_p0[0],sn21heh_p0[0],sn21pb_p0[0],sn21vgn_p0[0],sn22hnt_p0[0],sn22jpx_p0[0],sn22qzr_p0[0]]

#sep out initial guesses for multiproc
g_p0 = [p0s[i][0] for i in range(len(p0s))]
r_p0 = [p0s[i][1] for i in range(len(p0s))]

##############################################
#
#    RUN ALL MCMC AND PLOT RESULTS
#
###############################################
#k = 9
# G BAND FITS
# create method to run subset of sample
df_strs = df_str_list
df_subs = df_sub_ls
#r1_g_bools = r1_bool_dict['g']
gp0s = g_p0
pdicts = pdict_ls

# R BAND FITS
# create method to run subset of sample
rp0s = r_p0
#r1_r_bools = [r1_bool_dict['r'][df_str_list[k]]]

def run_mcmc(run=False, g=False, r=False, date='month00'):
    if run:
        print(f"RUNNING MCMC")
        if g:
            pool=Pool(len(df_strs))
            inputs = zip(df_strs, df_subs, ['g']*len(df_subs), r1_g_bools, gp0s, pdicts)
            all_fits_g = []

            with open(SAVE_DIR+'bestfits_g_'+date+'.txt', 'a') as savefile:
                for fit in pool.map(mp_fit_sne, inputs):
                    np.savetxt(savefile,fit)
                    all_fits_g.append(fit)
            pool.close()
            savefile.close()
        
        if r:
            # R BAND FITS
            pool=Pool(len(df_strs))
            inputs = zip(df_strs, df_subs, ['r']*len(df_subs), r1_r_bools, rp0s, pdicts)
            all_fits_r = []

            with open(SAVE_DIR+'bestfits_r_'+date+'.txt', 'a') as savefile:
                for fit in pool.map(mp_fit_sne, inputs):
                    np.savetxt(savefile,fit)
                    all_fits_r.append(fit)
            pool.close()
            savefile.close()
    else:
        print(f"NOT running MCMC")
    return

run_mcmc(run=True, g=True, r=True, date='sep10')
