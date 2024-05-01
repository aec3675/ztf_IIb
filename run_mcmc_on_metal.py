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
from IPython.display import display, Math
import matplotlib as mpl
from multiprocessing.pool import Pool
from mcmc_functions import *
import pickle
import h5py
import glob
import os

# read in the data (df created in fp jupyter notebook)
snztf18 = pd.read_csv('./forced_phot_data/snztf18_ztf_atlas_df.csv',index_col='index')
sn19rwd = pd.read_csv('./forced_phot_data/sn19rwd_ztf_atlas_df.csv',index_col='index')
sn20bio = pd.read_csv('./forced_phot_data/sn20bio_ztf_atlas_df.csv',index_col='index')
sn21aezx = pd.read_csv('./forced_phot_data/sn21aezx_ztf_atlas_df.csv',index_col='index')
sn21gno = pd.read_csv('./forced_phot_data/sn21gno_ztf_atlas_df.csv',index_col='index')
sn21heh = pd.read_csv('./forced_phot_data/sn21heh_ztf_atlas_df.csv',index_col='index')
sn21vgn = pd.read_csv('./forced_phot_data/sn21vgn_ztf_atlas_df.csv',index_col='index')
sn22hnt = pd.read_csv('./forced_phot_data/sn22hnt_ztf_atlas_df.csv',index_col='index')
sn22jpx = pd.read_csv('./forced_phot_data/sn22jpx_ztf_atlas_df.csv',index_col='index')
sn22qzr = pd.read_csv('./forced_phot_data/sn22qzr_ztf_atlas_df.csv',index_col='index')
# # added later
sn20ano = pd.read_csv('./forced_phot_data/sn20ano_ztf_atlas_df.csv', index_col='index')
sn20ikq = pd.read_csv('./forced_phot_data/sn20ikq_ztf_atlas_df.csv', index_col='index')
sn20rsc = pd.read_csv('./forced_phot_data/sn20rsc_ztf_atlas_df.csv', index_col='index')

df_list = [snztf18,sn19rwd,sn20ano,sn20bio,sn20ikq,sn20rsc,sn21aezx,sn21gno,sn21heh,sn21vgn,sn22hnt,sn22jpx,sn22qzr]
df_str_list = ['snztf18','sn19rwd','sn20ano','sn20bio','sn20ikq','sn20rsc','sn21aezx','sn21gno','sn21heh','sn21vgn','sn22hnt','sn22jpx','sn22qzr']
# df_list = [sn20ano, sn20ikq, sn20rsc]
# df_str_list = ['sn20ano', 'sn20ikq', 'sn20rsc']

# sn20ano.columns

for i,df in enumerate(df_list):
    convert_ztf_err(df)

# cut down the data
#df_str_list = ['snztf18','sn19rwd','sn20ano','sn20bio','sn20ikq','sn20rsc','sn21aezx','sn21gno','sn21heh','sn21vgn','sn22hnt','sn22jpx','sn22qzr']
#taken from forced_phot_nb
xlims_jd_ls = [[25+2.4582e6, 55+2.4582e6], [760+2.458e6, 800+2.458e6], [869+2.458e6,910+2.458e6],[2458875.5, 2458920.5], 
               [-35+2.459e6,20+2.459e6], [78+2.459e6,110+2.459e6],[530+2.459e6, 600+2.459e6], [290+2.459e6, 320+2.459e6], 
               [2459290.5, 2459340.5], [2459430.5, 2459480.5], [2459680.5, 2459730.5], [2459706.5, 2459740.5], [799+2.459e6, 875+2.459e6]]

snztf18_cut = slice_df(snztf18,xlims_jd_ls[0])
sn19rwd_cut = slice_df(sn19rwd,xlims_jd_ls[1])
sn20ano_cut = slice_df(sn20ano,xlims_jd_ls[2])
sn20bio_cut = slice_df(sn20bio,xlims_jd_ls[3])
sn20ikq_cut = slice_df(sn20ikq,xlims_jd_ls[4])
sn20rsc_cut = slice_df(sn20rsc,xlims_jd_ls[5])
sn21aezx_cut = slice_df(sn21aezx,xlims_jd_ls[6])
sn21gno_cut = slice_df(sn21gno,xlims_jd_ls[7])
sn21heh_cut = slice_df(sn21heh,xlims_jd_ls[8])
sn21vgn_cut = slice_df(sn21vgn,xlims_jd_ls[9])
sn22hnt_cut = slice_df(sn22hnt,xlims_jd_ls[10])
sn22jpx_cut = slice_df(sn22jpx,xlims_jd_ls[11])
sn22qzr_cut = slice_df(sn22qzr,xlims_jd_ls[12])

df_cut_list = [snztf18_cut,sn19rwd_cut,sn20ano_cut,sn20bio_cut,sn20ikq_cut,sn20rsc_cut,sn21aezx_cut,sn21gno_cut,sn21heh_cut,sn21vgn_cut,sn22hnt_cut,sn22jpx_cut,sn22qzr_cut]

# %matplotlib inline
# for i,df in enumerate(df_cut_list):
#     plt.figure()
#     plt.scatter(df[df['filter']=='ZTF_g']['JD'], df[df['filter']=='ZTF_g']['mag'],edgecolors='green',facecolors='none',label='ZTF_g',alpha=0.5)
#     plt.scatter(df[df['filter']=='ZTF_r']['JD'], df[df['filter']=='ZTF_r']['mag'],edgecolors='red',facecolors='none',label='ZTF_r',alpha=0.5)
#     plt.scatter(df[df['filter']=='c']['JD'], df[df['filter']=='c']['mag'],color='green', marker='d',label='ATLAS c',alpha=0.5)
#     plt.scatter(df[df['filter']=='o']['JD'], df[df['filter']=='o']['mag'],color='red', marker='d', label='ATLAS o',alpha=0.5)
#     plt.gca().invert_yaxis()
#     # plt.ylim(0,30)
#     plt.legend(loc='lower right')
#     plt.title(df_str_list[i])

a2_inds = [52, 25, 68, 22,73,27,5, 9, 10, 116, 11, 51, 2]

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
sn21aezx_sub,a6 = subselect(df_cut_list[6])
sn21gno_sub,a7 = subselect(df_cut_list[7])
sn21heh_sub,a8 = subselect(df_cut_list[8])
sn21vgn_sub,a9 = subselect(df_cut_list[9])
sn22hnt_sub,a10 = subselect(df_cut_list[10])
sn22jpx_sub,a11 = subselect(df_cut_list[11])
sn22qzr_sub,a12 = subselect(df_cut_list[12])

df_sub_ls = [snztf18_sub,sn19rwd_sub,sn20ano_sub,sn20bio_sub,sn20ikq_sub,sn20rsc_sub,sn21aezx_sub,sn21gno_sub,sn21heh_sub,sn21vgn_sub,sn22hnt_sub,sn22jpx_sub,sn22qzr_sub]
new_a2_inds = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12]

#create 1 column of all errors
for i,df in enumerate(df_sub_ls):
    merge_err_cols(df)

# %matplotlib inline
# for i,df in enumerate(df_sub_ls):
#     plt.figure()
#     plt.errorbar(df[df['filter']=='ZTF_g']['norm_t'], df[df['filter']=='ZTF_g']['norm_m'],df[df['filter']=='ZTF_g']['mag_err_all'],
#                  markeredgecolor='green',markerfacecolor='none',ecolor='green',label='ZTF_g',alpha=0.5,linestyle='',marker='o')
#     plt.errorbar(df[df['filter']=='ZTF_r']['norm_t'], df[df['filter']=='ZTF_r']['norm_m'],df[df['filter']=='ZTF_r']['mag_err_all'],
#                  markeredgecolor='red',markerfacecolor='none',ecolor='red',label='ZTF_r',alpha=0.5,linestyle='',marker='o')
#     plt.errorbar(df[df['filter']=='c']['norm_t'], df[df['filter']=='c']['norm_m'],df[df['filter']=='c']['mag_err_all'],
#                  color='green', marker='d',label='ATLAS c',alpha=0.5,linestyle='')
#     plt.errorbar(df[df['filter']=='o']['norm_t'], df[df['filter']=='o']['norm_m'],df[df['filter']=='o']['mag_err_all'],
#                  color='red', marker='d', label='ATLAS o',alpha=0.5,linestyle='')
#     plt.gca().invert_yaxis()
#     plt.legend(loc='lower right')
#     plt.title(df_str_list[i])
#     plt.xlabel('Normalized Time [days]')
#     plt.ylabel('Normalized Magnitude')

# def boolean list of rise1
r1_bools = [True,True,True,True,True,True,True,True,True,True,True,True,True]
r1_g_bools = [True, True, True, True, True,True, True, True, True, True, False, False, False]
r1_r_bools = [True,True,True,True,True,True,True,True,True,True,True,True,True]
r1_bool_dict = {
        "g": {"snztf18":True,
              "sn19rwd":True,
              "sn20bio":True,
              "sn21aezx":True,
              "sn21gno":True,
              "sn21heh":True,
              "sn21vgn":True,
              'sn22hnt':False,
              'sn22jpx':False,
              'sn22qzr':False,
              'sn20ano':True,
              'sn20ikq':True,
              'sn20rsc':True},
        "r": {"snztf18":True,
              "sn19rwd":True,
              "sn20bio":True,
              "sn21aezx":True,
              "sn21gno":True,
              "sn21heh":True,
              "sn21vgn":True,
              'sn22hnt':True,
              'sn22jpx':True,
              'sn22qzr':True,
              'sn20ano':True,
              'sn20ikq':True,
              'sn20rsc':True}
    }

# actually creating list of prior space limits
snztf18_pdict,sn19rwd_pdict,sn20ano_pdict,sn20bio_pdict,sn20ikq_pdict,sn20rsc_pdict,sn21aezx_pdict,sn21gno_pdict,sn21heh_pdict,sn21vgn_pdict,sn22hnt_pdict,sn22jpx_pdict,sn22qzr_pdict=[],[],[],[],[],[],[],[],[],[],[],[],[]
temp_pdict = [snztf18_pdict,sn19rwd_pdict,sn20ano_pdict,sn20bio_pdict,sn20ikq_pdict,sn20rsc_pdict,sn21aezx_pdict,sn21gno_pdict,sn21heh_pdict,sn21vgn_pdict,sn22hnt_pdict,sn22jpx_pdict,sn22qzr_pdict]

for i,sub in enumerate(df_sub_ls):
    # print('ON SNE', df_str_list[i])
    p = calc_prior(sub, r1_g=r1_g_bools[i], r1_r=r1_r_bools[i])
    temp_pdict[i].append(p) #applies extra 1st dimension

pdict_ls = [snztf18_pdict[0],sn19rwd_pdict[0],sn20ano_pdict[0],sn20bio_pdict[0],sn20ikq_pdict[0],sn20rsc_pdict[0],
            sn21aezx_pdict[0],sn21gno_pdict[0],sn21heh_pdict[0],sn21vgn_pdict[0],sn22hnt_pdict[0],sn22jpx_pdict[0],sn22qzr_pdict[0]]

# run p0_calc
snztf18_p0,sn19rwd_p0,sn20ano_p0,sn20bio_p0,sn20ikq_p0,sn20rsc_p0,sn21aezx_p0,sn21gno_p0,sn21heh_p0,sn21vgn_p0,sn22hnt_p0,sn22jpx_p0,sn22qzr_p0=[],[],[],[],[],[],[],[],[],[],[],[],[]
temp_p0s = [snztf18_p0,sn19rwd_p0,sn20ano_p0,sn20bio_p0,sn20ikq_p0,sn20rsc_p0,sn21aezx_p0,sn21gno_p0,sn21heh_p0,sn21vgn_p0,sn22hnt_p0,sn22jpx_p0,sn22qzr_p0]

for i,pdict in enumerate(pdict_ls):
    p = p0_calc(pdict)
    temp_p0s[i].append(p) #applies extra 1st dimension

p0s = [snztf18_p0[0],sn19rwd_p0[0],sn20ano_p0[0],sn20bio_p0[0],sn20ikq_p0[0],sn20rsc_p0[0],sn21aezx_p0[0],sn21gno_p0[0],sn21heh_p0[0],sn21vgn_p0[0],sn22hnt_p0[0],sn22jpx_p0[0],sn22qzr_p0[0]]

#sep out initial guesses for multiproc
g_p0 = [p0s[i][0] for i in range(len(p0s))]
r_p0 = [p0s[i][1] for i in range(len(p0s))]

##############################################
#
#    RUN ALL MCMC AND PLOT RESULTS
#
###############################################

# create method to run subset of sample
df_strs = [df_str_list[2],df_str_list[4], df_str_list[5]]
df_subs = [df_sub_ls[2], df_sub_ls[4], df_sub_ls[5]]
r1_bools = [True,True,True]
gp0s = [g_p0[2],g_p0[4], g_p0[5]]
pdicts = [pdict_ls[2], pdict_ls[4], pdict_ls[5]]

pool=Pool(3)
inputs = zip(df_strs, df_subs, ['g']*len(df_subs), r1_bools, gp0s, pdicts)
all_fits_g = []

with open(SAVE_DIR+'bestfits_g_may1.txt', 'a') as savefile:
    for fit in pool.map(mp_fit_sne, inputs):
        np.savetxt(savefile,fit)
        all_fits_g.append(fit)
pool.close()
savefile.close()


# # RUN IF WANT TO CONVERT H5 CHAIN FILES TO FLAT CHAINS LOCALLY
# PATH_TO_G_CHAINS = '/DATA/pnr5sh/mcmc_chains/g_chains/'
# PATH_TO_R_CHAINS = '/DATA/pnr5sh/mcmc_chains/r_chains/'
# h5_2_txt(PATH_TO_G_CHAINS)
# h5_2_txt(PATH_TO_R_CHAINS)