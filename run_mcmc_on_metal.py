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
snztf18 = pd.read_csv('./data/forced_phot_data/snztf18_ztf_atlas_df.csv',index_col='index')
sn19rwd = pd.read_csv('./data/forced_phot_data/sn19rwd_ztf_atlas_df.csv',index_col='index')
sn20ano = pd.read_csv('./data/forced_phot_data/sn20ano_ztf_atlas_df.csv', index_col='index')
sn20ikq = pd.read_csv('./data/forced_phot_data/sn20ikq_ztf_atlas_df.csv', index_col='index')
sn20rsc = pd.read_csv('./data/forced_phot_data/sn20rsc_ztf_atlas_df.csv', index_col='index')
sn21gno = pd.read_csv('./data/forced_phot_data/sn21gno_ztf_atlas_df.csv',index_col='index')
sn21heh = pd.read_csv('./data/forced_phot_data/sn21heh_ztf_atlas_df.csv',index_col='index')
sn21pb =  pd.read_csv('./data/forced_phot_data/sn21pb_ztf_atlas_df.csv', index_col='index')
sn21vgn = pd.read_csv('./data/forced_phot_data/sn21vgn_ztf_atlas_df.csv',index_col='index')
sn22hnt = pd.read_csv('./data/forced_phot_data/sn22hnt_ztf_atlas_df.csv',index_col='index')
sn22jpx = pd.read_csv('./data/forced_phot_data/sn22jpx_ztf_atlas_df.csv',index_col='index')
sn22qzr = pd.read_csv('./data/forced_phot_data/sn22qzr_ztf_atlas_df.csv',index_col='index')
#added sept
sn20sbw = pd.read_csv('./data/forced_phot_data/sn20sbw_ztf_atlas_df.csv',index_col='index')
sn18dfi = pd.read_csv('./data/forced_phot_data/sn18dfi_ztf_atlas_df.csv',index_col='index')

df_list = [snztf18,sn19rwd,sn20ano,sn20ikq,sn20rsc,sn21gno,sn21heh,
           sn21pb, sn21vgn,sn22hnt,sn22jpx,sn22qzr,sn20sbw,sn18dfi]
df_str_list = ['snztf18','sn19rwd','sn20ano','sn20ikq','sn20rsc','sn21gno','sn21heh',
               'sn21pb', 'sn21vgn','sn22hnt','sn22jpx','sn22qzr','sn20sbw','sn18dfi']

for i,df in enumerate(df_list):
    convert_ztf_err(df)
    merge_err_cols(df)


# cut down the data
xlims_jd_ls = [[25+2.4582e6, 55+2.4582e6], [761+2.458e6, 800+2.458e6], [869+2.458e6,910+2.458e6],
               [-35+2.459e6,20+2.459e6], [78+2.459e6,110+2.459e6], [290+2.459e6, 320+2.459e6], 
               [2459294.5, 2459340.5], [18+2.4592e6, 55+2.4592e6], [2459432.0, 2459480.5], [2459680.5, 2459730.5], 
               [2459706.5, 2459740.5], [800+2.459e6, 875+2.459e6], [2459086.5, 2459140.5], [2458305.0, 2458364.79]]

snztf18_cut = slice_df(snztf18,xlims_jd_ls[0])
sn19rwd_cut = slice_df(sn19rwd,xlims_jd_ls[1])
sn20ano_cut = slice_df(sn20ano,xlims_jd_ls[2])
sn20ikq_cut = slice_df(sn20ikq,xlims_jd_ls[3])
sn20rsc_cut = slice_df(sn20rsc,xlims_jd_ls[4])
sn21gno_cut = slice_df(sn21gno,xlims_jd_ls[5])
sn21heh_cut = slice_df(sn21heh,xlims_jd_ls[6])
sn21pb_cut = slice_df(sn21pb, xlims_jd_ls[7])
sn21vgn_cut = slice_df(sn21vgn,xlims_jd_ls[8])
sn22hnt_cut = slice_df(sn22hnt,xlims_jd_ls[9])
sn22jpx_cut = slice_df(sn22jpx,xlims_jd_ls[10])
sn22qzr_cut = slice_df(sn22qzr,xlims_jd_ls[11])
sn20sbw_cut = slice_df(sn20sbw,xlims_jd_ls[12])
sn18dfi_cut = slice_df(sn18dfi,xlims_jd_ls[13])


df_cut_list = [snztf18_cut,sn19rwd_cut,sn20ano_cut,sn20ikq_cut,sn20rsc_cut,sn21gno_cut,
               sn21heh_cut,sn21pb_cut,sn21vgn_cut,sn22hnt_cut,sn22jpx_cut,sn22qzr_cut, sn20sbw_cut, sn18dfi_cut]

a2_inds = [51, 23, 62, 80, 35, 51, 8, 35, 100, 9, 56, 1, 10, 45]

#actually normalizing the LCs to a2
for i,df in enumerate(df_cut_list):
    norm_LC_a2(df, a2_inds[i])

# run subselect func
snztf18_sub,a0 = subselect(df_cut_list[0])
sn19rwd_sub,a1 = subselect(df_cut_list[1])
sn20ano_sub,a2 = subselect(df_cut_list[2])
sn20ikq_sub,a4 = subselect(df_cut_list[3])
sn20rsc_sub,a5 = subselect(df_cut_list[4])
sn21gno_sub,a6 = subselect(df_cut_list[5])
sn21heh_sub,a7 = subselect(df_cut_list[6])
sn21pb_sub,a8 = subselect(df_cut_list[7])
sn21vgn_sub,a9 = subselect(df_cut_list[8])
sn22hnt_sub,a10 = subselect(df_cut_list[9])
sn22jpx_sub,a11 = subselect(df_cut_list[10])
sn22qzr_sub,a12 = subselect(df_cut_list[11])
sn20sbw_sub,_ = subselect(df_cut_list[12])
sn18dfi_sub,_ = subselect(df_cut_list[13])


df_sub_ls = [snztf18_sub,sn19rwd_sub,sn20ano_sub,sn20ikq_sub,sn20rsc_sub,sn21gno_sub,sn21heh_sub,
             sn21pb_sub,sn21vgn_sub,sn22hnt_sub,sn22jpx_sub,sn22qzr_sub, sn20sbw_sub,sn18dfi_sub]

# def boolean list of rise1
r1_bool_dict = {
        "g": {"snztf18":True,
              "sn19rwd":True,
              'sn20ano':False,
              'sn20ikq':True,
              'sn20rsc':True,
              "sn21gno":True,
              "sn21heh":False,
              'sn21pb':False,
              "sn21vgn":True,
              'sn22hnt':False,
              'sn22jpx':False,
              'sn22qzr':False,
              'sn20sbw':True,
              'sn18dfi':True,
             },          
        "r": {"snztf18":True,
              "sn19rwd":True,
              'sn20ano':True,
              'sn20ikq':True,
              'sn20rsc':True,
              "sn21gno":True,
              "sn21heh":True,
              'sn21pb':True,
              "sn21vgn":True,
              'sn22hnt':True,
              'sn22jpx':True,
              'sn22qzr':True,
              'sn20sbw':True,
              'sn18dfi':True,
            }
    }
r1_g_bools = r1_bool_dict['g'].values()
r1_r_bools = r1_bool_dict['r'].values()

pdict_all = {
    'snztf18':{"g": {"m1_l":-4.0,"m1_u":0.0,
            "m2_l":0.1,"m2_u":0.4,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-5.5,"b2_u":0.0,
            "a1_l":4.976226799655706, "a1_u": 10.0,
            "a2_l":10.0, "a2_u":16.5,
            "log_f_l":-6.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":4.981712899636477, "a1_u":11.024212900083512,
            "a2_l":11.024212900083512, "a2_u":20.957604099996388,
            "log_f_l":-6.0,"log_f_u":4.0}
            },
    'sn19rwd':{"g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.1,"m2_u":0.4,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-5.5,"b2_u":0.0,
            "a1_l":6.86876150034368, "a1_u": 13.057245300151408,
            "a2_l":13.057245300151408, "a2_u":17.885092600248754,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":6.840046300087124, "a1_u":12.083327900152653,
            "a2_l":12.083327900152653, "a2_u":17.99365740036592,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn20ano':{"g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.34,"m2_u":0.85,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-8.0,"b2_u":0.0,
            "a1_l":8.0, "a1_u": 11.0,
            "a2_l":10.0, "a2_u":18.0,
            "log_f_l":-6.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":6.156779400072992, "a1_u":12.1571254003793,
            "a2_l":12.1571254003793, "a2_u":17.063321799971163,
            "log_f_l":-6.0,"log_f_u":4.0}
    },
    'sn20ikq':{"g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.06,"m2_u":0.25,
            "m3_l":-0.1,"m3_u":0.05,
            "b2_l":-5.5,"b2_u":0.0,
            "a1_l":-1.1329234996810555, "a1_u": 13.5,
            "a2_l":9.5, "a2_u":16.980994999874383,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":-0.05,"m2_u":0.16,
            "m3_l":-0.08,"m3_u":0.04,
            "b2_l":-2.0,"b2_u":0.0,
            "a1_l":-1.0, "a1_u":4.0,
            "a2_l":10, "a2_u":20.0,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn20rsc':{"g": {"m1_l":-5.0,"m1_u":-0.1,
            "m2_l":0.1,"m2_u":0.4,
            "m3_l":-0.25,"m3_u":0.15,
            "b2_l":-6.6,"b2_u":-3.6,
            "a1_l":6.0, "a1_u": 11.5,
            "a2_l":14.4, "a2_u":20.0,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.6,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":6.0, "a1_u":12.00540510006249,
            "a2_l":12.00540510006249, "a2_u":17.936087999958545,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn21gno':{"g": {"m1_l":-4.0,"m1_u":0.0,
            "m2_l":-0.2,"m2_u":0.6,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-5.5,"b2_u":0.0,
            "a1_l":7.8987962999381125, "a1_u": 12.959537100046873,
            "a2_l":12.959537100046873, "a2_u":18.959502399899065,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-4.0,"m1_u":0.0,
            "m2_l":-0.12,"m2_u":0.6,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":1.0,
            "a1_l":7.959467599634081, "a1_u":12.964421299751848,
            "a2_l":12.964421299751848, "a2_u":18.86081019975245,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn21heh':{"g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.1,"m2_u":0.4,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-5.5,"b2_u":0.0,
            "a1_l":-2.069155099801719, "a1_u": 13.024780100211501,
            "a2_l":11.024780100211501, "a2_u":16.984375,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.6,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":1.0, "a1_u":12.151536900084466,
            "a2_l":12.151536900084466, "a2_u":16.949224499985576,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn21pb':{"g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.1,"m2_u":0.55,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-7.5,"b2_u":0.0,
            "a1_l":9.383196800015867, "a1_u": 13,
            "a2_l":10.0, "a2_u":18.0,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":7.026162000373006, "a1_u":13.5,
            "a2_l":11.0, "a2_u":17.902143600396812,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn21vgn':{"g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.1,"m2_u":0.4,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-5.5,"b2_u":0.0,
            "a1_l":1.9684039000421762, "a1_u": 12.935359899885952,
            "a2_l":12.935359899885952, "a2_u":16.5,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":3.0539090000092983, "a1_u":12.87760529993102,
            "a2_l":12.87760529993102, "a2_u":20.0,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn22hnt':{"g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.1,"m2_u":0.6,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-7.5,"b2_u":0.0,
            "a1_l":8.511018499732018, "a1_u": 9.011018499732018,
            "a2_l":11.5, "a2_u":17.5,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":5.956030100118369, "a1_u":13.035427199676633,
            "a2_l":13.035427199676633, "a2_u":16.5,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn22jpx':{"g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":-0.12,"m2_u":0.4,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-5.5,"b2_u":0.0,
            "a1_l":5.58201489970088, "a1_u": 6.08201489970088,
            "a2_l":7.0, "a2_u":14.5,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":1.9527329998090863, "a1_u":12.217712000012398,
            "a2_l":9.0, "a2_u":14.5,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn22qzr':{"g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.1,"m2_u":0.4,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-6.0,"b2_u":-2.0,
            "a1_l":8.524236100260168, "a1_u": 9.024236100260168,
            "a2_l":9.024236100260168, "a2_u":17.986064800061285,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-6.0,"b2_u":-1.5,
            "a1_l":5.87473600031808, "a1_u":10.0,
            "a2_l":12.010590299963951, "a2_u":16.5,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn20sbw':{"g": {"m1_l":-7.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-6.0,"b2_u":0.0,
            "a1_l":10, "a1_u": 15,
            "a2_l":10, "a2_u":20,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-6.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.8,
            "m3_l":-0.35,"m3_u":-0.12,
            "b2_l":-11.0,"b2_u":-4.5,
            "a1_l":7, "a1_u":15,
            "a2_l":13, "a2_u":17,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
    'sn18dfi':{"g": {"m1_l":-6.0,"m1_u":0.0,
            "m2_l":0.06,"m2_u":0.24,
            "m3_l":-0.15,"m3_u":0.0,
            "b2_l":-6.0,"b2_u":0.0,
            "a1_l":0, "a1_u": 15,
            "a2_l":10, "a2_u":20,
            "log_f_l":-4.0,"log_f_u":4.0},
        "r": {"m1_l":-3.5,"m1_u":0.1,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":0, "a1_u":8,
            "a2_l":10, "a2_u":20,
            "log_f_l":-4.0,"log_f_u":4.0}
    },
}

p0s = []
for i,sn in enumerate(df_str_list):
    p = p0_calc(pdict_all[sn])
    p0s.append(p)

#sep out initial guesses for multiproc
g_p0 = [p0s[i][0] for i in range(len(p0s))]
r_p0 = [p0s[i][1] for i in range(len(p0s))]

##############################################
#
#    RUN ALL MCMC AND PLOT RESULTS
#
###############################################

# G BAND FITS
df_strs = df_str_list
df_subs = df_sub_ls
#r1_g_bools = r1_bool_dict['g']
gp0s = g_p0
pdicts = [pdict_all[sn] for sn in df_str_list]

# R BAND FITS
rp0s = r_p0
#r1_r_bools = [r1_bool_dict['r'][df_str_list[k]]]


#to run subset:
#20sbw=12, 18dfi=13
df_strs = df_str_list[12:]
df_subs = df_sub_ls[12:]
pdicts = [pdict_all[sn] for sn in df_strs]
#G BAND
r1_g_bools = [r1_bool_dict['g'][sn] for sn in df_strs]
gp0s = g_p0[12:]
# R BAND FITS
rp0s = r_p0[12:]
r1_r_bools = [r1_bool_dict['r'][sn] for sn in df_strs]

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

run_mcmc(run=True, g=False, r=True, date='sep30')
