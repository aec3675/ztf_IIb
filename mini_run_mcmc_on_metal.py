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

sn20ano = pd.read_csv('./cleaning/outputs/sn20ano_cleaned_binned_cut.csv')
sn20ano['JD'] = sn20ano['mjd']+2400000.5
sn20ano['mag_err_all'] = sn20ano['err']
df_str = 'sn20ano'

a2_ind = 36

norm_LC_a2(sn20ano, a2_ind)

sn20ano_sub,a2 = subselect(sn20ano)

r1_bool_dict = {
        "g": {"sn20ano":False},
        "r": {"sn20ano":True}
}

pdict = calc_prior(sn20ano_sub, r1_g=r1_bool_dict['g'][df_str], r1_r=r1_bool_dict['r'][df_str])

p0 = p0_calc(pdict)
g_p0 = p0[0]
r_p0 = p0[1]


df_strs = [df_str]
df_subs = [sn20ano_sub]
r1_g_bools = [r1_bool_dict['g'][df_str]]
gp0s = [g_p0]
pdicts = [pdict]

# R BAND FITS
# create method to run subset of sample
rp0s = [r_p0]
r1_r_bools = [r1_bool_dict['r'][df_str]]

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

run_mcmc(run=True, g=True, r=True, date='aug29')