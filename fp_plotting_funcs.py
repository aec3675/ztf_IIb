import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

plt.rc('xtick', top=True, direction='in', labelsize=12)
plt.rc('ytick', right=True, direction='in', labelsize=12)
plt.rcParams.update({'font.size': 12})



##########################################################################
#                   READING AND FORMATTING DATA
##########################################################################



def format_df(df):
    '''
    redfines indeces for ztf_fp data
    '''
    df = df.set_index(df['index'])  # manually setting indeces
    df = df.drop(columns=['index']) # removing duplicated index column
    return df


def mjd_2_jd(df, source='atlas'):
    '''
    converting from MJD to JD
    '''
    if source == 'atlas':
        df.insert(0, 'JD', df['MJD']+2400000.5)
    else:
        df.insert(0, 'jd', df['mjd']+2400000.5)
    return df


def convert2mag(df):
    '''
    converting the ztf fp data flux to mag and appending to df
    '''
    m = df.zpdiff - 2.5*np.log10(df.forcediffimflux) #magnitude: should be around 12-20
    
    #calcing asym err bars
    flux_l = df.forcediffimflux - df.forcediffimfluxunc
    flux_u = df.forcediffimflux + df.forcediffimfluxunc
    mel = -2.5*np.log10(flux_l) + df.zpdiff #lower bound err
    meu = -2.5*np.log10(flux_u) + df.zpdiff #upper bound err
    
    #appending to df
    df['mag'] = m
    df['mag_err_lower'] = mel
    df['mag_err_upper'] = meu
    return df


def read_in_data(file_path, data_source=''):
    '''
    takes in forced_photometry file and reads in data as pd.DataFrame
    data_source can be: 'ztf_fp', 'alerce', 'atlas', 'asassn'
        converts flux to mag, adds jd/mjd columns, and formats df depending on data_source
    input: file_path
    output: pd.DataFrame
    '''
    if type(file_path)!=str:
        print('File Path must be str')
        return
    
    if data_source == 'ztf_fp':
        cols = ['index', 'field', 'ccdid', 'qid', 'filter', 'pid', 'infobitssci', 'sciinpseeing', 'scibckgnd', 'scisigpix', 'zpmaginpsci', 'zpmaginpsciunc', 'zpmaginpscirms', 'clrcoeff', 'clrcoeffunc', 'ncalmatches', 'exptime', 'adpctdif1', 'adpctdif2', 'diffmaglim', 'zpdiff', 'programid', 'jd', 'rfid', 'forcediffimflux', 'forcediffimfluxunc', 'forcediffimsnr', 'forcediffimchisq', 'forcediffimfluxap', 'forcediffimfluxuncap', 'forcediffimsnrap', 'aperturecorr', 'dnearestrefsrc', 'nearestrefmag', 'nearestrefmagunc', 'nearestrefchi', 'nearestrefsharp', 'refjdstart', 'refjdend', 'procstatus']
        df = pd.read_csv(file_path, names=cols, header=None, sep=" ", skiprows=57)
        df = format_df(df)
        df = convert2mag(df)
    elif data_source == 'alerce':
        df = pd.read_csv(file_path)
        df = mjd_2_jd(df,source='alerce')
    elif data_source == 'asassn':
        df = pd.read_csv(file_path, header='infer')
        df['mag'] = df['mag'].loc[df['mag'].str[0]=='>'].str[1:].astype('float') #asassn csv files have > before upper limit magnitude detections, so removing
    elif data_source == 'atlas':
        cols_atlas = ['MJD', 'm', 'dm', 'uJy', 'duJy', 'F', 'err', 'chi/N', 'RA', 'Dec', 'x', 'y', 'maj', 'min', 'phi', 'apfit', 'mag5sig', 'Sky', 'Obs']
        df = pd.read_csv(file_path, skiprows=1, names=cols_atlas, delim_whitespace=True)
        df = mjd_2_jd(df)
    else:
        print("Data source not supported")
        return
    
    return df
    

def export_combined_df(fp_df, atlas_df, SN_ID='temp', verbose=False, custom_path='.'):
    times = np.concatenate((np.array(fp_df['jd']),np.array(atlas_df['JD'])),axis=None)
    filters = np.concatenate((np.array(fp_df['filter']),np.array(atlas_df['F'])),axis=None)
    mags = np.concatenate((np.array(fp_df['mag']),np.array(atlas_df['m'])),axis=None)
    origins = np.concatenate((np.array(['ztf']*len(fp_df['jd'])),np.array(['atlas']*len(atlas_df['JD']))),axis=None)

    ztf_atlas_df = pd.DataFrame(columns=['JD', 'filter', 'mag', 'ztf_mag_lower', 'ztf_mag_upper', 'atlas_mag_err', 'data_origin'])

    ztf_atlas_df['JD'],ztf_atlas_df['filter'],ztf_atlas_df['mag'],ztf_atlas_df['data_origin'] = times,filters,mags,origins
    ztf_atlas_df['ztf_mag_lower'][ztf_atlas_df['data_origin']=='ztf'] = np.array(fp_df['mag_err_lower'])
    ztf_atlas_df['ztf_mag_upper'][ztf_atlas_df['data_origin']=='ztf'] = np.array(fp_df['mag_err_upper'])
    ztf_atlas_df['atlas_mag_err'][ztf_atlas_df['data_origin']=='atlas'] = np.array(atlas_df['dm'])

    if verbose:
        print(ztf_atlas_df)

    if len(custom_path)<2:
        ztf_atlas_df.to_csv('./data/forced_phot_data/'+SN_ID+'_ztf_atlas_df.csv',index_label='index')
    else:
        ztf_atlas_df.to_csv(custom_path,index_label='index')
    return 


def fp_alert_merge(fp_df, alert_df):
    '''
    create merged dataframe of ztf_fp and alerce dataframes
        merges on shared jd values
    output: pd.Dataframe
    '''
    merge_df = fp_df.merge(alert_df, how='inner', on='jd')
    return merge_df

def compare_mags(merge_df):
    '''
    subtracts alert/broker mag from forced photometry mag
    creates new column in input df with differences
    input: pd.DataFrame
    output: None
    '''
    # mag in fp_df and magpsf in alert df
    mag_diff = merge_df['mag'] - merge_df['magpsf']
    merge_df['mag_diff_fp-alert'] = mag_diff
    return

def normalize_time(merge_df):
    '''
    normalized jd time of df by subtracting the time of first row
    output: pd.Series 
    '''
    t = merge_df['jd']
    t_norm = t - t[0]
    return t_norm




##########################################################################
#                           PLOTTING ROUTINES
##########################################################################




def shape_err(df,filter='f',lower_err_col=None,upper_err_col=None):
    '''
    deals with uneven error bars on ztf forced photometry 
    output: np.array
    '''
    if (lower_err_col is None) and (upper_err_col is None):
        l = (df[df['filter']==filter].mag_err_lower - df[df['filter']==filter].mag).to_numpy()
        u = (df[df['filter']==filter].mag - df[df['filter']==filter].mag_err_upper).to_numpy()
    else:
        l = (df[df['filter']==filter][lower_err_col] - df[df['filter']==filter].mag).to_numpy()
        u = (df[df['filter']==filter].mag - df[df['filter']==filter][upper_err_col]).to_numpy()
    l = l.reshape(1,len(l))
    u = l.reshape(1,len(u))

    err_arr = np.concatenate((l,u))
    return err_arr

def plot_LC(df, col1, err, ID='SN ID',show=True, xlimit=True, xlims=[1,2], ylims=[0,0], save=False, flux=True, new_fig=True):
    """
    creates individual plot of all ztf data for single SN, can plot in flux or mag
        (err can string of column for flux plotting or (2,N) array for mag plotting)
        (does NOT automatically save image)
    """
    if new_fig:
        plt.figure(figsize=(9,5))
    
    if flux:
        #plot g band w/ err
        plt.errorbar(df[df['filter']=='ZTF_g'].jd, df[df['filter']=='ZTF_g'][col1], df[df['filter']=='ZTF_g'][err], 
                    fmt='o',color='green', label='ZTF_g', alpha=0.5)
        
        #plot r band w/ err
        plt.errorbar(df[df['filter']=='ZTF_r'].jd, df[df['filter']=='ZTF_r'][col1], df[df['filter']=='ZTF_r'][err], 
                    fmt='o',color='red', label='ZTF_r', alpha=0.5)

        #plot i band w/ err
        plt.errorbar(df[df['filter']=='ZTF_i'].jd, df[df['filter']=='ZTF_i'][col1], df[df['filter']=='ZTF_i'][err], 
                    fmt='o',color='goldenrod', label='ZTF_i', alpha=0.5)
    else:
        #plot g band w/ err
        err_g = shape_err(df, filter='ZTF_g')
        plt.errorbar(df[df['filter']=='ZTF_g'].jd, df[df['filter']=='ZTF_g'][col1], err_g, 
                    fmt='o',color='green', label='ZTF_g', alpha=0.5)
        
        #plot r band w/ err
        err_r = shape_err(df, filter='ZTF_r')
        plt.errorbar(df[df['filter']=='ZTF_r'].jd, df[df['filter']=='ZTF_r'][col1], err_r, 
                    fmt='o',color='red', label='ZTF_r', alpha=0.5)

        #plot i band w/ err
        err_i = shape_err(df, filter='ZTF_i')
        plt.errorbar(df[df['filter']=='ZTF_i'].jd, df[df['filter']=='ZTF_i'][col1], err_i, 
                    fmt='o',color='goldenrod', label='ZTF_i', alpha=0.5)

    if xlimit:
        plt.xlim(xlims[0], xlims[1])
    if ylims != [0,0]:
        plt.ylim(ylims[0], ylims[1])
    if flux:
        plt.ylim(-500,)
    plt.title(ID)
    plt.xlabel('JD')
    plt.legend()
    # plt.gca().invert_yaxis()
    if save and flux:
        plt.ylabel('Flux')
        plt.savefig('./figures/fp/'+ID+'_fp_LC_flux.png')
    elif save and not flux:
        plt.ylabel('Magnitude')
        plt.savefig('./figures/fp/'+ID+'_fp_LC_mag.png')
    if show:
        plt.show()
    return


def plot_fp_mag_LC(fp_df, df, df_non, ID='SN ID', show=True, alert=True, xlims=[0,0], ylims=[0,0], save=False, new_fig=True):
    """
    PLOTTING FP AND ALERT MAG DATA TOGETHER
    """
    if new_fig:
        plt.figure(figsize=(7,5))
    plt.rcParams.update({'font.size': 12})

    # FP DATA
    #plot g band w/ err
    err_g = shape_err(fp_df, filter='ZTF_g')
    plt.errorbar(fp_df[fp_df['filter']=='ZTF_g'].jd-2400000.5, fp_df[fp_df['filter']=='ZTF_g']['mag'], err_g, 
                fmt='s',markerfacecolor='none', markeredgecolor='green', ecolor='green', label='FP_g', alpha=0.3)
    
    #plot r band w/ err
    err_r = shape_err(fp_df, filter='ZTF_r')
    plt.errorbar(fp_df[fp_df['filter']=='ZTF_r'].jd-2400000.5, fp_df[fp_df['filter']=='ZTF_r']['mag'], err_r, 
                fmt='s',markerfacecolor='none', markeredgecolor='red', ecolor='red', label='FP_r', alpha=0.3)

    #plot i band w/ err
    err_i = shape_err(fp_df, filter='ZTF_i')
    plt.errorbar(fp_df[fp_df['filter']=='ZTF_i'].jd-2400000.5, fp_df[fp_df['filter']=='ZTF_i']['mag'], err_i, 
                fmt='s',markerfacecolor='none', markeredgecolor='goldenrod', ecolor='goldenrod', label='FP_i', alpha=0.3)
    
    
    # ALERT DATA
    if alert:
        #plot g band w/ err
        # plt.scatter(df[df['fid']==1].mjd,df[df['fid']==1].magpsf, color='green', label='alert_g') #detections
        plt.errorbar(df[df['fid']==1].mjd,df[df['fid']==1].magpsf,df[df['fid']==1].sigmapsf, fmt='o', 
                    color='lime', ecolor='lime', label='alert_g', alpha=0.5)
        plt.scatter(np.array(df_non[df_non['fid']==1].mjd),np.array(df_non[df_non['fid']==1].diffmaglim), 
                    color='lime', label='non_g', marker='v', alpha=0.3)
        
        #plot r band w/ err
        # plt.scatter(df[df['fid']==2].mjd,df[df['fid']==2].magpsf, color='red', label='alert_r') #detections
        plt.errorbar(df[df['fid']==2].mjd,df[df['fid']==2].magpsf,df[df['fid']==2].sigmapsf, fmt='o', 
                    color='magenta', ecolor='magenta', label='alert_r', alpha=0.5)
        plt.scatter(np.array(df_non[df_non['fid']==2].mjd),np.array(df_non[df_non['fid']==2].diffmaglim), 
                    color='magenta', label='non_r', marker='v', alpha=0.3)

    if xlims != [0,0]:
        # print('here')
        plt.xlim(xlims[0], xlims[1])
    if ylims != [0,0]:
        plt.ylim(ylims[0], ylims[1])
    plt.title(ID)
    plt.xlabel('MJD')
    plt.ylabel('Magnitude')
    plt.legend(loc='lower right')
    # plt.gca().invert_yaxis()
    if save:
        plt.savefig('./figures/fp/'+ID+'_combo_fp_LC_mag.png')
    if show:
        plt.show()
    return


def multi_band_plot(ztf_df, asassn_df, atlas_df, ID='temp', ztf=True, asassn=True, atlas=True, xlims=[0,0], ylims=[0,0], save=False):
    '''
    Creates plot of all forced photometry lightcurves overplotted for each object
        (does NOT automatically save)
    '''
    plt.figure(figsize=(8,6))
    if ztf:
        err_g = shape_err(ztf_df, filter='ZTF_g')
        err_r = shape_err(ztf_df, filter='ZTF_r')
        err_i = shape_err(ztf_df, filter='ZTF_i')
        
        plt.errorbar(ztf_df[ztf_df['filter']=='ZTF_g']['jd'], ztf_df[ztf_df['filter']=='ZTF_g']['mag'], err_g,
                     color='green', alpha=0.3, label='ZTF_g', linestyle='', marker='o')
        plt.errorbar(ztf_df[ztf_df['filter']=='ZTF_r']['jd'], ztf_df[ztf_df['filter']=='ZTF_r']['mag'], err_r,
                    color='red', alpha=0.3, label='ZTF_r', linestyle='', marker='o')
        plt.errorbar(ztf_df[ztf_df['filter']=='ZTF_i']['jd'], ztf_df[ztf_df['filter']=='ZTF_i']['mag'], err_i,
                    color='goldenrod', alpha=0.3, label='ZTF_i', linestyle='', marker='o')

    if asassn: #if want only good data then add in: [asassn_df['mag_err']!=99.99]
        if (asassn_df['Filter'].unique())[0] or (asassn_df['Filter'].unique())[1] == 'V':
            plt.errorbar(asassn_df[asassn_df['Filter']=='V']['HJD'], 
                        asassn_df[asassn_df['Filter']=='V']['mag'], 
                        asassn_df[asassn_df['Filter']=='V']['mag_err'], 
                        linestyle='', alpha=0.1, marker='d', color='darkkhaki', label='ASASSN V',zorder=50)
        if (asassn_df['Filter'].unique())[0] or (asassn_df['Filter'].unique())[1] == 'g':
            plt.errorbar(asassn_df[asassn_df['Filter']=='g']['HJD'], 
                        asassn_df[asassn_df['Filter']=='g']['mag'], 
                        asassn_df[asassn_df['Filter']=='g']['mag_err'], 
                        linestyle='', alpha=0.1, marker='d', color='limegreen',label='ASASSN g',zorder=50)
    if atlas:
        plt.errorbar(atlas_df[atlas_df['F']=='o']['MJD']+2400000.5, atlas_df[atlas_df['F']=='o']['m'], atlas_df[atlas_df['F']=='o']['dm'],
                    marker='s', color='orange', alpha=0.3, label='ATLAS o', linestyle='')
        plt.errorbar(atlas_df[atlas_df['F']=='c']['MJD']+2400000.5, atlas_df[atlas_df['F']=='c']['m'], atlas_df[atlas_df['F']=='c']['dm'],
                    marker='s', color='cyan', alpha=0.3, label='ATLAS c', linestyle='')
    
    # plt.gca().invert_yaxis()
    if xlims != [0,0]:
        plt.xlim(xlims[0], xlims[1])
    if ylims != [0,0]:
        plt.ylim(ylims[0], ylims[1])
    plt.ylabel('Apparent Magnitude')
    plt.xlabel('JD')
    plt.legend(loc='lower right')
    plt.title(ID+'-multiband')
    if save:
        plt.savefig('./figures/fp/'+ID+'_multiband.png')


def plot_residuals(merge_df, sn_name, save=False):
    '''
    plotting residual differences between ztf forced photometry and alert data, per sne
    '''
    fig, axs = plt.subplots(2, 1, figsize=(8,6), sharex=True, gridspec_kw={'height_ratios': [2,1]})
    fig.subplots_adjust(hspace=0)

    #light curves
    axs[0].scatter(merge_df[merge_df['filter']=='ZTF_g']['jd'], merge_df[merge_df['filter']=='ZTF_g']['mag'], color='lime', label='FP: g')    # fp data
    axs[0].scatter(merge_df[merge_df['filter']=='ZTF_r']['jd'], merge_df[merge_df['filter']=='ZTF_r']['mag'], color='fuchsia', label='FP: R') # fp data

    axs[0].scatter(merge_df[merge_df['filter']=='ZTF_g']['jd'], merge_df[merge_df['filter']=='ZTF_g']['magpsf'], 
                edgecolor='green', facecolor='none', marker='s', label='Alert: g')   # alert data
    axs[0].scatter(merge_df[merge_df['filter']=='ZTF_r']['jd'], merge_df[merge_df['filter']=='ZTF_r']['magpsf'], 
                edgecolor='firebrick', facecolor='none', marker='s', label='Alert: R')   # alert data

    axs[0].set_ylabel('Magnitude')
    axs[0].invert_yaxis()

    #residuals
    axs[1].scatter(merge_df[merge_df['filter']=='ZTF_g']['jd'], merge_df[merge_df['filter']=='ZTF_g']['mag_diff_fp-alert'], 
                edgecolor='lime', facecolor='none', marker='d', label='g')
    axs[1].scatter(merge_df[merge_df['filter']=='ZTF_r']['jd'], merge_df[merge_df['filter']=='ZTF_r']['mag_diff_fp-alert'], 
                edgecolor='fuchsia', facecolor='none', marker='d', label='R')
    axs[1].axhline(0, color='lightgray', zorder=0, linestyle='--')
    axs[1].set_xlabel('JD')
    axs[1].set_ylabel('Residuals')
    axs[1].yaxis.set_minor_locator(MultipleLocator(5))

    axs[0].set_title('(FP-Alert) for '+sn_name)
    # handles, labels = axs[0].get_legend_handles_labels()
    # axs[0].legend(handles, labels[0:15], loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    plt.tight_layout()
    if save:
        plt.savefig('./figures/fp/individ_residuals_'+sn_name+'.png')
    plt.show()