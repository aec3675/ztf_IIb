import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
import corner

SAVE_DIR = './mcmc_fp_results/'

def convert_ztf_err(df):
    l = (df.ztf_mag_lower - df.mag)
    u = (df.mag - df.ztf_mag_upper)
    df['ztf_mag_lower'] = l
    df['ztf_mag_upper'] = u
    return

def slice_df(df,jd_lims=[0,0]):
    low = jd_lims[0]
    high = jd_lims[1]
    return df[(low<=df['JD']) & (df['JD']<=high) & (df['mag']>0)].reset_index(drop=True)

def find_a2(df_cut):
    plt.figure(figsize=(10,8))
    plt.scatter(df_cut['JD'], df_cut['mag'],s=10)
    # for i in range(len(df_cut['JD'])):
    #     plt.text(np.array(df_cut['JD'])[i], np.array(df_cut['mag'])[i], str(i),fontsize=10)
    # plt.gca().invert_yaxis()
    plt.ylim(23,17)

def norm_LC_a2(df_cut, a2=[0]):
    """
    input: 
        df = SN df from read in
        a2 = index of array that trough is at from find_a2()
    output:
        norm_lc = 2D array of time and mag
    """
    a2_t = df_cut['JD'].loc[a2] #time
    a2_m = df_cut['mag'].loc[a2] #mag

    norm_t = ((df_cut['JD']) - a2_t)+15          #normalized time in g band
    norm_m = (df_cut['mag']) - a2_m              #normalized mag in g band
    df_cut['norm_t'] = norm_t
    df_cut['norm_m'] = norm_m
    return 

# FIXED DAY SUBSELECTION FROM LCs
def subselect(df_cut, t_upper=25):
    """
    input: 
        df_cut = sne dataframe cut down by eye
        norm_sne = [1,2,N] list of arrays of [time, mag] normalized 
        a2 = index of torugh in np.array(df_cut)
    output:
        [sub_t, sub_m, sub_e_zl, sub_e_zu, sub_e_a] = [1,5,N] list of time, mag, magerr1,magerr2,magerr3 subselcted around t_upper+1dp
    """    
    # creating sub df
    crit = (df_cut['norm_t']<=t_upper)
    df_sub = df_cut[crit].reset_index(drop=True)
    a2_ind = np.where(df_sub['norm_t']==15)[0][0] #15 is hardcoded value that we normalized the tough to in norm_LC_a2()

    # #if last dp before t=20 days is also a2, add the next dp in the LC to the sub_LC
    # if sub_t[-1] == a2_t:
    #     sub_t = np.append(sub_t, norm_t[a2+1])
    #     sub_m = np.append(sub_m, norm_m[a2+1])
    #     sub_e_zl = np.append(sub_e_zl, norm_e_zl[a2+1])
    #     sub_e_zu = np.append(sub_e_zu, norm_e_zu[a2+1])
    #     sub_e_a  = np.append(sub_e_a, norm_e_a[a2+1])

    # #add a datapoint after the last datapoint in sub_g_LC
    # p1 = len(sub_t) 
    # if p1 != len(norm_t):
    #     sub_t = np.append(sub_t, norm_t[p1])
    #     sub_m = np.append(sub_m, norm_m[p1])
    #     sub_e_zl = np.append(sub_e_zl, norm_e_zl[p1])
    #     sub_e_zu = np.append(sub_e_zu, norm_e_zl[p1])
    #     sub_e_a  = np.append(sub_e_a , norm_e_a[p1])

    return df_sub,a2_ind

def find_nearest_points(series,t):
    lessthan = series[series<t]
    greater = series[series>t]
    lowerbound = np.array(series[series==15-min(np.abs(15-lessthan))])[0]
    upperbound = np.array(series[series==15+min(np.abs(greater-15))])[0]
    return [lowerbound, upperbound]

def find_within_trange(series,t_norm, t_pm):
    lb = t_norm-t_pm
    ub = t_norm+t_pm
    t_around = series[(lb<=series)&(series<=ub)] #fixed day selection +/-3 around t=15
    bounds = find_nearest_points(series,t_norm) #for when fixed day selection wont work, need to do nearest datapoint

    # print(t_around)
    if len(t_around)>=3:
        # print('more than 3 pnts')
        lowerbound,upperbound = min(t_around),max(t_around)
        if upperbound<=15:
            upperbound = bounds[1]
        if lowerbound>=15.0:
            lowerbound = bounds[0]
    elif len(t_around)<3:
        # print('less than 3')
        bounds = find_nearest_points(series,t_norm)
        lowerbound,upperbound = bounds[0],bounds[1]
    else:
        print('smthn funky w t')
    if 15-lowerbound<1.5:
        lowerbound = 15-1.5
    if upperbound-15<1.5:
        upperbound=15+1.5
    print(lowerbound,upperbound)
    return[lowerbound,upperbound]

# defining func to create list of a1/a2 prior-space for each sne

def calc_prior(df_sub, r1_g=False, r1_r=True):
    """
    INPUT: 
        df_sub = df of subselected rows
        as = index of trough of np.array(df_sub)
    OUTPUT:
        prior_dict: 2D dict ['g', 'r'] of lower/upper bounds on all priors
    """
    sorted_df = df_sub.sort_values(by=['norm_t'],ignore_index=True)
    
    # "G" BAND
    g_df = sorted_df[(sorted_df['filter']=='ZTF_g') | (sorted_df['filter']=='c')]    
    sub_sn_t_g = np.array(g_df['norm_t']) #time g band
    
    # a2 bounds
    # l_g,u_g = find_nearest_points(g_df['norm_t'],t=15) 
    l_g,u_g = find_within_trange(g_df['norm_t'], t_norm=15, t_pm=3)               
    if len(sub_sn_t_g)>1:# and a2_g!=0:
        a2_g_lower = l_g
        a2_g_upper = u_g
    else:
        print('theres an oopsie in g')
        
    #a1 bounds
    if r1_g:
        a1_g_lower = sub_sn_t_g[0]
        a1_g_upper = a2_g_lower
    else:
        a1_g_lower = sub_sn_t_g[0]-0.5
        a1_g_upper = sub_sn_t_g[0]


    # "R" BAND
    r_df = sorted_df[(sorted_df['filter']=='ZTF_r') | (sorted_df['filter']=='o')]    
    sub_sn_t_r = np.array(r_df['norm_t']) #time r band

    # a2 bounds
    # l_r,u_r = find_nearest_points(r_df['norm_t'],t=15)
    l_r,u_r = find_within_trange(r_df['norm_t'], t_norm=15, t_pm=3)               
    if len(sub_sn_t_r)>1:#and a2!=0: #2nd clause makes sure there is another datapoint after a2
        a2_r_lower = l_r
        a2_r_upper = u_r
    else:
        print('theres an oopsie in r')
        
    #a1 bounds
    if r1_r:
        a1_r_lower = sub_sn_t_r[0]
        a1_r_upper = a2_r_lower
    else:
        a1_r_lower = sub_sn_t_r[0]-0.5
        a1_r_upper = sub_sn_t_r[0]
    

    # OUTPUTS
    prior_dict = {
        "g": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.1,"m2_u":0.4,
            "m3_l":-0.25,"m3_u":0.0,
            "b2_l":-5.5,"b2_u":0.0,
            "a1_l":a1_g_lower, "a1_u": a1_g_upper,
            "a2_l":a2_g_lower, "a2_u":a2_g_upper,
            "log_f_l":-4.0,"log_f_u":4.0},
        
        "r": {"m1_l":-2.0,"m1_u":0.0,
            "m2_l":0.0,"m2_u":0.4,
            "m3_l":-0.2,"m3_u":0.0,
            "b2_l":-5.0,"b2_u":0.0,
            "a1_l":a1_r_lower, "a1_u":a1_r_upper,
            "a2_l":a2_r_lower, "a2_u":a2_r_upper,
            "log_f_l":-4.0,"log_f_u":4.0}
    }
    
    #output
    # print(prior_dict)
    return prior_dict

def p0_calc(prior_dict):
    #g band
    m1g = np.average([prior_dict['g']['m1_l'],prior_dict['g']['m1_u']])
    m2g = np.average([prior_dict['g']['m2_l'],prior_dict['g']['m2_u']])
    m3g = np.average([prior_dict['g']['m3_l'],prior_dict['g']['m3_u']])
    b2g = np.average([prior_dict['g']['b2_l'],prior_dict['g']['b2_u']])
    a1g = np.average([prior_dict['g']['a1_l'],prior_dict['g']['a1_u']])
    a2g = np.average([prior_dict['g']['a2_l'],prior_dict['g']['a2_u']])
    log_fg = -2
    p0_g = [m1g,m2g,m3g,b2g,a1g,a2g,log_fg]

    #r band
    m1r = np.average([prior_dict['r']['m1_l'],prior_dict['r']['m1_u']])
    m2r = np.average([prior_dict['r']['m2_l'],prior_dict['r']['m2_u']])
    m3r = np.average([prior_dict['r']['m3_l'],prior_dict['r']['m3_u']])
    b2r = np.average([prior_dict['r']['b2_l'],prior_dict['r']['b2_u']])
    a1r = np.average([prior_dict['r']['a1_l'],prior_dict['r']['a1_u']])
    a2r = np.average([prior_dict['r']['a2_l'],prior_dict['r']['a2_u']])
    log_fr = -2
    p0_r = [m1r,m2r,m3r,b2r,a1r,a2r,log_fr]

    p0 = [p0_g,p0_r]
    print(p0)
    return p0

def merge_err_cols(df_sub):
    df_sub['max_ztf_err'] = df_sub[["ztf_mag_lower", "ztf_mag_upper"]].max(axis=1)
    df_sub['mag_err_all'] = df_sub.max_ztf_err.fillna(df_sub.atlas_mag_err)

#defining lines/liklihoods to be called in MCMC fitting
def model(theta, x, r1=True):
    m_eq = float()
    if r1:
        m1, m2, m3, b2, a1, a2, log_f = theta
        if x<= a1:
            m_eq = m1*x + (a1*(m2-m1)+b2)    
        elif a1< x <= a2:
            m_eq = m2*x + b2
        elif a2 < x:
            m_eq = (m3*x) + (a2*(m2-m3)+b2)
        else:
            print("smthn wrong 1") 
    else:
        m2, m3, b2, a2, log_f = theta
        if x <= a2:
            m_eq = m2*x + b2
        elif a2 < x:
            m_eq = (m3*x) + (a2*(m2-m3)+b2)
        else:
            print("smthn wrong 2") 
    return m_eq
    

def log_likelihood(theta, x, y, yerr, r1=True):
    if r1:
        m1, m2, m3, b2, a1, a2, log_f = theta
        model_all = np.array([model(theta, v) for v in x])
        sigma2 = yerr**2 + model_all**2 * np.exp(2 * log_f)
    else:
        m2, m3, b2, a2, log_f = theta
        model_all = np.array([model(theta, v, False) for v in x])
        sigma2 = yerr**2 + model_all**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model_all) ** 2 / sigma2 + np.log(sigma2))

def log_prior_combo(theta, priors, r1=True): 
    if r1:
        m1, m2, m3, b2, a1, a2, log_f = theta
        if (priors['m1_l']<m1<=priors['m1_u'] and priors['m2_l']<=m2<priors['m2_u'] and priors['m3_l']<m3<=priors['m3_u'] and 
            priors['b2_l']<b2<=priors['b2_u'] and priors['a1_l']<=a1<=priors['a1_u'] and priors['a2_l']<a2<=priors['a2_u'] and 
            priors['log_f_l']<log_f<priors['log_f_u']):
            return 0.0
        else:
            return -np.inf
    else:
        m2, m3, b2, a2, log_f = theta
        if (priors['m2_l']<=m2<priors['m2_u'] and priors['m3_l']<m3<=priors['m3_u'] and priors['b2_l']<b2<=priors['b2_u'] and 
            priors['a2_l']<a2<=priors['a2_u'] and priors['log_f_l']<log_f<priors['log_f_u']):
            return 0.0
        else:
            return -np.inf   
    
def log_probability_combo(theta, x, y, yerr, priors, r1=True):
    if r1:
        lp = log_prior_combo(theta, priors, r1=True)
    else:
        lp = log_prior_combo(theta, priors, r1=False)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, r1)

# input: x, y, yerr, [m0,b0,logf0], rise/fall, 
# output: [[m,ml,mu],[b,bl,bu],[logf,logfl,logfu]], best_fit_model[N,2]

def mcmc_fit(x, y, yerr, priors, p0=[-2,2,-2,0,0,15,-2], r1=True, band='g', sn_name='test'):
    np.random.seed(42)

    if r1:
        L1 = len(p0)
        pos = p0 + 1e-4 * np.random.randn(128, L1)
        nwalkers, ndim = pos.shape
    else:
        p0.remove(p0[0])
        p0.remove(p0[3]) #removes a1 but index is shifted because of previous line
        L2 = len(p0)
        pos = p0 + 1e-4 * np.random.randn(128, L2)
        nwalkers, ndim = pos.shape
    
    #selecting prior-space to feed in 
    if band=='g':
        p_dict = priors['g']
    if band=='r':
        p_dict = priors['r']

    #setting up backend saving
    filename = SAVE_DIR+"/mcmc_chains/"+sn_name+"_chains.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    #actually doing MCMC
    if r1:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_combo, args=(x, y, yerr, p_dict, True), backend=backend)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_combo, args=(x, y, yerr, p_dict, False), backend=backend)
    sampler.run_mcmc(pos, 1500000, progress=False)
    
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    mcmc_results = []
    #retrieve the 16th/50th/84th percentile for each param and the lower/upper bounds on each
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        arr = [mcmc[1], q[0], q[1]]
        mcmc_results.append(arr)
    
    return mcmc_results, sampler

def plot_mcmc_results(x, y, yerr, fit, mc, r1=True, sn_band='test_g', save=True):
    if r1:
        labels = ["m1","m2","m3","b2","a1","a2","log(f)"]
    else:
        labels = ["m2","m3","b2","a2","log(f)"]
    
    # ACTUAL FIT PLOT #########################################
    bestfit = np.array(fit).T[0] # or [f[0] for f in fit] 
    lower_bound = np.array(fit).T[0] - np.array(fit).T[1] #50th ptile - diff_16ptile
    upper_bound = np.array(fit).T[0] + np.array(fit).T[2] #50th ptile + diff_84ptile
    
    plt.figure(figsize=(7,5))
    x_plt = np.arange(1,40,0.01)
    if r1:
        bestfit_curve = np.array([model(bestfit, v, r1=True) for v in x_plt])
        lower_curve = np.array([model(lower_bound, v, r1=True) for v in x_plt])
        upper_curve = np.array([model(upper_bound, v, r1=True) for v in x_plt])

        plt.plot(x_plt, bestfit_curve, color='cornflowerblue')
        plt.fill_between(x_plt, lower_curve, upper_curve, color='cornflowerblue', alpha=0.2)
    else: 
        bestfit_curve = np.array([model(bestfit, v, r1=False) for v in x_plt])
        lower_curve = np.array([model(lower_bound, v, r1=False) for v in x_plt])
        upper_curve = np.array([model(upper_bound, v, r1=False) for v in x_plt]) 

        plt.plot(x_plt, bestfit_curve, color='cornflowerblue')
        plt.fill_between(x_plt, lower_curve, upper_curve, color='cornflowerblue', alpha=0.2)
        
    plt.errorbar(x, y, yerr, linestyle='', marker='o', color='black', label='') #ztf errors
    plt.axvline(bestfit[-3],alpha=0.3, color='black', linestyle=(5, (10, 3)), zorder=50)
    plt.axvline(bestfit[-2],alpha=0.3, color='black', linestyle=(5, (10, 3)), zorder=50)
    plt.gca().invert_yaxis()
    plt.xlim(3, )
    plt.ylim(6, -2.2)
    plt.xlabel('Time [days]')
    plt.ylabel('Magnitude')
    plt.title(sn_band)
    if save:
        plt.savefig(SAVE_DIR+'/figures/'+sn_band+'_mcmc.png')
    # plt.show()
    
    
    # CORNER PLOT #############################################
    plt.figure(figsize=(9,7))
    flat_samples = mc.get_chain(discard=100, thin=15, flat=True)
    
    fig = corner.corner(
        flat_samples, labels=labels, truths=bestfit, truth_color='cornflowerblue')
    fig.suptitle(sn_band)
    if save:
        plt.savefig(SAVE_DIR+'/figures/'+sn_band+'_mcmc_corner.png')
    # plt.show()

# AUTOCORRELATION FUNCS

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

#putting all the above functions together and plotting result --to call in for loop below
def do_gw_autocorr_and_plot(mc, sn_band):
    chain = mc.get_chain()[:, :, 0].T
    # Compute the estimators for a few different chain lengths
    N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
    gw2010 = np.empty(len(N))
    for i, n in enumerate(N):
        gw2010[i] = autocorr_gw2010(chain[:, :n])
    # print(gw2010)

    # Plot the comparisons
    plt.figure()
    plt.loglog(N, gw2010, "o-", label="G&W 2010")
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    plt.ylim(ylim)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)
    plt.savefig(SAVE_DIR+'/figures/'+sn_band+'_autocorr.png')
    # plt.show()

def mp_fit_sne(idfb):
    str_sn,df,band,r1_bools,p0,pdict_ls = idfb
    print('ON SN : ', str_sn)

    if band == 'g':
        g_df = df[(df['filter']=='ZTF_g') | (df['filter']=='c')]
        x,y,yerr = np.array(g_df['norm_t']),np.array(g_df['norm_m']),np.array(g_df['mag_err_all'])
    if band == 'r':
        r_df = df[(df['filter']=='ZTF_r') | (df['filter']=='o')]
        x,y,yerr = np.array(r_df['norm_t']),np.array(r_df['norm_m']),np.array(r_df['mag_err_all'])
        
    save_name = str_sn+'_'+band
    
    #run mcmc and save bestfit results, chains.h5
    fit, mc = mcmc_fit(x, y, yerr, pdict_ls, p0=p0, r1=r1_bools, band=band, sn_name=save_name)

    #calc and plot autocorr values
    do_gw_autocorr_and_plot(mc, save_name)

    #plot mcmc fit over ztf data and corner plots
    plot_mcmc_results(x, y, yerr, fit, mc, r1=r1_bools, sn_band=save_name, save=True)
    return fit