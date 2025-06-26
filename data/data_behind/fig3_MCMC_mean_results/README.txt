DESCRIPTION OF THE DATA

In Figure 3, the best-fit (50th percentile) models as returned from MCMC fitting for each individual supernova 
are shown in the colored lines with light gray error regions (16th-84 percentile). The model used is the lightning 
bolt model, described in equations 1 and 2. Overplotted in thicker black and vertical gray lines is the mean best-fit
model of the sample. 
 
Lighting Bolt Model Parameters:
m1 = slope of the first rise (to SCE peak)
m2 = slope of the first decline (from SCE peak)
m3 = slope of the second rise (to nickel peak)
b2 = magnitude-axis offset of the model
a1 = time of the SCE peak
a2 = time of the trough between the SCE and nickel peaks
log_f = aka log(f) 

The bestfit parameters for the g+c band (left panel of Fig. 3) are included in the file: individ_mcmc_bestfit_params_g.csv
The 16th percentile parameters for the g+c band (left panel of Fig. 3) are included in the file: individ_mcmc_16ptile_params_g.csv
The 84th percentile parameters for the g+c band (left panel of Fig. 3) are included in the file: individ_mcmc_84ptile_params_g.csv
Each line corresponds to an individual supernova, identified by the "SN_ID" column name. Note, that the name 'snztf18'
refers to ZTF18aalrxas. 

The bestfit parameters for the r+o band (right panel of Fig. 3) are included in the file: individ_mcmc_bestfit_params_r.csv
The 16th percentile parameters for the r+o band (right panel of Fig. 3) are included in the file: individ_mcmc_16ptile_params_r.csv
The 84th percentile parameters for the r+o band (right panel of Fig. 3) are included in the file: individ_mcmc_84ptile_params_r.csv
Each line corresponds to an individual supernova, identified by the "SN_ID" column name. Note, that the name 'snztf18'
refers to ZTF18aalrxas.

The sample's mean best-fit model parameters, for the g+c bands, are included in the file: sample_mean_mcmc_bestfit_g.csv
The sample's mean best-fit model parameters, for the r+o bands, are included in the file: sample_mean_mcmc_bestfit_r.csv
In both files, the column name 'sem' refers to the standard error on the mean. The lower/upper error bounds on the mean
can be calculated by subtracting/adding the sem from/to the mean. 

The start and end times, as defined in the relative time space, in unit of days, for each supernova are included in: time_ranges.csv