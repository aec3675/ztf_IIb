DESCRIPTION OF DATA

The light curves used in Fig. 4 in Crawford+2025 (submitted to ApJ) are created using ATLAS and ZTF forced photometry data.
The ATLAS data are cleaned by only including any observations with ['chi/N'] < 4.0 and binned using 3-day bins.
The ZTF data are cleaned by only including observations with ['infobitssci'] < 33554432, ['scisigpix'] <= 25, and ['sciinpseeing'] <= 4 and are binned using 3-day bins.
For both ATLAS and ZTF, detections are defined as having a signal-to-noise > 3, following the recommendations in F. J. Masci+2023 (https://ui.adsabs.harvard.edu/abs/2023arXiv230516279M).
We calculate detections and upper limits following Masci+2023's definition (seen on page 12). 
We exclude all ZTF i-band data for this study. 

For the left panel of Fig. 4 in Crawford+2025, we plot both ATLAS c-band and ZTF g-bands, ordered by MJD. 
For the right panel, we plot both ATLAS o-band and ZTF r-band, ordered by MJD.
Only observations with ['err']<2.0 are included in Fig. 4.
Below we include a short description of the column names in each object's csv. 

mag = apparent magnitude
err = error on the apparent magnitude (values of 9999 indicate an upper limit)
mjd = Modified Julian Date of the binned data point
filter = [c, o, ZTF_g, ZTF_r] where c and o are from ATLAS forced photometry ; ZTF_g and ZTF_r are ZTF forced photometry
norm_t = normalized time (as defined in Crawford+2025) such that trough between peaks occurs ~15 days
abs_mag = absolute magnitude, corrected for MW dust extinction (see Table 4 in Crawford+2025 for extinction values)