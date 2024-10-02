# Lightning Model Explainer

>This model was fit to an initial sample of 10 ZTF/ATLAS forced photometry light curves of doubly peaked SNe IIb light curves. The goal was to ascertain population statistics of these objects and use the best-fit parameters as individual features in a filter for use in alert stream to catch these objects early on (i.e. before re-rising to nickel-peak). 

Note: we do not intend this model to be fit to live-stream alert stream light curves.

## Parameter Definitions

```m1``` = slope of the rise to the SCE peak

```m2``` =  slope of the fall from the SCE peak

```m3``` = slope of the rise to the nickel-peak

```b2``` = vertical/magnitude shift of the whole model (i.e. "y-intercept")

```a1``` = time of the peak of the SCE peak [days]

```a2``` = time of the trough between SCE and nickel peak [days]

```log_f``` = estimation of the errors (MCMC parameter, not physical parameter)

## Parametric model

Note: this model is fit to g and r band light curves separately 
```
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
```
Note: ```r1``` (boolean) indicates whether there is data on the rise to the SCE peak (```True``` means there is)

## Prior Bounds

```
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
```

Note: ```a2_lower``` and ```a2_upper``` are roughly defined as 15-3 days and 15+3 days since we by-eye shifted all the light curves to align such that t(a2)=15 days

Note: ```a1_lower``` is generally set as the first observation and ```a1_upper = a2_lower```

## Population Statistics / Results

Quoted population statistics are the mean of the 10 individual object best-fit values in a given band. (The best-fit value for a single object is the median of the MCMC walkers/chains)

**g-band:** 
| Parameter | Mean Best-fit | Standard Deviation |
| :---: | :---: | :---: |
| m1 | -1.154 | 0.3296 |
| m2 | 0.2094 | 0.06119 |
| m3 | -0.1000 | 0.05152 |
| b2 | -2.756 | 0.9544 |
| a1 | 7.684 | 2.323 |
| a2 | 13.62 | 1.373 |
|log_f | -1.648 | 1.443 |


================================================

**r-band:**
| Parameter | Mean Best-fit | Standard Deviation |
| :---: | :---: | :---: |
| m1 | -1.003 | 0.3890 |
| m2 | 0.1794 | 0.08843 |
| m3 | -0.08751 | 0.03841 |
| b2 | -2.590 | 1.166 |
| a1 | 8.005 | 1.779 |
| a2 | 13.66 | 1.448 |
|log_f | -2.458 | 0.8952 |
