import rpy2.robjects as robjects
import numpy as np
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
import pandas as pd
from meld.meld import LMER
r = robjects.r
pandas2ri.activate()
# For a Pythonic interface to R
lme4 = importr('lme4')
rstats = importr('stats')
if hasattr(lme4,'coef'):
    r_coef = lme4.coef
else:
    r_coef = rstats.coef

def test_lmer_dyestuff1():
    """Simple test that LMER is working as expected based 
    on values from http://lme4.r-forge.r-project.org/book/Ch1.pdf"""
    dyestuff = r('Dyestuff')
    fm = LMER("Yield ~ 1 + (1|Batch)", dyestuff.to_records())
    fm.run()
    dyestuff_fe = np.array([[ 1527.5,    19.38341215,    78.80449469]])
    assert np.isclose(pandas2ri.ri2py((r_coef(r['summary'](fm._ms)))), dyestuff_fe).all
    ran_vars, ran_corrs = fm._get_re()
    assert ran_corrs is None
    dyestuff_re = pd.DataFrame({'Name': {'Batch': '(Intercept)', 'Residual': ''},
         'Var': {'Batch': 1764.0499999667695, 'Residual': 2451.2500000072018},
         'Std': {'Batch': 42.000595233481747, 'Residual': 49.510099979773841}})
    assert pd.DataFrame(ran_vars.to_dict()).equals(dyestuff_re)
    assert np.isclose(float(r['logLik'](fm._ms)[0]),-159.82713842112875)

def test_lmer_dyestuff2():
    """Simple test that LMER is working as expected based 
    on values from http://lme4.r-forge.r-project.org/book/Ch1.pdf"""
    dyestuff = r('Dyestuff2')
    fm = LMER("Yield ~ 1 + (1|Batch)", dyestuff.to_records())
    fm.run()
    dyestuff_fe = np.array([[ 5.6656    ,  0.67838803,  8.35156244]])
    assert (pandas2ri.ri2py((r_coef(r['summary'](fm._ms)))) == dyestuff_fe).all
    ran_vars, ran_corrs = fm._get_re()
    assert ran_corrs is None
    dyestuff_re = pd.DataFrame({'Name': {'Batch': '(Intercept)', 'Residual': ''},
     'Std': {'Batch': 0.0, 'Residual': 3.7156842744757266},
     'Var': {'Batch': 0.0, 'Residual': 13.806309627586206}})
    assert pd.DataFrame(ran_vars.to_dict()).equals(dyestuff_re)
    assert np.isclose(float(r['logLik'](fm._ms)[0]), -80.91413890614422)