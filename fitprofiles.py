import numpy as npy
import warnings
import cheasepy
import fusionfiles
import fusionprofit

warnings.filterwarnings("ignore")

eqdskpath     = "/home/ehab/MyFiles/CoDev/Python/Fusion/Discharges/DIIID162940/g162940.02944_670"
profilespath  = "/home/ehab/MyFiles/CoDev/Python/Fusion/Discharges/DIIID162940/p162940.02944_670"

eqdskdata     = fusionfiles.read_eqdsk(eqdskfpath=eqdskpath)
profilesdata  = fusionfiles.read_profiles(profilesfpath=profilespath,setParam={'nrhomesh':1},eqdsk=eqdskpath)

setparam = {}
setparam['Normalize'] = False
setparam['fit_plot']  = True

fitparam = {}
fitparam['alpha']      = None
fitparam['ped_sol']    = None
fitparam['ped_mid']    = None
fitparam['ped_width']  = None
fitparam['ped_height'] = None
fitparam['cor_exp']    = None
fitparam['cor_height'] = None
fitparam['cor_height'] = None

fitbounds = {}
fitbounds['alpha']      = None
fitbounds['ped_sol']    = None
fitbounds['ped_mid']    = None
fitbounds['ped_width']  = None
fitbounds['ped_height'] = None
fitbounds['cor_exp']    = None
fitbounds['cor_height'] = None
fitbounds['cor_height'] = None

# Fitting the Pressure Profile:
pres_fit_param = fusionprofit.fit_profile(profilesdata['rhotor'],profilesdata['pressure'],method='groebner',setParam=setparam,fitParam=fitparam,fitBounds=fitbounds)
print("Pressure Profile Fitting Parameters:")
for key in pres_fit_param:
    print(key,' = ',pres_fit_param[key])

setparam['Normalize'] = True
fitparam['ped_mid']   = pres_fit_param['ped_mid']

# Fitting the Electron Density Profile:
ne_fit_param = fusionprofit.fit_profile(profilesdata['rhotor'],profilesdata['ne'],method='groebner',setParam=setparam,fitParam=fitparam,fitBounds=fitbounds)
print("Electron Density Profile Fitting Parameters:")
for key in ne_fit_param:
    print(key,' = ',ne_fit_param[key])

# Fitting the Electron Temperature Profile:
te_fit_param = fusionprofit.fit_profile(profilesdata['rhotor'],profilesdata['Te'],method='stefanikova',setParam=setparam,fitParam=fitparam,fitBounds=fitbounds)
print("Electron Temperature Profile Fitting Parameters:")
for key in te_fit_param:
    print(key,' = ',te_fit_param[key])

# Fitting the Ion Density Profile:
ni_fit_param = fusionprofit.fit_profile(profilesdata['rhotor'],profilesdata['ni'],method='groebner',setParam=setparam,fitParam=fitparam,fitBounds=fitbounds)
print("Ion Density Profile Fitting Parameters:")
for key in ni_fit_param:
    print(key,' = ',ni_fit_param[key])

# Fitting the Ion Temperature Profile:
ti_fit_param = fusionprofit.fit_profile(profilesdata['rhotor'],profilesdata['Ti'],method='groebner',setParam=setparam,fitParam=fitparam,fitBounds=fitbounds)
print("Ion Temperature Profile Fitting Parameters:")
for key in ti_fit_param:
    print(key,' = ',ti_fit_param[key])

