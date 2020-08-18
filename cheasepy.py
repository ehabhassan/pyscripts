#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import time
import h5py
import traceback
import subprocess


import mathtools
import fusionfiles
import fusionprofit

import numpy             as npy
import statsmodels.api   as sm
import matplotlib.pyplot as plt

from glob import glob

from efittools import read_efit_file
from efittools import psi2phi,phi2psi
from efittools import read_iterdb_file
from efittools import magsurf_solvflines
from efittools import read_profiles_file

from scipy.optimize    import curve_fit
from scipy.integrate   import trapz,simps,quad
from scipy.interpolate import splrep,splev
from scipy.interpolate import interp1d,interp2d
from scipy.interpolate import CubicSpline,RectBivariateSpline

from matplotlib.backends.backend_pdf import PdfPages

mu0 = 4.0e-7*npy.pi
KBZ = 1.3806503e-23

def current_correction(chease,expeq={},setParam={}):
    if   'nsttp' in setParam:
         if   type(setParam['nsttp'])==list:
              current_type = setParam['nsttp'][0]
              current_src  = setParam['nsttp'][1]
         elif type(setParam['nsttp']) in [int,str]:
              current_type = setParam['nsttp']
              current_src  = 'chease'
    else:
              current_type = 'iprln'
              current_src  = 'chease'

    if type(current_type) == str: current_type.lower()
    if type(current_src)  == str: current_src.lower()

    if   expeq: expeqavail = True
    else:       expeqavail = False

    if current_src in [2,'expeq'] and not expeqavail:
       print('FATAL: Data from EXPEQ file is required but not provided')
       sys.exit()

    if 'ITEXP' in setParam: ITEXP = setParam['ITEXP']
    error = (chease['Itor']/ITEXP)-1.0
    
    if   current_type in [3,'iprl','iprln','iparallel','iparalleln']:
         if   current_src in [0,'chease']:
             #############
             # METHOD 01 #
             #############
              Iohmic  = (1.0-error)*(chease['Iprl']-chease['Ibs'])
             #############
             # METHOD 02 #
             #############
             #IPRLPSI = integrate(chease['CHI'],chease['IPRL']*chease['J']/chease['R'],axis=0,method='trapz')
             #IPRLT   = integrate(chease['PSI'],IPRLPSI,axis=0)
             #ISIGPSI = integrate(chease['CHI'],chease['signeo']*chease['J']/chease['R'],axis=0,method='trapz')
             #ISIGT   = integrate(chease['PSI'],ISIGPSI,axis=0)
             #Eparall = (ITEXP-IPRLT)/ISIGT
             #Iohmic  = chease['signeo']*Eparall
             #Iohmic  = (Iohmic-Iohmic[0])/(Iohmic[-1]-Iohmic[0])
             #############
             # METHOD 03 #
             #############
             #eqdskpath   = 'shots/CASE2_20190509/CASE2_20190509_EQDSK'
             #eqdskdata   = fusionfiles.read_eqdsk(eqdskfpath=eqdskpath)
             #cheasefiles = sorted(glob('chease_iter???.h5'))
             #cheasedata  = read_chease(cheasefpath=cheasefiles[-1],eqdsk=eqdskpath)
             #qratio      = cheasedata['q']/eqdskdata['q']
         elif current_src in [2,'expeq']:
              Iohmic  = (1.0-error)*((chease['B0EXP']*expeq['Iprl']*chease['R0EXP']/mu0)-chease['Ibs'])
         Iprl   = chease['Ibs'] + Iohmic
        #Iprl   = qratio*cheasedata['Iprl']
    elif current_type in [4,'jprl','jprln','jparallel','jparalleln']:
         if   current_src in [0,'chease']:
              Johmic  = (1.0-error)*(chease['Jprl']-chease['Jbs'])
             #JPRLPSI = integrate(chease['CHI'],chease['Jprl']*chease['J']/chease['R'],axis=0,method='trapz')
             #JPRLT   = integrate(chease['PSI'],JPRLPSI,axis=0)
             #JSIGPSI = integrate(chease['CHI'],chease['signeo']*chease['J']/chease['R'],axis=0,method='trapz')
             #JSIGT   = integrate(chease['PSI'],JSIGPSI,axis=0)
             #Eparall = (ITEXP-JPRLT)/JSIGT
             #Johmic  = chease['signeo']*Eparall
         elif current_src in [2,'expeq']:
              Johmic  = (1.0-error)*(expeq['Jprl']-chease['Jbs'])
              Johmic  = (1.0-error)*((chease['B0EXP']*expeq['Iprl']/chease['R0EXP']/mu0)-chease['Ibs'])
         Jprl   = chease['Jbs'] + Johmic

    if   'Iprl' in locals():
         IprlN      = mu0*Iprl/chease['R0EXP']/chease['B0EXP']
         correction = IprlN[:]
    elif 'Jprl' in locals():
         JprlN      = mu0*Jprl*chease['R0EXP']/chease['B0EXP']
         correction = JprlN[:]
    else:
         print('WARNING: return correction is not calculated!')
         correction = npy.zeros_like(chease['Iprl'])

    return correction


def pressure_correction(chease,expeq={},setParam={}):
    if   'nppfun' in setParam:
         if   type(setParam['nppfun'])==list:
              pressure_type = setParam['nppfun'][0]
              pressure_src  = setParam['nppfun'][1]
         elif type(setParam['nsttp']) in [int,str]:
              pressure_type = setParam['nppfun']
              pressure_src  = 'chease'
    else:
              pressure_type = 'pressure'
              pressure_src  = 'chease'

    if type(pressure_type) == str: pressure_type.lower()
    if type(pressure_src)  == str: pressure_src.lower()

    if   expeq: expeqavail = True
    else:       expeqavail = False

    if pressure_src in [2,'expeq'] and not expeqavail:
       print('FATAL: Data from EXPEQ file is required but not provided')
       sys.exit()

    if 'ITEXP' in setParam: ITEXP = setParam['ITEXP']
    error = (chease['Itor']/ITEXP)-1.0

    if   pressure_type in [8,'pressure']:
         if   pressure_src in [0,'chease']:
              pressure = (1.0-error)*chease['pressure']
         elif pressure_src in [2,'expeq']:
              pressure = (1.0-error)*(mu0*expeq['pressure']/chease['B0EXP']**2)
    elif pressure_type in [4,'pprime']:
         if   pressure_src in [0,'chease']:
              pprime = (1.0-error)*chease['pprime']
         elif pressure_src in [2,'expeq']:
              pprime = (1.0-error)*(mu0*expeq['pprime']*chease['R0EXP']**2/chease['B0EXP'])

    if   'pressure' in locals():
         correction = mu0*pressure/chease['B0EXP']**2
    elif 'pprime' in locals():
         correction = mu0*pprime*chease['R0EXP']**2/chease['B0EXP']
    else:
         print('WARNING: return correction is not calculated!')
         correction = npy.zeros_like(chease['pressure'])

    return correction


def findall(inlist,item):
   #Developed by Ehab Hassan on 2019-03-28
    inds = []
    for i,character in enumerate(inlist):
        if character==item: inds.append(i)
    return inds


def create_namelist(setParam={}):
    wfh = open('chease_namelist','w')
    wfh.write('*** for EQDSK file copied onto EXPEQ file \n')
    wfh.write('*** cp this file to "chease_namelist" and run chease \n')
    wfh.write('***  \n')
    wfh.write('***  \n')
    wfh.write(' &EQDATA \n')
    if   'RELAX'     not in setParam: setParam['RELAX']     = 0.0
    if   'NBSEXPQ'   not in setParam: setParam['NBSEXPQ']   = 1111
    if   'NEQDSK'    not in setParam: setParam['NEQDSK']    = 0
    if   'NS'        not in setParam: setParam['NS']        = 256
    if   'NT'        not in setParam: setParam['NT']        = 256
    if   'NISO'      not in setParam: setParam['NISO']      = 256
    if   'NPSI'      not in setParam: setParam['NPSI']      = 1024
    if   'NCHI'      not in setParam: setParam['NCHI']      = 1024
    if   'NRBOX'     not in setParam: setParam['NRBOX']     = 60
    if   'NZBOX'     not in setParam: setParam['NZBOX']     = 60
    if   'NCSCAL'    not in setParam: setParam['NCSCAL']    = 4
    if   'NOPT'      not in setParam: setParam['NOPT']      = 0
    if   'NSURF'     not in setParam: setParam['NSURF']     = 6
    if   'NFUNC'     not in setParam: setParam['NFUNC']     = 4
    if   'NRHOMESH'  not in setParam: setParam['NRHOMESH']  = 0
    if   'NFUNRHO'   not in setParam: setParam['NFUNRHO']   = setParam['NRHOMESH']
    if   'NSTTP'     not in setParam: setParam['NSTTP']     = 1
    if   'NPROPT'    not in setParam: setParam['NPROPT']    = setParam['NSTTP']  if setParam['NPPFUN']==4 else -setParam['NSTTP']
    else:                             setParam['NPROPT']    = setParam['NPROPT'] if setParam['NPPFUN']==4 else -abs(setParam['NPROPT'])
    if   'NPPFUN'    not in setParam: setParam['NPPFUN']    = 8
    if   'NVERBOSE'  not in setParam: setParam['NVERBOSE']  = 4
    if   'NDIAGOP'   not in setParam: setParam['NDIAGOP']   = 1
    if   'NIDEAL'    not in setParam: setParam['NIDEAL']    = 9
    if   'NDIFPS'    not in setParam: setParam['NDIFPS']    = 0
    if   'NDIFT'     not in setParam: setParam['NDIFT']     = 1
    if   'NMESHC'    not in setParam: setParam['NMESHC']    = 1
    if   'NPOIDC'    not in setParam: setParam['NPOIDC']    = 2
    if   'SOLPDC'    not in setParam: setParam['SOLPDC']    = 0.7
    if   'CPLACE'    not in setParam: setParam['CPLACE']    = '0.9,0.95,0.99,1.0'
    if   'CWIDTH'    not in setParam: setParam['CWIDTH']    = '0.1,0.05,0.01,0.025'
    if   'NMESHPOL'  not in setParam: setParam['NMESHPOL']  = 1
    if   'SOLPDPOL'  not in setParam: setParam['SOLPDPOL']  = 0.1
    if   'NTURN'     not in setParam: setParam['NTURN']     = 20
    if   'NBLC0'     not in setParam: setParam['NBLC0']     = 16
    if   'NPPR'      not in setParam: setParam['NPPR']      = 24
    if   'NINMAP'    not in setParam: setParam['NINMAP']    = 40
    if   'NINSCA'    not in setParam: setParam['NINSCA']    = 40
    if   'NSYM'      not in setParam: setParam['NSYM']      = 0
    if   'NEGP'      not in setParam: setParam['NEGP']      = 0
    if   'NER'       not in setParam: setParam['NER']       = 2
    if   'EPSLON'    not in setParam: setParam['EPSLON']    = 1.0E-10
    if   'ETAEI'     not in setParam: setParam['ETAEI']     = 3.0
    if   'RPEOP'     not in setParam: setParam['RPEOP']     = 0.5
    if   'RZION'     not in setParam: setParam['RZION']     = 1.0
    if   'GAMMA'     not in setParam: setParam['GAMMA']     = 1.6666666667
    if   'AT3(1)'    not in setParam: setParam['AT3(1)']    = -0.69
    if   'TENSPROF'  not in setParam: setParam['TENSPROF']  = 0.0
    if   'TENSBND'   not in setParam: setParam['TENSBND']   = 0.0
    if   'cocos_in'  not in setParam: setParam['cocos_in']  = 2
    if   'cocos_out' not in setParam: setParam['cocos_out'] = 12

    if   'RELAX'     in setParam: wfh.write(' RELAX=%2.2f, '      % (float(setParam['RELAX'])))
    if   'NBSEXPQ'   in setParam: wfh.write(' NBSEXPQ=%04d, '     % (int(setParam['NBSEXPQ'])))
    if   'NEQDSK'    in setParam: wfh.write(' NEQDSK=%1d,     \n' % (int(setParam['NEQDSK'])))
    if   'NS'        in setParam: wfh.write(' NS=%4d, '           % (int(setParam['NS'])))
    if   'NT'        in setParam: wfh.write(' NT=%4d,         \n' % (int(setParam['NT'])))
    if   'NPSI'      in setParam: wfh.write(' NPSI=%4d, '         % (int(setParam['NPSI'])))
    if   'NCHI'      in setParam: wfh.write(' NCHI=%4d, '         % (int(setParam['NCHI'])))
    if   'NISO'      in setParam: wfh.write(' NISO=%4d,       \n' % (int(setParam['NISO'])))
    if   'NRBOX'     in setParam: wfh.write(' NRBOX=%4d, '        % (int(setParam['NRBOX'])))
    if   'NZBOX'     in setParam: wfh.write(' NZBOX=%4d,      \n' % (int(setParam['NZBOX'])))
    if   'NCSCAL'    in setParam: wfh.write(' NCSCAL=%1d, '       % (float(setParam['NCSCAL'])))
    if   'NOPT'      in setParam: wfh.write(' NOPT=%1d,       \n' % (int(setParam['NOPT'])))
    if   'NSURF'     in setParam: wfh.write(' NSURF=%1d, '        % (int(setParam['NSURF'])))
    if   'NFUNC'     in setParam: wfh.write(' NFUNC=%1d,      \n' % (int(setParam['NFUNC'])))
    if   'NPPFUN'    in setParam: wfh.write(' NPPFUN=%1d,     \n' % (int(setParam['NPPFUN'])))
    if   'NFUNRHO'   in setParam: wfh.write(' NFUNRHO=%1d, '      % (int(setParam['NFUNRHO'])))
    if   'NRHOMESH'  in setParam: wfh.write(' NRHOMESH=%1d,   \n' % (int(setParam['NRHOMESH'])))
    if   'NSTTP'     in setParam: wfh.write(' NSTTP=%1d, '        % (int(setParam['NSTTP'])))
    if   'NPROPT'    in setParam: wfh.write(' NPROPT=%1d,     \n' % ( int(setParam['NPROPT'])))
    if   'NVERBOSE'  in setParam: wfh.write(' NVERBOSE=%1d,   \n' % (int(setParam['NVERBOSE'])))
    if   'QSPEC'     in setParam: wfh.write(' QSPEC=%3.3f, '      % (float(setParam['QSPEC'])))
    if   'CSSPEC'    in setParam: wfh.write(' CSSPEC=%3.3f,   \n' % (float(setParam['CSSPEC'])))
    if   'TRIANG'    in setParam: wfh.write(' TRIANG=%3.3f,   \n' % (float(setParam['TRIANG'])))
    if   'R0'        in setParam: wfh.write(' R0=%10.8f, '        % (float(setParam['R0'])))
    if   'RZ0'       in setParam: wfh.write(' RZ0=%10.8f,     \n' % (float(setParam['RZ0'])))
    if   'RBOXLEN'   in setParam: wfh.write(' RBOXLEN=%3.3f, '    % (float(setParam['RBOXLEN'])))
    if   'ZBOXLEN'   in setParam: wfh.write(' ZBOXLEN=%3.3f, '    % (float(setParam['ZBOXLEN'])))
    if   'RBOXLFT'   in setParam: wfh.write(' RBOXLFT=%3.3f,  \n' % (float(setParam['RBOXLFT'])))
    if   'R0EXP'     in setParam: wfh.write(' R0EXP=%3.3f, '      % (float(setParam['R0EXP'])))
    if   'B0EXP'     in setParam: wfh.write(' B0EXP=%3.3f,    \n' % (float(setParam['B0EXP'])))
    if   'NDIAGOP'   in setParam: wfh.write(' NDIAGOP=%1d, '      % (int(setParam['NDIAGOP'])))
    if   'NIDEAL'    in setParam: wfh.write(' NIDEAL=%1d,     \n' % (int(setParam['NIDEAL'])))
    if   'NDIFPS'    in setParam: wfh.write(' NDIFPS=%1d, '       % (int(setParam['NDIFPS'])))
    if   'NDIFT'     in setParam: wfh.write(' NDIFT=%1d,      \n' % (int(setParam['NDIFT'])))
    if   'NMESHC'    in setParam: wfh.write(' NMESHC=%1d, '       % (int(setParam['NMESHC'])))
    if   'NPOIDC'    in setParam: wfh.write(' NPOIDC=%1d, '       % (int(setParam['NPOIDC'])))
    if   'SOLPDC'    in setParam: wfh.write(' SOLPDC=%2.2f,   \n' % (float(setParam['SOLPDC'])))
    if   'CPLACE'    in setParam: wfh.write(' CPLACE=%s,      \n' % (str(setParam['CPLACE'])))
    if   'CWIDTH'    in setParam: wfh.write(' CWIDTH=%s,      \n' % (str(setParam['CWIDTH'])))
    if   'NMESHPOL'  in setParam: wfh.write(' NMESHPOL=%4d, '     % (int(setParam['NMESHPOL'])))
    if   'SOLPDPOL'  in setParam: wfh.write(' SOLPDPOL=%2.2f, \n' % (float(setParam['SOLPDPOL'])))
    if   'NTURN'     in setParam: wfh.write(' NTURN=%2d, '        % (int(setParam['NTURN'])))
    if   'NBLC0'     in setParam: wfh.write(' NBLC0=%2d, '        % (int(setParam['NBLC0'])))
    if   'NPPR'      in setParam: wfh.write(' NPPR=%2d,       \n' % (int(setParam['NPPR'])))
    if   'NINMAP'    in setParam: wfh.write(' NINMAP=%2d, '       % (int(setParam['NINMAP'])))
    if   'NINSCA'    in setParam: wfh.write(' NINSCA=%2d,     \n' % (int(setParam['NINSCA'])))
    if   'NSYM'      in setParam: wfh.write(' NSYM=%1d, '         % (int(setParam['NSYM'])))
    if   'NEGP'      in setParam: wfh.write(' NEGP=%1d, '         % (int(setParam['NEGP'])))
    if   'NER'       in setParam: wfh.write(' NER=%1d,        \n' % (int(setParam['NER'])))
    if   'EPSLON'    in setParam: wfh.write(' EPSLON=%6.2E,   \n' % (float(setParam['EPSLON'])))
    if   'ETAEI'     in setParam: wfh.write(' ETAEI=%2.1f, '      % (float(setParam['ETAEI'])))
    if   'RPEOP'     in setParam: wfh.write(' RPEOP=%2.1f, '      % (float(setParam['RPEOP'])))
    if   'RZION'     in setParam: wfh.write(' RZION=%2.1f, '      % (float(setParam['RZION'])))
    if   'GAMMA'     in setParam: wfh.write(' GAMMA=%12.11f,  \n' % (float(setParam['GAMMA'])))
    if   'AT3(1)'    in setParam: wfh.write(' AT3(1)=%2.2f,   \n' % (float(setParam['AT3(1)'])))
    if   'TENSPROF'  in setParam: wfh.write(' TENSPROF=%2.2f, \n' % (float(setParam['TENSPROF'])))
    if   'TENSBND'   in setParam: wfh.write(' TENSBND=%2.2f,  \n' % (float(setParam['TENSBND'])))
    if   'cocos_in'  in setParam: wfh.write(' cocos_in=%2d,   \n' % (int(setParam['cocos_in'])))
    if   'cocos_out' in setParam: wfh.write(' cocos_out=%2d   \n' % (int(setParam['cocos_out'])))

    wfh.write(' &END \n')
    wfh.write('\n')
    wfh.close()

    return setParam


def find_boundary(eqdsk='',setParam={}):
    if eqdsk:
       eqdskflag  = True
       if   type(eqdsk)==str and os.path.isfile(eqdsk.strip()):
                               eqdskdata = fusionfiles.read_eqdsk(eqdskfpath=eqdsk.strip())
       elif type(eqdsk)==dict: eqdskdata = eqdsk.copy()
       else:
            eqdskflag = False
    else:
       eqdskflag = False

    if not eqdskflag: raise IOError('FATAL: EQDSK FILE IS NOT PROVIDED. EXIT!')

    asisflag = False; interpflag = False
    if 'boundary_type' in setParam:
       if   setParam['boundary_type'] in [0,'asis']:   asisflag   = True
       elif setParam['boundary_type'] in [1,'interp']: interpflag = True
       else:                                           asisflag   = True

    if   asisflag:
         rbound = eqdskdata['rbound']
         zbound = eqdskdata['zbound']
    elif interpflag:
       rbndtst = int(eqdskdata['RLEN']/(max(eqdskdata['rbound'])-abs(min(eqdskdata['rbound']))))
       zbndtst = int(eqdskdata['ZLEN']/(max(eqdskdata['zbound'])+abs(min(eqdskdata['zbound']))))
       if  rbndtst==1 and zbndtst==1:
           rbound,zbound = magsurf_solvflines(eqdskdata=eqdskdata,psi=0.999,eps=1.0e-16)
       else:
           rbound=npy.zeros(2*len(eqdskdata['rbound'])-1)
           zbound=npy.zeros(2*len(eqdskdata['zbound'])-1)
           rbound[0] = eqdskdata['rbound'][0]
           zbound[0] = eqdskdata['zbound'][0]
           for i in range(1,len(eqdskdata['rbound'])):
               rbound[i]  = eqdskdata['rbound'][i]
               rbound[-i] = eqdskdata['rbound'][i]
               zbound[i]  = eqdskdata['zbound'][i]
               zbound[-i] =-eqdskdata['zbound'][i]

    return rbound,zbound


def plot_chease(OSPATH,reportpath='',skipfigs=1):
    from matplotlib.backends.backend_pdf import PdfPages
    from glob import glob

    if reportpath == '':
       report = False
    else:
       report = True

    if not os.path.isfile(OSPATH):
       srhpath    = os.path.join(OSPATH,'chease*.h5')
       h5list     = sorted(glob(srhpath))
    else:
       h5list     = [OSPATH]

    if not report:
       srhpath    = os.path.join(OSPATH,'*_EQDSK')
       eqdsklist  = sorted(glob(srhpath))

       profpath   = os.path.join(OSPATH,'*_PROFILES')
       proflist   = sorted(glob(profpath))

       srhpath    = os.path.join(OSPATH,'EXPEQ_iter*.OUT')
       expeqlist  = sorted(glob(srhpath))

       srhpath    = os.path.join(OSPATH,'EXPTNZ_iter*.OUT')
       exptnzlist = sorted(glob(srhpath))

       icounter = 1
       for h5fid in h5list[0::skipfigs+1]:
           print('Plotting CHEASE data in: %s ...' % h5fid)
           if   h5fid[13:17] == 'iter':
                caselabel  = h5fid[13:21]
           elif h5fid[8:14] in ['KEFITD','MEFITD']:
                caselabel  = h5fid[8:21]
           else:
                caselabel  = h5fid

           CHEASEdata = read_chease(cheasefpath=h5fid)
           CHEASEdataKeys = CHEASEdata.keys()

           EXPTNZdata   = fusionfiles.read_exptnz(exptnzlist[h5list.index(h5fid)],eqdsk=eqdsklist[0])
           EXPEQdata    = fusionfiles.read_expeq(expeqlist[h5list.index(h5fid)])
           EQDSKdata    = fusionfiles.read_eqdsk(eqdsklist[0])
           if proflist:
              PROFILESdata = fusionfiles.read_profiles(proflist[0])

           EDENfig = plt.figure("Electron Density")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['ne'],linestyle='-',label='CHESAE-'+caselabel[-6:-3])
           plt.plot(EXPTNZdata['rhopsi'],EXPTNZdata['ne'],linestyle=':',label='EXPTNZ-'+caselabel[-6:-3])
           plt.title('Electron Density Profiles')
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel('$n_e$')
           plt.legend()
       
           GDNEfig = plt.figure("Electron Density Gradient")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['nePrime'],label=caselabel)
           plt.title('Electron Density Gradient Profiles')
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel('$\\nabla{n_e}$')
           plt.legend()
       
           ETMPfig = plt.figure("Electron Temperature")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['Te'],linestyle='-',label='CHESAE-'+caselabel[-6:-3])
           plt.plot(EXPTNZdata['rhopsi'],EXPTNZdata['Te'],linestyle=':',label='EXPTNZ-'+caselabel[-6:-3])
           plt.title('Electron Temperature Profiles')
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel('$T_e$')
           plt.legend()

           GDTEfig = plt.figure("Electron Temperature Gradient")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['TePrime'],label=caselabel)
           plt.title('Electron Temperature Gradient Profiles')
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel('$\\nabla{T_e}$')
           plt.legend()
       
           SFTYfig = plt.figure("Safety Factor (q)")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['q'],linestyle='-',label='CHESAE-'+caselabel[-6:-3])
           if 'q' in EXPEQdata:
              plt.plot(EXPEQdata['rhopsi'],EXPEQdata['q'],linestyle='--',label='EXPEQ-'+caselabel[-6:-3])
           plt.plot(EQDSKdata['rhopsi'], EQDSKdata['q'], linestyle=':',label='EQDSK')
           plt.title("Safety Factor Profiles")
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel("q")
           plt.legend()

           TPPRfig = plt.figure("Plasma Pressure")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['pressure'],  linestyle='solid', label='CHESAE-'+caselabel[-6:-3])
           plt.plot(EQDSKdata['rhopsi'], EQDSKdata['pressure'],   linestyle='dotted', label='EQDSK')
           if 'PROFILESdata' in locals() and 'pressure' in PROFILESdata:
              plt.plot(PROFILESdata['rhopsi'],PROFILESdata['pressure'],linestyle=(0,(5,1)),label='PROFILES')
           if 'EXPTNZdata' in locals() and 'pressure' in EXPTNZdata:
              plt.plot(EXPTNZdata['rhopsi'],EXPTNZdata['pressure'],  linestyle='dashdot',label='EXPTNZ-'+caselabel[-6:-3])
           if 'EXPEQdata' in locals() and 'pressure' in EXPEQdata:
              EXPEQdata['pressure'] = EXPEQdata['pressure']*CHEASEdata['B0EXP']**2/mu0
              plt.plot(EXPEQdata['rhopsi'],EXPEQdata['pressure'],linestyle='dashed',label='EXPEQ-'+caselabel[-6:-3])
           plt.title('Plasma Pressure Profiles')
           plt.xlabel('$\\rho_{psi_N}$')
           plt.ylabel('$P$')
           plt.legend()
       
           PPRMfig = plt.figure("P'")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['pprime'], linestyle='solid', label='CHESAE-'+caselabel[-6:-3])
           plt.plot(EQDSKdata['rhopsi'], EQDSKdata['pprime'],  linestyle='dotted', label='EQDSK')
           if 'PROFILESdata' in locals() and 'pprime' in PROFILESdata:
              plt.plot(PROFILESdata['rhopsi'],PROFILESdata['pprime'],linestyle=(0,(5,1)),label='PROFILES')
           if 'EXPTNZdata' in locals() and 'pprime' in EXPTNZdata:
              plt.plot(EXPTNZdata['rhopsi'],EXPTNZdata['pprime'],linestyle=(0,(5,1)),label='EXPTNZ')
           if 'EXPEQdata' in locals() and 'pprime' in EXPEQdata:
              EXPEQdata['pprime'] = EXPEQdata['pprime']*CHEASEdata['B0EXP']/mu0/CHEASEdata['R0EXP']**2
              plt.plot(EXPEQdata['rhopsi'],EXPEQdata['pprime'],linestyle='dashed',label='EXPEQ-'+caselabel[-6:-3])
           plt.title("P' Profiles")
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel("P'")
           plt.legend()
       
           TTPMfig = plt.figure("FF'")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['ffprime'],linestyle='-',label='CHESAE-'+caselabel[-6:-3])
           plt.plot( EQDSKdata['rhopsi'], EQDSKdata['ffprime'],linestyle=':',label='EQDSK')
           if 'EXPEQdata' in locals() and 'ffprime' in EXPEQdata:
              EXPEQdata['ffprime'] = EXPEQdata['ffprime']*CHEASEdata['B0EXP']
              plt.plot(EXPEQdata['rhopsi'],EXPEQdata['ffprime'],linestyle='--',label='EXPEQ-'+caselabel[-6:-3])
           plt.title("TT' Profiles")
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel("TT'")
           plt.legend()
       
           ISTRfig = plt.figure("I*")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['Istr'],linestyle='-',label='CHESAE-'+caselabel[-6:-3])
           if 'EXPEQdata' in locals() and 'Istr' in EXPEQdata:
              EXPEQdata['Istr'] = EXPEQdata['Istr']*CHEASEdata['R0EXP']*CHEASEdata['B0EXP']/mu0
              plt.plot(EXPEQdata['rhopsi'],EXPEQdata['Istr'],linestyle='--',label='EXPEQ-'+caselabel[-6:-3])
           plt.title("I* Profiles")
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel("I*")
           plt.legend()
       
           ICRTfig = plt.figure("Parallel Current")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['Iprl'],linestyle='-',label='CHESAE-'+caselabel[-6:-3])
           if 'EXPEQdata' in locals() and 'Iprl' in EXPEQdata:
              EXPEQdata['Iprl'] = EXPEQdata['Iprl']*CHEASEdata['R0EXP']*CHEASEdata['B0EXP']/mu0
              plt.plot(EXPEQdata['rhopsi'],EXPEQdata['Iprl'],linestyle='--',label='EXPEQ-'+caselabel[-6:-3])
           plt.title('Parallel Current Profiles')
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel('$I_{||}$')
           plt.legend()
       
           JCRTfig = plt.figure("Parallel Current Density")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['Jprl'],linestyle='-',label='CHESAE-'+caselabel[-6:-3])
           if 'EXPEQdata' in locals() and 'Jprl' in EXPEQdata:
              EXPEQdata['Jprl'] = EXPEQdata['Jprl']*CHEASEdata['B0EXP']/CHEASEdata['B0EXP']/mu0
              plt.plot(EXPEQdata['rhopsi'],EXPEQdata['Jprl'],linestyle='--',label='EXPEQ-'+caselabel[-6:-3])
           plt.title('Parallel Current Density Profiles')
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel('$J_{||}$')
           plt.legend()
       
           SCRTfig = plt.figure("Bootstrap Currents")
           plt.plot(CHEASEdata['rhopsi'],CHEASEdata['Jbs'],label='$J_{BS}$-'+caselabel)
           plt.title('Bootstrap Current Density Profiles')
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel('$J_{BS}$')
           plt.legend()
       
           (CHEASEdata['PSIN2D'],CHEASEdata['CHIN2D']) = npy.meshgrid(CHEASEdata['PSIN'],CHEASEdata['CHIN'])
           BF2Dfig = plt.figure("Magnetic Field, B($\psi$,$\chi$)")
           plt.contour(CHEASEdata['CHIN2D'],CHEASEdata['PSIN2D'],CHEASEdata['B'])
           plt.title('Magnetic Field Profiles')
           plt.xlabel('$\chi$')
           plt.ylabel('$\psi$')
       
           JPHIfig = plt.figure("Toroidal Current")
           plt.contour(CHEASEdata['CHIN2D'],CHEASEdata['PSIN2D'],CHEASEdata['Jphi'],cmap=plt.cm.hot)
           plt.title('Toroidal Current Profiles')
           plt.xlabel('$\\rho_{\psi_N}$')
           plt.ylabel('$J_{\phi}$')
       
           BFRZfig = plt.figure("Magnetic Field, B(R,Z}")
           plt.contour(CHEASEdata['R'],CHEASEdata['Z'],CHEASEdata['B'])
           plt.title('Magnetic Field Profiles, B(R,Z}')
           plt.xlabel('$R$')
           plt.ylabel('$Z$')

           PSRZfig = plt.figure("Magnetic Poloidal Boundary Surface")
           plt.plot(CHEASEdata['rbound'],CHEASEdata['zbound'],label=caselabel)
           plt.title('Magnetic Poloidal Boundary Surface')
           plt.xlabel('$R$')
           plt.ylabel('$Z$')
           plt.legend()

           del(CHEASEdata)

           chsfigs = PdfPages('cheaseresults.pdf')
           chsfigs.savefig(EDENfig)
           chsfigs.savefig(GDNEfig)
           chsfigs.savefig(ETMPfig)
           chsfigs.savefig(GDTEfig)
           chsfigs.savefig(SFTYfig)
           chsfigs.savefig(TPPRfig)
           chsfigs.savefig(PPRMfig)
           chsfigs.savefig(TTPMfig)
           chsfigs.savefig(ISTRfig)
           chsfigs.savefig(ICRTfig)
           chsfigs.savefig(JPHIfig)
           chsfigs.savefig(JCRTfig)
           chsfigs.savefig(SCRTfig)
           chsfigs.savefig(BF2Dfig)
           chsfigs.savefig(BFRZfig)
           chsfigs.savefig(PSRZfig)
           chsfigs.close()

    elif report:
       for h5fid in h5list[0::skipfigs+1]:
           print('Plotting CHEASE data in: %s ...' % h5fid)

           if reportpath[-1] != "/": reportpath += "/"
           if 'report' not in reportpath:
              reportpath += "report/"
           if not os.path.isdir(reportpath):
              os.system('mkdir %s' % reportpath)

           CHEASEdata = read_chease(cheasefpath=h5fid)
           CHEASEdataKeys = CHEASEdata.keys()

           ETMPfig = plt.figure("Electron Temperature")
           plt.plot(CHEASEdata['PSIN'],CHEASEdata['Te'],label='Te')
           plt.plot(CHEASEdata['PSIN'],CHEASEdata['Ti'],label='Ti')
           plt.title('Temperature Profiles')
           plt.xlabel('$\psi$')
           plt.ylabel('$T_e$')
           plt.legend()
           ETMPfig.savefig(reportpath+"chease_temperature.png")
           plt.close(ETMPfig)

           EDENfig = plt.figure("Electron Density")
           plt.plot(CHEASEdata['PSIN'],CHEASEdata['ne'],label='ne')
           plt.plot(CHEASEdata['PSIN'],CHEASEdata['ni'],label='ni')
           plt.title('Density Profiles')
           plt.xlabel('$\psi$')
           plt.ylabel('$n_e$')
           plt.legend()
           EDENfig.savefig(reportpath+"chease_density.png")
           plt.close(EDENfig)
           
           PPRMfig = plt.figure("P'")
           plt.plot(CHEASEdata['PSIN'],CHEASEdata['pprime'])
           plt.title("P' Profiles")
           plt.xlabel('$\psi$')
           plt.ylabel("P'")
           PPRMfig.savefig(reportpath+"chease_pprime.png")
           plt.close(PPRMfig)
       
           TTPMfig = plt.figure("TT'")
           plt.plot(CHEASEdata['PSIN'],CHEASEdata['ffprime'])
           plt.title("FF' Profiles")
           plt.xlabel('$\psi$')
           plt.ylabel("TT'")
           TTPMfig.savefig(reportpath+"chease_ffprime.png")
           plt.close(TTPMfig)
       
           SFTYfig = plt.figure("Safety Factor (q)")
           plt.plot(CHEASEdata['PSIN'],CHEASEdata['q'])
           plt.title("Safety Factor Profiles")
           plt.xlabel('$\psi$')
           plt.ylabel("q")
           SFTYfig.savefig(reportpath+"chease_safetyfactor.png")
           plt.close(SFTYfig)

           PSRZfig = plt.figure("Magnetic Poloidal Boundary Surface")
           plt.plot(CHEASEdata['rbound'],CHEASEdata['zbound'])
           plt.title('Magnetic Poloidal Boundary Surface')
           plt.xlabel('$R$')
           plt.ylabel('$Z$')
           PSRZfig.savefig(reportpath+"chease_magsurfbound.png")
           plt.close(PSRZfig)

           del(CHEASEdata)

    return 1


def cheasepy(srcVals={},namelistVals={},pltVals={},cheaseVals={},importedVals={}):
    if   sys.version_info[0] > 3:
         PYTHON3 = True; PYTHON2 = False
    elif sys.version_info[0] < 3:
         PYTHON2 = True; PYTHON3 = False

    CGRN = '\x1b[32m'
    CYLW = '\x1b[33m'
    CBLU = '\x1b[34m'
    CRED = '\x1b[91m'
    CEND = '\x1b[0m'

    while True:
          print(CGRN+'Select on of the following options:'+CEND)
          print(CGRN+'(1) Run the Code and Plot the Outputs.'+CEND)
          print(CGRN+'(2) Plot the Available Outputs.'+CEND)
          print(CGRN+'(3) Remove Input/Output Files.'+CEND)
          print(CGRN+'(4) Exit.'+CEND)
          try:
              if 'runmode' in cheaseVals:
                 selection = int(cheaseVals['runmode'])
                 print('Selected Option: %d' %selection)
              else:
                 selection = int(input('Selected Option: '))
              if    selection in [1,2,3,4]:
                    if    selection == 3:
                       if glob('./NGA'):                   os.system('rm NGA')
                       if glob('./NDES'):                  os.system('rm NDES')
                       if glob('./*OUT*'):                 os.system('rm *OUT*')
                       if glob('./*.pdf'):                 os.system('rm *.pdf')
                       if glob('./EXPEQ'):                 os.system('rm EXPEQ')
                       if glob('./EXPTNZ'):                os.system('rm EXPTNZ')
                       if glob('./*_EQDSK'):               os.system('rm *_EQDSK')
                       if glob('./*CHEASE'):               os.system('rm *CHEASE')
                       if glob('./*ITERDB'):               os.system('rm *ITERDB')
                       if glob('./*_EXPTNZ'):              os.system('rm *_EXPTNZ')
                       if glob('./ogyropsi*'):             os.system('rm ogyropsi*')
                       if glob('./*PROFILES'):             os.system('rm *PROFILES')
                       if glob('./chease_iter*'):          os.system('rm chease_iter*')
                       if glob('./EXPEQ_EQDSK*'):          os.system('rm EXPEQ_EQDSK*')
                       if glob('./EXPEQ_EXPEQ*'):          os.system('rm EXPEQ_EXPEQ*')
                       if glob('./EXPEQ_iter*.IN'):        os.system('rm EXPEQ_iter*.IN')
                       if glob('./EXPTNZ_iter*.IN'):       os.system('rm EXPTNZ_iter*.IN')
                       if glob('./chease_namelist*'):      os.system('rm chease_namelist*')
                       sys.exit()
                    elif  selection == 4: sys.exit()
                    else: break
              else: raise(NameError)
          except NameError:
             print(CRED+'Select between 1,2,3, and 4 options.'+CEND)
             continue

    '''
    "iterTotal     = n" where n is the number of chease iterations
                              n = 0, 1, 2, ...
                              n = 0 just reproduce the equilibrium
                                    given in the input sources based
                                    on the input set of parameters.
    "boundary_type  = n" where n in geometry options
    Boundary Surface options:
    n = 0 or 'asis' --- Take the boundary surface 'as is'
    n = 1 or 'interp' - Interpolate the given boundary surface

    "current_src   = n" where n in geometry options
    Geometry Source options:
    n = 0 or 'chease'
    n = 1 or 'eqdsk'
    n = 2 or 'expeq'
    n = 7 or 'imported'

    "rhomesh_src   = n" where n in meshgrid options
    Meshgrid Source options:
    n = 0 or 'chease'
    n = 1 or 'eqdsk'

    "pressure_src  = n" where n in profiles options
    "eprofiles_src = n" where n in profiles options
    "iprofiles_src = n" where n in profiles options
    Profiles Source options:
    n = 0 or 'chease'
    n = 1 or 'eqdsk'
    n = 2 or 'expeq'
    n = 3 or 'exptnz'
    n = 4 or 'profiles'
    n = 5 or 'iterdb'
    n = 7 or 'imported'
    '''

    srcValsKeys     = srcVals.keys()
    if   'iterTotal'   in srcVals: iterTotal     = srcVals['iterTotal']
    else:                          iterTotal     = 0  

    if 'boundary_type' in srcVals: boundary_type = srcVals['boundary_type']   
    else:                          boundary_type = 0

    rhomesh_update = False
    if   'rhomesh_src' in srcVals:
         if   type(srcVals['rhomesh_src']) == list:
              rhomesh_src    = srcVals['rhomesh_src'][0]
              rhomesh_update = True
         else:
              rhomesh_src    = srcVals['rhomesh_src']
    else:
              rhomesh_src    = None

    current_update = False
    if   'current_src' in srcVals:
         if   type(srcVals['current_src']) == list:
              current_src    = srcVals['current_src'][0]
              current_update = True
         else:
              current_src    = srcVals['current_src']
    else:
              current_src    = None

    pressure_update = False
    if   'pressure_src' in srcVals:
         if   type(srcVals['pressure_src']) == list:
              pressure_src    = srcVals['pressure_src'][0]
              pressure_update = True
         else:
              pressure_src    = srcVals['pressure_src']
    else:
              pressure_src    = None

    eprofiles_update = False
    if   'eprofiles_src' in srcVals:
         if   type(srcVals['eprofiles_src']) == list:
              eprofiles_src    = srcVals['eprofiles_src'][0]
              eprofiles_update = True
         else:
              eprofiles_src    = srcVals['eprofiles_src']
    else:
              eprofiles_src    = None

    iprofiles_update = False
    if   'iprofiles_src' in srcVals:
         if   type(srcVals['iprofiles_src']) == list:
              iprofiles_src    = srcVals['iprofiles_src'][0]
              iprofiles_update = True
         else:
              iprofiles_src    = srcVals['iprofiles_src']
    else:
              iprofiles_src    = None


    eqdskrequired    = False
    expeqrequired    = False
    cheaserequired   = False
    exptnzrequired   = False
    iterdbrequired   = False
    profilesrequired = False

    if   rhomesh_src  in [0,'chease']:   cheaserequired   = True
    elif rhomesh_src  in [1,'eqdsk']:    eqdskrequired    = True
    elif rhomesh_src  in [7,'imported']: importedrequired = True

    if   current_src  in [0,'chease']:   cheaserequired   = True
    elif current_src  in [1,'eqdsk']:    eqdskrequired    = True
    elif current_src  in [2,'expeq']:    expeqrequired    = True
    elif current_src  in [7,'imported']: importedrequired = True

   #if not (cheaserequired or eqdskrequired or expeqrequired):
   #   raise IOError('FATAL: No Geometry File is provided!')

    if   pressure_src in [0,'chease']:   cheaserequired   = True
    elif pressure_src in [1,'eqdsk']:    eqdskrequired    = True
    elif pressure_src in [2,'expeq']:    expeqrequired    = True
    elif pressure_src in [3,'exptnz']:   exptnzrequired   = True
    elif pressure_src in [4,'profiles']: profilesrequired = True
    elif pressure_src in [5,'iterdb']:   iterdbrequired   = True
    elif pressure_src in [7,'imported']: importedrequired = True

    if   eprofiles_src in [0,'chease']:   cheaserequired   = True
    elif eprofiles_src in [3,'exptnz']:   exptnzrequired   = True
    elif eprofiles_src in [4,'profiles']: profilesrequired = True
    elif eprofiles_src in [5,'iterdb']:   iterdbrequired   = True
    elif eprofiles_src in [7,'imported']: importedrequired = True

    if   iprofiles_src in [0,'chease']:   cheaserequired   = True
    elif iprofiles_src in [3,'exptnz']:   exptnzrequired   = True
    elif iprofiles_src in [4,'profiles']: profilesrequired = True
    elif iprofiles_src in [5,'iterdb']:   iterdbrequired   = True
    elif iprofiles_src in [7,'imported']: importedrequired = True

    
   #if not (cheaserequired or exptnzrequired or profilesrequired or iterdbrequired):
   #   raise IOError('FATAL: No Profiles File is provided!')

    if selection == 1:
       if glob('./EQSDK')!=[]:     EQDSKexist  = True
       else:                       EQDSKexist  = False

       if glob('./EXPEQ')!=[]:     EXPEQexist  = True
       else:                       EXPEQexist  = False

       if glob('./EXPTNZ')!=[]:    EXPTNZexist = True
       else:                       EXPTNZexist = False

       if glob('./*CHEASE')!=[]:   CHEASEexist = True
       else:                       CHEASEexist = False

       if glob('./*ITERDB')!=[]:   ITERDBexist = True
       else:                       ITERDBexist = False

       if glob('./*PROFILES')!=[]: PROFILESexist = True
       else:                       PROFILESexist = False

       if   'removeinputs' in cheaseVals:
            if type(cheaseVals['removeinputs'])==str:
              removeinputs = cheaseVals['removeinputs'].lower()
            else:
              removeinputs = cheaseVals['removeinputs']
       elif EXPEQexist or EXPTNZexist or EQDSKexist or ITERDBexist or PROFILESexist:
            os.system('ls')
            if   PYTHON3:
               removeinputs = str(input(CRED+'Remove input files from a previous run (yes/no)? '+CEND)).lower()
            elif PYTHON2:
               removeinputs = raw_input(CRED+'Remove input files from a previous run (yes/no)? '+CEND).lower()
       else:
            removeinputs = True
 
       if removeinputs in ['yes','y',1,True]:
          removeoutputs = True
          if glob('./*_PROFILES'): os.system('rm ./*_PROFILES')
          if glob('./*_EXPTNZ'):   os.system('rm ./*_EXPTNZ')
          if glob('./*_CHEASE'):   os.system('rm ./*_CHEASE')
          if glob('./*_ITERDB'):   os.system('rm ./*_ITERDB')
          if glob('./*_EQDSK'):    os.system('rm ./*_EQDSK')
          if glob('./*_EXPEQ'):    os.system('rm ./*_EXPEQ')
          PROFILESexist = False
          CHEASEexist   = False
          EXPTNZexist   = False
          ITERDBexist   = False
          EXPEQexist    = False
          EQDSKexist    = False
         #if os.path.isfile('./chease_parameters.csv'): os.system('rm chease_parameters.csv')

       while True:
             print(CYLW+'Select CHEASE running mode:'+CEND)
             print(CYLW+'(1) Check Equilibrium Preservation Over Multiple Iterations.'+CEND)
             print(CYLW+'(2) Converge to Total Current by correcting Current.'+CEND)
             print(CYLW+'(3) Converge to Total Current by correcting Pressure.'+CEND)
             try:
                if 'cheasemode' in cheaseVals:
                   cheasemode = cheaseVals['cheasemode']
                   print('Selected Option: %d' %cheasemode)
                else:
                   cheasemode = int(input('Selected Option: '))
                if    cheasemode in [1,2,3]: break
                else: raise(NameError)
             except NameError:
                print(CRED+'Select between 1, 2, and 3 options.'+CEND)
                continue
    
       namelistParam = {}
       namelistUpdate = False
       if namelistVals:
          namelistValsKeys = list(namelistVals.keys())
          if type(namelistVals[namelistValsKeys[0]]) in [float,int,complex]:
             for ikey in namelistValsKeys:
                 namelistParam[ikey] = namelistVals[ikey]
          else:
             namelistrecs   = len(namelistVals[namelistVals.keys()[0]])
             if namelistrecs > 1: namelistUpdate = True
             for ikey in namelistVals.keys():
                 namelistParam[ikey] = namelistVals[ikey][0]

       if   'removeoutputs' in cheaseVals:
            if type(cheaseVals['removeoutputs'])==str:
               removeoutputs = cheaseVals['removeoutputs'].lower()
            else:
               removeoutputs = cheaseVals['removeoutputs']
            print(CBLU+'Remove output files of previous a run (yes/no)? '+str(removeoutputs)+CEND)
       else:
            os.system('ls')
            if   PYTHON3:
                 removeoutputs = str(input(CBLU+'Remove output files from previous a run (yes/no)? '+CEND)).lower()
            elif PYTHON2:
                 removeoutputs = raw_input(CBLU+'Remove output files from previous a run (yes/no)? '+CEND).lower()

       if removeoutputs in ['yes','y',1,True]:
          if glob('./NGA'):              os.system('rm NGA')
          if glob('./NDES'):             os.system('rm NDES')
          if glob('./*OUT*'):            os.system('rm *OUT*')
          if glob('./*.pdf'):            os.system('rm *.pdf')
          if glob('./EXPEQ'):            os.system('rm EXPEQ')
          if glob('./EXPTNZ'):           os.system('rm EXPTNZ')
          if glob('./ogyropsi*'):        os.system('rm ogyropsi*')
          if glob('./chease_iter*'):     os.system('rm chease_iter*')
          if glob('./EXPEQ_EQDSK*'):     os.system('rm EXPEQ_EQDSK*')
          if glob('./EXPEQ_EXPEQ*'):     os.system('rm EXPEQ_EXPEQ*')
          if glob('./EXPEQ_iter*.IN'):   os.system('rm EXPEQ_iter*.IN')
          if glob('./EXPTNZ_iter*.IN'):  os.system('rm EXPTNZ_iter*.IN')
          if glob('./chease_namelist*'): os.system('rm chease_namelist*')
          print(CRED+'List of Available CHEASE Files:'+CEND)
          os.system('ls')
    
       shotlist = sorted(glob('./shots/*'))
       if 'shotpath' in cheaseVals:
          shotrec = cheaseVals['shotpath']
          if os.path.isdir(shotrec):
              shotpath = shotrec
              if shotpath[-1]!='/': shotpath+='/'
              slashpos = findall(shotpath,'/')
              if len(slashpos)>=2:
                  shotfile = shotpath[slashpos[-2]+1:-1]
              else:
                  shotfile = shotpath[0:]
          elif type(shotrec)==int:
              print('Select Shot Number: %d' %shotrec)
              shotpath = shotlist[shotrec-1]
              shotfile = shotlist[shotrec-1][8:]
          else:
              raise IOError(shotrec+' IS NOT A DIRECTORY OR SHOT NUMBER. Exit!')
       else:
          while True:
             print(CYLW+'List of the available shots:'+CEND)
             for ishot in range(len(shotlist)):
                 print(CYLW+'(%02d) %s' % (ishot+1,shotlist[ishot][8:])+CEND)
             try:
                shotrec = int(input('Select Shot Number: '))
                if shotrec-1 in range(len(shotlist)):
                   print(CGRN+'Chease runs the %s shot.' % shotlist[shotrec-1][8:]+CEND)
                   shotpath = shotlist[shotrec-1]
                   if shotpath[-1]!='/': shotpath+='/'
                   shotfile = shotlist[shotrec-1][8:]
                   break
                else:
                   raise(NameError)
             except NameError:
                print(CRED+'Choose ONLY from the %0d available shots.' % len(shotlist)+CEND)
                continue

       if os.path.isfile('%s/%s_EXPEQ' % (shotpath,shotfile)):    EXPEQexist = True
       if os.path.isfile('%s/%s_EQDSK' % (shotpath,shotfile)):    EQDSKexist = True
       if os.path.isfile('%s/%s_CHEASE' % (shotpath,shotfile)):   CHEASEexist = True
       if os.path.isfile('%s/%s_EXPTNZ' % (shotpath,shotfile)):   EXPTNZexist = True
       if os.path.isfile('%s/%s_ITERDB' % (shotpath,shotfile)):   ITERDBexist = True
       if os.path.isfile('%s/%s_PROFILES' % (shotpath,shotfile)): PROFILESexist = True


       if eprofiles_src == None:
           if   EXPTNZexist:   eprofiles_src = 3; exptnzrequired = True
           elif PROFILESexist: eprofiles_src = 4; proflilesrequired = True
           elif ITERDBexist:   eprofiles_src = 5; iterdbrequired = True
           elif CHEASEexist:   eprofiles_src = 0; cheaserequired = True

       if iprofiles_src == None:
           if   EXPTNZexist:   iprofiles_src = 3; exptnzrequired = True
           elif PROFILESexist: iprofiles_src = 4; profilesrequired = True
           elif ITERDBexist:   iprofiles_src = 5; iterdbrequired = True
           elif CHEASEexist:   iprofiles_src = 0; cheaserequired = True

       if pressure_src == None:
           if   EQDSKexist:    pressure_src = 1;  eqdskrequired = True
           elif EXPEQexist:    pressure_src = 2;  expeqrequired = True
           elif EXPTNZexist:   pressure_src = 3;  exptnzrequired = True
           elif PROFILESexist: pressure_src = 4;  profilesrequired = True
           elif ITERDBexist:   pressure_src = 5;  iterdbrequired = True
           elif CHEASEexist:   pressure_src = 0;  cheaserequired = True

       if current_src == None:
           if   EQDSKexist:    current_src = 1;   eqdskrequired = True
           elif EXPEQexist:    current_src = 2;   expeqrequired = True
           elif CHEASEexist:   current_src = 0;   cheaserequired = True

       namelist = create_namelist(setParam=namelistParam)

       if   'NFUNRHO'  in namelist: rhomesh_type = int(namelist['NFUNRHO'])
       elif 'NRHOMESH' in namelist: rhomesh_type = int(namelist['NRHOMESH'])
       else:                        rhomesh_type = 0

       if   'NPPFUN'   in namelist: pressure_type = int(namelist['NPPFUN'])
       else:                        pressure_type = 8

       if   'NSTTP'    in namelist: current_type  = int(namelist['NSTTP'])
       else:                        current_type  = 1

       if rhomesh_src in [7,'imported']:
          if   'rhopsi' not in importedVals and 'rhotor' not in importedVals:
               raise ValueError('importedVals MUST contain rhopsi and rhotor with rhomesh_src = 7 or "imported"')
       if eprofiles_src in [7,'imported']:
          if   'Te' not in importedVals and 'ne' not in importedVals:
               raise ValueError('importedVals MUST contain Te and ne with eprofiles_src = 7 or "imported"')
       if iprofiles_src in [7,'imported']:
          if   'Ti' not in importedVals and 'ni' not in importedVals and 'Zeff' not in importedVals:
               raise ValueError('importedVals MUST contain Ti, ni, and Zeff with iprofiles_src = 7 or "imported"')
       if pressure_src in [7,'imported']:
          if   pressure_type in [8,'pressure'] and 'pressure' not in importedVals:
               raise ValueError('importedVals MUST contain pressure with pressure_src = 7 or "imported"')
          elif pressure_type in [4,'pprime']   and 'pprime'   not in importedVals:
               raise ValueError('importedVals MUST contain pprime with pressure_src = 7 or "imported"')
       if current_src in [7,'imported']:
          if   current_type in [1,'ffprime'] and 'ffprime' not in importedVals:
               raise ValueError('importedVals MUST contain ffprime current_src = 7 or "imported"')
          elif current_type in [2,'istr'] and 'Istr' not in importedVals:
               raise ValueError('importedVals MUST contain Istr current_src = 7 or "imported"')
          elif current_type in [3,'iprl'] and 'Iprl' not in importedVals:
               raise ValueError('importedVals MUST contain Iprl current_src = 7 or "imported"')
          elif current_type in [4,'jprl'] and 'Jprl' not in importedVals:
               raise ValueError('importedVals MUST contain Jprl current_src = 7 or "imported"')
          elif current_type in [5,'q'] and 'q' not in importedVals:
               raise ValueError('importedVals MUST contain q current_src = 7 or "imported"')

       print(shotpath,shotfile)
       if   eqdskrequired and EQDSKexist:
            os.system('cp   %s/%s_EQDSK .'          % (shotpath,shotfile))
            eqdskfpath =      '%s_EQDSK'            %          (shotfile)
       elif eqdskrequired:
            raise IOError('EQDSK file NOT FOUND in the given path!')

       if   cheaserequired and CHEASEexist:
            os.system('cp   %s/%s_CHEASE .'         % (shotpath,shotfile))
            cheasefpath =     '%s_CHEASE'           %          (shotfile)
       elif cheaserequired:
            raise IOError('CHEASE file NOT FOUND in the given path!')

       if   expeqrequired and EXPEQexist:
            os.system('cp     %s/%s_EXPEQ ./EXPEQ'    % (shotpath,shotfile))
            expeqfpath = 'EXPEQ'
       elif expeqrequired:
            raise IOError('EXPEQ file NOT FOUND in the given path!')

       if   exptnzrequired  and EXPTNZexist:
            os.system('cp   %s/%s_EXPTNZ .'    % (shotpath,shotfile))
            exptnzfpath =     '%s_EXPTNZ'      %          (shotfile)
       elif exptnzrequired:
            raise IOError('EXPTNZ file NOT FOUND in the given path!')

       if   profilesrequired and PROFILESexist:
            os.system('cp   %s/%s_PROFILES .'  % (shotpath,shotfile))
            profilesfpath =   '%s_PROFILES'    %          (shotfile)
       elif profilesrequired:
            raise IOError('Profiles file NOT FOUND in the given path!')

       if   iterdbrequired and ITERDBexist:
            os.system('cp   %s/%s_ITERDB .'    % (shotpath,shotfile))
            iterdbfpath =     '%s_ITERDB'      %          (shotfile)
       elif iterdbrequired:
            raise IOError('ITERDB file NOT FOUND in the given path!')

       exptnzParam = {}
       if int(namelist['NBSEXPQ']) != 0:
          exptnzParam['nrhomesh']  = [rhomesh_type,rhomesh_src]
          exptnzParam['eprofiles'] = eprofiles_src
          exptnzParam['iprofiles'] = iprofiles_src

          if   rhomesh_src in [0,'chease']:
               if   eprofiles_src in [0,'chease']   and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
          elif rhomesh_src in [1,'eqdsk']:
               if   eprofiles_src in [0,'chease']   and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,imported=importedVals)
          elif rhomesh_src in [7,'imported',None]:
               if   eprofiles_src in [0,'chease']   and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [0,'chease']   and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,profiles=profilesfpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,profiles=profilesfpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [4,'profiles'] and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,iterdb=iterdbfpath,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,iterdb=iterdbfpath,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [0,'chease']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [3,'exptnz']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [4,'profiles']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,profiles=profilesfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [5,'iterdb']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,iterdb=iterdbfpath,imported=importedVals)
               elif eprofiles_src in [7,'imported'] and iprofiles_src in [7,'imported']:
                    fusionfiles.write_exptnz(setParam=exptnzParam,imported=importedVals)


       os.system('ls')
       if cheaseVals!={}:
          runchease = 'yes'
          print(CBLU+'Do you want to continue? (yes/no)? '+str(runchease)+CEND)
       else:
          while True:
             if   PYTHON3:
                  runchease = str(input(CBLU+'Do you want to continue? (yes/no)? '+CEND)).lower()
             elif PYTHON2:
                  runchease = raw_input(CBLU+'Do you want to continue? (yes/no)? '+CEND).lower()
             if    runchease in ['yes','y','no','n',1,True,False]: break
             else: print('Enter a valid value (y,yes,no,n)');      continue

       if runchease not in ['yes','y',1,True]: sys.exit()

       eqdskdata = fusionfiles.read_eqdsk(eqdskfpath=eqdskfpath)
       if 'R0EXP' in namelistVals:
           R0EXP = namelistVals['R0EXP']
       else:
           R0EXP = abs(eqdskdata['RCTR'])
       if 'B0EXP' in namelistVals:
           B0EXP = namelistVals['B0EXP']
       else:
           B0EXP = abs(eqdskdata['BCTR'])
       if 'ITEXP' in namelistVals:
           ITEXP = namelistVals['ITEXP']
       else:
           ITEXP = abs(eqdskdata['CURNT'])

       expeqParam = {}
       if   int(namelist['NEQDSK']) == 1:
            print('Reading from EQDSK file.')
            os.system('cp *_EQDSK  EXPEQ')
       elif int(namelist['NEQDSK']) == 0:
            print('Reading from EXPEQ file.')
            expeqParam['nrhomesh']   = [rhomesh_type,rhomesh_src]
            expeqParam['nppfun']     = [pressure_type,pressure_src]
            expeqParam['nsttp']      = [current_type,current_src]
            expeqParam['boundary']   =  boundary_type
            expeqParam['cheasemode'] =  1
            expeqParam['ITEXP']      =  ITEXP
            expeqParam['R0EXP']      =  R0EXP
            expeqParam['B0EXP']      =  B0EXP

            if not os.path.isfile('EXPEQ'):
               if   rhomesh_src in [0,'chease']:
                    if   pressure_src in [0,'chease']   and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [0,'chease']   and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [0,'chease']   and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [0,'chease']   and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,exptnz=exptnzfpath,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,exptnz=exptnzfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,profiles=profilesfpath,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,profiles=profilesfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,iterdb=iterdbfpath,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,iterdb=iterdbfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)

               elif rhomesh_src in [1,'eqdsk']:
                    if   pressure_src in [0,'chease']   and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [0,'chease']   and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [0,'chease']   and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [0,'chease']   and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,profiles=profilesfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals) 

               elif rhomesh_src in [7,'imported',None]:
                    if   pressure_src in [0,'chease']   and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [0,'chease']   and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [0,'chease']   and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [0,'chease']   and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [1,'eqdsk']    and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [2,'expeq']    and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,exptnz=exptnzfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [3,'exptnz']   and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,exptnz=exptnzfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,profiles=profilesfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [4,'profiles'] and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,profiles=profilesfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,iterdb=iterdbfpath,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [5,'iterdb']   and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,iterdb=iterdbfpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [0,'chease']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [1,'eqdsk']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [2,'expeq']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,expeq=expeqfpath,imported=importedVals)
                    elif pressure_src in [7,'imported'] and current_src in [7,'imported']:
                         expeqdata = fusionfiles.write_expeq(setParam=expeqParam,imported=importedVals)

       namelistParam['R0EXP'] = R0EXP
       namelistParam['B0EXP'] = B0EXP

       if   current_src in [0,'chease']:
            if   rhomesh_src in [0,'chease',None]:
                 cheasedata = read_chease(cheasefpath=cheasefpath)
            elif rhomesh_src in [1,'eqdsk']:
                 cheasedata = read_chease(cheasefpath=cheasefpath,eqdsk=eqdskfpath)
            if 'QSPEC' in namelistVals and 'CSSPEC' in namelistVals:
               namelistParam['QSPEC']  = namelistVals['QSPEC']
               namelistParam['CSSPEC'] = namelistVals['CSSPEC']
            else:
               if 'q' in importedVals:
                  namelistParam['QSPEC'] = importedVals['q'][0]
               else:
                  namelistParam['QSPEC'] = cheasedata['q'][0]
               namelistParam['CSSPEC'] = 0.0
       elif current_src in [1,'eqdsk']:
            if   rhomesh_src in [0,'chease']:
                 eqdskdata = fusionfiles.read_eqdsk(eqdskfpath=eqdskfpath,chease=cheasefpath)
            elif rhomesh_src in [1,'eqdsk',None]:
                 eqdskdata = fusionfiles.read_eqdsk(eqdskfpath=eqdskfpath)
            if 'QSPEC' in namelistVals and 'CSSPEC' in namelistVals:
               namelistParam['QSPEC']  = namelistVals['QSPEC']
               namelistParam['CSSPEC'] = namelistVals['CSSPEC']
            else:
               if 'q' in importedVals:
                  namelistParam['QSPEC'] = importedVals['q'][0]
               else:
                  namelistParam['QSPEC'] = eqdskdata['q'][0]
               namelistParam['CSSPEC'] = 0.0
       elif current_src in [2,'expeq']:
            if   rhomesh_src in [0,'chease']:
                 expeqdata = fusionfiles.read_expeq(expeqfpath=expeqfpath,chease=cheasefpath)
            elif rhomesh_src in [1,'eqdsk']:
                 expeqdata = fusionfiles.read_expeq(expeqfpath=expeqfpath,eqdsk=eqdskfpath)
            elif rhomesh_src in [None]:
                 expeqdata = fusionfiles.read_expeq(expeqfpath=expeqfpath)
            if 'QSPEC' in namelistVals and 'CSSPEC' in namelistVals:
               namelistParam['QSPEC']  = namelistVals['QSPEC']
               namelistParam['CSSPEC'] = namelistVals['CSSPEC']
            else:
               if   'q' in importedVals:
                    namelistParam['QSPEC'] = importedVals['q'][0]
               elif 'q' in expeqdata:
                    namelistParam['QSPEC'] = expeqdata['q'][0]
               namelistParam['CSSPEC'] = 0.0
       namelistParam['NCSCAL'] = 4

       namelist = create_namelist(setParam=namelistParam)
       
       cheasefname = sorted(glob('./chease_*.h5'))
       if   len(cheasefname)==0:
            it=0
           #exit_status = os.system('./chease_hdf5 chease_namelist > iter%03d.OUT' % it)
            exit_status = os.system('./chease_hdf5 chease_namelist')
           #exit_status = subprocess.call(['./chease_hdf5','chease_namelist'])
            if abs(exit_status) > 0: sys.exit()
            if os.path.isfile('./chease_namelist'): os.system('cp ./chease_namelist ./chease_namelist_iter%03d' % it)
            if os.path.isfile('./ogyropsi.dat'):    os.system('mv ./ogyropsi.dat chease_iter%03d.dat'         % it)
            if os.path.isfile('./ogyropsi.h5'):     os.system('mv ./ogyropsi.h5 chease_iter%03d.h5'           % it)
            if os.path.isfile('./EXPEQ'):           os.system('cp ./EXPEQ EXPEQ_iter%03d.IN'                    % it)
            if os.path.isfile('./EXPTNZ'):          os.system('cp ./EXPTNZ EXPTNZ_iter%03d.IN'                  % it)
            if os.path.isfile('./EXPEQ.OUT'):       os.system('mv ./EXPEQ.OUT EXPEQ_iter%03d.OUT'               % it)
            if os.path.isfile('./EXPTNZ.OUT'):      os.system('mv ./EXPTNZ.OUT EXPTNZ_iter%03d.OUT'             % it)
            if os.path.isfile('./EXPEQ_EXPEQ.IN'):  os.system('mv ./EXPEQ_EXPEQ.IN EXPEQ_EXPEQ_iter%03d.IN'     % it)
            if os.path.isfile('./EQDSK_EXPEQ.IN'):  os.system('mv ./EQDSK_EXPEQ.IN EQDSK_EXPEQ_iter%03d.IN'     % it)
       else:
            it=int(cheasefname[-1][-6:-3])+1

       eqdskrequired    = False
       expeqrequired    = False
       cheaserequired   = False
       exptnzrequired   = False
       iterdbrequired   = False
       profilesrequired = False

       if   rhomesh_update:
            rhomesh_src = srcVals['rhomesh_src'][min(len(srcVals['rhomesh_src'])-1,it+1)]
       if   rhomesh_src  in [0,'chease']:   cheaserequired   = True
       elif rhomesh_src  in [1,'eqdsk']:    eqdskrequired    = True
       elif rhomesh_src  in [2,'expeq']:    expeqrequired    = True

       if   current_update:
            current_src = srcVals['current_src'][min(len(srcVals['current_src'])-1,it+1)]
       if   current_src  in [0,'chease']:   cheaserequired   = True
       elif current_src  in [1,'eqdsk']:    eqdskrequired    = True
       elif current_src  in [2,'expeq']:    expeqrequired    = True

       if not (cheaserequired or eqdskrequired or expeqrequired):
          raise IOError('FATAL: No Geometry File is provided!')

       if   pressure_update:
            pressure_src = srcVals['pressure_src'][min(len(srcVals['pressure_src'])-1,it+1)]
       if   pressure_src in [0,'chease']:   cheaserequired   = True
       elif pressure_src in [1,'eqdsk']:    eqdskrequired    = True
       elif pressure_src in [2,'expeq']:    expeqrequired    = True
       elif pressure_src in [3,'exptnz']:   exptnzrequired   = True
       elif pressure_src in [4,'profiles']: profilesrequired = True
       elif pressure_src in [5,'iterdb']:   iterdbrequired   = True

       if   eprofiles_update:
            eprofiles_src = srcVals['eprofiles_src'][min(len(srcVals['eprofiles_src'])-1,it+1)]
       if   eprofiles_src in [0,'chease']:   cheaserequired   = True
       elif eprofiles_src in [3,'exptnz']:   exptnzrequired   = True
       elif eprofiles_src in [4,'profiles']: profilesrequired = True
       elif eprofiles_src in [5,'iterdb']:   iterdbrequired   = True

       if   iprofiles_update:
            iprofiles_src = srcVals['iprofiles_src'][min(len(srcVals['iprofiles_src'])-1,it+1)]
       if   iprofiles_src in [0,'chease']:   cheaserequired   = True
       elif iprofiles_src in [3,'exptnz']:   exptnzrequired   = True
       elif iprofiles_src in [4,'profiles']: profilesrequired = True
       elif iprofiles_src in [5,'iterdb']:   iterdbrequired   = True

      #if not (exptnzrequired or profilesrequired or iterdbrequired or cheaserequired):
      #   raise IOError('FATAL: No Profiles File is provided!')

       if eqdskrequired    and os.path.isfile('%s_EQDSK'   %(shotfile)):
          if os.path.isfile('%s/%s_EQDSK'   %(shotpath,shotfile)):
             os.system('cp   %s/%s_EQDSK .'     % (shotpath,shotfile))
             eqdskfpath =      '%s_EQDSK'       %          (shotfile)
          elif eqdskrequired:
             raise IOError('EQDSK file NOT FOUND in the given path!')

       if profilesrequired and not os.path.isfile('%s_PROFILES' %(shotfile)):
          if os.path.isfile('%s/%s_PROFILES' %(shotpath,shotfile)):
             os.system('cp   %s/%s_PROFILES .'  % (shotpath,shotfile))
             profilesfpath =   '%s_PROFILES'    %          (shotfile)
          elif profilesrequired:
               raise IOError('Profiles file NOT FOUND in the given path!')

       if iterdbrequired   and not os.path.isfile('%s_ITERDB'   %(shotfile)):
          if os.path.isfile('%s/%s_ITERDB'   %(shotpath,shotfile)):
             os.system('cp   %s/%s_ITERDB .'    % (shotpath,shotfile))
             iterdbfpath =     '%s_ITERDB'      %          (shotfile)
          elif iterdbrequired:
             raise IOError('ITERDB file NOT FOUND in the given path!')


       if 'NPROPT' in namelist:
          current_type = abs(int(namelist['NPROPT']))
       else:
          current_type = int(namelist['NSTTP'])
       pressure_type   = int(namelist['NPPFUN'])
       rhomesh_type    = int(namelist['NRHOMESH'])

       expeqParam['nsttp']      = [current_type,current_src]
       expeqParam['nppfun']     = [pressure_type,pressure_src]
       expeqParam['nrhomesh']   = [rhomesh_type,rhomesh_src]
       expeqParam['boundary']   =  boundary_type
       expeqParam['cheasemode'] =  cheasemode

       exptnzParam['nrhomesh']  = [rhomesh_type,rhomesh_src]
       exptnzParam['eprofiles'] = eprofiles_src
       exptnzParam['iprofiles'] = iprofiles_src

       cheasefpath = 'chease_iter%03d.h5' % it
       cheasedata = read_chease(cheasefpath=cheasefpath)

       ITErr = (cheasedata['Itor']-ITEXP)/ITEXP
       print('Iter  = ', it)
       print('ITOR  = ', cheasedata['Itor'])
       print('ITEXP = ', ITEXP)
       print('ITErr = ', abs(ITErr))
    
       while (abs(ITErr) > 1.0e-6):
           if (cheasemode == 1) and (it >= iterTotal): break

           expeqfpath  = 'EXPEQ_iter%03d.OUT'   % it
           exptnzfpath = 'EXPTNZ_iter%03d.OUT'  % it
           cheasefpath = 'chease_iter%03d.h5' % it

           if   rhomesh_src in [0,'chease']:
                if   pressure_src in [0,'chease']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath)
                elif pressure_src in [0,'chease']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath)
                elif pressure_src in [0,'chease']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath)
                elif pressure_src in [0,'chease']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [1,'eqdsk']    and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath)
                elif pressure_src in [1,'eqdsk']    and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath)
                elif pressure_src in [1,'eqdsk']    and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,eqdsk=eqdskfpath)
                elif pressure_src in [1,'eqdsk']    and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,imported=importedVals)
                elif pressure_src in [2,'expeq']    and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath)
                elif pressure_src in [2,'expeq']    and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,eqdsk=eqdskfpath)
                elif pressure_src in [2,'expeq']    and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath)
                elif pressure_src in [2,'expeq']    and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                elif pressure_src in [3,'exptnz']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,exptnz=exptnzfpath)
                elif pressure_src in [3,'exptnz']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,exptnz=exptnzfpath)
                elif pressure_src in [3,'exptnz']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,exptnz=exptnzfpath)
                elif pressure_src in [3,'exptnz']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                elif pressure_src in [4,'profiles'] and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,profiles=profilesfpath)
                elif pressure_src in [4,'profiles'] and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,profiles=profilesfpath)
                elif pressure_src in [4,'profiles'] and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,profiles=profilesfpath)
                elif pressure_src in [4,'profiles'] and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                elif pressure_src in [5,'iterdb']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,iterdb=iterdbfpath)
                elif pressure_src in [5,'iterdb']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,iterdb=iterdbfpath)
                elif pressure_src in [5,'iterdb']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,iterdb=iterdbfpath)
                elif pressure_src in [5,'iterdb']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                elif pressure_src in [7,'imported'] and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [7,'imported'] and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,eqdsk=eqdskfpath,imported=importedVals)
                elif pressure_src in [7,'imported'] and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,expeq=expeqfpath,imported=importedVals)
                elif pressure_src in [7,'imported'] and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)

           elif rhomesh_src in [1,'eqdsk']:
                if   pressure_src in [0,'chease']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath)
                elif pressure_src in [0,'chease']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath)
                elif pressure_src in [0,'chease']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,chease=cheasefpath)
                elif pressure_src in [0,'chease']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [1,'eqdsk']    and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath)
                elif pressure_src in [1,'eqdsk']    and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath)
                elif pressure_src in [1,'eqdsk']    and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath)
                elif pressure_src in [1,'eqdsk']    and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                elif pressure_src in [2,'expeq']    and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,chease=cheasefpath)
                elif pressure_src in [2,'expeq']    and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath)
                elif pressure_src in [2,'expeq']    and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath)
                elif pressure_src in [2,'expeq']    and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                elif pressure_src in [3,'exptnz']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,exptnz=exptnzfpath)
                elif pressure_src in [3,'exptnz']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,exptnz=exptnzfpath)
                elif pressure_src in [3,'exptnz']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,exptnz=exptnzfpath)
                elif pressure_src in [3,'exptnz']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
                elif pressure_src in [4,'profiles'] and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,profiles=profilesfpath)
                elif pressure_src in [4,'profiles'] and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,profiles=profilesfpath)
                elif pressure_src in [4,'profiles'] and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,profiles=profilesfpath)
                elif pressure_src in [4,'profiles'] and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
                elif pressure_src in [5,'iterdb']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,iterdb=iterdbfpath)
                elif pressure_src in [5,'iterdb']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,iterdb=iterdbfpath)
                elif pressure_src in [5,'iterdb']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,iterdb=iterdbfpath)
                elif pressure_src in [5,'iterdb']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
                elif pressure_src in [7,'imported'] and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [7,'imported'] and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                elif pressure_src in [7,'imported'] and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                elif pressure_src in [7,'imported'] and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)

           elif rhomesh_src in [7,'imported',None]:
                if   pressure_src in [0,'chease']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [0,'chease']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [0,'chease']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,expeq=expeqfpath,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [0,'chease']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [1,'eqdsk']    and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [1,'eqdsk']    and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                elif pressure_src in [1,'eqdsk']    and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                elif pressure_src in [1,'eqdsk']    and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,imported=importedVals)
                elif pressure_src in [2,'expeq']    and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,expeq=expeqfpath,chease=cheasefpath,imported=importedVals)
                elif pressure_src in [2,'expeq']    and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                elif pressure_src in [2,'expeq']    and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,expeq=expeqfpath,imported=importedVals)
                elif pressure_src in [2,'expeq']    and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,imported=importedVals)
                elif pressure_src in [3,'exptnz']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                elif pressure_src in [3,'exptnz']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
                elif pressure_src in [3,'exptnz']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,expeq=expeqfpath,exptnz=exptnzfpath,imported=importedVals)
                elif pressure_src in [3,'exptnz']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,exptnz=exptnzfpath,imported=importedVals)
                elif pressure_src in [4,'profiles'] and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                elif pressure_src in [4,'profiles'] and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
                elif pressure_src in [4,'profiles'] and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,expeq=expeqfpath,profiles=profilesfpath,imported=importedVals)
                elif pressure_src in [4,'profiles'] and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,profiles=profilesfpath,imported=importedVals)
                elif pressure_src in [5,'iterdb']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                elif pressure_src in [5,'iterdb']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
                elif pressure_src in [5,'iterdb']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,expeq=expeqfpath,iterdb=iterdbfpath,imported=importedVals)
                elif pressure_src in [5,'iterdb']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
                elif pressure_src in [7,'imported']   and current_src in [0,'chease']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                elif pressure_src in [7,'imported']   and current_src in [1,'eqdsk']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
                elif pressure_src in [7,'imported']   and current_src in [2,'expeq']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,expeq=expeqfpath,imported=importedVals)
                elif pressure_src in [7,'imported']   and current_src in [7,'imported']:
                     expeqdata = fusionfiles.write_expeq(setParam=expeqParam,imported=importedVals)


           if   rhomesh_src in [0,'chease']: 
                if   eprofiles_src in [0,'chease']   and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,profiles=profilesfpath)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,iterdb=iterdbfpath)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,exptnz=exptnzfpath)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,iterdb=iterdbfpath)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,exptnz=exptnzfpath)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,profiles=profilesfpath)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)

           elif rhomesh_src in [1,'eqdsk']: 
                if   eprofiles_src in [0,'chease']   and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,exptnz=exptnzfpath)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,profiles=profilesfpath)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,iterdb=iterdbfpath)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,exptnz=exptnzfpath)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,profiles=profilesfpath)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,iterdb=iterdbfpath)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,profiles=profilesfpath)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath,exptnz=exptnzfpath)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath,iterdb=iterdbfpath)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,iterdb=iterdbfpath)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,exptnz=exptnzfpath)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,profiles=profilesfpath)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,chease=cheasefpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,exptnz=exptnzfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,eqdsk=eqdskfpath,imported=importedVals)

           elif rhomesh_src in [7,'imported',None]:
                if   eprofiles_src in [0,'chease']   and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [0,'chease']   and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,exptnz=exptnzfpath,imported=importedVals)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,imported=importedVals)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [3,'exptnz']   and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [4,'profiles'] and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,profiles=profilesfpath,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,iterdb=iterdbfpath,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [5,'iterdb']   and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [0,'chease']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,chease=cheasefpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [3,'exptnz']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,exptnz=exptnzfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [4,'profiles']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,profiles=profilesfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [5,'iterdb']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,iterdb=iterdbfpath,imported=importedVals)
                elif eprofiles_src in [7,'imported'] and iprofiles_src in [7,'imported']:
                     exptnzdata = fusionfiles.write_exptnz(setParam=exptnzParam,imported=importedVals)


           namelistParam['QSPEC'] = cheasedata['q'][0]

           if namelistUpdate:
              irun = min(namelistrecs-1,it+1)
              for ikey in namelistVals.keys():
                  namelistParam[ikey] = namelistVals[ikey][irun]

           namelist = create_namelist(setParam=namelistParam)
           os.system('cp chease_namelist chease_namelist_iter%03d' % (it+1))

           exit_status = os.system('./chease_hdf5 chease_namelist > iter%03d.OUT' % (it+1))
          #exit_status = os.system('./chease_hdf5 chease_namelist')
          #exit_status = subprocess.call(['./chease_hdf5','chease_namelist'])
           if abs(exit_status) > 0: sys.exit()
           if os.path.isfile('./ogyropsi.dat'):   os.system('mv ./ogyropsi.dat chease_iter%03d.dat'       % (it+1))
           if os.path.isfile('./ogyropsi.h5'):    os.system('mv ./ogyropsi.h5 chease_iter%03d.h5'         % (it+1))
           if os.path.isfile('./EXPEQ'):          os.system('cp ./EXPEQ EXPEQ_iter%03d.IN'                % (it+1))
           if os.path.isfile('./EXPTNZ'):         os.system('cp ./EXPTNZ EXPTNZ_iter%03d.IN'              % (it+1))
           if os.path.isfile('./EXPEQ.OUT'):      os.system('mv ./EXPEQ.OUT EXPEQ_iter%03d.OUT'           % (it+1))
           if os.path.isfile('./EXPTNZ.OUT'):     os.system('mv ./EXPTNZ.OUT EXPTNZ_iter%03d.OUT'         % (it+1))
           if os.path.isfile('./EXPEQ_EXPEQ.IN'): os.system('mv ./EXPEQ_EXPEQ.IN EXPEQ_EXPEQ_iter%03d.IN' % (it+1))
           if os.path.isfile('./EQDSK_EXPEQ.IN'): os.system('mv ./EQDSK_EXPEQ.IN EQDSK_EXPEQ_iter%03d.IN' % (it+1))
           EQDSK_COCOS = glob('./EQDSK_COCOS_*')
           if EQDSK_COCOS:
                                                  os.system('cp '+EQDSK_COCOS[1]+' EQDSK_COCOS_iter%03d'  % (it+1))
                                                  os.system('mv '+EQDSK_COCOS[1]+' EQDSK')

           cheasepath = 'chease_iter%03d.h5' % (it+1)
           cheasedata = read_chease(cheasefpath=cheasepath)

           ITErr = (cheasedata['Itor']-ITEXP)/ITEXP
           print('Iter  = ', it+1)
           print('ITOR = ', cheasedata['Itor'])
           print('ITEXP = ', ITEXP)
           print('ITErr = ', abs(ITErr))
    
           it+=1
    
       pltValsKeys = pltVals.keys()
       if 'skipfigs' in pltValsKeys: skipfigs = pltVals['skipfigs']     
       else:                         skipfigs = 0  
       plot_chease('./',skipfigs=0)
    elif selection == 2:
       plot_chease('./',skipfigs=0)
    
    return 1
