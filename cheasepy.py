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


<<<<<<< HEAD
def getrecord(rec,datain):
    if type(datain) == dict:
       dataout = {}
       keys = datain.keys()
       if type(datain[keys[0]]) in [int,float,complex]:
          dataout = datain.copy()
       else:
          irec = min(irec,npy.size(datain[keys[0]]))
          for ikey in keys:
              dataout[ikey] = datain[ikey][irec]

    return dataout


def read_csv(csvfn):
    with open(csvfn, mode='r') as csvfid:
         csvdata = csv.DictReader(csvfid)
         recordid = 0
         csvdict  = {}
         for record in csvdata:
             headers = record.keys()
             for iheader in sorted(headers):
                 if recordid < len(headers):
                    csvdict[iheader] = []
                    recordid += 1
                 if len(record[iheader]) == 0:
                    csvdict[iheader].append(None)
                 else:
                    try:
                       csvdict[iheader].append(int(record[iheader]))
                    except ValueError:
                       csvdict[iheader].append(npy.float64(record[iheader]))
         for iheader in csvdict:
             csvdict[iheader] = npy.array(csvdict[iheader])
    return csvdict

def write_csv(csvfn,csvdict):
    headers   = sorted(csvdict.keys())
    nheaders  = npy.size(headers)
    for iheader in headers:
        if npy.size(csvdict[iheader])==0:
           del csvdict[iheader]
           headers.remove(iheader)
           nheaders = npy.size(headers)
        else:
           nrows = npy.size(csvdict[iheader])

    with open(csvfn, mode='w') as csvfid:
         writer = csv.DictWriter(csvfid, fieldnames=headers)
         writer.writeheader()
         for irow in range(nrows):
             record = {}
             for iheader in headers:
                 if   npy.size(csvdict[iheader]) < nrows: continue
                 if   type(csvdict[iheader][irow]) == int:
                      record.update({iheader:"%d" % csvdict[iheader][irow]})
                 elif type(csvdict[iheader][irow]) == float:
                      record.update({iheader:"%8.4e" % csvdict[iheader][irow]})
                 elif type(csvdict[iheader][irow]) == npy.float64:
                      record.update({iheader:"%8.4e" % csvdict[iheader][irow]})
                 elif type(csvdict[iheader][irow]) == npy.float128:
                      record.update({iheader:"%8.4e" % csvdict[iheader][irow]})
             writer.writerow(record)
    return 1


=======
>>>>>>> 7c0d900362339af9fc7ea7341e74170f46748f28
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

<<<<<<< HEAD
def read_chease(cheasefpath,setParam={},Normalized=False,**kwargs):
    if os.path.isfile(cheasefpath) == False:
       errorFunc = traceback.extract_stack(limit=2)[-2][3]
       errorLine = traceback.extract_stack(limit=2)[-2][1]
       errorFile = traceback.extract_stack(limit=2)[-2][2]
       errMSG    = 'Call %s line %5d in file %s Failed.\n'
       errMSG   += 'Fatal: file %s not found.'
       raise IOError(errMSG %(errorFunc,errorLine,errorFile,cheasefpath))

    extendCHI  = False
    reverseCHI = False

    hdffh = h5py.File(cheasefpath,'r')
    hdffhKeys = list(hdffh.keys())[0]
    datag = hdffh[hdffhKeys]

    CHEASEdata                = {}

    CHEASEdata['NPSI']        = datag.attrs["NPSI"]
    CHEASEdata['NCHI']        = datag.attrs["NCHI"]
    CHEASEdata['NRBOX']       = datag.attrs["NRBOX"]
    CHEASEdata['NZBOX']       = datag.attrs["NZBOX"]
    CHEASEdata['NBOUND']      = datag.attrs["NBOUND"]
    CHEASEdata['B0EXP']       = datag.attrs["B0EXP"]
    CHEASEdata['R0EXP']       = datag.attrs["R0EXP"]

    CHEASEdata['PSI']         = npy.array(datag["grid"]["PSI"])
    CHEASEdata['rhopsi_SI']   = npy.sqrt(CHEASEdata['PSI']/CHEASEdata['PSI'][-1])
    CHEASEdata['rhotor_SI']   = npy.array(datag["var1d"]["rho_tor"])

    CHEASEdata['CHI']         = npy.array(datag["grid"]["CHI"])
    if CHEASEdata['CHI'][0]>CHEASEdata['CHI'][1]:
       CHEASEdata['CHI']      = CHEASEdata["CHI"][::-1]
       reverseCHI             = True
    if min(abs(CHEASEdata['CHI']-2.0*npy.pi))>=min(npy.diff(CHEASEdata['CHI'])):
       CHEASEdata['CHI']      = npy.append(CHEASEdata['CHI'],2.0*npy.pi)
       extendCHI              = True

    CHEASEdata['Jbs']         = npy.array(datag["var1d"]["jbsBav"])
    CHEASEdata['Zeff']        = npy.array(datag["var1d"]["zeff"])
    CHEASEdata['kappa']       = npy.array(datag["var1d"]["kappa"])
    CHEASEdata['shear']       = npy.array(datag["var1d"]["shear"])
    CHEASEdata['signeo']      = npy.array(datag["var1d"]["signeo"])
    CHEASEdata['pprime']      = 2.0*npy.pi*npy.array(datag["var1d"]["dpdpsi"])

    CHEASEdata['q']           = npy.array(datag["var1d"]["q"])
    CHEASEdata['R_av']        = npy.array(datag["var1d"]["R_av"])
    CHEASEdata['ageom']       = npy.array(datag["var1d"]["ageom"])
    CHEASEdata['Rgeom']       = npy.array(datag["var1d"]["Rgeom"])
    CHEASEdata['Volume']      = npy.array(datag["var1d"]["Volume"])
    CHEASEdata['pressure']    = npy.array(datag["var1d"]["p"])
    CHEASEdata['GDPSI_av']    = npy.array(datag["var1d"]["GDPSI_av"])
    CHEASEdata['radius_av']   = npy.array(datag["var1d"]["radius_av"])

    CHEASEdata['Te']          = npy.array(datag["var1d"]["Te"])
    CHEASEdata['Ti']          = npy.array(datag["var1d"]["Ti"])
    CHEASEdata['ne']          = npy.array(datag["var1d"]["ne"])
    CHEASEdata['ni']          = npy.array(datag["var1d"]["ni"])

    CHEASEdata['Zi']          = 1.0
    CHEASEdata['Zz']          = 6.0
    CHEASEdata['nz']          = CHEASEdata['Zeff']*CHEASEdata['ne']
    CHEASEdata['nz']         -= CHEASEdata['ni']*CHEASEdata['Zi']**2
    CHEASEdata['nz']         /= CHEASEdata['Zz']**2

    CHEASEdata['TePrime']     = npy.array(datag["var1d"]["dTedpsi"])
    CHEASEdata['TiPrime']     = npy.array(datag["var1d"]["dTidpsi"])
    CHEASEdata['nePrime']     = npy.array(datag["var1d"]["dnedpsi"])
    CHEASEdata['niPrime']     = npy.array(datag["var1d"]["dnidpsi"])

    CHEASEdata['f']           = npy.array(datag["var1d"]["f"])
    CHEASEdata['ffprime']     = 2.0*npy.pi*npy.array(datag["var1d"]["fdfdpsi"])
    CHEASEdata['fprime']      = CHEASEdata['ffprime']/CHEASEdata['f']

    CHEASEdata['rmesh']       = npy.array(datag["var1d"]["rmesh"])
    CHEASEdata['zmesh']       = npy.array(datag["var1d"]["zmesh"])
    CHEASEdata['rbound']      = npy.array(datag["var1d"]["rboundplasma"])
    CHEASEdata['zbound']      = npy.array(datag["var1d"]["zboundplasma"])
    CHEASEdata['delta_upper'] = npy.array(datag["var1d"]["delta_upper"])
    CHEASEdata['delta_lower'] = npy.array(datag["var1d"]["delta_lower"])

    CHEASEdata['dpdpsi']      = npy.array(datag["var1d"]["dpdpsi"])
    CHEASEdata['dqdpsi']      = npy.array(datag["var1d"]["dqdpsi"])
    CHEASEdata['dVdpsi']      = npy.array(datag["var1d"]["dVdpsi"])
    CHEASEdata['d2qdpsi2']    = npy.array(datag["var1d"]["d2qdpsi2"])
    CHEASEdata['dsheardpsi']  = npy.array(datag["var1d"]["dsheardpsi"])
    CHEASEdata['dpsidrhotor'] = npy.array(datag["var1d"]["dpsidrhotor"])


   #THE DIMENSION OF ALL THE FOLLOWING QUNATITIES ARE (NCHI,NPSI)
    CHEASEdata['R']           = npy.array(datag["var2d"]["R"])
    CHEASEdata['Z']           = npy.array(datag["var2d"]["Z"])
    CHEASEdata['B']           = npy.array(datag["var2d"]["B"])
    CHEASEdata['J']           = npy.array(datag["var2d"]["Jacobian"])
    CHEASEdata['g11']         = npy.array(datag["var2d"]["g11"])
    CHEASEdata['g22']         = npy.array(datag["var2d"]["g22"])
    CHEASEdata['g33']         = npy.array(datag["var2d"]["g33"])
    CHEASEdata['dBdpsi']      = npy.array(datag["var2d"]["dBdpsi"])
    CHEASEdata['dBdchi']      = npy.array(datag["var2d"]["dBdchi"])
    CHEASEdata['dChidZ']      = npy.array(datag["var2d"]["dChidZ"])
    CHEASEdata['dPsidZ']      = npy.array(datag["var2d"]["dPsidZ"])
    CHEASEdata['dChidR']      = npy.array(datag["var2d"]["dChidR"])
    CHEASEdata['dPsidR']      = npy.array(datag["var2d"]["dPsidR"])

    if extendCHI:
       CHEASEdata['R']      = npy.vstack((CHEASEdata['R'],CHEASEdata['R'][0,:]))
       CHEASEdata['Z']      = npy.vstack((CHEASEdata['Z'],CHEASEdata['Z'][0,:]))
       CHEASEdata['B']      = npy.vstack((CHEASEdata['B'],CHEASEdata['B'][0,:]))
       CHEASEdata['J']      = npy.vstack((CHEASEdata['J'],CHEASEdata['J'][0,:]))
       CHEASEdata['g11']    = npy.vstack((CHEASEdata['g11'],CHEASEdata['g11'][0,:]))
       CHEASEdata['g22']    = npy.vstack((CHEASEdata['g22'],CHEASEdata['g22'][0,:]))
       CHEASEdata['g33']    = npy.vstack((CHEASEdata['g33'],CHEASEdata['g33'][0,:]))
       CHEASEdata['dBdpsi'] = npy.vstack((CHEASEdata['dBdpsi'],CHEASEdata['dBdpsi'][0,:]))
       CHEASEdata['dBdchi'] = npy.vstack((CHEASEdata['dBdchi'],CHEASEdata['dBdchi'][0,:]))
       CHEASEdata['dChidZ'] = npy.vstack((CHEASEdata['dChidZ'],CHEASEdata['dChidZ'][0,:]))
       CHEASEdata['dPsidZ'] = npy.vstack((CHEASEdata['dPsidZ'],CHEASEdata['dPsidZ'][0,:]))
       CHEASEdata['dChidR'] = npy.vstack((CHEASEdata['dChidR'],CHEASEdata['dChidR'][0,:]))
       CHEASEdata['dPsidR'] = npy.vstack((CHEASEdata['dPsidR'],CHEASEdata['dPsidR'][0,:]))

   #THE DIMENSION OF ALL THE FOLLOWING QUNATITIES ARE (NRBOX,NZBOX)
    CHEASEdata['psiRZ']    = npy.array(datag["var2d"]["psiRZ"])
    CHEASEdata['chiRZ']    = npy.array(datag["var2d"]["chiRZ"])

    CHEASEdata['C0']       = npy.trapz(y=CHEASEdata['J']/CHEASEdata['R'],                    x=CHEASEdata['CHI'],axis=0)
    CHEASEdata['C1']       = npy.trapz(y=CHEASEdata['J'],                                    x=CHEASEdata['CHI'],axis=0)
    CHEASEdata['C2']       = npy.trapz(y=CHEASEdata['J']/CHEASEdata['R']**2,                 x=CHEASEdata['CHI'],axis=0)
    CHEASEdata['C3']       = npy.trapz(y=CHEASEdata['J']*CHEASEdata['g11']*CHEASEdata['g33'],x=CHEASEdata['CHI'],axis=0)

    CHEASEdata['y1']       = 1.0+CHEASEdata['C3']/CHEASEdata['C2']/CHEASEdata['f']**2/4.0/npy.pi**2

    CHEASEdata['<B2>']     = npy.trapz(y=CHEASEdata['J']*CHEASEdata['B']**2,x=CHEASEdata['CHI'],axis=0)/CHEASEdata['C1']
    CHEASEdata['<JdotB>']  =-CHEASEdata['f']*CHEASEdata['pprime']-CHEASEdata['fprime']*CHEASEdata['<B2>']/mu0
    CHEASEdata['Jprl']     = CHEASEdata['<JdotB>']/CHEASEdata['B0EXP']

    CHEASEdata['<T/R2>']   = npy.trapz(y=CHEASEdata['J']*CHEASEdata['f']*CHEASEdata['g33'],x=CHEASEdata['CHI'],axis=0)/CHEASEdata['C1']
    CHEASEdata['Iprl']     = CHEASEdata['R0EXP']*CHEASEdata['<JdotB>']/CHEASEdata['<T/R2>']

    CHEASEdata['Istr']     =-((CHEASEdata['C2']/CHEASEdata['C0'])*(CHEASEdata['ffprime']/mu0))
    CHEASEdata['Istr']    +=-((CHEASEdata['C1']/CHEASEdata['C0'])*CHEASEdata['pprime'])
    CHEASEdata['Istr']    *= CHEASEdata['R0EXP']**2

    CHEASEdata['Jphi']     =-(CHEASEdata['R']*CHEASEdata['pprime'])-(CHEASEdata['ffprime']/(mu0*CHEASEdata['R']))

    CHEASEdata['Jtor']     = npy.trapz(y=CHEASEdata['Jphi']*CHEASEdata['J']/CHEASEdata['R'],x=CHEASEdata['CHI'],axis=0)
    CHEASEdata['Itor']     = npy.trapz(y=CHEASEdata['Jtor'],x=CHEASEdata['PSI'],axis=0)


    B2AVG = npy.trapz(y=CHEASEdata['<B2>'],x=CHEASEdata['PSI'],axis=0)/CHEASEdata['Volume'][-1]
    PTAVG = npy.trapz(y=CHEASEdata['pressure'],x=CHEASEdata['PSI'],axis=0)/CHEASEdata['Volume'][-1]
    CHEASEdata['beta_T']   = PTAVG/(B2AVG/2.0/mu0)
    CHEASEdata['beta_N']   = 100.0*CHEASEdata['beta_T']/((CHEASEdata['Itor']/1.0e6)/npy.max(CHEASEdata['ageom'])/CHEASEdata['B0EXP'])

    CHEASEdata['Ibs']      = CHEASEdata['R0EXP']*CHEASEdata['Jbs']/CHEASEdata['<T/R2>']
    CHEASEdata['Iohmic']   = CHEASEdata['Iprl']-CHEASEdata['Ibs']
    CHEASEdata['Johmic']   = CHEASEdata['Jprl']-CHEASEdata['Jbs']

    extendPSI    = npy.linspace(CHEASEdata['PSI'][0],CHEASEdata['PSI'][-1],10*npy.size(CHEASEdata['PSI']))
    extendPHI    = npy.empty_like(extendPSI)
    extendPHI[0] = 0.0
    qfunc        = CubicSpline(CHEASEdata['PSI'],CHEASEdata['q'])
    for i in range(1,npy.size(extendPSI)):
        x           = extendPSI[:i+1]
        y           = qfunc(x)
        extendPHI[i]= npy.trapz(y,x)

    CHEASEdata['PHI'] = npy.empty_like(CHEASEdata['PSI'])
    phifunc           = CubicSpline(extendPSI,extendPHI)
    for i in range(npy.size(CHEASEdata['PSI'])):
        CHEASEdata['PHI'][i] = phifunc(CHEASEdata['PSI'][i])

    CHEASEdata['PHIN']     = (CHEASEdata['PHI']-CHEASEdata['PHI'][0])/(CHEASEdata['PHI'][-1]-CHEASEdata['PHI'][0])
    CHEASEdata['PSIN']     = (CHEASEdata['PSI']-CHEASEdata['PSI'][0])/(CHEASEdata['PSI'][-1]-CHEASEdata['PSI'][0])
    CHEASEdata['CHIN']     = (CHEASEdata['CHI']-CHEASEdata['CHI'][0])/(CHEASEdata['CHI'][-1]-CHEASEdata['CHI'][0])

   #CHEASEdata['rhopsi']   = (CHEASEdata['rhopsi_SI']-CHEASEdata['rhopsi_SI'][0])/(CHEASEdata['rhopsi_SI'][-1]-CHEASEdata['rhopsi_SI'][0])
   #CHEASEdata['rhotor']   = (CHEASEdata['rhotor_SI']-CHEASEdata['rhotor_SI'][0])/(CHEASEdata['rhotor_SI'][-1]-CHEASEdata['rhotor_SI'][0])
    CHEASEdata['rhopsi']   = npy.sqrt(CHEASEdata['PSIN'])
    CHEASEdata['rhotor']   = npy.sqrt(CHEASEdata['PHIN'])

    #Implementing Interpolation to EQDSK Grid
    rhopsiflag = False; rhotorflag = False
    if 'nrhomesh' in setParam:
       if   setParam['nrhomesh'] in [0,'rhopsi']: rhopsiflag = True
       elif setParam['nrhomesh'] in [1,'rhotor']: rhotorflag = True
    else:                                         rhopsiflag = True

    eqdskflag    = False
    expeqflag    = False
    interpflag   = False
    importedflag = False
    for key,value in kwargs.items():
        if   key in ['eqdsk','eqdskdata','eqdskfpath']:
             if    type(value)==str and os.path.isfile(value.strip()):
                   eqdskdata = read_eqdsk(eqdskfpath=value.strip())
             else: raise IOError('%s file not found!' % value.strip())
             if 'rhopsi' in eqdskdata: rhopsi = eqdskdata['rhopsi'][:]
             if 'rhotor' in eqdskdata: rhotor = eqdskdata['rhotor'][:]
             if 'PSIN'   in eqdskdata: psi    = eqdskdata['PSIN'][:];  interpflag = True
             if 'PHIN'   in eqdskdata: phi    = eqdskdata['PHIN'][:];  interpflag = True
             eqdskflag = True
        elif key in ['imported','external','other']:
             imported = value.copy()
             if 'rhopsi' in imported: rhopsi = imported['rhopsi'][:]
             if 'rhotor' in imported: rhotor = imported['rhotor'][:]
             if 'PSIN'   in imported: psi = imported['PSIN'][:];       interpflag = True
             else:                    psi = importeddata['rhopsi']**2; interpflag = True
             if 'PHIN'   in imported: phi = imported['PHIN'][:];       interpflag = True
             else:                    phi = importeddata['rhotor']**2; interpflag = True
             importedflag = True

    if interpflag:
       if   rhopsiflag:
            CHEASEdata['q']        = interp(CHEASEdata['PSIN'],CHEASEdata['q'],psi)
            CHEASEdata['f']        = interp(CHEASEdata['PSIN'],CHEASEdata['f'],psi)
            CHEASEdata['Te']       = interp(CHEASEdata['PSIN'],CHEASEdata['Te'],psi)
            CHEASEdata['Ti']       = interp(CHEASEdata['PSIN'],CHEASEdata['Ti'],psi)
            CHEASEdata['ne']       = interp(CHEASEdata['PSIN'],CHEASEdata['ne'],psi)
            CHEASEdata['ni']       = interp(CHEASEdata['PSIN'],CHEASEdata['ni'],psi)
            CHEASEdata['nz']       = interp(CHEASEdata['PSIN'],CHEASEdata['nz'],psi)
            CHEASEdata['Ibs']      = interp(CHEASEdata['PSIN'],CHEASEdata['Ibs'],psi)
            CHEASEdata['Jbs']      = interp(CHEASEdata['PSIN'],CHEASEdata['Jbs'],psi)
            CHEASEdata['Zeff']     = interp(CHEASEdata['PSIN'],CHEASEdata['Zeff'],psi)
            CHEASEdata['Istr']     = interp(CHEASEdata['PSIN'],CHEASEdata['Istr'],psi)
            CHEASEdata['Iprl']     = interp(CHEASEdata['PSIN'],CHEASEdata['Iprl'],psi)
            CHEASEdata['Jprl']     = interp(CHEASEdata['PSIN'],CHEASEdata['Jprl'],psi)
            CHEASEdata['kappa']    = interp(CHEASEdata['PSIN'],CHEASEdata['kappa'],psi)
            CHEASEdata['shear']    = interp(CHEASEdata['PSIN'],CHEASEdata['shear'],psi)
            CHEASEdata['signeo']   = interp(CHEASEdata['PSIN'],CHEASEdata['signeo'],psi)
            CHEASEdata['fprime']   = interp(CHEASEdata['PSIN'],CHEASEdata['fprime'],psi)
            CHEASEdata['Iohmic']   = interp(CHEASEdata['PSIN'],CHEASEdata['Iohmic'],psi)
            CHEASEdata['Johmic']   = interp(CHEASEdata['PSIN'],CHEASEdata['Johmic'],psi)
            CHEASEdata['pprime']   = interp(CHEASEdata['PSIN'],CHEASEdata['pprime'],psi)
            CHEASEdata['ffprime']  = interp(CHEASEdata['PSIN'],CHEASEdata['ffprime'],psi)
            CHEASEdata['pressure'] = interp(CHEASEdata['PSIN'],CHEASEdata['pressure'],psi)

            CHEASEdata['PSIN']     = psi[:]
            CHEASEdata['PHIN']     = phi[:]
            CHEASEdata['rhopsi']   = rhopsi[:]
            CHEASEdata['rhotor']   = rhotor[:]

       elif rhotorflag:
            CHEASEdata['q']        = interp(CHEASEdata['PSIN'],CHEASEdata['q'],psi,phi,phi)
            CHEASEdata['f']        = interp(CHEASEdata['PSIN'],CHEASEdata['f'],psi,phi,phi)
            CHEASEdata['Te']       = interp(CHEASEdata['PSIN'],CHEASEdata['Te'],psi,phi,phi)
            CHEASEdata['Ti']       = interp(CHEASEdata['PSIN'],CHEASEdata['Ti'],psi,phi,phi)
            CHEASEdata['ne']       = interp(CHEASEdata['PSIN'],CHEASEdata['ne'],psi,phi,phi)
            CHEASEdata['ni']       = interp(CHEASEdata['PSIN'],CHEASEdata['ni'],psi,phi,phi)
            CHEASEdata['nz']       = interp(CHEASEdata['PSIN'],CHEASEdata['nz'],psi,phi,phi)
            CHEASEdata['Ibs']      = interp(CHEASEdata['PSIN'],CHEASEdata['Ibs'],psi,phi,phi)
            CHEASEdata['Jbs']      = interp(CHEASEdata['PSIN'],CHEASEdata['Jbs'],psi,phi,phi)
            CHEASEdata['Zeff']     = interp(CHEASEdata['PSIN'],CHEASEdata['Zeff'],psi,phi,phi)
            CHEASEdata['Istr']     = interp(CHEASEdata['PSIN'],CHEASEdata['Istr'],psi,phi,phi)
            CHEASEdata['Iprl']     = interp(CHEASEdata['PSIN'],CHEASEdata['Iprl'],psi,phi,phi)
            CHEASEdata['Jprl']     = interp(CHEASEdata['PSIN'],CHEASEdata['Jprl'],psi,phi,phi)
            CHEASEdata['kappa']    = interp(CHEASEdata['PSIN'],CHEASEdata['kappa'],psi,phi,phi)
            CHEASEdata['shear']    = interp(CHEASEdata['PSIN'],CHEASEdata['shear'],psi,phi,phi)
            CHEASEdata['signeo']   = interp(CHEASEdata['PSIN'],CHEASEdata['signeo'],psi,phi,phi)
            CHEASEdata['fprime']   = interp(CHEASEdata['PSIN'],CHEASEdata['fprime'],psi,phi,phi)
            CHEASEdata['Iohmic']   = interp(CHEASEdata['PSIN'],CHEASEdata['Iohmic'],psi,phi,phi)
            CHEASEdata['Johmic']   = interp(CHEASEdata['PSIN'],CHEASEdata['Johmic'],psi,phi,phi)
            CHEASEdata['pprime']   = interp(CHEASEdata['PSIN'],CHEASEdata['pprime'],psi,phi,phi)
            CHEASEdata['ffprime']  = interp(CHEASEdata['PSIN'],CHEASEdata['ffprime'],psi,phi,phi)
            CHEASEdata['pressure'] = interp(CHEASEdata['PSIN'],CHEASEdata['pressure'],psi,phi,phi)

            CHEASEdata['PSIN']     = psi[:]
            CHEASEdata['PHIN']     = phi[:]
            CHEASEdata['rhotor']   = rhopsi[:]
            CHEASEdata['rhopsi']   = rhotor[:]

    if Normalized:
       CHEASEdata['R']        = CHEASEdata['R']/CHEASEdata['R0EXP']
       CHEASEdata['Z']        = CHEASEdata['Z']/CHEASEdata['R0EXP']
       CHEASEdata['B']        = CHEASEdata['B']/CHEASEdata['B0EXP']
       CHEASEdata['Ibs']      = CHEASEdata['Ibs']*mu0/CHEASEdata['R0EXP']/CHEASEdata['B0EXP']
       CHEASEdata['Jbs']      = CHEASEdata['Jbs'] *mu0*CHEASEdata['R0EXP']/CHEASEdata['B0EXP']
       CHEASEdata['Iprl']     = CHEASEdata['Iprl']*mu0/CHEASEdata['R0EXP']/CHEASEdata['B0EXP']
       CHEASEdata['Jprl']     = CHEASEdata['Jprl']*mu0*CHEASEdata['R0EXP']/CHEASEdata['B0EXP']
       CHEASEdata['Istr']     = CHEASEdata['Istr']*mu0/CHEASEdata['R0EXP']/CHEASEdata['B0EXP']
       CHEASEdata['Jphi']     = CHEASEdata['Jphi']*mu0*CHEASEdata['R0EXP']/CHEASEdata['B0EXP']
       CHEASEdata['Itor']     = CHEASEdata['Itor']*mu0/CHEASEdata['R0EXP']/CHEASEdata['B0EXP']
       CHEASEdata['Iohmic']   = CHEASEdata['Iprl']-CHEASEdata['Ibs']
       CHEASEdata['Johmic']   = CHEASEdata['Jprl']-CHEASEdata['Jbs']
       CHEASEdata['pprime']   = CHEASEdata['pprime']*mu0*CHEASEdata['R0EXP']**2/CHEASEdata['B0EXP']
       CHEASEdata['ffprime']  = CHEASEdata['ffprime']/CHEASEdata['B0EXP']
       CHEASEdata['pressure'] = CHEASEdata['pressure']*mu0/CHEASEdata['B0EXP']**2
 
    return CHEASEdata

def read_eqdsk(eqdskfpath,setParam={},Normalized=False,**kwargs):
    if os.path.isfile(eqdskfpath) == False:
       errorFunc = traceback.extract_stack(limit=2)[-2][3]
       errorLine = traceback.extract_stack(limit=2)[-2][1]
       errorFile = traceback.extract_stack(limit=2)[-2][2]
       errMSG    = 'Call %s line %5d in file %s Failed.\n'
       errMSG   += 'Fatal: file %s not found.'
       raise IOError(errMSG %(errorFunc,errorLine,errorFile,eqdskfpath))
    
    rhopsiflag = False; rhotorflag = False
    if 'nrhomesh' in setParam:
        if   setParam['nrhomesh'] in [0,'rhopsi']: rhopsiflag = True
        elif setParam['nrhomesh'] in [1,'rhotor']: rhotorflag = True
    else:                                          rhopsiflag = True

    interpflag   = False
    cheaseflag   = False
    importedflag = False
    for key,value in kwargs.items():
        if   key in ['chease','cheasedata','cheasefpath']:
             if    type(value)==str and os.path.isfile(value.strip()):
                   cheasedata = read_chease(cheasefpath=value.strip())
             else: raise IOError('%s file not found!' % value.strip())
             if 'rhopsi' in cheasedata: rhopsi = cheasedata['rhopsi'][:]
             if 'rhotor' in cheasedata: rhotor = cheasedata['rhotor'][:]
             if 'PSIN'   in cheasedata: psi    = cheasedata['PSIN'][:]; interpflag = True
             if 'PHIN'   in cheasedata: phi    = cheasedata['PHIN'][:]; interpflag = True
             cheaseflag = True
        elif key in ['imported','external','other']:
             imported = value.copy()
             if 'rhopsi' in imported: rhopsi = imported['rhopsi'][:]
             if 'rhotor' in imported: rhotor = imported['rhotor'][:]
             if 'PSIN'   in imported: psi    = imported['PSIN'];      interpflag = True
             else:                    psi    = imported['rhopsi']**2; interpflag = True
             if 'PHIN'   in imported: phi    = imported['PHIN'];      interpflag = True
             else:                    phi    = imported['rhotor']**2; interpflag = True
             importedflag = True

    EQDSKdata = read_efit_file(eqdskfpath)

    if interpflag:
       if   rhopsiflag:
            EQDSKdata['q']        = interp(EQDSKdata['PSIN'],EQDSKdata['qpsi'],psi) 
            EQDSKdata['f']        = interp(EQDSKdata['PSIN'],EQDSKdata['fpol'],psi)
            EQDSKdata['pprime']   = interp(EQDSKdata['PSIN'],EQDSKdata['pprime'],psi)
            EQDSKdata['ffprime']  = interp(EQDSKdata['PSIN'],EQDSKdata['ffprime'],psi)
            EQDSKdata['pressure'] = interp(EQDSKdata['PSIN'],EQDSKdata['pressure'],psi) 
            EQDSKdata['rhopsi']   = rhopsi[:]
            EQDSKdata['rhotor']   = rhotor[:]
       elif rhotorflag:
            EQDSKdata['q']        = interp(EQDSKdata['PSIN'],EQDSKdata['qpsi'],psi,phi,phi) 
            EQDSKdata['f']        = interp(EQDSKdata['PSIN'],EQDSKdata['fpol'],psi,phi,phi)
            EQDSKdata['pprime']   = interp(EQDSKdata['PSIN'],EQDSKdata['pprime'],psi,phi,phi)
            EQDSKdata['ffprime']  = interp(EQDSKdata['PSIN'],EQDSKdata['ffprime'],psi,phi,phi)
            EQDSKdata['pressure'] = interp(EQDSKdata['PSIN'],EQDSKdata['pressure'],psi,phi,phi)
            EQDSKdata['rhopsi']   = rhopsi[:]
            EQDSKdata['rhotor']   = rhotor[:]
    else:
            EQDSKdata['q']        = EQDSKdata['qpsi'][:]
            EQDSKdata['f']        = EQDSKdata['fpol'][:]

    if Normalized:
       EQDSKdata['RCTR']          = abs(EQDSKdata['RCTR'])
       EQDSKdata['BCTR']          = abs(EQDSKdata['BCTR'])
       EQDSKdata['rbound']        = EQDSKdata['rbound']/EQDSKdata['RCTR']
       EQDSKdata['zbound']        = EQDSKdata['zbound']/EQDSKdata['RCTR']
       if 'rlimit' in EQDSKdata:
          EQDSKdata['rlimit']        = EQDSKdata['rlimit']/EQDSKdata['RCTR']
       if 'zlimit' in EQDSKdata:
          EQDSKdata['zlimit']        = EQDSKdata['zlimit']/EQDSKdata['RCTR']
       EQDSKdata['pressure']      = EQDSKdata['pressure']*mu0/EQDSKdata['BCTR']**2
       EQDSKdata['pprime']        = EQDSKdata['pprime']*mu0*EQDSKdata['RCTR']**2/EQDSKdata['BCTR']
       EQDSKdata['ffprime']       = EQDSKdata['ffprime']/EQDSKdata['BCTR']

    return EQDSKdata

def read_expeq(expeqfpath,setParam={},**kwargs):
    if os.path.isfile(expeqfpath) == False:
       errorFunc = traceback.extract_stack(limit=2)[-2][3]
       errorLine = traceback.extract_stack(limit=2)[-2][1]
       errorFile = traceback.extract_stack(limit=2)[-2][2]
       errMSG    = 'Call %s line %5d in file %s Failed.\n'
       errMSG   += 'Fatal: file %s not found.'
       raise IOError(errMSG %(errorFunc,errorLine,errorFile,expeqfpath))


    rhopsiflag = False; rhotorflag = False
    if 'nrhomesh' in setParam:
        if   setParam['nrhomesh'] in [0,'rhopsi']: rhopsiflag = True
        elif setParam['nrhomesh'] in [1,'rhotor']: rhotorflag = True
    else:                                          rhopsiflag = True

    eqdskflag    = False
    cheaseflag   = False
    interpflag   = False
    importedflag = False
    for key,value in kwargs.items():
        if   key in ['chease','cheasedata','cheasefpath']:
             if    type(value)==str and os.path.isfile(value.strip()):
                   cheasedata = read_chease(cheasefpath=value.strip())
             else: raise IOError('%s file not found!' % value.strip())
             if 'rhopsi' in cheasedata: rhopsi = cheasedata['rhopsi'][:]; interpflag = True
             if 'rhotor' in cheasedata: rhotor = cheasedata['rhotor'][:]; interpflag = True
             cheaseflag = True
        elif key in ['eqdsk','eqdskdata','eqdskfpath']:
             if    type(value)==str and os.path.isfile(value.strip()):
                   eqdskdata = read_eqdsk(eqdskfpath=value.strip())
             else: raise IOError('%s file not found!' % value.strip())
             if 'rhopsi' in eqdskdata: rhopsi = eqdskdata['rhopsi'][:]; interpflag = True
             if 'rhotor' in eqdskdata: rhotor = eqdskdata['rhotor'][:]; interpflag = True
             eqdskflag = True
        elif key in ['imported','external','other']:
             imported = value.copy()
             if 'rhopsi' in imported: rhopsi = imported['rhopsi'][:]; interpflag = True
             if 'rhotor' in imported: rhotor = imported['rhotor'][:]; interpflag = True
             importedflag = True

    ofh = open(expeqfpath,'r')
    EXPEQOUT = ofh.readlines()
    ofh.close()

    EXPEQdata                  = {} 
    EXPEQdata['aspect']        = float(EXPEQOUT[0])
    EXPEQdata['zgeom']         = float(EXPEQOUT[1])
    EXPEQdata['pedge']         = float(EXPEQOUT[2])
    nRZmesh                    =   int(EXPEQOUT[3])
    EXPEQdata['nRZmesh']       = nRZmesh
    EXPEQdata['rbound']        = npy.array([irec.split()[0] for irec in EXPEQOUT[4:nRZmesh+4]],dtype=float)
    EXPEQdata['zbound']        = npy.array([irec.split()[1] for irec in EXPEQOUT[4:nRZmesh+4]],dtype=float)
    
    nrhomesh                   = int(EXPEQOUT[nRZmesh+4].split()[0])
    EXPEQdata['nrhomesh']      = nrhomesh
    EXPEQdata['nppfun']        = int(EXPEQOUT[nRZmesh+4].split()[1])
    EXPEQdata['nsttp']         = int(EXPEQOUT[nRZmesh+5].split()[0])
    EXPEQdata['nrhotype']      = int(EXPEQOUT[nRZmesh+5].split()[1])

    if   EXPEQdata['nrhotype']  == 0:
         EXPEQdata['rhopsi']    = npy.array(EXPEQOUT[nRZmesh+6+0*nrhomesh:nRZmesh+6+1*nrhomesh],dtype=float)
    elif EXPEQdata['nrhotype']  == 1:
         EXPEQdata['rhotor']    = npy.array(EXPEQOUT[nRZmesh+6+0*nrhomesh:nRZmesh+6+1*nrhomesh],dtype=float)
    
    if   EXPEQdata['nppfun']    == 4:
         EXPEQdata['pprime']    = npy.array(EXPEQOUT[nRZmesh+6+1*nrhomesh:nRZmesh+6+2*nrhomesh],dtype=float)
    elif EXPEQdata['nppfun']    == 8:
         EXPEQdata['pressure']  = npy.array(EXPEQOUT[nRZmesh+6+1*nrhomesh:nRZmesh+6+2*nrhomesh],dtype=float)
  
    if   EXPEQdata['nsttp']     == 1:
         EXPEQdata['ffprime']   = npy.array(EXPEQOUT[nRZmesh+6+2*nrhomesh:nRZmesh+6+3*nrhomesh],dtype=float)
    elif EXPEQdata['nsttp']     == 2:
         EXPEQdata['Istr']      = npy.array(EXPEQOUT[nRZmesh+6+2*nrhomesh:nRZmesh+6+3*nrhomesh],dtype=float)
    elif EXPEQdata['nsttp']     == 3:
         EXPEQdata['Iprl']      = npy.array(EXPEQOUT[nRZmesh+6+2*nrhomesh:nRZmesh+6+3*nrhomesh],dtype=float)
    elif EXPEQdata['nsttp']     == 4:
         EXPEQdata['Jprl']      = npy.array(EXPEQOUT[nRZmesh+6+2*nrhomesh:nRZmesh+6+3*nrhomesh],dtype=float)
    elif EXPEQdata['nsttp']     == 5:
         EXPEQdata['q']         = npy.array(EXPEQOUT[nRZmesh+6+2*nrhomesh:nRZmesh+6+3*nrhomesh],dtype=float)

    if interpflag:
       if   rhopsiflag and EXPEQdata['nrhotype']==0:
            if   EXPEQdata['nppfun']    == 4:
                 EXPEQdata['pprime']    = interp(EXPEQdata['rhopsi'],EXPEQdata['pprime'],rhopsi)
            elif EXPEQdata['nppfun']    == 8:
                 EXPEQdata['pressure']  = interp(EXPEQdata['rhopsi'],EXPEQdata['pressure'],rhopsi)
            if   EXPEQdata['nsttp']     == 1:
                 EXPEQdata['ffprime']   = interp(EXPEQdata['rhopsi'],EXPEQdata['ffprime'],rhopsi)
            elif EXPEQdata['nsttp']     == 2:
                 EXPEQdata['Istr']      = interp(EXPEQdata['rhopsi'],EXPEQdata['Istr'],rhopsi)
            elif EXPEQdata['nsttp']     == 3:
                 EXPEQdata['Iprl']      = interp(EXPEQdata['rhopsi'],EXPEQdata['Iprl'],rhopsi)
            elif EXPEQdata['nsttp']     == 4:
                 EXPEQdata['Jprl']      = interp(EXPEQdata['rhopsi'],EXPEQdata['Jprl'],rhopsi)
            elif EXPEQdata['nsttp']     == 5:
                 EXPEQdata['q']         = interp(EXPEQdata['rhopsi'],EXPEQdata['q'],rhopsi)
            EXPEQdata['rhopsi']         = rhopsi[:]
            EXPEQdata['rhotor']         = rhotor[:]
       elif rhotorflag and EXPEQdata['nrhotype']==1:
            if   EXPEQdata['nppfun']    == 4:
                 EXPEQdata['pprime']    = interp(EXPEQdata['rhotor'],EXPEQdata['pprime'],rhotor)
            elif EXPEQdata['nppfun']    == 8:
                 EXPEQdata['pressure']  = interp(EXPEQdata['rhotor'],EXPEQdata['pressure'],rhotor)
            if   EXPEQdata['nsttp']     == 1:
                 EXPEQdata['ffprime']   = interp(EXPEQdata['rhotor'],EXPEQdata['ffprime'],rhotor)
            elif EXPEQdata['nsttp']     == 2:
                 EXPEQdata['Istr']      = interp(EXPEQdata['rhotor'],EXPEQdata['Istr'],rhotor)
            elif EXPEQdata['nsttp']     == 3:
                 EXPEQdata['Iprl']      = interp(EXPEQdata['rhotor'],EXPEQdata['Iprl'],rhotor)
            elif EXPEQdata['nsttp']     == 4:
                 EXPEQdata['Jprl']      = interp(EXPEQdata['rhotor'],EXPEQdata['Jprl'],rhotor)
            elif EXPEQdata['nsttp']     == 5:
                 EXPEQdata['q']         = interp(EXPEQdata['rhotor'],EXPEQdata['q'],rhotor)
            EXPEQdata['rhopsi']         = rhopsi[:]
            EXPEQdata['rhotor']         = rhotor[:]
       elif rhopsiflag and EXPEQdata['nrhotype']==1:
            if   EXPEQdata['nppfun']    == 4:
                 EXPEQdata['pprime']    = interp(EXPEQdata['rhotor'],EXPEQdata['pprime'],rhotor,rhopsi,rhopsi)
            elif EXPEQdata['nppfun']    == 8:
                 EXPEQdata['pressure']  = interp(EXPEQdata['rhotor'],EXPEQdata['pressure'],rhotor,rhopsi,rhopsi)
            if   EXPEQdata['nsttp']     == 1:
                 EXPEQdata['ffprime']   = interp(EXPEQdata['rhotor'],EXPEQdata['ffprime'],rhotor,rhopsi,rhopsi)
            elif EXPEQdata['nsttp']     == 2:
                 EXPEQdata['Istr']      = interp(EXPEQdata['rhotor'],EXPEQdata['Istr'],rhotor,rhopsi,rhopsi)
            elif EXPEQdata['nsttp']     == 3:
                 EXPEQdata['Iprl']      = interp(EXPEQdata['rhotor'],EXPEQdata['Iprl'],rhotor,rhopsi,rhopsi)
            elif EXPEQdata['nsttp']     == 4:
                 EXPEQdata['Jprl']      = interp(EXPEQdata['rhotor'],EXPEQdata['Jprl'],rhotor,rhopsi,rhopsi)
            elif EXPEQdata['nsttp']     == 5:
                 EXPEQdata['q']         = interp(EXPEQdata['rhotor'],EXPEQdata['q'],rhotor,rhopsi,rhopsi)
            EXPEQdata['rhopsi']         = rhopsi[:]
            EXPEQdata['rhotor']         = rhotor[:]
       elif rhotorflag and EXPEQdata['nrhotype']==0:
            if   EXPEQdata['nppfun']    == 4:
                 EXPEQdata['pprime']    = interp(EXPEQdata['rhopsi'],EXPEQdata['pprime'],rhopsi,rhotor,rhotor)
            elif EXPEQdata['nppfun']    == 8:
                 EXPEQdata['pressure']  = interp(EXPEQdata['rhopsi'],EXPEQdata['pressure'],rhopsi,rhotor,rhotor)
            if   EXPEQdata['nsttp']     == 1:
                 EXPEQdata['ffprime']   = interp(EXPEQdata['rhopsi'],EXPEQdata['ffprime'],rhopsi,rhotor,rhotor)
            elif EXPEQdata['nsttp']     == 2:
                 EXPEQdata['Istr']      = interp(EXPEQdata['rhopsi'],EXPEQdata['Istr'],rhopsi,rhotor,rhotor)
            elif EXPEQdata['nsttp']     == 3:
                 EXPEQdata['Iprl']      = interp(EXPEQdata['rhopsi'],EXPEQdata['Iprl'],rhopsi,rhotor,rhotor)
            elif EXPEQdata['nsttp']     == 4:
                 EXPEQdata['Jprl']      = interp(EXPEQdata['rhopsi'],EXPEQdata['Jprl'],rhopsi,rhotor,rhotor)
            elif EXPEQdata['nsttp']     == 5:
                 EXPEQdata['q']         = interp(EXPEQdata['rhopsi'],EXPEQdata['q'],rhopsi,rhotor,rhotor)
            EXPEQdata['rhopsi']         = rhopsi[:]
            EXPEQdata['rhotor']         = rhotor[:]

    return EXPEQdata

def write_expeq(setParam={},outfile=True,**kwargs):
    '''
    nrhomesh=[rho_type(0:rhopsi,1:rhotor),rho_src(0:chease,1:eqdsk)]
    nppfun  =[pressure_type(8:pressure,4:pprime),pressure_src(0:chease,1:eqdsk,2:expeq,3:exptnz,4:iterdb,5:profiles)]
    nsttp   =[current_type(1:ffprime,2:istar,3:iprl,4:jprl,5:q),current_src(0:chease,1:eqdsk,2:expeq)]
    boundary= boundary_type(0:asis,1,interpolate)
    '''

    eqdskflag    = False
    expeqflag    = False
    interpflag   = False
    cheaseflag   = False
    iterdbflag   = False
    exptnzflag   = False
    importedflag = False
    profilesflag = False
    for key,value in kwargs.items():
        if key in ['chease','cheasedata','cheasefpath']:
           if os.path.isfile(value.strip()):
              cheasepath = value.strip()
              cheasedata = read_chease(cheasefpath=cheasepath)
              cheaseflag = True

        if key in ['eqdsk','eqdskdata','eqdskfpath']:
           if os.path.isfile(value.strip()):
              eqdskpath = value.strip()
              eqdskdata = read_eqdsk(eqdskfpath=eqdskpath)
              eqdskflag = True

        if key in ['expeq','expeqdata','expeqfpath']:
           if os.path.isfile(value.strip()):
              expeqpath = value.strip()
              expeqdata = read_expeq(expeqfpath=expeqpath)
              expeqflag = True

        if key in ['exptnz','exptnzdata','exptnzfpath']:
           if os.path.isfile(value.strip()):
              exptnzpath = value.strip()
              exptnzflag = True

        if key in ['profiles','profilesdata','profilesfpath']:
           if os.path.isfile(value.strip()):
              profilespath = value.strip()
              profilesflag = True

        if key in ['iterdb','iterdbdata','iterdbfpath']:
           if os.path.isfile(value.strip()):
              iterdbpath = value.strip()
              iterdbflag = True

        if key in ['imported','external','others']:
              imported = value.copy()
              importedflag = True


    if not (cheaseflag or expeqflag  or eqdskflag or exptnzflag or profilesflag or iterdbflag or importedflag):
       raise IOError('FATAL: NO VALID INPUT PROFILES AVAILABLE. EXIT!')


    if 'nrhomesh' in setParam.keys():
       if   type(setParam['nrhomesh'])==list:
            if   type(setParam['nrhomesh'][0])==float: setParam['nrhomesh'][0] = int(setParam['nrhomesh'][0])
            elif type(setParam['nrhomesh'][0])==str:   setParam['nrhomesh'][0] = setParam['nrhomesh'][0].lower()
            if   type(setParam['nrhomesh'][1])==float: setParam['nrhomesh'][1] = int(setParam['nrhomesh'][1])
            elif type(setParam['nrhomesh'][1])==str:   setParam['nrhomesh'][1] = setParam['nrhomesh'][1].lower()
            elif      setParam['nrhomesh'][1] ==None:  setParam['nrhomesh'][1] = None
            nrhotype = setParam['nrhomesh'][:]
       elif type(setParam['nrhomesh']) in [int,str,float]:
            if   type(setParam['nrhomesh'])==float: setParam['nrhomesh'] = int(setParam['nrhomesh'])
            elif type(setParam['nrhomesh'])==str:   setParam['nrhomesh'] = setParam['nrhomesh'].lower()
            if   cheaseflag:   nrhotype=[setParam['nrhomesh'],0]
            elif eqdskflag:    nrhotype=[setParam['nrhomesh'],1]
            elif importedflag: nrhotype=[setParam['nrhomesh'],7]
    else:
            if   cheaseflag:   nrhotype=[0,0]
            elif eqdskflag:    nrhotype=[0,1]
            elif importedflag: nrhotype=[0,7]


    if 'nppfun' in setParam.keys():
       if   type(setParam['nppfun'])==list:
            if   type(setParam['nppfun'][0])==float: setParam['nppfun'][0] = int(setParam['nppfun'][0])
            elif type(setParam['nppfun'][0])==str:   setParam['nppfun'][0] = setParam['nppfun'][0].lower()
            if   type(setParam['nppfun'][1])==float: setParam['nppfun'][1] = int(setParam['nppfun'][1])
            elif type(setParam['nppfun'][1])==str:   setParam['nppfun'][1] = setParam['nppfun'][1].lower()
            nppfun=setParam['nppfun'][:]
       elif type(setParam['nppfun']) in [int,float,str]:
            if   type(setParam['nppfun'])==float: setParam['nppfun'] = int(setParam['nppfun'])
            elif type(setParam['nppfun'])==str:   setParam['nppfun'] = setParam['nppfun'].lower()
            if   cheaseflag:   nppfun=[setParam['nppfun'],0]
            elif eqdskflag:    nppfun=[setParam['nppfun'],1]
            elif expeqflag:    nppfun=[setParam['nppfun'],2]
            elif exptnzflag:   nppfun=[setParam['nppfun'],3]
            elif profilesflag: nppfun=[setParam['nppfun'],4]
            elif iterdbflag:   nppfun=[setParam['nppfun'],5]
            elif importedflag: nppfun=[setParam['nppfun'],7]
    else:
            if   cheaseflag:   nppfun=[4,0]
            elif eqdskflag:    nppfun=[4,1]
            elif expeqflag:    nppfun=[4,2]
            elif exptnzflag:   nppfun=[4,3]
            elif profilesflag: nppfun=[4,4]
            elif iterdbflag:   nppfun=[4,5]
            elif importedflag: nppfun=[4,7]

    if 'nsttp' in setParam.keys():
       if   type(setParam['nsttp'])==list:
            if   type(setParam['nsttp'][0])==float: setParam['nsttp'][0] = int(setParam['nsttp'][0])
            elif type(setParam['nsttp'][0])==str:   setParam['nsttp'][0] = setParam['nsttp'][0].lower()
            if   type(setParam['nsttp'][1])==float: setParam['nsttp'][1] = int(setParam['nsttp'][1])
            elif type(setParam['nsttp'][1])==str:   setParam['nsttp'][1] = setParam['nsttp'][1].lower()
            nsttp=setParam['nsttp'][:]
       elif type(setParam['nsttp']) in [int,float,str]:
            if   type(setParam['nsttp'])==float: setParam['nsttp'] = int(setParam['nsttp'])
            elif type(setParam['nsttp'])==str:   setParam['nsttp'] = setParam['nsttp'].lower()
            if   cheaseflag:   nsttp=[setParam['nsttp'],0]
            elif eqdskflag:    nsttp=[setParam['nsttp'],1]
            elif expeqflag:    nsttp=[setParam['nsttp'],2]
            elif importedflag: nsttp=[setParam['nsttp'],7]
    else:
            if   cheaseflag:   nsttp=[1,0]
            elif eqdskflag:    nsttp=[1,1]
            elif expeqflag:    nsttp=[1,2]
            elif importedflag: nsttp=[1,7]

    if 'boundary' in setParam.keys():
         if   type(setParam['boundary'])==float:
              boundary = int(setParam['boundary'])
         elif type(setParam['boundary'])==str:
              boundary = setParam['boundary'].lower()
    else:
              boundary = 0

    if   'cheasemode' in setParam.keys():
         cheasemode = int(setParam['cheasemode'])
    else:
         cheasemode = 1

    if   nrhotype[1] in [0,'chease'] and not cheaseflag:
         raise IOError('chease.h5 FILE IS NOT PROVIDED. EXIT!')
    elif nrhotype[1] in [1,'eqdsk'] and not eqdskflag:
         raise IOError('EQDSK FILE IS NOT PROVIDED. EXIT!')
    elif nrhotype[1] in [7,'imported'] and not eqdskflag:
         raise IOError('IMPORTED DATA IS NOT PROVIDED. EXIT!')

    if nrhotype[0] in [1,'rhotor'] and nsttp[0] in [1,'ffprime','ffprimen']:
       raise NameError('FATAL: nrhotype (must) = 0 or rhopsi. Exit!')

    if   nsttp[1] in [0,'chease'] and not cheaseflag:
         raise IOError('chease.h5 FILE IS NOT PROVIDED. EXIT!')
    elif nsttp[1] in [1,'eqdsk'] and not eqdskflag:
         raise IOError('EQSSK FILE IS NOT PROVIDED. EXIT!')
    elif nsttp[1] in [2,'expeq'] and not expeqflag:
         raise IOError('EXPEQ FILE IS NOT PROVIDED. EXIT!')
    elif nsttp[1] in [7,'imported'] and not importedflag:
         raise IOError('IMPORTED DATA IS NOT PROVIDED. EXIT!')

    if   nppfun[1] in [0,'chease'] and not cheaseflag:
         raise IOError('chease.h5 FILE IS NOT PROVIDED. EXIT!')
    elif nppfun[1] in [1,'eqdsk'] and not eqdskflag:
         raise IOError('EQDSK FILE IS NOT PROVIDED. EXIT!')
    elif nppfun[1] in [2,'expeq'] and not expeqflag:
         raise IOError('EXPEQ FILE IS NOT PROVIDED. EXIT!')
    elif nppfun[1] in [3,'exptnz'] and not exptnzflag:
         raise IOError('EXPTNZ FILE IS NOT PROVIDED. EXIT!')
    elif nppfun[1] in [4,'profiles'] and not profilesflag:
         raise IOError('PROFILES FILE IS NOT PROVIDED. EXIT!')
    elif nppfun[1] in [5,'iterdb'] and not iterdbflag:
         raise IOError('ITERDB FILE IS NOT PROVIDED. EXIT!')
    elif nppfun[1] in [7,'imported'] and not importedflag:
         raise IOError('IMPORTED DATA IS NOT PROVIDED. EXIT!')

    currents = []
    currents.append([2,'istr','istrn','istar','istarn'])
    currents.append([3,'iprl','iprln','iparallel','iparalleln'])
    currents.append([4,'jprl','jprln','jparallel','jparalleln'])
    if (nsttp[1] in [1,'eqdsk']) and (nsttp[0] in currents):
       raise IOError("FATAL: eqdsk option is not accepted with nsttp = 2, 3 or 4")

    expeq = {}

    if   'imported' in locals():
         if   'R0EXP' in imported:
              expeq['R0EXP'] = imported['R0EXP']
         else:
              expeq['R0EXP'] = 1.0
         if   'B0EXP' in imported:
              expeq['B0EXP'] = imported['B0EXP']
         else:
              expeq['B0EXP'] = 1.0
         expeq['nRZmesh'] = npy.size(imported['rbound'])
         expeq['rbound']  = imported['rbound']/expeq['R0EXP']
         expeq['zbound']  = imported['zbound']/expeq['R0EXP']
         expeq['aspect']  = (max(imported['rbound'])-min(imported['rbound']))
         expeq['aspect'] /= (max(imported['rbound'])+min(imported['rbound']))
         if   'ZMAX' in setParam.keys():
              expeq['zgeom'] = setParam['ZMAX']
         else:
              expeq['zgeom'] = 0.0
    elif 'cheasedata' in locals():
         expeq['R0EXP']   = cheasedata['R0EXP']
         expeq['B0EXP']   = cheasedata['B0EXP']
         expeq['nRZmesh'] = npy.size(cheasedata['rbound'])
         expeq['rbound']  = cheasedata['rbound']/expeq['R0EXP']
         expeq['zbound']  = cheasedata['zbound']/expeq['R0EXP']
         expeq['aspect']  = (max(cheasedata['rbound'])-min(cheasedata['rbound']))
         expeq['aspect'] /= (max(cheasedata['rbound'])+min(cheasedata['rbound']))
         expeq['zgeom']   = npy.mean(cheasedata['zmesh'])/expeq['R0EXP']
    elif 'eqdskdata' in locals():
         expeq['R0EXP'] = abs(eqdskdata['RCTR'])
         expeq['B0EXP'] = abs(eqdskdata['BCTR'])
         eqdskParam = {'boundary_type':boundary}
         rbound,zbound    = find_boundary(eqdskdata,setParam=eqdskParam)
         expeq['nRZmesh'] = npy.size(rbound)
         expeq['rbound']  = rbound/expeq['R0EXP']
         expeq['zbound']  = zbound/expeq['R0EXP']
         expeq['aspect']  = (max(rbound)-min(rbound))
         expeq['aspect'] /= (max(rbound)+min(rbound))
         expeq['zgeom']   = eqdskdata['ZMAX']/expeq['R0EXP']
    elif 'expeqdata' in locals():
         if   'R0EXP' in setParam.keys():
              expeq['R0EXP'] = setParam['R0EXP']
         else:
              expeq['R0EXP'] = 1.0
         if   'B0EXP' in setParam.keys():
              expeq['B0EXP'] = setParam['B0EXP']
         else:
              expeq['B0EXP'] = 1.0
         expeq['nRZmesh'] = npy.size(expeqdata['rbound'])
         expeq['rbound']  = expeqdata['rbound'][:]
         expeq['zbound']  = expeqdata['zbound'][:]
         expeq['aspect']  = expeqdata['aspect']
         expeq['zgeom']   = expeqdata['zgeom']

    if   nrhotype[0] in [0,'rhopsi','rhopsin']:
         rhopsiflag = True
         rhotorflag = False
    elif nrhotype[0] in [1,'rhotor','rhotorn']:
         rhopsiflag = False
         rhotorflag = True

    SetParam = {'nrhomesh':nrhotype[0]}

    if   nrhotype[1] in [0,'chease']:
         if cheaseflag:
            cheasedata = read_chease(cheasefpath=cheasepath,setParam=SetParam)
         if eqdskflag:
            eqdskdata = read_eqdsk(eqdskfpath=eqdskpath,setParam=SetParam,chease=cheasepath)
         if expeqflag:
            expeqdata = read_expeq(expeqfpath=expeqpath,setParam=SetParam,chease=cheasepath)
         if exptnzflag:
            exptnzdata = read_exptnz(exptnzfpath=exptnzpath,setParam=SetParam,chease=cheasepath)
         if iterdbflag:
            iterdbdata = read_iterdb(iterdbfpath=iterdbpath,setParam=SetParam,chease=cheasepath)
         if profilesflag:
            profilesdata = read_profiles(profilesfpath=profilespath,setParam=SetParam,chease=cheasepath)
         if importedflag:
            importeddata = read_imported(importeddata=imported,setParam=SetParam,chease=cheasepath)
         if   rhopsiflag:
              expeq['rhopsi'] = cheasedata['rhopsi'][:]
              expeq['nrhomesh']= npy.size(expeq['rhopsi'])
         elif rhotorflag:
              expeq['rhotor'] = cheasedata['rhotor'][:]
              expeq['nrhomesh']= npy.size(expeq['rhotor'])
    elif nrhotype[1] in [1,'eqdsk']:
         if cheaseflag:
            cheasedata = read_chease(cheasefpath=cheasepath,setParam=SetParam,eqdsk=eqdskpath)
         if eqdskflag:
            eqdskdata = read_eqdsk(eqdskfpath=eqdskpath,setParam=SetParam)
         if expeqflag:
            expeqdata = read_expeq(expeqfpath=expeqpath,setParam=SetParam,eqdsk=eqdskpath)
         if exptnzflag:
            exptnzdata = read_exptnz(exptnzfpath=exptnzpath,setParam=SetParam,eqdsk=eqdskpath)
         if iterdbflag:
            iterdbdata = read_iterdb(iterdbfpath=iterdbpath,setParam=SetParam,eqdsk=eqdskpath)
         if profilesflag:
            profilesdata = read_profiles(profilesfpath=profilespath,setParam=SetParam,eqdsk=eqdskpath)
         if importedflag:
            importeddata = read_imported(importeddata=imported,setParam=SetParam,eqdsk=eqdskpath)
         if   rhopsiflag:
              expeq['rhopsi'] = eqdskdata['rhopsi'][:]
              expeq['nrhomesh']= npy.size(expeq['rhopsi'])
         elif rhotorflag:
              expeq['rhotor'] = eqdskdata['rhotor'][:]
              expeq['nrhomesh']= npy.size(expeq['rhotor'])
    elif nrhotype[1] in [7,'imported']:
         if cheaseflag:
            cheasedata = read_chease(cheasefpath=cheasepath,setParam=SetParam,imported=imported)
         if eqdskflag:
            eqdskdata = read_eqdsk(eqdskfpath=eqdskpath,setParam=SetParam,imported=imported)
         if expeqflag:
            expeqdata = read_expeq(expeqfpath=expeqpath,setParam=SetParam,imported=imported)
         if exptnzflag:
            exptnzdata = read_exptnz(exptnzfpath=exptnzpath,setParam=SetParam,imported=imported)
         if iterdbflag:
            iterdbdata = read_iterdb(iterdbfpath=iterdbpath,setParam=SetParam,imported=imported)
         if profilesflag:
            profilesdata = read_profiles(profilesfpath=profilespath,setParam=SetParam,imported=imported)
         if importedflag:
            importeddata = read_imported(importeddata=imported,setParam=SetParam)
         if   rhopsiflag:
              expeq['rhopsi'] = imported['rhopsi'][:]
              expeq['nrhomesh']= npy.size(expeq['rhopsi'])
         elif rhotorflag:
              expeq['rhotor'] = imported['rhotor'][:]
              expeq['nrhomesh']= npy.size(expeq['rhotor'])
    elif nrhotype[1] in [None]:
         if cheaseflag:
            cheasedata = read_chease(cheasefpath=cheasepath,setParam=SetParam)
            if   rhopsiflag:
                 expeq['rhopsi'] = cheasedata['rhopsi'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhopsi'])
            elif rhotorflag:
                 expeq['rhotor'] = cheasedata['rhotor'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhotor'])
         if eqdskflag:
            eqdskdata = read_eqdsk(eqdskfpath=eqdskpath,setParam=SetParam)
            if   rhopsiflag:
                 expeq['rhopsi'] = eqdskdata['rhopsi'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhopsi'])
            elif rhotorflag:
                 expeq['rhotor'] = eqdskdata['rhotor'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhotor'])
         if expeqflag:
            expeqdata = read_expeq(expeqfpath=expeqpath,setParam=SetParam)
            if   rhopsiflag:
                 expeq['rhopsi'] = expeqdata['rhopsi'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhopsi'])
            elif rhotorflag:
                 expeq['rhotor'] = expeqdata['rhotor'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhotor'])
         if exptnzflag:
            exptnzdata = read_exptnz(exptnzfpath=exptnzpath,setParam=SetParam)
            if   rhopsiflag:
                 expeq['rhopsi'] = exptnzdata['rhopsi'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhopsi'])
            elif rhotorflag:
                 expeq['rhotor'] = exptnzdata['rhotor'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhotor'])
         if iterdbflag:
            iterdbdata = read_iterdb(iterdbfpath=iterdbpath,setParam=SetParam)
            if   rhopsiflag:
                 expeq['rhopsi'] = iterdbdata['rhopsi'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhopsi'])
            elif rhotorflag:
                 expeq['rhotor'] = iterdbdata['rhotor'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhotor'])
         if profilesflag:
            profilesdata = read_profiles(profilesfpath=profilespath,setParam=SetParam)
            if   rhopsiflag:
                 expeq['rhopsi'] = profilesdata['rhopsi'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhopsi'])
            elif rhotorflag:
                 expeq['rhotor'] = profilesdata['rhotor'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhotor'])
         if importedflag:
            importeddata = read_imported(importeddata=imported,setParam=SetParam)
            if   rhopsiflag:
                 expeq['rhopsi'] = importeddata['rhopsi'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhopsi'])
            elif rhotorflag:
                 expeq['rhotor'] = importeddata['rhotor'][:]
                 expeq['nrhomesh']= npy.size(expeq['rhotor'])


    if   nppfun[1] in [0,'chease'] and cheaseflag:
         if   cheasemode != 3:
              if importedflag and 'pressure' in importeddata:
                 expeq['pedge'] = mu0*importeddata['pressure'][-1]/expeq['B0EXP']**2
              else:
                 expeq['pedge'] = mu0*cheasedata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [4,'pprime','pprimen']:
                   if importedflag and 'pprime' in importeddata:
                      expeq['pprime'] = mu0*importeddata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
                   else:
                      expeq['pprime'] = mu0*cheasedata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
              elif nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   if importedflag and 'pressure' in importeddata:
                      expeq['pressure'] = mu0*importeddata['pressure']/expeq['B0EXP']**2
                   else:
                      expeq['pressure'] = mu0*cheasedata['pressure']/expeq['B0EXP']**2
         elif cheasemode == 3:
              correctParam = {'nppfun':nppfun,'ITEXP':setParam['ITEXP']}
              correction   = pressure_correction(chease=cheasedata,setParam=correctParam)
              expeq['pedge'] = mu0*cheasedata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   expeq['pressure'] = correction[:]
              elif nppfun[0] in [4,'pprime','pprimen']:
                   expeq['pprime'] = correction[:]

    elif nppfun[1] in [1,'eqdsk'] and eqdskflag:
         if   cheasemode != 3:
              if importedflag and 'pressure' in importeddata:
                 expeq['pedge'] = mu0*importeddata['pressure'][-1]/expeq['B0EXP']**2
              else:
                 expeq['pedge'] = mu0*eqdskdata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [4,'pprime','pprimen']:
                   if importedflag and 'pprime' in importeddata:
                      expeq['pprime'] = mu0*importeddata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
                   else:
                      expeq['pprime'] = mu0*eqdskdata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
              elif nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   if importedflag and 'pressure' in importeddata:
                      expeq['pressure'] = mu0*importeddata['pressure']/expeq['B0EXP']**2
                   else:
                      expeq['pressure'] = mu0*eqdskdata['pressure']/expeq['B0EXP']**2
         elif cheasemode == 3:
              correctParam = {'nppfun':nppfun,'ITEXP':setParam['ITEXP']}
              correction   = pressure_correction(chease=cheasedata,setParam=correctParam)
              expeq['pedge'] = mu0*cheasedata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   expeq['pressure'] = correction[:]
              elif nppfun[0] in [4,'pprime','pprimen']:
                   expeq['pprime'] = correction[:]

    elif nppfun[1] in [2,'expeq'] and expeqflag:
         if   cheasemode != 3:
              if importedflag and 'pressure' in importeddata:
                 expeq['pedge'] = mu0*importeddata['pressure'][-1]/expeq['B0EXP']**2
              else:
                 expeq['pedge'] = expeqdata['pedge']
              if   nppfun[0] in [4,'pprime','pprimen']:
                   if importedflag and 'pprime' in importeddata:
                      expeq['pprime'] = mu0*importeddata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
                   else:
                      expeq['pprime'] = expeqdata['pprime'][:]
              elif nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   if importedflag and 'pressure' in importeddata:
                      expeq['pressure'] = mu0*importeddata['pressure']/expeq['B0EXP']**2
                   else:
                      expeq['pressure'] = expeqdata['pressure'][:]
         elif cheasemode == 3:
              correctParam = {'nppfun':nppfun,'ITEXP':setParam['ITEXP']}
              correction   = pressure_correction(chease=cheasedata,expeq=expeqdata,setParam=correctParam)
              expeq['pedge'] = mu0*cheasedata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   expeq['pressure'] = correction[:]
              elif nppfun[0] in [4,'pprime','pprimen']:
                   expeq['pprime'] = correction[:]

    elif nppfun[1] in [3,'exptnz'] and exptnzflag:
         if   cheasemode != 3:
              if importedflag and 'pressure' in importeddata:
                 expeq['pedge'] = mu0*importeddata['pressure'][-1]/expeq['B0EXP']**2
              else:
                 expeq['pedge'] = mu0*exptnzdata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [4,'pprime','pprimen']:
                   if importedflag and 'pprime' in importeddata:
                      expeq['pprime'] = mu0*importeddata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
                   else:
                      expeq['pprime'] = mu0*exptnzdata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
              elif nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   if importedflag and 'pressure' in importeddata:
                      expeq['pressure'] = mu0*importeddata['pressure']/expeq['B0EXP']**2
                   else:
                      expeq['pressure'] = mu0*exptnzdata['pressure']/expeq['B0EXP']**2
         elif cheasemode == 3:
              correctParam = {'nppfun':nppfun,'ITEXP':setParam['ITEXP']}
              correction   = pressure_correction(chease=cheasedata,setParam=correctParam)
              expeq['pedge'] = mu0*cheasedata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   expeq['pressure'] = correction[:]
              elif nppfun[0] in [4,'pprime','pprimen']:
                   expeq['pprime'] = correction[:]

    elif nppfun[1] in [4,'profiles'] and profilesflag:
         if   cheasemode != 3:
              if importedflag and 'pressure' in importeddata:
                 expeq['pedge'] = mu0*importeddata['pressure'][-1]/expeq['B0EXP']**2
              else:
                 expeq['pedge'] = mu0*profilesdata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [4,'pprime','pprimen']:
                   if importedflag and 'pprime' in importeddata:
                      expeq['pprime'] = mu0*importeddata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
                   else:
                      expeq['pprime'] = mu0*profilesdata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
              elif nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   if importedflag and 'pressure' in importeddata:
                      expeq['pressure'] = mu0*importeddata['pressure']/expeq['B0EXP']**2
                   else:
                      expeq['pressure'] = mu0*profilesdata['pressure']/expeq['B0EXP']**2
         elif cheasemode == 3:
              correctParam = {'nppfun':nppfun,'ITEXP':setParam['ITEXP']}
              correction   = pressure_correction(chease=cheasedata,setParam=correctParam)
              expeq['pedge'] = mu0*cheasedata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   expeq['pressure'] = correction[:]
              elif nppfun[0] in [4,'pprime','pprimen']:
                   expeq['pprime'] = correction[:]

    elif nppfun[1] in [5,'iterdb'] and iterdbflag:
         if   cheasemode != 3:
              if importedflag and 'pressure' in importeddata:
                 expeq['pedge'] = mu0*importeddata['pressure'][-1]/expeq['B0EXP']**2
              else:
                 expeq['pedge'] = mu0*iterdbdata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [4,'pprime','pprimen']:
                   if importedflag and 'pprime' in importeddata:
                      expeq['pprime'] = mu0*importeddata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
                   else:
                      expeq['pprime'] = mu0*iterdbdata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
              elif nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   if importedflag and 'pressure' in importeddata:
                      expeq['pressure'] = mu0*importeddata['pressure']/expeq['B0EXP']**2
                   else:
                      expeq['pressure'] = mu0*iterdbdata['pressure']/expeq['B0EXP']**2
         elif cheasemode == 3:
              correctParam = {'nppfun':nppfun,'ITEXP':setParam['ITEXP']}
              correction   = pressure_correction(chease=cheasedata,setParam=correctParam)
              expeq['pedge'] = mu0*cheasedata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   expeq['pressure'] = correction[:]
              elif nppfun[0] in [4,'pprime','pprimen']:
                   expeq['pprime'] = correction[:]

    elif nppfun[1] in [7,'imported'] and importedflag:
         if   cheasemode != 3:
              if 'pressure' in importeddata:
                 expeq['pedge'] = mu0*importeddata['pressure'][-1]/expeq['B0EXP']**2
              else:
                 expeq['pedge'] = mu0*eqdskdata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [4,'pprime','pprimen']:
                   expeq['pprime'] = mu0*importeddata['pprime']*expeq['R0EXP']**2/expeq['B0EXP']
              elif nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   expeq['pressure'] = mu0*importeddata['pressure']/expeq['B0EXP']**2
         elif cheasemode == 3:
              correctParam = {'nppfun':nppfun,'ITEXP':setParam['ITEXP']}
              correction   = pressure_correction(chease=cheasedata,setParam=correctParam)
              expeq['pedge'] = mu0*cheasedata['pressure'][-1]/expeq['B0EXP']**2
              if   nppfun[0] in [8,'pressure','pressuren','p','pn']:
                   expeq['pressure'] = correction[:]
              elif nppfun[0] in [4,'pprime','pprimen']:
                   expeq['pprime'] = correction[:]


    if   nsttp[1] in [0,'chease'] and cheaseflag:
         if   cheasemode != 2:
              if   nsttp[0] in [1,'ffprime','ffprimen']:
                   if importedflag and 'ffprime' in importeddata:
                      expeq['ffprime'] = importeddata['ffprime']/expeq['B0EXP']
                   else:
                      expeq['ffprime'] = cheasedata['ffprime']/expeq['B0EXP']
              elif nsttp[0] in [2,'istr','istrn','istar','istarn']:
                   if importedflag and 'Istr' in importeddata:
                      expeq['Istr'] = importeddata['Istr']*mu0/expeq['R0EXP']/expeq['B0EXP']
                   else:
                      expeq['Istr'] = cheasedata['Istr']*mu0/expeq['R0EXP']/expeq['B0EXP']
              elif nsttp[0] in [3,'iprl','iprln','iparallel','iparalleln']:
                   if importedflag and 'Iprl' in importeddata:
                      expeq['Iprl'] = importeddata['Iprl']*mu0/expeq['R0EXP']/expeq['B0EXP']
                   else:
                      expeq['Iprl'] = cheasedata['Iprl']*mu0/expeq['R0EXP']/expeq['B0EXP']
              elif nsttp[0] in [4,'jprl','jprln','jparallel','jparalleln']:
                   if importedflag and 'Jprl' in importeddata:
                      expeq['Jprl'] = importeddata['Jprl']*mu0*expeq['R0EXP']/expeq['B0EXP']
                   else:
                      expeq['Jprl'] = cheasedata['Jprl']*mu0*expeq['R0EXP']/expeq['B0EXP']
              elif nsttp[0] in [5,'q','qpsi','qtor']:
                   if importedflag and 'q' in importeddata:
                      expeq['q'] = importeddata['q'][:]
                   else:
                      expeq['q'] = cheasedata['q'][:]
         elif cheasemode == 2:
              correctParam = {'nsttp':nsttp,'ITEXP':setParam['ITEXP']}
              correction   = current_correction(chease=cheasedata,setParam=correctParam)
              if   nsttp[0] in [1,'ffprime','ffprimen']:
                   expeq['ffprime'] = correction[:]
              elif nsttp[0] in [2,'istr','istrn','istar','istarn']:
                   expeq['Istr'] = correction[:]
              elif nsttp[0] in [3,'iprl','iprln','iparallel','iparalleln']:
                   expeq['Iprl'] = correction[:]
              elif nsttp[0] in [4,'jprl','jprln','jparallel','jparalleln']:
                   expeq['Jprl'] = correction[:]
    elif nsttp[1] in [2,'expeq'] and expeqflag:
         if   cheasemode != 2:
              if   nsttp[0] in [1,'ffprime','ffprimen']:
                   if importedflag and 'ffprime' in importeddata:
                      expeq['ffprime'] = importeddata['ffprime']/expeq['B0EXP']
                   else:
                      expeq['ffprime'] = expeqdata['ffprime'][:]
              elif nsttp[0] in [2,'istr','istrn','istar','istarn']:
                   if importedflag and 'Istr' in importeddata:
                      expeq['Istr'] = importeddata['Istr']*mu0/expeq['R0EXP']/expeq['B0EXP']
                   else:
                      expeq['Istr'] = expeqdata['Istr'][:]
              elif nsttp[0] in [3,'iprl','iprln','iparallel','iparalleln']:
                   if importedflag and 'Iprl' in importeddata:
                      expeq['Iprl'] = importeddata['Iprl']*mu0/expeq['R0EXP']/expeq['B0EXP']
                   else:
                      expeq['Iprl'] = expeqdata['Iprl'][:]
              elif nsttp[0] in [4,'jprl','jprln','jparallel','jparalleln']:
                   if importedflag and 'Jprl' in importeddata:
                      expeq['Jprl'] = importeddata['Jprl']*mu0*expeq['R0EXP']/expeq['B0EXP']
                   else:
                      expeq['Jprl'] = expeqdata['Jprl'][:]
              elif nsttp[0] in [5,'q','qpsi','qtor']:
                   if importedflag and 'q' in importeddata:
                      expeq['q'] = importeddata['q']
                   else:
                      expeq['q'] = expeqdata['q'][:]
         elif cheasemode == 2:
              correctParam = {'nsttp':nsttp,'ITEXP':setParam['ITEXP']}
              correction   = current_correction(chease=cheasedata,expeq=expeqdata,setParam=correctParam)
              if   nsttp[0] in [1,'ffprime','ffprimen']:
                   expeq['ffprime'] = correction[:]
              elif nsttp[0] in [2,'istr','istrn','istar','istarn']:
                   expeq['Istr'] = correction[:]
              elif nsttp[0] in [3,'iprl','iprln','iparallel','iparalleln']:
                   expeq['Iprl'] = correction[:]
              elif nsttp[0] in [4,'jprl','jprln','jparallel','jparalleln']:
                   expeq['Jprl'] = correction[:]
    elif nsttp[1] in [1,'eqdsk'] and eqdskflag:
         if   cheasemode != 2:
              if   nsttp[0] in [1,'ffprime','ffprimen']:
                   if importedflag and 'ffprime' in importeddata:
                      expeq['ffprime'] = importeddata['ffprime']/expeq['B0EXP']
                   else:
                      expeq['ffprime'] = eqdskdata['ffprime']/expeq['B0EXP']
              elif nsttp[0] in [5,'q','qpsi','qtor']:
                   if importedflag and 'q' in importeddata:
                      expeq['q'] = importeddata['q']
                   else:
                      expeq['q'] = eqdskdata['q']
              else:
                   raise ValueError("FATAL: nsttp[0] (must) = 1 or 5.")
         elif cheasemode in [2,3]:
              raise ValueError("FATAL: eqdsk option is not accepted with cheasemode = 2 or 3")
    elif nsttp[1] in [7,'imported'] and importedflag:
         if   cheasemode != 2:
              if   nsttp[0] in [1,'ffprime','ffprimen']:
                   expeq['ffprime'] = importeddata['ffprime']/expeq['B0EXP']
              elif nsttp[0] in [2,'istr','istrn','istar','istarn']:
                   expeq['Istr'] = importeddata['Istr']*mu0/expeq['R0EXP']/expeq['B0EXP']
              elif nsttp[0] in [3,'iprl','iprln','iparallel','iparalleln']:
                   expeq['Iprl'] = importeddata['Iprl']*mu0/expeq['R0EXP']/expeq['B0EXP']
              elif nsttp[0] in [4,'jprl','jprln','jparallel','jparalleln']:
                   expeq['Jprl'] = importeddata['Jprl']*mu0*expeq['R0EXP']/expeq['B0EXP']
              elif nsttp[0] in [5,'q','qpsi','qtor']:
                   expeq['q'] = importeddata['q'][:]
         elif cheasemode == 2:
              raise ValueError("FATAL: imported option is not accepted with cheasemode = 2")

    if outfile:
       ofh = open("EXPEQ",'w')
       ofh.write('%18.8E\n'           % expeq['aspect'])
       ofh.write('%18.8E\n'           % expeq['zgeom'])
       ofh.write('%18.8E\n'           % expeq['pedge'])
       ofh.write('%5d\n'              % expeq['nRZmesh'])
       for i in range(expeq['nRZmesh']):
           ofh.write('%18.8E%18.8E\n' % (expeq['rbound'][i],expeq['zbound'][i]))
       ofh.write('%5d%5d\n'           % (expeq['nrhomesh'],nppfun[0]))
       ofh.write('%5d%5d\n'           % (nsttp[0],nrhotype[0]))


       if   nrhotype[0] in [0,'rhopsi']:
            for i in range(expeq['nrhomesh']):
                ofh.write('%18.8E\n'       % expeq['rhopsi'][i])
       elif nrhotype[0] in [1,'rhotor']:
            for i in range(expeq['nrhomesh']):
                ofh.write('%18.8E\n'       % expeq['rhotor'][i])

       if   nppfun[0] in [4,'pprime','pprimen']:
            for i in range(expeq['nrhomesh']):
                ofh.write('%18.8E\n'       % expeq['pprime'][i])
       elif nppfun[0] in [8,'pressure','pressuren','p','pn']:
            for i in range(expeq['nrhomesh']):
                ofh.write('%18.8E\n'       % expeq['pressure'][i])

       if   nsttp[0] in [1,'ffprime','ffprimen']:
            for i in range(expeq['nrhomesh']):
                ofh.write('%18.8E\n'       % expeq['ffprime'][i])
       elif nsttp[0] in [2,'istar','istarn','istr','istrn']:
            for i in range(expeq['nrhomesh']):
                ofh.write('%18.8E\n'       % expeq['Istr'][i])
       elif nsttp[0] in [3,'iprl','iprln','iparallel','iparalleln']:
            for i in range(expeq['nrhomesh']):
                ofh.write('%18.8E\n'       % expeq['Iprl'][i])
       elif nsttp[0] in [4,'jprl','jprln','jparallel','jparalleln']:
            for i in range(expeq['nrhomesh']):
                ofh.write('%18.8E\n'       % expeq['Jprl'][i])
       elif nsttp[0] in [5,'q']:
            for i in range(expeq['nrhomesh']):
                ofh.write('%18.8E\n'       % expeq['q'][i])
       ofh.close()

    return expeq

=======
>>>>>>> 7c0d900362339af9fc7ea7341e74170f46748f28

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
