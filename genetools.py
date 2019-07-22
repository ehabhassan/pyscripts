#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import math
import cmath
import numpy as npy

import efittools
import read_iterdb

from ParIO import *
from momlib import *
from fieldlib import *
from finite_differences import *
from read_write_geometry import *

from scipy.interpolate import interp1d

def create_k_grid(x_grid):
   #Developed by Ehab Hassan on 2019-04-28
    nx  = npy.size(x_grid)
    dx  = min(npy.diff(x_grid))
    dKx = 2.0*npy.pi/nx/dx
    iKx = npy.zeros(nx)
    for ix in range(nx):
        if   ix<=nx/2-1:
             iKx[ix] = ix*dx
        elif ix>=nx/2:
             iKx[ix] =-(nx-ix)*dx
    return npy.fft.fftshift(iKx)


def findall(inlist,item):
   #Developed by Ehab Hassan on 2019-03-28
    inds = []
    for i,character in enumerate(inlist):
        if character==item: inds.append(i)
    return inds

def str2bool(vin):
   #Developed by Ehab Hassan on 2019-03-28
    return (vin.strip()).lower() in ('t','.t.','true','.true.')

def read_parameters(paramfpath):
   #Developed by Ehab Hassan on 2019-02-07
    #Modified by Ehab Hassan on 2019-03-09
    if   "parameters_" in paramfpath.strip():
         paramflist=[paramfpath.strip()]
    elif "parameters.dat" in paramfpath.strip():
         paramflist=[paramfpath.strip()]
    elif "nrg_" in paramfpath.strip():
        paramflist=[paramfpath.strip()[:-8]+'parameters_'+paramfpath.strip()[-4:]]
    else: 
         if paramfpath[-1] != "/": paramfpath+="/"
         paramflist = sorted(glob.glob(paramfpath+"parameters_*"))

    geneparam = {'filepath':paramfpath}
    for fid in range(len(paramflist)):
        if not os.path.isfile(paramflist[fid]):
           print('FATAL in read_parameters:')
           print(paramflist[fid]+' FILE NOT FOUND. Exit!'); sys.exit()
    
        ofh = open(paramflist[fid],'r')
        lines = ofh.readlines()
        ofh.close()

        nspecs = 1
        for line in lines:
            pkeys = geneparam.keys()
            if   line[0] == '&':
                 pkey = line[1:].strip()
                 if pkey == 'species':
                    pkey += str(nspecs)
                    nspecs += 1
                 if pkey not in pkeys: geneparam[pkey] = {}
            elif line.strip() in ['/', '!','']:
                 continue
            else:
                 skeys = geneparam[pkey].keys()
                 items = (line.strip()).split('=')
                 skey  = items[0].strip()
                 if   pkey == "parallelization":
                      if skey not in skeys:
                         geneparam[pkey][skey] = int(items[1])
                      elif type(geneparam[pkey][skey]) == list:
                         geneparam[pkey][skey].append(int(items[1]))
                      else:
                         if geneparam[pkey][skey] != int(items[1]):
                            geneparam[pkey][skey]  = [geneparam[pkey][skey],int(items[1])]
                 elif pkey == "units":
                      if skey not in skeys:
                         geneparam[pkey][skey] = float(items[1])
                      elif type(geneparam[pkey][skey]) == list:
                         geneparam[pkey][skey].append(float(items[1]))
                      else:
                         if abs(geneparam[pkey][skey]-float(items[1])) >= 1.0e-9:
                            geneparam[pkey][skey]  = [geneparam[pkey][skey],float(items[1])]
                 elif pkey == "bdgrid":
                      if skey not in skeys:
                         geneparam[pkey][skey] = str2bool(items[1])
                      elif type(geneparam[pkey][skey]) == list:
                         geneparam[pkey][skey].append(str2bool(items[1]))
                      else:
                         if geneparam[pkey][skey] != str2bool(items[1]):
                            geneparam[pkey][skey]  = [geneparam[pkey][skey],str2bool(items[1])]
                 elif pkey == "box":
                      if skey not in skeys:
                         if   skey in ['lx','lx_a','x0','kymin','lv','lw','kx_center']:
                              geneparam[pkey][skey] = float(items[1])
                         elif skey in ['adapt_lx','adapt_ly']:
                              geneparam[pkey][skey] = bool(items[1])
                         elif skey in ['mu_grid_type']:
                              geneparam[pkey][skey] = str(items[1])
                         else:
                              geneparam[pkey][skey] = int(items[1])
                      elif type(geneparam[pkey][skey]) == list:
                         if   skey in ['lx','lx_a','x0','kymin','lv','lw','kx_center']:
                              geneparam[pkey][skey].append(float(items[1]))
                         elif skey in ['adapt_lx','adapt_ly']:
                              geneparam[pkey][skey].append(str2bool(items[1]))
                         elif skey in ['mu_grid_type']:
                              geneparam[pkey][skey].append(str(items[1]))
                         else:
                              geneparam[pkey][skey].append(int(items[1]))
                      else:
                         if   skey in ['lx','lx_a','x0','kymin','lv','lw','kx_center']:
                              if geneparam[pkey][skey] != float(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],float(items[1])]
                         elif skey in ['adapt_lx','adapt_ly']:
                              if geneparam[pkey][skey] != str2bool(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],str2bool(items[1])]
                         elif skey in ['mu_grid_type']:
                              if geneparam[pkey][skey] != str(items[1]):
                                 geneparam[pkey][skey] = [geneparam[pkey][skey],str(items[1])]
                         else:
                              if geneparam[pkey][skey] != int(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],int(items[1])]
                 elif pkey == "in_out":
                      if skey not in skeys:
                         if   skey in ['iterdb_time']:
                              geneparam[pkey][skey] = float(items[1])
                         elif skey in ['diagdir','chptdir','iterdb_file']:
                              geneparam[pkey][skey] = items[1].strip()[1:-1]
                         elif skey in ['write_std','read_checkpoint','write_checkpoint','write_h5','chpt_read_h5','chpt_write_h5','chpt_read_hac','chpt_write_hac','many_chpts']:
                              geneparam[pkey][skey] = str2bool(items[1])
                         else:
                              geneparam[pkey][skey] = int(items[1])
                      elif type(geneparam[pkey][skey]) == list:
                         if   skey in ['iterdb_time']:
                              geneparam[pkey][skey].append(float(items[1]))
                         elif skey in ['diagdir','chptdir','iterdb_file']:
                              geneparam[pkey][skey].append(items[1].strip()[1:-1])
                         elif skey in ['write_std','read_checkpoint','write_checkpoint','write_h5','chpt_read_h5','chpt_write_h5','chpt_read_hac','chpt_write_hac','many_chpts']:
                              geneparam[pkey][skey].append(str2bool(items[1]))
                         else:
                              geneparam[pkey][skey].append(int(items[1]))
                      else:
                         if   skey in ['iterdb_time']:
                              if geneparam[pkey][skey] != float(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],float(items[1])]
                         elif skey in ['diagdir','chptdir','iterdb_file']:
                              if geneparam[pkey][skey] != items[1].strip()[1:-1]:
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],items[1].strip()[1:-1]]
                         elif skey in ['write_std','read_checkpoint','write_checkpoint','write_h5','chpt_read_h5','chpt_write_h5','chpt_read_hac','chpt_write_hac','many_chpts']:
                              if geneparam[pkey][skey] != str2bool(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],str2bool(items[1])]
                         else:
                              if geneparam[pkey][skey] != int(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],int(items[1])]
                 elif "species" in pkey:
                      if skey not in skeys:
                         if   skey in ['omn','omt','mass','temp','dens']:
                              geneparam[pkey][skey] = float(items[1])
                         elif skey in ['charge','prof_type']:
                              geneparam[pkey][skey] = int(items[1])
                         elif skey in ['passive']:
                              geneparam[pkey][skey] = str2bool(items[1])
                         elif skey in ['name']:
                              geneparam[pkey][skey] = items[1].strip()[1:-1]
                      elif type(geneparam[pkey][skey]) == list:
                         if   skey in ['omn','omt','mass','temp','dens']:
                              geneparam[pkey][skey].append(float(items[1]))
                         elif skey in ['charge','prof_type']:
                              geneparam[pkey][skey].append(int(items[1]))
                         elif skey in ['passive']:
                              geneparam[pkey][skey].append(str2bool(items[1]))
                         elif skey in ['name']:
                              geneparam[pkey][skey].append(items[1].strip()[1:-1])
                      else:
                         if   skey in ['omn','omt','mass','temp','dens']:
                              if geneparam[pkey][skey] != float(items[1]):
                                 geneparam[pkey][skey] = [geneparam[pkey][skey],float(items[1])]
                         elif skey in ['charge','prof_type']:
                              if geneparam[pkey][skey] != int(items[1]):
                                 geneparam[pkey][skey] = [geneparam[pkey][skey],int(items[1])]
                         elif skey in ['passive']:
                              if geneparam[pkey][skey] != str2bool(items[1]):
                                 geneparam[pkey][skey] = [geneparam[pkey][skey],str2bool(items[1])]
                         elif skey in ['name']:
                              if geneparam[pkey][skey] != items[1].strip()[1:-1]:
                                 geneparam[pkey][skey] = [geneparam[pkey][skey],items[1].strip()[1:-1]]
                 elif pkey == "geometry":
                      if skey not in skeys:
                         if   skey in ['norm_flux_projection','mag_prof']:
                              geneparam[pkey][skey] = str2bool(items[1])
                         elif skey in ['magn_geometry','geomdir','geomfile','x_def','dpdx_term']:
                              geneparam[pkey][skey] = items[1].strip()[1:-1]
                         else:
                              geneparam[pkey][skey] = float(items[1])
                      elif type(geneparam[pkey][skey]) == list:
                         if   skey in ['norm_flux_projection']:
                              geneparam[pkey][skey].append(str2bool(items[1]))
                         elif skey in ['magn_geometry','geomdir','geomfile','x_def','dpdx_term']:
                              geneparam[pkey][skey].append(items[1].strip()[1:-1])
                         else:
                              geneparam[pkey][skey].append(float(items[1]))
                      else:
                         if   skey in ['norm_flux_projection']:
                              if geneparam[pkey][skey] != str2bool(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],str2bool(items[1])]
                         elif skey in ['magn_geometry','geomdir','geomfile','x_def','dpdx_term']:
                              if geneparam[pkey][skey] != items[1].strip()[1:-1]:
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],items[1].strip()[1:-1]]
                         else:
                              if geneparam[pkey][skey] != float(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],float(items[1])]
                 elif pkey == "external_contr":
                      if skey not in skeys:
                         if   skey in ['with_coriolis','with_centrifugal','with_comoving_other']:
                              geneparam[pkey][skey] = str2bool(items[1])
                         elif skey in ['kxind_phi_ext','kxind_omn_ext','kxind_omt_ext','kxind_apar_ext','phase_phi_ext','phase_omn_ext','phase_omt_ext','phase_apar_ext']:
                              geneparam[pkey][skey] = int(items[1])
                         else:
                              geneparam[pkey][skey] = float(items[1])
                      elif type(geneparam[pkey][skey]) == list:
                         if   skey in ['with_coriolis','with_centrifugal','with_comoving_other']:
                              geneparam[pkey][skey].append(str2bool(items[1]))
                         elif skey in ['kxind_phi_ext','kxind_omn_ext','kxind_omt_ext','kxind_apar_ext','phase_phi_ext','phase_omn_ext','phase_omt_ext','phase_apar_ext']:
                              geneparam[pkey][skey].append(int(items[1]))
                         else:
                              geneparam[pkey][skey].append(float(items[1]))
                      else:
                         if   skey in ['with_coriolis','with_centrifugal','with_comoving_other']:
                              if geneparam[pkey][skey] != str2bool(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],str2bool(items[1])]
                         elif skey in ['kxind_phi_ext','kxind_omn_ext','kxind_omt_ext','kxind_apar_ext','phase_phi_ext','phase_omn_ext','phase_omt_ext','phase_apar_ext']:
                              if geneparam[pkey][skey] != int(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],int(items[1])]
                         else:
                              if geneparam[pkey][skey] != float(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],float(items[1])]
                 elif pkey == "nonlocal_x":
                      if skey not in skeys:
                         if   skey in ['rad_bc_type','lpow_krook','upow_krook','psource_type','ck_filter_type','lckpow_krook','uckpow_krook']:
                              geneparam[pkey][skey] = int(items[1])
                         elif skey in ['buffer_on_ky0','explicit_buffer','shifted_metric','drive_buffer','ga_spatial_var']:
                              geneparam[pkey][skey] = str2bool(items[1])
                         else:
                              geneparam[pkey][skey] = float(items[1])
                      elif type(geneparam[pkey][skey]) == list:
                         if   skey in ['rad_bc_type','lpow_krook','upow_krook','psource_type','ck_filter_type','lckpow_krook','uckpow_krook']:
                              geneparam[pkey][skey].append(int(items[1]))
                         elif skey in ['buffer_on_ky0','explicit_buffer','shifted_metric','drive_buffer','ga_spatial_var']:
                              geneparam[pkey][skey].append(str2bool(items[1]))
                         else:
                              geneparam[pkey][skey].append(float(items[1]))
                      else:
                         if   skey in ['rad_bc_type','lpow_krook','upow_krook','psource_type','ck_filter_type','lckpow_krook','uckpow_krook']:
                              if geneparam[pkey][skey] != int(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],int(items[1])]
                         elif skey in ['buffer_on_ky0','explicit_buffer','shifted_metric','drive_buffer','ga_spatial_var']:
                              if geneparam[pkey][skey] != str2bool(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],str2bool(items[1])]
                         else:
                              if geneparam[pkey][skey] != float(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],float(items[1])]
                 elif pkey == "general":
                      if skey not in skeys:
                         if   skey in ['antenna_type','perf_tsteps','hyp_z_order','hyp_y_order','hyp_x_order','ev_max_it','n_ev','timelim','ntimesteps']:
                              geneparam[pkey][skey] = int(items[1])
                         elif skey in ['ntimesteps']:
                              geneparam[pkey][skey] = long(items[1])
                         elif skey in ['nonlinear','hypz_compensation','x_local','calc_dt','include_f0_contr','bpar','delzonal','delzonal_fields','arakawa_zv','check qn gradients']:
                              geneparam[pkey][skey] = str2bool(items[1])
                         elif skey in ['comp_type','timescheme','coll_split_scheme','which_ev','init_cond','collision_op','coll_cons_model']:
                              geneparam[pkey][skey] = items[1].strip()[1:-1]
                         elif skey in ['lv_antenna_freq','lv_antenna_initamp','lv_antenna_amp','ev_shift']:
                              geneparam[pkey][skey] = complex(items[1])
                         elif skey in ['perf_vec']:
                              geneparam[pkey][skey] = tuple([int(item) for item in items[1].split()])
                         elif skey in ['lv_antenna_modes']:
                              try:
                                  geneparam[pkey][skey].append(items[1])
                              except ValueError: 
                                  geneparam[pkey][skey] = [items[1]]
                         else:
                              geneparam[pkey][skey] = float(items[1])
                      elif type(geneparam[pkey][skey]) == list:
                         if   skey in ['antenna_type','perf_tsteps','hyp_z_order','hyp_y_order','hyp_x_order','ev_max_it','n_ev','timelim','ntimesteps']:
                              geneparam[pkey][skey].append(int(items[1]))
                         elif skey in ['ntimesteps']:
                              geneparam[pkey][skey].append(long(items[1]))
                         elif skey in ['nonlinear','hypz_compensation','x_local','calc_dt','include_f0_contr','bpar','delzonal','delzonal_fields','arakawa_zv','check qn gradients']:
                              geneparam[pkey][skey].append(str2bool(items[1]))
                         elif skey in ['comp_type','timescheme','coll_split_scheme','which_ev','init_cond','collision_op','coll_cons_model']:
                              geneparam[pkey][skey].append(items[1].strip()[1:-1])
                         elif skey in ['lv_antenna_freq','lv_antenna_initamp','lv_antenna_amp','ev_shift']:
                              geneparam[pkey][skey].append(complex(items[1]))
                         elif skey in ['perf_vec']:
                              geneparam[pkey][skey].append(tuple([int(item) for item in items[1].split()]))
                         elif skey in ['lv_antenna_modes']:
                              continue
                         else:
                              geneparam[pkey][skey].append(float(items[1]))
                      else:
                         if   skey in ['antenna_type','perf_tsteps','hyp_z_order','hyp_y_order','hyp_x_order','ev_max_it','n_ev','timelim','ntimesteps']:
                              if geneparam[pkey][skey] != int(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],int(items[1])]
                         elif skey in ['ntimesteps']:
                              if geneparam[pkey][skey] != long(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],long(items[1])]
                         elif skey in ['nonlinear','hypz_compensation','x_local','calc_dt','include_f0_contr','bpar','delzonal','delzonal_fields','arakawa_zv','check qn gradients']:
                              if geneparam[pkey][skey] != str2bool(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],str2bool(items[1])]
                         elif skey in ['comp_type','timescheme','coll_split_scheme','which_ev','init_cond','collision_op','coll_cons_model']:
                              if geneparam[pkey][skey] != items[1].strip()[1:-1]:
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],items[1].strip()[1:-1]]
                         elif skey in ['lv_antenna_freq','lv_antenna_initamp','lv_antenna_amp','ev_shift']:
                              if geneparam[pkey][skey] != complex(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],complex(items[1])]
                         elif skey in ['perf_vec']:
                              perf_vec_list = tuple([int(item) for item in items[1].split()])
                              if geneparam[pkey][skey] != perf_vec_list:
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],perf_vec_list]
                         elif skey in ['lv_antenna_modes']:
                              continue
                         else:
                              if geneparam[pkey][skey] != float(items[1]):
                                 geneparam[pkey][skey]  = [geneparam[pkey][skey],float(items[1])]

    return geneparam

def read_nrg(nrgfpath,nspecs=0,parameters={},normalized=True):
   #Developed by Ehab Hassan on 2019-03-12
    if "nrg_" in nrgfpath.strip():
       nrgflist=[nrgfpath.strip()]
       parameters = read_parameters(nrgflist[0][:-8]+'parameters'+nrgflist[0][-5:])
    elif "nrg.dat" in nrgfpath.strip():
       nrgflist=[nrgfpath.strip()]
       parameters = read_parameters(nrgflist[0][:-7]+'parameters'+nrgflist[0][-4:])
    else: 
       if nrgfpath[-1] != "/": nrgfpath+="/"
       nrgflist = sorted(glob.glob(nrgfpath+"nrg*"))
       if "nrg_" in nrgflist[0]:
          parameters = read_parameters(nrgflist[0][:-8]+'parameters'+nrgflist[0][-5:])
       elif "nrg.dat" in nrgflist[0]:
          parameters = read_parameters(nrgflist[0][:-7]+'parameters'+nrgflist[0][-4:])

    nspecs    = parameters['box']['n_spec']
    specstype = []
    for ispec in range(1,nspecs+1):
        specname = 'species'+str(ispec)
        specstype.append(parameters[specname]['name'])

    if not normalized:
       units = units_conversion(parameters=parameters)

    nrgdata = {}
    for inrgf in nrgflist:
        if not os.path.isfile(inrgf):
           print(inrgf+' FILE NOT FOUND. Exit!'); sys.exit()

        if "nrg.dat" in inrgf:
           inrgfkey = inrgf[-7:]
        else:
           inrgfkey = inrgf[-8:]

        nrgdata[inrgfkey] = {}
        nrgfhand = open(inrgf,'r')
        while True:
              try:
                 ctime = float(nrgfhand.readline())
                 nrgdata[inrgfkey][ctime]={}
                 for ispecs in range(nspecs):
                     linedata = nrgfhand.readline().split()
                     specdata = [float(item) for item in linedata]
                     if not normalized:
                        specdata[0]*=(units['nref']*units['rhostar'])**2
                        specdata[1]*=(units['vref']*units['rhostar'])**2
                        specdata[2]*=(units['Tref']*units['rhostar'])**2
                        specdata[3]*=(units['Tref']*units['rhostar'])**2
                        specdata[4]*=(units['Ggb'])
                        specdata[5]*=(units['Ggb'])
                        specdata[6]*=(units['Qgb'])
                        specdata[7]*=(units['Qgb'])
                        specdata[8]*=(units['Pgb'])
                        specdata[9]*=(units['Pgb'])

                     nrgdata[inrgfkey][ctime][specstype[ispecs]] = {}

                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['n']        = specdata[0]
                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['upara']    = specdata[1]
                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['Tpara']    = specdata[2]
                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['Tperp']    = specdata[3]
                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['PFluses']  = specdata[4]
                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['PFluxem']  = specdata[5]
                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['HFluxes']  = specdata[6]
                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['HFluxem']  = specdata[7]
                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['Visocses'] = specdata[8]
                     nrgdata[inrgfkey][ctime][specstype[ispecs]]['Viscosem'] = specdata[9]
              except ValueError:
                  break
        nrgfhand.close()

    return nrgdata


def read_scanfile(scanfpath):
   #Developed by Ehab Hassan on 2019-02-01
    if   "scan.log" not in scanfpath:
         if scanfpath[-1] != "/": scanfpath += "/"
    else:
         scanfpath = scanfpath[0:-8]
    if not os.path.isfile(scanfpath+'scan.log'):
       print('FATAL in read_scanfile:')
       print(scanfpath+'scan.log FILE NOT FOUND. Exit!'); sys.exit()

    ofh = open(scanfpath+'scan.log','r')
    lines = ofh.readlines()
    ofh.close()

    scandata = {'scanpath':scanfpath}
    hlist = lines[0].split('|')[1:]
    vlist = []
    for ihead in hlist:
        vlist.append(ihead.split()[0])
    nrecs = npy.size(lines)-1
    vscan = npy.zeros((len(vlist),nrecs))
    gamma = npy.zeros(nrecs)
    omega = npy.zeros(nrecs)
    for irec in range(nrecs):
        line = lines[irec+1].split('|')
        for ivar in range(1,len(vlist)+1):
            vscan[ivar-1,irec] = float(line[ivar])
        gamma[irec] = float(line[-1].split()[0])
        omega[irec] = float(line[-1].split()[1])
    scandata['gamma'] = gamma
    scandata['omega'] = omega
    for ivar in range(len(vlist)):
        scandata[vlist[ivar]] = vscan[ivar,:]
    return scandata


def read_omega(omegafpath):
   #Developed by Ehab Hassan on 2019-03-13
    if   "omega" in omegafpath.strip():
         omegaflist=[omegafpath.strip()]
    else: 
         if omegafpath[-1] != "/": omegafpath+="/"
         omegaflist = sorted(glob.glob(omegafpath+"omega*"))

    kymin = []
    gamma = []
    omega = []
    for iomegaf in omegaflist:
        if not os.path.isfile(iomegaf):
           print('FATAL in read_omega:')
           print(iomegaf+' FILE NOT FOUND. Exit!'); sys.exit()
    
        recvr = npy.genfromtxt(iomegaf)
        kymin.append(float(recvr[0]))
        gamma.append(float(recvr[1]))
        omega.append(float(recvr[2]))

    omegadata = {'kymin':npy.array(kymin),'gamma':npy.array(gamma),'omega':npy.array(omega)}
    return omegadata 

def omega_to_hz(genefpath):
   #Developed by Ehab Hassan on 2019-05-16
    if   "omega" in genefpath.strip():
         omegaflist=[genefpath.strip()]
    else: 
         if genefpath[-1] != "/": genefpath+="/"
         omegaflist = sorted(glob.glob(genefpath+"omega*"))

    kymin     = []
    frequency = []
    for iomegaf in omegaflist:
        if not os.path.isfile(iomegaf):
           raise IOError('FATAL: '+iomegaf+' FILE NOT FOUND. Exit!')

        if  'dat' in iomegaf:
            if not os.path.isfile(iomegaf[:-9]+'parameters.dat'):
               raise IOError('FATAL: '+iomegaf[:-9]+'parameters.dat FILE NOT FOUND. Exit!')
            iparamf = iomegaf[:-9]+'parameters.dat'
        else:
            if not os.path.isfile(iomegaf[:-10]+'parameters'+iomegaf[-5:]):
               raise IOError('FATAL: '+iomegaf[:-10]+'parameters'+iomegaf[-5:]+' FILE NOT FOUND. Exit!')
            iparamf = iomegaf[:-10]+'parameters'+iomegaf[-5:]

        paramdata = read_parameters(iparamf)
        omegadata = read_omega(iomegaf)

        Cs = npy.sqrt(paramdata['units']['Tref']*1000.0*1.602e-19/paramdata['units']['mref']/1.6726e-27)
        omegaref = Cs/paramdata['units']['Lref']

        kymin.append(omegadata['kymin'][omegaflist.index(iomegaf)])
        frequency.append(omegadata['omega'][omegaflist.index(iomegaf)]*omegaref/2.0/npy.pi)

    return kymin,frequency


def my_corr_func_complex(v1,v2,time,show_plot=False,v1eqv2=True):
   #Developed by David Hatch on ????-??-??
    dt=time[1]-time[0]
    N=len(time)
    cfunc=npy.zeros(N,dtype='complex')
    for i in range(N):
        i0=i+1
        cfunc[-i0]=npy.sum(npy.conj(v1[-i0:])*v2[:i0])
    tau=npy.arange(N)
    tau=tau*dt
    if v1eqv2:
        cfunc=npy.real(cfunc)
    max_corr=max(npy.abs(cfunc))
    corr_time=0.0
    i=0
    while corr_time==0.0:
        if (abs(cfunc[i])-max_corr/npy.e) > 0.0 and \
           (abs(cfunc[i+1])-max_corr/npy.e) <= 0.0:
            slope=(cfunc[i+1]-cfunc[i])/(tau[i+1]-tau[i])
            zero=cfunc[i]-slope*tau[i]
            corr_time=(max_corr/npy.e-zero)/slope
        i+=1
    neg_loc = 10000.0
    i=0
    while neg_loc==10000.0 and i < N:
        if cfunc[i] < 0.0:
            neg_loc = tau[i]
        i+=1

    if neg_loc < corr_time:
        print "WARNING: neg_loc < corr_time"
        corr_time = neg_loc

    if show_plot:
        plt.plot(tau,cfunc,'x-')
        ax=plt.axis()
        plt.vlines(corr_time,ax[2],ax[3])
        plt.show()
    return cfunc,tau,corr_time

def read_mom(momfpath,specs='',timeslot=-1):
   #Developed by Ehab Hassan on 2019-03-15
    if   "mom_"+specs in momfpath.strip():
         momflist=[momfpath.strip()]
    else: 
         if momfpath[-1] != "/": momfpath+="/"
         momflist = sorted(glob.glob(momfpath+"mom_%s*" % specs))

    momdata = {}
    for imomf in momflist:
        if not os.path.isfile(imomf):
           print('FATAL in read_mom:')
           print(imomf+' FILE NOT FOUND. Exit!'); sys.exit()

        momdata[imomf] = {}

        par0 = Parameters()
        par0.Read_Pars(imomf[:-10]+"parameters_"+imomf[-4:])
        pars = par0.pardict
        mom = momfile(imomf,pars)
        mom.set_time(mom.tmom[timeslot])

        momdata[imomf]['dens']  = mom.dens()[:,:,:]
        momdata[imomf]['tpar']  = mom.tpar()[:,:,:]
        momdata[imomf]['tperp'] = mom.tperp()[:,:,:]
        momdata[imomf]['qpar']  = mom.qpar()[:,:,:]
        momdata[imomf]['qperp'] = mom.qperp()[:,:,:]
        momdata[imomf]['upar']  = mom.upar()[:,:,:]

       #nx  = int(mom.pars['nx0'])
       #nky = int(mom.pars['nky0'])
       #nz  = int(mom.pars['nz0'])
       #dz  = 2.0/nz

       #if 'lx_a' in mom.pars:
       #     xgrid = npy.arange(nx)/float(nx-1)*float(mom.pars['lx_a']) + float(mom.pars['x0']) - float(mom.pars['lx_a'])/2.0
       #else:
       #     xgrid = npy.arange(nx)/float(nx-1)*float(mom.pars['lx'])   - float(mom.pars['lx'])/2.0
       #zgrid = npy.arange(nz)/float(nz-1)*(2.0-dz)-1.0

       #momdata[imomf]['xgrid'] = xgrid
       #momdata[imomf]['zgrid'] = xgrid

    return momdata


def read_field(fieldfpath,timeslot=None,fieldfmt=None):
   #Developed by Ehab Hassan on 2019-03-13
    '''
    fieldfmt argument takes three choices:
    fieldfmt = 'original' returns the fields as read from file
    fieldfmt = 'central'  returns the fields at the central point
    fieldfmt = 'overall'  returns the fields with fixed phase
    '''

    if fieldfmt not in ['central','overall']: fieldfmt = 'original'

    if   "field_" in fieldfpath.strip():
         fieldflist=[fieldfpath.strip()]
    else: 
         if fieldfpath[-1] != "/": fieldfpath+="/"
         fieldflist = sorted(glob.glob(fieldfpath+"field_*"))

    fielddata={}
    for ifieldf in fieldflist:
        if not os.path.isfile(ifieldf):
           print('FATAL in read_field:')
           print(ifieldf+' FILE NOT FOUND. Exit!'); sys.exit()

        fielddata[ifieldf] = {}

        par0 = Parameters()
        par0.Read_Pars(ifieldf[:-10]+"parameters_"+ifieldf[-4:])
        pars = par0.pardict
        field = fieldfile(ifieldf,pars)

        if timeslot == None: t_ind = -1
        else: t_ind = np.argmin(abs(npy.array(field.tfld)-timeslot))
        field.set_time(field.tfld[t_ind])

        if 'x_local' in pars:
           if pars['x_local']:
              x_local = True
           else:
              x_local = False
        else:
           x_local = True

        if fieldfmt=='original':
           phi  = field.phi()
           apar = field.apar()

        elif fieldfmt in ['central','overall']:
           if x_local:
              if 'n0_global' in pars:
                 n0_global = int(pars['n0_global'])
                 q0        = float(pars['q0'])
                 phase     = -npy.e**(-2.0*npy.pi*(0.0+1.0J)*n0_global*q0)
              else:
                 shat      = float(pars['shat'])
                 kymin     = float(pars['kymin'])
                 lx        = float(pars['lx'])
                 phase     = -npy.e**(-npy.pi*(0.0+1.0J)*shat*kymin*lx)

              if   fieldfmt=='central':
                   nx   = 1
                   nz   = field.nz
                   phi  = npy.zeros(nz,dtype='complex128')
                   apar = npy.zeros(nz,dtype='complex128')
              elif fieldfmt=='overall':
                   nx   = field.nx
                   nz   = field.nz
                   phi  = npy.zeros(nx*nz,dtype='complex128')
                   apar = npy.zeros(nx*nz,dtype='complex128')

              phi3d  = field.phi()
              apar3d = field.apar()

              shatsgn = int(npy.sign(float(pars['shat'])))
              for i in range(-nx/2+1,nx/2+1):
                  phi[(i+nx/2-1)*nz:(i+nx/2)*nz]=phi3d[:,0,i*shatsgn]*phase**i
                  if int(field.nfields)>1  and float(pars['beta'])!=0:
                     apar[(i+nx/2-1)*nz:(i+nx/2)*nz]=apar3d[:,0,i*shatsgn]*phase**i

        fielddata[ifieldf]['t']       = field.tfld
        fielddata[ifieldf]['nx']      = field.nx
        fielddata[ifieldf]['ny']      = field.ny
        fielddata[ifieldf]['nz']      = field.nz
        fielddata[ifieldf]['phi']     = phi
        fielddata[ifieldf]['apar']    = apar
        fielddata[ifieldf]['nfields'] = field.nfields

    return fielddata


def field_info(field,param={}):
   #Developed by Ehab Hassan on 2019-03-14
    for ifieldf in field:
        if not param.keys():
           param = read_parameters(ifieldf[:-10]+"parameters_"+ifieldf[-4:])

        if 'x_local' in param['general'].keys():
           if param['general']['x_local']:
            x_local = True
           else:
            x_local = False
        else:
            x_local = True

        if x_local:
           nx      = field[ifieldf]['nx']
           ny      = field[ifieldf]['ny']
           nz      = field[ifieldf]['nz']
           phi     = field[ifieldf]['phi']
           apar    = field[ifieldf]['apar']
           nfields = field[ifieldf]['nfields']

           zgrid = npy.arange(nx*nz)/float(nx*nz-1)*(2.0*nx-(2.0/nz))-nx

           if 'n0_global' in param['box']:
               phase = -npy.e**(-2.0*npy.pi*(0.0+1.0J)*param['box']['n0_global']*param['geometry']['q0'])
           else:
               phase = -npy.e**(-npy.pi*(0.0+1.0J)*param['geometry']['shat']*param['box']['kymin']*param['box']['lx'])

           phi1d  = npy.zeros(nx*nz,dtype='complex128')
           apar1d = npy.zeros(nx*nz,dtype='complex128')

           shatsgn = int(npy.sign(param['geometry']['shat']))
           for i in range(nx/2):
               phi1d[(i+nx/2)*nz:(i+nx/2+1)*nz]=phi[:,0,i*shatsgn]*phase**i
               if i < nx/2:
                   phi1d[(nx/2-i-1)*nz:(nx/2-i)*nz]=phi[:,0,-(i+1)*shatsgn]*phase**(-(i+1))
               if int(nfields)>1:
                  apar1d[(i+nx/2)*nz:(i+nx/2+1)*nz]=apar[:,0,i*shatsgn]*phase**i
                  if i < nx/2:
                       apar1d[(nx/2-i-1)*nz:(nx/2-i)*nz]=apar[:,0,-(i+1)*shatsgn]*phase**(-(i+1))

           phi  = phi1d/phi[nz/2,0,0]
           apar = apar1d/apar[nz/2,0,0]

           field_info = {}

           field_info['phi'] = phi
           field_info['apar']= apar

           field_info['zavg'] = npy.sum(npy.abs(phi)*npy.abs(zgrid))/npy.sum(npy.abs(phi))
           cfunc,zed,corr_len=my_corr_func_complex(phi,phi,zgrid,show_plot=False)
           field_info['corr_len'] = corr_len
           field_info['parity_factor_apar'] = npy.abs(npy.sum(apar))/npy.sum(npy.abs(apar))
           field_info['parity_factor_phi']  = npy.abs(npy.sum(phi))/npy.sum(npy.abs(phi))

           #Calculating E|| Cancellation
           gpars,geometry = read_geometry_local(ifieldf[:-10]+param['geometry']['magn_geometry']+'_'+ifieldf[-4:])
           jacxB = geometry['gjacobian']*geometry['gBfield']
           omegadata = read_omega(ifieldf[:-10]+"omega_"+ifieldf[-4:])
           omega_complex = (omegadata['omega']*(0.0+1.0J) + omegadata['gamma'])
           gradphi = fd_d1_o4(phi,zgrid)
           for j in range(int(param['box']['nx0'])):
               gradphi[int(param['box']['nz0'])*j:int(param['box']['nz0'])*(j+1)] = gradphi[int(param['box']['nz0'])*j:int(param['box']['nz0'])*(j+1)]/jacxB[:]/npy.pi
           diff = npy.sum(npy.abs(gradphi + omega_complex*apar))
           phi_cont = npy.sum(npy.abs(gradphi))
           apar_cont = npy.sum(npy.abs(omega_complex*apar))
           field_info['Epar_Cancellation'] = diff/(phi_cont+apar_cont)

           #Calculating Global Factor
           phikx = field[ifieldf]['phi'][:,0,:]
           aparkx = field[ifieldf]['apar'][:,0,:]
           phi0 = npy.empty(npy.shape(phikx),dtype = 'complex')
           apar0 = npy.empty(npy.shape(aparkx),dtype = 'complex')
           phi0 = phikx
           apar0 = aparkx
           #Calculate <gamma_HB> / gamma
           geomfile = param['geometry']['magn_geometry'][1:-1]+'_'+ifieldf[-4:]
           #Estimate of global relevance
           dkx = 2.0*npy.pi*float(param['geometry']['shat'])
           phi2tot = npy.sum(npy.sum(abs(phikx)**2,axis=0),axis=0)
           kxgrid = npy.empty(int(param['box']['nx0']),dtype='float64')
           if 'kx_center' in param['box'].keys():
              kxgrid[0] = float(param['box']['kx_center'])
           else:
              kxgrid[0] = 0.0
           for k in range(int(param['box']['nx0'])/2):
              kxgrid[k+1] = kxgrid[0] + (k+1)*dkx
           for k in range((int(param['box']['nx0'])-1)/2):
              kxgrid[-k-1] = kxgrid[0] - (k+1)*dkx
           kxavg = 0.0
           #Get eigenmode averaged |kx|
           for k in range(int(param['box']['nx0'])):
              kxavg += abs(kxgrid[k])*npy.sum(abs(phikx[:,k])**2)
           kxavg = kxavg/phi2tot
           #global factor is kxavg / (rho/L_{T,n})--i.e. rho/L_eigenmode / (rho / L_gradients) ~ L_gradients / L_eigenmode
           #==> if L_gradients / L_eigenmode is large then the mode can survive in global 
           #Note: filter_local_modes in plot_scan_info_efit.py
           field_info['global_factor'] = kxavg/(float(param['geometry']['rhostar'])*0.5*(param['species1']['omn']+param['species1']['omt']))

        elif not x_local:
           nx      = field[ifieldf]['nx']
           ny      = field[ifieldf]['ny']
           nz      = field[ifieldf]['nz']
           phi     = field[ifieldf]['phi']
           apar    = field[ifieldf]['apar']

           zgrid = npy.arange(nz+4)/float(nz+4-1)*(2.0+3.0*(2.0/nz))-(1.0+2.0*(2.0/nz))
           xgrid = npy.arange(nx)/float(nx-1)*param['box']['lx_a']+param['box']['x0']-param['box']['lx_a']/2.0

           gpars,geometry = read_geometry_global(ifieldf[:-10]+param['geometry']['magn_geometry']+'_'+ifieldf[-4:])

           phase = (0.0+1.0J)*param['box']['n0_global']*2.0*npy.pi*geometry['q']

           phi_bnd = npy.zeros((nz+4,ny,nx),dtype = 'complex128')
           phi_bnd[2:-2,:,:] = phi
           for j in range(nx):
               phi_bnd[-2,0,j] = phi_bnd[ 2,0,j]*npy.e**(-phase[j])
               phi_bnd[-1,0,j] = phi_bnd[ 3,0,j]*npy.e**(-phase[j])
               phi_bnd[ 0,0,j] = phi_bnd[-4,0,j]*npy.e**( phase[j])
               phi_bnd[ 1,0,j] = phi_bnd[-3,0,j]*npy.e**( phase[j])

           gradphi= npy.zeros((nz+4,nx),dtype = 'complex128')
           for j in range(nx):
               gradphi[:,j]    = fd_d1_o4(phi_bnd[:,0,j],zgrid)
               gradphi[2:-2,j] = gradphi[2:-2,j]/npy.pi/(geometry['jacobian'][:,j]*geometry['Bfield'][:,j])

           field_info = {}

           field_info['phi'] = phi
           field_info['apar']= apar

           field_info['zavg'] = npy.nan

           imaxphi  = np.unravel_index(npy.argmax(abs(phi)),(nz,nx))
           imaxapar = np.unravel_index(npy.argmax(abs(apar)),(nz,nx))
           cfunc,zed,corr_len=my_corr_func_complex(phi[:,0,imaxphi[1]],phi[:,0,imaxphi[1]],zgrid,show_plot=False)
           field_info['corr_len'] = corr_len
           field_info['parity_factor_apar'] = npy.abs(npy.sum(apar[:,0,imaxapar[1]]))/npy.sum(npy.abs(apar[:,0,imaxapar[1]]))
           field_info['parity_factor_phi']  = npy.abs(npy.sum( phi[:,0,imaxphi[1]])) /npy.sum(npy.abs( phi[:,0,imaxphi[1]]))

           omegadata = read_omega(ifieldf[:-10]+"omega_"+ifieldf[-4:])
           omega_complex = (omegadata['omega']*(0.0+1.0J) + omegadata['gamma'])

           diff = np.sum(np.abs(gradphi[2:-2,:] + omega_complex*apar[:,0,:]))
           phi_cont = np.sum(np.abs(gradphi[2:-2,:]))
           apar_cont = np.sum(np.abs(omega_complex*apar[:,0,:]))
           field_info['Epar_Cancellation'] = diff/(phi_cont+apar_cont)

           field_info['global_factor'] = npy.nan

    return field_info

def find_mode_frequency(fieldfpath,bgn_t=None,end_t=None):
   #Developed by Ehab Hassan on 2019-04-18
    if "field_" not in fieldfpath:
       if fieldfpath[-1]!="/": fieldfpath+="/"
       fieldflist = glob.glob(fieldfpath+"field_*")
    else:
       fieldflist = [fieldfpath]

    frequency = {}
    for ifieldf in fieldflist:
        if os.path.isfile(ifieldf):
           field   = read_field(ifieldf,fieldfmt='overall')
           param   = read_parameters(ifieldf[:-10]+"parameters"+ifieldf[-5:])
           tlist   = field[ifieldf]['t']
           nx      = field[ifieldf]['nx']
           ny      = field[ifieldf]['ny']
           nz      = field[ifieldf]['nz']
           nfields = field[ifieldf]['nfields']

           if bgn_t == None: bgn_t  = tlist[-1]*0.9
           if end_t == None: end_t  = tlist[-1]
           bgn_t_ind = npy.argmin(abs(npy.array(tlist)-bgn_t))
           end_t_ind = npy.argmin(abs(npy.array(tlist)-end_t))
           ntimes    = end_t_ind-bgn_t_ind

           if nfields>1:
              time  = npy.empty(0,dtype='complex128')
              phi   = npy.empty(0,dtype='complex128')
              apr   = npy.empty(0,dtype='complex128')
              phi_t = []
              apr_t = []
           else:
              time  = npy.empty(0,dtype='complex128')
              phi   = npy.empty(0,dtype='complex128')
              phi_t = []

           for tind in range(bgn_t_ind,end_t_ind):
               field = read_field(ifieldf,timeslot=tlist[tind],fieldfmt='overall')
               time  = npy.append(time,tlist[tind])

               phi_t.append(field[ifieldf]['phi'])
               if nfields>1:
                  apr_t.append(field[ifieldf]['apar'])

           phi_t = npy.array(phi_t)
           apr_t = npy.array(apr_t)

           comega_t   = []
           weight_t   = []
           comega_avg = npy.empty(0,dtype='complex128')

           max_t_ind,max_z_ind = npy.shape(phi_t)

           for tind in range(1,max_t_ind):
               for zind in range(max_z_ind):
                   comega_t.append(cmath.log((phi_t[tind,zind]/phi_t[tind-1,zind]))/(tlist[tind]-tlist[tind-1]))
                   weight_t.append(abs(phi_t[tind,zind])+abs(phi_t[tind-1,zind]))
               comega_diffs = npy.array([comega_t[i*max_z_ind:(i+1)*max_z_ind] for i in range(npy.size(comega_t)//max_z_ind)],dtype='complex128')
               weight = npy.array([weight_t[i*max_z_ind:(i+1)*max_z_ind] for i in range(npy.size(weight_t)//max_z_ind)],dtype='float128')
               comega_avg = npy.append(comega_avg,npy.sum(comega_diffs[:,tind]*weight[:,tind])/npy.sum(weight[:,tind]))
               gamma = comega_avg[tind-1].real
               omega = comega_avg[tind-1].imag
               gamma_std = npy.sum(weight[:,tind-1]*(gamma-comega_diffs[:,tind-1].real)**2)/npy.sum(weight[:,tind-1])
               omega_std = npy.sum(weight[:,tind-1]*(omega-comega_diffs[:,tind-1].imag)**2)/npy.sum(weight[:,tind-1])
               print tlist[tind],gamma,omega,gamma_std,omega_std
           
          #My Traditional Method
          #for tind in range(bgn_t_ind,end_t_ind):
          #    field = read_field(ifieldf,timeslot=tind,fieldfmt='original')
          #    time  = npy.append(time,tlist[tind])

          #    (iphiz,iphix) = npy.unravel_index(np.argmax(abs(field[ifieldf]['phi'][:,0,:])),(nz,nx))
          #    phi  = npy.append(phi,field[ifieldf]['phi'][iphiz,0,iphix])
          #    if nfields>1:
          #       (iaprz,iaprx) = np.unravel_index(np.argmax(abs(field[ifieldf]['apar'][:,0,:])),(nz,nx))
          #       apr = npy.append(apr,field[ifieldf]['apar'][iaprz,0,iaprx])

          #dt        = npy.diff(time)
          #time      = npy.delete(time,0)
          #phi_omega = npy.log(phi/np.roll(phi,1))
          #phi_omega = npy.delete(phi_omega,0)
          #phi_omega/= dt
          #apr_omega = npy.log(apr/np.roll(apr,1))
          #apr_omega = npy.delete(apr_omega,0)
          #apr_omega/= dt
          #gamma_avg = 0.5*(npy.average(npy.real(phi_omega))+npy.average(npy.real(apr_omega)))
          #omega_avg = 0.5*(npy.average(npy.imag(phi_omega))+npy.average(npy.imag(apr_omega)))
          #frequency["mode"+ifieldf[-5:]]={'ky':param['box']['kymin'],'omega':omega_avg,'gamma':gamma_avg,'phi_omega':phi_omega,'apr_omega':apr_omega}

    return frequency

def mode_type(modeinfo,parameters):
   #Developed by Ehab Hassan on 2019-03-28
   #Based on the work done by David Hatch
    if 'external contr' in parameters.keys():
       if 'ExBrate' in parameters['external contr'].keys() and parameters['external contr']['ExBrate'] != 0.0: wExBshear = True
    else:                                                                                                      wExBshear = False
    epar_threshold = 0.2
    epar_threshold2 = 0.6
    if 'x_local' in parameters['general'].keys() and not parameters['general']['x_local']:
        epar_threshold = 0.4
        epar_threshold2 = 0.7
    dz = 2.0/parameters['box']['nz0']

    mode_type=''

    #Test for numerical modes (i.e. grid-scale z)
    if modeinfo['corr_len'] <= 2*dz:
        if modeinfo['parity_factor_apar']/modeinfo['parity_factor_phi'] > 1.0:
            #Numerical microtearing
            mode_type = 'NMTM'
        else:
            #Numerical ballooning parity
            mode_type = 'NBP'

    #Test for MTM
    #Negative frequency
    #Mostly tearing parity
    #Mostly EM transport
    if modeinfo['Qem/Qes']>0.5 and modeinfo['parity_factor_apar']/modeinfo['parity_factor_phi'] >= 0.8 and not mode_type:
        if not wExBshear:
            if modeinfo[imode,5] < 0.0:
               mode_type = 'MTM'
        else:
            mode_type = 'MTM'

    #Test for KBM
    #Positive frequency
    #Peaked mode structure
    #Mostly ballooning parity
    #Epar cancelation
    if modeinfo['parity_factor_apar']/modeinfo['parity_factor_phi']<0.1 and modeinfo['Epar_Cancellation']<epar_threshold and not mode_type:
        if not wExBshear:
            if modeinfo['omega'] > 0.0:
               mode_type = 'KBM'
        else:
            mode_type = 'KBM'

    if modeinfo['Epar_Cancellation']<epar_threshold2 and not mode_type:
       mode_type = 'ID'

    if not mode_type:
       mode_type = 'Other'

    return mode_type


def mode_info(modesfpath):
   #Developed by Ehab Hassan on 2019-03-13
    if   "_" in modesfpath:
         inds = findall(modesfpath,"/")
         modesfpath = modesfpath.strip()[:inds[-1]+1]
    elif modesfpath[-1] != "/":
         modesfpath+="/"

    omegadata = read_omega(modesfpath)
    nmodes    = len(omegadata['kymin'])
    mode_info = {}

    for imode in range(nmodes):
        nrgdata   = read_nrg(modesfpath+"nrg_%04d" % (imode+1))
        paramdata = read_parameters(modesfpath+"parameters_%04d" % (imode+1))
        field     = read_field(modesfpath+"field_%04d" % (imode+1))
       #fieldplot = plot_field(field,paramdata)
        fieldinfo = field_info(field,paramdata)

        iky = omegadata['kymin'][imode]
        mode_info[iky]={}

        mode_info[iky]['kymin'] = paramdata['box']['kymin']
        if   'x0'        in paramdata['box'].keys():
             mode_info[iky]['x0'] = paramdata['box']['x0']
        else:
             mode_info[iky]['x0'] = npy.nan
        if   'kx_center' in paramdata['box'].keys():
             mode_info[iky]['kx_center'] = paramdata['box']['kx_center']
        else:
             mode_info[iky]['kx_center'] = 0.0
        if   'n0_global' in paramdata['box'].keys():
             mode_info[iky]['n0_global'] = paramdata['box']['n0_global']
        else:
             mode_info[iky]['n0_global'] = npy.nan
        if   omegadata['gamma'][imode] != 0.0:
             mode_info[iky]['gamma'] = omegadata['gamma'][imode]
             mode_info[iky]['omega'] = omegadata['omega'][imode]
        else:
             mode_info[iky]['gamma'] = npy.nan
             mode_info[iky]['omega'] = npy.nan

        if 'zavg' in fieldinfo.keys():
             mode_info[iky]['zavg'] = fieldinfo['zavg']
        else:
             mode_info[iky]['zavg'] = npy.nan
        if 'corr_len' in fieldinfo.keys():
             mode_info[iky]['corr_len'] = fieldinfo['corr_len']
        else:
             mode_info[iky]['corr_len'] = npy.nan
        if 'parity_factor_apar' in fieldinfo.keys():
             mode_info[iky]['parity_factor_apar'] = fieldinfo['parity_factor_apar']
        else:
             mode_info[iky]['parity_factor_apar'] = npy.nan
        if 'parity_factor_phi' in fieldinfo.keys():
             mode_info[iky]['parity_factor_phi'] = fieldinfo['parity_factor_phi']
        else:
             mode_info[iky]['parity_factor_phi'] = npy.nan
        if 'Epar_Cancellation' in fieldinfo.keys():
             mode_info[iky]['Epar_Cancellation'] = fieldinfo['Epar_Cancellation']
        else:
             mode_info[iky]['Epar_Cancellation'] = npy.nan
        if 'global_factor' in fieldinfo.keys():
             mode_info[iky]['glabal_factor'] = fieldinfo['global_factor']
        else:
             mode_info[iky]['glabal_factor'] = npy.nan


        tslots = nrgdata['nrg_%04d' % (imode+1)].keys()
        tind   = tslots.index(max(tslots))
        if   len(nrgdata['nrg_%04d' % (imode+1)].keys()) >= 3:
              mode_info[iky]['Qem/Qes']  = nrgdata['nrg_%04d' % (imode+1)][tslots[tind]]['i'][7]
              mode_info[iky]['Qem/Qes'] /= (abs(nrgdata['nrg_%04d' % (imode+1)][tslots[tind]]['i'][6])+abs(nrgdata['nrg_%04d' % (imode+1)][tslots[tind]]['e'][6]))
        elif len(nrgdata['nrg_%04d' % (imode+1)].keys()) == 2:
              mode_info[iky]['Qem/Qes']  = nrgdata['nrg_%04d' % (imode+1)][tslots[tind]]['e'][7]
              mode_info[iky]['Qem/Qes'] /= abs(nrgdata['nrg_%04d' % (imode+1)][tslots[tind]]['e'][6])

        mode_info[iky]['Type']=mode_type(mode_info[iky],paramdata)

    return mode_info


def flux_type(fluxinfo,parameters,tol=1.0e-2):
   #Developed by Ehab Hassan on 2019-03-XX
    Xi_over_Xe = fluxinfo['i']['Chi']/fluxinfo['e']['Chi']
    Xe_over_Xi = fluxinfo['e']['Chi']/fluxinfo['i']['Chi']
    De_over_Xe = fluxinfo['e']['Dee']/fluxinfo['e']['Chi']
    Dz_over_Xe = fluxinfo['z']['Dee']/fluxinfo['e']['Chi']
    De_over_Xt = fluxinfo['e']['Dee']/(fluxinfo['i']['Chi']+fluxinfo['e']['Chi'])
    Dz_over_Xt = fluxinfo['z']['Dee']/(fluxinfo['i']['Chi']+fluxinfo['e']['Chi'])

    if   all(npy.array([abs(Xi_over_Xe-1.0),abs(De_over_Xe-0.33),abs(Dz_over_Xe-0.33)])<tol): flux_type = 'KBM'
    elif all(npy.array([abs(Xi_over_Xe-0.1),abs(De_over_Xe-0.10),abs(Dz_over_Xe-0.10)])<tol): flux_type = 'MTM'
    elif all(npy.array([abs(Xi_over_Xe-0.1),abs(De_over_Xe-0.05),abs(Dz_over_Xe-0.05)])<tol): flux_type = 'ETG'
    elif abs(Dz_over_Xt-1.0)<tol:
      if (De_over_Xt>=-1.0 and De_over_Xt<=0.33) and (Xe_over_Xi>=0.25 and Xe_over_Xi<=1.0):flux_type = 'ITG/TEM'
    else:                                                                                   flux_type = 'N/A'
    return flux_type


def flux_info(simfpath):
   #Developed by Ehab Hassan on 2019-03-27
    if simfpath[-1] != "/": simfpath+="/"
    if "_" in simfpath:
       slashinds = findall(simfpath,"/")
       simfpath  = simfpath[:slashinds[-2]+1]
    paramdata = read_parameters(simfpath)
    flux_info = {}
    if type(paramdata['box']['kymin'])==list:
       kyminlist = paramdata['box']['kymin']
    else:
       kyminlist = [paramdata['box']['kymin']]
    for imode in range(len(kyminlist)):
       #iky = paramdata['box']['kymin'][imode]
        iky = kyminlist[imode]

        nrgid     = "nrg_%04d" % (imode+1)
        nrgdata   = read_nrg(simfpath+nrgid)

        tslots = nrgdata[nrgid].keys()
        tind   = tslots.index(max(tslots))

        flux_info[iky]={}
        for ispecs in range(paramdata['box']['n_spec']):
             specid    = 'species'+str(ispecs+1)
             specname  = paramdata[specid]['name']
             PFlux_es  = nrgdata[nrgid][tslots[tind]][specname][4]
             PFlux_em  = nrgdata[nrgid][tslots[tind]][specname][5]
             PFlux     = PFlux_es + PFlux_em
             HFlux_es  = nrgdata[nrgid][tslots[tind]][specname][6]
             if type(paramdata[specid]['temp'])==list:
                HFlux_es -= (3./2.)*PFlux_es*max(paramdata[specid]['temp'])
             else:
                HFlux_es -= (3./2.)*PFlux_es*paramdata[specid]['temp']
             HFlux_em  = nrgdata[nrgid][tslots[tind]][specname][7]
             if type(paramdata[specid]['temp'])==list:
                HFlux_em -= (3./2.)*PFlux_em*max(paramdata[specid]['temp'])
             else:
                HFlux_em -= (3./2.)*PFlux_em*paramdata[specid]['temp']
             HFlux     = HFlux_es + HFlux_em
             Dee       = PFlux
             if type(paramdata[specid]['omn'])==list:
                Dee      /= max(paramdata[specid]['omn'])
             else:
                Dee      /= paramdata[specid]['omn']
             if type(paramdata[specid]['dens'])==list:
                Dee      /= max(paramdata[specid]['dens'])
             else:
                Dee      /= paramdata[specid]['dens']
             Chi       = HFlux
             if type(paramdata[specid]['omn'])==list:
                Chi      /= max(paramdata[specid]['omt'])
             else:
                Chi      /= paramdata[specid]['omt']
             if type(paramdata[specid]['dens'])==list:
                Chi      /= max(paramdata[specid]['dens'])
             else:
                Chi      /= paramdata[specid]['dens']
             if type(paramdata[specid]['temp'])==list:
                Chi      /= max(paramdata[specid]['temp'])
             else:
                Chi      /= paramdata[specid]['temp']
             flux_info[iky][specname]={'PFlux_es':PFlux_es,'PFlux_em':PFlux_em,'HFlux_es':HFlux_es,'HFlux_em':HFlux_em,'Dee':Dee,'Chi':Chi}

        flux_info[iky]['Type']=flux_type(flux_info[iky],paramdata)

    return flux_info

def calc_tau(psi,profilepath='',iterdbpath=''):
   #Developed by Ehab Hassan on 2019-03-27
    if   profilepath:
         profiles,units = efittools.read_profiles(profilepath)
         psinorm = profiles['psinorm']
         Te      = profiles['te']
         Ti      = profiles['ti']

         Zeff    = profiles['z'][1]**2*profiles['ni']
         Zeff   += profiles['z'][0]**2*profiles['nz1']
         Zeff   /= profiles['ne']

         tau     = Zeff*Te/Ti

         taupsi  = interp1d(psinorm,tau,kind='linear')
         return taupsi(psi)
    elif iterdbpath:
         iterdbdata = read_iterdb.read_iterdb(iterdbpath)
    else:
         print('No Valid Input')
         return 0


def units_conversion(paramfpath='',parameters={}):
   #Developed by Ehab Hassan on 2019-05-27
    if paramfpath!='' and not parameters:
         parameters = read_parameters(parafpath.strip())

    units               = {}

    units['nref']       = parameters['units']['nref']*1.0e19
    units['Lref']       = parameters['units']['Lref']
    units['Bref']       = parameters['units']['Bref']
    units['Tref']       = parameters['units']['Tref']*1.60218e-19*1.0e3
    units['mref']       = parameters['units']['mref']*1.6726e-27

    units['qref']       = 1.60218e-19
    units['vref']       = npy.sqrt(1.0*units['Tref']/units['mref'])
    units['cref']       = npy.sqrt(units['Tref']/units['mref'])
    units['gyrofreq']   = units['qref']*units['Bref']/units['mref']
    units['gyroradius'] = units['cref']/units['gyrofreq']
    units['rhostar']    = units['gyroradius']/units['Lref']

    units['pref']       = units['nref']*units['Tref']
    units['Ggb']        = units['cref']*units['nref']*units['rhostar']**2
    units['Qgb']        = units['cref']*units['pref']*units['rhostar']**2
    units['Pgb']        = units['nref']*units['mref']*(units['cref']*units['rhostar'])**2

   #nspecs = parameters['box']['n_spec']
   #for ispecs in range(nspecs):
   #    specname                      = 'species'+str(ispecs+1)
   #    units[specname]               = {}
   #    units[specname]['qspec']      = abs(parameters[specname]['charge'])*1.60218e-19
   #    units[specname]['nspec']      = units['nref']*parameters[specname]['dens']
   #    units[specname]['Tspec']      = units['Tref']*parameters[specname]['temp']
   #    units[specname]['mspec']      = units['mref']*parameters[specname]['mass']
   #    units[specname]['pspec']      = units[specname]['nspec']*units[specname]['Tspec']
   #    units[specname]['vspec']      = npy.sqrt(2*units[specname]['Tspec']/units[specname]['mspec'])
   #    units[specname]['cspec']      = npy.sqrt(units[specname]['Tspec']/units['mref'])
   #    units[specname]['gyrofreq']   = units[specname]['qspec']*units['Bref']/units[specname]['mspec']
   #    units[specname]['gyroradius'] = units[specname]['cspec']/units[specname]['gyrofreq']
   #    units[specname]['rhostar']    = units[specname]['gyroradius']/units['Lref']
   #    units[specname]['Ggb']        = units[specname]['cspec']*units[specname]['nspec']*units[specname]['rhostar']**2
   #    units[specname]['Qgb']        = units[specname]['cspec']*units[specname]['pspec']*units[specname]['rhostar']**2
   #    units[specname]['Pgb']        = units[specname]['nspec']*units[specname]['mspec']*(units[specname]['cspec']*units[specname]['rhostar'])**2
    return units


def merge_runs(runspathlist,destination='./'):
   #Developed by Ehab Hassan on 2019-05-21
    if destination[-1]!='/': destination+='/merged'
    else:                    destination+= 'merged'
    if not os.path.isdir(destination):
       os.system('mkdir %s'   %(destination))
       os.system('mkdir %s/%s'%(destination,'in_par'))
    elif not os.path.isdir(destination+'/in_par'):
       os.system('mkdir %s/%s'%(destination,'in_par'))
    ntotalfiles = 0
    for runpath in runspathlist:
        if runpath[-1]=='/': runpath=runpath[:-1]
        nlocalfiles = len(glob.glob(runpath+'/nrg*'))
        for ifile in range(1,nlocalfiles+1):
            os.system('cp %s/nrg_%04d %s/nrg_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            os.system('cp %s/vsp_%04d %s/vsp_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            os.system('cp %s/field_%04d %s/field_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            os.system('cp %s/omega_%04d %s/omega_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            os.system('cp %s/energy_%04d %s/energy_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            os.system('cp %s/parameters_%04d %s/parameters_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            os.system('cp %s/tracer_efit_%04d %s/tracer_efit_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            os.system('cp %s/checkpoint_%04d %s/checkpoint_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            os.system('cp %s/s_checkpoint_%04d %s/s_checkpoint_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            if os.path.isfile('%s/mom_i_%04d'%(runpath,ifile)):
               os.system('cp %s/mom_i_%04d %s/mom_i_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            if os.path.isfile('%s/mom_e_%04d'%(runpath,ifile)):
               os.system('cp %s/mom_e_%04d %s/mom_e_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
            if os.path.isfile('%s/mom_z_%04d'%(runpath,ifile)):
               os.system('cp %s/mom_z_%04d %s/mom_z_%04d'%(runpath,ifile,destination,ntotalfiles+ifile))
        ntotalfiles += nlocalfiles

    wfh = open('%s/scan.log' %destination,'w')
    wfh.write('#Run  | kymin     1  /Eigenvalue1 \n')
    nlocalfiles = len(glob.glob(destination+'/omega*'))
    for ifile in range(1,nlocalfiles+1):
        omega=read_omega('%s/omega_%04d' %(destination,ifile))
        wfh.write("%04d | %12.6E | %6.4f %6.4f \n" %(ifile,omega['kymin'][0],omega['gamma'][0],omega['omega'][0]))
    wfh.close()

    return ntotalfiles
