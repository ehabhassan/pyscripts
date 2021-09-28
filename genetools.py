#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import math
import numpy as npy

import mathtools
import efittools
import read_iterdb

from ParIO import *
from momlib import *
from fieldlib import *
from finite_differences import *
from read_write_geometry import *

from scipy.interpolate import interp1d


if   sys.version_info[0] >= 3:
     PYTHON3 = True; PYTHON2 = False
elif sys.version_info[0] <  3:
     PYTHON2 = True; PYTHON3 = False

def convert_extension(genefpath,targetext=''):
    if targetext == 'dat': targetext = '.dat'

    if genefpath[-1] != '/': genefpath+='/'
    flist = glob.glob(genefpath+"*")
    sourceext = flist[0][-4:]
    if sourceext == targetext:
       print('There is no conversion to do.')
       return 0
    
    for ifile in flist:
        if sourceext not in ifile[-4:]: continue
        if sourceext == '.dat':
           if targetext == '': targetext = '0001'
           fname = ifile[:-4]+'_'+targetext
        else:
           if targetext == '': targetext = '.dat'
           fname = ifile[:-4]+targetext
        os.system('mv '+ifile+' '+fname)

    return 1


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

def read_profiles(profilesfpath,setParam={}):
   #Developed by Ehab Hassan on 2020-04-13
    if not os.path.isfile(profilesfpath):
       print('Fatal: file %s not found.' % profilesfpath)
       sys.exit()

    ofh = open(profilesfpath,'r')

    profiles = {}
    lines = ofh.readlines()
    for line in lines:
        recs = line.split()
        if   'x/a' in recs and 'x/rho_ref' in recs:
             for i in range(1,len(recs)):
                 profiles[recs[i]] = []
        elif '#' in recs:
             continue
        else:
             profiles['x/a']       = recs[0]
             profiles['x/rho_ref'] = recs[1]
             profiles['T']         = recs[2]
             profiles['n']         = recs[3]
             profiles['omt']       = recs[4]
             profiles['omn']       = recs[5]

    return profiles


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
         if os.path.isfile(paramfpath+"parameters_0001"):
            paramflist = sorted(glob.glob(paramfpath+"parameters_*"))
         else:
            paramflist = sorted(glob.glob(paramfpath+"parameters.dat"))

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
                         if   skey in ['omn','omt','mass','temp','dens','charge']:
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
                         if   skey in ['omn','omt','mass','temp','dens','charge']:
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
                         if   skey in ['norm_flux_projection','mag_prof']:
                              geneparam[pkey][skey].append(str2bool(items[1]))
                         elif skey in ['magn_geometry','geomdir','geomfile','x_def','dpdx_term']:
                              geneparam[pkey][skey].append(items[1].strip()[1:-1])
                         else:
                              geneparam[pkey][skey].append(float(items[1]))
                      else:
                         if   skey in ['norm_flux_projection','mag_prof']:
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
                         elif skey in ['hyp_z_with_dz_prefactor','hyp_v_with_dv_prefactor']:
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
                         elif skey in ['hyp_z_with_dz_prefactor','hyp_v_with_dv_prefactor']:
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
                         elif skey in ['hyp_z_with_dz_prefactor','hyp_v_with_dv_prefactor']:
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


def read_nrg(nrgfpath,nspecs=0,parameters={},normalized=True,timeslot=None):
   #Developed by Ehab Hassan on 2019-03-12
    if "nrg_" in nrgfpath.strip():
       nrgflist=[nrgfpath.strip()]
       parameters = read_parameters(nrgflist[0][:-8]+'parameters'+nrgflist[0][-5:])
    elif "nrg.dat" in nrgfpath.strip():
       nrgflist=[nrgfpath.strip()]
       parameters = read_parameters(nrgflist[0][:-7]+'parameters'+nrgflist[0][-4:])
    else: 
       if nrgfpath[-1] != "/": nrgfpath+="/"
       nrgflist = sorted(glob.glob(nrgfpath+"nrg_*"))
       if nrgflist==[]:
          nrgflist = sorted(glob.glob(nrgfpath+"nrg.*"))
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

        inrgfkey = inrgf
        nrgdata[inrgfkey] = {}
        nrgfhand = open(inrgf,'r')

        nrgdata[inrgfkey]['time']=npy.empty(0,dtype=float)
        for ispecs in specstype:
            nrgdata[inrgfkey][ispecs]={}
            nrgdata[inrgfkey][ispecs]['n']=npy.empty(0,dtype=float)
            nrgdata[inrgfkey][ispecs]['upara']=npy.empty(0,dtype=float)
            nrgdata[inrgfkey][ispecs]['Tpara']=npy.empty(0,dtype=float)
            nrgdata[inrgfkey][ispecs]['Tperp']=npy.empty(0,dtype=float)
            nrgdata[inrgfkey][ispecs]['PFluxes']=npy.empty(0,dtype=float)
            nrgdata[inrgfkey][ispecs]['PFluxem']=npy.empty(0,dtype=float)
            nrgdata[inrgfkey][ispecs]['HFluxes']=npy.empty(0,dtype=float)
            nrgdata[inrgfkey][ispecs]['HFluxem']=npy.empty(0,dtype=float)
            nrgdata[inrgfkey][ispecs]['Viscoses']=npy.empty(0,dtype=float)
            nrgdata[inrgfkey][ispecs]['Viscosem']=npy.empty(0,dtype=float)

        while True:
              try:
                 ctime = float(nrgfhand.readline())
                 nrgdata[inrgfkey]['time']=npy.append(nrgdata[inrgfkey]['time'],ctime)
                 for ispecs in specstype:
                     linedata = nrgfhand.readline().split()
                     specdata = [float(item) for item in linedata]
                     if len(specdata)==0:
                        nrgdata[inrgfkey]['time'] = npy.delete(nrgdata[inrgfkey]['time'],-1)
                        ntimes = npy.size(nrgdata[inrgfkey]['time'])
                        specsid = specstype.index(ispecs)
                        if npy.size(nrgdata[inrgfkey][ispecs]['n']) > ntimes:
                           for specind in range(specid):
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['n'],-1)
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['upara'],-1)
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['Tpara'],-1)
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['Tperp'],-1)
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['PFluxes'],-1)
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['PFluxem'],-1)
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['HFluxes'],-1)
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['HFluxem'],-1)
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['Viscoses'],-1)
                                npy.delete(nrgdata[inrgfkey][spectype[specind]]['Viscosem'],-1)
                        break

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

                     nrgdata[inrgfkey][ispecs]['n']=npy.append(nrgdata[inrgfkey][ispecs]['n'],specdata[0])
                     nrgdata[inrgfkey][ispecs]['upara']=npy.append(nrgdata[inrgfkey][ispecs]['upara'],specdata[1])
                     nrgdata[inrgfkey][ispecs]['Tpara']=npy.append(nrgdata[inrgfkey][ispecs]['Tpara'],specdata[2])
                     nrgdata[inrgfkey][ispecs]['Tperp']=npy.append(nrgdata[inrgfkey][ispecs]['Tperp'],specdata[3])
                     nrgdata[inrgfkey][ispecs]['PFluxes']=npy.append(nrgdata[inrgfkey][ispecs]['PFluxes'],specdata[4])
                     nrgdata[inrgfkey][ispecs]['PFluxem']=npy.append(nrgdata[inrgfkey][ispecs]['PFluxem'],specdata[5])
                     nrgdata[inrgfkey][ispecs]['HFluxes']=npy.append(nrgdata[inrgfkey][ispecs]['HFluxes'],specdata[6])
                     nrgdata[inrgfkey][ispecs]['HFluxem']=npy.append(nrgdata[inrgfkey][ispecs]['HFluxem'],specdata[7])
                     nrgdata[inrgfkey][ispecs]['Viscoses']=npy.append(nrgdata[inrgfkey][ispecs]['Viscoses'],specdata[8])
                     nrgdata[inrgfkey][ispecs]['Viscosem']=npy.append(nrgdata[inrgfkey][ispecs]['Viscosem'],specdata[9])

              except ValueError:
                  break
        nrgfhand.close()

        if timeslot != None:
           t_ind = npy.argmin([abs(itime-timeslot) for itime in nrgdata[inrgfkey]['time']])
           nrgdata[inrgfkey]['time']             = nrgdata[inrgfkey]['time'][t_ind]
           nrgdata[inrgfkey][ispecs]['n']        = nrgdata[inrgfkey][ispecs]['n'][t_ind]
           nrgdata[inrgfkey][ispecs]['upara']    = nrgdata[inrgfkey][ispecs]['upara'][t_ind]
           nrgdata[inrgfkey][ispecs]['Tpara']    = nrgdata[inrgfkey][ispecs]['Tpara'][t_ind]
           nrgdata[inrgfkey][ispecs]['Tperp']    = nrgdata[inrgfkey][ispecs]['Tperp'][t_ind]
           nrgdata[inrgfkey][ispecs]['PFluxes']  = nrgdata[inrgfkey][ispecs]['PFluxes'][t_ind]
           nrgdata[inrgfkey][ispecs]['PFluxem']  = nrgdata[inrgfkey][ispecs]['PFluxem'][t_ind]
           nrgdata[inrgfkey][ispecs]['HFluxes']  = nrgdata[inrgfkey][ispecs]['HFluxes'][t_ind]
           nrgdata[inrgfkey][ispecs]['HFluxem']  = nrgdata[inrgfkey][ispecs]['HFluxem'][t_ind]
           nrgdata[inrgfkey][ispecs]['Viscoses'] = nrgdata[inrgfkey][ispecs]['Viscoses'][t_ind]
           nrgdata[inrgfkey][ispecs]['Viscosem'] = nrgdata[inrgfkey][ispecs]['Viscosem'][t_ind]

    return nrgdata


def read_neoclass(neoclassfpath,nspecs=0,parameters={},normalized=True):
   #Developed by Ehab Hassan on 2019-03-12
    neoclassfpath = neoclassfpath.strip()
    if   "neoclass.dat" in neoclassfpath[-13:]:
          neoclassflist=[neoclassfpath]
          parameters = read_parameters(neoclassflist[0][:-12]+'parameters'+neoclassflist[0][-4:])
    elif "neoclass_" in neoclassfpath[-13:]:
          neoclassflist=[neoclassfpath]
          parameters = read_parameters(neoclassflist[0][:-13]+'parameters'+neoclassflist[0][-5:])
    else:
         if neoclassfpath[-1] != "/": neoclassfpath+="/"
         neoclassflist = sorted(glob(neoclassfpath+"neoclass_*"))
         if    neoclassflist==[]:
               neoclassflist = sorted(glob.glob(neoclassfpath+"neoclass.*"))
         if   "neoclass.dat" in neoclassflist[0][-13:]:
               parameters = read_parameters(neoclassflist[0][:-12]+'parameters'+neoclassflist[0][-4:])
         elif "neoclass_" in neoclassflist[0][-13:]:
               parameters = read_parameters(neoclassflist[0][:-13]+'parameters'+neoclassflist[0][-5:])

    nspecs    = parameters['box']['n_spec']
    specstype = []
    for ispec in range(1,nspecs+1):
        specname = 'species'+str(ispec)
        specstype.append(parameters[specname]['name'])

    if not normalized:
       units = units_conversion(parameters=parameters)

    neoclassdata = {}
    for ineoclassf in neoclassflist:
        if not os.path.isfile(ineoclassf):
           print(ineoclassf+' FILE NOT FOUND. Exit!'); sys.exit()

        ineoclassfkey = ineoclassf
        neoclassdata[ineoclassfkey] = {}
        neoclassfhand = open(ineoclassf,'r')
        pagetitle = neoclassfhand.readline()

        neoclassdata[ineoclassfkey]['time']=npy.empty(0,dtype=float)
        for ispecs in specstype:
            neoclassdata[ineoclassfkey][ispecs]={}
            neoclassdata[ineoclassfkey][ispecs]['PFlux']=npy.empty(0,dtype=float)
            neoclassdata[ineoclassfkey][ispecs]['HFlux']=npy.empty(0,dtype=float)
            neoclassdata[ineoclassfkey][ispecs]['Viscos']=npy.empty(0,dtype=float)
            neoclassdata[ineoclassfkey][ispecs]['JBS']=npy.empty(0,dtype=float)

        while True:
              try:
                 ctime = float(neoclassfhand.readline())
                 if not normalized:
                    ctime*=(units['Lref']/units['cref'])
                 neoclassdata[ineoclassfkey]['time']=npy.append(neoclassdata[ineoclassfkey]['time'],ctime)

                 for ispecs in specstype:
                     linedata = neoclassfhand.readline().split()
                     specdata = [float(item) for item in linedata]
                     if len(specdata)==0:
                        neoclassdata[ineoclassfkey]['time'] = npy.delete(neoclassdata[ineoclassfkey]['time'],-1)
                        ntimes = npy.size(neoclassdata[ineoclassfkey]['time'])
                        specsid = specstype.index(ispecs)
                        if npy.size(neoclassdata[ineoclassfkey][ispecs]['PFlux']) > ntimes:
                           for specind in range(specid):
                               npy.delete(neoclassdata[ineoclassfkey][spectype[specind]]['PFlux'],-1)
                               npy.delete(neoclassdata[ineoclassfkey][spectype[specind]]['HFlux'],-1)
                               npy.delete(neoclassdata[ineoclassfkey][spectype[specind]]['Viscos'],-1)
                               npy.delete(neoclassdata[ineoclassfkey][spectype[specind]]['JBS'],-1)
                        break

                     if not normalized:
                        specdata[0]*=(units['Ggb'])
                        specdata[1]*=(units['Qgb'])
                        specdata[2]*=(units['Pgb'])
                        specdata[3]*=(units['nref']*units['cref']*units['Bref']*units['rhostar'])
                     neoclassdata[ineoclassfkey][ispecs]['PFlux']=npy.append(neoclassdata[ineoclassfkey][ispecs]['PFlux'],specdata[0])
                     neoclassdata[ineoclassfkey][ispecs]['HFlux']=npy.append(neoclassdata[ineoclassfkey][ispecs]['HFlux'],specdata[1])
                     neoclassdata[ineoclassfkey][ispecs]['Viscos']=npy.append(neoclassdata[ineoclassfkey][ispecs]['Viscos'],specdata[2])
                     neoclassdata[ineoclassfkey][ispecs]['JBS']=npy.append(neoclassdata[ineoclassfkey][ispecs]['JBS'],specdata[3])

              except ValueError:
                  break
        neoclassfhand.close()

    return neoclassdata


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


def read_omega(omegafpath,calc_omega=True):
   #Developed by Ehab Hassan on 2019-03-13
   #Modified  by Ehab Hassan on 2020-02-04
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
        if   len(recvr) == 0 and calc_omega:
             recvr = [None,None,None]
        elif len(recvr) == 0:
             if   PYTHON3:
                  response = str(input('Do you want to calculate omega? (Y/N)'))
             elif PYTHON2:
                  response = raw_input('Do you want to calculate omega? (Y/N)')
             if response.lower() in ['1','y','yes','yup']:
                frequency = find_mode_frequency(fieldfpath,fraction=0.95,method='quick')
                recvr = npy.genfromtxt(iomegaf)
             else:
                print('Omega has to have value to continue!'); sys.exit()
        try:
           kymin.append(float(recvr[0]))
           gamma.append(float(recvr[1]))
           omega.append(float(recvr[2]))

        except TypeError:
           kymin.append(None)
           gamma.append(None)
           omega.append(None)

    if len(omegaflist)>1:
       omegadata = {'kymin':npy.array(kymin),'gamma':npy.array(gamma),'omega':npy.array(omega)}
    else:
       omegadata = {'kymin':kymin[0],'gamma':gamma[0],'omega':omega[0]}

    return omegadata 

def omega_to_hz(genefpath,calc_omega=False):
   #Developed by Ehab Hassan on 2019-05-16
    if   "omega" in genefpath.strip():
         omegaflist=[genefpath.strip()]
    else: 
         if genefpath[-1] != "/": genefpath+="/"
         omegaflist = sorted(glob.glob(genefpath+"omega*"))

    kymin      = []
    frequency  = []
    growthrate = []
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
        normvals  = units_conversion(parameters=paramdata)
        omegadata = read_omega(iomegaf,calc_omega=calc_omega)

        cs       = normvals['cref']
        Lref     = normvals['Lref']
        omegaref = cs/Lref

        if type(omegadata['kymin']) in [list,tuple]:
           kymin.append(omegadata['kymin'][omegaflist.index(iomegaf)])
           if omegadata['omega'][omegaflist.index(iomegaf)] == None:
              frequency.append(None)
              growthrate.append(None)
           else:
              frequency.append(omegadata['omega'][omegaflist.index(iomegaf)]*omegaref/2.0/npy.pi)
              growthrate.append(omegadata['gamma'][omegaflist.index(iomegaf)])
        else:
           kymin = omegadata['kymin']
           if omegadata['omega'] == None:
              frequency  = None
              growthrate = None
           else:
              frequency  = omegadata['omega']*omegaref/2.0/npy.pi
              growthrate = omegadata['gamma']*omegaref/2.0/npy.pi

    return kymin,frequency,growthrate


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
        print("WARNING: neg_loc < corr_time")
        corr_time = neg_loc

    if show_plot:
        plt.plot(tau,cfunc,'x-')
        ax=plt.axis()
        plt.vlines(corr_time,ax[2],ax[3])
        plt.show()
    return cfunc,tau,corr_time

def read_mom(momfpath,specs='',Normalized=True,timeslot=None,momfmt=None):
   #Developed by Ehab Hassan on 2019-03-15
    '''
    momfmt argument takes three choices:
    momfmt = 'original' returns the moms as read from file
    momfmt = 'local-central' returns the moms at the central point
    momfmt = 'local-flatten' returns the moms flattened over all dimensions
    '''

    momfmttypes = ['local-central','local-flatten','global-modes','global-flatten']
    if momfmt not in momfmttypes: momfmt = 'original'

    momfpath = momfpath.strip()
    if os.path.isfile(momfpath):
       momflist=[momfpath]
    else: 
      if momfpath[-1] != "/": momfpath+="/"
      if   specs in ['e','i','z']:
           momflist = sorted(glob.glob(momfpath+"mom_%s*" % specs))
      else:
           raise ValueError('Species type is missing as a function input')

    momdata = {}
    for imomf in momflist:
        if not os.path.isfile(imomf):
           print('FATAL in read_mom:')
           print(imomf+' FILE NOT FOUND. Exit!'); sys.exit()

        if 'dat' in momfpath:
           paramfpath = imomf[:-9]+"parameters"+imomf[-4:]
        else:
           paramfpath = imomf[:-10]+"parameters"+imomf[-5:]

        momdata[imomf] = {}

        par0 = Parameters()
        par0.Read_Pars(paramfpath)
        pars = par0.pardict

        if 'x_local' in pars:
           if pars['x_local'].lower() in ['t','true']: x_local = True
           else:                                       x_local = False
        else:
                                                       x_local = True
        mom = momfile(imomf,pars)
        if timeslot == None: t_ind = -1
        else:                t_ind = npy.argmin([abs(itime-timeslot) for itime in mom.tmom])
        mom.set_time(mom.tmom[t_ind])

        momdata[imomf]['t'] = mom.tmom[t_ind]

        nx = int(mom.pars['nx0'])
        nz = int(mom.pars['nz0'])
        ny = int(mom.pars['nky0'])

        if momfmt=='original':
           if Normalized:
              momdata[imomf]['dens']  = mom.dens()[:,:,:]
              momdata[imomf]['tpar']  = mom.tpar()[:,:,:]
              momdata[imomf]['tperp'] = mom.tperp()[:,:,:]
              momdata[imomf]['qpar']  = mom.qpar()[:,:,:]
              momdata[imomf]['qperp'] = mom.qperp()[:,:,:]
              momdata[imomf]['upar']  = mom.upar()[:,:,:]
           else:
              units = units_conversion(paramfpath)
              momdata[imomf]['dens']  = units['rhostar']*mom.dens()[:,:,:]
              momdata[imomf]['tpar']  = units['rhostar']*mom.tpar()[:,:,:]
              momdata[imomf]['tperp'] = units['rhostar']*mom.tperp()[:,:,:]
              momdata[imomf]['qpar']  = units['cref']*units['rhostar']*mom.qpar()[:,:,:]
              momdata[imomf]['qperp'] = units['cref']*units['rhostar']*mom.qperp()[:,:,:]
              momdata[imomf]['upar']  = units['cref']*units['rhostar']*mom.upar()[:,:,:]

              dz  = 2.0/nz
              if 'lx_a' in pars:
                   xgrid = npy.arange(nx)/float(nx-1)*float(pars['lx_a'])+float(pars['x0'])-float(pars['lx_a'])/2.0
              else:
                   xgrid = npy.arange(nx)/float(nx-1)*float(pars['lx'])-float(pars['lx'])/2.0
              zgrid = npy.arange(nz)/float(nz-1)*(2.0-dz)-1.0

              momdata[imomf]['xgrid'] = xgrid
              momdata[imomf]['zgrid'] = xgrid

        elif momfmt in momfmttypes:
           dens3d  = mom.dens()[:,:,:]
           tpar3d  = mom.tpar()[:,:,:]
           tperp3d = mom.tperp()[:,:,:]
           qpar3d  = mom.qpar()[:,:,:]
           qperp3d = mom.qperp()[:,:,:]
           upar3d  = mom.upar()[:,:,:]

           if x_local:
              if   momfmt=='local-central': nx = 1
              dens  = npy.zeros(nx*nz,dtype='complex128')
              tpar  = npy.zeros(nx*nz,dtype='complex128')
              tperp = npy.zeros(nx*nz,dtype='complex128')
              qpar  = npy.zeros(nx*nz,dtype='complex128')
              qperp = npy.zeros(nx*nz,dtype='complex128')
              upar  = npy.zeros(nx*nz,dtype='complex128')

              zgrid = npy.arange(nx*nz)/float(nx*nz-1)*(2.0*nx-(2.0/nz))-nx

              if 'n0_global' in pars:
                 n0_global = int(pars['n0_global'])
                 q0        = float(pars['q0'])
                 phase     = -npy.e**(-2.0*npy.pi*(0.0+1.0J)*n0_global*q0)
              else:
                 shat      = float(pars['shat'])
                 kymin     = float(pars['kymin'])
                 lx        = float(pars['lx'])
                 phase     = -npy.e**(-npy.pi*(0.0+1.0J)*shat*kymin*lx)
              shatsgn = int(npy.sign(float(pars['shat'])))

              for iy in range(ny):
                for ix in range(nx/2):
                  dens[ (ix+nx/2)*nz:(ix+nx/2+1)*nz] = dens3d[ :,iy,ix*shatsgn]*phase**ix
                  tpar[ (ix+nx/2)*nz:(ix+nx/2+1)*nz] = tpar3d[ :,iy,ix*shatsgn]*phase**ix
                  tperp[(ix+nx/2)*nz:(ix+nx/2+1)*nz] = tperp3d[:,iy,ix*shatsgn]*phase**ix
                  qpar[ (ix+nx/2)*nz:(ix+nx/2+1)*nz] = qpar3d[ :,iy,ix*shatsgn]*phase**ix
                  qperp[(ix+nx/2)*nz:(ix+nx/2+1)*nz] = qperp3d[:,iy,ix*shatsgn]*phase**ix
                  upar[ (ix+nx/2)*nz:(ix+nx/2+1)*nz] = upar3d[ :,iy,ix*shatsgn]*phase**ix
                  if ix < nx/2:
                     dens[ (nx/2-ix-1)*nz:(nx/2-ix)*nz] = dens3d[ :,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                     tpar[ (nx/2-ix-1)*nz:(nx/2-ix)*nz] = tpar3d[ :,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                     tperp[(nx/2-ix-1)*nz:(nx/2-ix)*nz] = tperp3d[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                     qpar[ (nx/2-ix-1)*nz:(nx/2-ix)*nz] = qpar3d[ :,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                     qperp[(nx/2-ix-1)*nz:(nx/2-ix)*nz] = qperp3d[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                     upar[ (nx/2-ix-1)*nz:(nx/2-ix)*nz] = upar3d[ :,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))

              momdata[imomf]['nx']    = nx
              momdata[imomf]['ny']    = ny
              momdata[imomf]['nz']    = nz
              momdata[imomf]['dens']  = dens
              momdata[imomf]['tpar']  = tpar
              momdata[imomf]['tperp'] = tperp
              momdata[imomf]['qpar']  = qpar
              momdata[imomf]['qperp'] = qperp
              momdata[imomf]['upar']  = upar
              momdata[imomf]['zgrid'] = zgrid

    return momdata


def read_field(fieldfpath,Normalized=True,timeslot=None,fieldfmt=None):
   #Developed by Ehab Hassan on 2019-03-13
    '''
    fieldfmt argument takes three choices:
    fieldfmt = 'original' returns the fields as read from file
    fieldfmt = 'local-central' returns the fields at the central point
    fieldfmt = 'local-flatten' returns the fields flattened over all dimensions
    fieldfmt = 'global-modes'  returns the fields and the global modes
    '''

    fieldfmttypes = ['local-central','local-flatten','global-modes','global-flatten']
    if fieldfmt not in fieldfmttypes: fieldfmt = 'original'

    if   "field" in fieldfpath.strip():
         fieldflist=[fieldfpath.strip()]
    else: 
         if fieldfpath[-1] != "/": fieldfpath+="/"
         fieldflist = sorted(glob.glob(fieldfpath+"field*"))

    fielddata={}
    for ifieldf in fieldflist:
        if not os.path.isfile(ifieldf):
           print('FATAL in read_field:')
           print(ifieldf+' FILE NOT FOUND. Exit!'); sys.exit()

        fielddata[ifieldf] = {}

        par0 = Parameters()
        if 'field.dat' in ifieldf:
           paramfpath = ifieldf[:-9]+"parameters"+ifieldf[-4:]
        else:
           paramfpath = ifieldf[:-10]+"parameters"+ifieldf[-5:]
        par0.Read_Pars(paramfpath)
        pars = par0.pardict
        field = fieldfile(ifieldf,pars)

        if timeslot == None: t_ind = -1
        else:                t_ind = npy.argmin([abs(itime-timeslot) for itime in field.tfld])
        field.set_time(field.tfld[t_ind])

        if 'x_local' in pars:
           if pars['x_local'].lower() in ['t','true']:  x_local = True
           else:                                        x_local = False
        else:                                           x_local = True

        if fieldfmt=='original':
           phi  = field.phi()
           apar = field.apar()

           nx = field.nx
           ny = field.ny
           nz = field.nz
           zgrid = npy.arange(nz)/float(nz-1)*(2.0-(2.0/nz))-1.0
           if 'lx_a' in field.pars:
               xgrid = npy.arange(nx)/float(nx-1)*float(pars['lx_a'])+float(pars['x0'])-float(pars['lx_a'])/2.0
           else:
               xgrid = npy.arange(nx)/float(nx-1)*float(pars['lx'])-float(pars['lx'])/2.0

           fielddata[ifieldf]['t']        = field.tfld[t_ind]
           fielddata[ifieldf]['nx']       = field.nx
           fielddata[ifieldf]['ny']       = field.ny
           fielddata[ifieldf]['nz']       = field.nz
           fielddata[ifieldf]['zgrid']    = zgrid
           fielddata[ifieldf]['xgrid']    = xgrid
           fielddata[ifieldf]['nfields']  = field.nfields
           if Normalized:
              fielddata[ifieldf]['phi']   = phi
              fielddata[ifieldf]['apar']  = apar
           else:
              units = units_conversion(paramfpath)
              fielddata[ifieldf]['phi']   = units['Tref']*units['rhostar']*phi/units['qref']
              fielddata[ifieldf]['apar']  = units['Bref']*units['gyroradius']*units['rhostar']*apar

        elif fieldfmt in fieldfmttypes:
           phi3d  = field.phi()
           apar3d = field.apar()

           if x_local:
              if   fieldfmt=='local-central': nx = 1;        ikx_grid = npy.array([0])
              elif fieldfmt=='local-flatten': nx = field.nx; ikx_grid = npy.arange(-nx//2+1,nx//2+1,dtype=int)
              ny   = field.ny
              nz   = field.nz
              phi  = npy.zeros(nx*nz,dtype='complex128')
              apar = npy.zeros(nx*nz,dtype='complex128')

              zgrid = npy.arange(nx*nz)/float(nx*nz-1)*(2.0*nx-(2.0/nz))-nx

              if 'n0_global' in pars:
                 n0_global = int(pars['n0_global'])
                 q0        = float(pars['q0'])
                 phase     = -npy.e**(-2.0*npy.pi*(0.0+1.0J)*n0_global*q0)
              else:
                 shat      = float(pars['shat'])
                 kymin     = float(pars['kymin'])
                 lx        = float(pars['lx'])
                 phase     = -npy.e**(-npy.pi*(0.0+1.0J)*shat*kymin*lx)
              shatsgn = int(npy.sign(float(pars['shat'])))

             #for iy in range(ny):
             #    for ix in ikx_grid:
             #        phi[(ix-ikx_grid[0])*nz:(ix-ikx_grid[0]+1)*nz] = phi3d[:,iy,ix*shatsgn]*phase**ix
             #        if int(field.nfields)>1  and float(pars['beta'])!=0:
             #            apar[(ix-ikx_grid[0])*nz:(ix-ikx_grid[0]+1)*nz] = apar3d[:,iy,ix*shatsgn]*phase**ix

              for iy in range(ny):
                for ix in range(nx//2):
                  phi[(ix+nx//2)*nz:(ix+nx//2+1)*nz]=phi3d[:,iy,ix*shatsgn]*phase**ix
                  if ix < nx//2:
                     phi[(nx//2-ix-1)*nz:(nx//2-ix)*nz]=phi3d[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                  if int(field.nfields)>1  and float(pars['beta'])!=0:
                     apar[(ix+nx//2)*nz:(ix+nx//2+1)*nz]=apar3d[:,iy,ix*shatsgn]*phase**ix
                     if ix < nx//2:
                        apar[(nx//2-ix-1)*nz:(nx//2-ix)*nz]=apar3d[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))

              fielddata[ifieldf]['t']       = field.tfld[t_ind]
              fielddata[ifieldf]['nx']      = field.nx
              fielddata[ifieldf]['ny']      = field.ny
              fielddata[ifieldf]['nz']      = field.nz
              fielddata[ifieldf]['phi']     = phi
              fielddata[ifieldf]['apar']    = apar
              fielddata[ifieldf]['zgrid']   = zgrid
              fielddata[ifieldf]['nfields'] = field.nfields

           elif not x_local:
              if 'dat' in ifieldf[-4:]:
                 gpars,geometry = read_geometry_global(ifieldf[:-9]+pars['magn_geometry'][1:-1]+ifieldf[-4:])
              else:
                 gpars,geometry = read_geometry_global(ifieldf[:-10]+pars['magn_geometry'][1:-1]+'_'+ifieldf[-4:])
              n0_global      = int(pars['n0_global'])
              q              = geometry['q']
              phase          = 2.0*npy.pi*(0.0+1.0J)*n0_global*q

              nx    = field.nx
              ny    = field.ny
              nz    = field.nz
              zgrid = npy.arange(nz)/float(nz-1)*(2.0-(2.0/nz))-1.0
              if 'lx_a' in field.pars:
                  xgrid = npy.arange(nx)/float(nx-1)*float(pars['lx_a'])+float(pars['x0'])-float(pars['lx_a'])/2.0
              else:
                  xgrid = npy.arange(nx)/float(nx-1)*float(pars['lx'])-float(pars['lx'])/2.0

              phi           = npy.empty((nz+4,ny,nx),dtype = 'complex128')
              phi[2:-2,:,:] = phi3d
             #phi[2:-2,:,:] = phi3d*float(pars['rhostar'])
              for iy in range(ny):
                for ix in range(nx):
                  phi[-2,iy,ix] = phi[ 2,iy,ix]*npy.exp(-phase[ix])
                  phi[-1,iy,ix] = phi[ 3,iy,ix]*npy.exp(-phase[ix])
                  phi[ 0,iy,ix] = phi[-4,iy,ix]*npy.exp( phase[ix])
                  phi[ 1,iy,ix] = phi[-3,iy,ix]*npy.exp( phase[ix])
              if field.nfields>1:
                 apar = npy.empty((nz,ny,nx),dtype = 'complex128')
                 apar = apar3d
             #   apar = apar3d*float(pars['rhostar'])

              fielddata[ifieldf]['t']       = field.tfld[t_ind]
              fielddata[ifieldf]['nx']      = field.nx
              fielddata[ifieldf]['ny']      = field.ny
              fielddata[ifieldf]['nz']      = field.nz
              fielddata[ifieldf]['phi']     = phi 
              fielddata[ifieldf]['apar']    = apar 
              fielddata[ifieldf]['xgrid']   = xgrid
              fielddata[ifieldf]['zgrid']   = zgrid
              fielddata[ifieldf]['nfields'] = field.nfields

              if fieldfmt=='global-flatten':
                 phix = npy.empty(nx*(nz+4),dtype = 'complex128')
                 if field.nfields>1:
                    aparx = npy.empty(nx*nz,dtype = 'complex128')
                 if 'n0_global' in pars:
                    n0_global = int(pars['n0_global'])
                    q0        = float(gpars['q0'])
                    phase     = -npy.e**(-2.0*npy.pi*(0.0+1.0J)*n0_global*q0)
                 else:
                    shat      = float(pars['shat'])
                    kymin     = float(pars['kymin'])
                    lx        = float(pars['lx'])
                    phase     = -npy.e**(-npy.pi*(0.0+1.0J)*shat*kymin*lx)
                 shatsgn = int(npy.sign(float(gpars['shat'])))

                 for iy in range(ny):
                   for ix in range(nx/2):
                     phix[(ix+nx/2)*(nz+4):(ix+nx/2+1)*(nz+4)]=phi[:,iy,ix*shatsgn]*phase**ix
                     if ix < nx/2:
                        phix[(nx/2-ix-1)*(nz+4):(nx/2-ix)*(nz+4)]=phi[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                     if int(field.nfields)>1  and float(pars['beta'])!=0:
                        aparx[(ix+nx/2)*nz:(ix+nx/2+1)*nz]=apar[:,iy,ix*shatsgn]*phase**ix
                        if ix < nx/2:
                           aparx[(nx/2-ix-1)*nz:(nx/2-ix)*nz]=apar[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                 fielddata[ifieldf]['phi']     = phix
                 fielddata[ifieldf]['apar']    = aparx

              elif fieldfmt=='global-modes':
                 min_mode     = math.ceil( npy.min(q)*n0_global)
                 max_mode     = math.floor(npy.max(q)*n0_global)
                 mode_aray    = npy.arange(min_mode,max_mode+1)
                 mode_zgrid   = npy.arange(max_mode*20)/float(max_mode*20)*2.0-1.0
                 n_mode_zgrid = npy.size(mode_zgrid)

                 phi_theta = npy.empty((n_mode_zgrid,ny,nx),dtype = 'complex128')
                 phi_modes = npy.empty((n_mode_zgrid,ny,nx),dtype = 'complex128')
                 if field.nfields>1:
                    apar_theta = npy.empty((n_mode_zgrid,ny,nx),dtype = 'complex128')
                    apar_modes = npy.zeros((n_mode_zgrid,ny,nx),dtype = 'complex128')

                 zgridx = npy.arange(nz+4)/float(nz+4-1)*(2.0+3.0*(2.0/nz))-(1.0+2.0*(2.0/nz))

                 for iy in range(ny):
                   for ix in range(nx):
                     phi_theta[:,iy,ix] = interp(zgridx,npy.real(phi[:,iy,ix]),mode_zgrid)
                     phi_theta[:,iy,ix] = phi_theta[:,iy,ix]*npy.exp(phase[ix]*mode_zgrid)
                     phi_modes[:,iy,ix] = npy.fft.fft(phi_theta[:,iy,ix])

                     if field.nfields>1:
                        apar_theta[:,iy,ix] = interp(zgrid,npy.real(apar[:,iy,ix]),mode_zgrid)
                        apar_theta[:,iy,ix] = apar_theta[:,iy,ix]*npy.exp(phase[ix]*mode_zgrid)
                        apar_modes[:,iy,ix] = npy.fft.fft(apar_theta[:,iy,ix])

                 fielddata[ifieldf]['zgridx']     = zgridx
                 fielddata[ifieldf]['phi_theta']  = phi_theta
                 fielddata[ifieldf]['phi_modes']  = phi_modes
                 fielddata[ifieldf]['apar_theta'] = apar_theta
                 fielddata[ifieldf]['apar_modes'] = apar_modes

    return fielddata


def field_info(field,param={}):
   #Developed by Ehab Hassan on 2019-03-14
    for ifieldf in field:
        if not param.keys():
           if os.path.isfile(ifieldf[:-10]+"parameters_"+ifieldf[-4:]):
              param = read_parameters(ifieldf[:-10]+"parameters_"+ifieldf[-4:])
           else:
              param = read_parameters(ifieldf[:-9]+"parameters.dat")

        if 'x_local' in param['general']: x_local = param['general']['x_local']
        else:                             x_local = True

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
           for i in range(nx//2):
               phi1d[(i+nx//2)*nz:(i+nx//2+1)*nz]=phi[:,0,i*shatsgn]*phase**i
               if i < nx//2:
                   phi1d[(nx//2-i-1)*nz:(nx//2-i)*nz]=phi[:,0,-(i+1)*shatsgn]*phase**(-(i+1))
               if int(nfields)>1:
                  apar1d[(i+nx//2)*nz:(i+nx//2+1)*nz]=apar[:,0,i*shatsgn]*phase**i
                  if i < nx/2:
                       apar1d[(nx//2-i-1)*nz:(nx//2-i)*nz]=apar[:,0,-(i+1)*shatsgn]*phase**(-(i+1))

           phi  = phi1d.copy()
           apar = apar1d.copy()

           field_info = {}

           field_info['phi'] = phi
           field_info['apar']= apar

           field_info['zavg'] = npy.sum(npy.abs(phi)*npy.abs(zgrid))/npy.sum(npy.abs(phi))
           cfunc,zed,corr_len=my_corr_func_complex(phi,phi,zgrid,show_plot=False)
           field_info['corr_len'] = corr_len
           field_info['parity_factor_apar'] = npy.abs(npy.sum(apar))/npy.sum(npy.abs(apar))
           field_info['parity_factor_phi']  = npy.abs(npy.sum(phi))/npy.sum(npy.abs(phi))

           #Calculating E|| Cancellation
           if os.path.isfile(ifieldf[:-10]+param['geometry']['magn_geometry']+'_'+ifieldf[-4:]):
              gpars,geometry = read_geometry_local(ifieldf[:-10]+param['geometry']['magn_geometry']+'_'+ifieldf[-4:])
              omegafpath = ifieldf[:-10]+"omega_"+ifieldf[-4:]
           else:
              gpars,geometry = read_geometry_local(ifieldf[:-9]+param['geometry']['magn_geometry']+'.'+ifieldf[-3:])
              omegafpath = ifieldf[:-9]+"omega."+ifieldf[-3:]

           jacxB = geometry['gjacobian']*geometry['gBfield']
           omegadata = read_omega(omegafpath)
           omega_complex = (omegadata['omega']*(0.0+1.0J) + omegadata['gamma'])
           gradphi = fd_d1_o4(phi,zgrid)
           for j in range(int(param['box']['nx0'])):
               gradphi[param['box']['nz0']*j:param['box']['nz0']*(j+1)] /= (jacxB[:]*npy.pi)
           diff = npy.sum(npy.abs(gradphi + omega_complex*apar))
           phi_cont  = npy.sum(npy.abs(gradphi))
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
           for k in range(int(param['box']['nx0'])//2):
              kxgrid[k+1] = kxgrid[0] + (k+1)*dkx
           for k in range((int(param['box']['nx0'])-1)//2):
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
           field_info['zgrid'] = zgrid

        elif not x_local:
           nx      = field[ifieldf]['nx']
           ny      = field[ifieldf]['ny']
           nz      = field[ifieldf]['nz']
           phi     = field[ifieldf]['phi']
           apar    = field[ifieldf]['apar']

           zgrid = npy.arange(nz+4)/float(nz+4-1)*(2.0+3.0*(2.0/nz))-(1.0+2.0*(2.0/nz))
           xgrid = npy.arange(nx)/float(nx-1)*param['box']['lx_a']+param['box']['x0']-param['box']['lx_a']/2.0

           if os.path.isfile( ifieldf[:-9]+param['geometry']['magn_geometry']+'.dat'):
              geomfpath = ifieldf[:-9]+param['geometry']['magn_geometry']+'.dat'
           else:
              geomfpath = ifieldf[:-10]+param['geometry']['magn_geometry']+'_'+ifieldf[-4:]
           gpars,geometry = read_geometry_global(geomfpath)

           phase = (0.0+1.0J)*param['box']['n0_global']*2.0*npy.pi*geometry['q']

           phi_bnd = npy.zeros((nz+4,ny,nx),dtype = 'complex128')
           gradphi = npy.zeros((nz+4,ny,nx),dtype = 'complex128')
           phi_bnd[2:-2,:,:] = phi

           for j in range(nx):
               phi_bnd[-2,:,j]   = phi_bnd[ 2,:,j]*npy.e**(-phase[j])
               phi_bnd[-1,:,j]   = phi_bnd[ 3,:,j]*npy.e**(-phase[j])
               phi_bnd[ 0,:,j]   = phi_bnd[-4,:,j]*npy.e**( phase[j])
               phi_bnd[ 1,:,j]   = phi_bnd[-3,:,j]*npy.e**( phase[j])

               gradphi[:,0,j]    = fd_d1_o4(phi_bnd[:,0,j],zgrid)
               gradphi[2:-2,0,j] = gradphi[2:-2,0,j]/npy.pi/(geometry['jacobian'][:,j]*geometry['Bfield'][:,j])

           field_info = {}

           field_info['phi']   = phi
           field_info['apar']  = apar
           field_info['dphi']  = gradphi

           field_info['zavg']  = npy.nan
           field_info['xgrid'] = xgrid
           field_info['zgrid'] = zgrid

           imaxphi  = npy.unravel_index(npy.argmax(abs(phi)),(nz,nx))
           imaxapar = npy.unravel_index(npy.argmax(abs(apar)),(nz,nx))
           cfunc,zed,corr_len=my_corr_func_complex(phi[:,0,imaxphi[1]],phi[:,0,imaxphi[1]],zgrid,show_plot=False)
           field_info['corr_len'] = corr_len
           field_info['parity_factor_phi']   = npy.abs(npy.sum( phi[:,0,imaxphi[1]]))
           field_info['parity_factor_phi']  /= npy.sum(npy.abs( phi[:,0,imaxphi[1]]))
           field_info['parity_factor_apar']  = npy.abs(npy.sum(apar[:,0,imaxapar[1]]))
           field_info['parity_factor_apar'] /= npy.sum(npy.abs(apar[:,0,imaxapar[1]]))

           if os.path.isfile(ifieldf[:-10]+"omega_"+ifieldf[-4:]):
              omegadata = read_omega(ifieldf[:-10]+"omega_"+ifieldf[-4:])
           else:
              omegadata = read_omega(ifieldf[:-9]+"omega.dat")
           omega_complex = (omegadata['omega']*(0.0+1.0J) + omegadata['gamma'])

           if 'ExBrate' in param['external_contr'] and param['external_contr']['ExBrate'] == -1111:
              if 'iterdb_file' in param['in_out'] and os.path.isfile(param['in_out']['iterdb_file']):
                  iterdbfpath = param['in_out']['iterdb_file']
              else:      
                  iterdbfpath = input("Path to ITERDB File: ")
              if 'dat' in ifieldf[-4:]: profilesfpath = ifieldf[:-9]+"profiles_e.dat"
              else:                     profilesfpath = ifieldf[:-10]+"profiles_e_"+ifieldf[-4:]
              if not os.path.isfile(profilesfpath):
                 profilesfpath = input("Path to Profiles File:\n")
              rhot_idb,profs_idb,units_idb = read_iterdb.read_iterdb(iterdbfpath)
              profs = np.genfromtxt(profilesfpath)
              omegator0 = interp(rhot_idb['VROT'],profs_idb['VROT'],profs[:,0])
              mi = 1.673e-27
              ee = 1.602e-19
              mref = param['units']['mref']*mi
              time_ref = param['units']['Lref']/(param['units']['Tref']*1000.0*ee/mref)**0.5
              apar_cont = 0.0
              diff = 0.0
              apar_cont_2D = npy.empty_like(apar[:,0,:],dtype='complex128')
              for i in range(param['box']['nx0']):
                  diff += npy.sum(npy.abs(gradphi[2:-2,0,i]+(omega_complex+(0.0+1.0J)*param['box']['n0_global']*omegator0[i]*time_ref)*apar[:,0,i]))
                  apar_cont += npy.sum(npy.abs((omega_complex+(0.0+1.0J)*param['box']['n0_global']*omegator0[i]*time_ref)*apar[:,0,i]))
                  apar_cont_2D[:,i] = (omega_complex+(0.0+1.0J)*param['box']['n0_global']*omegator0[i]*time_ref)*apar[:,0,i]
           else:
               diff = npy.sum(npy.abs(gradphi[2:-2,0,:]+omega_complex*apar[:,0,:]))
               apar_cont = npy.sum(npy.abs(omega_complex*apar[:,0,:]))
           phi_cont = npy.sum(npy.abs(gradphi[2:-2,0,:]))

           field_info['Epar_Cancellation'] = diff/(phi_cont+apar_cont)

           field_info['global_factor'] = npy.nan

    return field_info

def find_mode_frequency(fieldfpath,fraction=0.9,bgn_t=None,end_t=None,method='fast-mode'):
    '''
    Developed by Ehab Hassan on 2019-04-18
    Modified  by Ehab Hassan on 2019-09-03
    '''
    if   "omega" in fieldfpath[-11:]:
         if fieldfpath[-1]!="/": fieldfpath+="/"
         if "omega.dat" in fieldfpath:
            fieldflist = glob.glob(fieldfpath[:-10]+"field*")
         else:
            fieldflist = glob.glob(fieldfpath[:-11]+"field*")
    elif "field" not in fieldfpath[-11:]:
         if fieldfpath[-1]!="/": fieldfpath+="/"
         fieldflist = glob.glob(fieldfpath+"field*")
    elif "field" in fieldfpath[-11:]:
         fieldflist = [fieldfpath]

    frequency = {}
    for ifieldf in fieldflist:
        if os.path.isfile(ifieldf):
           if 'field.dat' in ifieldf:
              param   = read_parameters(ifieldf[:-9]+"parameters"+ifieldf[-4:])
           else:
              param   = read_parameters(ifieldf[:-10]+"parameters"+ifieldf[-5:])
           if 'x_local' in param['general']: x_local = param['general']['x_local']
           else:                             x_local = True

           if 'field.dat' in ifieldf:
              modeid = "mode"+ifieldf[-4:]
           else:
              modeid = "mode"+ifieldf[-5:]

           par0 = Parameters()
           if 'field.dat' in ifieldf:
              par0.Read_Pars(ifieldf[:-9]+"parameters"+ifieldf[-4:])
           else:
              par0.Read_Pars(ifieldf[:-10]+"parameters"+ifieldf[-5:])
           pars = par0.pardict
           field = fieldfile(ifieldf,pars)

           field.set_time(field.tfld[-1])

           tlist   = field.tfld
           nx      = field.nx
           ny      = field.ny
           nz      = field.nz
           nfields = field.nfields

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

           if end_t == None: end_t  = tlist[-1]
           end_t_ind = npy.argmin(abs(npy.array(tlist)-end_t))
           if bgn_t == None: bgn_t  = tlist[-1]*fraction
           bgn_t_ind = npy.argmin(abs(npy.array(tlist)-bgn_t))
           if bgn_t == None:
              bgn_t     = tlist[-1]*fraction
              bgn_t_ind = npy.argmin(abs(npy.array(tlist)-bgn_t))
              if end_t_ind-bgn_t_ind > 150:
                 bgn_t_ind = npy.size(tlist)-151
                 bgn_t = tlist[bgn_t_ind]
           ntimes    = end_t_ind-bgn_t_ind

           frequency[modeid]={}
           frequency[modeid]['ky']=param['box']['kymin']

           if method.lower() in ['quick','fast-mode','standard']:
              for tind in range(bgn_t_ind,end_t_ind):
                  field.set_time(field.tfld[tind])
                  time  = npy.append(time,tlist[tind])

                  (iphiz,iphix) = npy.unravel_index(np.argmax(abs(field.phi()[:,0,:])),(nz,nx))
                  phi  = npy.append(phi,field.phi()[iphiz,0,iphix])
                  if nfields>1:
                     (iaprz,iaprx) = npy.unravel_index(np.argmax(abs(field.apar()[:,0,:])),(nz,nx))
                     apr  = npy.append(apr,field.apar()[iaprz,0,iaprx])

              dt        = npy.diff(time)
              time      = npy.delete(time,0)

              omega_phi     = npy.log(phi/np.roll(phi,1))
              omega_phi     = npy.delete(omega_phi,0)
              omega_phi    /= dt
              omega_phi_avg = npy.average(omega_phi.imag)
              gamma_phi_avg = npy.average(omega_phi.real)

              frequency[modeid]['omega_phi'] = omega_phi_avg
              frequency[modeid]['gamma_phi'] = gamma_phi_avg

              if nfields>1:
                 omega_apr     = npy.log(apr/np.roll(apr,1))
                 omega_apr     = npy.delete(omega_apr,0)
                 omega_apr    /= dt
                 omega_apr_avg = npy.average(omega_apr.imag)
                 gamma_apr_avg = npy.average(omega_apr.real)

                 frequency[modeid]['omega_apr'] = omega_apr_avg
                 frequency[modeid]['gamma_apr'] = gamma_apr_avg

           elif method.lower() in ['slow','general-mode','thorough']:
              for tind in range(bgn_t_ind,end_t_ind+1):
                field.set_time(field.tfld[tind])
                time  = npy.append(time,tlist[tind])

                if x_local:
                  phi  = npy.zeros(nx*nz,dtype='complex128')
                  if nfields>1:
                     apar = npy.zeros(nx*nz,dtype='complex128')

                  if 'n0_global' in pars:
                     n0_global = int(pars['n0_global'])
                     q0        = float(pars['q0'])
                     phase     = -npy.e**(-2.0*npy.pi*(0.0+1.0J)*n0_global*q0)
                  else:
                     shat      = float(pars['shat'])
                     kymin     = float(pars['kymin'])
                     lx        = float(pars['lx'])
                     phase     = -npy.e**(-npy.pi*(0.0+1.0J)*shat*kymin*lx)
                  shatsgn = int(npy.sign(float(pars['shat'])))

                  for iy in range(ny):
                    for ix in range(nx//2):
                      phi[(ix+nx//2)*nz:(ix+nx//2+1)*nz]=field.phi()[:,iy,ix*shatsgn]*phase**ix
                      if ix < nx//2:
                         phi[(nx//2-ix-1)*nz:(nx//2-ix)*nz]=field.phi()[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                      if int(field.nfields)>1  and float(pars['beta'])!=0:
                         apar[(ix+nx//2)*nz:(ix+nx//2+1)*nz]=field.apar()[:,iy,ix*shatsgn]*phase**ix
                         if ix < nx//2:
                            apar[(nx//2-ix-1)*nz:(nx//2-ix)*nz]=field.apar()[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                elif not x_local:
                  if 'field.dat' in ifieldf:
                     gpars,geometry = read_geometry_global(ifieldf[:-9]+pars['magn_geometry'][1:-1]+ifieldf[-4:])
                  else:
                     gpars,geometry = read_geometry_global(ifieldf[:-10]+pars['magn_geometry'][1:-1]+ifieldf[-5:])
                  n0_global      = int(pars['n0_global'])
                  q              = geometry['q']
                  phase          = 2.0*npy.pi*(0.0+1.0J)*n0_global*q

                  phix           = npy.empty((nz+4,ny,nx),dtype = 'complex128')
                  phix[2:-2,:,:] = field.phi()
                  for iy in range(ny):
                    for ix in range(nx):
                      phix[-2,iy,ix] = phix[ 2,iy,ix]*npy.exp(-phase[ix])
                      phix[-1,iy,ix] = phix[ 3,iy,ix]*npy.exp(-phase[ix])
                      phix[ 0,iy,ix] = phix[-4,iy,ix]*npy.exp( phase[ix])
                      phix[ 1,iy,ix] = phix[-3,iy,ix]*npy.exp( phase[ix])
                  if field.nfields>1:
                     aparx = npy.empty((nz,ny,nx),dtype = 'complex128')
                     aparx = field.apar()

                  phi = npy.empty(nx*(nz+4),dtype = 'complex128')
                  if field.nfields>1:
                     apar = npy.empty(nx*nz,dtype = 'complex128')
                  if 'n0_global' in pars:
                     n0_global = int(pars['n0_global'])
                     q0        = float(gpars['q0'])
                     phase     = -npy.e**(-2.0*npy.pi*(0.0+1.0J)*n0_global*q0)
                  else:
                     shat      = float(pars['shat'])
                     kymin     = float(pars['kymin'])
                     lx        = float(pars['lx'])
                     phase     = -npy.e**(-npy.pi*(0.0+1.0J)*shat*kymin*lx)
                  shatsgn = int(npy.sign(float(gpars['shat'])))

                  phi  = npy.zeros(nx*(nz+4),dtype='complex128')
                  if nfields>1:
                     apar = npy.zeros(nx*(nz+4),dtype='complex128')

                  for iy in range(ny):
                    for ix in range(nx//2):
                      phi[(ix+nx//2)*(nz+4):(ix+nx//2+1)*(nz+4)]=phix[:,iy,ix*shatsgn]*phase**ix
                      if ix < nx//2:
                         phi[(nx//2-ix-1)*(nz+4):(nx//2-ix)*(nz+4)]=phix[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))
                      if nfields>1  and float(pars['beta'])!=0:
                         apar[(ix+nx//2)*nz:(ix+nx//2+1)*nz]=aparx[:,iy,ix*shatsgn]*phase**ix
                         if ix < nx//2:
                            apar[(nx//2-ix-1)*nz:(nx//2-ix)*nz]=aparx[:,iy,-(ix+1)*shatsgn]*phase**(-(ix+1))

                phi_t.append(phi)
                if nfields>1:
                   apr_t.append(apar)

              phi_t = npy.array(phi_t)
              max_t_ind,max_z_ind = npy.shape(phi_t)

              z0=[]
              for zind in range(max_z_ind):
                  if abs(phi_t[0,zind])==0.0:
                     z0.append(zind)
              phi_t = npy.delete(phi_t,z0,axis=1)
              max_t_ind,max_z_ind = npy.shape(phi_t)

              comega_phi_t   = npy.empty((max_t_ind-1,max_z_ind),dtype='complex128')
              weight_phi_t   = npy.empty((max_t_ind-1,max_z_ind),dtype='complex128')
              gamma_phi_avg  = npy.empty(0,dtype='float')
              omega_phi_avg  = npy.empty(0,dtype='float')
              gamma_phi_std  = npy.empty(0,dtype='float')
              omega_phi_std  = npy.empty(0,dtype='float')

              for tind in range(0,max_t_ind-1):
                  for zind in range(max_z_ind):
                      comega_phi_t[tind,zind] = npy.log(phi_t[tind+1,zind]/phi_t[tind,zind])/(tlist[tind+1]-tlist[tind])
                      weight_phi_t[tind,zind] = abs(phi_t[tind+1,zind])+abs(phi_t[tind,zind])

                  comega_phi_avg = npy.sum(comega_phi_t[tind,:]*weight_phi_t[tind,:])/npy.sum(weight_phi_t[tind,:])
                  gamma_phi_avg  = npy.append(gamma_phi_avg,comega_phi_avg.real)
                  omega_phi_avg  = npy.append(omega_phi_avg,comega_phi_avg.imag)
                  gamma_phi_std  = npy.append(gamma_phi_std,npy.sqrt(npy.sum(weight_phi_t[tind,:]*(gamma_phi_avg[tind]-comega_phi_t[tind,:].real)**2)/npy.sum(weight_phi_t[tind,:])))
                  omega_phi_std  = npy.append(omega_phi_std,npy.sqrt(npy.sum(weight_phi_t[tind,:]*(omega_phi_avg[tind]-comega_phi_t[tind,:].imag)**2)/npy.sum(weight_phi_t[tind,:])))

              gamma_phi_avg = npy.average(gamma_phi_avg[:-1])
              gamma_phi_std = npy.average(gamma_phi_std[:-1].real)
              omega_phi_avg = npy.average(omega_phi_avg[:-1])
              omega_phi_std = npy.average(omega_phi_std[:-1].real)

              frequency[modeid]['omega_phi']=omega_phi_avg
              frequency[modeid]['gamma_phi']=gamma_phi_avg

              if nfields>1:
                 apr_t = npy.array(apr_t)
                 max_t_ind,max_z_ind = npy.shape(apr_t)
                 z0=[]
                 for zind in range(max_z_ind):
                     if abs(apr_t[0,zind])==0.0:
                        z0.append(zind)
                 apr_t = npy.delete(apr_t,z0,axis=1)
                 max_t_ind,max_z_ind = npy.shape(apr_t)

                 comega_apr_t   = npy.empty((max_t_ind-1,max_z_ind),dtype='complex128')
                 weight_apr_t   = npy.empty((max_t_ind-1,max_z_ind),dtype='complex128')
                 gamma_apr_avg  = npy.empty(0,dtype='float')
                 omega_apr_avg  = npy.empty(0,dtype='float')
                 gamma_apr_std  = npy.empty(0,dtype='float')
                 omega_apr_std  = npy.empty(0,dtype='float')

                 for tind in range(0,max_t_ind-1):
                     for zind in range(max_z_ind):
                         comega_apr_t[tind,zind] = npy.log(apr_t[tind+1,zind]/apr_t[tind,zind])/(tlist[tind+1]-tlist[tind])
                         weight_apr_t[tind,zind] = abs(apr_t[tind+1,zind])+abs(apr_t[tind,zind])

                     comega_apr_avg = npy.sum(comega_apr_t[tind,:]*weight_apr_t[tind,:])/npy.sum(weight_apr_t[tind,:])
                     gamma_apr_avg  = npy.append(gamma_apr_avg,comega_apr_avg.real)
                     omega_apr_avg  = npy.append(omega_apr_avg,comega_apr_avg.imag)
                     gamma_apr_std  = npy.append(gamma_apr_std,npy.sqrt(npy.sum(weight_apr_t[tind,:]*(gamma_apr_avg[tind]-comega_apr_t[tind,:].real)**2)/npy.sum(weight_apr_t[tind,:])))
                     omega_apr_std  = npy.append(omega_apr_std,npy.sqrt(npy.sum(weight_apr_t[tind,:]*(omega_apr_avg[tind]-comega_apr_t[tind,:].imag)**2)/npy.sum(weight_apr_t[tind,:])))

                 gamma_apr_avg = npy.average(gamma_apr_avg[:-1])
                 gamma_apr_std = npy.average(gamma_apr_std[:-1].real)
                 omega_apr_avg = npy.average(omega_apr_avg[:-1])
                 omega_apr_std = npy.average(omega_apr_std[:-1].real)

                 frequency[modeid]['omega_apr']=omega_apr_avg
                 frequency[modeid]['gamma_apr']=gamma_apr_avg
           
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
            if modeinfo['omega'] < 0.0:
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
       mode_type = 'N/A'

    return mode_type


def mode_info(genefpath):
   #Developed by Ehab Hassan on 2019-03-13
    if   'parameters' in genefpath:
          paramflist = [genefpath]
    elif 'nrg' in genefpath:
         if 'dat' in genefpath[-4:]:
            paramflist = [genefpath[:-7]+'parameters.dat']
         else:
            paramflist = [genefpath[:-8]+'parameters_'+genefpath[-4:]]
    elif 'omega' in genefpath:
         if 'dat' in genefpath[-4:]:
            paramflist = [genefpath[:-9]+'parameters.dat']
         else:
            paramflist = [genefpath[:-10]+'parameters_'+genefpath[-4:]]
    else:
         if genefpath[-1] != "/": genefpath+="/"
         paramflist = sorted(glob.glob(genefpath+'parameters_????'))
         if paramflist == []:
            paramflist = glob.glob(genefpath+'parameters.dat')

    mode_info = {}

    for paramid in paramflist:
        if 'dat' in paramid:
           imode   = 'dat'
           nrgid   = paramid[:-14]+'nrg.dat'
           omegaid = paramid[:-14]+'omega.dat'
           fieldid = paramid[:-14]+'field.dat'
        else:
           imode   = int(paramid[-4:])-1
           nrgid   = paramid[:-15]+'nrg_%04d'   % (imode+1)
           omegaid = paramid[:-15]+'omega_%04d' % (imode+1)
           fieldid = paramid[:-15]+'field_%04d' % (imode+1)

        if len(paramflist)==1: imode = 0

        nrgfpath   = os.path.abspath(nrgid)
        fieldfpath = os.path.abspath(fieldid)
        omegafpath = os.path.abspath(omegaid)
        paramfpath = os.path.abspath(paramid)

        nrgdata   = read_nrg(nrgfpath)
        fielddata = read_field(fieldfpath)
        omegadata = read_omega(omegafpath)
        paramdata = read_parameters(paramfpath)
        fieldinfo = field_info(fielddata,paramdata)

        if type(omegadata['kymin']) in [list,tuple,npy.ndarray]:
           iky = omegadata['kymin'][imode]
        else:
           iky = omegadata['kymin']
        mode_info[iky] = {}

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

        if   omegadata:
             if type(omegadata['kymin']) in [list,tuple,npy.ndarray]:
                mode_info[iky]['gamma'] = omegadata['gamma'][imode]
                mode_info[iky]['omega'] = omegadata['omega'][imode]
             else:
                mode_info[iky]['gamma'] = omegadata['gamma']
                mode_info[iky]['omega'] = omegadata['omega']
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

        try:
           inrgf  = nrgdata.keys()[0]
        except TypeError:
           inrgf  = list(nrgdata.keys())[0]

        if 'e' in nrgdata[inrgf]:
           mode_info[iky]['Qem/Qes']  = abs(nrgdata[inrgf]['e']['HFluxem'][-1])
           mode_info[iky]['Qem/Qes'] /= abs(nrgdata[inrgf]['e']['HFluxes'][-1])

        if 'i' in nrgdata[inrgf]:
           mode_info[iky]['Qim/Qis']  = abs(nrgdata[inrgf]['i']['HFluxem'][-1])
           mode_info[iky]['Qim/Qis'] /= abs(nrgdata[inrgf]['i']['HFluxes'][-1])

        if 'z' in nrgdata[inrgf]:
           mode_info[iky]['Qzm/Qzs']  = abs(nrgdata[inrgf]['z']['HFluxem'][-1])
           mode_info[iky]['Qzm/Qzs'] /= abs(nrgdata[inrgf]['z']['HFluxes'][-1])

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

    flux_type = 'N/A'
   #if   all(npy.array([abs(Xi_over_Xe-1.0),abs(De_over_Xe-0.33),abs(Dz_over_Xe-0.33)])<tol):  flux_type = 'KBM'
   #elif all(npy.array([abs(Xi_over_Xe-0.1),abs(De_over_Xe-0.10),abs(Dz_over_Xe-0.10)])<tol):  flux_type = 'MTM'
   #elif all(npy.array([abs(Xi_over_Xe-0.1),abs(De_over_Xe-0.05),abs(Dz_over_Xe-0.05)])<tol):  flux_type = 'ETG'
   #elif abs(Dz_over_Xt-1.0)<tol:
   #     if (De_over_Xt>=-1.0 and De_over_Xt<=0.33) and (Xe_over_Xi>=0.25 and Xe_over_Xi<=1.0):flux_type = 'ITG/TEM'
    return flux_type


def flux_info(genefpath,timeslot=None):
   #Developed by Ehab Hassan on 2019-03-27
    if   'parameters' in genefpath:
          paramflist = [genefpath]
    elif 'nrg' in genefpath:
         if 'dat' in genefpath[-4:]:
            paramflist = [genefpath[:-7]+'parameters.dat']
         else:
            paramflist = [genefpath[:-8]+'parameters_'+genefpath[-4:]]
    else:
         if genefpath[-1] != "/": genefpath+="/"
         paramflist = sorted(glob.glob(genefpath+'parameters_????'))
         if paramflist == []:
            paramflist = glob.glob(genefpath+'parameters.dat')

    flux_info = {}

    for paramid in paramflist:
        paramfpath = os.path.abspath(paramid)
        paramdata = read_parameters(paramfpath)
        iky = paramdata['box']['kymin']
        if 'dat' in paramid:
            nrgid = paramid[:-14]+'nrg.dat'
        else:
            nrgid = paramid[:-15]+'nrg_%04d' % int(paramid[-4:])

        nrgfpath = os.path.abspath(nrgid)
        nrgdata  = read_nrg(nrgfpath,timeslot=timeslot)
        try:
           nrgid = nrgdata.keys()[0]
        except TypeError:
           nrgid = list(nrgdata.keys())[0]

        flux_info[iky]={}
        for ispecs in range(paramdata['box']['n_spec']):
             specid    = 'species'+str(ispecs+1)
             specname  = paramdata[specid]['name']
             try:
                PFlux_es  = nrgdata[nrgid][specname]['PFluxes'][-1]
                PFlux_em  = nrgdata[nrgid][specname]['PFluxem'][-1]
                PFlux     = PFlux_es + PFlux_em
                HFlux_es  = nrgdata[nrgid][specname]['HFluxes'][-1]
                if type(paramdata[specid]['temp'])==list:
                   HFlux_es -= (3./2.)*PFlux_es*max(paramdata[specid]['temp'])
                else:
                   HFlux_es -= (3./2.)*PFlux_es*paramdata[specid]['temp']
                HFlux_em  = nrgdata[nrgid][specname]['HFluxem'][-1]
                if type(paramdata[specid]['temp'])==list:
                   HFlux_em -= (3./2.)*PFlux_em*max(paramdata[specid]['temp'])
                else:
                   HFlux_em -= (3./2.)*PFlux_em*paramdata[specid]['temp']
                HFlux     = HFlux_es + HFlux_em
                Dee       = PFlux
             except IndexError:
                PFlux_es  = nrgdata[nrgid][specname]['PFluxes']
                PFlux_em  = nrgdata[nrgid][specname]['PFluxem']
                PFlux     = PFlux_es + PFlux_em
                HFlux_es  = nrgdata[nrgid][specname]['HFluxes']
                if type(paramdata[specid]['temp'])==list:
                   HFlux_es -= (3./2.)*PFlux_es*max(paramdata[specid]['temp'])
                else:
                   HFlux_es -= (3./2.)*PFlux_es*paramdata[specid]['temp']
                HFlux_em  = nrgdata[nrgid][specname]['HFluxem']
                if type(paramdata[specid]['temp'])==list:
                   HFlux_em -= (3./2.)*PFlux_em*max(paramdata[specid]['temp'])
                else:
                   HFlux_em -= (3./2.)*PFlux_em*paramdata[specid]['temp']
                HFlux     = HFlux_es + HFlux_em
                Dee       = PFlux
             if 'omn' in paramdata[specid]:
                if type(paramdata[specid]['omn'])==list:
                   Dee      /= max(paramdata[specid]['omn'])
                else:
                   Dee      /= paramdata[specid]['omn']
             if type(paramdata[specid]['dens'])==list:
                Dee      /= max(paramdata[specid]['dens'])
             else:
                Dee      /= paramdata[specid]['dens']
             Chi       = HFlux
             if 'omt' in paramdata[specid]:
                 if type(paramdata[specid]['omt'])==list:
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
             flux_info[iky][specname]={'PFluxes':PFlux_es,'PFluxem':PFlux_em,'HFluxes':HFlux_es,'HFluxem':HFlux_em,'Dee':Dee,'Chi':Chi}

        flux_info[iky]['Type']=flux_type(flux_info[iky],paramdata)

    return flux_info

def fluct_info(genefpath,timeslot=None,setParam={}):
   #Developed by Ehab Hassan on 2020-02-17
    if   'parameters' in genefpath:
          paramflist = [genefpath]
    elif 'nrg' in genefpath:
         if 'dat' in genefpath[-4:]:
            paramflist = [genefpath[:-7]+'parameters.dat']
         else:
            paramflist = [genefpath[:-8]+'parameters_'+genefpath[-4:]]
    else:
         if genefpath[-1] != "/": genefpath+="/"
         paramflist = sorted(glob.glob(genefpath+'parameters_????'))
         if paramflist == []:
            paramflist = glob.glob(genefpath+'parameters.dat')

    if 'local_central' in setParam: local_central = setParam['local_central']
    else:                           local_central = False

    if 'local_flatten' in setParam: local_flatten = setParam['local_flatten']
    else:                           local_flatten = False

    fluct_info = {}

    for paramid in paramflist:
        paramfpath = os.path.abspath(paramid)
        paramdata = read_parameters(paramfpath)
        iky = paramdata['box']['kymin']
        if 'x_local' in paramdata['general']: x_local = paramdata['general']['x_local']
        else:                                 x_local = True

        fluct_info[iky]={}

        units = units_conversion(parameters=paramdata)

        for ispecs in range(paramdata['box']['n_spec']):
            if paramdata['species'+str(ispecs+1)]['name'] == 'e':
               if 'dat' in paramid: momeid = paramid[:-14]+'mom_e.dat'
               else:                momeid = paramid[:-15]+'mom_e_%04d' % int(paramid[-4:])
               momefpath = os.path.abspath(momeid)
               if   not x_local:
                    momedata  = read_mom(momefpath,specs='e',timeslot=timeslot)
               elif local_flatten:
                    momedata  = read_mom(momefpath,specs='e',timeslot=timeslot,momfmt = 'local-flatten')
               elif local_central:
                    momedata  = read_mom(momefpath,specs='e',timeslot=timeslot,momfmt = 'local-central')
               else:
                    momedata  = read_mom(momifpath,specs='e',timeslot=timeslot,momfmt = 'original')
               momeid    = momedata.keys()[0]
               fluct_info[iky]['e']=momedata[momeid]
            if paramdata['species'+str(ispecs+1)]['name'] == 'i':
               if 'dat' in paramid: momiid = paramid[:-14]+'mom_i.dat'
               else:                momiid = paramid[:-15]+'mom_i_%04d' % int(paramid[-4:])
               momifpath = os.path.abspath(momiid)
               if   not x_local:
                    momidata  = read_mom(momifpath,specs='i',timeslot=timeslot)
               elif local_flatten:
                    momidata  = read_mom(momifpath,specs='i',timeslot=timeslot,momfmt = 'local-flatten')
               elif local_central:
                    momidata  = read_mom(momifpath,specs='i',timeslot=timeslot,momfmt = 'local-central')
               else:
                    momidata  = read_mom(momifpath,specs='i',timeslot=timeslot,momfmt = 'original')
               momiid    = momidata.keys()[0]
               fluct_info[iky]['i']=momidata[momiid]
            if paramdata['species'+str(ispecs+1)]['name'] == 'z':
               if 'dat' in paramid: momzid = paramid[:-14]+'mom_z.dat'
               else:                momzid = paramid[:-15]+'mom_z_%04d' % int(paramid[-4:])
               momzfpath = os.path.abspath(momzid)
               if   not x_local:
                    momzdata  = read_mom(momzfpath,specs='z',timeslot=timeslot)
               elif local_flatten:
                    momzdata  = read_mom(momzfpath,specs='z',timeslot=timeslot,momfmt = 'local-flatten')
               elif local_central:
                    momzdata  = read_mom(momzfpath,specs='z',timeslot=timeslot,momfmt = 'local-central')
               else:
                    momzdata  = read_mom(momifpath,specs='z',timeslot=timeslot,momfmt = 'original')
               momzid    = momzdata.keys()[0]
               fluct_info[iky]['z']=momzdata[momzid]

        if 'dat' in paramid: fieldid = paramid[:-14]+'field.dat'
        else:                fieldid = paramid[:-15]+'field_%04d' % int(paramid[-4:])

        fieldfpath = os.path.abspath(fieldid)
        if   not x_local:
             fielddata  = read_field(fieldfpath,timeslot=momedata[momeid]['t'])
        elif local_flatten:
             fielddata  = read_field(fieldfpath,timeslot=momedata[momeid]['t'],fieldfmt = 'local-flatten')
             kperp,vcurv,gradB = get_kperp(paramfpath=paramid,setParam={'local_flatten':True})
        elif local_central:
             fielddata  = read_field(fieldfpath,timeslot=momedata[momeid]['t'],fieldfmt = 'local-central')
             kperp,vcurv,gradB = get_kperp(paramfpath=paramid,setParam={'local_central':True})
        else:
             fielddata  = read_field(fieldfpath,timeslot=momedata[momeid]['t'],fieldfmt = 'original')
        fieldid    = fielddata.keys()[0]


        if 'bpar' in fielddata[fieldid]:
           fluct_info[iky]['bpar']  = fielddata[fieldid]['bpar']
        if 'apar' in fielddata[fieldid]:
           fluct_info[iky]['apar']  = fielddata[fieldid]['apar']
           fluct_info[iky]['bperp'] = npy.zeros_like(fielddata[fieldid]['apar'])
           if fluct_info[iky]['bperp'].ndim == 3:
              fluct_info[iky]['bperp'][:,0,:] = -iky*fielddata[fieldid]['apar'][:,0,:]
           else:
              fluct_info[iky]['bperp'] = -iky*fielddata[fieldid]['apar']

        fluct_info[iky]['nx']       = fielddata[fieldid]['nx']
        fluct_info[iky]['ny']       = fielddata[fieldid]['ny']
        fluct_info[iky]['nz']       = fielddata[fieldid]['nz']
        if 'xgrid' in fielddata[fieldid]:
           fluct_info[iky]['xgrid']    = fielddata[fieldid]['xgrid']
        if 'zgrid' in fielddata[fieldid]:
           fluct_info[iky]['zgrid']    = fielddata[fieldid]['zgrid']

        if 'dat' in paramid: nrgid = paramid[:-14]+'nrg.dat'
        else:                nrgid = paramid[:-15]+'nrg_%04d' % int(paramid[-4:])
        nrgfpath = os.path.abspath(nrgid)
        nrgdata  = read_nrg(nrgfpath,timeslot=momedata[momeid]['t'])
        nrgid    = nrgdata.keys()[0]

        for ispecs in range(paramdata['box']['n_spec']):
            specname = paramdata['species'+str(ispecs+1)]['name']
            fluct_info[iky][specname].update({'HFluxes':nrgdata[nrgid][specname]['HFluxes']})
            fluct_info[iky][specname].update({'HFluxem':nrgdata[nrgid][specname]['HFluxem']})
            fluct_info[iky][specname].update({'PFluxes':nrgdata[nrgid][specname]['PFluxes']})
            fluct_info[iky][specname].update({'PFluxem':nrgdata[nrgid][specname]['PFluxem']})

    return fluct_info


def calc_tau(psi=None,profilepath='',iterdbpath='',parampath=''):
   #Developed by Ehab Hassan on 2019-03-27
    if   profilepath:
         profiles,units = efittools.read_profiles_file(profilepath)
         psinorm = profiles['psinorm']
         Te      = profiles['te']
         Ti      = profiles['ti']

         Zeff    = profiles['z'][1]**2*profiles['ni']
         Zeff   += profiles['z'][0]**2*profiles['nz1']
         Zeff   /= profiles['ne']

         tau     = Zeff*Te/Ti

         taupsi  = interp1d(psinorm,tau,kind='linear')
         Zeffpsi = interp1d(psinorm,Zeff,kind='linear')
         return taupsi(psi),Zeffpsi(psi)
    elif iterdbpath:
         iterdbdata = read_iterdb.read_iterdb(iterdbpath)
    elif parampath:
         paramdata = read_parameters(parampath)
         if 'species1' in paramdata and 'species2' in paramdata and 'species3' in paramdata:
            if   'e' == paramdata['species1']['name']:
                   Te = paramdata['species1']['temp']
                   ne = paramdata['species1']['dens']
                   me = paramdata['species1']['mass']
                   qe = paramdata['species1']['charge']
            elif 'i' == paramdata['species1']['name']:
                   Ti = paramdata['species1']['temp']
                   ni = paramdata['species1']['dens']
                   mi = paramdata['species1']['mass']
                   qi = paramdata['species1']['charge']
            elif 'z' == paramdata['species1']['name']:
                   Tz = paramdata['species1']['temp']
                   nz = paramdata['species1']['dens']
                   mz = paramdata['species1']['mass']
                   qz = paramdata['species1']['charge']
            if   'e' == paramdata['species2']['name']:
                   Te = paramdata['species2']['temp']
                   ne = paramdata['species2']['dens']
                   me = paramdata['species2']['mass']
                   qe = paramdata['species2']['charge']
            elif 'i' == paramdata['species2']['name']:
                   Ti = paramdata['species2']['temp']
                   ni = paramdata['species2']['dens']
                   mi = paramdata['species2']['mass']
                   qi = paramdata['species2']['charge']
            elif 'z' == paramdata['species2']['name']:
                   Tz = paramdata['species2']['temp']
                   nz = paramdata['species2']['dens']
                   mz = paramdata['species2']['mass']
                   qz = paramdata['species2']['charge']
            if   'e' == paramdata['species3']['name']:
                   Te = paramdata['species3']['temp']
                   ne = paramdata['species3']['dens']
                   me = paramdata['species3']['mass']
                   qe = paramdata['species3']['charge']
            elif 'i' == paramdata['species3']['name']:
                   Ti = paramdata['species3']['temp']
                   ni = paramdata['species3']['dens']
                   mi = paramdata['species3']['mass']
                   qi = paramdata['species3']['charge']
            elif 'z' == paramdata['species3']['name']:
                   Tz = paramdata['species3']['temp']
                   nz = paramdata['species3']['dens']
                   mz = paramdata['species3']['mass']
                   qz = paramdata['species3']['charge']
            Zeff = (qi**2*ni + qz**2*nz)/ne
            tau  = Zeff*(Te/Ti)
         else:
             Zeff = 1.0
             tau  = 1.0
         return tau,Zeff
    else:
         print('No Valid Input')
         return 0


def units_conversion(paramfpath='',parameters={}):
   #Developed by Ehab Hassan on 2019-05-27
    if paramfpath!='' and not parameters:
         parameters = read_parameters(paramfpath.strip())

    units               = {}

    if parameters['units']:
       units['nref']       = parameters['units']['nref']*1.0e19
       units['Lref']       = parameters['units']['Lref']
       units['Bref']       = parameters['units']['Bref']
       units['Tref']       = parameters['units']['Tref']*1.60218e-19*1.0e3

       units['qref']       = 1.60218e-19
       units['mref']       = parameters['units']['mref']*1.67262e-27

       units['vref']       = npy.sqrt(1.0*units['Tref']/units['mref'])
       units['cref']       = npy.sqrt(units['Tref']/units['mref'])
       units['gyrofreq']   = units['qref']*units['Bref']/units['mref']
       units['gyroradius'] = units['cref']/units['gyrofreq']
       units['rhostar']    = units['gyroradius']/units['Lref']

       units['pref']       = units['nref']*units['Tref']
       units['Ggb']        = units['cref']*units['nref']*units['rhostar']**2
       units['Qgb']        = units['cref']*units['pref']*units['rhostar']**2
       units['Pgb']        = units['nref']*units['mref']*(units['cref']*units['rhostar'])**2

    return units

def calculate_surface_area(geometry,parameters):
    if   type(parameters)==str:
         paramdata = read_parameters(parameters.strip())
    else:
         paramdata = parameters.copy()

    if 'x_local' in paramdata['general']: x_local = paramdata['general']['x_local']
    else:                                 x_local = True

    if type(geometry)==str:
       if x_local:
          params,geomdata = read_geometry_local(geometry.strip())
       else:
          params,geomdata = read_geometry_global(geometry.strip())
    else:
       geomdata = geometry.copy()

    surface_area = 0.0

    surface_area = (2.0*npy.pi*paramdata['units']['Lref'])**2
    if x_local:
       if  paramdata['geometry']['norm_flux_projection']:
           surface_area *= npy.sum(geomdata['gjacobian']*npy.sqrt(geomdata['ggxx']))
       else:
           surface_area *= npy.sum(geomdata['gjacobian'])
       surface_area *= npy.abs(params['Cy'])
       surface_area /= paramdata['box']['nz0']
    else:
       if  paramdata['geometry']['norm_flux_projection']:
           surface_area *= npy.sum(npy.sum(geomdata['jacobian'][:,paramdata['box']['nx0']/2]*npy.sqrt(geomdata['ggxx'][:,paramdata['box']['nx0']/2]),0))
       else:
           surface_area *= npy.sum(npy.sum(geomdata['jacobian'][:,paramdata['box']['nx0']/2],0))
       surface_area *= npy.abs(geomdata['C_y'][paramdata['box']['nx0']/2])
       surface_area /= paramdata['box']['nz0']

    return surface_area


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
            if os.path.isfile('%s/nrg.dat' % runpath):
                if os.path.isfile('%s/nrg.dat'%(runpath)):
                    os.system('cp %s/nrg.dat %s/nrg_%04d'                   %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/vsp.dat'%(runpath)):
                    os.system('cp %s/vsp.dat %s/vsp_%04d'                   %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/field.dat'%(runpath)):
                    os.system('cp %s/field.dat %s/field_%04d'               %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/omega.dat'%(runpath)):
                    os.system('cp %s/omega.dat %s/omega_%04d'               %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/energy.dat'%(runpath)):
                    os.system('cp %s/energy.dat %s/energy_%04d'             %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/parameters.dat'%(runpath)):
                    os.system('cp %s/parameters.dat %s/parameters_%04d'     %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/tracer_efit.dat'%(runpath)):
                    os.system('cp %s/tracer_efit.dat %s/tracer_efit_%04d'   %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/checkpoint.dat'%(runpath)):
                    os.system('cp %s/checkpoint.dat %s/checkpoint_%04d'     %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/s_checkpoint.dat'%(runpath)):
                    os.system('cp %s/s_checkpoint.dat %s/s_checkpoint_%04d' %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/mom_i.dat'%(runpath)):
                   os.system('cp %s/mom_i.dat %s/mom_i_%04d'                %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/mom_e.dat'%(runpath)):
                   os.system('cp %s/mom_e.dat %s/mom_e_%04d'                %(runpath,destination,ntotalfiles+ifile))
                if os.path.isfile('%s/mom_z.dat'%(runpath)):
                   os.system('cp %s/mom_z.dat %s/mom_z_%04d'                %(runpath,destination,ntotalfiles+ifile))
            else:
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

   #wfh = open('%s/scan.log' %destination,'w')
   #wfh.write('#Run  | kymin     1  /Eigenvalue1 \n')
   #nlocalfiles = len(glob.glob(destination+'/omega*'))
   #for ifile in range(1,nlocalfiles+1):
   #    omega=read_omega('%s/omega_%04d' %(destination,ifile))
   #    wfh.write("%04d | %12.6E | %6.4f %6.4f \n" %(ifile,omega['kymin'][0],omega['gamma'][0],omega['omega'][0]))
   #wfh.close()

    return ntotalfiles


def read_geometry(geomfpath):
    try:
       params,geomdata = read_geometry_local(geomfpath.strip())
    except ValueError:
       params,geomdata = read_geometry_global(geomfpath.strip())
    return params,geomdata


def get_kperp(paramfpath,setParam={}):
    #Developed by Ehab Hassan on 2020-03-01
    #Modified  by Ehab Hassan on 2020-03-17
    paramdata = read_parameters(paramfpath=paramfpath)
    paramfpath = paramdata['filepath']
    if 'dat' in paramfpath[-15:]: datext = True
    else:                         datext = False

    for geomftype in ['tracer_efit','miller','s_alpha','chease']:
        if datext:
           geomfpath = paramfpath[:-14]+geomftype+'.dat'
        else:
           paramind = paramfpath[-5:]
           geomfpath = paramfpath[:-15]+geomftype+paramind
        if os.path.isfile(geomfpath): break
    params,geomdata = read_geometry(geomfpath=geomfpath)

    if 'x_local' in paramdata['general']: x_local = paramdata['general']['x_local']
    else:                                 x_local = True

    if 'local_central' in setParam:       local_central = setParam['local_central']
    else:                                 local_central = False

    if 'local_flatten' in setParam:       local_flatten = setParam['local_flatten']
    else:                                 local_flatten = False

    nx = paramdata['box']['nx0']
    if local_central: ikx_grid = [0]
    else:             ikx_grid = npy.arange(-nx//2+1,nx//2+1,dtype=int)
    if   'lx' in paramdata['box']:  lx = paramdata['box']['lx']
    elif 'lx' in paramdata['info']: lx = paramdata['info']['lx']
    else:                           lx = None

    if 'kx_center' in paramdata['box']: kx_center = paramdata['box']['kx_center']
    else:                               kx_center = 0.0

    nz = paramdata['box']['nz0']

    ky = paramdata['box']['kymin']
    if 'shat' in paramdata['geometry']: shat = paramdata['geometry']['shat']
    else:                               shat = 1.0
    dkx = 2.0*npy.pi*shat*ky

    if   x_local and local_central:
         kperp = np.zeros(nz,dtype='float128')
         vcurv = np.zeros(nz,dtype='float128')
         gradB = np.zeros(nz,dtype='float128')
    elif x_local and local_flatten:
         kperp = np.zeros(nx*nz,dtype='float128')
         vcurv = np.zeros(nx*nz,dtype='float128')
         gradB = np.zeros(nx*nz,dtype='float128')
    else:
         kperp = np.zeros((nx,nz),dtype='float128')
         vcurv = np.zeros((nx,nz),dtype='float128')
         gradB = np.zeros((nx,nz),dtype='float128')

    if x_local:
       gxx    = geomdata['ggxx']
       gxy    = geomdata['ggxy']
       gyy    = geomdata['ggyy']
       gyz    = geomdata['ggyz']
       gxz    = geomdata['ggxz']
       gzz    = geomdata['ggzz']
       dBdx   = geomdata['gdBdx']
       dBdy   = geomdata['gdBdy']
       dBdz   = geomdata['gdBdz']
       Bfield = geomdata['gBfield']
    else:
       gxx    = geomdata['gxx']
       gxy    = geomdata['gxy']
       gyy    = geomdata['gyy']
       gyz    = geomdata['gyz']
       gxz    = geomdata['gxz']
       gzz    = geomdata['gzz']
       dBdx   = geomdata['dBdx']
       dBdy   = geomdata['dBdy']
       dBdz   = geomdata['dBdz']
       Bfield = geomdata['Bfield']

    gamma1 = gxx*gyy-gxy**2
    gamma2 = gxx*gyz-gxy*gxz
    gamma3 = gxy*gyz-gyy*gxz

    gradBx =-dBdy-(gamma2/gamma1)*dBdz
    gradBy = dBdx-(gamma3/gamma1)*dBdz

    if geomftype == 's_alpha':
        gradBx = gradBx/Bfield
        gradBy = gradBy/Bfield

  # for i in ikx_grid:
  #     kx        = i*dkx + kx_center
  #     loc_kperp = npy.sqrt(gxx*kx**2 + 2.0*gxy*kx*ky + gyy*ky**2)
  #     loc_gradB =-(kx*gradBx + ky*gradBy)/Bfield
  #     loc_vcurv = loc_gradB + ky*paramdata['geometry']['dpdx_pm']/Bfield**2/2.0

  #     if x_local:
  #        kperp[(i-ikx_grid[0])*nz:(i-ikx_grid[0]+1)*nz] = loc_kperp
  #        vcurv[(i-ikx_grid[0])*nz:(i-ikx_grid[0]+1)*nz] = loc_vcurv
  #        gradB[(i-ikx_grid[0])*nz:(i-ikx_grid[0]+1)*nz] = loc_gradB
  #     else:
  #        kperp = loc_kperp
  #        vcurv = loc_vcurv
  #        gradB = loc_gradB

    for ix in range(nx//2):
        kx        = ix*dkx + kx_center
        if ix < nx//2: kx     = -ix*dkx + kx_center

        loc_kperp = npy.sqrt(gxx*kx**2 + 2.0*gxy*kx*ky + gyy*ky**2)
        loc_gradB =-(kx*gradBx + ky*gradBy)/Bfield
        loc_vcurv = loc_gradB + ky*paramdata['geometry']['dpdx_pm']/Bfield**2/2.0

        if x_local:
           kperp[(ix+nx//2)*nz:(ix+nx//2+1)*nz] = loc_kperp
           if ix < nx//2:
              kperp[(nx//2-ix-1)*nz:(nx//2-ix)*nz] = loc_kperp
           vcurv[(ix+nx//2)*nz:(ix+nx//2+1)*nz] = loc_vcurv
           if ix < nx//2:
              vcurv[(nx//2-ix-1)*nz:(nx//2-ix)*nz] = loc_vcurv
           gradB[(ix+nx//2)*nz:(ix+nx//2+1)*nz] = loc_gradB
           if ix < nx//2:
              gradB[(nx//2-ix-1)*nz:(nx//2-ix)*nz] = loc_gradB
        else:
           kperp = loc_kperp
           vcurv = loc_vcurv
           gradB = loc_gradB

    if geomftype == 's_alpha':
        if 'amhd' in paramdata['geometry']: amhd = paramdata['geometry']['amhd']
        else:                               amhd = 0.

        z_grid    = npy.linspace(-1.,1.,nz,endpoint = False)
        Kx0       =-npy.sin(z_grid*npy.pi)/paramdata['geometry']['major_R']
        Ky0       =-(npy.cos(z_grid*npy.pi)+npy.sin(z_grid*npy.pi)*(shat*z_grid*npy.pi-amhd*npy.sin(z_grid*npy.pi)))
        Ky0      /= paramdata['geometry']['major_R']
        omega_d0  =-(Kx0*kx_center+Ky0*ky)
        omega_d00 = omega_d0+amhd/paramdata['geometry']['q0']**2/paramdata['geometry']['major_R']/2.*ky/Bfield**2
        gxx0 = 1.0
        gxy0 = shat*z_grid*npy.pi-amhd*npy.sin(z_grid*npy.pi)
        gyy0 = 1.0+(shat*z_grid*npy.pi-amhd*npy.sin(z_grid*npy.pi))**2
        kperp0 = npy.sqrt(gxx0*kx_center**2+2.*gxy0*kx_center*ky+gyy0*ky**2)

    if 'plot' in setParam: plot = setParam['plot']
    else:                  plot = False
    if plot:
       plt.plot(kperp,label='kperp')
       plt.title('entire simulation domain')
       if geomftype == 's_alpha' and plot and center_only:
           plt.plot(kperp0,label='check')
           plt.title('local_central')
       plt.legend()
       plt.show()
       plt.plot(curv,label='curv')
       plt.plot(gradB,label='gradB')
       plt.title('entire simulation domain')
       if geomftype == 's_alpha' and plot and center_only:
           plt.plot(omega_d0,label='check')
           plt.plot(omega_d00,label='check')
           plt.title('center only')
       plt.legend()
       plt.show()

    return kperp,vcurv,gradB


def get_zgrid(paramfpath,setParam={}):
    #Modified by Ehab Hassan on 2020-03-18
    #following Xing work.
    paramdata = read_parameters(paramfpath=paramfpath)
    paramfpath = paramdata['filepath']
    if 'dat' in paramfpath[-15:]: datext = True
    else:                         datext = False

    if 'x_local' in paramdata['general']: x_local = paramdata['general']['x_local']
    else:                                 x_local = True

    for geomftype in ['tracer_efit','miller','s_alpha','chease']:
        if datext:
           geomfpath = paramfpath[:-14]+geomftype+'.dat'
        else:
           paramind = paramfpath[-5:]
           geomfpath = paramfpath[:-15]+geomftype+paramind
        if os.path.isfile(geomfpath): break
    params,geomdata = read_geometry(geomfpath=geomfpath)

    if 'center_only' in setParam: center_only = setParam['center_only']
    else:                         center_only = False

    nx = paramdata['box']['nx0']
    nz = paramdata['box']['nz0']
    zgrid_even_center = npy.linspace(-1.,1.,nz,endpoint=False)

    if   'gBfield' in geomdata: gBfield = geomdata['gBfield']
    elif 'Bfield'  in geomdata: gBfield = geomdata['Bfield']

    if   'gjacobian' in geomdata: gjacobian = geomdata['gjacobian']
    elif 'jacobian'  in geomdata: gjacobian = geomdata['jacobian']

    if center_only: ikx_grid = [0]
    else:           ikx_grid = npy.arange(-nx//2+1,nx//2+1)

    if not center_only:
        if nx%2 == 1:
           zgrid_even = npy.linspace(-nx,nx,nx*nz,endpoint=False)
        else:
           zgrid_even = npy.linspace(-(nx-1),(nx+1),nx*nz,endpoint=False)

    if 'edge_opt' in paramdata['geometry']:
       if 'edge_opt' in setParam: edge_opt = float(setParam['edge_opt'])
       else:                      edge_opt = paramdata['geometry']['edge_opt'] 

       zgrid_edge = npy.zeros(nx*nz,dtype='float128')
       N = npy.arcsinh(edge_opt*zgrid_even_center[0]*npy.pi)/zgrid_even_center[0]/npy.pi
       zgrid_edge_center = 1.0/edge_opt*npy.sinh(N*zgrid_even_center*npy.pi)/npy.pi

       dz = npy.zeros(nz,dtype='float128')
       for i in npy.arange(nz/2+1,nz):
           dz[i] = zgrid_edge_center[i]-zgrid_edge_center[i-1]
       for i in npy.arange(nz/2-1,-1,-1):
           dz[i] = zgrid_edge_center[i+1]-zgrid_edge_center[i]

       for i in ikx_grid:
           zgrid_edge[(i-ikx_grid[0])*nz:(i-ikx_grid[0]+1)*nz] = i*2.0+zgrid_edge_center
       if center_only: zgrid = zgrid_edge_center
       else:           zgrid = zgrid_edge
    else:
       edge_opt = 0.0

       if 'edge_opt' in paramdata['geometry'] and paramdata['geometry']['edge_opt'] != 0:
           print('Warning:edge_opt ~= 0 code for zgrid is not ready.')
       if center_only: zgrid = zgrid_even_center
       else:           zgrid = zgrid_even
       dz = np.ones(nz, dtype = 'float128')*2./nz

    jacobian_center = 1.0/npy.pi/gjacobian/gBfield

    if x_local:
       if   not center_only:
            jacobian = npy.zeros(nx*nz,dtype='float128')
            for i in ikx_grid:
                jacobian[(i-ikx_grid[0])*nz:(i-ikx_grid[0]+1)*nz] = jacobian_center
       elif center_only:
            jacobian = jacobian_center
    else:
            jacobian = jacobian_center

    if 'plot' in setParam: plot = setParam['plot']
    else:                  plot = False
    if plot:
        plt.plot(zgrid, label = 'zgrid')
        plt.legend(loc=2)
        plt.show()
        plt.plot(jacobian, label = 'jacobian')
        plt.legend(loc=2)
        plt.show()

    return zgrid,jacobian


def eigenfunction_average_bessel(kperp, omegaD, field, jacobian, zgrid, mass_ratio=1, charge=1):

    alpha = 2.0/3.0
    kperp = kperp*npy.sqrt(mass_ratio)/abs(charge)
    bessel_factor = 1.0 / npy.sqrt(1.0 + 2.0 * (kperp**2 + npy.pi * alpha * kperp**4) / (1.0 + alpha * kperp**2))

    numer = 0.
    denom = 0.
    
    for i in npy.arange(npy.size(field)-1):
        loc_numer  = (kperp[i]**2 * abs(field[i])**2 + kperp[i+1]**2 * abs(field[i+1])**2) / 2.0
        loc_numer *= (zgrid[i+1] - zgrid[i]) / jacobian[i]

        numer += loc_numer

        loc_denom  = (abs(field[i])**2 + abs(field[i+1])**2)/2.0
        loc_denom *= (zgrid[i+1] - zgrid[i]) / jacobian[i]

        denom += loc_denom

    kperp_avg = npy.sqrt(numer/denom)

    numer = 0.
    denom = 0.
    for i in npy.arange(len(field)-1):
        loc_numer  = (omegaD[i] * abs(field[i])**2 * bessel_factor[i] + omegaD[i+1] * abs(field[i+1])**2 * bessel_factor[i+1])/2.0
        loc_numer *= (zgrid[i+1] - zgrid[i]) / jacobian[i]

        numer += loc_numer

        loc_denom  = (abs(field[i])**2 * bessel_factor[i] + abs(field[i+1])**2 * bessel_factor[i+1])/2.0
        loc_denom *= (zgrid[i+1] - zgrid[i]) / jacobian[i]

        denom += loc_denom

    omegaD_avg = numer/denom

    return kperp_avg,omegaD_avg


def get_magnetic_drift_frequency(paramfpath,setParam={}):

    if "dat" in paramfpath: paramind = ".dat"
    else:                   paramind = paramfpath[-5:]
    paramdata = read_parameters(paramfpath)

    if 'center_only' in setParam:   center_only = setParam['center_only']
    else:                           center_only = False
    if 'local_central' in setParam: local_central = setParam['local_central']
    else:                           local_central = False
    if 'local_flatten' in setParam: local_flatten = setParam['local_flatten']
    else:                           local_flatten = False

    setparam                   = {}
    if center_only:   setparam['center_only']   = center_only
    if local_central: setparam['local_central'] = local_central
    if local_flatten: setparam['local_flatten'] = local_flatten

    kperp, omega_curvB, omega_gradB = get_kperp(paramfpath, setParam=setparam)
    zgrid, jacobian = get_zgrid(paramfpath, setParam=setparam)

    if 'dat' in paramfpath[-15:]:
        fieldfpath = paramfpath[:-14]+'field.dat'
    else:
        fieldfpath = paramfpath[:-15]+'field'+paramfpath[-5:]

    if center_only: fielddata = read_field(fieldfpath=fieldfpath,fieldfmt='local-central')
    else:           fielddata = read_field(fieldfpath=fieldfpath,fieldfmt='local-flatten')

    mainKey = list(fielddata.keys())[0]
    if 'phi' in setParam and setParam['phi']:
        field = fielddata[mainKey]['phi']
    else:
        field = fielddata[mainKey]['apar']
    fldmx = field[npy.argmax(abs(field))]
    field = field / fldmx
    omegaD = omega_curvB
    kperp_avg, omegaD_avg = eigenfunction_average_bessel(kperp, omegaD, field, jacobian, zgrid)

   #fldamp = 1.0
   #plt.plot(zgrid,omegaD,label="$\\omega_{\\nabla{B}}$")
   #plt.plot(zgrid,kperp,label="$k_{\\perp}$")
   #if "phi" in setParam and setParam["phi"]:
   #    plt.plot(zgrid,fldamp*abs(field),label="|$\\phi$|")
   #    plt.plot(zgrid,fldamp*npy.real(field),label="Re($\\phi$)")
   #    plt.plot(zgrid,fldamp*npy.imag(field),label="Im($\\phi$)")
   #    plt.ylabel("$\\omega_{\\nabla{B}}$,$\\phi$")
   #else:
   #    plt.plot(zgrid,fldamp*abs(field),label="|$A_{\\parallel}$|")
   #    plt.plot(zgrid,fldamp*npy.real(field),label="Re($A_{\\parallel}$)")
   #    plt.plot(zgrid,fldamp*npy.imag(field),label="Im($A_{\\parallel}$)")
   #    plt.ylabel("$\\omega_{\\nabla{B}}$,$A_{\\parallel}$")
   #plt.xlabel("Z")
   #plt.title("$n_{\\phi}$ = %02d" % paramdata['box']['n0_global'])
   #plt.legend()
   #plt.show()

    return kperp_avg, omegaD_avg


def get_kpar(paramfpath,fieldname='phi',setParam={}):
    #Modified by Ehab Hassan on 2020-03-18
    #Based on Xing work.
    paramdata = read_parameters(paramfpath=paramfpath)
    paramfpath = paramdata['filepath']
    if 'dat' in paramfpath[-15:]: datext = True
    else:                         datext = False

    if 'x_local' in paramdata['general']: x_local = paramdata['general']['x_local']
    else:                                 x_local = True

    if 'center_only' in setParam: center_only = setParam['center_only']
    else:                         center_only = False


    if 'scale_field' in setParam: scale_field = setParam['scale_field']
    else:                         scale_field = False

    if datext:
       fieldfpath = paramfpath[:-14]+'field.dat'
    else:
       paramind = paramfpath[-5:]
       fieldfpath = paramfpath[:-15]+'field'+paramind

    nx = paramdata['box']['nx0']
    nz = paramdata['box']['nz0']

    if x_local:
       if center_only:
          ikx_grid  = [0]
          fielddata = read_field(fieldfpath=fieldfpath,fieldfmt='local-central')
       else:
          ikx_grid  = npy.arange(-nx/2+1,nx/2+1)
          fielddata = read_field(fieldfpath=fieldfpath,fieldfmt='local-flatten')
    else:
          ikx_grid  = npy.arange(-nx/2+1,nx/2+1)
          fielddata = read_field(fieldfpath=fieldfpath)

    if 'zrange' in setParam:
       zrng = setParam['zrange']
       if   len(zrng)==1: zbgn = zrng[0]; zend=-1
       elif len(zrng)==2: zbgn = zrng[0]; zend = zrng[1]
    else:
       zbgn = 0.95; zend = 0.95

    zgrid,jacobian = get_zgrid(paramfpath=paramfpath,setParam={})

    if   PYTHON3:
         fieldfnID = list(fielddata.keys())[0]
    elif PYTHON2:
         fieldfnID = fielddata.keys()[0]

    if fieldname=='phi': field = fielddata[fieldfnID]['phi']
    else:                field = fielddata[fieldfnID]['apar']

    if scale_field:
       field =field/npy.max(abs(field))

    if x_local:
       dfielddz = npy.empty(npy.size(field),dtype='complex128')
       for i in range(len(field)-1):
           dfielddz[i] = (field[i+1]-field[i])/(zgrid[i+1]-zgrid[i])*jacobian[i]
    else:
       dfielddz = npy.empty((nx,nz),dtype='complex128')
       for i in range(nx):
           for j in range(nz):
               dfielddz[i,j] = (field[i+1,0,j]-field[i,0,j])/(zgrid[i+1]-zgrid[i])*jacobian[i,j]

    zbgn_ind = npy.argmin(abs(zgrid-zbgn))
    zend_ind = npy.argmin(abs(zgrid-zend))

    avgfdz  = 0.
    avgf2dz = 0.
    for i in range(zbgn_ind,zend_ind+1):
        avgf2dz = avgf2dz + 0.5*(abs(dfielddz[i])**2+abs(dfielddz[i+1])**2)*(zgrid[i+1]-zgrid[i])/jacobian[i]
        avgfdz  = avgfdz  + 0.5*(abs(field[i])**2+abs(field[i+1])**2)*(zgrid[i+1]-zgrid[i])/jacobian[i]
    kpar = npy.sqrt(avgf2dz/avgfdz)

    if 'plot' in setParam: plot = setParam['plot']
    else:                  plot = False
    if plot:
        plt.plot(zgrid,npy.abs(dfielddz), label='abs d' +fieldname+'/dz')
        plt.plot(zgrid,npy.real(dfielddz),label='real d'+fieldname+'/dz')
        plt.plot(zgrid,npy.imag(dfielddz),label='imag d'+fieldname+'/dz')
        plt.legend()
        plt.xlabel('z')
        plt.show()
 
    return kpar

def get_plasma_info(genefpath='',setParam={},timeslot=None):
   #Developed by Ehab Hassan on 2020-04-14
    if   'parameters' in genefpath:
          paramflist = [genefpath]
    elif 'nrg' in genefpath:
         if 'dat' in genefpath[-4:]:
            paramflist = [genefpath[:-7]+'parameters.dat']
         else:
            paramflist = [genefpath[:-8]+'parameters_'+genefpath[-4:]]
    else:
         if genefpath[-1] != "/": genefpath+="/"
         paramflist = sorted(glob.glob(genefpath+'parameters_????'))
         if paramflist == []:
            paramflist = genefpath+'parameters.dat'

    if 'dat' in paramflist[0][-4:]: datext = True
    else:                           datext = False

    mu0 = 4.0e-7*npy.pi

    plasma_info = {}

    for paramid in paramflist:
        paramdata = read_parameters(paramfpath=paramid)

        if datext: 
                   nrgid   = paramid[:-14]+'nrg.dat'
                   momeid  = paramid[:-14]+'mom_e.dat'
                   momiid  = paramid[:-14]+'mom_i.dat'
                   momzid  = paramid[:-14]+'mom_z.dat'
                   fieldid = paramid[:-14]+'field.dat'
        else:
                   nrgid   = paramid[:-15]+'nrg'  +paramfpath[-5:]
                   momeid  = paramid[:-15]+'mom_e'+paramfpath[-5:]
                   momiid  = paramid[:-15]+'mom_i'+paramfpath[-5:]
                   momzid  = paramid[:-15]+'mom_z'+paramfpath[-5:]
                   fieldid = paramid[:-15]+'field'+paramfpath[-5:]

        momedata  = read_mom(momfpath=momeid,specs='e',timeslot=timeslot)
        momidata  = read_mom(momfpath=momiid,specs='i',timeslot=timeslot)
        momzdata  = read_mom(momfpath=momzid,specs='z',timeslot=timeslot)

        momtslot  = momedata[momeid]['t']

        nrgdata   = read_nrg(nrgfpath=nrgid,timeslot=momtslot)
        fielddata = read_field(fieldfpath=fieldid,timeslot=momtslot)

        nrgtslot  = nrgdata[nrgid]['time']    
        fieldtslot= fielddata[fieldid]['t']


       #plasma_info['ky'] = paramdata['box']['kymin']
       #plasma_info['n0'] = paramdata['box']['n0_global']

       #print(nrgdata[nrgid]['e'].keys())

       #mp = 1.6726e-27
       #Mi = paramdata['units']['mref']/mp
       #plasma_info['alfven_drift'] = paramdata['units']['Bref']/npy.sqrt(mu0*npy.sqrt(nrgdata[nrgid]['i']['n'])*Mi)
       #plasma_info['diamag_freq'] = 

    return plasma_info


def get_vAlfven(parampath='',parameters={},setParam={}):
    if   parampath and not parameters:
         parameters = read_parameters(paramfpath=parampath)
    elif not parameters:
         raise IOError('Parameters can not be retrieved from the inputs. EXIT!')
         sys.exit()

    mu0 = 4.0e-7*npy.pi

    if 'e' in setParam and setParam['e'] == True: inc_e = True
    else:                                         inc_e = False
    if 'i' in setParam and setParam['i'] == True: inc_i = True
    else:                                         inc_i = False
    if 'z' in setParam and setParam['z'] == True: inc_z = True
    else:                                         inc_z = False

    normvals = units_conversion(parameters=parameters)

    mi   = parameters['species1']['mass']*normvals['mref']
    ni   = parameters['species1']['dens']*normvals['nref']
    me   = parameters['species2']['mass']*normvals['mref']
    ne   = parameters['species2']['dens']*normvals['nref']
    mz   = parameters['species3']['mass']*normvals['mref']
    nz   = parameters['species3']['dens']*normvals['nref']

    Bref = parameters['units']['Bref']

    if   inc_e and inc_i and inc_z:
         vAlfven  = Bref/npy.sqrt(mu0*(ni*mi+ne*me+nz*mz))
    elif inc_e and inc_i:
         vAlfven  = Bref/npy.sqrt(mu0*(ni*mi+ne*me))
    elif inc_i:
         vAlfven  = Bref/npy.sqrt(mu0*(ni*mi))
    elif inc_e:
         vAlfven  = Bref/npy.sqrt(mu0*(ne*me))
    elif inc_z:
         vAlfven  = Bref/npy.sqrt(mu0*(nz*mz))
    else:
         vAlfven  = Bref/npy.sqrt(mu0*(ni*mi+ne*me+nz*mz))

    return vAlfven


def n0_to_ky(parampath='',parameters={},setParam={}):
    if   parampath and not parameters:
         parameters = read_parameters(paramfpath=parampath)
    elif not parameters:
         raise IOError('Parameters can not be retrieved from the inputs. EXIT!')
         sys.exit()

    if    'q'    in setParam:   q = setParam['q0']
    elif  'q0'   in parameters: q = parameters['q0']
    else: raise ValueError('q-profile not found. Exit!');    sys.exit()

    if    'x'    in setParam:   x = setParam['x']
    elif  'x0'   in parameters: x = parameters['x0']
    else: raise ValueError('x-location not found. Exit!');   sys.exit()

    if    'a'    in setParam:   a = setParam['a']
    elif  'Lref' in parameters: a = parameters['Lref']
    else: raise ValueError('minor radius not found. Exit!'); sys.exit()

    if    'n'    in setParam:   n = setParam['n']
    elif  'n0'   in parameters: n = parameters['n0']
    else: raise ValueError('mode number not found. Exit!');  sys.exit()

    if    'gyroR' in setParam:
          gyroR = setParam['gyroradius']
    elif  parameters:
          normvals = units_conversion(parameters=parameters)
          gyroR = normvals['gyroradius']
    else: raise ValueError('gyro-radius not found. Exit!');  sys.exit()

    ky = n*q*gyroR/a/x

    return ky


def main():

    scanlist = []
    for iscan in sys.argv[1:]:
        scanlist.append(iscan)
    merge_runs(runspathlist=scanlist,destination='./')
    sys.exit()

    if len(sys.argv) > 1:
        if "dat" in sys.argv[1]: paramfpath = "parameters.dat"
        else:                    paramfpath = "parameters_"+sys.argv[1]
    setparam = {}
    setparam['phi'] = True
  # setparam['center_only']   = True
  # setparam['local_central'] = True
    setparam['local_flatten'] = True
    kperp, omegaD = get_magnetic_drift_frequency(paramfpath,setParam=setparam)
    conv_units = units_conversion(paramfpath=paramfpath)
    Lref = conv_units['Lref']
    cref = conv_units['cref']
    print(kperp,omegaD*(cref/Lref)*1.0e-6)

    sys.exit()

    fieldfname = sys.argv[1]
    fieldpath  = os.path.abspath(fieldfname)
    fielddata  = read_field(fieldfpath=fieldpath)
    fieldinfo  = field_info(field=fielddata)
    print(fieldinfo['Epar_Cancellation'])
    sys.exit()

    paramfname = sys.argv[1]
    paramfpath = os.path.abspath(paramfname)
    convert_extension(genefpath=paramfname,targetext='')
    sys.exit()
    kpar = get_kpar(paramfpath=paramfpath,setParam={'center_only':True,'scale_field':True,'plot':True})
    print(kpar)
    sys.exit()
    plasmainfo = get_plasma_info(genefpath=paramfpath)
    kperp,vcurv,gradB = get_kperp(paramfpath=paramfpath,setParam={'center_only':True})
    zgrid,jacobian = get_zgrid(paramfpath=paramfpath,setParam={'center_only':False})
    kpar = get_kpar(paramfpath=paramfpath,setParam={'center_only':False,'scale_field':True})
    
if __name__ == "__main__":
    main()



#def get_eigenfunctions(paramfpath,timeslot=-1,Normalize=True,setParam={}):
#    #Modified by Ehab Hassan on 2020-03-18
#    #following Xing work.
#    paramdata = read_parameters(paramfpath=paramfpath)
#    paramfpath = paramdata['filepath']
#    if 'dat' in paramfpath[-15:]: datext = True
#    else:                         datext = False
#
#    if 'center_only' in setParam: center_only = setParam['center_only']
#    else:                         center_only = False
#
#    if 'smooth_field' in setParam: smooth_field = setParam['smooth_field']
#    else:                          smooth_field = False
#
#    if datext:
#       fieldfpath = paramfpath[:-14]+'field.dat'
#    else:
#       paramind = paramfpath[-5:]
#       fieldfpath = paramfpath[:-15]+'field'+paramind
#
#    if center_only:
#       ikx_grid  = [0]
#       phi       = np.zeros(nz,dtype='complex128')
#       apar      = np.zeros(nz,dtype='complex128')
#       fielddata = read_field(fieldfpath=fieldfpath,Normalize=Normalize,fieldfmt='local-central')
#    else:
#       ikx_grid  = np.arange(-nx/2+1,nx/2+1)
#       phi       = np.zeros(nx*nz,dtype='complex128')
#       apar      = np.zeros(nx*nz,dtype='complex128')
#       fielddata = read_field(fieldfpath=fieldfpath,Normalize=Normalize,fieldfmt='local-flatten')
#
#    nz = paramdata['box']['nz0']
#    nx = paramdata['box']['nx0']
#
#    if 'n0_global' in paramdata['box']:
#        phase_fac = -npy.e**(-2.0*npy.pi*(0.0+1.0J)*paramdata['box']['n0_global']*paramdata['geometry']['q0'])
#    else:
#        phase_fac = -1.0
#
#    if float(pars['shat']) > 0.:
#        for i in ikx_grid:
#            this_phi = field.phi()[:,0,i]*phase_fac**i
#            phi[(i-ikx_grid[0])*nz:(i-ikx_grid[0]+1)*nz]=this_phi
#            if int(pars['n_fields']) > 1 and float(pars['beta']) !=0:
#                this_apar = field.apar()[:,0,i]*phase_fac**i
#                apar[(i-ikx_grid[0])*nz:(i-ikx_grid[0]+1)*nz]=\
#                                                                 this_apar
#    else:
#        for i in ikx_grid:
#            this_phi = field.phi()[:,0,-i]*phase_fac**i
#            phi[(i-ikx_grid[0])*nz:(i-ikx_grid[0]+1)*nz]=this_phi
#            if pars['n_fields'] > 1 and pars['beta'] !=0:
#                this_apar = field.apar()[:,0,-i]*phase_fac**i
#                apar[(i-ikx_grid[0])*nz:(i-ikx_grid[0]+1)*nz]=this_apar
#
#    # Normalize phi and apar by highest value so that the peak abs val = 1
#    if scale_field:
#        phi = phi/np.max(abs(field.phi()[:,0,:]))
#        if int(pars['n_fields']) > 1 and float(pars['beta']) !=0:
#            apar = apar/np.max(abs(field.apar()[:,0,:]))
#    if plot:
#        if (timeslot == -1):
#            figTitle='t = '+ str(field.tfld[setTime])
#        else:
#            figTitle='t = '+ str(field.tfld[isetTime])
#        if center_only:
#            figTitle = figTitle+' center only'
#        else:
#            figTitle = figTitle+' entire simulation domain'
#        plt.plot(np.real(phi),label='Re(phi)')
#        plt.plot(np.imag(phi),label='Im(phi)')
#        plt.plot(abs(phi),label='abs(phi)')
#        plt.title(figTitle)
#        plt.legend()
#        plt.show()
#    if plot and fielddata['nfields']>1 and paramdata['geometry']['beta'] !=0:
#        plt.plot(np.real(apar),label='Re(apar)')
#        plt.plot(np.imag(apar),label='Im(apar)')
#        plt.plot(abs(apar),label='abs(apar)')
#        plt.title(figTitle)
#        plt.legend()
#        plt.show()
#    return phi,apar

