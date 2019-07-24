#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import numpy as npy
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfh

from read_write_geometry import *
from scipy.interpolate import interp1d

import genetools
import efittools
import read_iterdb


def create_k_grid(x_grid):
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
    inds = []
    for i,character in enumerate(inlist):
        if character==item: inds.append(i)
    return inds

def str2bool(vin):
    return (vin.strip()).lower() in ('t','.t.','true','.true.')


def plot_nrg(nrgdata,reportpath='',mergeplots=False):
   #Developed by Ehab Hassan on 2019-07-22
    inrgf=nrgdata.keys()[0]
    if    inrgf[-3:] == 'dat': inrgfpath = inrgf[:-7]
    else:                      inrgfpath = inrgf[:-8]
    if 'report' not in reportpath:
       if not os.path.isdir(inrgfpath+"report"):
          os.system('mkdir '+inrgfpath+"report")
          reportpath = inrgfpath+"report/"
       else:
          reportpath = inrgfpath+"report/"
    elif reportpath[-1] != "/":
       reportpath += "/"

    time      = nrgdata[inrgf]['time']
    specstype = nrgdata[inrgf].keys()
    specstype.remove('time')

    for inrgf in nrgdata:
        if 'dat' in inrgf:
           isimfpath = inrgf[:-7]
           inrgfpath = inrgf[-7:]
           inrgfext  = inrgf[-4:]
        else:
           isimfpath = inrgf[:-8]
           inrgfpath = inrgf[-8:]
           inrgfext  = inrgf[-5:]

        parameters = genetools.read_parameters("%sparameters%s" %(isimfpath,inrgfext))
        if 'n0_global' in parameters['box']:
           titletxt = '(k_y=%6.3f,n_0=%5d)' % (parameters['box']['kymin'],parameters['box']['n0_global'])
        else:
           titletxt = '(k_y=%6.3f)' % (parameters['box']['kymin'])

        time = nrgdata[inrgf]['time']
        for ispecs in specstype:
            nfig   = plt.figure('n-'+inrgfpath)
            axhand = nfig.add_subplot(1,1,1)
            axhand.plot(time,nrgdata[inrgf][ispecs]['n'],label=ispecs)
            axhand.set_title('n%s' %(titletxt))
            axhand.set_xlabel('Time')
            axhand.set_ylabel('Density')
            axhand.legend()

            uparafig = plt.figure('upara-'+inrgfpath)
            axhand = uparafig.add_subplot(1,1,1)
            axhand.plot(time,nrgdata[inrgf][ispecs]['upara'],label=ispecs)
            axhand.set_title('$U_{||}%s$' %(titletxt))
            axhand.set_xlabel('Time')
            axhand.set_ylabel('Parallel Velocity')
            axhand.legend()

            if mergeplots:
               Tfig = plt.figure('T-'+inrgfpath)
               axhand = Tfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['Tpara'],linestyle='-', label='$T_{\\parallel,%s}$' % ispecs)
               axhand.plot(time,nrgdata[inrgf][ispecs]['Tperp'],linestyle='--',label='$T_{\\perp,%s}$' % ispecs)
               axhand.set_title('$T_{\\parallel,\\perp}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Temperature')
               axhand.set_yscale('log')
               axhand.legend()
            else:
               Tparafig = plt.figure('Tpara-'+inrgfpath)
               axhand = Tparafig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['Tpara'],label=ispecs)
               axhand.set_title('$T_{\\parallel}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Parallel Temperature')
               axhand.legend()

               Tperpfig = plt.figure('Tperp-'+inrgfpath)
               axhand = Tperpfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['Tperp'],label=ispecs)
               axhand.set_title('$T_{\\perp}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Transverse Temperature')
               axhand.legend()

            if mergeplots:
               PFluxfig = plt.figure('PFlux-'+inrgfpath)
               axhand = PFluxfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['PFluxes'],linestyle='-', label='$\\Gamma_{es,%s}$' % ispecs)
               axhand.plot(time,nrgdata[inrgf][ispecs]['PFluxem'],linestyle='--',label='$\\Gamma_{em,%s}$' % ispecs)
               axhand.set_title('$\\Gamma_{es,em}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Particle Flux')
               axhand.set_yscale('log')
               axhand.legend()
            else:
               PFluxesfig = plt.figure('PFluxes-'+inrgfpath)
               axhand = PFluxesfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['PFluxes'],label=ispecs)
               axhand.set_title('$\\Gamma_{es}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Electrostatic Particle Flux')
               axhand.legend()

               PFluxemfig = plt.figure('PFluxem-'+inrgfpath)
               axhand = PFluxemfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['PFluxem'],label=ispecs)
               axhand.set_title('$\\Gamma_{em}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Electromagnetic Particle Flux')
               axhand.legend()

            if mergeplots:
               HFluxfig = plt.figure('HFlux-'+inrgfpath)
               axhand = HFluxfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['HFluxes'],linestyle='-', label='$Q_{es,%s}$' % ispecs)
               axhand.plot(time,nrgdata[inrgf][ispecs]['HFluxem'],linestyle='--',label='$Q_{em,%s}$' % ispecs)
               axhand.set_title('$Q_{es,em}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Heat Flux')
               axhand.set_yscale('log')
               axhand.legend()
            else:
               HFluxesfig = plt.figure('HFluxes-'+inrgfpath)
               axhand = HFluxesfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['HFluxes'],label=ispecs)
               axhand.set_title('$Q_{es}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Electrostatic Heat Flux')
               axhand.legend()

               HFluxemfig = plt.figure('HFluxem-'+inrgfpath)
               axhand = HFluxemfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['HFluxem'],label=ispecs)
               axhand.set_title('$Q_{em}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Electromagnetic Heat Flux')
               axhand.legend()

            if mergeplots:
               PFluxfig = plt.figure('Viscos-'+inrgfpath)
               axhand = PFluxfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['Viscoses'],linestyle='-', label='$\\Pi_{es,%s}$' % ispecs)
               axhand.plot(time,nrgdata[inrgf][ispecs]['Viscosem'],linestyle='--',label='$\\Pi_{em,%s}$' % ispecs)
               axhand.set_title('$\\Pi_{es,em}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Stress Tensor')
               axhand.set_yscale('log')
               axhand.legend()
            else:
               Viscosesfig = plt.figure('Viscoses-'+inrgfpath)
               axhand = Viscosesfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['Viscoses'],label=ispecs)
               axhand.set_title('$\\Pi_{es}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Electrostatic Stress Tensor')
               axhand.legend()

               Viscosemfig = plt.figure('Viscosem-'+inrgfpath)
               axhand = Viscosemfig.add_subplot(1,1,1)
               axhand.plot(time,nrgdata[inrgf][ispecs]['Viscosem'],label=ispecs)
               axhand.set_title('$\\Pi_{em}%s$' %(titletxt))
               axhand.set_xlabel('Time')
               axhand.set_ylabel('Electromagnetic Stress Tensor')
               axhand.legend()

        nfig.savefig(reportpath+'n_%s.png' % (inrgfpath))
        plt.close(nfig)

        uparafig.savefig(reportpath+'upara_%s.png' % (inrgfpath))
        plt.close(uparafig)

        if mergeplots:
           Tfig.savefig(reportpath+'T_%s.png' % (inrgfpath))
           plt.close(Tfig)
        else:
           Tparafig.savefig(reportpath+'Tpara_%s.png' % (inrgfpath))
           plt.close(Tparafig)

           Tperpfig.savefig(reportpath+'Tperp_%s.png' % (inrgfpath))
           plt.close(Tperpfig)

        if mergeplots:
           PFluxfig.savefig(reportpath+'PFlux_%s.png' % (inrgfpath))
           plt.close(PFluxfig)
        else:
           PFluxesfig.savefig(reportpath+'PFluxes_%s.png' % (inrgfpath))
           plt.close(PFluxesfig)

           PFluxemfig.savefig(reportpath+'PFluxem_%s.png' % (inrgfpath))
           plt.close(PFluxemfig)

        if mergeplots:
           HFluxfig.savefig(reportpath+'HFlux_%s.png' % (inrgfpath))
           plt.close(HFluxfig)
        else:
           HFluxesfig.savefig(reportpath+'HFluxes_%s.png' % (inrgfpath))
           plt.close(HFluxesfig)

           HFluxemfig.savefig(reportpath+'HFluxem_%s.png' % (inrgfpath))
           plt.close(HFluxemfig)

        if mergeplots:
           Viscosfig.savefig(reportpath+'Viscos_%s.png' % (inrgfpath))
           plt.close(Viscosfig)
        else:
           Viscosesfig.savefig(reportpath+'Viscoses_%s.png' % (inrgfpath))
           plt.close(Viscosesfig)

           Viscosemfig.savefig(reportpath+'Viscosem_%s.png' % (inrgfpath))
           plt.close(Viscosemfig)

    return 1 
    

def plot_scandata(scandata,params={},normalize=True):
   #Developed by Ehab Hassan on 2019-02-05
    #Modified by Ehab Hassan on 2019-03-11
    slashinds=findall(scandata['scanpath'],"/")
    scan_title = scandata['scanpath'][slashinds[-2]+1:slashinds[-1]]
    if not params.keys():
       params = genetools.read_parameters(scandata['scanpath'])
    vlist = list(set(scandata.keys())-{'omega','gamma','scanpath'})
    if   len(vlist) == 1:
         if   'kymin' in vlist:
              if   'x0' in params['box']:
                   plabel = '$x_0$='+str(params['box']['x0'])
              elif 'flux_pos' in params['geometry']:
                   plabel = '$x_0$='+str(params['geometry']['flux_pos'])
              else:
                   plabel = '$x_0$=?'
              if 'kx_center' in params['box'].keys():
                 plabel += ", $k_{x_center}$="+str(params['box']['kx_center'])
         else:
              plabel = '$\\rho_{ref}k_y$ = '+str(rhoref*params['box']['kymin'])
         gammafig = [plt.figure(1)]
         omegafig = [plt.figure(2)]
         axhand01 = gammafig[0].add_subplot(1,1,1)
         axhand02 = omegafig[0].add_subplot(1,1,1)
         if normalize:
              axhand01.plot(scandata[vlist[0]],scandata['gamma'],marker='*',label=plabel)
              axhand02.plot(scandata[vlist[0]],scandata['omega'],marker='*',label=plabel)
         elif not normalize:
              normfactor = params['units']['Lref']/npy.sqrt(params['units']['Tref']/params['units']['mref'])
              axhand01.plot(scandata[vlist[0]],normfactor*scandata['gamma'],marker='*',label=plabel)
              axhand02.plot(scandata[vlist[0]],normfactor*scandata['omega'],marker='*',label=plabel)
         if vlist[0] == 'kymin':
            axhand01.set_title('$k_y$ vs $\gamma$')
            axhand02.set_title('$k_y$ vs $\omega$')
            axhand01.set_xlabel("$\\rho_{ref}k_y$")
            axhand02.set_xlabel("$\\rho_{ref}k_y$")
         elif vlist[0] == 'kx_center':
            axhand01.set_title('$k_{x_{center}}$ vs $\gamma$')
            axhand02.set_title('$k_{x_{center}}$ vs $\omega$')
            axhand01.set_xlabel("$\\rho_{ref}k_{x_{center}}$")
            axhand02.set_xlabel("$\\rho_{ref}k_{x_{center}}$")
         else:
            axhand01.set_title(vlist[0]+' vs $\gamma$')
            axhand02.set_title(vlist[0]+' vs $\omega$')
            axhand01.set_xlabel(vlist[0])
            axhand02.set_xlabel(vlist[0])
         axhand01.set_ylabel('$\gamma/\Omega_{ref}$')
         axhand02.set_ylabel('$\omega/\Omega_{ref}$')
         axhand01.legend()
         axhand02.legend()
    elif len(vlist) == 2:
         if   'kymin' not in vlist:
              iscan = 'kymin = '+str(params['box']['kymin'])+': '
         else:
              iscan = ''
         gammafig = []
         omegafig = []
         maxrepeat = 1
         for var in vlist:
             for irepeat in range(1,len(scandata[var])):
                 if scandata[var][irepeat] == scandata[var][0]: break
             if irepeat >= maxrepeat: maxrepeatvar = var
             maxrepeat = max(maxrepeat,irepeat)
         gammafig.append(plt.figure(1))
         omegafig.append(plt.figure(2))
         axhand01 = gammafig[0].add_subplot(1,1,1)
         axhand02 = omegafig[0].add_subplot(1,1,1)
         for iplot in range(npy.size(scandata[vlist[0]])/maxrepeat):
             sindex = iplot*maxrepeat
             eindex = (iplot+1)*maxrepeat
             for var in list(set(vlist)-{maxrepeatvar}):
                 plabel = iscan+var+'='+str(scandata[var][sindex:eindex][0])+' '
             axhand01.plot(scandata[maxrepeatvar][sindex:eindex],scandata['gamma'][sindex:eindex],marker='*',label=plabel)
             axhand02.plot(scandata[maxrepeatvar][sindex:eindex],scandata['omega'][sindex:eindex],marker='*',label=plabel)
         axhand01.set_title(maxrepeatvar+' vs $\gamma$')
         axhand02.set_title(maxrepeatvar+' vs $\omega$')
         axhand01.set_xlabel(maxrepeatvar)
         axhand02.set_xlabel(maxrepeatvar)
         axhand01.set_ylabel('$\gamma$')
         axhand02.set_ylabel('$\omega$')
         axhand01.legend()
         axhand02.legend()

         gammafig.append(plt.figure(3))
         omegafig.append(plt.figure(4))
         pvarname = list(set(vlist)-{maxrepeatvar})
         axhand01 = gammafig[1].add_subplot(1,1,1)
         axhand02 = omegafig[1].add_subplot(1,1,1)
         for iplot in range(maxrepeat):
             sindex = iplot
             stprng = npy.size(scandata[pvarname[0]])/maxrepeat
             plabel = iscan+maxrepeatvar+'='+str(scandata[maxrepeatvar][sindex:eindex][0])+' '
             axhand01.plot(scandata[pvarname[0]][sindex::stprng],scandata['gamma'][sindex::stprng],marker='*',label=plabel)
             axhand02.plot(scandata[pvarname[0]][sindex::stprng],scandata['omega'][sindex::stprng],marker='*',label=plabel)
         axhand01.set_title(pvarname[0]+' vs $\gamma$')
         axhand02.set_title(pvarname[0]+' vs $\omega$')
         axhand01.set_xlabel(pvarname[0])
         axhand02.set_xlabel(pvarname[0])
         axhand01.set_ylabel('$\gamma$')
         axhand02.set_ylabel('$\omega$')
         axhand01.legend()
         axhand02.legend()

    return gammafig,omegafig

def plot_scans(scanfpaths,geneparams={},normalize=True):
   #Developed by Ehab Hassan on 2019-02-07
    #Modified by Ehab Hassan on 2019-03-07
    if type(scanfpaths)==list:
       for iscan in scanfpaths:
           scanvals = genetools.read_scanfile(iscan)
           gammafig,omegafig = plot_scandata(scandata=scanvals,params=geneparams,normalize=normalize)
       reportpath = "./"
    else:
           scanvals = genetools.read_scanfile(scanfpaths)
           gammafig,omegafig = plot_scandata(scandata=scanvals,params=geneparams,normalize=normalize)
           if scanfpaths[-1] != "/": scanfpaths += "/"
           reportpath = scanfpaths+"report/"
           if not os.path.isdir(reportpath):
              os.system('mkdir '+reportpath)

    pdfpages = pdfh.PdfPages(reportpath+'scanfigs.pdf')
    for item in range(len(gammafig)):
        gammafig[item].savefig(reportpath+'gamma%02d.png' % (item+1))
        omegafig[item].savefig(reportpath+'omega%02d.png' % (item+1))
        pdfpages.savefig(gammafig[item])
        pdfpages.savefig(omegafig[item])
    pdfpages.close()

    return gammafig,omegafig


from ParIO import *
from momlib import *
from fieldlib import *
from finite_differences import *
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

def plot_mom(mom,param={},reportpath=''):
   #Developed by Ehab Hassan on 2019-03-14
    imomf=mom.keys()[0]
    if reportpath == '':
       if not os.path.isdir(imomf[:-10]+"report"):
          os.system('mkdir '+imomf[:-10]+"report")
          reportpath = imomf[:-10]+"report/"
       else:
          reportpath = imomf[:-10]+"report/"
    elif reportpath[-1] != "/":
       reportpath += "/"

    for imomf in mom:
        if not param.keys():
           param = genetools.read_parameters(imomf[:-10]+"parameters_"+imomf[-4:])

        if 'x_local' in param['general']:
            if param['general']['x_local']:
                x_local = True
            else:
                x_local = False
        else:
            x_local = True

        if x_local:
           dens  = mom[imomf]['dens']
           tpar  = mom[imomf]['tpar'] 
           tperp = mom[imomf]['tperp']
           qpar  = mom[imomf]['qpar'] 
           qperp = mom[imomf]['qperp']
           upar  = mom[imomf]['upar'] 

           nx = param['box']['nx0']
           ny = param['box']['nky0']
           nz = param['box']['nz0']

           if 'lx_a' in param['box']:
                xgrid = param['box']['x0']+npy.arange(nx)/float(nx-1)*param['box']['lx_a']-param['box']['lx_a']/2.0
           elif 'lx' in param['box']:
                xgrid = npy.arange(nx)/float(nx-1)*param['box']['lx']-param['box']['lx']/2.0
           elif 'adapt_lx' in param['box']:
                lx = 1.0/(param['box']['kymin']*param['geometry']['shat'])
                xgrid = npy.arange(nx)/float(nx-1)*lx-lx/2.0
           zgrid = npy.arange(nz)/float(nz-1)*(2.0-(2.0/nz))-1.0

           plt.contourf(xgrid,zgrid,npy.abs(npy.fft.fftshift(dens[:,0,:])))
           plt.show()
           plt.contourf(xgrid,zgrid,npy.abs(npy.fft.fftshift(tpar[:,0,:])))
           plt.show()
           plt.contourf(xgrid,zgrid,npy.abs(npy.fft.fftshift(tperp[:,0,:])))
           plt.show()

    return

def plot_field(field,param={},reportpath=''):
   #Developed by Ehab Hassan on 2019-03-14
    ifieldf=field.keys()[0]
    if 'report' not in reportpath:
       if not os.path.isdir(ifieldf[:-10]+"report"):
          os.system('mkdir '+ifieldf[:-10]+"report")
          reportpath = ifieldf[:-10]+"report/"
       else:
          reportpath = ifieldf[:-10]+"report/"
    elif reportpath[-1] != "/":
       reportpath += "/"

    for ifieldf in field:
        if not param.keys():
           param = genetools.read_parameters(ifieldf[:-10]+"parameters_"+ifieldf[-4:])

        if 'x_local' in param['general']:
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

           phi1d  = npy.zeros(nx*nz,dtype='complex128')
           apar1d = npy.zeros(nx*nz,dtype='complex128')

           if 'n0_global' in param['box']:
               phase = -npy.e**(-2.0*npy.pi*(0.0+1.0J)*param['box']['n0_global']*param['geometry']['q0'])
           else:
               phase = -npy.e**(-npy.pi*(0.0+1.0J)*param['geometry']['shat']*param['box']['kymin']*param['box']['lx'])

           shatsgn = int(npy.sign(param['geometry']['shat']))
           for i in range(nx/2):
               phi1d[(i+nx/2)*nz:(i+nx/2+1)*nz]=phi[:,0,i*shatsgn]*phase**i
               if i < nx/2:
                   phi1d[(nx/2-i-1)*nz:(nx/2-i)*nz]=phi[:,0,-(i+1)*shatsgn]*phase**(-(i+1))
               if int(nfields)>1:
                  apar1d[(i+nx/2)*nz:(i+nx/2+1)*nz]=apar[:,0,i*shatsgn]*phase**i
                  if i < nx/2:
                       apar1d[(nx/2-i-1)*nz:(nx/2-i)*nz]=apar[:,0,-(i+1)*shatsgn]*phase**(-(i+1))

           phi1d  = phi1d/phi[nz/2,0,0]
           apar1d = apar1d/apar[nz/2,0,0]

           phinds = (abs(phi1d)>=1.0e-4)
           figure = plt.figure('Phi_'+ifieldf[-4:])
           axhand = figure.add_subplot(1,1,1)
           axhand.plot(zgrid[phinds],npy.real(phi1d[phinds]),color='red',label=r'$Re[\phi]$')
           axhand.plot(zgrid[phinds],npy.imag(phi1d[phinds]),color='blue',label=r'$Im[\phi]$')
           axhand.plot(zgrid[phinds],npy.abs(phi1d[phinds]),color='black',label=r'$|\phi|$')
           axhand.set_title(r'$\phi(k_y=%1.3f)$' % float(param['box']['kymin']) )
           axhand.set_xlabel(r'$z/\pi$',size=18)
           axhand.legend()
           figure.savefig(reportpath+'phi_mode_%s.png' % (ifieldf[-4:]))
           plt.close(figure)

           aparinds = (abs(apar1d)>=1.0e-4)
           figure = plt.figure('Apar_'+ifieldf[-4:])
           axhand = figure.add_subplot(1,1,1)
           axhand.plot(zgrid[aparinds],npy.real(apar1d[aparinds]),color='red',label=r'$Re[A_{||}]$')
           axhand.plot(zgrid[aparinds],npy.imag(apar1d[aparinds]),color='blue',label=r'$Im[A_{||}]$')
           axhand.plot(zgrid[aparinds],npy.abs(apar1d[aparinds]),color='black',label=r'$|A_{||}|$')
           axhand.set_title(r'$A_{||}(k_y=%1.3f)$' % float(param['box']['kymin']))
           axhand.set_xlabel(r'$z/\pi$',size=18)
           axhand.legend()
           figure.savefig(reportpath+'apar_mode_%s.png' % (ifieldf[-4:]))
           plt.close(figure)

           if os.path.isfile(ifieldf[:-10]+'omega'+ifieldf[-5:]):
               om = np.genfromtxt(ifieldf[:-10]+'omega'+ifieldf[-5:])

          ##Note:  the complex frequency is (gamma + i*omega)
           gpars,geometry = read_geometry_local(ifieldf[:-10]+param['geometry']['magn_geometry']+'_'+ifieldf[-4:])
           jacxB = geometry['gjacobian']*geometry['gBfield']
           omega_complex = (om[2]*(0.0+1.0J) + om[1])

           gradphi = fd_d1_o4(phi1d,zgrid)
           for i in range(int(param['box']['nx0'])):
               gradphi[int(param['box']['nz0'])*i:int(param['box']['nz0'])*(i+1)] = gradphi[int(param['box']['nz0'])*i:int(param['box']['nz0'])*(i+1)]/jacxB[:]/npy.pi

           genlist = list(zip(abs(npy.real(gradphi))>=1.0e-6,abs(npy.real(omega_complex*apar1d))>=1.0e-6,abs(npy.imag(gradphi))>=1.0e-6,abs(npy.imag(omega_complex*apar1d))>=1.0e-6))
           geninds = [(i or j or k or l) for (i,j,k,l) in genlist]
           figure = plt.figure('wApar_dPhi_'+ifieldf[-4:])
           axhand = figure.add_subplot(1,1,1)
           axhand.plot(zgrid[geninds],npy.real(gradphi)[geninds],'-',color = 'red',label=r'$Re[\nabla \phi]$')
           axhand.plot(zgrid[geninds],npy.imag(gradphi)[geninds],'-.',color = 'red',label=r'$Im[\nabla \phi]$')
           axhand.plot(zgrid[geninds],-npy.real(omega_complex*apar1d)[geninds],'-',color = 'black',label=r'$Re[\omega A_{||}]$')
           axhand.plot(zgrid[geninds],-npy.imag(omega_complex*apar1d)[geninds],'-.',color = 'black',label=r'$Im[\omega A_{||}]$')
           axhand.set_title(r'$\nabla\phi,\partial_tA_{||}(k_y=%1.3f)$' % float(param['box']['kymin']))
           axhand.set_xlabel(r'$z/\pi$',size=18)
           axhand.legend()
           figure.savefig(reportpath+'wApar_dPhi_mode_%s.png' % (ifieldf[-4:]))
           plt.close(figure)

        elif not x_local:
           nx      = field[ifieldf]['nx']
           ny      = field[ifieldf]['ny']
           nz      = field[ifieldf]['nz']
           phi     = field[ifieldf]['phi']
           apar    = field[ifieldf]['apar']

           zgrid = npy.arange(nz+4)/float(nz+4-1)*(2.0+3.0*(2.0/nz))-(1.0+2.0*(2.0/nz))
           xgrid = npy.arange(nx)/float(nx-1)*param['box']['lx_a']+param['box']['x0']-param['box']['lx_a']/2.0

           gpars,geometry = read_geometry_global(ifieldf[:-10]+param['geometry']['magn_geometry']+'_'+ifieldf[-4:])
           #Find rational q surfaces
           qmin = npy.min(geometry['q'])
           qmax = npy.max(geometry['q'])
           mmin = math.ceil( qmin*param['box']['n0_global'])
           mmax = math.floor(qmax*param['box']['n0_global'])
           mnums = npy.arange(mmin,mmax+1)
           qrats = mnums/float(param['box']['n0_global'])
           zgridm = np.arange(mmax*20)/float(mmax*20)*2.0-1.0
           nm = int(mmax*20)

           phase = (0.0+1.0J)*param['box']['n0_global']*2.0*npy.pi*geometry['q']

           phi_bnd = npy.zeros((nz+4,ny,nx),dtype = 'complex128')
           phi_bnd[2:-2,:,:] = phi
           for j in range(nx):
               phi_bnd[-2,0,j] = phi_bnd[ 2,0,j]*npy.e**(-phase[j])
               phi_bnd[-1,0,j] = phi_bnd[ 3,0,j]*npy.e**(-phase[j])
               phi_bnd[ 0,0,j] = phi_bnd[-4,0,j]*npy.e**( phase[j])
               phi_bnd[ 1,0,j] = phi_bnd[-3,0,j]*npy.e**( phase[j])

           gradphi= npy.zeros((nz+4,ny,nx),dtype = 'complex128')     
           for i in range(nx):
               gradphi[:,0,i] = fd_d1_o4(phi_bnd[:,0,i],zgrid)
               gradphi[2:-2:,0,i] = gradphi[2:-2,0,i]/npy.pi/(geometry['jacobian'][:,i]*geometry['Bfield'][:,i])

           phi_theta = np.zeros((nm,ny,nx),dtype = 'complex128')     
           for i in range(len(xgrid)):
              #phi_theta[:,0,i] = interp(zgrid,phi_bnd[:,0,i],zgridm)
              #Check using the real, imaginary, or absolute value in the interpolation
               phi_theta[:,0,i] = interp(zgrid,npy.real(phi_bnd[:,0,i]),zgridm)
               phi_theta[:,0,i] = phi_theta[:,0,i]*npy.e**(1J*float(param['box']['n0_global'])*geometry['q'][i]*npy.pi*zgridm)

           phi_m = npy.zeros((nm,ny,nx),dtype = 'complex128')     
           for i in range(len(xgrid)):
               phi_m[:,0,i] = npy.fft.fft(phi_theta[:,0,i])

           imax = np.unravel_index(np.argmax(abs(phi)),(nz,nx))
           plot_ballooning = True
           if plot_ballooning:
              figure = plt.figure('Phi_'+ifieldf[-4:])
              axhand = figure.add_subplot(3,1,1)
              cp=axhand.contourf(xgrid,zgrid,npy.abs(phi_bnd[:,0,:]),70)
              for i in range(len(qrats)):
                  ix = np.argmin(abs(geometry['q']-qrats[i])) 
                  axhand.axvline(xgrid[ix],color='white')
              axhand.plot(xgrid[imax[1]],zgrid[imax[0]],'x')
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$|\phi|$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand.set_title(r'Electric Potential ($\phi$)',fontsize=13)
              axhand = figure.add_subplot(3,1,2)
              cp=axhand.contourf(xgrid,zgrid,npy.real(phi_bnd[:,0,:]),70)
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$Re[\phi]$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand = figure.add_subplot(3,1,3)
              cp=axhand.contourf(xgrid,zgrid,npy.imag(phi_bnd[:,0,:]),70)
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$Im[\phi]$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand.set_xlabel(r'$\rho_{tor}$',fontsize=13)
              figure.savefig(reportpath+'phi_mode_%s_2d.png' % (ifieldf[-4:]))
              plt.close(figure)

#          if plot_ballooning:
#              plt.xlabel(r'$z/\pi$',fontsize=13)
#              plt.ylabel(r'$|\phi|$')
#              plt.plot(zgrid,npy.abs(phi_bnd[:,0,param['box']['nx0']/4]),label='nx0/4')
#              plt.plot(zgrid,npy.abs(phi_bnd[:,0,param['box']['nx0']/2]),label='nx0/2')
#              plt.plot(zgrid,npy.abs(phi_bnd[:,0,3*param['box']['nx0']/4]),label='3/4*nx0')
#              plt.legend()
#              plt.show()

           plot_theta = True
           if plot_theta:
              figure = plt.figure('Phi_'+ifieldf[-4:])
              axhand = figure.add_subplot(1,1,1)
              for i in range(int(mmin),int(mmax)+1):
                  axhand.plot(xgrid,npy.abs(phi_m[i,0,:]))
              for i in range(len(qrats)):
                  ix = npy.argmin(abs(geometry['q']-qrats[i])) 
                  axhand.axvline(xgrid[ix],color='black')
              axhand.set_xlabel(r'$\rho_{tor}$',fontsize=13)
              axhand.set_ylabel(r'$\phi_{m}$',fontsize=13)
              axhand.set_title(r'$\phi_m$')
              figure.savefig(reportpath+'phi_mode_%s.png' % (ifieldf[-4:]))
              plt.close(figure)

           zgrid = np.arange(nz)/float(nz-1)*(2.0-(2.0/nz))-1.0

           apar_theta = npy.zeros((nm,nx),dtype = 'complex128')     
           for i in range(len(xgrid)):
              #apar_theta[:,i] = interp(zgrid,apar[:,0,i],zgridm)
              #Check using the real, imaginary, or absolute value in the interpolation
               apar_theta[:,i] = interp(zgrid,npy.real(apar[:,0,i]),zgridm)
               apar_theta[:,i] = apar_theta[:,i]*npy.e**(1J*float(param['box']['n0_global'])*geometry['q'][i]*npy.pi*zgridm)

           apar_m     = npy.zeros((nm,nx),dtype = 'complex128')     
           for i in range(len(xgrid)):
               apar_m[:,i] = npy.fft.fft(apar_theta[:,i])

           if plot_theta:
              figure = plt.figure('Apar_'+ifieldf[-4:])
              axhand = figure.add_subplot(1,1,1)
              for i in range(int(mmin),int(mmax)+1):
                  axhand.plot(xgrid,npy.abs(apar_m[i,:]))
              for i in range(len(qrats)):
                  ix = npy.argmin(abs(geometry['q']-qrats[i])) 
                  axhand.axvline(xgrid[ix],color='black')
              axhand.set_xlabel(r'$\rho_{tor}$',fontsize=13)
              axhand.set_ylabel(r'$A_{||m}$',fontsize=13)
              axhand.set_title(r'$A_{||m}$')
              figure.savefig(reportpath+'apar_mode_%s.png' % (ifieldf[-4:]))
              plt.close(figure)

           if plot_ballooning:
              figure = plt.figure('Apar'+ifieldf[-4:])
              axhand = figure.add_subplot(3,1,1)
              cp=axhand.contourf(xgrid,zgrid,npy.abs(apar[:,0,:]),70)
              for i in range(len(qrats)):
                  ix = npy.argmin(abs(geometry['q']-qrats[i])) 
                  axhand.axvline(xgrid[ix],color='white')
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$|A_{||}|$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand.set_title(r'Magnetic Potential ($A_{||}$)',fontsize=13)
              axhand = figure.add_subplot(3,1,2)
              cp=axhand.contourf(xgrid,zgrid,npy.real(apar[:,0,:]),70)
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$Re[A_{||}]$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand = figure.add_subplot(3,1,3)
              cp=axhand.contourf(xgrid,zgrid,npy.imag(apar[:,0,:]),70)
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$Im[A_{||}]$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand.set_xlabel(r'$\rho_{tor}$',fontsize=13)
              figure.savefig(reportpath+'apar_mode_%s_2d.png' % (ifieldf[-4:]))
              plt.close(figure)

           if os.path.isfile(ifieldf[:-10]+'omega'+ifieldf[-5:]):
               om = np.genfromtxt(ifieldf[:-10]+'omega'+ifieldf[-5:])

          ##Note:  the complex frequency is (gamma + i*omega)
           omega_complex = (om[2]*(0.0+1.0J) + om[1])

           if 'ExBrate' in param['external_contr'] and param['external_contr']['ExBrate'] == -1111: 
              if os.path.isfile(param['in_out']['iterdb_file']):
                 rhot_idb,profs_idb,units_idb = read_iterdb.read_iterdb(param['in_out']['iterdb_file'])
              else:
                 raise IOError('Iterdb file NOT FOUND in the given path!')
              if param['geometry']['geomdir'][-1]=='/':
                 prof_file = param['geometry']['geomdir'][:-6]+'PROFILES/'+param['geometry']['geomfile'][:-5]+'Profiles'
                 geom_file = param['geometry']['geomdir']+param['geometry']['geomfile']
              else:
                 prof_file = param['geometry']['geomdir'][:-5]+'PROFILES/'+param['geometry']['geomfile'][:-5]+'Profiles'
                 geom_file = param['geometry']['geomdir']+'/'+param['geometry']['geomfile']
              if os.path.isfile(prof_file) and os.path.isfile(geom_file):
                 profs,units = efittools.read_profiles(prof_file,setParam={'rhotor':param['box']['nx0'],'eqdskfpath':geom_file})
              else:
                 raise IOError('Profile and/or Geometry files NOT FOUND in the given path!')
              omegator0 = interp(rhot_idb['VROT'],profs_idb['VROT'],profs['rhotor'])
              mi = 1.673e-27
              ee = 1.602e-19
              mref = param['units']['mref']*mi
              time_ref = param['units']['Lref']/(param['units']['Tref']*1000.0*ee/mref)**0.5
              apar_cont = 0.0
              diff = 0.0
              apar_cont_2D = npy.empty(npy.shape(apar),dtype='complex128')
              for i in range(param['box']['nx0']):
                  diff += npy.sum(npy.abs(gradphi[2:-2,:,i] + (omega_complex+(0.0+1.0J)*param['box']['n0_global']*omegator0[i]*time_ref)*apar[:,:,i]))
                  apar_cont += npy.sum(npy.abs((omega_complex+(0.0+1.0J)*param['box']['n0_global']*omegator0[i]*time_ref)*apar[:,:,i]))
                  apar_cont_2D[:,:,i] =  (omega_complex+(0.0+1.0J)*param['box']['n0_global']*omegator0[i]*time_ref)*apar[:,:,i]
           else:
              diff = npy.sum(npy.abs(gradphi[2:-2,:,:] + omega_complex*apar[:,:,:]))
              apar_cont = npy.sum(npy.abs(omega_complex*apar[:,:,:]))
           phi_cont = np.sum(npy.abs(gradphi[2:-2,:,:]))

           if 'ExBrate' in param['external_contr'] and param['external_contr']['ExBrate'] == -1111 and plot_ballooning: 
              figure = plt.figure('real_Epar'+ifieldf[-4:])
              axhand = figure.add_subplot(3,1,1)
              cp=axhand.contourf(xgrid,zgrid,npy.real(gradphi[2:-2,0,:]),70,vmin = npy.min(np.real(gradphi[2:-2,0,:])),vmax = npy.max(np.real(gradphi[2:-2,0,:])))
              for i in range(len(qrats)):
                  ix = npy.argmin(abs(geometry['q']-qrats[i])) 
                  axhand.axvline(xgrid[ix],color='white')
              axhand.plot(xgrid[imax[1]],zgrid[imax[0]],'x')
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$Re(grad phi)$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand.set_title(r'Parallel Electric Field ($Re[E_{||}]$)',fontsize=13)
              axhand = figure.add_subplot(3,1,2)
              cp=axhand.contourf(xgrid,zgrid,-npy.real(apar_cont_2D[:,0,:]),70,vmin = npy.min(np.real(gradphi[2:-2,0,:])),vmax = npy.max(np.real(gradphi[2:-2,0,:])))
              cbar= figure.colorbar(cp)
              cbar.set_label(r'$Re[ omega Apar]$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand=figure.add_subplot(3,1,3)
              cp=axhand.contourf(xgrid,zgrid,npy.real(gradphi[2:-2,0,:]+apar_cont_2D[:,0,:]),70,vmin = npy.min(npy.real(gradphi[2:-2,0,:])),vmax = npy.max(np.real(gradphi[2:-2,0,:])))
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$Re[Diff]$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand.set_xlabel(r'$\rho_{tor}$',fontsize=13)
              figure.savefig(reportpath+'real_Epar_mode_%s_2d.png' % (ifieldf[-4:]))
              plt.close(figure)
           
              figure = plt.figure('imag_Epar'+ifieldf[-4:])
              axhand = figure.add_subplot(3,1,1)
              cp=axhand.contourf(xgrid,zgrid,npy.imag(gradphi[2:-2,0,:]),70,vmin = npy.min(npy.imag(gradphi[2:-2,0,:])),vmax = npy.max(npy.imag(gradphi[2:-2,0,:])))
              for i in range(len(qrats)):
                  ix = npy.argmin(abs(geometry['q']-qrats[i])) 
                  axhand.axvline(xgrid[ix],color='white')
              axhand.plot(xgrid[imax[1]],zgrid[imax[0]],'x')
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$Im(grad phi)$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand.set_title(r'Parallel Electric Field ($Im[E_{||}]$)',fontsize=13)
              axhand = figure.add_subplot(3,1,2)
              cp=axhand.contourf(xgrid,zgrid,-npy.imag(apar_cont_2D[:,0,:]),70,vmin = npy.min(npy.imag(gradphi[2:-2,0,:])),vmax = npy.max(npy.imag(gradphi[2:-2,0,:])))
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$Im[ omega Apar]$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand = figure.add_subplot(3,1,3)
              cp=axhand.contourf(xgrid,zgrid,npy.imag(gradphi[2:-2,0,:]+apar_cont_2D[:,0,:]),70,vmin = npy.min(npy.imag(gradphi[2:-2,0,:])),vmax = npy.max(npy.imag(gradphi[2:-2,0,:])))
              cbar=figure.colorbar(cp)
              cbar.set_label(r'$Im[Diff]$')
              axhand.set_ylabel(r'$z/\pi$',fontsize=13)
              axhand.set_xlabel(r'$\rho_{tor}$',fontsize=13)
              figure.savefig(reportpath+'imag_Epar_mode_%s_2d.png' % (ifieldf[-4:]))
              plt.close(figure)

    return 1

def plot_geometry(geometryfpath):
    if type(geometryfpath)==str:
       gfpathlist=[geoemtryfpath]
    elif type(geometryfpath)==list:
       gfpathlist=geometryfpath

    geomfigs = pdfh.PdfPages('geometry.pdf')

    for ifile in gfpathlist:
        parameters,geometry = read_geometry_local(ifile)
        Lref = parameters['Lref']
        nz = parameters['gridpoints']
        zgrid = npy.arange(nz)/float(nz-1)*(2.0-(2.0/nz))-1.0

        if ifile[-1]!="/": ifile+="/"
        slashinds=findall(ifile,"/")
        gfname=ifile[slashinds[-2]:slashinds[-1]]

        GGfig = plt.figure("ggCoeff",dpi=500)
        ax  = GGfig.add_subplot(2,3,1)
        ax.plot(zgrid,geometry['ggxx'],label=gfname)
        ax.set_title('ggxx')
        ax.set_xticks([])

        ax  = GGfig.add_subplot(2,3,2)
        ax.plot(zgrid,geometry['ggyy'],label=gfname)
        ax.set_title('ggyy')
        ax.set_xticks([])

        ax  = GGfig.add_subplot(2,3,3)
        ax.plot(zgrid,geometry['ggzz'],label=gfname)
        ax.set_title('ggzz')
        ax.set_xticks([])

        ax  = GGfig.add_subplot(2,3,4)
        ax.plot(zgrid,geometry['ggxy'],label=gfname)
        ax.set_xlabel('$Z/\\pi$')
        ax.set_title('ggxy')

        ax  = GGfig.add_subplot(2,3,5)
        ax.plot(zgrid,geometry['ggxz'],label=gfname)
        ax.set_xlabel('$Z/\\pi$')
        ax.set_title('ggxz')

        ax  = GGfig.add_subplot(2,3,6)
        ax.plot(zgrid,geometry['ggyz'],label=gfname)
        ax.set_xlabel('$Z/\\pi$')
        ax.set_title('ggyz')

        GLfig = plt.figure("gl_R",dpi=500)
        ax  = GLfig.add_subplot(2,2,1)
        ax.plot(zgrid,geometry['gl_R'],label=gfname)
        ax.set_title('gl_R')
        ax.set_xticks([])

        ax  = GLfig.add_subplot(2,2,2)
        ax.plot(zgrid,geometry['gl_z'],label=gfname)
        ax.set_title('gl_R')
        ax.set_xticks([])

        ax  = GLfig.add_subplot(2,2,3)
        ax.plot(zgrid,geometry['gl_dxdR'],label=gfname)
        ax.set_xlabel('$Z/\\pi$')
        ax.set_title('gl_dxdR')

        ax  = GLfig.add_subplot(2,2,4)
        ax.plot(zgrid,geometry['gl_dxdZ'],label=gfname)
        ax.set_xlabel('$Z/\\pi$')
        ax.set_title('gl_dxdZ')

        gBfig = plt.figure("gBfield",dpi=500)
        ax  = gBfig.add_subplot(2,2,1)
        ax.plot(zgrid,geometry['gBfield'],label=gfname)
        ax.set_title('gBfield')
        ax.set_xticks([])

        ax  = gBfig.add_subplot(2,2,2)
        ax.plot(zgrid,geometry['gdBdx'],label=gfname)
        ax.set_title('gdBdx')
        ax.set_xticks([])

        ax  = gBfig.add_subplot(2,2,3)
        ax.plot(zgrid,geometry['gdBdy'],label=gfname)
        ax.set_xlabel('$Z/\\pi$')
        ax.set_title('gdBdy')
        plt.legend()

        ax12  = gBfig.add_subplot(2,2,4)
        ax12.plot(zgrid,geometry['gdBdz'],label=gfname)
        ax12.set_xlabel('$Z/\\pi$')
        ax12.set_title('gdBdz')

    geomfigs.savefig(gBfig)
    geomfigs.savefig(GLfig)
    geomfigs.savefig(GGfig)
    
    geomfigs.close()

    return gBfig,GLfig,GGfig

