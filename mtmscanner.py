import os
import sys

import scipy             as spy
import numpy             as npy
import warnings          as warnings
import cheasepy          as cheasepy
import genetools         as genetools
import matplotlib.pyplot as plt

from finite_differences import *
from matplotlib.backends.backend_pdf import PdfPages


def main():

    geompath    = "DIIID162940/tracer_efit.dat"
    eqdskpath   = "DIIID162940/DIIID162940.eqdsk"
    iterdbpath  = "DIIID162940/DIIID162940.iterdb"

   #'collective' is used to plot the rational surfaces for all
   #toroidal mode numbers in the same figure, however, 'individual'
   #is used to plot the rational surface of each toroidal mode number
   #in a separate figure.
    collective = True
    individual = False

   #If 'plot_rejected' is set to True, the rational surfaces for the
   #toroidal mode numbers outside the given frequency range(s) will
   #also be plotted if there is any.
    plot_rejected = False

    q_import = False
    q_eqdsk  = True

    eqdskdata = cheasepy.read_eqdsk(eqdskfpath=eqdskpath)

    setParam = {}

   #Setup a range of toroidal mode numbers to search for MTM over this range
    setParam.update({'n0':(1,30)})

   #Setup a range or multiple ranges of frequencies from spectrogram
   #setParam.update({'frequency':(1.0e3,550.0e3)})
    setParam.update({'frequency':((50.0e3,55.0e3),(85.0e3,105.0e3),(270.0e3,510.0e3))})

   #Setup the center of the location you're searching in
    setParam.update({'rhotor':0.970})

   #Setup the limits for the location you're searching in
   #setParam.update({'rholim':(0.950,0.990)})

   #Setup the a percentage from maximum diamagnetic frequency to match
    setParam.update({'omegapct':99.9})

    if   q_import:
         newrhotor = npy.linspace(eqdskdata['rhotor'][0],eqdskdata['rhotor'][-1],10*npy.size(eqdskdata['rhotor']))
         newq      = npy.interp(newrhotor,eqdskdata['rhotor'],eqdskdata['q'])
         
         paramdata,geomdata  = genetools.read_geometry(geomfpath=geompath)
         bgnid = npy.argmin(abs(newq-geomdata['q'][0]))
         endid = npy.argmin(abs(newq-geomdata['q'][-1]))
         geomdata['rhotor'] = npy.linspace(newrhotor[bgnid],newrhotor[endid],npy.size(geomdata['q']))
         
         modrhotor = newrhotor[:bgnid]
         modrhotor = npy.append(modrhotor,geomdata['rhotor'])
         modrhotor = npy.append(modrhotor,newrhotor[endid+1:])
         
         modq = newq[:bgnid]
         modq = npy.append(modq,geomdata['q'])
         modq = npy.append(modq,newq[endid+1:])

         importeddata = {}
         importeddata.update({'q':modq,'rhotor':modrhotor})

         mtmfreq = get_mtm_frequency(iterdbfpath=iterdbpath,eqdskfpath=eqdskpath,imported=importeddata,setParam=setParam)

    elif q_eqdsk:
         mtmfreq = get_mtm_frequency(iterdbfpath=iterdbpath,eqdskfpath=eqdskpath,setParam=setParam)


    setparam = {'frequency':setParam['frequency']}
    pltparam = {}
    if   individual:
         pltparam.update({'plot':'individual','rotation':'vertical'})
    elif collective:
         pltparam.update({'plot':'collective','rotation':'horizontal'})

    if   plot_rejected:
         pltparam.update({'plot_rejected':plot_rejected})

    plot_mtm_frequency(mtm_frequency=mtmfreq,setParam=setparam,pltParam=pltparam)

    return 1


def plot_mtm_frequency(mtm_frequency,setParam={},pltParam={}):

    eqdskpath   = "DIIID162940/DIIID162940.eqdsk"
    eqdskdata = cheasepy.read_eqdsk(eqdskfpath=eqdskpath)

    iterdbpath  = "DIIID162940/DIIID162940.iterdb"
    iterdbdata = cheasepy.read_iterdb(iterdbfpath=iterdbpath)

    q_rhotor = interp(eqdskdata['PSIN'],eqdskdata['q'],iterdbdata['rhotor'])

    n0  = mtm_frequency.keys()

    if 'frequency' in setParam:
       f0range = setParam['frequency']
       if type(f0range[0]) in [float,int,str]:
          f0bgn = float(f0range[0])
          f0end = float(f0range[1])
          fmax = f0end
       else:
          f0bgn = []
          f0end = []
          for i in range(len(f0range)):
              f0bgn.append(f0range[i][0])
              f0end.append(f0range[i][1])
          fmax = max(f0end)
    else:
        f0bgn = 0.0
        f0end = 0.0


    if 'plot' in pltParam:
       if   pltParam['plot'].lower() == 'individual':
            individual = True
            collective = False
       elif pltParam['plot'].lower() == 'collective':
            individual = False
            collective = True
    else:
            individual = False
            collective = False

    if 'rotation' in pltParam:
       if   pltParam['rotation'].lower() == 'vertical':
            vertical   = True
            horizontal = False
       elif pltParam['rotation'].lower() == 'horizontal':
            vertical   = False
            horizontal = True
    else:
            vertical   = False
            horizontal = True

    if 'plot_rejected' in pltParam:
        plot_rejected = pltParam['plot_rejected']
    else:
        plot_rejected = False


    mtmfreqfigs = PdfPages('MTM_Frequencies.pdf')
    if   collective:
         fig, ax1 = plt.subplots()
         ax2 = ax1.twinx()
         red_color = 'tab:red'
         blu_color = 'tab:blue'
         grn_color = 'tab:green'
         for in0 in n0:
             if type(in0) == str: continue
             ax1.set_xlabel('$\\rho_{\\phi}$')
             ax1.set_ylabel('$f(\\rho_{\\phi}$) (kHz)', color=grn_color)
             ax1.set_title('Rational Surfaces for Microtearing Modes')
             ax1.tick_params(axis='y', labelcolor=grn_color)
             ax1.set_ylim((0.0,1.1*fmax/1.0e3))
             font = {'family':'serif','color':'blue','weight': 'normal','size': 10}
             if   'select' in mtm_frequency[in0]:
                  ax1.plot(mtm_frequency['rho'], mtm_frequency[in0]['select']['omega']/1.0e3, color=grn_color)
                  omega_max = npy.max(mtm_frequency[in0]['select']['omega'])
             elif 'reject' in mtm_frequency[in0] and plot_rejected:
                  ax1.plot(mtm_frequency['rho'], mtm_frequency[in0]['reject']['omega']/1.0e3, color=red_color)
                  omega_max = npy.max(mtm_frequency[in0]['reject']['omega'])

             if   'select' in mtm_frequency[in0]:
                  for ind in range(npy.size(mtm_frequency[in0]['select']['rho_rat_surf'])):
                      linelabel = "(%d,%d)" % (in0,mtm_frequency[in0]['select']['m0_rat_surf'][ind])
                      ax1.axvline(mtm_frequency[in0]['select']['rho_rat_surf'][ind],color=grn_color)
                      if   horizontal:
                           ax1.text(mtm_frequency[in0]['select']['rho_rat_surf'][ind],omega_max/1.0e3,linelabel,fontdict=font)
                      elif vertical:
                           ax1.text(mtm_frequency[in0]['select']['rho_rat_surf'][ind],0.9*omega_max/1.0e3,linelabel,fontdict=font,rotation=90.0)

             elif 'reject' in mtm_frequency[in0] and plot_rejected:
                  for ind in range(npy.size(mtm_frequency[in0]['reject']['rho_rat_surf'])):
                      linelabel = "(%d,%d)" % (in0,mtm_frequency[in0]['reject']['m0_rat_surf'][ind])
                      ax1.axvline(mtm_frequency[in0]['reject']['rho_rat_surf'][ind],color=red_color)
                      if   horizontal:
                           ax1.text(mtm_frequency[in0]['reject']['rho_rat_surf'][ind],omega_max/1.0e3,linelabel,fontdict=font)
                      elif vertical:
                           ax1.text(mtm_frequency[in0]['reject']['rho_rat_surf'][ind],0.9*omega_max/1.0e3,linelabel,fontdict=font,rotation=90.0)

             ax2.set_ylabel('$q(\\rho_{\\phi})$', color=blu_color)
             ax2.plot(mtm_frequency['rho'], mtm_frequency['qtor'], color=blu_color)
             ax2.tick_params(axis='y', labelcolor=blu_color)

             unity = 0.001*npy.ones(npy.size(mtm_frequency['rho']))
             if   type(f0bgn) == float and f0end <= omega_max:
                      ax1.fill_between(mtm_frequency['rho'],f0bgn*unity,f0end*unity,color='orange',alpha=0.9)
             elif type(f0bgn) == list:
                  for i in range(len(f0bgn)):
                      ax1.fill_between(mtm_frequency['rho'],f0bgn[i]*unity,f0end[i]*unity,color="orange",alpha=0.9)

         mtmfreqfigs.savefig(fig)
         plt.close(fig)
    elif individual:
         for in0 in n0:
             fig, ax1 = plt.subplots()
             ax2 = ax1.twinx()
             red_color = 'tab:red'
             blu_color = 'tab:blue'
             grn_color = 'tab:green'

             if type(in0) == str: continue
             ax1.set_xlabel('$\\rho_{\\phi}$')
             ax1.set_ylabel('$f(\\rho_{\\phi}$) (kHz)', color=grn_color)
             ax1.set_title('Rational Surfaces for Microtearing Modes')
             ax1.tick_params(axis='y', labelcolor=grn_color)
             ax1.set_ylim((0.0,1.1*fmax/1.0e3))
             font = {'family':'serif','color':'blue','weight': 'normal','size': 10}
             if    'select' in mtm_frequency[in0]:
                   ax1.plot(mtm_frequency['rho'], mtm_frequency[in0]['select']['omega']/1.0e3, color=grn_color)
                   omega_max = npy.max(mtm_frequency[in0]['select']['omega'])
             elif  'reject' in mtm_frequency[in0] and plot_rejected:
                   ax1.plot(mtm_frequency['rho'], mtm_frequency[in0]['reject']['omega']/1.0e3, color=red_color)
                   omega_max = npy.max(mtm_frequency[in0]['reject']['omega'])

             if   'select' in mtm_frequency[in0]:
                  for ind in range(npy.size(mtm_frequency[in0]['select']['rho_rat_surf'])):
                      linelabel = "(%d,%d)" % (in0,mtm_frequency[in0]['select']['m0_rat_surf'][ind])
                      ax1.axvline(mtm_frequency[in0]['select']['rho_rat_surf'][ind],color=grn_color)
                      if   horizontal:
                           ax1.text(mtm_frequency[in0]['select']['rho_rat_surf'][ind],omega_max/1.0e3,linelabel,fontdict=font)
                      elif vertical:
                           ax1.text(mtm_frequency[in0]['select']['rho_rat_surf'][ind],0.9*omega_max/1.0e3,linelabel,fontdict=font,rotation=90.0)

             elif 'reject' in mtm_frequency[in0] and plot_rejected:
                  for ind in range(npy.size(mtm_frequency[in0]['reject']['rho_rat_surf'])):
                      linelabel = "(%d,%d)" % (in0,mtm_frequency[in0]['reject']['m0_rat_surf'][ind])
                      ax1.axvline(mtm_frequency[in0]['reject']['rho_rat_surf'][ind],color=red_color)
                      if   horizontal:
                           ax1.text(mtm_frequency[in0]['reject']['rho_rat_surf'][ind],omega_max/1.0e3,linelabel,fontdict=font)
                      elif vertical:
                           ax1.text(mtm_frequency[in0]['reject']['rho_rat_surf'][ind],0.9*omega_max/1.0e3,linelabel,fontdict=font,rotation=90.0)

             ax2.set_ylabel('$q(\\rho_{\\phi})$', color=blu_color)
             ax2.plot(mtm_frequency['rho'], mtm_frequency['qtor'], color=blu_color)
             ax2.tick_params(axis='y', labelcolor=blu_color)

             unity = 0.001*npy.ones(npy.size(mtm_frequency['rho']))
             if   type(f0bgn) == float:
                      ax1.fill_between(mtm_frequency['rho'],f0bgn*unity,f0end*unity,color='orange',alpha=0.5)
             elif type(f0bgn) == list:
                  for i in range(len(f0bgn)):
                      ax1.fill_between(mtm_frequency['rho'],f0bgn[i]*unity,f0end[i]*unity,color="orange",alpha=0.9)

             mtmfreqfigs.savefig(fig)
             plt.close(fig)


    mtmfreqfigs.close()

    return 1


def get_mtm_frequency(iterdbfpath,eqdskfpath,imported={},setParam={}):
    if os.path.isfile(eqdskfpath):
       eqdskdata  = cheasepy.read_eqdsk(eqdskfpath=eqdskfpath)
       eqdskfile  = True
    else:
       print("The path to the EQDSK file is INCORRECT!")

    if os.path.isfile(iterdbfpath):
       if 'rhotor' in setParam:
          iterdbdata = cheasepy.read_iterdb(iterdbfpath=iterdbfpath)
          ped_mid = setParam['rhotor']
       else:
          pedestalParam,iterdbdata = cheasepy.find_pedestal(profilefpath=iterdbfpath,profileftype='iterdb')
          ped_mid = pedestalParam['Te']['ped_mid']
    else:
       print("The path to the ITERDB file is INCORRECT!")

    if 'n0' in setParam:
       if type(setParam['n0']) in [int,float,str]:
          n0bgn = int(setParam['n0'])
          n0end = 100
       else:
          n0bgn = int(setParam['n0'][0])
          n0end = int(setParam['n0'][1])
    else:
          n0bgn = 1
          n0end = 100


    if 'frequency' in setParam:
       if type(setParam['frequency']) in [int,float,str]:
          f0bgn = float(setParam['frequency'])
          f0end = 500.0e3
       else:
          if type(setParam['frequency'][0]) in [int,float,str]:
             f0bgn = float(setParam['frequency'][0])
             f0end = float(setParam['frequency'][1])
          else:
             f0bgn = []
             f0end = []
             for i in range(len(setParam['frequency'])):
                 f0bgn.append(setParam['frequency'][i][0])
                 f0end.append(setParam['frequency'][i][1])
    else:
          f0bgn = 1.0e3
          f0end = 500.0e3

    if 'rholim' in setParam:
       if type(setParam['rholim']) == float:
          bgnrho = setParam['rholim']
       else:
          bgnrho = setParam['rholim'][0]
          endrho = setParam['rholim'][-1]

    if 'omegapct' in setParam:
       if setParam['omegapct']> 1.0:
          omegapct = setParam['omegapct']/100.0
       else:
          omegapct = setParam['omegapct']
    else:
          omegapct = 0.8

    qe   = 1.6022e-19
    me   = 9.1093e-31
    mp   = 1.6726e-27
    mref = 2.0

    mi   = mref*mp
    Bref = eqdskdata['BCTR']
    Lref = npy.mean(npy.sqrt((eqdskdata['rbound']-eqdskdata['RCTR'])**2+(eqdskdata['zbound']-eqdskdata['ZMID'])**2))

    rho = iterdbdata['rhotor']
    if 'q' in imported:
      q_rhotor = interp(imported['PSIN'],imported['q'],iterdbdata['rhotor'])
    else:
      q_rhotor = interp(eqdskdata['rhotor'],eqdskdata['q'],iterdbdata['rhotor'])

    ped_mid_id = npy.argmin(abs(rho-ped_mid))     # Location of Pedestal Middle Point
    q0_ped_mid = q_rhotor[ped_mid_id]             # Value of q-profile at ped_mid
    ne_ped_mid = iterdbdata['ne'][ped_mid_id]     # Value of ne-profile at ped_mid
    Te_ped_mid = qe*iterdbdata['Te'][ped_mid_id]  # Value of Te-profile at ped_mid
    Cy_ped_mid = ped_mid/q0_ped_mid               # Value of Cy-coefficient at ped_mid
    cs_ped_mid = npy.sqrt(Te_ped_mid/mi)          # Value of acoustic speed at ped_mid
    fg_ped_mid = qe*Bref/mi                       # Value of Gyrofrequency
    rs_ped_mid = cs_ped_mid/fg_ped_mid            # Value of rho_s at ped_mid

    TePrime = -fd_d1_o4(iterdbdata['Te'],rho)/iterdbdata['Te']
    nePrime = -fd_d1_o4(iterdbdata['ne'],rho)/iterdbdata['ne']

    mtm_frequency = {}

    if 'bgnrho' in locals():
       bgnid = npy.argmin(abs(iterdbdata['rhotor']-bgnrho))
    else:
       bgnid = ped_mid_id-100

    if 'endrho' in locals():
       endid = npy.argmin(abs(iterdbdata['rhotor']-endrho))
    else:
       endid = ped_mid_id+100
    rho = iterdbdata['rhotor'][bgnid:endid]

    mtm_frequency['rho']  = iterdbdata['rhotor'][bgnid:endid]
    mtm_frequency['qtor'] = q_rhotor[bgnid:endid]
    for n0 in range(n0bgn,n0end+1):
        m0        = n0*q0_ped_mid
        rs        = npy.sqrt(qe*iterdbdata['Te'][bgnid:endid]/mi)/(qe*Bref/mi)
        kymin     = (n0*q_rhotor[bgnid:endid]*rs)/(Lref*iterdbdata['rhotor'][bgnid:endid])

        omegastar    = kymin*npy.sqrt(iterdbdata['Te'][bgnid:endid]/iterdbdata['Te'][ped_mid_id])*TePrime[bgnid:endid]
        omegastar   += kymin*npy.sqrt(iterdbdata['ne'][bgnid:endid]/iterdbdata['ne'][ped_mid_id])*nePrime[bgnid:endid]
        omegagyro    = npy.sqrt(qe*iterdbdata['Te'][bgnid:endid]/mp/mref)/Lref
        omegaMTM     = omegastar*omegagyro/2.0/npy.pi
        omegaDoppler = iterdbdata['Vrot'][bgnid:endid]*n0/2.0/npy.pi
        omega        = omegaMTM+omegaDoppler

        omega_max = npy.max(omega)
        omega_min = omega_max*omegapct

        select_ind     = []
        m0_rat_surf    = []
        rho_rat_surf   = []
        omega_rat_surf = []

        reject_ind            = []
        reject_m0_rat_surf    = []
        reject_rho_rat_surf   = []
        reject_omega_rat_surf = []

        for i in range(npy.size(rho)):
            if omega[i] >= omega_min:
               if type(f0bgn) == float:
                  if omega[i] >= f0bgn and omega[i] <= f0end:
                     select_ind.append(i)
                  else:
                     reject_ind.append(i)
               elif type(f0bgn) == list:
                    for j in range(len(f0bgn)):
                        if omega[i] >= f0bgn[j] and omega[i] <= f0end[j]:
                           select_ind.append(i)
                        else:
                           reject_ind.append(i)

        mtm_frequency[n0] = {}
        if select_ind:
           mtm_frequency[n0]['select'] = {}
           mtm_frequency[n0]['select']['kymin'] = kymin
           mtm_frequency[n0]['select']['omega'] = omega

          #Method 01:
          #q_vals = mtm_frequency['qtor'][select_ind[0]:select_ind[-1]+1]
          #m0vals = n0*q_vals
          #q_rat_surf = []
          #m0_rat_surf = []
          #rho_rat_surf = []
          #omega_rat_surf = []
          #for iq in range(npy.size(m0vals)):
          #    if round(m0vals[iq]) not in m0_rat_surf:
          #       m0_rat_surf.append(round(m0vals[iq]))
          #       ix = npy.argmin(abs(mtm_frequency['qtor']-q_vals[iq]))
          #       q_rat_surf.append(mtm_frequency['qtor'][ix])
          #       rho_rat_surf.append(rho[ix])
          #       omega_rat_surf.append(omega[ix])

          #Method 02:
          #q_min  = npy.min(mtm_frequency['qtor'][select_ind[0]:select_ind[-1]+1])
          #q_max  = npy.max(mtm_frequency['qtor'][select_ind[0]:select_ind[-1]+1])
          #m_min  = npy.ceil(q_min*n0)
          #m_max  = npy.floor(q_max*n0)
          #if m_min>m_max:
          #   q_rational_surfaces = npy.array([q_max])
          #   m_nums = npy.array([int(q_rational_surfaces[0]*n0)])
          #else:
          #   m_nums = npy.arange(m_min,m_max+1)
          #   q_rational_surfaces = m_nums/float(n0)

          #Method 03:
           m0ceil = set([npy.ceil(jjj)  for jjj in n0*mtm_frequency['qtor'][select_ind[0]:select_ind[-1]+1]])
           m0flor = set([npy.floor(jjj) for jjj in n0*mtm_frequency['qtor'][select_ind[0]:select_ind[-1]+1]])
           m_nums = npy.array(list(m0flor.union(m0ceil)))
           q_rational_surfaces = m_nums/float(n0)
           
           for i in range(npy.size(q_rational_surfaces)):
               ix = npy.argmin(abs(mtm_frequency['qtor']-q_rational_surfaces[i]))
               if rho[ix]>=rho[select_ind[0]] and rho[ix]<=rho[select_ind[-1]]:
                  m0_rat_surf.append(m_nums[i])
                  rho_rat_surf.append(rho[ix])
                  omega_rat_surf.append(omega[ix])
           if  not rho_rat_surf:
               q_mean = npy.mean(mtm_frequency['qtor'][select_ind[0]:select_ind[-1]+1])
               ix = npy.argmin(abs(mtm_frequency['qtor']-q_mean))
               m0_rat_surf.append(npy.ceil(n0*q_mean))
               rho_rat_surf.append(rho[ix])
               omega_rat_surf.append(omega[ix])

           mtm_frequency[n0]['select']['m0_rat_surf']    = m0_rat_surf
           mtm_frequency[n0]['select']['rho_rat_surf']   = rho_rat_surf
           mtm_frequency[n0]['select']['omega_rat_surf'] = omega_rat_surf

        if reject_ind:
           mtm_frequency[n0]['reject'] = {}
           mtm_frequency[n0]['reject']['kymin'] = kymin
           mtm_frequency[n0]['reject']['omega'] = omega

           q_min  = npy.min(mtm_frequency['qtor'][reject_ind[0]:reject_ind[-1]+1])
           q_max  = npy.max(mtm_frequency['qtor'][reject_ind[0]:reject_ind[-1]+1])
           m_min  = npy.ceil(q_min*n0)
           m_max  = npy.floor(q_max*n0)
           if m_min>m_max:
              q_rational_surfaces = npy.array([q_min])
              m_nums = npy.array([int(q_rational_surfaces[0]*n0)])
           else:
              m_nums = npy.arange(m_min,m_max+1)
              q_rational_surfaces = m_nums/float(n0)

           for i in range(npy.size(q_rational_surfaces)):
               ix = npy.argmin(abs(mtm_frequency['qtor']-q_rational_surfaces[i]))
               if rho[ix]>=rho[reject_ind[0]] and rho[ix]<=rho[reject_ind[-1]]:
                  reject_m0_rat_surf.append(m_nums[i])
                  reject_rho_rat_surf.append(rho[ix])
                  reject_omega_rat_surf.append(omega[ix])

           mtm_frequency[n0]['reject']['m0_rat_surf']    = reject_m0_rat_surf
           mtm_frequency[n0]['reject']['rho_rat_surf']   = reject_rho_rat_surf
           mtm_frequency[n0]['reject']['omega_rat_surf'] = reject_omega_rat_surf

    return mtm_frequency


if __name__ == '__main__':
   print('Finding MTM Fequencies')
   main()



