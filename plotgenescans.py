#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as npy
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfh


def read_parameters(paramfpath):
   #Developed by Ehab Hassan on 2019-02-05
    if "parameters" not in paramfpath:
       if    paramfpath[-1] == "/": paramfpath+="parametrers"
       else:                        paramfpath+="/parameters"
    if os.path.isfile(paramfpath):
         print(paramfpath+' FILE FOUND ...')
    else:
         print(paramfpath+' FILE NOT FOUND. Exit!'); sys.exit()

    ofh = open(paramfpath,'r')
    lines = ofh.readlines()
    ofh.close()

    geneparam = {'filepath':paramfpath}

    nspecs = 0
    for line in lines:
        if   line[0] == '&':
             ckey = line[1:].strip()
             if ckey=="species":
                nspecs += 1
                ckey += "_"+str(nspecs)
             geneparam[ckey] = {}
        elif line[0] == '/' or line[0] == '!':
             continue
        elif line.strip() == '':
             continue
        else:
             items = line.split('=')
             values = items[1].split()
             if "!scanlist" in items[1]:
                cvalues = []
                for item in values[2:]:
                    if item == '!': break
                    cvalues.append(float(item[0:-1]))
                cvalues = npy.array(cvalues)
             else:
               if    values[0].isdigit(): cvalues = float(values[0].strip())
               else:                      cvalues = values[0].strip()
             geneparam[ckey][items[0].strip()] = cvalues

    return geneparam


def read_scanfile(scanfpath):
   #Developed by Ehab Hassan on 2019-02-01
    if "scan.log" not in scanfpath:
       if    scanfpath[-1] == "/": scanfpath+="scan.log"
       else:                       scanfpath+="/scan.log"
    if os.path.isfile(scanfpath):
         print(scanfpath+' FILE FOUND ...')
    else:
         print(scanfpath+' FILE NOT FOUND. Exit!'); sys.exit()

    ofh = open(scanfpath,'r')
    lines = ofh.readlines()
    ofh.close()

    scandata = {'filepath':scanfpath}
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

def plot_scandata(scandata,normalize=False):
   #Developed by Ehab Hassan on 2019-02-05
    paramsx = read_parameters(scandata['filepath'][0:-9]+'/parameters_0001')
    Tref    = float(paramsx['units']['Tref'])
    nref    = float(paramsx['units']['nref'])
    Bref    = float(paramsx['units']['Bref'])
    mref    = float(paramsx['units']['mref'])
    Lref    = float(paramsx['units']['Lref'])
    qref    = 1 #1.60217662e-19
    cref    = npy.sqrt(Tref/mref)
    frqref  = qref*Bref/mref/cref
    rhoref  = cref/frqref
    rhostr  = rhoref/Lref

    vlist = list(set(scandata.keys())-{'omega','gamma','filepath'})
    if   len(vlist) == 1:
         if   'kymin' in vlist:
              usrdeflabel = raw_input('What parameter should be used as a label for '+scandata['filepath'][0:-9]+"? ")
              if len(usrdeflabel.strip()) == 0:
                 plabel = scandata['filepath'][0:-9]
              else:
                 if   usrdeflabel.strip() in paramsx['box'].keys():
                      plabel = usrdeflabel.strip()+" = "+paramsx['box'][usrdeflabel.strip()]
                 elif usrdeflabel.strip() in paramsx['geometry'].keys():
                      plabel = usrdeflabel.strip()+" = "+paramsx['geometry'][usrdeflabel.strip()]
                 else:
                      plabel = usrdeflabel.strip()
         else:
              params = read_parameters(scandata['filepath'][0:-9]+'/parameters')
              plabel = '$\\rho_{ref}k_y$ = '+str(rhoref*float(params['box']['kymin']))
         gammafig = [plt.figure(1)]
         omegafig = [plt.figure(2)]
         axhand01 = gammafig[0].add_subplot(1,1,1)
         axhand02 = omegafig[0].add_subplot(1,1,1)
         if   normalize:
              axhand01.plot(scandata[vlist[0]],scandata['gamma'],marker='*',label=plabel)
              axhand02.plot(scandata[vlist[0]],scandata['omega'],marker='*',label=plabel)
         elif not normalize:
              axhand01.plot(scandata[vlist[0]],scandata['gamma']*(cref/Lref),marker='*',label=plabel)
              axhand02.plot(scandata[vlist[0]],scandata['omega']*(cref/Lref),marker='*',label=plabel)
         if vlist[0] == 'kymin':
            axhand01.set_title('$k_y$ vs $\gamma$')
            axhand02.set_title('$k_y$ vs $\omega$')
            if normalize:
               axhand01.set_xlabel("$\\rho_{ref}k_y$")
               axhand02.set_xlabel("$\\rho_{ref}k_y$")
            elif not normalize:
               axhand01.set_xlabel("$k_y$")
               axhand02.set_xlabel("$k_y$")
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
         if normalize:
            axhand01.set_ylabel('$\gamma/\Omega_{ref}$')
            axhand02.set_ylabel('$\omega/\Omega_{ref}$')
         elif not normalize:
            axhand01.set_ylabel('$\gamma$')
            axhand02.set_ylabel('$\omega$')

        #newlabel = [273.15,290,310,330,350,373.15] # labels of the xticklabels: the position in the new x-axis
        #k2degc = lambda t: t-273.15 # convert function: from Kelvin to Degree Celsius
        #newpos   = [k2degc(t) for t in newlabel]   # position of the xticklabels in the old x-axis
        #ax2.set_xticks(newpos)
        #ax2.set_xticklabels(newlabel)
        #
        #ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
        #ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
        #ax2.spines['bottom'].set_position(('outward', 36))
        #ax2.set_xlabel('Temperature [K]')
        #ax2.set_xlim(ax1.get_xlim())

         axhand01.legend()
         axhand02.legend()
    elif len(vlist) == 2:
         if   'kymin' not in vlist:
              params = read_parameters(scandata['filepath'][0:-9]+'/parameters')
              iscan = 'kymin = '+params['box']['kymin']+': '
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


def plot_scans(scanfiles,normalize=False):
   #Developed by Ehab Hassan on 2019-02-07
    if type(scanfiles)==list:
       for iscan in scanfiles:
           scanvals = read_scanfile(iscan)
           gammafig,omegafig = plot_scandata(scandata=scanvals,normalize=normalize)
    else:       
           scanvals = read_scanfile(scanfiles)
           gammafig,omegafig = plot_scandata(scandata=scanvals,normalize=normalize)

    for item in range(len(gammafig)):
        plt.show(gammafig[item])
        plt.show(omegafig[item])

    if raw_input('Do you want to save these figures [Yes/No]? ').lower() in ['yes','y']:
       pdfpages = pdfh.PdfPages('scanfigs.pdf')
       for item in range(len(gammafig)):
           gammafig[item].savefig('gamma%02d.png' % (item+1))
           omegafig[item].savefig('omega%02d.png' % (item+1))
           pdfpages.savefig(gammafig[item])
           pdfpages.savefig(omegafig[item])
       pdfpages.close()

    return 1


scanfpath = []
if   len(sys.argv[1:]) >= 1:
     for iarg in sys.argv[1:]:
         scanfpath.append(iarg)
else:
     if raw_input('Do you want to compare several scans in several files? (yes/no)').lower() in ['yes','y']:
        while True:
              infpath = raw_input('Path to scan file: ')
              if len(infpath) > 0: scanfpath.append(infpath)
              else: break
     else:
        scanfpath.append(raw_input('Path to the scan file: '))

#plot_scans(scanfiles=scanfpath,normalize=True)
plot_scans(scanfiles=scanfpath,normalize=False)


