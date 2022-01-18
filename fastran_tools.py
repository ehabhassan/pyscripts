import os
import re
import sys
import json
import argparse
import efit_eqdsk
import subprocess

import numpy               as npy
import matplotlib.pyplot   as plt
import matplotlib.colors   as clr
import matplotlib.gridspec as gds

from matplotlib.lines                import Line2D
from matplotlib.backends.backend_pdf import PdfPages

from glob                import glob
from netCDF4             import Dataset
#from fastran.equilibrium import efit_eqdsk

CEND    = '\033[0m'
CRED    = '\33[31m'
CBLUE   = '\33[34m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'


def read_dcon(fn_log):
    for line in open(fn_log,"r").readlines():
        if re.compile("\s*betaN_ideal-nowall").search(line):
            betan_ideal_nowall = float(line.split(":")[-1])
        if re.compile("\s*betaN_ideal-wall").search(line):
            betan_ideal_wall = float(line.split(":")[-1])
    if "betan_ideal_nowall" not in locals(): betan_ideal_nowall = 0.0
    if "betan_ideal_wall"   not in locals(): betan_ideal_wall = 0.0
    return betan_ideal_nowall,betan_ideal_wall


#def read_state_outputs(WORK_DIR,CURRENT_EQSTATE):
#    PATH_TO_FILE = os.path.join(WORK_DIR,CURRENT_STATE)
#    cdffh = Dataset(PATH_TO_FILE, mode='r')
def read_state_outputs(statefpath):
    cdffh = Dataset(statefpath, mode='r')

    state = {}
    for name, variable in cdffh.variables.items():
        state[name]                  = {}
        state[name]['data']          = cdffh.variables[name][:]
        state[name]['units']         = ""
        state[name]['long_name']     = ""
        state[name]['specification'] = ""
        if   name == 'curt':            state[name]['symbol'] = "$I_{TOT}$"
        elif name == 'curlh':           state[name]['symbol'] = "$I_{LH}$"
        elif name == 'curech':          state[name]['symbol'] = "$I_{ECH}$"
        elif name == 'curich':          state[name]['symbol'] = "$I_{ICH}$"
        elif name == 'curbeam':         state[name]['symbol'] = "$I_{NB}$"
        elif name == 'curr_ohmic':      state[name]['symbol'] = "$I_{OH}$"
        elif name == 'curr_bootstrap':  state[name]['symbol'] = "$I_{BS}$"
        else:                           state[name]['symbol'] = ""
        varattrs = variable.ncattrs()
        if varattrs:
           for attrname in varattrs:
               if   attrname == 'units':         state[name]['units']         = getattr(variable,attrname)
               elif attrname == 'long_name':     state[name]['long_name']     = getattr(variable,attrname)
               elif attrname == 'specification': state[name]['specification'] = getattr(variable,attrname)
    return state


#def read_fastran_outputs(WORK_DIR,CURRENT_FASTRAN):
#    PATH_TO_FILE = os.path.join(WORK_DIR,CURRENT_FASTRAN)
#    cdffh = Dataset(PATH_TO_FILE, mode='r')
def read_fastran_outputs(fastranfpath):
    cdffh = Dataset(fastranfpath, mode='r')

    fastran = {}
    for name, variable in cdffh.variables.items():
        fastran[name]                  = {}
        fastran[name]['data']          = cdffh.variables[name][:]
        fastran[name]['units']         = ""
        fastran[name]['symbol']        = ""
        fastran[name]['long_name']     = ""
        fastran[name]['specification'] = ""
        varattrs = variable.ncattrs()
        if varattrs:
           for attrname in varattrs:
               if   attrname == 'units':         fastran[name]['units']         = getattr(variable,attrname)
               elif attrname == 'long_name':     fastran[name]['long_name']     = getattr(variable,attrname)
               elif attrname == 'specification': fastran[name]['specification'] = getattr(variable,attrname)
        if name == 'q':         fastran[name]['symbol'] = "$q$"
        if name == 'a':         fastran[name]['symbol'] = "$a$"
        if name == 'R':         fastran[name]['symbol'] = "$R$"
        if name == 'b0':        fastran[name]['symbol'] = "$B_0$"
        if name == 'te':        fastran[name]['symbol'] = "$T_e$"           # DONE
        if name == 'ti':        fastran[name]['symbol'] = "$T_i$"           # DONE
        if name == 'ne':        fastran[name]['symbol'] = "$n_e$"           # DONE
        if name == 'ni':        fastran[name]['symbol'] = "$n_i$"           # DONE
        if name == 'pe':        fastran[name]['symbol'] = "$P_e$"           # DONE
        if name == 'pi':        fastran[name]['symbol'] = "$P_i$"           # DONE
        if name == 'nz0':       fastran[name]['symbol'] = "$n_z$"
        if name == 'rho':       fastran[name]['symbol'] = "$\\rho$"
        if name == 'fpol':      fastran[name]['symbol'] = "$f$"
        if name == 'j_bs':      fastran[name]['symbol'] = "$J_{BS}$"        # DONE
        if name == 'j_nb':      fastran[name]['symbol'] = "$J_{NB}$"        # DONE
        if name == 'j_rf':      fastran[name]['symbol'] = "$J_{RF}$"        # DONE
        if name == 'j_oh':      fastran[name]['symbol'] = "$J_{OH}$"        # DONE
        if name == 'shat':      fastran[name]['symbol'] = "$\\hat{s}$"
        if name == 'chii':      fastran[name]['symbol'] = "$\\chi_i$"
        if name == 'chie':      fastran[name]['symbol'] = "$\\chi_e$"
        if name == 'pnbe':      fastran[name]['symbol'] = "$P_{NB_e}$"
        if name == 'pnbi':      fastran[name]['symbol'] = "$P_{NB_i}$"
        if name == 'prfe':      fastran[name]['symbol'] = "$P_{RF_e}$"
        if name == 'prfi':      fastran[name]['symbol'] = "$P_{RF_i}$"
        if name == 'pe_nb':     fastran[name]['symbol'] = "$P_{NB_e}$"
        if name == 'pi_nb':     fastran[name]['symbol'] = "$P_{NB_i}$"
        if name == 'pe_rf':     fastran[name]['symbol'] = "$P_{RF_e}$"
        if name == 'pi_rf':     fastran[name]['symbol'] = "$P_{RF_i}$"
        if name == 'fluxe':     fastran[name]['symbol'] = "$Q_e$"
        if name == 'fluxi':     fastran[name]['symbol'] = "$Q_i$"
        if name == 'j_tot':     fastran[name]['symbol'] = "$J_{TOT}$"       # DONE
        if name == 'omega':     fastran[name]['symbol'] = "$\\Omega$"
        if name == 'delta':     fastran[name]['symbol'] = "$\\delta$"
        if name == 'kappa':     fastran[name]['symbol'] = "$\\kappa$"
        if name == 'betan':     fastran[name]['symbol'] = "$\\beta_n$"
        if name == 'shift':     fastran[name]['symbol'] = "$\\delta_r$"
        if name == 'pfuse':     fastran[name]['symbol'] = "$P_{FUS_e}$"
        if name == 'pfusi':     fastran[name]['symbol'] = "$P_{FUS_i}$"
        if name == 'pe_fus':    fastran[name]['symbol'] = "$P_{FUS_e}$"
        if name == 'pi_fus':    fastran[name]['symbol'] = "$P_{FUS_i}$"
        if name == 'j_bs_0':    fastran[name]['symbol'] = "$J_{BS_0}$"
        if name == 'chieneo':   fastran[name]['symbol'] = "$\\chi_{e_{neo}}$"
        if name == 'chiineo':   fastran[name]['symbol'] = "$\\chi_{i_{neo}}$"
        if name == 'chieexp':   fastran[name]['symbol'] = "$\\chi_{e_{exp}}$"
        if name == 'chiiexp':   fastran[name]['symbol'] = "$\\chi_{i_{exp}}$"
        if name == 'fluxe_exp': fastran[name]['symbol'] = "$Q_{e_{exp}}$"
        if name == 'fluxi_exp': fastran[name]['symbol'] = "$Q_{i_{exp}}$"
    return fastran


def read_fastran(WORK_DIR):
    fastrandata = {}

    CASE_ID = 0
    shotref = ""

    for iWORK_DIR in WORK_DIR:
        iWORK_DIR_PATH = '%s/work/plasma_state' % os.path.abspath(iWORK_DIR)
        WORK_FILES = glob('%s/f*' % (iWORK_DIR_PATH))
        if WORK_FILES:
           FASTRAN_FILEPATH = WORK_FILES[0]
           FASTRAN_FILENAME = FASTRAN_FILEPATH.replace(iWORK_DIR_PATH+'/','')
           SHOT_NUMBER, TIME_ID = FASTRAN_FILENAME[1:].split('.')
           shot = SHOT_NUMBER + '.' + TIME_ID
           if shot == shotref:
               CASE_ID += 1
           elif shotref == "":
               shotref = shot
               CASE_ID += 1
           elif shotref != "" and shot != shotref:
               shotref = shot
               CASE_ID = 1
           shot = SHOT_NUMBER + '.' + TIME_ID + '.%02d' % CASE_ID

           fastrandata[shot] = read_fastran_outputs(FASTRAN_FILEPATH)
           print(CGREEN + 'READING CURRENT_FASTRAN IN %s ... PASSED' % iWORK_DIR_PATH + CEND)
        else:
           print(CRED   + 'READING CURRENT_FASTRAN IN %s ... FAILED' % iWORK_DIR_PATH + CEND)

    return fastrandata


def fastran_summary(fastrandata,**kwargs):
    shots = list(fastrandata.keys())
    nshot = len(shots)

    reportpath = os.path.abspath(".")+"/fastran_report/"
    if not os.path.isdir(reportpath):
       os.system('mkdir '+reportpath)

    infopath = os.path.abspath(".")+"/fastran_report/Summary/"
    if not os.path.isdir(infopath):
       os.system('mkdir '+infopath)

    for shot in shots:
        fhand = open("%ssummary_%s.dat" % (infopath,shot),"w")

        Pext  = 0.0
        for ipext in ["prfi","prfe","pnbi","pnbe"]:
            Pext += fastrandata[shot][ipext]['data'][-1]
        Pfus  = 0.0
        for ipfus in ["pfusi","pfuse"]:
            Pfus += fastrandata[shot][ipfus]['data'][-1]
        Q = 5.0*Pfus/Pext

        INI  = 0.0
        for iINI in ["ibs","inb","irf"]:
            INI += fastrandata[shot][iINI]['data'][-1]
        fNI = INI/fastrandata[shot]["ip"]['data'][-1]
        fBS = fastrandata[shot]["ibs"]['data'][-1]/fastrandata[shot]["ip"]['data'][-1]

        H98 = fastrandata[shot]["tauth"]['data'][-1]/fastrandata[shot]["tau98"]['data'][-1]
        H89 = fastrandata[shot]["tauth"]['data'][-1]/fastrandata[shot]["tau89"]['data'][-1]

        fhand.write("%s\t%5.3f\n" % ("Q",Q))
        fhand.write("%s\t%5.3f\n" % ("fNI",fNI))
        fhand.write("%s\t%5.3f\n" % ("fBS",fBS))
        fhand.write("%s\t%5.3f\n" % ("H98",H98))
        fhand.write("%s\t%5.3f\n" % ("H89",H89))
        fhand.write("%s\t%5.3f\n" % ("betan",fastrandata[shot]["betan"]['data'][-1]))

    fhand.close()

    return 1

def plot_fastran_outputs(fastrandata,plotparam={},**kwargs):
    shots = list(fastrandata.keys())
    nshot = len(shots)

    if 'newplot' in plotparam:  newplot = plotparam['newplot']
    else:                       newplot = False

    if 'savepng' in plotparam:  savepng = plotparam['savepng']
    else:                       savepng = True

    if 'figspec' in plotparam:  figspec = plotparam['figspec']
    else:                       figspec = False

    reportpath = os.path.abspath(".")+"/fastran_report/"
    if not os.path.isdir(reportpath):
       os.system('mkdir '+reportpath)

    figurepath = os.path.abspath(".")+"/fastran_report/Figures/"
    if not os.path.isdir(figurepath):
       os.system('mkdir '+figurepath)

    if newplot and glob("%s*.png" % figurepath):
        os.system("rm %s*.png" % figurepath)

    fastranfigs = PdfPages(figurepath+'fastran_plots.pdf')
    colornames = clr.cnames.items()

    colors = ['red','blue','green','orange','black','cyan','sienna','magenta','teal','olive','steelblue','deeppink','indigo','purple','maroon','lime']
    styles = [(0,()),(0,(1,10)),(0,(1,1)),(0,(5,1)),(0,(5,5)),(0,(5,10))]
    styles.extend([(0,(3,10,1,10)),(0,(3,5,1,5)),(0,(3,1,1,1)),(0,(3,5,1,5,1,5)),(0,(3,10,1,10,1,10)),(0,(3,1,1,1,1,1))])

    for key,value in kwargs.items():
        if key in ['state','statedata']:
           statedata = value.copy()
           stateflag = True

    if figspec:
        if not os.path.isfile("figspec.json"):
            print("figspec.json DOES NOT EXIST ... EXIT!")
            sys.exit()
        jsonfhand = open('figspec.json')
        jsonfdata = json.load(jsonfhand)
        jsonfhand.close()

        axs = [None for i in range(9)]
        for ifig in range(len(jsonfdata["figures"])):
            fig = plt.figure(jsonfdata["figures"][ifig]["name"],dpi=200)
            axa = []
            for isubfig in range(len(jsonfdata["figures"][ifig]["subplots"])):
                figgrd = int("%d%d%d" % (jsonfdata["figures"][ifig]['grid'][0],jsonfdata["figures"][ifig]['grid'][1],isubfig+1))
                axs[isubfig] = fig.add_subplot(figgrd)
                if type(jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"]) not in [list,tuple]:
                    jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"] = [jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"]]
                llabel2 = []

                for ifield in jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"]:
                    for shot in shots:
                        if len(shots) > 1:
                            lcolor = colors[shots.index(shot)]
                            lstyle = styles[jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"].index(ifield)]
                        else:
                            lcolor = colors[jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"].index(ifield)]
                            lstyle = styles[shots.index(shot)]

                        if jsonfdata["figures"][ifig]["subplots"][isubfig]['label2']:
                            legned2 = True
                        else:
                            legned2 = False

                        if legned2 and ifield == jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"][0]:
                            if not jsonfdata["figures"][ifig]["subplots"][isubfig]['label2']:
                                llabel2.append(shot)
                            else:
                                if type(jsonfdata["figures"][ifig]["subplots"][isubfig]['label2']) in [list,tuple]:
                                    try:
                                        llabel2.append(jsonfdata["figures"][ifig]["subplots"][isubfig]['label2'][shots.index(shot)])
                                    except IndexError:
                                        pass
                                else:
                                        llabel2.append(jsonfdata["figures"][ifig]["subplots"][isubfig]['label2'])
                                        legend2 = False

                        if ifield in fastrandata[shot]:
                            if shot == shots[0]:
                                if not jsonfdata["figures"][ifig]["subplots"][isubfig]['label']:
                                    if "symbol" in fastrandata[shot][ifield]:
                                        llabel = fastrandata[shot][ifield]['symbol']
                                    else:
                                        llabel = ifield
                                else:
                                    if jsonfdata["figures"][ifig]["subplots"][isubfig]['label'] in fastrandata[shot]:
                                        llabel = fastrandata[shot][jsonfdata["figures"][ifig]["subplots"][isubfig]['label']]['symbol']
                                    else:
                                        llabel = jsonfdata["figures"][ifig]["subplots"][isubfig]['label']
                            else:
                                        llabel = ""

                            axs[isubfig].plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot][ifield]['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)

                axs[isubfig].set_title( jsonfdata["figures"][ifig]["subplots"][isubfig]["title"])
                if not jsonfdata["figures"][ifig]["subplots"][isubfig]["ylabel"]:
                    axs[isubfig].set_ylabel(ifield)
                else:
                    axs[isubfig].set_ylabel(jsonfdata["figures"][ifig]["subplots"][isubfig]["ylabel"])
                if not jsonfdata["figures"][ifig]["subplots"][isubfig]["xlabel"]:
                    axs[isubfig].set_xlabel("$\\rho$")
                else:
                    axs[isubfig].set_xlabel(jsonfdata["figures"][ifig]["subplots"][isubfig]['xlabel'])
                if not jsonfdata["figures"][ifig]["subplots"][isubfig]['xticks']:
                   axs[isubfig].set_xticks([])
                if 'legned2' in locals() and legned2:
                    lines = [Line2D([0],[0],color=colors[c],linestyle='-') for c in range(len(llabel2))]
                    labels = llabel2
                    if jsonfdata["figures"][ifig]["subplots"][isubfig]['legend']:
                        if jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc2']:
                            leg2 = axs[isubfig].legend(lines, labels,loc=jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc2'],frameon=False,fontsize='7')
                        else:
                            leg2 = axs[isubfig].legend(lines, labels,loc='upper center',frameon=False,fontsize='7')
                    else:
                        if jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc2']:
                            leg2 = axs[isubfig].legend(lines, labels,loc=jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc2'],frameon=False,fontsize='7')
                        else:
                            leg2 = axs[isubfig].legend(lines, labels,loc='upper right',frameon=Falsei,fontsize='7')
                if jsonfdata["figures"][ifig]["subplots"][isubfig]['legend']:
                    if jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc']:
                        leg1 = axs[isubfig].legend(loc=jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc'],markerscale=19,frameon=False,fontsize='10')
                    else:
                        leg1 = axs[isubfig].legend(loc='upper right',markerscale=19,frameon=False,fontsize='10')
                if 'legned2' in locals() and legned2:
                    axs[isubfig].add_artist(leg2)
                if jsonfdata["figures"][ifig]["subplots"][isubfig]['yaxdir'] == "right":
                   axs[isubfig].yaxis.tick_right()
                   axs[isubfig].yaxis.set_label_position("right")
                if jsonfdata["figures"][ifig]["subplots"][isubfig]['xaxdir'] == "top":
                   axs[isubfig].xaxis.tick_top()
                   axs[isubfig].xaxis.set_label_position("top")
                if jsonfdata["figures"][ifig]["subplots"][isubfig]["ylimit"]:
                   axs[isubfig].set_ylim(jsonfdata["figures"][ifig]["subplots"][isubfig]["ylimit"])

            if jsonfdata["figures"][ifig]["title"]:
                fig.suptitle(jsonfdata["figures"][ifig]["title"])
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.subplots_adjust(wspace=0,hspace=0)
            fastranfigs.savefig(fig)
            if savepng: fig.savefig(figurepath+"%s.png" % jsonfdata["figures"][ifig]['name'])
            plt.close(fig)

        fastranfigs.close()

    else:
        Tefig = plt.figure("ELECTRON TEMPERATURE PROFILE",dpi=200)
        Teaxs = Tefig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lstyle = styles[0]
            llabel = shot
            Teaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['te']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        Teaxs.set_title("Electron Temperature Profile")
        Teaxs.set_ylabel("$T_e$")
        Teaxs.set_xlabel("$\\rho$")
        Teaxs.legend()
        fastranfigs.savefig(Tefig)
        if savepng: Tefig.savefig(figurepath+"Te.png")
        plt.close(Tefig)

        Tifig = plt.figure("ION TEMPERATURE PROFILE",dpi=200)
        Tiaxs = Tifig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lstyle = styles[0]
            llabel = shot
            Tiaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['ti']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        Tiaxs.set_title("Ion Temperature Profile")
        Tiaxs.set_ylabel("$T_i$")
        Tiaxs.set_xlabel("$\\rho$")
        Tiaxs.legend()
        fastranfigs.savefig(Tifig)
        if savepng: Tifig.savefig(figurepath+"Ti.png")
        plt.close(Tifig)


        nefig = plt.figure("ELECTRON DENSITY PROFILE",dpi=200)
        neaxs = nefig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lstyle = styles[0]
            llabel = shot
            neaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['ne']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        neaxs.set_title("Electron Density Profile")
        neaxs.set_ylabel("$n_e$")
        neaxs.set_xlabel("$\\rho$")
        neaxs.legend()
        fastranfigs.savefig(nefig)
        if savepng: nefig.savefig(figurepath+"ne.png")
        plt.close(nefig)

        nifig = plt.figure("ION DENSITY PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lsytle = styles[0]
            llabel = shot
            niaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['ni']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Ion Density Profile")
        niaxs.set_ylabel("$n_i$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"ni.png")
        plt.close(nifig)

        nifig = plt.figure("BOOTSTRAP CURRENT PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lsytle = styles[0]
            llabel = shot
            niaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_bs']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Bootstrap Current Profile")
        niaxs.set_ylabel("$J_{bs}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"jbs.png")
        plt.close(nifig)

        nifig = plt.figure("OHMIC CURRENT PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lsytle = styles[0]
            llabel = shot
            niaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_oh']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Ohmic Current Profile")
        niaxs.set_ylabel("$J_{oh}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"joh.png")
        plt.close(nifig)

        nifig = plt.figure("RF CURRENT PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lsytle = styles[0]
            llabel = shot
            niaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_rf']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("RF Current Profile")
        niaxs.set_ylabel("$J_{rf}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"jrf.png")
        plt.close(nifig)

        nifig = plt.figure("TOTAL CURRENT PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lsytle = styles[0]
            llabel = shot
            niaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_tot']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Total Current Profile")
        niaxs.set_ylabel("$J_{tot}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"jtot.png")
        plt.close(nifig)

        nifig = plt.figure("SAFETY FACTOR PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lsytle = styles[0]
            llabel = shot
            niaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['q']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Safety Factor Profile")
        niaxs.set_ylabel("$q$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"q.png")
        plt.close(nifig)

        nifig = plt.figure("MAGNETIC SHEAR PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lsytle = styles[0]
            llabel = shot
            niaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['shat']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Magnetic Shear Profile")
        niaxs.set_ylabel("$\\hat{s}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"shat.png")
        plt.close(nifig)

        nifig = plt.figure("MAGNETIC PERTURBATION PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for shot in shots:
            lcolor = colors[shots.index(shot)]
            lsytle = styles[0]
            llabel = shot
            niaxs.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['betan_loc']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Magnetic Perturbation Profile")
        niaxs.set_ylabel("$\\beta_N$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"betan.png")
        plt.close(nifig)

        fastranfigs.close()

#   else:
#       Tnfig = plt.figure("T-n PROFILES",figsize=(9,7),dpi=200)
#       Tnfig.tight_layout()
#       Tnfig.subplots_adjust(wspace=0,hspace=0)
#       Tnax1 = Tnfig.add_subplot(221)
#       Tnax2 = Tnfig.add_subplot(222)
#       Tnax3 = Tnfig.add_subplot(223)
#       Tnax4 = Tnfig.add_subplot(224)

#       Jfig = plt.figure("J PROFILES",figsize=(9,7),dpi=200)
#       Jfig.tight_layout()
#       Jfig.subplots_adjust(wspace=0,hspace=0)
#       Jax1 = Jfig.add_subplot(221)
#       Jax2 = Jfig.add_subplot(222)
#       Jax3 = Jfig.add_subplot(223)
#       Jax4 = Jfig.add_subplot(224)

#       SHPfig = plt.figure("Species HP PROFILES",figsize=(9,9),dpi=200)
#       SHPfig.tight_layout()
#       SHPfig.subplots_adjust(wspace=0,hspace=0)
#       SHPax1 = SHPfig.add_subplot(421)
#       SHPax2 = SHPfig.add_subplot(422)
#       SHPax3 = SHPfig.add_subplot(423)
#       SHPax4 = SHPfig.add_subplot(424)
#       SHPax5 = SHPfig.add_subplot(425)
#       SHPax6 = SHPfig.add_subplot(426)
#       SHPax7 = SHPfig.add_subplot(427)
#       SHPax8 = SHPfig.add_subplot(428)

#       OHPfig = plt.figure("Other HP PROFILES",figsize=(9,9),dpi=200)
#       OHPfig.tight_layout()
#       OHPfig.subplots_adjust(wspace=0,hspace=0)
#       OHPax1 = OHPfig.add_subplot(321)
#       OHPax2 = OHPfig.add_subplot(322)
#       OHPax3 = OHPfig.add_subplot(323)
#       OHPax4 = OHPfig.add_subplot(324)
#       OHPax5 = OHPfig.add_subplot(325)
#       OHPax6 = OHPfig.add_subplot(326)

#       HTRfig = plt.figure("Heat TR PROFILES",figsize=(9,9),dpi=200)
#       HTRfig.tight_layout()
#       HTRfig.subplots_adjust(wspace=0,hspace=0)
#       HTRax1 = HTRfig.add_subplot(321)
#       HTRax2 = HTRfig.add_subplot(322)
#       HTRax3 = HTRfig.add_subplot(323)
#       HTRax4 = HTRfig.add_subplot(324)
#       HTRax5 = HTRfig.add_subplot(325)
#       HTRax6 = HTRfig.add_subplot(326)

#       QFXfig = plt.figure("HEAT FX PROFILES",figsize=(9,7),dpi=200)
#       QFXfig.tight_layout()
#       QFXfig.subplots_adjust(wspace=0,hspace=0)
#       QFXax1 = QFXfig.add_subplot(221)
#       QFXax2 = QFXfig.add_subplot(222)
#       QFXax3 = QFXfig.add_subplot(223)
#       QFXax4 = QFXfig.add_subplot(224)

#       OZRfig = plt.figure("Other PROFILES",figsize=(9,9),dpi=200)
#       OZRfig.tight_layout()
#       OZRfig.subplots_adjust(wspace=0,hspace=0)
#       OZRax1 = OZRfig.add_subplot(321)
#       OZRax2 = OZRfig.add_subplot(322)
#       OZRax3 = OZRfig.add_subplot(323)
#       OZRax4 = OZRfig.add_subplot(324)
#       OZRax5 = OZRfig.add_subplot(325)
#       OZRax6 = OZRfig.add_subplot(326)

#      #DTLfig = plt.figure("Simulation Details",figsize=(9,9),dpi=200,constrained_layout=False)
#      #if   nshot == 1: widthratios = [2, 8]
#      #elif nshot == 2: widthratios = [2, 4, 4]
#      #elif nshot == 3: widthratios = [1, 3, 3, 3]
#      #elif nshot == 4: widthratios = [2, 2, 2, 2, 2]
#      #DTLgds = gds.GridSpec(nrows=1, ncols=nshot, width_ratios=widthratios) 
#      #DTLgds = DTLfig.add_gridspec(nrows=1, ncols=nshot, left=0.05, right=0.48, wspace=0.05)
#      #DTLax1 = DTLfig.add_subplot(DTLgds[0])
#      #if   nshot == 1:
#      #     DTLax2 = DTLfig.add_subplot(DTLgds[1])
#      #elif nshot == 2:
#      #     DTLax2 = DTLfig.add_subplot(DTLgds[1])
#      #     DTLax3 = DTLfig.add_subplot(DTLgds[2])
#      #elif nshot == 3:
#      #     DTLax2 = DTLfig.add_subplot(DTLgds[1])
#      #     DTLax3 = DTLfig.add_subplot(DTLgds[2])
#      #     DTLax4 = DTLfig.add_subplot(DTLgds[3])
#      #elif nshot == 4:
#      #     DTLax2 = DTLfig.add_subplot(DTLgds[1])
#      #     DTLax3 = DTLfig.add_subplot(DTLgds[2])
#      #     DTLax4 = DTLfig.add_subplot(DTLgds[3])
#      #     DTLax5 = DTLfig.add_subplot(DTLgds[4])

#       colors = ['red','blue','green','orange','black','cyan','sienna','magenta','teal','olive']

#       for shot in shots:
#           print(CYELLOW + 'PLOTTING FASTRAN Results from %s' % shot + CEND)
#           print("betan = ", fastrandata[shot]['betan']['data'])
#      #  # print("pfuse = ", fastrandata[shot]['pfuse']['data'])
#      #  # print("pfusi = ", fastrandata[shot]['pfusi']['data'])
#      #  # print("pnbe = ", fastrandata[shot]['pnbe']['data'])
#      #  # print("pnbi = ", fastrandata[shot]['pnbi']['data'])
#      #  # print("prfe = ", fastrandata[shot]['prfe']['data'])
#      #  # print("prfi = ", fastrandata[shot]['prfi']['data'])
#      #  # print("prad = ", fastrandata[shot]['prad']['data'])
#      #  # print("pei = ", fastrandata[shot]['pei']['data'])
#      #  # print("poh = ", fastrandata[shot]['pei']['data'])
#      #  # print("we = ", fastrandata[shot]['we']['data'])
#      #  # print("wi = ", fastrandata[shot]['wi']['data'])
#      #  # print("wb = ", fastrandata[shot]['wb']['data'])
#      #  # print("taue = ", fastrandata[shot]['taue']['data'])
#      #  # print("taui = ", fastrandata[shot]['taui']['data'])
#      #  # print("tauth = ", fastrandata[shot]['tauth']['data'])
#      #  # print("tautot = ", fastrandata[shot]['tautot']['data'])
#      #  # print("tau98 = ", fastrandata[shot]['tau98']['data'])
#      #  # print("tau89 = ", fastrandata[shot]['tau89']['data'])
#      #  # print("taunc = ", fastrandata[shot]['taunc']['data'])
#      #  # print("tea = ", fastrandata[shot]['tea']['data'])
#      #  # print("tia = ", fastrandata[shot]['tia']['data'])
#      #  # print("nebar = ", fastrandata[shot]['nebar']['data'])
#      #  # print("ip = ", fastrandata[shot]['ip']['data'])
#      #  # print("ibs = ", fastrandata[shot]['ibs']['data'])
#      #  # print("inb = ", fastrandata[shot]['inb']['data'])
#      #  # print("irf = ", fastrandata[shot]['irf']['data'])
#      #  # print("sn = ", fastrandata[shot]['sn']['data'])
#      #  # print("aimp= ", fastrandata[shot]['aimp']['data'])
#      #  # print("zimp = ", fastrandata[shot]['zimp']['data'])
#      #  # print("amain = ", fastrandata[shot]['amain']['data'])
#      #  # print("zmain = ", fastrandata[shot]['zmain']['data'])
#      #  # print("Time = ", fastrandata[shot]['time']['data'])
#      #  # print("r0 = ", fastrandata[shot]['r0']['data'])
#      #  # print("b0 = ", fastrandata[shot]['b0']['data'])
#      #  # print("a0 = ", fastrandata[shot]['a0']['data'])


#           shotcolor = colors.pop()
#           for item in fastrandata[shots[0]]:
#               if item in ['te','ti','ne','ni']:
#                  if   item == 'te':
#                       Tnax1.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['te']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       Tnax1.set_xticks([])
#                       Tnax1.set_ylabel("$T_e$")
#                  elif item == 'ti':
#                       Tnax2.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['ti']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       Tnax2.set_xticks([])
#                       Tnax2.yaxis.tick_right()
#                       Tnax2.yaxis.set_label_position("right")
#                       Tnax2.set_ylabel("$T_i$")
#                  elif item == 'ne':
#                       Tnax3.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['ne']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       Tnax3.set_ylabel("$n_e$")
#                       Tnax3.set_xlabel("$\\rho$")
#                  elif item == 'ni':
#                       Tnax4.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['ni']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       Tnax4.yaxis.tick_right()
#                       Tnax4.yaxis.set_label_position("right")
#                       Tnax4.set_ylabel("$n_i$")
#                       Tnax4.set_xlabel("$\\rho$")
#                       Tnax4.legend()

#               if item in ['j_bs','j_rf','j_nb','j_oh','j_bs_0','j_tot']:
#                  if   item == 'j_bs':
#                       Jax1.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_bs']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       Jax1.set_xticks([])
#                       Jax1.set_ylabel("$J_{BS}$")
#                  elif item == 'j_oh':
#                       Jax2.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_oh']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       Jax2.set_xticks([])
#                       Jax2.yaxis.tick_right()
#                       Jax2.yaxis.set_label_position("right")
#                       Jax2.set_ylabel("$J_{OH}$")
#                  elif item == 'j_nb':
#                       Jax3.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_nb']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       Jax3.set_ylabel("$J_{NB}$")
#                       Jax3.set_xlabel("$\\rho$")
#                  elif item == 'j_rf':
#                       Jax4.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_rf']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       Jax4.yaxis.tick_right()
#                       Jax4.yaxis.set_label_position("right")
#                       Jax4.set_ylabel("$J_{RF}$")
#                       Jax4.set_xlabel("$\\rho$")
#                       Jax4.legend()
#                  elif item == 'j_tot':
#                       Jax1.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_tot']['data'][-1,:],color=shotcolor,linestyle='--')
#                       Jax2.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_tot']['data'][-1,:],color=shotcolor,linestyle='--')
#                       Jax3.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_tot']['data'][-1,:],color=shotcolor,linestyle='--')
#                       Jax4.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['j_tot']['data'][-1,:],color=shotcolor,linestyle='--')

#               if item in ['pe_nb','pi_nb','pe_rf','pi_rf','pe_fus','pi_fus','pe_ionization','pi_ionization']:
#                  if   item == 'pe_nb':
#                       SHPax1.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pe_nb']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       SHPax1.set_xticks([])
#                       SHPax1.set_ylabel("$P_{NB_e}$")
#                  elif item == 'pi_nb':
#                       SHPax2.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pi_nb']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       SHPax2.set_xticks([])
#                       SHPax2.yaxis.tick_right()
#                       SHPax2.yaxis.set_label_position("right")
#                       SHPax2.set_ylabel("$P_{NB_i}$")
#                  elif item == 'pe_rf':
#                       SHPax3.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pe_rf']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       SHPax3.set_xticks([])
#                       SHPax3.set_ylabel("$P_{RF_e}$")
#                       SHPax3.set_xlabel("$\\rho$")
#                  elif item == 'pi_rf':
#                       SHPax4.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pi_rf']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       SHPax4.set_xticks([])
#                       SHPax4.yaxis.tick_right()
#                       SHPax4.yaxis.set_label_position("right")
#                       SHPax4.set_ylabel("$P_{RF_i}$")
#                       SHPax4.set_xlabel("$\\rho$")
#                  elif item == 'pe_fus':
#                       SHPax5.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pe_fus']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       SHPax5.set_xticks([])
#                       SHPax5.set_ylabel("$P_{FUS_e}$")
#                       SHPax5.set_xlabel("$\\rho$")
#                  elif item == 'pi_fus':
#                       SHPax6.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pi_fus']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       SHPax6.set_xticks([])
#                       SHPax6.yaxis.tick_right()
#                       SHPax6.yaxis.set_label_position("right")
#                       SHPax6.set_ylabel("$P_{FUS_i}$")
#                       SHPax6.set_xlabel("$\\rho$")
#                  elif item == 'pe_ionization':
#                       SHPax7.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pe_ionization']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       SHPax7.set_ylabel("$P_{INZ_e}$")
#                       SHPax7.set_xlabel("$\\rho$")
#                  elif item == 'pi_ionization':
#                       SHPax8.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pi_ionization']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       SHPax8.yaxis.tick_right()
#                       SHPax8.yaxis.set_label_position("right")
#                       SHPax8.set_ylabel("$P_{INZ_i}$")
#                       SHPax8.set_xlabel("$\\rho$")
#                       SHPax8.legend()

#               if item in ['pe','pi','p_ei','pi_cx','p_rad','p_ohm']:
#                  if   item == 'pe':
#                       OHPax1.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pe']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OHPax1.set_xticks([])
#                       OHPax1.set_ylabel("$P_e$")
#                  elif item == 'pi':
#                       OHPax2.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pi']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OHPax2.set_xticks([])
#                       OHPax2.yaxis.tick_right()
#                       OHPax2.yaxis.set_label_position("right")
#                       OHPax2.set_ylabel("$P_i$")
#                  elif item == 'p_ei':
#                       OHPax3.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['p_ei']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OHPax3.set_xticks([])
#                       OHPax3.set_ylabel("$P_{ei}$")
#                       OHPax3.set_xlabel("$\\rho$")
#                  elif item == 'pi_cx':
#                       OHPax4.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['pi_cx']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OHPax4.set_xticks([])
#                       OHPax4.yaxis.tick_right()
#                       OHPax4.yaxis.set_label_position("right")
#                       OHPax4.set_ylabel("$P_{CX_i}$")
#                       OHPax4.set_xlabel("$\\rho$")
#                  elif item == 'p_rad':
#                       OHPax5.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['p_rad']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OHPax5.set_ylabel("$P_{RAD}$")
#                       OHPax5.set_xlabel("$\\rho$")
#                  elif item == 'p_ohm':
#                       OHPax6.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['p_ohm']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OHPax6.yaxis.tick_right()
#                       OHPax6.yaxis.set_label_position("right")
#                       OHPax6.set_ylabel("$P_{OHM}$")
#                       OHPax6.set_xlabel("$\\rho$")
#                       OHPax6.legend()

#               if item in ['chie','chii','chieneo','chiineo','chie_exp','chii_exp']:
#                  if   item == 'chie':
#                       HTRax1.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['chie']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       HTRax1.set_xticks([])
#                       HTRax1.set_ylabel("$\\chi_e$")
#                  elif item == 'chii':
#                       HTRax2.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['chii']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       HTRax2.set_xticks([])
#                       HTRax2.yaxis.tick_right()
#                       HTRax2.yaxis.set_label_position("right")
#                       HTRax2.set_ylabel("$\\chi_i$")
#                  elif item == 'chieneo':
#                       HTRax3.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['chieneo']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       HTRax3.set_xticks([])
#                       HTRax3.set_ylabel("$\\chi_{NEO_e}$")
#                       HTRax3.set_xlabel("$\\rho$")
#                  elif item == 'chiineo':
#                       HTRax4.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['chiineo']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       HTRax4.set_xticks([])
#                       HTRax4.yaxis.tick_right()
#                       HTRax4.yaxis.set_label_position("right")
#                       HTRax4.set_ylabel("$\\chi_{NEO_i}$")
#                       HTRax4.set_xlabel("$\\rho$")
#                  elif item == 'chie_exp':
#                       HTRax5.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['chie_exp']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       HTRax5.set_ylabel("$\\chi_{EXP_e}$")
#                       HTRax5.set_xlabel("$\\rho$")
#                  elif item == 'chii_exp':
#                       HTRax6.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['chii_exp']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       HTRax6.yaxis.tick_right()
#                       HTRax6.yaxis.set_label_position("right")
#                       HTRax6.set_ylabel("$\\chi_{EXP_i}$")
#                       HTRax6.set_xlabel("$\\rho$")
#                       HTRax6.legend()

#               if item in ['fluxe','fluxi','fluxe_exp','fluxi_exp']:
#                  if   item == 'fluxe':
#                       QFXax1.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['fluxe']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       QFXax1.set_xticks([])
#                       QFXax1.set_ylabel("$Q_e$")
#                  elif item == 'fluxi':
#                       QFXax2.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['fluxi']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       QFXax2.set_xticks([])
#                       QFXax2.yaxis.tick_right()
#                       QFXax2.yaxis.set_label_position("right")
#                       QFXax2.set_ylabel("$Q_i$")
#                  elif item == 'fluxe_exp':
#                       QFXax3.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['fluxe_exp']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       QFXax3.set_ylabel("$Q_{EXP_e}$")
#                       QFXax3.set_xlabel("$\\rho$")
#                  elif item == 'fluxi_exp':
#                       QFXax4.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['fluxi_exp']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       QFXax4.yaxis.tick_right()
#                       QFXax4.yaxis.set_label_position("right")
#                       QFXax4.set_ylabel("$Q_{EXP_i}$")
#                       QFXax4.set_xlabel("$\\rho$")
#                       QFXax4.legend()

#               if item in ['omega','betan_loc','q','shat','sigma','vloop']:
#                  if   item == 'omega':
#                       OZRax1.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['omega']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OZRax1.set_xticks([])
#                       OZRax1.set_ylabel("$\\Omega$")
#                  elif item == 'betan_loc':
#                       OZRax2.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['betan_loc']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OZRax2.set_xticks([])
#                       OZRax2.yaxis.tick_right()
#                       OZRax2.yaxis.set_label_position("right")
#                       OZRax2.set_ylabel("$\\beta_n$")
#                  elif item == 'q':
#                       OZRax3.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['q']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OZRax3.set_xticks([])
#                       OZRax3.set_ylabel("$q$")
#                       OZRax3.set_xlabel("$\\rho$")
#                  elif item == 'shat':
#                       OZRax4.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['shat']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OZRax4.set_xticks([])
#                       OZRax4.yaxis.tick_right()
#                       OZRax4.yaxis.set_label_position("right")
#                       OZRax4.set_ylabel("$\\hat{s}$")
#                       OZRax4.set_xlabel("$\\rho$")
#                  elif item == 'sigma':
#                       OZRax5.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['sigma']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OZRax5.set_ylabel("$\\sigma$")
#                       OZRax5.set_xlabel("$\\rho$")
#                  elif item == 'vloop':
#                       OZRax6.plot(fastrandata[shot]['rho']['data'][:],fastrandata[shot]['vloop']['data'][-1,:],color=shotcolor,linestyle='-',label=shot)
#                       OZRax6.yaxis.tick_right()
#                       OZRax6.yaxis.set_label_position("right")
#                       OZRax6.set_ylabel("$V_{loop}$")
#                       OZRax6.set_xlabel("$\\rho$")
#                       OZRax6.legend()

#       Jfig.suptitle("Currents Profiles")
#       Tnfig.suptitle("Density and Temperature Profiles")
#       OZRfig.suptitle("Other Profiles")
#       QFXfig.suptitle("Heat Flux Profiles")
#       HTRfig.suptitle("Heat Transport Profiles")
#       OHPfig.suptitle("Other Heating Powers Profiles")
#       SHPfig.suptitle("Species Heating Powers Profiles")

#       fastranfigs.savefig(Jfig)
#       fastranfigs.savefig(Tnfig)
#       fastranfigs.savefig(SHPfig)
#       fastranfigs.savefig(OHPfig)
#       fastranfigs.savefig(HTRfig)
#       fastranfigs.savefig(QFXfig)
#       fastranfigs.savefig(OZRfig)

#       if savepng:
#          Jfig.savefig(  figurepath+"currents.png")
#          Tnfig.savefig( figurepath+"profiles.png")
#          QFXfig.savefig(figurepath+"heatflux.png")
#          OZRfig.savefig(figurepath+"geometry.png")
#          SHPfig.savefig(figurepath+"mainheating.png")
#          OHPfig.savefig(figurepath+"otherheating.png")
#          HTRfig.savefig(figurepath+"heattransport.png")

#       plt.close(Jfig)
#       plt.close(Tnfig)
#       plt.close(SHPfig)
#       plt.close(OHPfig)
#       plt.close(HTRfig)
#       plt.close(QFXfig)
#       plt.close(OZRfig)

    return True
        

def fastran_plot(WORK_DIR,plotparam={}):
    if 'addbeta' in plotparam:  addbeta = plotparam['addbeta']
    else:                       addbeta = False
    if 'nostate' in plotparam:  nostate = plotparam['nostate']
    else:                       nostate = False
    statedata = {}
    fastrandata = {}
    
    fCURRENT_BC      = [False for i in range(len(WORK_DIR))]
    fCURRENT_STATE   = [False for i in range(len(WORK_DIR))]
    fCURRENT_EQDSK   = [False for i in range(len(WORK_DIR))]
    fCURRENT_EQSTATE = [False for i in range(len(WORK_DIR))]
    fCURRENT_INSTATE = [False for i in range(len(WORK_DIR))]
    fCURRENT_FASTRAN = [False for i in range(len(WORK_DIR))]

    CASE_ID = 0
    shotref = ""

    if addbeta and not summary:
        plotparam['betan_ideal_wall']   = {}
        plotparam['betan_ideal_nowall'] = {}

    for iWORK_DIR in WORK_DIR:
        if iWORK_DIR[-1] != "/":
            WORK_DIR[WORK_DIR.index(iWORK_DIR)] = iWORK_DIR + "/"
            iWORK_DIR += "/"
        if not summary:
            WORK_DIR[WORK_DIR.index(iWORK_DIR)] = iWORK_DIR + 'work/plasma_state/'
            iWORK_DIR += 'work/plasma_state/'
        WORK_FILES = glob(iWORK_DIR+'*')
        
        for FILE in WORK_FILES:
            FILENAME = FILE.replace(iWORK_DIR,'')
            if FILENAME[0] == 'b': CURRENT_BC      = FILENAME;  fCURRENT_BC[WORK_DIR.index(iWORK_DIR)]      =  True
            if FILENAME[0] == 's': CURRENT_STATE   = FILENAME;  fCURRENT_STATE[WORK_DIR.index(iWORK_DIR)]   =  True
            if FILENAME[0] == 'g': CURRENT_EQDSK   = FILENAME;  fCURRENT_EQDSK[WORK_DIR.index(iWORK_DIR)]   =  True
            if FILENAME[0] == 'k': CURRENT_EQSTATE = FILENAME;  fCURRENT_EQSTATE[WORK_DIR.index(iWORK_DIR)] =  True
            if FILENAME[0] == 'i': CURRENT_INSTATE = FILENAME;  fCURRENT_INSTATE[WORK_DIR.index(iWORK_DIR)] =  True
            if FILENAME[0] == 'f': CURRENT_FASTRAN = FILENAME;  fCURRENT_FASTRAN[WORK_DIR.index(iWORK_DIR)] =  True

        if fCURRENT_INSTATE[WORK_DIR.index(iWORK_DIR)]:
           SHOT_NUMBER, TIME_ID = CURRENT_STATE[1:].split('.')
           shot = SHOT_NUMBER + '.' + TIME_ID
           if shot == shotref:
               CASE_ID += 1
           elif shotref == "":
               shotref = shot
               CASE_ID += 1
           elif shotref != "" and shot != shotref:
               shotref = shot
               CASE_ID = 1
           shot = SHOT_NUMBER + '.' + TIME_ID + '.%02d' % CASE_ID

       #if fCURRENT_STATE[WORK_DIR.index(iWORK_DIR)] and not nostate:
       #   statedata[shot] = read_state_outputs(os.path.join(iWORK_DIR,CURRENT_STATE))
       #   print(CGREEN + 'READING CURRENT_STATE from %s ... PASSED' % iWORK_DIR + CEND)
       ##  for ikey in list(statedata[shot].keys()):
       ##      print(ikey,npy.size(statedata[shot][ikey]['data']),npy.shape(statedata[shot][ikey]['data']))
       ##  plt.plot(statedata[shot]['rho']['data'][:-1],statedata[shot]['curech']['data']*1.0e-6, label="ECH")
       ##  plt.plot(statedata[shot]['rho']['data'][:-1],statedata[shot]['curich']['data']*1.0e-6, label="ICH")
       ##  plt.plot(statedata[shot]['rho']['data'][:-1],statedata[shot]['curbeam']['data']*1.0e-6,label="NB")
       ##  plt.plot(statedata[shot]['rho']['data'][:-1],statedata[shot]['curr_ohmic']['data']*1.0e-6, label="OHMIC")
       ##  plt.plot(statedata[shot]['rho']['data'][:-1],statedata[shot]['curr_bootstrap']['data']*1.0e-6, label="BS")
       ##  plt.title("Current Drives")
       ##  plt.xlabel("$\\rho_{\\psi}$")
       ##  plt.ylabel("J ($MA/m^2$)")
       ##  plt.legend()
       ##  plt.show()
       #  #if iWORK_DIR == WORK_DIR[0]:
       #  #    print(CRED + "STATE DATA" + CEND)
       #  #    f = open("statedata.dat", "w")
       #  #    for ikey in sorted(list(statedata[shot].keys())):
       #  #        f.write("%s\t\t%s\t\t%s\n" % (ikey,str(npy.shape(statedata[shot][ikey]['data'])),statedata[shot][ikey]['long_name']))
       #  #        print(ikey,npy.shape(statedata[shot][ikey]['data']),statedata[shot][ikey]['long_name'])
       #  #        if ikey in ['triang','Rmajor_mean','rMinor_mean','elong','B_axis']:
       #  #          print(ikey,": ",statedata[shot][ikey]['long_name']," = ",statedata[shot][ikey]['data'])
       #  #    f.close()

        if fCURRENT_FASTRAN[WORK_DIR.index(iWORK_DIR)]:
           fastrandata[shot] = read_fastran_outputs(os.path.join(iWORK_DIR,CURRENT_FASTRAN))
           print(CGREEN + 'READING CURRENT_FASTRAN from %s ... PASSED' % iWORK_DIR + CEND)
         # for ikey in list(fastrandata[shot].keys()):
         #     print(ikey,npy.size(fastrandata[shot][ikey]['data']),npy.shape(fastrandata[shot][ikey]['data']))
          #if iWORK_DIR == WORK_DIR[0]:
          #    print(CRED + "FASTRAN DATA" + CEND)
          #    f = open("fastrandata.dat", "w")
          #    for ikey in sorted(list(fastrandata[shot].keys())):
          #        f.write("%s\t\t%s\n" % (ikey,str(npy.shape(fastrandata[shot][ikey]['data']))))
          #        if ikey in ['pnbe','pnbi','poh','prad','prfe','prfi']: 
          #            print(ikey,fastrandata[shot][ikey]['data'][-1])
          #        print(ikey,npy.shape(fastrandata[shot][ikey]['data']))
          #        if ikey in ['aminor','rmajor','delta','kappa']:
          #             print(ikey,fastrandata[shot][ikey]['data'])
          #    f.close()
        else:
           print(CRED + 'READING CURRENT_FASTRAN from %s ... FAILED' % iWORK_DIR + CEND)

        if addbeta and not summary:
            portdirs = glob(iWORK_DIR[:-13] + "/*")
            betan_ideal_wall   = 0.0
            betan_ideal_nowall = 0.0
            for iport in portdirs:
                if "fastran_betaiw_dcon_corsica" in iport:
                    xdconfpath = iport+"/xdcon.log"
                    loc_betan_ideal_nowall,loc_betan_ideal_wall = read_dcon(xdconfpath)
                    betan_ideal_wall   = max(betan_ideal_wall,  loc_betan_ideal_wall)
                    betan_ideal_nowall = max(betan_ideal_nowall,loc_betan_ideal_nowall)
                elif "fastran_betanw_dcon_corsica" in iport:
                    xdconfpath = iport+"/xdcon.log"
                    loc_betan_ideal_nowall,loc_betan_ideal_wall = read_dcon(xdconfpath)
                    betan_ideal_wall   = max(betan_ideal_wall,  loc_betan_ideal_wall)
                    betan_ideal_nowall = max(betan_ideal_nowall,loc_betan_ideal_nowall)
            plotparam['betan_ideal_wall'][shot]   = betan_ideal_wall
            plotparam['betan_ideal_nowall'][shot] = betan_ideal_nowall

   #shots = list(fastrandata.keys())
   #print(CRED + "STATE" + CEND)
   #for ikey in list(statedata[shots[0]].keys()):   print(ikey,npy.size(statedata[shots[-1]][ikey]))
   #print(CRED + "FASTRAN" + CEND)
   #for ikey in list(fastrandata[shots[0]].keys()): print(ikey,npy.size(fastrandata[shots[-1]][ikey]))

    shots = list(fastrandata.keys())
   #for ikey in list(fastrandata[shots[-1]].keys()): print(ikey)
    if any(fCURRENT_FASTRAN) or any(fCURRENT_STATE):
        if fCURRENT_FASTRAN[WORK_DIR.index(iWORK_DIR)] and fCURRENT_STATE[WORK_DIR.index(iWORK_DIR)]:
            fastranplot = plot_fastran_outputs(fastrandata,plotparam=plotparam,state=statedata)
            return fastrandata,statedata,fastranplot
        elif fCURRENT_FASTRAN[WORK_DIR.index(iWORK_DIR)]:
            fastranplot = plot_fastran_outputs(fastrandata,plotparam=plotparam)
            return fastrandata,fastranplot
        elif fCURRENT_STATE[WORK_DIR.index(iWORK_DIR)]:
            fastranplot = plot_state_outputs(statedata,plotparam=plotparam)
            return statedata,fastranplot

    return 0
    
def calc_dcon_betan(WORK_DIR):
    dconfpath = "/global/project/projectdirs/atom/users/ehab/DCONfiles"
    dconfexec = "/global/common/software/atom/cori/binaries/dcon/bin/caltrans"

    reportpath = os.path.abspath(".")+"/fastran_report/"
    if not os.path.isdir(reportpath):
       os.system('mkdir '+reportpath)

    dconpath = os.path.abspath(".")+"/fastran_report/DCON/"
    if not os.path.isdir(dconpath):
       os.system('mkdir '+dconpath)

    shots   = []
    shotref = ""
    CASE_ID = 0

    fCURRENT_BC      = [False for i in range(len(WORK_DIR))]
    fCURRENT_STATE   = [False for i in range(len(WORK_DIR))]
    fCURRENT_EQDSK   = [False for i in range(len(WORK_DIR))]
    fCURRENT_EQSTATE = [False for i in range(len(WORK_DIR))]
    fCURRENT_INSTATE = [False for i in range(len(WORK_DIR))]
    fCURRENT_FASTRAN = [False for i in range(len(WORK_DIR))]

    mainpath = os.path.abspath(".")

    for iWORK_DIR in WORK_DIR:
        iWORK_DIR_PATH = os.path.abspath(iWORK_DIR)
        WORK_FILES = glob('%s/work/plasma_state/*' % (iWORK_DIR_PATH))

        for FILE in WORK_FILES:
            FILENAME = FILE.replace('%s/work/plasma_state/' % (iWORK_DIR_PATH),'')
            if FILENAME[0] == 'b': CURRENT_BC      = FILENAME;  fCURRENT_BC[WORK_DIR.index(iWORK_DIR)]      =  True
            if FILENAME[0] == 's': CURRENT_STATE   = FILENAME;  fCURRENT_STATE[WORK_DIR.index(iWORK_DIR)]   =  True
            if FILENAME[0] == 'g': CURRENT_EQDSK   = FILENAME;  fCURRENT_EQDSK[WORK_DIR.index(iWORK_DIR)]   =  True
            if FILENAME[0] == 'k': CURRENT_EQSTATE = FILENAME;  fCURRENT_EQSTATE[WORK_DIR.index(iWORK_DIR)] =  True
            if FILENAME[0] == 'i': CURRENT_INSTATE = FILENAME;  fCURRENT_INSTATE[WORK_DIR.index(iWORK_DIR)] =  True
            if FILENAME[0] == 'f': CURRENT_FASTRAN = FILENAME;  fCURRENT_FASTRAN[WORK_DIR.index(iWORK_DIR)] =  True

        geqdskfpath = os.path.abspath("%s/work/plasma_state/%s" % (iWORK_DIR_PATH,CURRENT_EQDSK))

        if fCURRENT_FASTRAN[WORK_DIR.index(iWORK_DIR)]:
           SHOT_NUMBER, TIME_ID = CURRENT_FASTRAN[1:].split('.')
           shot = SHOT_NUMBER + '.' + TIME_ID
           if shot == shotref:
               CASE_ID += 1
           elif shotref == "":
               shotref = shot
               CASE_ID += 1
           elif shotref != "" and shot != shotref:
               shotref = shot
               CASE_ID = 1
           shot = SHOT_NUMBER + '.' + TIME_ID + '.%02d' % CASE_ID

           shotpath = mainpath+"/fastran_report/DCON/"+shot
           if not os.path.isdir(shotpath):
              os.system('mkdir '+shotpath)

           for ind in npy.arange(0.4,2.2,0.4):
               betaiwfname = "betaiw_%dp%d" % (int(ind),int(round((10*(ind-int(ind))))))
               betaiwpath = shotpath+"/"+betaiwfname
               if not os.path.isdir(betaiwpath):
                  os.system('mkdir '+betaiwpath)
               os.system('cp %s/%s.bas %s/.'         % (dconfpath,betaiwfname,betaiwpath))
               os.system('cp %s %s/eqdsk'            % (geqdskfpath,betaiwpath))
               os.chdir('%s'                         % (betaiwpath))
               scale_geqdsk_file(cur_eqdsk_file="eqdsk",R0_scale=1.7,B0_scale=2.0)
               print(CGREEN + "Finding Stability Factor for %s.bas in %s" % (betaiwfname,iWORK_DIR) + CEND)
               os.system("%s %s.bas >> xdcon.log"    % (dconfexec,betaiwfname))

           betanwpath = shotpath+"/"+"betanw"
           if not os.path.isdir(betanwpath):
              os.system('mkdir '+betanwpath)
           os.system('cp %s/betanw.bas %s/.'         % (dconfpath,betanwpath))
           os.system('cp %s %s/eqdsk'                % (geqdskfpath,betanwpath))
           os.chdir('%s'                             % (betanwpath))
           scale_geqdsk_file(cur_eqdsk_file="eqdsk",R0_scale=1.7,B0_scale=2.0)
           print(CGREEN + "Finding Stability Factor for betanw.bas" + CEND)
           os.system("%s betanw.bas >> xdcon.log"    % (dconfexec))

           os.chdir('%s'                         % (mainpath))
           shots.append(shot)

    return 1

def scale_geqdsk_file(cur_eqdsk_file,R0_scale,B0_scale):
    gfdata = efit_eqdsk.readg(cur_eqdsk_file)
    Rs = gfdata['rzero']/R0_scale
    Bs = gfdata['bcentr']/B0_scale
    gfdata = efit_eqdsk.scaleg(gfdata, R0=1.0/Rs, B0=1.0/Bs)
    cur_eqdsk_file = efit_eqdsk.writeg(gfdata,00000,0000)
    os.system("mv %s eqdsk" % cur_eqdsk_file)

    return 1


       
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--plot'   , '-plot',     action='store_const',const=1,help='Plot FASTRAN ouputs.')
   parser.add_argument('--state',   '-state',    action='store_const',const=1,help='Include STATE data in the summary and plots.')
   parser.add_argument('--summary', '-summary',  action='store_const',const=1,help='Plot FASTRAN output from Summary Folder.')
   parser.add_argument('--newplot', '-newplot',  action='store_const',const=1,help='Remove the old figures and plot new ones.')
   parser.add_argument('--figspec', '-figspec',  action='store_const',const=1,help='Create Figures Based on Specifications provided by the user.')
   parser.add_argument('--getbeta', '-getbeta',  action='store_const',const=1,help='Calculate the beta values using DCON stability code.')
   parser.add_argument('inputs',nargs='*')

   if parser.parse_args():
       args    = parser.parse_args()
       plot    = args.plot
       state   = args.state
       inputs  = args.inputs
       newplot = args.newplot
       figspec = args.figspec
       summary = args.summary
       getbeta = args.getbeta

   plotparam = {}
   if newplot: plotparam['newplot'] = True
   if figspec: plotparam['figspec'] = True

   if inputs == []:
       print(CRED + 'SIMULATION FOLDER(S) NOT FOUND/PROVIDED ... EXIT!' + CEND)
       sys.exit()

   if   getbeta:
        dcon_betan = calc_dcon_betan(WORK_DIR=inputs)
   elif summary:
        fastrandata = read_fastran(WORK_DIR=inputs)
        returnvals  = fastran_summary(fastrandata)
        plotsreturn   = plot_fastran_outputs(fastrandata,plotparam=plotparam)
   elif plot or figspec:
        fastrandata = read_fastran(WORK_DIR=inputs)
        returnvals  = plot_fastran_outputs(fastrandata,plotparam=plotparam)

    
