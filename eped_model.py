import os
import sys
import math
import glob
import numpy as npy
import fastran_tools
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

colors = ['red','blue','green','orange','black','cyan','sienna','magenta','teal','olive','steelblue','deeppink','indigo','purple','maroon','lime']
colors.extend(['gold','lawngreen','sandybrown','cadetblue','deepskyblue'])
colors.extend(['#0000FF','#FF4040','#8A2BE2','#458B00','#E3CF57','#8EE5EE','#FF7F24','#FF1493','#97FFFF','#FFD700','#FF69B4','#27408B'])
colors.extend(['#FFF68F','#7A8B8B','#FFAEB9','#EE9572','#20B2AA','#B3EE3A','#FFBBFF','#8E8E38','#FF8247','#EE82EE','#FFFF00','#8B8B00'])

CEND    = '\033[0m'
CRED    = '\33[31m'
CBLUE   = '\33[34m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'

eped_folders  = []
eped_folders += sorted(glob.glob("FNSF_EPED_scan_NEPED_IP_??p??"))

IP_ORDER   = []
NPED_ORDER = []
for ifolder in eped_folders:
    eped_files = sorted(glob.glob("%s/SUMMARY/e*" % ifolder))
    for ifile in eped_files:
        epeddata = fastran_tools.read_eped_outputs(ifile)
        IP_ORDER.append(epeddata['ip']['data'][-1])
        NPED_ORDER.append(epeddata['neped']['data'][-1])

IP_ORDER   = sorted(list(set(IP_ORDER)))
NPED_ORDER = sorted(list(set(NPED_ORDER)))

NIP   = len(IP_ORDER)
NNPED = len(NPED_ORDER)

IP2D   = npy.zeros((NIP,NNPED),dtype=float); IP2D[:]   = None
NPED2D = npy.zeros((NIP,NNPED),dtype=float); NPED2D[:] = None
PPED2D = npy.zeros((NIP,NNPED),dtype=float); PPED2D[:] = None
WPED2D = npy.zeros((NIP,NNPED),dtype=float); WPED2D[:] = None
 
reportpath = os.path.abspath(".")+"/fastran_report/"
if not os.path.isdir(reportpath):
   os.system('mkdir '+reportpath)

figurepath = os.path.abspath(".")+"/fastran_report/Figures/"
if not os.path.isdir(figurepath):
   os.system('mkdir '+figurepath)

epedfigs = PdfPages(figurepath+'eped_fit_plots.pdf')

individual_flag = False
collective_flag = True

for ifolder in eped_folders:
    eped_files = sorted(glob.glob("%s/SUMMARY/e*" % ifolder))
    IP1D = []; NPED1D = []; PPED1D = []; WPED1D = []
    for ifile in eped_files:
        epeddata = fastran_tools.read_eped_outputs(ifile)
       #crnt_index = IP_ORDER.index(epeddata['ip']['data'][-1])
        crnt_index = NPED_ORDER.index(epeddata['neped']['data'][-1])
        if epeddata['p_E1']['data'][-1] <= 0.0 or epeddata['wid_E1']['data'][-1] <= 0.0: continue
        if epeddata['neped']['data'][-1] < 10.5: continue
        IP2D[eped_folders.index(ifolder),  crnt_index] = epeddata['ip']['data'][-1]
        NPED2D[eped_folders.index(ifolder),crnt_index] = epeddata['neped']['data'][-1]
        PPED2D[eped_folders.index(ifolder),crnt_index] = epeddata['p_E1']['data'][-1]
        WPED2D[eped_folders.index(ifolder),crnt_index] = epeddata['wid_E1']['data'][-1]

eped_fit_model = lambda X, A, B, C: A * X[0]**B * X[1]**C 
IP11D   = [x for x in IP2D.flatten()   if math.isnan(x) == False]
NPED11D = [x for x in NPED2D.flatten() if math.isnan(x) == False]
PPED11D = [x for x in PPED2D.flatten() if math.isnan(x) == False]
WPED11D = [x for x in WPED2D.flatten() if math.isnan(x) == False]
(PPED_A1, PPED_B1, PPED_C1), pcov = curve_fit(eped_fit_model, (NPED11D,IP11D), PPED11D)
(WPED_A1, WPED_B1, WPED_C1), pcov = curve_fit(eped_fit_model, (NPED11D,IP11D), WPED11D)

if collective_flag:
    fig = plt.figure("EPED Model",dpi=200)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

[nrow,ncol] = npy.shape(IP2D)

irow = 0
for irow in range(nrow):
    IP1_TEST   = [x for x in IP2D[irow,:]               if math.isnan(x) == False]
    IP2_TEST   = [x for x in IP2D.transpose()[irow,:]   if math.isnan(x) == False]
    NPED1_TEST = [x for x in NPED2D[irow,:]             if math.isnan(x) == False]
    NPED2_TEST = [x for x in NPED2D.transpose()[irow,:] if math.isnan(x) == False]
    if IP1_TEST and IP2_TEST and NPED1_TEST and NPED2_TEST:
        if individual_flag:
            fig = plt.figure("EPED Model_%04d" % irow ,dpi=200)
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

        if all(IP2_TEST == IP2_TEST[0]):
            IP   = IP2D.transpose()[irow,:]
            NPED = NPED2D.transpose()[irow,:]
            PPED = PPED2D.transpose()[irow,:]
            WPED = WPED2D.transpose()[irow,:]

            PPED_Model = PPED_A1*NPED**PPED_B1*IP**PPED_C1
            PPED_exind = abs(PPED - PPED_Model) > 0.1
            ax2.plot(NPED,1.0e3*PPED,color=colors[irow],marker='o',linestyle="")
            ax2.plot(NPED,1.0e3*PPED_A1*NPED**PPED_B1*IP**PPED_C1,color=colors[irow])
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            ax2.set_ylabel("$P_{ped}$",fontsize=12)
            ax2.set_title("$W_{ped}$ = %3.2f$N_{eped}^{%3.2f}$$I_P^{%3.2f}$" % (WPED_A1,WPED_B1,WPED_C1),fontsize=14)

            WPED_Model = WPED_A1*NPED**WPED_B1*IP**WPED_C1
            WPED_exind = abs(WPED - WPED_Model) > 0.1
            ax4.plot(NPED,WPED,color=colors[irow],marker='o',linestyle="")
            ax4.plot(NPED,WPED_A1*NPED**WPED_B1*IP**WPED_C1,color=colors[irow])
            ax4.yaxis.tick_right()
            ax4.yaxis.set_label_position("right")
            ax4.set_ylabel("$W_{ped}$",fontsize=12)
            ax4.set_xlabel("$N_{PED}$",fontsize=12)

            IP   = IP2D[irow,:]
            NPED = NPED2D[irow,:]
            PPED = PPED2D[irow,:]
            WPED = WPED2D[irow,:]

            ax1.plot(IP,1.0e3*PPED,color=colors[irow],marker='o',linestyle="")
            ax1.plot(IP,1.0e3*PPED_A1*NPED**PPED_B1*IP**PPED_C1,color=colors[irow])
            ax1.set_ylabel("$P_{ped}$",fontsize=12)
            ax1.set_title("$P_{ped}$ = %3.2f$N_{eped}^{%3.2f}$$I_P^{%3.2f}$" % (1.0e3*PPED_A1,PPED_B1,PPED_C1),fontsize=14)

            ax3.plot(IP,WPED,color=colors[irow],marker='o',linestyle="")
            ax3.plot(IP,WPED_A1*NPED**WPED_B1*IP**WPED_C1,color=colors[irow])
            ax3.set_ylabel("$W_{ped}$",fontsize=12)
            ax3.set_xlabel("$I_{P}$",fontsize=12)

        elif all(NPED2_TEST == NPED2_TEST[0]):
            IP   = IP2D[irow,:]
            NPED = NPED2D[irow,:]
            PPED = PPED2D[irow,:]
            WPED = WPED2D[irow,:]

            PPED_Model = PPED_A1*NPED**PPED_B1*IP**PPED_C1
            PPED_exind = abs(PPED - PPED_Model) < 0.008
            ax2.plot(NPED[PPED_exind],1.0e3*PPED[PPED_exind],color=colors[irow],marker='o',linestyle="")
            ax2.plot(NPED,1.0e3*PPED_A1*NPED**PPED_B1*IP**PPED_C1,color=colors[irow])
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            ax2.set_ylabel("$P_{ped}$ (kPa)",fontsize=12)
            ax2.set_title("$W_{ped}$ = %3.2f$N_{eped}^{%3.2f}$$I_P^{%3.2f}$" % (WPED_A1,WPED_B1,WPED_C1),fontsize=14)

            WPED_Model = WPED_A1*NPED**WPED_B1*IP**WPED_C1
            WPED_exind = abs(WPED - WPED_Model) < 0.0010
            ax4.plot(NPED[WPED_exind],WPED[WPED_exind],color=colors[irow],marker='o',linestyle="")
            ax4.plot(NPED,WPED_A1*NPED**WPED_B1*IP**WPED_C1,color=colors[irow])
            ax4.yaxis.tick_right()
            ax4.yaxis.set_label_position("right")
            ax4.set_ylabel("$W_{ped}$",fontsize=12)
            ax4.set_xlabel("$N_{PED}$",fontsize=12)

            IP   = IP2D.transpose()[irow,:]
            NPED = NPED2D.transpose()[irow,:]
            PPED = PPED2D.transpose()[irow,:]
            WPED = WPED2D.transpose()[irow,:]

            PPED_Model = PPED_A1*NPED**PPED_B1*IP**PPED_C1
            PPED_exind = abs(PPED - PPED_Model) < 0.008
            ax1.plot(IP[PPED_exind],1.0e3*PPED[PPED_exind],color=colors[irow],marker='o',linestyle="")
            ax1.plot(IP,1.0e3*PPED_A1*NPED**PPED_B1*IP**PPED_C1,color=colors[irow])
            ax1.set_ylabel("$P_{ped}$ (kPa)",fontsize=12)
            ax1.set_title("$P_{ped}$ = %3.2f$N_{eped}^{%3.2f}$$I_P^{%3.2f}$" % (1.0e3*PPED_A1,PPED_B1,PPED_C1),fontsize=14)

            WPED_Model = WPED_A1*NPED**WPED_B1*IP**WPED_C1
            WPED_exind = abs(WPED - WPED_Model) < 0.0025
            ax3.plot(IP[WPED_exind],WPED[WPED_exind],color=colors[irow],marker='o',linestyle="")
            ax3.plot(IP,WPED_A1*NPED**WPED_B1*IP**WPED_C1,color=colors[irow])
            ax3.set_ylabel("$W_{ped}$",fontsize=12)
            ax3.set_xlabel("$I_{P}$ (MA)",fontsize=12)

        if individual_flag:
            fig.suptitle("EPED Fitting Model",fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.subplots_adjust(wspace=0,hspace=0)
            
            epedfigs.savefig(fig)
            plt.close(fig)

if collective_flag:
    fig.suptitle("EPED Fitting Model",fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(wspace=0,hspace=0)
    
    epedfigs.savefig(fig)
    plt.close(fig)

epedfigs.close()
