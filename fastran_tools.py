import os
import re
import sys
import json
import random
import argparse
import efit_eqdsk
import subprocess

import numpy               as npy
import matplotlib.pyplot   as plt
import matplotlib.colors   as clr
import matplotlib.gridspec as gds

from glob                            import glob
from netCDF4                         import Dataset
from scipy.optimize                  import curve_fit
from scipy.interpolate               import CubicSpline
from matplotlib.lines                import Line2D
from matplotlib.backends.backend_pdf import PdfPages

from Namelist import Namelist

CEND    = '\033[0m'
CRED    = '\33[31m'
CBLUE   = '\33[34m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'

#colors = []
#ncolors = 150; icolors = 0
#while True:
#    rand_color = "#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
#    if rand_color not in colors: colors.append(rand_color); icolors += 1
#    if icolors > ncolors: break

colors = ['red','blue','green','orange','black','cyan','sienna','magenta','teal','olive','steelblue','deeppink','indigo','purple','maroon','lime']
colors.extend(['gold','lawngreen','sandybrown','cadetblue','deepskyblue'])
colors.extend(['#0000FF','#FF4040','#8A2BE2','#458B00','#E3CF57','#8EE5EE','#FF7F24','#FF1493','#97FFFF','#FFD700','#FF69B4','#27408B'])
colors.extend(['#FFF68F','#7A8B8B','#FFAEB9','#EE9572','#20B2AA','#B3EE3A','#FFBBFF','#8E8E38','#FF8247','#EE82EE','#FFFF00','#8B8B00'])

styles = [(0,()),(0,(5,1)),(0,(1,1)),(0,(3,1,1,1)),(0,(3,1,1,1,1,1)),(0,(5,5)),(0,(5,10)),(0,(1,10))]
styles.extend([(0,(3,10,1,10)),(0,(3,5,1,5)),(0,(3,5,1,5,1,5)),(0,(3,10,1,10,1,10)),(0,(3,1,1,1,1,1))])

sublabels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']


def listndims(mylist):
    maxdims = 0
    if type(mylist) in [list,tuple]: maxdims += 1
    for i in mylist:
        if type(i)  in [list,tuple]: maxdims += 1; break
    return maxdims


def read_dcon(fn_log):
    for line in open(fn_log,"r").readlines():
        if re.compile("\s*betaN_ideal-nowall").search(line):
            betan_ideal_nowall = float(line.split(":")[-1])
        if re.compile("\s*betaN_ideal-wall").search(line):
            betan_ideal_wall = float(line.split(":")[-1])
    if "betan_ideal_nowall" not in locals(): betan_ideal_nowall = 0.0
    if "betan_ideal_wall"   not in locals(): betan_ideal_wall = 0.0
    return betan_ideal_nowall,betan_ideal_wall

def dcon_bas_file_data(iwall = 20.0):
    bas_text  = ''
    bas_text += 'read "d3.bas"\n'
    bas_text += 'read "sewall.bas"\n'
    bas_text += 'character*128 eqdsk\n'
    bas_text += 'character*32 fileid\n'
    bas_text += 'character*32 inv_save_file\n'
    bas_text += 'eqdsk = "eqdsk"\n'
    bas_text += 'd3(eqdsk, 0, 0.995, 1, 360, 0.001)\n'
    bas_text += 'residj\n'
    bas_text += 'fileid = trim(shotName) // "_" // format(shotTime*1e3, -4, 0, 1)\n'
    bas_text += 'inv_save_file = trim(fileid) // "_inv.sav"\n'
    bas_text += 'restore ^inv_save_file\n'
    bas_text += 'sewall\n'
    bas_text += 'real pmult_stable = 1.e-4\n'
    bas_text += 'real pmult_unstable = 1.e4\n'
    bas_text += 'real pstep_min = 0.9\n'
    bas_text += 'real pstep_max = 1.4\n'
    bas_text += 'real pstep\n'
    bas_text += 'real pmult = 1.0\n'
    bas_text += 'real pmult_k(100), betan_k(100), li_k(100), wtot_k(100)\n'
    bas_text += 'integer k = 0\n'
    bas_text += 'logical stable\n'
    bas_text += 'real psave0 = psave\n'
    bas_text += 'real jtsrf0 = jtsrf\n'
    bas_text += 'real qsrf0 = qsrf\n'
    bas_text += 'character*32 file_check\n'
    bas_text += 'dcn.nn = 1\n'
    bas_text += 'dcn.ishape = 6\n'
    bas_text += 'dcn.a = %5.3f\n' % iwall
    bas_text += 'dcn.delta_mlow = 6\n'
    bas_text += 'dcn.delta_mhigh = 27\n'
    bas_text += 'while (pmult_unstable/pmult_stable > 1.05)\n'
    bas_text += '    k = k + 1\n'
    bas_text += '    psave = pmult*psave0\n'
    bas_text += '    teq_inv\n'
    bas_text += '    call dcon\n'
    bas_text += '    stable = (et(1) > 0)\n'
    bas_text += '    betan_k(k) = ctroy\n'
    bas_text += '    li_k(k)    = li(1)\n'
    bas_text += '    wtot_k(k)  = et(1)\n'
    bas_text += '    pmult_k(k) = pmult\n'
    bas_text += '    if (stable) then\n'
    bas_text += '      # Increase pressure\n'
    bas_text += '      pstep = min(1+0.5*( pmult_unstable - pmult)/pmult, pstep_max)\n'
    bas_text += '      pmult_stable = pmult\n'
    bas_text += '    else\n'
    bas_text += '      # Decrease pressure\n'
    bas_text += '      pstep = max(1+ 0.5*( pmult_stable - pmult)/pmult, pstep_min)\n'
    bas_text += '      pmult_unstable = pmult\n'
    bas_text += '    endif\n'
    bas_text += '    pmult = pmult*pstep\n'
    bas_text += '    remark "Next pressure multiplier"\n'
    bas_text += '    k\n'
    bas_text += '    pmult\n'
    bas_text += '  endwhile\n'
    bas_text += '  remark "betaN_ideal-nowall: " // format(betan_k(k),5, 2, 1)\n'
    bas_text += '  quit(0)\n'

    return bas_text

def calc_dcon_betan(WORK_DIR):
    dconfpath = "/global/project/projectdirs/atom/users/ehab/DCONfiles"
    dconfexec = os.path.abspath("/global/common/software/atom/cori/binaries/dcon/bin/caltrans")

    os.environ["CPU"]               = "LINUX"
    os.environ["DCON"]              = "/global/common/software/atom/cori/binaries/dcon/dcon"
    os.environ["GCC_PATH"]          = "/opt/gcc/11.2.0"
    os.environ["NCARG_ROOT"]        = "/global/common/cori/software/ncl/6.5.0"
    os.environ["CORSICA_PFB"]       = "/global/common/software/atom/cori/binaries/dcon/scripts"
    os.environ["CORSICA_ROOT"]      = "/global/common/software/atom/cori/binaries/dcon"
    os.environ["CORSICA_SCRIPTS"]   = "/global/common/software/atom/cori/binaries/dcon/scripts"
    os.environ["CALTRANS_BIN_DIR"]  = "/global/common/software/atom/cori/binaries/dcon/bin"
    os.environ["CALTRANS_BIN_NAME"] = "caltrans"

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
    CASE_ID = max(CASE_ID,len(glob(mainpath+"/fastran_report/DCON/*")))

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
           if os.path.isfile(shotpath+"/dcon.dat"):
               fcollect = open(shotpath+"/dcon.dat", "a")
           else:
               fcollect = open(shotpath+"/dcon.dat", "x")

           for ind in npy.arange(0.4,2.2,0.4):
               betaiwfname = "betaiw_%dp%d" % (int(ind),int(round((10*(ind-int(ind))))))
               betaiwpath = shotpath+"/"+betaiwfname
               if not os.path.isdir(betaiwpath):
                  os.system('mkdir '+betaiwpath)
               fhandle = open('%s/%s.bas'            % (betaiwpath,betaiwfname), "w")
               fhandle.write(dcon_bas_file_data(iwall = ind))
               fhandle.close()
               os.system('cp %s %s/eqdsk'            % (geqdskfpath,betaiwpath))
               os.chdir('%s'                         % (betaiwpath))
               scale_geqdsk_file(cur_eqdsk_file="eqdsk",R0_scale=1.7,B0_scale=2.0)
               print(CGREEN + "Finding Stability Factor for %s.bas in %s" % (betaiwfname,iWORK_DIR) + CEND)
               os.system("%s %s.bas >> xdcon.log"    % (dconfexec,betaiwfname))
               fbetanw,fbetaiw = read_dcon("xdcon.log")
               fcollect.write("betan_iwall (d = %5.3fa) = %5.3f\n" % (ind,max(fbetanw,fbetaiw)))

           betanwpath = shotpath+"/"+"betanw"
           if not os.path.isdir(betanwpath):
              os.system('mkdir '+betanwpath)
           fhandle = open('%s/betanw.bas'            % (betanwpath), "w")
           fhandle.write(dcon_bas_file_data(iwall = 20.0))
           fhandle.close()
           os.system('cp %s %s/eqdsk'                % (geqdskfpath,betanwpath))
           os.chdir('%s'                             % (betanwpath))
           scale_geqdsk_file(cur_eqdsk_file="eqdsk",R0_scale=1.7,B0_scale=2.0)
           print(CGREEN + "Finding Stability Factor for betanw.bas" + CEND)
           os.system("%s betanw.bas >> xdcon.log"    % (dconfexec))
           fbetanw,fbetaiw = read_dcon("xdcon.log")
           fcollect.write("betan_nwall (d = %5.3fa) = %5.3f\n" % (ind,fbetanw))

           fcollect.close()

           os.chdir('%s'                             % (mainpath))
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

def read_geqdsk_outputs(fpath):
   #if os.path.isfile(fpath):
   #    print(CGREEN + "FINDING EPED OUTPUT AT %s: PASSED" % (fpath) + CEND)
   #else:
    
   #Developed by Ehab Hassan on 2019-02-27
    if os.path.isfile(fpath) == False:
       errorFunc = traceback.extract_stack(limit=2)[-2][3]
       errorLine = traceback.extract_stack(limit=2)[-2][1]
       errorFile = traceback.extract_stack(limit=2)[-2][2]
       errMSG    = 'Call %s line %5d in file %s Failed.\n'
       errMSG   += 'Fatal: file %s not found.'
       raise IOError(errMSG %(errorFunc,errorLine,errorFile,fpath))

    ofh = open(fpath,'r')
    geqdsk = {}
    geqdsk['idum']   = {} 
    geqdsk['RDIM']   = {} 
    geqdsk['ZDIM']   = {} 
    geqdsk['RLEN']   = {} 
    geqdsk['ZLEN']   = {} 
    geqdsk['RCTR']   = {} 
    geqdsk['RLFT']   = {} 
    geqdsk['ZMID']   = {} 
    geqdsk['RMAX']   = {} 
    geqdsk['ZMAX']   = {} 
    geqdsk['PSIMAX'] = {} 
    geqdsk['PSIBND'] = {} 
    geqdsk['BCTR']   = {} 
    geqdsk['CURNT']  = {} 
    geqdsk['PSIMAX'] = {} 
    geqdsk['XDUM']   = {} 
    geqdsk['RMAX']   = {} 
    geqdsk['XDUM']   = {} 
    geqdsk['ZMAX']   = {} 
    geqdsk['XDUM']   = {} 
    geqdsk['PSIBND'] = {} 
    geqdsk['XDUM']   = {} 
    geqdsk['XDUM']   = {} 

    cline = ofh.readline()
    geqdsk['idum']['data']   = int(cline[48:52])
    geqdsk['RDIM']['data']   = int(cline[52:56])
    geqdsk['ZDIM']['data']   = int(cline[56:61])
    cline = ofh.readline()
    geqdsk['RLEN']['data']   = float(cline[0:16])
    geqdsk['ZLEN']['data']   = float(cline[16:32])
    geqdsk['RCTR']['data']   = float(cline[32:48])
    geqdsk['RLFT']['data']   = float(cline[48:64])
    geqdsk['ZMID']['data']   = float(cline[64:80])
    cline = ofh.readline()
    geqdsk['RMAX']['data']   = float(cline[0:16])
    geqdsk['ZMAX']['data']   = float(cline[16:32])
    geqdsk['PSIMAX']['data'] = float(cline[32:48])
    geqdsk['PSIBND']['data'] = float(cline[48:64])
    geqdsk['BCTR']['data']   = float(cline[64:80])
    cline = ofh.readline()
    geqdsk['CURNT']['data']  = float(cline[0:16])
    geqdsk['PSIMAX']['data'] = float(cline[16:32])
    geqdsk['XDUM']['data']   = float(cline[32:48])
    geqdsk['RMAX']['data']   = float(cline[48:64])
    geqdsk['XDUM']['data']   = float(cline[64:80])
    cline = ofh.readline()
    geqdsk['ZMAX']['data']   = float(cline[0:16])
    geqdsk['XDUM']['data']   = float(cline[16:32])
    geqdsk['PSIBND']['data'] = float(cline[32:48])
    geqdsk['XDUM']['data']   = float(cline[48:64])
    geqdsk['XDUM']['data']   = float(cline[64:80])

    nlines1D = int(npy.ceil(geqdsk['RDIM']['data']/5.0))

    geqdsk['fpol'] = {}
    geqdsk['fpol']['data'] = npy.zeros(geqdsk['RDIM']['data'])
    for iline in range(nlines1D):
        cline = ofh.readline()
        try:
            geqdsk['fpol']['data'][iline*5+0] = float(cline[0:16])
            geqdsk['fpol']['data'][iline*5+1] = float(cline[16:32])
            geqdsk['fpol']['data'][iline*5+2] = float(cline[32:48])
            geqdsk['fpol']['data'][iline*5+3] = float(cline[48:64])
            geqdsk['fpol']['data'][iline*5+4] = float(cline[64:80])
        except:
            error = 'empty records'

    geqdsk['pressure'] = {}
    geqdsk['pressure']['data'] = npy.zeros(geqdsk['RDIM']['data'])
    for iline in range(nlines1D):
        cline = ofh.readline()
        try:
            geqdsk['pressure']['data'][iline*5+0] = float(cline[0:16])
            geqdsk['pressure']['data'][iline*5+1] = float(cline[16:32])
            geqdsk['pressure']['data'][iline*5+2] = float(cline[32:48])
            geqdsk['pressure']['data'][iline*5+3] = float(cline[48:64])
            geqdsk['pressure']['data'][iline*5+4] = float(cline[64:80])
        except:
            error = 'empty records'
   
    geqdsk['ffprime'] = {}
    geqdsk['ffprime']['data'] = npy.zeros(geqdsk['RDIM']['data'])
    for iline in range(nlines1D):
        cline = ofh.readline()
        try:
            geqdsk['ffprime']['data'][iline*5+0] = float(cline[0:16])
            geqdsk['ffprime']['data'][iline*5+1] = float(cline[16:32])
            geqdsk['ffprime']['data'][iline*5+2] = float(cline[32:48])
            geqdsk['ffprime']['data'][iline*5+3] = float(cline[48:64])
            geqdsk['ffprime']['data'][iline*5+4] = float(cline[64:80])
        except:
            error = 'empty records'

    geqdsk['pprime'] = {}
    geqdsk['pprime']['data'] = npy.zeros(geqdsk['RDIM']['data'])
    for iline in range(nlines1D):
        cline = ofh.readline()
        try:
            geqdsk['pprime']['data'][iline*5+0] = float(cline[0:16])
            geqdsk['pprime']['data'][iline*5+1] = float(cline[16:32])
            geqdsk['pprime']['data'][iline*5+2] = float(cline[32:48])
            geqdsk['pprime']['data'][iline*5+3] = float(cline[48:64])
            geqdsk['pprime']['data'][iline*5+4] = float(cline[64:80])
        except:
            error = 'empty records'

    nlines2D = int(npy.ceil(geqdsk['RDIM']['data']*geqdsk['ZDIM']['data']/5.0))

    geqdsk['psiRZ'] = {}
    geqdsk['psiRZ']['data'] = npy.zeros(geqdsk['RDIM']['data']*geqdsk['ZDIM']['data'])
    for iline in range(nlines2D):
        cline = ofh.readline()
        try:
            geqdsk['psiRZ']['data'][iline*5+0] = float(cline[0:16])
            geqdsk['psiRZ']['data'][iline*5+1] = float(cline[16:32])
            geqdsk['psiRZ']['data'][iline*5+2] = float(cline[32:48])
            geqdsk['psiRZ']['data'][iline*5+3] = float(cline[48:64])
            geqdsk['psiRZ']['data'][iline*5+4] = float(cline[64:80])
        except:
            error = 'empty records'
    geqdsk['psiRZ']['data'] = npy.reshape(geqdsk['psiRZ']['data'],(geqdsk['ZDIM']['data'],geqdsk['RDIM']['data']))

    geqdsk['qpsi'] = {}
    geqdsk['qpsi']['data'] = npy.zeros(geqdsk['RDIM']['data'])
    for iline in range(nlines1D):
        cline = ofh.readline()
        try:
            geqdsk['qpsi']['data'][iline*5+0] = float(cline[0:16])
            geqdsk['qpsi']['data'][iline*5+1] = float(cline[16:32])
            geqdsk['qpsi']['data'][iline*5+2] = float(cline[32:48])
            geqdsk['qpsi']['data'][iline*5+3] = float(cline[48:64])
            geqdsk['qpsi']['data'][iline*5+4] = float(cline[64:80])
        except:
            error = 'empty records'

    geqdsk['nbound'] = {} 
    geqdsk['rbound'] = {} 
    geqdsk['zbound'] = {} 

    geqdsk['nlimit'] = {} 
    geqdsk['rlimit'] = {} 
    geqdsk['zlimit'] = {} 

    cline = ofh.readline()
    geqdsk['nbound']['data'] = int(cline[0:5])
    geqdsk['nlimit']['data'] = int(cline[5:10])

    if geqdsk['nbound']['data'] > 0:
       nlines1D = int(npy.ceil(2*geqdsk['nbound']['data']/5.0))

       Ary1D = npy.zeros(2*geqdsk['nbound']['data'])
       for iline in range(nlines1D):
           cline = ofh.readline()
           try:
               Ary1D[iline*5+0] = float(cline[0:16])
               Ary1D[iline*5+1] = float(cline[16:32])
               Ary1D[iline*5+2] = float(cline[32:48])
               Ary1D[iline*5+3] = float(cline[48:64])
               Ary1D[iline*5+4] = float(cline[64:80])
           except:
               error = 'empty records'

       geqdsk['rbound']['data'] = Ary1D[0::2]
       geqdsk['zbound']['data'] = Ary1D[1::2]


    if geqdsk['nlimit']['data'] > 0:
       nlines1D = int(npy.ceil(2*geqdsk['nlimit']['data']/5.0))

       Ary1D = npy.zeros(2*geqdsk['nlimit']['data'])
       for iline in range(nlines1D):
           cline = ofh.readline()
           try:
               Ary1D[iline*5+0] = float(cline[0:16])
               Ary1D[iline*5+1] = float(cline[16:32])
               Ary1D[iline*5+2] = float(cline[32:48])
               Ary1D[iline*5+3] = float(cline[48:64])
               Ary1D[iline*5+4] = float(cline[64:80])
           except:
               error = 'empty records'

       geqdsk['rlimit']['data'] = Ary1D[0::2]
       geqdsk['zlimit']['data'] = Ary1D[1::2]


    geqdsk['ZR1D']          = {}
    geqdsk['ZR1D']['data']  = npy.arange(geqdsk['ZDIM']['data'],dtype=float)*geqdsk['ZLEN']['data']/(geqdsk['ZDIM']['data']-1.0)
    geqdsk['ZR1D']['data'] += geqdsk['ZMID']['data']-geqdsk['ZMID']['data']/2.0

    geqdsk['RR1D']          = {}
    geqdsk['RR1D']['data']  = npy.arange(geqdsk['RDIM']['data'],dtype=float)*geqdsk['RLEN']['data']/(geqdsk['RDIM']['data']-1.0)
    geqdsk['RR1D']['data'] += geqdsk['RLFT']['data']

    geqdsk['psiRZ']['data'] = (geqdsk['psiRZ']['data']-geqdsk['PSIMAX']['data'])/(geqdsk['PSIBND']['data']-geqdsk['PSIMAX']['data'])

    geqdsk['PSI']            = {}
    geqdsk['PSI']['data']    = (geqdsk['PSIBND']['data']-geqdsk['PSIMAX']['data'])*npy.arange(geqdsk['RDIM']['data'])/(geqdsk['RDIM']['data']-1.0)
    geqdsk['PSIN']           = {}
    geqdsk['PSIN']['data']   = (geqdsk['PSI']['data']-geqdsk['PSI']['data'][0])/(geqdsk['PSI']['data'][-1]-geqdsk['PSI']['data'][0])
    geqdsk['rhopsi']         = {}
    geqdsk['rhopsi']['data'] = npy.sqrt(geqdsk['PSIN']['data'])

    extendPSI    = npy.linspace(geqdsk['PSI']['data'][0],geqdsk['PSI']['data'][-1],10*npy.size(geqdsk['PSI']['data']))
    extendPHI    = npy.empty_like(extendPSI)
    extendPHI[0] = 0.0
    qfunc        = CubicSpline(geqdsk['PSI']['data'],geqdsk['qpsi']['data'])
    for i in range(1,npy.size(extendPSI)):
        x           = extendPSI[:i+1]
        y           = qfunc(x)
        extendPHI[i]= npy.trapz(y,x)

    geqdsk['PHI']         = {}
    geqdsk['PHI']['data'] = npy.empty_like(geqdsk['PSI']['data'])
    phifunc       = CubicSpline(extendPSI,extendPHI)
    for i in range(npy.size(geqdsk['PSI']['data'])):
        geqdsk['PHI']['data'][i] = phifunc(geqdsk['PSI']['data'][i])

    geqdsk['PHIN']           = {}
    geqdsk['PHIN']['data']   = (geqdsk['PHI']['data']-geqdsk['PHI']['data'][0])/(geqdsk['PHI']['data'][-1]-geqdsk['PHI']['data'][0])
    geqdsk['rhotor']         = {}
    geqdsk['rhotor']['data'] = npy.sqrt(geqdsk['PHIN']['data'])

    return geqdsk

def read_geqdsk(WORK_DIR):
    geqdskdata = {}

    if type(WORK_DIR) == str: WORK_DIR = [WORK_DIR]

    CASE_ID = 0
    shotref = ""

    for iWORK_DIR in WORK_DIR:
        if os.path.isfile(iWORK_DIR):
            WORK_FILES = ['%s' % (iWORK_DIR)]
            iWORK_DIR_PATH = WORK_FILES[0].replace(os.path.basename(WORK_FILES[0]),"")
        else:
            WORK_FILES = []
        if not WORK_FILES:
            iWORK_DIR_PATH = '%s' % os.path.abspath(iWORK_DIR)
            WORK_FILES = glob('%s/g*' % (iWORK_DIR_PATH))
            if not WORK_FILES:
                iWORK_DIR_PATH = '%s/work/plasma_state' % os.path.abspath(iWORK_DIR)
                WORK_FILES = glob('%s/g*' % (iWORK_DIR_PATH))
        if WORK_FILES:
            for iWORK_FILES in WORK_FILES:
                GEQDSK_FILEPATH = iWORK_FILES
                GEQDSK_FILENAME = GEQDSK_FILEPATH.replace(iWORK_DIR_PATH+'/','')
                SHOT_NUMBER, TIME_ID = GEQDSK_FILENAME[1:].split('.')
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

                geqdskdata[shot] = read_geqdsk_outputs(GEQDSK_FILEPATH)
                if len(geqdskdata[shot]['pressure']['data']) == 0:
                    del geqdskdata[shot]
                    print(CRED   + 'READING GEQDSK FILE %s IN %s ... FAILED' % (os.path.basename(GEQDSK_FILENAME),iWORK_DIR_PATH) + CEND)
                else:
                    print(CGREEN + 'READING GEQDSK FILE %s IN %s ... PASSED' % (os.path.basename(GEQDSK_FILENAME),iWORK_DIR_PATH) + CEND)
        else:
           print(CRED   + 'READING GEQDSK FILE IN %s ... FAILED' % iWORK_DIR_PATH + CEND)

    return geqdskdata

def plot_geqdsk_outputs(geqdskdata, plotparam={}):
    sims  = list(geqdskdata.keys())
    nsims = len(sims)

    reportpath = os.path.abspath(".")+"/fastran_report/"
    if not os.path.isdir(reportpath):
       os.system('mkdir '+reportpath)

    figurepath = os.path.abspath(".")+"/fastran_report/Figures/"
    if not os.path.isdir(figurepath):
       os.system('mkdir '+figurepath)

    if newplot and glob("%sgeqdsk_*.png" % figurepath):
        os.system("rm %sgeqdsk_*.png" % figurepath)

    if 'savepng' in plotparam:  savepng = plotparam['savepng']
    else:                       savepng = True

    if 'figspec' in plotparam:  figspec = plotparam['figspec']
    else:                       figspec = False

    geqdskfigs = PdfPages(figurepath+'geqdsk_plots.pdf')

    for sim in sims:
        # PLOTTING EPED PROFILES
       #lcmsfig = plt.figure("GEQDSK LMS",dpi=200)
       #lcmsaxs = lcmsfig.add_subplot(111)
       #lcolor = colors[0]
       #lstyle = styles[0]
       #lcmsaxs.plot(geqdskdata[sim]['rbound']['data'][:],geqdskdata[sim]['zbound']['data'][:],color=lcolor,linestyle=lstyle)
       #lcmsaxs.set_xlabel("$r$")
       #lcmsaxs.set_ylabel("$z$")

       #lcmsfig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
       #lcmsfig.subplots_adjust(wspace=0,hspace=0)
       #geqdskfigs.savefig(lcmsfig)
       #if savepng: lcmsfig.savefig(figurepath+"geqdsk_lcms_%04d.png" % sims.index(sim))
       #plt.close(lcmsfig)

        fig = plt.figure("GEQDSK psiRZ",dpi=200)
        fig.suptitle('FNSF Equilibrium')
        grds = gds.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1])
        ax01 = fig.add_subplot(grds[0:,0:3])
        ax02 = fig.add_subplot(grds[0,3:])
        ax03 = fig.add_subplot(grds[1,3:])

        contours = ax01.contour(geqdskdata[sim]['RR1D']['data'][:],geqdskdata[sim]['ZR1D']['data'][:],geqdskdata[sim]['psiRZ']['data'][:],20,colors='black')
        ax01.clabel(contours, inline=True, fontsize=8)
        ax01.set_xlabel("$R (m)$")
        ax01.set_ylabel("$Z (m)$")

        contours = ax02.plot(geqdskdata[sim]['PSIN']['data'][:],geqdskdata[sim]['qpsi']['data'][:],color='black')
        ax02.set_xlabel("$\\rho$")
        ax02.set_ylabel("$q$")
        ax02.set_ylim([1,12])
        ax02.yaxis.tick_right()
        ax02.yaxis.set_label_position("right")

        contours = ax03.plot(geqdskdata[sim]['PSIN']['data'][:],geqdskdata[sim]['pressure']['data'][:]/1.0e6,color='black')
        ax03.set_xlabel("$\\rho$")
        ax03.set_ylabel("$Pressure$ (MPa)")
        ax03.set_ylim([0,1.5])
        ax03.yaxis.tick_right()
        ax03.yaxis.set_label_position("right")

        fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        fig.subplots_adjust(wspace=0,hspace=0)
        geqdskfigs.savefig(fig)
        if savepng: fig.savefig(figurepath+"geqdsk_basics_%04d.png" % sims.index(sim))
        plt.close(fig)


    geqdskfigs.close()
    return 1


       
def read_eped_outputs(fpath):
   #if os.path.isfile(fpath):
   #    print(CGREEN + "FINDING EPED OUTPUT AT %s: PASSED" % (fpath) + CEND)
   #else:
   #    print(CRED + "FINDING EPED OUTPUT AT %s: FAILED" % (fpath) + CEND)
    cdffh = Dataset(fpath, mode='r')

    eped = {}
    for name, variable in cdffh.variables.items():
        eped[name]                  = {}
        eped[name]['data']          = cdffh.variables[name][:]
        if hasattr(variable, "unit"):
            eped[name]['units']     = getattr(variable, "units")
        else:
            eped[name]['units']     = ""
        if hasattr(variable, "long_name"):
            eped[name]['long_name']     = getattr(variable, "long_name")
        else:
            eped[name]['long_name']     = ""

    return eped

def read_eped(WORK_DIR):
    epeddata = {}

    if type(WORK_DIR) == str: WORK_DIR = [WORK_DIR]

    CASE_ID = 0
    shotref = ""

    for iWORK_DIR in WORK_DIR:
        if os.path.isfile(iWORK_DIR):
            WORK_FILES = ['%s' % (iWORK_DIR)]
            iWORK_DIR_PATH = WORK_FILES[0].replace("/"+os.path.basename(WORK_FILES[0]),"")
        else:
            WORK_FILES = []
        if not WORK_FILES:
            iWORK_DIR_PATH = '%s' % os.path.abspath(iWORK_DIR)
            WORK_FILES = sorted(glob('%s/e*' % (iWORK_DIR_PATH)))
            if not WORK_FILES:
                iWORK_DIR_PATH = '%s/work/plasma_state' % os.path.abspath(iWORK_DIR)
                WORK_FILES = sorted(glob('%s/e*' % (iWORK_DIR_PATH)))
        if WORK_FILES:
            for iWORK_FILES in WORK_FILES:
                EPED_FILEPATH = iWORK_FILES
                EPED_FILENAME = EPED_FILEPATH.replace(iWORK_DIR_PATH+'/','')
                SHOT_NUMBER, TIME_ID = EPED_FILENAME[1:].split('.')
                shot = SHOT_NUMBER + '.' + TIME_ID
                if shot == shotref:
                    CASE_ID += 1
                elif shotref == "":
                    shotref = shot
                    CASE_ID += 1
                elif shotref != "" and shot != shotref:
                    shotref = shot
                    CASE_ID = 1
                shot = SHOT_NUMBER + '.' + TIME_ID + '.%02d.%02d.%02d' % (CASE_ID,WORK_DIR.index(iWORK_DIR),WORK_FILES.index(iWORK_FILES))

                epeddata[shot] = read_eped_outputs(EPED_FILEPATH)
                if len(epeddata[shot]['gamma']['data']) == 0:
                    del epeddata[shot]
                    print(CRED   + 'READING EPED FILE %s IN %s ... FAILED' % (os.path.basename(EPED_FILENAME),iWORK_DIR_PATH) + CEND)
                else:
                    print(CGREEN + 'READING EPED FILE %s IN %s ... PASSED' % (os.path.basename(EPED_FILENAME),iWORK_DIR_PATH) + CEND)
        else:
           print(CRED   + 'READING EPED FILE IN %s ... FAILED' % iWORK_DIR_PATH + CEND)

    return epeddata

#def eped_model(epeddata):
#    sims  = list(epeddata.keys())
#    nsims = len(sims)
#
#    IP   = []
#    IP1  = []
#    NPED = []
#    NPED1= []
#    PPED = []
#    WPED = []
#    for isim in sims:
#        if epeddata[isim]['p_E1']['data'][-1] <= 0.0 or epeddata[isim]['wid_E1']['data'][-1] <= 0.0: continue
#        print(epeddata[isim]['ip']['data'][-1],epeddata[isim]['neped']['data'][-1])
#        if epeddata[isim]['ip']['data'][-1] not in IP1:        IP1.append(epeddata[isim]['ip']['data'][-1])
#        if epeddata[isim]['neped']['data'][-1] not in NPED1: NPED1.append(epeddata[isim]['neped']['data'][-1]) 
#        IP.append(epeddata[isim]['ip']['data'][-1])
#        NPED.append(epeddata[isim]['neped']['data'][-1])
#        PPED.append(epeddata[isim]['p_E1']['data'][-1])
#        WPED.append(epeddata[isim]['wid_E1']['data'][-1])
#
#    IP   = npy.array(IP)
#    NPED = npy.array(NPED)
#    PPED = npy.array(PPED)
#    WPED = npy.array(WPED)
#
#    IP1D   = npy.linspace(min(IP),  max(IP),100)
#    NPED1D = npy.linspace(min(NPED),max(NPED),100)
#
#    eped_fit_model = lambda X, A, B, C: A * X[0]**B * X[1]**C 
#    (PPED_A, PPED_B, PPED_C), pcov = curve_fit(eped_fit_model, (NPED,IP), PPED)
#    (WPED_A, WPED_B, WPED_C), pcov = curve_fit(eped_fit_model, (NPED,IP), WPED)
#
#    fig = plt.figure("EPED Model",dpi=200)
#    ax1 = fig.add_subplot(221)
#    ax2 = fig.add_subplot(222)
#    ax3 = fig.add_subplot(223)
#    ax4 = fig.add_subplot(224)
#    ax1.plot(NPED,PPED,'ro')
#    ax1.plot(NPED1D,PPED_A*NPED1D**PPED_B*IP1D**PPED_C,'k')
#    ax2.plot(IP,PPED,'go')
#    ax2.plot(IP1D,  PPED_A*NPED1D**PPED_B*IP1D**PPED_C,'k')
#    ax3.plot(NPED,WPED,'b*')
#    ax3.plot(NPED1D,WPED_A*NPED1D**WPED_B*IP1D**WPED_C,'k')
#    ax4.plot(IP,WPED,'m*')
#    ax4.plot(IP1D,  WPED_A*NPED1D**WPED_B*IP1D**WPED_C,'k')
#    ax2.yaxis.tick_right()
#    ax2.yaxis.set_label_position("right")
#    ax4.yaxis.tick_right()
#    ax4.yaxis.set_label_position("right")
#    ax1.set_ylabel("$P_{ped}$",fontsize=12)
#    ax3.set_ylabel("$W_{ped}$",fontsize=12)
#    ax3.set_xlabel("$N_{EPED}$",fontsize=12)
#    ax2.set_ylabel("$P_{ped}$",fontsize=12)
#    ax4.set_ylabel("$W_{ped}$",fontsize=12)
#    ax4.set_xlabel("$I_{P}$",fontsize=12)
#    ax1.set_title("$P_{ped}$ = %3.2f$N_{eped}^{%3.2f}$$I_P^{%3.2f}$" % (PPED_A,PPED_B,PPED_C),fontsize=14)
#    ax2.set_title("$W_{ped}$ = %3.2f$N_{eped}^{%3.2f}$$I_P^{%3.2f}$" % (WPED_A,WPED_B,WPED_C),fontsize=14)
#    plt.suptitle("EPED Fitting Model",fontsize=16)
#    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#    fig.subplots_adjust(wspace=0,hspace=0)
#    plt.show()
#    plt.close(fig)
#
#    model = {}
#    model['PPED'] = (PPED_A, PPED_B, PPED_C)
#    model['WPED'] = (WPED_A, WPED_B, WPED_C)
#
#    return model


def plot_eped_outputs(epeddata, plotparam={}):
    sims  = list(epeddata.keys())
    nsims = len(sims)

    reportpath = os.path.abspath(".")+"/fastran_report/"
    if not os.path.isdir(reportpath):
       os.system('mkdir '+reportpath)

    figurepath = os.path.abspath(".")+"/fastran_report/Figures/"
    if not os.path.isdir(figurepath):
       os.system('mkdir '+figurepath)

    if newplot and glob("%seped_*.png" % figurepath):
        os.system("rm %seped_*.png" % figurepath)

    if 'savepng' in plotparam:  savepng = plotparam['savepng']
    else:                       savepng = True

    if 'figspec' in plotparam:  figspec = plotparam['figspec']
    else:                       figspec = False

    epedfigs = PdfPages(figurepath+'eped_plots.pdf')

    for sim in sims:
        print(CBLUE + "PLOTTING %s" % sim + CEND)
        modes = epeddata[sim]["nmodes"]['data']
        gamma = epeddata[sim]["gamma_PB"]['data']
        teped = epeddata[sim]["teped_list"]['data']
        betan = epeddata[sim]["eq_betanped"]['data']
        kEPED = epeddata[sim]["k_EPED"]['data'][:][0]
        nGr = epeddata[sim]["ip"]['data'][-1]/npy.pi/epeddata[sim]["a"]['data'][-1]**2
        FGW_PED_INPUT = epeddata[sim]['neped']['data'][-1]/10.0/nGr

        # PLOTTING EPED PROFILES
        k_EPED = epeddata[sim]['k_EPED']['data'][-1]
        rhok = [x for x in epeddata[sim]['profile_rho']['data'][k_EPED,:] if x != 0.0]
       #rhos = [x for x in epeddata[sim]['profile_rho']['data'][0,:] if x != 0.0]
       #rhoe = [x for x in epeddata[sim]['profile_rho']['data'][-1,:] if x != 0.0]

        proffig = plt.figure("EPED PROFILES",dpi=200)
        teaxs = proffig.add_subplot(321)
        lcolor = colors[0]
        lstyle = styles[0]
        prof = [x for x in epeddata[sim]['profile_Te']['data'][k_EPED,:]  if x != 0.0]
        teaxs.plot(rhok,prof,color=lcolor,linestyle=lstyle)
        teaxs.axvline(1.0-epeddata[sim]['wid_E1']['data'][-1],   ls="--",color="r")
        teaxs.axvline(1.0-epeddata[sim]['widtop_E1']['data'][-1],ls="--",color="b")
        teaxs.set_xticks([])
        teaxs.set_ylabel("$T_{e}$")
        teaxs.set_xlabel("$\\rho$")
        teaxs.set_xticks([])

        tiaxs = proffig.add_subplot(323)
        lcolor = colors[0]
        lstyle = styles[0]
        prof = [x for x in epeddata[sim]['profile_Ti']['data'][k_EPED,:]  if x != 0.0]
        tiaxs.plot(rhok,prof,color=lcolor,linestyle=lstyle)
        tiaxs.axvline(1.0-epeddata[sim]['wid_E1']['data'][-1],   ls="--",color="r")
        tiaxs.axvline(1.0-epeddata[sim]['widtop_E1']['data'][-1],ls="--",color="b")
        tiaxs.set_xticks([])
        tiaxs.set_ylabel("$T_{i}$")
        tiaxs.set_xlabel("$\\rho$")
        teaxs.set_xticks([])

        neaxs = proffig.add_subplot(325)
        lcolor = colors[0]
        lstyle = styles[0]
        prof = [x for x in epeddata[sim]['profile_ne']['data'][k_EPED,:]  if x != 0.0]
        neaxs.plot(rhok,prof,color=lcolor,linestyle=lstyle)
        neaxs.axvline(1.0-epeddata[sim]['wid_E1']['data'][-1],   ls="--",color="r")
        neaxs.axvline(1.0-epeddata[sim]['widtop_E1']['data'][-1],ls="--",color="b")
        neaxs.set_ylabel("$N_e$")
        neaxs.set_xlabel("$\\rho_{PED}$")

        paxs = proffig.add_subplot(322)
        lcolor = colors[0]
        lstyle = styles[0]
        prof = [x for x in epeddata[sim]['profile_ptot']['data'][k_EPED,:]  if x != 0.0]
        paxs.plot(rhok,prof,color=lcolor,linestyle=lstyle)
        paxs.axvline(1.0-epeddata[sim]['wid_E1']['data'][-1],   ls="--",color="r")
        paxs.axvline(1.0-epeddata[sim]['widtop_E1']['data'][-1],ls="--",color="b")
        paxs.set_ylabel("$P_{TOT}$")
        paxs.set_xlabel("$\\rho_{PED}$")
        paxs.yaxis.tick_right()
        paxs.yaxis.set_label_position("right")
        paxs.set_xticks([])

        jaxs = proffig.add_subplot(324)
        lcolor = colors[0]
        lstyle = styles[0]
        prof = [x for x in epeddata[sim]['profile_jtot']['data'][k_EPED,:]  if x != 0.0]
        jaxs.plot(rhok,prof,color=lcolor,linestyle=lstyle)
        jaxs.axvline(1.0-epeddata[sim]['wid_E1']['data'][-1],   ls="--",color="r")
        jaxs.axvline(1.0-epeddata[sim]['widtop_E1']['data'][-1],ls="--",color="b")
        jaxs.set_ylabel("$J_{TOT}$")
        jaxs.set_xlabel("$\\rho$")
        jaxs.yaxis.tick_right()
        jaxs.yaxis.set_label_position("right")
        jaxs.set_xticks([])

        qaxs = proffig.add_subplot(326)
        lcolor = colors[0]
        lstyle = styles[0]
        prof = [x for x in epeddata[sim]['profile_q']['data'][k_EPED,:]  if x != 0.0]
        qaxs.plot(rhok,prof,color=lcolor,linestyle=lstyle)
        qaxs.axvline(1.0-epeddata[sim]['wid_E1']['data'][-1],   ls="--",color="r")
        qaxs.axvline(1.0-epeddata[sim]['widtop_E1']['data'][-1],ls="--",color="b")
        qaxs.set_ylabel("$q$")
        qaxs.set_xlabel("$\\rho$")
        qaxs.yaxis.tick_right()
        qaxs.yaxis.set_label_position("right")

        title_txt_01 = "$I_P$ = %5.3f, $N_{e,ped}$ = %5.3f, $\\beta_n$ = %5.3f, $B_{T}$ = %5.3f" \
                     % (epeddata[sim]["ip"]['data'], epeddata[sim]["neped"]['data'], epeddata[sim]["betan"]['data'], epeddata[sim]["bt"]['data'])
        title_txt_02 = "$\\kappa$ = %5.3f, $\\delta$ = %5.3f, $R$ = %5.3f, $a$ = %5.3f" \
                     % (epeddata[sim]["kappa"]['data'], epeddata[sim]["delta"]['data'], epeddata[sim]["r"]['data'], epeddata[sim]["a"]['data'])
        proffig.suptitle("EPED PROFILES\n%s\n%s" % (title_txt_01,title_txt_02))
        proffig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        proffig.subplots_adjust(wspace=0,hspace=0)
        epedfigs.savefig(proffig)
        if savepng: proffig.savefig(figurepath+"eped_tpped_%04d.png" % sims.index(sim))
        plt.close(proffig)


        gamma_all = npy.array([ max(max(igamma),1.0e-30) for igamma in gamma ])
        k_start = 0
        k_EPED_0 = k_start+npy.where(gamma[k_start:] > 1.0)[0][0]

        # PLOTTING GAMMA PROFILES
        gammaTepedfig = plt.figure("GAMMA PROFILES Teped",dpi=200)
        gammaTepedaxs = [None for i in range(9)]
        for n in range(len(modes)):
            figgrd = int("%d%d%d" % (3,3,n+1))
            gammaTepedaxs[n] = gammaTepedfig.add_subplot(figgrd)
            lstyle = styles[n]
            gammaTepedaxs[n].plot(teped,gamma_all,  color='r', linestyle = "-")
            gammaTepedaxs[n].plot(teped,gamma[:,n], color='k', linestyle = "--", label="mode = %d" % modes[n])
            gammaTepedaxs[n].axhline(1.0,ls="--",color="g")
            if k_EPED > 0:
                gammaTepedaxs[n].axvline(teped[k_EPED],   ls="--", color="g")
                gammaTepedaxs[n].axvline(teped[k_EPED_0], ls="--", color="g")
            gammaTepedaxs[n].set_yscale("symlog")
            if n+1 in [1,4,7]:
                gammaTepedaxs[n].set_ylabel("$\gamma/(\omega_{*}/2)$")
            if n+1 in [7,8,9]:
                gammaTepedaxs[n].set_xlabel("$T_{e,ped} (eV)$")
            if n+1 in [2,3,5,6,8,9]:
                gammaTepedaxs[n].yaxis.tick_right()
                gammaTepedaxs[n].yaxis.set_label_position("right")
                gammaTepedaxs[n].yaxis.set_ticks_position("none")
                gammaTepedaxs[n].yaxis.set_ticklabels([])
            gammaTepedaxs[n].legend()

        title_txt_01 = "$I_P$ = %3.2f, $N_{e,ped}$ = %3.2f, $\\beta_n$ = %3.2f, $B_{T}$ = %3.2f" \
                     % (epeddata[sim]["ip"]['data'], epeddata[sim]["neped"]['data'], epeddata[sim]["betan"]['data'], epeddata[sim]["bt"]['data'])
        title_txt_02 = "$\\kappa$ = %3.2f, $\\delta$ = %3.2f, $R$ = %3.2f, $a$ = %3.2f" \
                     % (epeddata[sim]["kappa"]['data'], epeddata[sim]["delta"]['data'], epeddata[sim]["r"]['data'], epeddata[sim]["a"]['data'])
        title_txt_03 = "$q_{95}$ = %3.2f, $\\ell_i$ = %3.2f" \
                     % (epeddata[sim]["eq_q95"]['data'][k_EPED], epeddata[sim]["eq_li"]['data'][k_EPED])

        gammaTepedfig.suptitle("GAMMA PROFILES\n%s\n%s\n%s" % (title_txt_01,title_txt_02,title_txt_03), fontsize = "10")
        gammaTepedfig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        gammaTepedfig.subplots_adjust(wspace=0,hspace=0)
        epedfigs.savefig(gammaTepedfig)
        if savepng: gammaTepedfig.savefig(figurepath+"eped_gamma_%04d_teped.png" % sims.index(sim))
        plt.close(gammaTepedfig)


        gammaBetanfig = plt.figure("GAMMA PROFILES Betan",dpi=200)
        figgrd = int("%d%d%d" % (1,1,1))
        gammaBetanaxs = gammaBetanfig.add_subplot(figgrd)
        lstyle = styles[0]
        gammaBetanaxs.plot(betan,gamma_all,  color='b', linestyle = "-",)
        gammaBetanaxs.set_yscale("symlog")
        gammaBetanaxs.axhline(1.0,ls="--",color="g")
        if k_EPED > 0:
            gammaBetanaxs.axvline(betan[k_EPED],   ls="--", color="g")
            gammaBetanaxs.axvline(betan[k_EPED_0], ls="--", color="g")
        gammaBetanaxs.set_xlabel("$\\beta_n$")
        gammaBetanaxs.set_ylabel("$\gamma/(\omega_{*}/2)$")

        gammaBetanfig.suptitle("GAMMA PROFILES\n%s\n%s\n%s" % (title_txt_01,title_txt_02,title_txt_03), fontsize = "10")
        gammaBetanfig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        gammaBetanfig.subplots_adjust(wspace=0,hspace=0)
        epedfigs.savefig(gammaBetanfig)
        if savepng: gammaBetanfig.savefig(figurepath+"eped_gamma_betan_%04d.png" % sims.index(sim))
        plt.close(gammaBetanfig)

    epedfigs.close()

    return 1

def read_nubeam_outputs(fpath):
   #if os.path.isfile(fpath):
   #    print(CGREEN + "FINDING NUBEAM OUTPUT AT %s: PASSED" % (fpath) + CEND)
   #else:
   #    print(CRED + "FINDING NUBEAM OUTPUT AT %s: FAILED" % (fpath) + CEND)
    cdffh = Dataset(fpath, mode='r')

    nubeam = {}
    for name, variable in cdffh.variables.items():
        nubeam[name]                  = {}
        nubeam[name]['data']          = cdffh.variables[name][:]
        if hasattr(variable, "unit"):
            nubeam[name]['units']     = getattr(variable, "units")
        else:
            nubeam[name]['units']     = ""
        if hasattr(variable, "long_name"):
            nubeam[name]['long_name']     = getattr(variable, "long_name")
        else:
            nubeam[name]['long_name']     = ""

    return nubeam

def read_nubeam(WORK_DIR):
    nubeamdata = {}

    if type(WORK_DIR) == str: WORK_DIR = [WORK_DIR]

    CASE_ID = 0
    shotref = ""

    for iWORK_DIR in WORK_DIR:
        iWORK_DIR_PATH = glob('%s/work/fastran_nb_nubeam_*' % os.path.abspath(iWORK_DIR))[0]
        WORK_FILES = glob('%s/ips_nbi_fld_state.cdf' % (iWORK_DIR_PATH))
        if WORK_FILES:
           NUBEAM_FILEPATH = WORK_FILES[0]
           NUBEAM_FILENAME = NUBEAM_FILEPATH.replace(iWORK_DIR_PATH+'/','')
           SHOT_NUMBER, TIME_ID = NUBEAM_FILENAME[1:].split('.')
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

           nubeamdata[shot] = read_nubeam_outputs(NUBEAM_FILEPATH)
           if len(nubeamdata[shot]['rtcena']['data']) == 0:
               del nubeamdata[shot]
               print(CRED   + 'READING CURRENT NUBEAM IN %s ... FAILED' % iWORK_DIR_PATH + CEND)
           else:
               print(CGREEN + 'READING CURRENT NUBEAM IN %s ... PASSED' % iWORK_DIR_PATH + CEND)
        else:
           print(CRED   + 'READING CURRENT NUBEAM IN %s ... FAILED' % iWORK_DIR_PATH + CEND)

    return nubeamdata

def read_genray_outputs(fpath):
    if os.path.isfile(fpath):
        print(CGREEN + "FINDING GENRAY OUTPUT AT %s: PASSED" % (fpath[0]) + CEND)
    else:
        print(CRED + "FINDING GENRAY OUTPUT AT %s: FAILED" % (fpath[0]) + CEND)
    cdffh = Dataset(fpath, mode='r')

    genray = {}
    for name, variable in cdffh.variables.items():
        genray[name]                  = {}
        genray[name]['data']          = cdffh.variables[name][:]
        if hasattr(variable, "unit"):
            genray[name]['units']     = getattr(variable, "units")
        else:
            genray[name]['units']     = ""
        if hasattr(variable, "long_name"):
            genray[name]['long_name']     = getattr(variable, "long_name")
        else:
            genray[name]['long_name']     = ""

    return genray

def read_genray(WORK_DIR,model):
    genraydata = {}

    if type(WORK_DIR) == str: WORK_DIR = [WORK_DIR]

    CASE_ID = 0
    shotref = ""

    for iWORK_DIR in WORK_DIR:
        if model == "lh":
            iWORK_DIR_PATH = glob('%s/work/fastran_lh_genray_*' % os.path.abspath(iWORK_DIR))[0]
        elif model == "hc":
            iWORK_DIR_PATH = glob('%s/work/fastran_hc_genray_*' % os.path.abspath(iWORK_DIR))[0]
        elif model == "ec":
            iWORK_DIR_PATH = glob('%s/work/fastran_ec_genray_*' % os.path.abspath(iWORK_DIR))[0]
        elif model == "ic":
            iWORK_DIR_PATH = glob('%s/work/fastran_ic_genray_*' % os.path.abspath(iWORK_DIR))[0]
        WORK_FILES = glob('%s/genray.nc' % (iWORK_DIR_PATH))
        if WORK_FILES:
           GENRAY_FILEPATH = WORK_FILES[0]
           GENRAY_FILENAME = GENRAY_FILEPATH.replace(iWORK_DIR_PATH+'/','')
           SHOT_NUMBER, TIME_ID = GENRAY_FILENAME[1:].split('.')
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

           genraydata[shot] = read_genray_outputs(GENRAY_FILEPATH)
           print(CGREEN + 'READING CURRENT GENRAY IN %s ... PASSED' % iWORK_DIR_PATH + CEND)
          #if len(genraydata[shot]['power_inj_total']['data']) == 0:
          #    del genraydata[shot]
          #    print(CRED   + 'READING CURRENT GENRAY IN %s ... FAILED' % iWORK_DIR_PATH + CEND)
          #else:
          #    print(CGREEN + 'READING CURRENT GENRAY IN %s ... PASSED' % iWORK_DIR_PATH + CEND)
        else:
           print(CRED   + 'READING CURRENT GENRAY IN %s ... FAILED' % iWORK_DIR_PATH + CEND)

    return genraydata       


def read_toray_outputs(fpath):
    if os.path.isfile(fpath[0]):
        print(CGREEN + "FINDING TORAY OUTPUT AT %s: PASSED" % (fpath[0]) + CEND)
    else:
        print(CRED + "FINDING TORAY OUTPUT AT %s: FAILED" % (fpath[0]) + CEND)
    cdffh = Dataset(fpath[0], mode='r')

    toray = {}
    for name, variable in cdffh.variables.items():
        toray[name]                  = {}
        toray[name]['data']          = cdffh.variables[name][:]
        if hasattr(variable, "unit"):
            toray[name]['units']     = getattr(variable, "units")
        else:
            toray[name]['units']     = ""
        if hasattr(variable, "long_name"):
            toray[name]['long_name']     = getattr(variable, "long_name")
        else:
            toray[name]['long_name']     = ""

        print(name, toray[name]['long_name'], npy.shape(toray[name]['data']))

    return toray


def read_cql3d_outputs(fpath):
    if os.path.isfile(fpath[0]):
        print(CGREEN + "FINDING CQL3D OUTPUT AT %s: PASSED" % (fpath[0]) + CEND)
    else:
        print(CRED + "FINDING CQL3D OUTPUT AT %s: FAILED" % (fpath[0]) + CEND)
    cdffh = Dataset(fpath[0], mode='r')

    cql3d = {}
    for name, variable in cdffh.variables.items():
        cql3d[name]                  = {}
        cql3d[name]['data']          = cdffh.variables[name][:]
        if hasattr(variable, "unit"):
            cql3d[name]['units']     = getattr(variable, "units")
        else:
            cql3d[name]['units']     = ""
        if hasattr(variable, "long_name"):
            cql3d[name]['long_name']     = getattr(variable, "long_name")
        else:
            cql3d[name]['long_name']     = ""

        if npy.size(cql3d[name]['data']) == 1:
            print(name, cql3d[name]['long_name'], npy.shape(cql3d[name]['data']), cql3d[name]['data'])
        else:
            print(name, cql3d[name]['long_name'], npy.shape(cql3d[name]['data']))

    return cql3d


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

def read_instate_inputs(instatefpath):
    namelistdata = Namelist(instatefpath)

    instatedata = {}
    for ikey in list(namelistdata['INSTATE'].keys()):
        instatedata[ikey] = {}
        instatedata[ikey]['data'] = namelistdata['INSTATE'][ikey]

    return instatedata

def read_instate(WORK_DIR):
    instatedata = {}

    if type(WORK_DIR) == str: WORK_DIR = [WORK_DIR]

    CASE_ID = 0
    shotref = ""

    for iWORK_DIR in WORK_DIR:
        iWORK_DIR_PATH = '%s' % os.path.abspath(iWORK_DIR)
        WORK_FILES = glob('%s/i*' % (iWORK_DIR_PATH))
        if not WORK_FILES:
            iWORK_DIR_PATH = '%s/work/plasma_state' % os.path.abspath(iWORK_DIR)
            WORK_FILES = glob('%s/i*' % (iWORK_DIR_PATH))
            if not WORK_FILES:
                iWORK_DIR_PATH = '%s' % os.path.abspath(iWORK_DIR)
                WORK_FILES = glob('%s/i*' % (iWORK_DIR_PATH))
        if WORK_FILES:
           INSTATE_FILEPATH = WORK_FILES[0]
           INSTATE_FILENAME = INSTATE_FILEPATH.replace(iWORK_DIR_PATH+'/','')
           SHOT_NUMBER, TIME_ID = INSTATE_FILENAME[1:].split('.')
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

           instatedata[shot] = read_instate_inputs(INSTATE_FILEPATH)
           if len(instatedata[shot]['NE_PED']['data']) == 0:
               del instatedata[shot]
               print(CRED   + 'READING CURRENT_INSTATE IN %s ... FAILED' % iWORK_DIR_PATH + CEND)
           else:
               print(CGREEN + 'READING CURRENT_INSTATE IN %s ... PASSED' % iWORK_DIR_PATH + CEND)
        else:
           print(CRED   + 'READING CURRENT_INSTATE IN %s ... FAILED' % iWORK_DIR_PATH + CEND)

    return instatedata

def read_fastran_outputs(fastranfpath):
    cdffh = Dataset(fastranfpath, mode='r')

    fastran = {}
    for name, variable in cdffh.variables.items():
        fastran[name]                  = {}
        fastran[name]['data']          = cdffh.variables[name][:]
        fastran[name]['units']         = ""
        fastran[name]['symbol']        = ""
        fastran[name]['long_name']     = ""

        if hasattr(variable, "unit"):      fastran[name]['units']     = getattr(variable, "units")
        elif name == 'time':               fastran[name]['units']     = "s"
        elif name == 'r0':                 fastran[name]['units']     = "m"
        elif name == 'a0':                 fastran[name]['units']     = "m"
        elif name == 'b0':                 fastran[name]['units']     = "T"
        elif name == 'wi':                 fastran[name]['units']     = "MJ"
        elif name == 'we':                 fastran[name]['units']     = "MJ"
        elif name == 'wb':                 fastran[name]['units']     = "MJ"
        elif name == 'pei':                fastran[name]['units']     = "MW"
        elif name == 'poh':                fastran[name]['units']     = "MW"
        elif name == 'taue':               fastran[name]['units']     = "s"
        elif name == 'taui':               fastran[name]['units']     = "s"
        elif name == 'tauth':              fastran[name]['units']     = "s"
        elif name == 'tau98':              fastran[name]['units']     = "s"
        elif name == 'tau89':              fastran[name]['units']     = "s"
        elif name == 'taunc':              fastran[name]['units']     = "s"
        elif name == 'tautot':             fastran[name]['units']     = "s"
        elif name == 'pnbe':               fastran[name]['units']     = "MW"
        elif name == 'pnbi':               fastran[name]['units']     = "MW"
        elif name == 'prfe':               fastran[name]['units']     = "MW"
        elif name == 'prfi':               fastran[name]['units']     = "MW"
        elif name == 'prad':               fastran[name]['units']     = "MW"
        elif name == 'pfuse':              fastran[name]['units']     = "MW"
        elif name == 'pfusi':              fastran[name]['units']     = "MW"
        elif name == 'pfuse_equiv':        fastran[name]['units']     = "MW"
        elif name == 'pfusi_equiv':        fastran[name]['units']     = "MW"
        else:                              fastran[name]['units']     = ""

        if hasattr(variable, "long_name"): fastran[name]['long_name'] = getattr(variable, "long_name")
        elif name == 'time':               fastran[name]['long_name'] = "Time"
        elif name == 'r0':                 fastran[name]['long_name'] = "Major Radius"
        elif name == 'a0':                 fastran[name]['long_name'] = "Minor Radius"
        elif name == 'b0':                 fastran[name]['long_name'] = "Toroidal Magnetic Field at r0"
        elif name == 'wi':                 fastran[name]['long_name'] = "Ion Thermal Stored Energy"
        elif name == 'we':                 fastran[name]['long_name'] = "Electron Thermal Stored Energy"
        elif name == 'wb':                 fastran[name]['long_name'] = "Fast Ion Thermal Stored Energy"
        elif name == 'pei':                fastran[name]['long_name'] = "Total e-i Exchange"
        elif name == 'poh':                fastran[name]['long_name'] = "Ohmic Heating"
        elif name == 'taue':               fastran[name]['long_name'] = "Electron Energy Confinement Time"
        elif name == 'taui':               fastran[name]['long_name'] = "Ion Energy Confinement Time"
        elif name == 'tauth':              fastran[name]['long_name'] = "Thermal Energy Confinement Time"
        elif name == 'tau98':              fastran[name]['long_name'] = "H-mode Confinement Time Scaling"
        elif name == 'tau89':              fastran[name]['long_name'] = "L-mode Confinement Time Scaling"
        elif name == 'taunc':              fastran[name]['long_name'] = "Neoclassical Confinement Time Scaling"
        elif name == 'tautot':             fastran[name]['long_name'] = "Total Energy Confinement Time"
        elif name == 'pnbe':               fastran[name]['long_name'] = "NB Electron Heating"
        elif name == 'pnbi':               fastran[name]['long_name'] = "NB Ion Heating"
        elif name == 'prfe':               fastran[name]['long_name'] = "RF Electron Heating"
        elif name == 'prfi':               fastran[name]['long_name'] = "RF Ion Heating"
        elif name == 'prad':               fastran[name]['long_name'] = "Total Radiation"
        elif name == 'pfuse':              fastran[name]['long_name'] = "Electron Alpha Heating"
        elif name == 'pfusi':              fastran[name]['long_name'] = "Ion Alpha Heating"
        elif name == 'betan':              fastran[name]['long_name'] = "Normalized Beta"
        elif name == 'amain':              fastran[name]['long_name'] = "Atomic Number of Main Ion"
        elif name == 'zmain':              fastran[name]['long_name'] = "Charge Number of Main Ion"
        elif name == 'pfuse_equiv':        fastran[name]['long_name'] = "Equilvalent Electron Alpha Heating for Deutron Plasma"
        elif name == 'pfusi_equiv':        fastran[name]['long_name'] = "Equilvalent Ion Alpha Heating for Deutron Plasma"
        else:                              fastran[name]['long_name'] = ""

        if   name == 'q':                  fastran[name]['symbol']    = "$q$"
        elif name == 'a':                  fastran[name]['symbol']    = "$a$"
        elif name == 'R':                  fastran[name]['symbol']    = "$R$"
        elif name == 'b0':                 fastran[name]['symbol']    = "$B_0$"
        elif name == 'te':                 fastran[name]['symbol']    = "$T_e$"           # DONE
        elif name == 'ti':                 fastran[name]['symbol']    = "$T_i$"           # DONE
        elif name == 'ne':                 fastran[name]['symbol']    = "$n_e$"           # DONE
        elif name == 'ni':                 fastran[name]['symbol']    = "$n_i$"           # DONE
        elif name == 'pe':                 fastran[name]['symbol']    = "$P_e$"           # DONE
        elif name == 'pi':                 fastran[name]['symbol']    = "$P_i$"           # DONE
        elif name == 'nz0':                fastran[name]['symbol']    = "$n_z$"
        elif name == 'rho':                fastran[name]['symbol']    = "$\\rho$"
        elif name == 'fpol':               fastran[name]['symbol']    = "$f$"
        elif name == 'j_bs':               fastran[name]['symbol']    = "$J_{BS}$"        # DONE
        elif name == 'j_nb':               fastran[name]['symbol']    = "$J_{NB}$"        # DONE
        elif name == 'j_rf':               fastran[name]['symbol']    = "$J_{RF}$"        # DONE
        elif name == 'j_oh':               fastran[name]['symbol']    = "$J_{OH}$"        # DONE
        elif name == 'shat':               fastran[name]['symbol']    = "$\\hat{s}$"
        elif name == 'chii':               fastran[name]['symbol']    = "$D_i$"
        elif name == 'chie':               fastran[name]['symbol']    = "$D_e$"
        elif name == 'pnbe':               fastran[name]['symbol']    = "$P_{NB_e}$"
        elif name == 'pnbi':               fastran[name]['symbol']    = "$P_{NB_i}$"
        elif name == 'prfe':               fastran[name]['symbol']    = "$P_{RF_e}$"
        elif name == 'prfi':               fastran[name]['symbol']    = "$P_{RF_i}$"
        elif name == 'pe_nb':              fastran[name]['symbol']    = "$P_{NB_e}$"
        elif name == 'pi_nb':              fastran[name]['symbol']    = "$P_{NB_i}$"
        elif name == 'pe_rf':              fastran[name]['symbol']    = "$P_{RF_e}$"
        elif name == 'pi_rf':              fastran[name]['symbol']    = "$P_{RF_i}$"
        elif name == 'fluxe':              fastran[name]['symbol']    = "$Q_e$"
        elif name == 'fluxi':              fastran[name]['symbol']    = "$Q_i$"
        elif name == 'j_tot':              fastran[name]['symbol']    = "$J_{TOT}$"       # DONE
        elif name == 'omega':              fastran[name]['symbol']    = "$\\Omega$"
        elif name == 'delta':              fastran[name]['symbol']    = "$\\delta$"
        elif name == 'kappa':              fastran[name]['symbol']    = "$\\kappa$"
        elif name == 'betan':              fastran[name]['symbol']    = "$\\beta_n$"
        elif name == 'shift':              fastran[name]['symbol']    = "$\\delta_r$"
        elif name == 'pfuse':              fastran[name]['symbol']    = "$P_{FUS_e}$"
        elif name == 'pfusi':              fastran[name]['symbol']    = "$P_{FUS_i}$"
        elif name == 'pe_fus':             fastran[name]['symbol']    = "$P_{FUS_e}$"
        elif name == 'pi_fus':             fastran[name]['symbol']    = "$P_{FUS_i}$"
        elif name == 'j_bs_0':             fastran[name]['symbol']    = "$J_{BS_0}$"
        elif name == 'chieneo':            fastran[name]['symbol']    = "$D_{e_{neo}}$"
        elif name == 'chiineo':            fastran[name]['symbol']    = "$D_{i_{neo}}$"
        elif name == 'chie_exp':           fastran[name]['symbol']    = "$D_{e,{balance}}$"
        elif name == 'chii_exp':           fastran[name]['symbol']    = "$D_{i,{balance}}}$"
        elif name == 'fluxe_exp':          fastran[name]['symbol']    = "$Q_{e,{balance}}$"
        elif name == 'fluxi_exp':          fastran[name]['symbol']    = "$Q_{i,{balance}}$"
    return fastran

#tea(time) : volume average electron temperature [keV]
#tia(time) : volume average ion temperature [keV]
#nebar(time) : line-average density [10^19/m^3]
#ip(time) : plasma current [MA]
#ibs(time) : bootstrap current [MA]
#inb(time) : NB current [MA]
#irf(time) : RF current [MA]
#sn(time) : particle source [10^19#]
#rhob(time, rho) : sqrt(toroidal_flux/B0) [m]
#te(time, rho) : electron temperature [keV]
#ti(time, rho) : ion temperature [keV]
#ne(time, rho) : electron density [10^19/m^3]
#ni(time, rho) : ion density [10^19/m^3]
#nz0(time, rho) : impurity 1 density [10^19/m^3]
#omega(time, rho) : rotation [rad/sec]
#zeff(time, rho) : zeff []
#aimp(imp) : mass number of impurities []
#zimp(imp) : charge number of impurities []
#tep(time, rho) : radial derivative of electron temperature with respect to rho [keV]
#tip(time, rho) : radial derivative of ion temperature with respect to rho [keV]
#nep(time, rho) : radial derivative of electron density with respect to rho [10^19/m^3]
#nip(time, rho) : radial derivative of electron density with respect to rho [10^19/m^3]
#j_tot(time, rho) : plasma current [MA/m^2]
#j_bs(time, rho) : bootstrap current [MA/m^2]
#j_bs_0(time, rho) : bootstrap current w/o smooth [MA/m^2]
#j_nb(time, rho) : NB current [MA/m^2]
#j_rf(time, rho) : RF current [MA/m^2]
#j_oh(time, rho) : Ohmic current [MA/m^2]
#pe_nb(time, rho) : NB electron heating [MW/m^3]
#pe_rf(time, rho) : RF electron heating [MW/m^3]
#p_rad(time, rho) : radiation [MW/m^3]
#p_ohm(time, rho) : Ohmic heating [MW/m^3]
#p_ei(time, rho) : electron-ion collisional heating [MW/m^3]
#pe_fus(time, rho) : electron alpha heating [MW/m^3]
#pe_ionization(time, rho) : electron ionization loss [MW/m^3]
#pi_nb(time, rho) : NB ion heating [MW/m^3]
#pi_rf(time, rho) : RF ion heating [MW/m^3]
#pi_fus(time, rho) : ion alpha heating [MW/m^3]
#pi_cx(time, rho) : charge exchange loss [MW/m^3]
#pi_ionization(time, rho) : ion ionization loss [MW/m^3]
#pe(time, rho) : total electron heating [MW/m^3]
#pi(time, rho) : total ion heating [MW/m^3]
#torque_nb(time, rho) : NB torque [Nm]
#q(time, rho) : safety factor []
#shat(time, rho) : magnetic shear []
#fpol(time, rho) : poloidal magentic flux [Wb]
#sigma(time, rho) : parallel electric conductivity [MS/m]
#vloop(time, rho) : loop voltage [V]
#pmhd(time, rho) : pressure from MHD equilibrium [Pa]
#qmhd(time, rho) : safety factor from MhD equilibrium []
#ermhd(time, rho) : radial electric field [V/m]
#wbeam(time, rho) : beam stored energy [MJ/m^3]
#nbeam(time, rho) : beam density [10^19/m^3]
#sion(time, rho) : particle source [10^19/m^3/s]
#walp(time, rho) : alpha stored energy [MJ]
#betan_loc(time, rho) : local normalized beta
#nhe(time, rho) : He ash density [10^19/m^3]
#chie(time, rho) : electron heat duffusivity [m^2/s]
#chie_etg(time, rho) : high-k electron heat duffusivity [m^2/s]
#chii(time, rho) : ion heat duffusivity [m^2/s]
#chiv(time, rho) : momentum duffusivity [m^2/s]
#chin(time, rho) : particle duffusivity [m^2/s]
#chieneo(time, rho) : neoclassical electron heat duffusivity [m^2/s]
#chiineo(time, rho) : neoclassical ion heat duffusivity [m^2/s]
#fluxe(time, rho) : electron energy flux [MW/m^2]
#fluxi(time, rho) : ion energy flux [MW/m^2]
#fluxe_exp(time, rho) : power balance electron energy flux [MW/m^2]
#fluxi_exp(time, rho) : power balance ion energy flux [MW/m^2]
#chie_exp(time, rho) : power balance electron heat duffusivity [m^2/s]
#chii_exp(time, rho) : power balance ion heat duffusivity [m^2/s]
#chiv_exp(time, rho) : power balance momentum duffusivity [m^2/s]
#volp(time, rho) : dV/drhob (m^2)
#ipol(time, rho) : rbt/(r0b0) []
#g11(time, rho) : volp < (grad rhob)^2 > [m^2]
#g22(time, rho) : volp / (4*pi**2) < (grad rhob/r)^2 > * r0/ipol [m]
#g33(time, rho) : < (r0/r)**2 > []
#gradrho(time, rho) : < grad rhob > []
#area(time, rho) : surface area [m^2]
#aminor(time, rho) : minor radius [m]
#rmajor(time, rho) : major radius [m]
#shift(time, rho) : shafranov shift [m]
#delta(time, rho) : triangularity []
#kappa(time, rho) : elongation []
#chin_exp(time, rho) : power balance electron heat duffusivity [m^2/s]
#chie_paleo(time, rho) : paleo-classical electron heat duffusivity [m^2/s]
#nue_s(time, rho) : normalized electron collision frequency []

def read_fastran(WORK_DIR):
    fastrandata = {}

    if type(WORK_DIR) == str: WORK_DIR = [WORK_DIR]

    CASE_ID = 0
    shotref = ""

    for iWORK_DIR in WORK_DIR:
        iWORK_DIR_PATH = '%s' % os.path.abspath(iWORK_DIR)
        WORK_FILES = glob('%s/f*' % (iWORK_DIR_PATH))
        if not WORK_FILES:
            iWORK_DIR_PATH = '%s/work/plasma_state' % os.path.abspath(iWORK_DIR)
            WORK_FILES = glob('%s/f*' % (iWORK_DIR_PATH))
            if not WORK_FILES:
                iWORK_DIR_PATH = '%s' % os.path.abspath(iWORK_DIR)
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
           if len(fastrandata[shot]['j_rf']['data']) == 0:
               del fastrandata[shot]
               print(CRED   + 'READING CURRENT_FASTRAN IN %s ... FAILED' % iWORK_DIR_PATH + CEND)
           else:
               print(CGREEN + 'READING CURRENT_FASTRAN IN %s ... PASSED' % iWORK_DIR_PATH + CEND)
        else:
           print(CRED   + 'READING CURRENT_FASTRAN IN %s ... FAILED' % iWORK_DIR_PATH + CEND)

    return fastrandata


def fastran_summary(fastrandata, write_to_file = True, **kwargs):
    shots = list(fastrandata.keys())
    nshot = len(shots)

    if write_to_file:
        reportpath = os.path.abspath(".")+"/fastran_report/"
        if not os.path.isdir(reportpath):
           os.system('mkdir '+reportpath)

        infopath = os.path.abspath(".")+"/fastran_report/Summary/"
        if not os.path.isdir(infopath):
           os.system('mkdir '+infopath)

    summary = {}

    for shot in shots:
        try:
            summary['ip']  = fastrandata[shot]["ip"]['data'][-1]
        except IndexError:
            return {}

        Pext  = 0.0
        for ipext in ["prfi","prfe","pnbi","pnbe"]:
            Pext += fastrandata[shot][ipext]['data'][-1]
        summary['Pext'] = Pext

        Pfus  = 0.0
        for ipfus in ["pfusi","pfuse"]:
            Pfus += fastrandata[shot][ipfus]['data'][-1]
        Pfus *= 5.0
        summary['Pfus'] = Pfus

        Q = Pfus/Pext
        summary['Q'] = Q

        INI  = 0.0
        for iINI in ["ibs","inb","irf"]:
            INI += fastrandata[shot][iINI]['data'][-1]
        summary['irf'] = fastrandata[shot]["irf"]['data'][-1]
        summary['inb'] = fastrandata[shot]["inb"]['data'][-1]
        summary['ibs'] = fastrandata[shot]["ibs"]['data'][-1]
        summary['poh'] = fastrandata[shot]["poh"]['data'][-1]

        fNI = INI/fastrandata[shot]["ip"]['data'][-1]
        summary['fNI'] = fNI

        fBS = fastrandata[shot]["ibs"]['data'][-1]/fastrandata[shot]["ip"]['data'][-1]
        summary['fBS'] = fBS

        H98 = fastrandata[shot]["tauth"]['data'][-1]/fastrandata[shot]["tau98"]['data'][-1]
        H89 = fastrandata[shot]["tauth"]['data'][-1]/fastrandata[shot]["tau89"]['data'][-1]
        summary['H98'] = H98

        nGW = fastrandata[shot]["ip"]['data'][-1]/npy.pi/fastrandata[shot]["a0"]['data'][-1]**2
        fGW = fastrandata[shot]["nebar"]['data'][-1]/10.0/nGW
        summary['fGW'] = fGW

        summary['betan'] = fastrandata[shot]["betan"]['data'][-1]

        if write_to_file:
            fhand = open("%ssummary_%s.dat" % (infopath,shot),"w")
            fhand.write("%s\t%5.3f\n" % ("Q",Q))
            fhand.write("%s\t%5.3f\n" % ("Poh",summary['poh']))
            fhand.write("%s\t%5.3f\n" % ("Ibs",summary['ibs']))
            fhand.write("%s\t%5.3f\n" % ("Inb",summary['inb']))
            fhand.write("%s\t%5.3f\n" % ("Irf",summary['irf']))
            fhand.write("%s\t%5.3f\n" % ("fNI",fNI))
            fhand.write("%s\t%5.3f\n" % ("fBS",fBS))
            fhand.write("%s\t%5.3f\n" % ("H98",H98))
            fhand.write("%s\t%5.3f\n" % ("H89",H89))
            fhand.write("%s\t%5.3f\n" % ("nGW",nGW))
            fhand.write("%s\t%5.3f\n" % ("fGW",fGW))
            fhand.write("%s\t%5.3f\n" % ("<ne>",fastrandata[shot]['nebar']['data'][-1]))
            fhand.write("%s\t%5.3f\n" % ("<Te>",fastrandata[shot]['tea']['data'][-1]))
            fhand.write("%s\t%5.3f\n" % ("<Ti>",fastrandata[shot]['tia']['data'][-1]))
            fhand.write("%s\t%5.3f\n" % ("Prad",fastrandata[shot]['prad']['data'][-1]))
            fhand.write("%s\t%5.3f\n" % ("Pext",npy.round(Pext)))
            fhand.write("%s\t%5.3f\n" % ("Pfus",Pfus))
            fhand.write("%s\t%5.3f\n" % ("betan",fastrandata[shot]["betan"]['data'][-1]))
            fhand.write("%s\t%5.3f\n" % ("WTHSTR",fastrandata[shot]['we']['data'][-1]+fastrandata[shot]['wi']['data'][-1]))
            fhand.close()

    return summary

def plot_fastran_outputs(fastrandata,plotparam={},**kwargs):
    sims  = list(fastrandata.keys())
    nsims = len(sims)

    if 'newplot' in plotparam:  newplot = plotparam['newplot']
    else:                       newplot = False

    if 'savepng' in plotparam:  savepng = plotparam['savepng']
    else:                       savepng = True

    if 'figspec' in plotparam:  figspec = plotparam['figspec']
    else:                       figspec = False

    if 'timetrace' in plotparam:  timetrace = plotparam['timetrace']
    else:                         timetrace = False

    reportpath = os.path.abspath(".")+"/fastran_report/"
    if not os.path.isdir(reportpath):
       os.system('mkdir '+reportpath)

    figurepath = os.path.abspath(".")+"/fastran_report/Figures/"
    if not os.path.isdir(figurepath):
       os.system('mkdir '+figurepath)

    if newplot and glob("%s*.png" % figurepath):
        os.system("rm %s*.png" % figurepath)

    fastranfigs = PdfPages(figurepath+'fastran_plots.pdf')

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

            if "groups" in jsonfdata["figures"][ifig]:
                groups = jsonfdata["figures"][ifig]["groups"]
                if   type(groups) in [int,float]: groups = [[groups]]
                elif listndims(groups) == 1 and type(groups) in [list,tuple]: groups = [groups]
            else:
                    groups = []

            nSubFigs = len(jsonfdata["figures"][ifig]["subplots"])
            for isubfig in range(len(jsonfdata["figures"][ifig]["subplots"])):
                figgrd = int("%d%d%d" % (jsonfdata["figures"][ifig]['grid'][0],jsonfdata["figures"][ifig]['grid'][1],isubfig+1))
                axs[isubfig] = fig.add_subplot(figgrd)
                if type(jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"]) not in [list,tuple]:
                    jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"] = [jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"]]
                llabel2 = []

                if jsonfdata["figures"][ifig]["subplots"][isubfig]['label2']:
                    legned2 = True
                else:
                    legned2 = False

                iline = 0; nlines = 0

                if groups:
                    if isubfig == 0:
                        bgnsim = 0
                        endsim = bgnsim+sum(groups[isubfig])
                    else:
                        bgnsim = endsim
                        endsim = bgnsim+sum(groups[isubfig])
                else:
                        bgnsim = 0
                        endsim = nsims
                iline = 0; nlines = 0
                lines = []

                if "yfactor" in jsonfdata["figures"][ifig]["subplots"][isubfig] and jsonfdata["figures"][ifig]["subplots"][isubfig]['yfactor']:
                    yfactor = jsonfdata["figures"][ifig]["subplots"][isubfig]['yfactor']
                else:                                                           
                    yfactor = 1

                for sim in sims[bgnsim:endsim]:
#                   if ifig == 0 and isubfig == 0: print(fastrandata[sim]['ip']);print(fastrandata[sim].keys())
                    if groups and (sims.index(sim)+1 - bgnsim) > groups[isubfig][iline]+nlines: iline += 1; nlines = sum(groups[isubfig][:iline])

                    for ifield in jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"]:
                        if nsims > 1:
                            lcolor = colors[sims.index(sim)]
                            lstyle = styles[jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"].index(ifield)+iline]
                        else:
                            lcolor = colors[jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"].index(ifield)+iline]
                            lstyle = styles[sims.index(sim)]

                        if legned2 and ifield == jsonfdata["figures"][ifig]["subplots"][isubfig]["fields"][0]:
                            if not jsonfdata["figures"][ifig]["subplots"][isubfig]['label2']:
                                llabel2.append(sim)
                                lines.append(Line2D([0],[0],color=lcolor,linestyle=lstyle))
                            else:
                                if type(jsonfdata["figures"][ifig]["subplots"][isubfig]['label2']) in [list,tuple]:
                                    try:
                                        if groups:
                                            if len(jsonfdata["figures"][ifig]["subplots"][isubfig]['label2']) > 1:
                                                llabel2.append(jsonfdata["figures"][ifig]["subplots"][isubfig]['label2'][sims.index(sim) - bgnsim])
                                            elif (sims.index(sim)+1 - bgnsim) == groups[isubfig][iline]+nlines:
                                                llabel2.append(jsonfdata["figures"][ifig]["subplots"][isubfig]['label2'][iline])
                                            lines.append(Line2D([0],[0],color=lcolor,linestyle=lstyle))
                                        elif not groups:
                                            llabel2.append(jsonfdata["figures"][ifig]["subplots"][isubfig]['label2'][sims.index(sim)])
                                            lines.append(Line2D([0],[0],color=lcolor,linestyle=lstyle))
                                    except IndexError:
                                        pass
                                else:
                                    if sim == sims[0]:
                                        llabel2.append(jsonfdata["figures"][ifig]["subplots"][isubfig]['label2'])
                                        lines.append(Line2D([0],[0],color=lcolor,linestyle=lstyle))

                        if ifield in fastrandata[sim]:
                            if type(jsonfdata["figures"][ifig]["subplots"][isubfig]['label']) == str:
                                if sim == sims[nlines]:
                                    if not jsonfdata["figures"][ifig]["subplots"][isubfig]['label']:
                                        if "symbol" in fastrandata[sim][ifield]:
                                            llabel = fastrandata[sim][ifield]['symbol']
                                        else:
                                            llabel = ifield
                                    else:
                                        if jsonfdata["figures"][ifig]["subplots"][isubfig]['label'] in fastrandata[shot]:
                                            llabel = fastrandata[shot][jsonfdata["figures"][ifig]["subplots"][isubfig]['label']]['symbol']
                                        else:
                                            llabel = jsonfdata["figures"][ifig]["subplots"][isubfig]['label']
                                else:
                                            llabel = ""
                            else:
                                try:
                                    llabel = jsonfdata["figures"][ifig]["subplots"][isubfig]['label'][sims.index(sim)-bgnsim]
                                except IndexError:
                                    llabel = ""

                            axs[isubfig].plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim][ifield]['data'][-1,:]/yfactor,color=lcolor,linestyle=lstyle,label=llabel)

                if "legncol" in jsonfdata["figures"][ifig]["subplots"][isubfig] and jsonfdata["figures"][ifig]["subplots"][isubfig]['legncol']:
                    legncol = int(jsonfdata["figures"][ifig]["subplots"][isubfig]['legncol'])
                else:                                                           
                    legncol = 1
                if "legncol2" in jsonfdata["figures"][ifig]["subplots"][isubfig] and jsonfdata["figures"][ifig]["subplots"][isubfig]['legncol2']:
                    legncol2 = int(jsonfdata["figures"][ifig]["subplots"][isubfig]['legncol2'])
                else:                                                           
                    legncol2 = 1
                if 'legfs' in jsonfdata["figures"][ifig]["subplots"][isubfig] and jsonfdata["figures"][ifig]["subplots"][isubfig]['legfs']:
                    legfs  = jsonfdata["figures"][ifig]["subplots"][isubfig]['legfs']
                else:
                    legfs  = 10
                if 'legfs2' in jsonfdata["figures"][ifig]["subplots"][isubfig] and jsonfdata["figures"][ifig]["subplots"][isubfig]['legfs2']:
                    legfs2  = jsonfdata["figures"][ifig]["subplots"][isubfig]['legfs2']
                else:
                    legfs2 = 10
                if 'yscale' in jsonfdata["figures"][ifig]["subplots"][isubfig] and jsonfdata["figures"][ifig]["subplots"][isubfig]['yscale']:
                    yscale  = jsonfdata["figures"][ifig]["subplots"][isubfig]['yscale']
                else:
                    yscale = 'linear'
                if 'xscale' in jsonfdata["figures"][ifig]["subplots"][isubfig] and jsonfdata["figures"][ifig]["subplots"][isubfig]['xscale']:
                    xscale  = jsonfdata["figures"][ifig]["subplots"][isubfig]['xscale']
                else:
                    xscale = 'linear'

                axs[isubfig].set_title( jsonfdata["figures"][ifig]["subplots"][isubfig]["title"],fontsize="10")
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

                if 'sublabel' in jsonfdata["figures"][ifig]["subplots"][isubfig] and jsonfdata["figures"][ifig]["subplots"][isubfig]['sublabel']:
                    sublabeltxt = sublabels[isubfig]
                    sublabelpos = jsonfdata["figures"][ifig]["subplots"][isubfig]['sublabel']
                    axs[isubfig].text(sublabelpos[0], sublabelpos[1], sublabeltxt, style='italic')
                else:
                    sublabeltxt = ""
                    sublabelpos = []

                if 'legned2' in locals() and legned2:
                    labels = llabel2
                    if jsonfdata["figures"][ifig]["subplots"][isubfig]['legend']:
                        if jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc2']:
                            leg2 = axs[isubfig].legend(lines, labels,loc=jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc2'],ncol=legncol2,frameon=False,fontsize=legfs2)
                        else:
                            leg2 = axs[isubfig].legend(lines, labels,loc='upper center',ncol=legncol2,frameon=False,fontsize='8')
                    else:
                        if jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc2']:
                            leg2 = axs[isubfig].legend(lines, labels,loc=jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc2'],ncol=legncol2,frameon=False,fontsize=legfs2)
                        else:
                            leg2 = axs[isubfig].legend(lines, labels,loc='upper right',ncol=legncol2,frameon=False,fontsize='8')
                if jsonfdata["figures"][ifig]["subplots"][isubfig]['legend']:
                    if jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc']:
                        leg1 = axs[isubfig].legend(loc=jsonfdata["figures"][ifig]["subplots"][isubfig]['legloc'],markerscale=19,ncol=legncol,frameon=False,fontsize=legfs)
                    else:
                        leg1 = axs[isubfig].legend(loc='upper right',markerscale=19,ncol=legncol,frameon=False,fontsize=legfs)
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
                if jsonfdata["figures"][ifig]["subplots"][isubfig]["xlimit"]:
                   axs[isubfig].set_xlim(jsonfdata["figures"][ifig]["subplots"][isubfig]["xlimit"])
                axs[isubfig].set_xscale(xscale.strip())
                axs[isubfig].set_yscale(yscale.strip())

            if jsonfdata["figures"][ifig]["title"]:
                fig.suptitle(jsonfdata["figures"][ifig]["title"])
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
            fig.subplots_adjust(wspace=0,hspace=0)
            fastranfigs.savefig(fig)
            if savepng: fig.savefig(figurepath+"fastran_%s.png" % jsonfdata["figures"][ifig]['name'])
            plt.close(fig)

        fastranfigs.close()

    else:
        Tefig = plt.figure("ELECTRON TEMPERATURE PROFILE",dpi=200)
        Teaxs = Tefig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lstyle = styles[0]
            llabel = sim
            Teaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['te']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        Teaxs.set_title("Electron Temperature Profile")
        Teaxs.set_ylabel("$T_e$")
        Teaxs.set_xlabel("$\\rho$")
        Teaxs.legend()
        fastranfigs.savefig(Tefig)
        if savepng: Tefig.savefig(figurepath+"fastran_Te.png")
        plt.close(Tefig)

        Tifig = plt.figure("ION TEMPERATURE PROFILE",dpi=200)
        Tiaxs = Tifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lstyle = styles[0]
            llabel = sim
            Tiaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['ti']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        Tiaxs.set_title("Ion Temperature Profile")
        Tiaxs.set_ylabel("$T_i$")
        Tiaxs.set_xlabel("$\\rho$")
        Tiaxs.legend()
        fastranfigs.savefig(Tifig)
        if savepng: Tifig.savefig(figurepath+"fastran_Ti.png")
        plt.close(Tifig)


        nefig = plt.figure("ELECTRON DENSITY PROFILE",dpi=200)
        neaxs = nefig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lstyle = styles[0]
            llabel = sim
            neaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['ne']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        neaxs.set_title("Electron Density Profile")
        neaxs.set_ylabel("$n_e$")
        neaxs.set_xlabel("$\\rho$")
        neaxs.legend()
        fastranfigs.savefig(nefig)
        if savepng: nefig.savefig(figurepath+"fastran_ne.png")
        plt.close(nefig)

        nifig = plt.figure("ION DENSITY PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lsytle = styles[0]
            llabel = sim
            niaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['ni']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Ion Density Profile")
        niaxs.set_ylabel("$n_i$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"fastran_ni.png")
        plt.close(nifig)

        nifig = plt.figure("BOOTSTRAP CURRENT PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lsytle = styles[0]
            llabel = sim
            niaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['j_bs']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Bootstrap Current Profile")
        niaxs.set_ylabel("$J_{bs}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"fastran_jbs.png")
        plt.close(nifig)

        nifig = plt.figure("OHMIC CURRENT PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lsytle = styles[0]
            llabel = sim
            niaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['j_oh']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Ohmic Current Profile")
        niaxs.set_ylabel("$J_{oh}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"fastran_joh.png")
        plt.close(nifig)

        nifig = plt.figure("NB CURRENT PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lsytle = styles[0]
            llabel = sim
            niaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['j_nb']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("NB Current Profile")
        niaxs.set_ylabel("$J_{nb}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"fastran_jnb.png")
        plt.close(nifig)

        nifig = plt.figure("RF CURRENT PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lsytle = styles[0]
            llabel = sim
            niaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['j_rf']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("RF Current Profile")
        niaxs.set_ylabel("$J_{rf}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"fastran_jrf.png")
        plt.close(nifig)

        nifig = plt.figure("TOTAL CURRENT PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lsytle = styles[0]
            llabel = sim
            niaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['j_tot']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Total Current Profile")
        niaxs.set_ylabel("$J_{tot}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"fastran_jtot.png")
        plt.close(nifig)

        nifig = plt.figure("SAFETY FACTOR PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lsytle = styles[0]
            llabel = sim
            niaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['q']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Safety Factor Profile")
        niaxs.set_ylabel("$q$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"fastran_q.png")
        plt.close(nifig)

        nifig = plt.figure("MAGNETIC SHEAR PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lsytle = styles[0]
            llabel = sim
            niaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['shat']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Magnetic Shear Profile")
        niaxs.set_ylabel("$\\hat{s}$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"fastran_shat.png")
        plt.close(nifig)

        nifig = plt.figure("MAGNETIC PERTURBATION PROFILE",dpi=200)
        niaxs = nifig.add_subplot(111)
        for sim in sims:
            lcolor = colors[sims.index(sim)]
            lsytle = styles[0]
            llabel = sim
            niaxs.plot(fastrandata[sim]['rho']['data'][:],fastrandata[sim]['betan_loc']['data'][-1,:],color=lcolor,linestyle=lstyle,label=llabel)
        niaxs.set_title("Magnetic Perturbation Profile")
        niaxs.set_ylabel("$\\beta_N$")
        niaxs.set_xlabel("$\\rho$")
        niaxs.legend()
        fastranfigs.savefig(nifig)
        if savepng: nifig.savefig(figurepath+"fastran_betan.png")
        plt.close(nifig)

        fastranfigs.close()

    return True
        

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--lh',      '-lh',           action='store_const',const=1,help='Lower-Hybrid HCD Model.')
   parser.add_argument('--hc',      '-hc',           action='store_const',const=1,help='Helicon HCD Model.')
   parser.add_argument('--ic',      '-ic',           action='store_const',const=1,help='Ion-Cyclotron HCD Model.')
   parser.add_argument('--ec',      '-ec',           action='store_const',const=1,help='Electron-Cyclotron HCD Model.')
   parser.add_argument('--plot',    '-plot',         action='store_const',const=1,help='Plot FASTRAN ouputs.')
   parser.add_argument('--dcon',    '-dcon',         action='store_const',const=1,help='Calculate the beta values using DCON stability code.')
   parser.add_argument('--eped',    '-eped',         action='store_const',const=1,help='Read and Plot EPED Simulation Output file eped_state.nc.')
   parser.add_argument('--model',   '-model',        action='store_const',const=1,help='Create a MODEL from EPED data.')
   parser.add_argument('--state',   '-state',        action='store_const',const=1,help='Include STATE data in the summary and plots.')
   parser.add_argument('--toray',   '-toray',        action='store_const',const=1,help='Read and Plot TORAY Simulation Output file toray.nc.')
   parser.add_argument('--cql3d',   '-cql3d',        action='store_const',const=1,help='Read and Plot CQL3D Simulation Output file cql3d.nc.')
   parser.add_argument('--genray',  '-genray',       action='store_const',const=1,help='Read and Plot GENRAY Simulation Output file genray.nc.')
   parser.add_argument('--nubeam',  '-nubeam',       action='store_const',const=1,help='Read and Plot NUBEAM Simulation Output file ips_nbi_fld_state.cdf.')
   parser.add_argument('--fields',  '-fields',       action='store_const',const=1,help='Print Fields in target output, defualt: fastran.')
   parser.add_argument('--geqdsk',  '-geqdsk',  action='store_const',const=1,help='Read and Plot Equilibrium State from FASTRAN Simulation geqdsk file.')
   parser.add_argument('--summary', '-summary',      action='store_const',const=1,help='Plot FASTRAN output from Summary Folder.')
   parser.add_argument('--figspec', '-figspec',      action='store_const',const=1,help='Create Figures Based on Specifications provided by the user.')
   parser.add_argument('--newplot', '-newplot',      action='store_const',const=1,help='Remove the old figures and plot new ones.')
   parser.add_argument('--fastran', '-fastran',      action='store_const',const=1,help='Read and Plot FASTRAN Simulation Output file fastran.nc.')

   parser.add_argument('inputs',nargs='*')

   if parser.parse_args():
       args    = parser.parse_args()
       lh      = args.lh
       hc      = args.hc
       ec      = args.ec
       ic      = args.ic
       plot    = args.plot
       dcon    = args.dcon
       eped    = args.eped
       model   = args.model
       state   = args.state
       toray   = args.toray
       cql3d   = args.cql3d
       genray  = args.genray
       nubeam  = args.nubeam
       inputs  = args.inputs
       fields  = args.fields
       geqdsk  = args.geqdsk
       summary = args.summary
       figspec = args.figspec
       newplot = args.newplot
       fastran = args.fastran

  #if not (fields or plot or figspec or summary or state or newplot or getbeta or genray or toray or cql3d or eped): plot = True
   if not (fastran or state or genray or toray or cql3d or eped or nubeam or geqdsk): fastran = True
   if fastran and not (summary or figspec or plot or fields): plot = True

   if   lh: hcd_model = "lh"
   elif hc: hcd_model = "hc"
   elif ec: hcd_model = "ec"
   elif ic: hcd_model = "ic"

   plotparam = {}
   if newplot: plotparam['newplot'] = True
   if figspec: plotparam['figspec'] = True

   for i in inputs:
       if i.isnumeric(): inputs[inputs.index(i)] = glob("simulation_*_%04d" % int(i))[0]
   if figspec:
       if not os.path.isfile("figspec.json"):
           print("figspec.json DOES NOT EXIST ... EXIT!")
           sys.exit()
       jsonfhand = open('figspec.json')
       jsonfdata = json.load(jsonfhand)
       jsonfhand.close()
       if 'simulations' in jsonfdata:
          for i in jsonfdata['simulations']:
              if type(i) == int:
                  inputs.append(glob("simulation_*_%04d" % i)[0])
              elif type(i) == float:
                  inputs.append(glob("simulation_*_%04d" % int(i))[0])
              elif type(i) == str and i.isnumeric():
                  inputs.append(glob("simulation_*_%04d" % int(i))[0])
              elif type(i) == str and os.path.isdir(i):
                  inputs.append(i)

   if inputs == []:
       print(CRED + 'SIMULATION FOLDER(S) NOT FOUND/PROVIDED ... EXIT!' + CEND)
       sys.exit()

   if   dcon:
        dcon_betan = calc_dcon_betan(WORK_DIR=inputs)
        
   if fastran:
        if (plot or figspec) and summary:
             fastrandata = read_fastran(WORK_DIR=inputs)
             if fastrandata:
                 returnvals  = fastran_summary(fastrandata)
                 plotsreturn = plot_fastran_outputs(fastrandata,plotparam=plotparam)
        elif summary or fields:
             fastrandata = read_fastran(WORK_DIR=inputs)
             if fastrandata and summary:
                 returnvals  = fastran_summary(fastrandata)
             else:
                 ikeys   = list(fastrandata.keys())
                 for ikey in ikeys:
                   #print(ikey,fastrandata[ikey]['poh']['data'][-1],fastrandata[ikey]['ip']['data'][-1])
                   #print(ikey,fastrandata[ikey]['j_oh']['data'][-1,10],fastrandata[ikey]['poh']['data'][-1],fastrandata[ikey]['ip']['data'][-1])
                    print(ikey,fastrandata[ikey]['betan']['data'][-1])
                 #  if abs(fastrandata[ikey]['j_oh']['data'][-1,10]) < 0.1:
                 #      print(ikey,fastrandata[ikey]['j_oh']['data'][-1,10],fastrandata[ikey]['poh']['data'][-1],fastrandata[ikey]['ip']['data'][-1])
                   #print(fastrandata[ikey]['ip']['data'][-1])
                   #print(fastrandata[ikey]['zeff']['data'][-1])
                #ikey   = list(fastrandata.keys())[0]
                #fields = list(fastrandata[ikey].keys())
                #for ifield in fields:
                #    print(ifield, fastrandata[ikey][ifield]['long_name'], npy.shape(fastrandata[ikey][ifield]['data']))
        elif plot or figspec:
             fastrandata = read_fastran(WORK_DIR=inputs)
             if fastrandata:
                 returnvals  = plot_fastran_outputs(fastrandata,plotparam=plotparam)

   if genray:
       #genraydata = read_genray_outputs(fpath=inputs)
       #print(genraydata['power_inj_total']['data'],genraydata['powtot_e']['data'])
        genraydata = read_genray(WORK_DIR=inputs,model=hcd_model.strip())
        sims = list(genraydata.keys())
        for isim in sims:
            print(genraydata[isim]['power_inj_total']['data'],genraydata[isim]['powtot_e']['data'])
       #if genraydata:
       #    returnvals  = plot_genray_outputs(genraydata,plotparam=plotparam)

   if toray:
        toraydata = read_toray_outputs(fpath=inputs)
       #toraydata = read_toray(WORK_DIR=inputs)
       #if toraydata:
       #    returnvals  = plot_toray_outputs(toraydata,plotparam=plotparam)

   if cql3d:
        cql3ddata = read_cql3d_outputs(fpath=inputs)
       #cql3ddata = read_cql3d(WORK_DIR=inputs)
       #if cql3ddata:
       #    returnvals  = plot_cql3d_outputs(cql3ddata,plotparam=plotparam)

   if nubeam:
       #nubeamdata = read_nubeam_outputs(fpath=inputs)
        nubeamdata = read_nubeam(WORK_DIR=inputs)
        sims = list(nubeamdata.keys())
        for isim in sims:
            print(nubeamdata[isim]['pinja']['data'][0])
       #    print('rtcena = %7.4f, xlbtna = %7.4f, xybsca = %7.4f, xlbapa = %7.4f, xybapa = %7.4f, foclra = %7.4f, foclza = %7.4f, divra = %7.4f, divza = %7.4f' %
       #          (
       #              nubeamdata[isim]['rtcena']['data'][0], nubeamdata[isim]['xlbtna']['data'][0], nubeamdata[isim]['xybsca']['data'][0],
       #              nubeamdata[isim]['xlbapa']['data'][0], nubeamdata[isim]['xybapa']['data'][0],
       #              nubeamdata[isim]['foclra']['data'][0], nubeamdata[isim]['foclza']['data'][0],
       #              nubeamdata[isim]['divra']['data'][0],  nubeamdata[isim]['divza']['data'][0]
       #          )
       #         )
       #for ikey in list(nubeamdata.keys()):
       #    print(ikey,nubeamdata[ikey]['long_name'],npy.shape(nubeamdata[ikey]['data']))
       #nubeamdata = read_nubeam(WORK_DIR=inputs)
       #if nubemadata:
       #    returnvals  = plot_nubeam_outputs(nubeamdata,plotparam=plotparam)

   if eped:
       #epeddata = read_eped_outputs(fpath=inputs)
       #for ikey in list(epeddata.keys()):
       #    print(ikey,epeddata[ikey]['long_name'],npy.shape(epeddata[ikey]['data']))
        epeddata = read_eped(WORK_DIR=inputs)
        if epeddata and plot:
            returnvals  = plot_eped_outputs(epeddata,plotparam=plotparam)
        elif epeddata and model:
            returnvals  = eped_model(epeddata)
            print(returnvals['PPED'],returnvals['WPED'])
        elif epeddata and fields:
            #field_list  = []
            #field_list += ['ip','bt','r','a','kappa','delta','zeta','betan','k_EPED','neped','zeffped']
            #field_list += ['p_E1','wid_E1','ptop_E1','widtop_E1']
            #field_list += ['eq_tped','eq_pped','eq_ptop','eq_ttop','eq_betanped','eq_wped_rho','eq_wped_psi']
            #ikeys   = sorted(list(epeddata.keys()))
            #for ikey in ikeys:
            #    print(CBLUE + ikey + CEND)
            #    for ifield in field_list:
            #        if ifield in ['eq_tped','eq_pped','eq_ptop','eq_ttop','eq_betanped','eq_wped_rho','eq_wped_psi']:
            #            k_EPED = epeddata[ikey]['k_EPED']['data'][-1]
            #            print(ifield," = ",epeddata[ikey][ifield]['data'][k_EPED])
            #        else:
            #            print(ifield," = ",epeddata[ikey][ifield]['data'][-1])
             ikey   = list(epeddata.keys())[0]
             fields = list(epeddata[ikey].keys())
             for ifield in fields:
                 print(ifield, epeddata[ikey][ifield]['long_name'], npy.shape(epeddata[ikey][ifield]['data']))

   if geqdsk:
       geqdskdata = read_geqdsk(WORK_DIR=inputs)
       if geqdskdata and plot:
           returnvals  = plot_geqdsk_outputs(geqdskdata,plotparam=plotparam)

 
