#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Developed by Ehab Hassan on 2021-07-01

import os
import sys
import argparse

import numpy as npy

from glob import glob
from Namelist import Namelist
from fastran_plot import fastran_plot

CEND    = '\033[0m'
CRED    = '\33[31m'
CBLUE   = '\33[34m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'

def findall(inlist,item):
    inds = []
    for i,character in enumerate(inlist):
        if character==item: inds.append(i)
    return inds


def fastran_report(fastranfpath="./",reportparam={}):
    if type(fastranfpath) not in [list]:
       fastranfpath = [fastranfpath]

    hcd      = {}
    tglf     = {}
    lshots   = []
    nubeam   = {}
    nfreya   = {}
    genray   = {}
    fastran  = {}
    genraylh = {}
    genrayhc = {}

    CASE_ID = 0
    shotref = ""

    for ifastran in fastranfpath:
        inputpath = "./" + ifastran + "/input"
        inputfile = glob(inputpath + "/*")

        path_to_plamsa_state = "./" + ifastran + "/work/plasma_state"
        plasma_state_file = glob(path_to_plamsa_state + "/*")
        slashinds = findall(plasma_state_file[0],"/")
        SHOT_NUM, TIME_ID = plasma_state_file[0][slashinds[-1]+2:].split(".")

        fastran_output_file = path_to_plamsa_state + "/f" + SHOT_NUM + "." + TIME_ID
        if fastran_output_file not in plasma_state_file: continue

        SHOT_ID = SHOT_NUM+"."+TIME_ID
        if SHOT_ID == shotref:
            CASE_ID += 1
        elif shotref == "":
            shotref = SHOT_ID
            CASE_ID += 1
        elif shotref != "" and SHOT_ID != shotref:
            shotref = SHOT_ID
            CASE_ID = 1
        SHOT_ID += ".%02d" % CASE_ID
        lshots.append(SHOT_ID)

        print(CYELLOW + "Reading INPUT Files for %s Case" % SHOT_ID)
        if inputpath + "/inhcd" in inputfile:
            hcd[SHOT_ID]    = Namelist(inputpath+"/inhcd")
            print(CGREEN + 'Reading inhcd ......... PASSED' + CEND)
            hcd['status']   = True
        else:
            hcd['status']   = False
        if inputpath + "/intglf" in inputfile:
            tglf[SHOT_ID]     = Namelist(inputpath+"/intglf")
            print(CGREEN + 'Reading intglf ... PASSED' + CEND)
            tglf['status']   = True
        else:
            tglf['status']   = False
        if inputpath + "/innubeam" in inputfile:
            nubeam[SHOT_ID]   = Namelist(inputpath+"/innubeam")
            print(CGREEN + 'Reading innubeam ...... PASSED' + CEND)
            nubeam['status']   = True
        else:
            nubeam['status']   = False
        if inputpath + "/innfreya" in inputfile:
            nfreya[SHOT_ID]   = Namelist(inputpath+"/innfreya")
            print(CGREEN + 'Reading innfreya ...... PASSED' + CEND)
            nfreya['status']   = True
        else:
            nfreya['status']   = False
        if inputpath + "/infastran" in inputfile:
            fastran[SHOT_ID]  = Namelist(inputpath+"/infastran",case='lower')
            print(CGREEN + 'Reading infastran ..... PASSED' + CEND)
            fastran['status']   = True
        else:
            fastran['status']   = False
        if inputpath + "/ingenray_LH" in inputfile:
            genraylh[SHOT_ID] = Namelist(inputpath+"/ingenray_LH")
            print(CGREEN + 'Reading ingenray_LH ... PASSED' + CEND)
            genraylh['status']   = True
        else:
            genraylh['status']   = False
        if inputpath + "/ingenray_HC" in inputfile:
            genrayhc[SHOT_ID] = Namelist(inputpath+"/ingenray_HC")
            print(CGREEN + 'Reading ingenray_HC ... PASSED' + CEND)
            genrayhc['status']   = True
        else:
            genrayhc['status']   = False

    hcddiff      = []
    tglfdiff     = []
    nubeamdiff   = []
    nfreyadiff   = []
    fastrandiff  = []
    genraylhdiff = []
    genrayhcdiff = []

    for ishot in range(len(lshots)-1):
        if hcd['status']:
            for ikey in hcd[lshots[ishot]]['inhcd'].keys():
                if   ikey in hcddiff: continue
                elif hcd[lshots[ishot]]['inhcd'][ikey] != hcd[lshots[ishot+1]]['inhcd'][ikey]:
                     hcddiff.append(ikey)

        if tglf['status']:
            for ikey in tglf[lshots[ishot]]['intglf'].keys():
                if   ikey in tglfdiff: continue
                elif tglf[lshots[ishot]]['intglf'][ikey] != tglf[lshots[ishot+1]]['intglf'][ikey]:
                     tglfdiff.append(ikey)

        if fastran['status']:
            for ikey in fastran[lshots[ishot]]['infastran'].keys():
                if   ikey in fastrandiff: continue
                elif fastran[lshots[ishot]]['infastran'][ikey] != fastran[lshots[ishot+1]]['infastran'][ikey]:
                     fastrandiff.append(ikey)

        if nfreya['status']:
            for ikey in nfreya[lshots[ishot]]['innfreya'].keys():
                if   ikey in nfreyadiff: continue
                elif nfreya[lshots[ishot]]['innfreya'][ikey] != nfreya[lshots[ishot+1]]['innfreya'][ikey]:
                     nfreyadiff.append(ikey)

        if nubeam['status']:
            for ikey in nubeam[lshots[ishot]]['NBI_CONFIG'].keys():
                if   ikey in nubeamdiff: continue
                elif nubeam[lshots[ishot]]['NBI_CONFIG'][ikey] != nubeam[lshots[ishot+1]]['NBI_CONFIG'][ikey]:
                     nubeamdiff.append(ikey)
            for ikey in nubeam[lshots[ishot]]['NBI_MODEL'].keys():
                if   ikey in nubeamdiff: continue
                elif nubeam[lshots[ishot]]['NBI_MODEL'][ikey] != nubeam[lshots[ishot+1]]['NBI_MODEL'][ikey]:
                     nubeamdiff.append(ikey)
            for ikey in nubeam[lshots[ishot]]['NBI_INIT'].keys():
                if   ikey in nubeamdiff: continue
                elif nubeam[lshots[ishot]]['NBI_INIT'][ikey] != nubeam[lshots[ishot+1]]['NBI_INIT'][ikey]:
                     nubeamdiff.append(ikey)

        if genraylh['status']:
            for ikey in genraylh[lshots[ishot]]['GENR'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['GENR'][ikey] != genraylh[lshots[ishot+1]]['GENR'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['TOKAMAK'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['TOKAMAK'][ikey] != genraylh[lshots[ishot+1]]['TOKAMAK'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['WAVE'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['WAVE'][ikey] != genraylh[lshots[ishot+1]]['WAVE'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['SCATNPER'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['SCATNPER'][ikey] != genraylh[lshots[ishot+1]]['SCATNPER'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['DISPERS'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['DISPERS'][ikey] != genraylh[lshots[ishot+1]]['DISPERS'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['NUMERCL'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['NUMERCL'][ikey] != genraylh[lshots[ishot+1]]['NUMERCL'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['OUTPUT'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['OUTPUT'][ikey] != genraylh[lshots[ishot+1]]['OUTPUT'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['PLASMA'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['PLASMA'][ikey] != genraylh[lshots[ishot+1]]['PLASMA'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['EDGE_PROF_NML'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['EDGE_PROF_NML'][ikey] != genraylh[lshots[ishot+1]]['EDGE_PROF_NML'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['ADJ_NML'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['ADJ_NML'][ikey] != genraylh[lshots[ishot+1]]['ADJ_NML'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['OX'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['OX'][ikey] != genraylh[lshots[ishot+1]]['OX'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['EMISSION'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['EMISSION'][ikey] != genraylh[lshots[ishot+1]]['EMISSION'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['READ_DISKF'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['READ_DISKF'][ikey] != genraylh[lshots[ishot+1]]['READ_DISKF'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['ZEFTAB_NONUNIFORM_LINE'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['ZEFTAB_NONUNIFORM_LINE'][ikey] != genraylh[lshots[ishot+1]]['ZEFTAB_NONUNIFORM_LINE'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['VFLOWTAB_NONUNIFORM_LINE'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['VFLOWTAB_NONUNIFORM_LINE'][ikey] != genraylh[lshots[ishot+1]]['VFLOWTAB_NONUNIFORM_LINE'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['TPOP_NONUNIFORM_LINE'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['TPOP_NONUNIFORM_LINE'][ikey] != genraylh[lshots[ishot+1]]['TPOP_NONUNIFORM_LINE'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['TEMTAB_NONUNIFORM_LINE'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['TEMTAB_NONUNIFORM_LINE'][ikey] != genraylh[lshots[ishot+1]]['TEMTAB_NONUNIFORM_LINE'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['DENTAB_NONUNIFORM_LINE'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['DENTAB_NONUNIFORM_LINE'][ikey] != genraylh[lshots[ishot+1]]['DENTAB_NONUNIFORM_LINE'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['ECCONE'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['ECCONE'][ikey] != genraylh[lshots[ishot+1]]['ECCONE'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['GRILL'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['GRILL'][ikey] != genraylh[lshots[ishot+1]]['GRILL'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['SPECIES'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['SPECIES'][ikey] != genraylh[lshots[ishot+1]]['SPECIES'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['VARDEN'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['VARDEN'][ikey] != genraylh[lshots[ishot+1]]['VARDEN'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['DENPROF'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['DENPROF'][ikey] != genraylh[lshots[ishot+1]]['DENPROF'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['TPOPPROF'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['TPOPPROF'][ikey] != genraylh[lshots[ishot+1]]['TPOPPROF'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['VFLPROF'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['VFLPROF'][ikey] != genraylh[lshots[ishot+1]]['VFLPROF'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['ZPROF'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['ZPROF'][ikey] != genraylh[lshots[ishot+1]]['ZPROF'][ikey]:
                     genraylhdiff.append(ikey)
            for ikey in genraylh[lshots[ishot]]['TPROF'].keys():
                if   ikey in genraylhdiff: continue
                elif genraylh[lshots[ishot]]['TPROF'][ikey] != genraylh[lshots[ishot+1]]['TPROF'][ikey]:
                     genraylhdiff.append(ikey)

        if genrayhc['status']:
            for ikey in genrayhc[lshots[ishot]]['GENR'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['GENR'][ikey] != genrayhc[lshots[ishot+1]]['GENR'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['TOKAMAK'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['TOKAMAK'][ikey] != genrayhc[lshots[ishot+1]]['TOKAMAK'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['WAVE'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['WAVE'][ikey] != genrayhc[lshots[ishot+1]]['WAVE'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['SCATNPER'].keys():
                if   ikey in genraylhdiff: continue
                elif genrayhc[lshots[ishot]]['SCATNPER'][ikey] != genrayhc[lshots[ishot+1]]['SCATNPER'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['DISPERS'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['DISPERS'][ikey] != genrayhc[lshots[ishot+1]]['DISPERS'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['NUMERCL'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['NUMERCL'][ikey] != genrayhc[lshots[ishot+1]]['NUMERCL'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['OUTPUT'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['OUTPUT'][ikey] != genrayhc[lshots[ishot+1]]['OUTPUT'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['PLASMA'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['PLASMA'][ikey] != genrayhc[lshots[ishot+1]]['PLASMA'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['EDGE_PROF_NML'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['EDGE_PROF_NML'][ikey] != genrayhc[lshots[ishot+1]]['EDGE_PROF_NML'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['ADJ_NML'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['ADJ_NML'][ikey] != genrayhc[lshots[ishot+1]]['ADJ_NML'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['OX'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['OX'][ikey] != genrayhc[lshots[ishot+1]]['OX'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['EMISSION'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['EMISSION'][ikey] != genrayhc[lshots[ishot+1]]['EMISSION'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['READ_DISKF'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['READ_DISKF'][ikey] != genrayhc[lshots[ishot+1]]['READ_DISKF'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['ZEFTAB_NONUNIFORM_LINE'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['ZEFTAB_NONUNIFORM_LINE'][ikey] != genrayhc[lshots[ishot+1]]['ZEFTAB_NONUNIFORM_LINE'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['VFLOWTAB_NONUNIFORM_LINE'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['VFLOWTAB_NONUNIFORM_LINE'][ikey] != genrayhc[lshots[ishot+1]]['VFLOWTAB_NONUNIFORM_LINE'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['TPOP_NONUNIFORM_LINE'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['TPOP_NONUNIFORM_LINE'][ikey] != genrayhc[lshots[ishot+1]]['TPOP_NONUNIFORM_LINE'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['TEMTAB_NONUNIFORM_LINE'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['TEMTAB_NONUNIFORM_LINE'][ikey] != genrayhc[lshots[ishot+1]]['TEMTAB_NONUNIFORM_LINE'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['DENTAB_NONUNIFORM_LINE'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['DENTAB_NONUNIFORM_LINE'][ikey] != genrayhc[lshots[ishot+1]]['DENTAB_NONUNIFORM_LINE'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['ECCONE'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['ECCONE'][ikey] != genrayhc[lshots[ishot+1]]['ECCONE'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['GRILL'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['GRILL'][ikey] != genrayhc[lshots[ishot+1]]['GRILL'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['SPECIES'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['SPECIES'][ikey] != genrayhc[lshots[ishot+1]]['SPECIES'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['VARDEN'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['VARDEN'][ikey] != genrayhc[lshots[ishot+1]]['VARDEN'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['DENPROF'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['DENPROF'][ikey] != genrayhc[lshots[ishot+1]]['DENPROF'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['TPOPPROF'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['TPOPPROF'][ikey] != genrayhc[lshots[ishot+1]]['TPOPPROF'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['VFLPROF'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['VFLPROF'][ikey] != genrayhc[lshots[ishot+1]]['VFLPROF'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['ZPROF'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['ZPROF'][ikey] != genrayhc[lshots[ishot+1]]['ZPROF'][ikey]:
                     genrayhcdiff.append(ikey)
            for ikey in genrayhc[lshots[ishot]]['TPROF'].keys():
                if   ikey in genrayhcdiff: continue
                elif genrayhc[lshots[ishot]]['TPROF'][ikey] != genrayhc[lshots[ishot+1]]['TPROF'][ikey]:
                     genrayhcdiff.append(ikey)




    report_title = "FASTRAN Report"
    reportpath = "./fastran_report/"
    if not os.path.isdir(reportpath):
       os.system('mkdir '+reportpath)

    figurepath = os.path.abspath(".")+"/fastran_report/Figures/"
    if not os.path.isdir(figurepath):
       os.system('mkdir '+figurepath)

    if 'noplot' in reportparam and reportparam['noplot']:
        print(CYELLOW + "WARNING: NO PLOTS CREATED" + CEND)
    else:
        fastrandata,fastranplot = fastran_plot(WORK_DIR=fastranfpath)

    texfname = 'fastran_report.tex'
    texfhand = open(reportpath+texfname,'w')
    texfhand.write("\\documentclass[]{book} \n")
    texfhand.write("\\usepackage{float} \n")    
    texfhand.write("\\usepackage{amsmath} \n")    
    texfhand.write("\\usepackage{placeins} \n")    
    texfhand.write("\\usepackage{graphicx} \n")    
    texfhand.write("\\usepackage{colortbl} \n")    
    texfhand.write("\\usepackage{makecell} \n")    
    texfhand.write("\\usepackage{geometry} \n")    
    texfhand.write("\\geometry{legalpaper, landscape, margin=0.7in} \n")    
    texfhand.write("\\usepackage{multirow,tabularx} \n")    
    texfhand.write("\\setcellgapes{4pt} \n")    

    texfhand.write("\\usepackage{scrpage2} \n")
    texfhand.write("\\pagestyle{scrheadings} \n")
    texfhand.write("\\clearscrheadfoot{} \n")
    texfhand.write("\\ohead{\\rightmark} \n")
    texfhand.write("\\cfoot[\\pagemark]{\\pagemark} \n")

    texfhand.write("\n")    
    textitle = "Reporting FASTRAN Results \\\\ "
    for ishot in lshots:
        textitle += ishot+" \\\\ "
    texfhand.write("\\title{%s} \n" % textitle)

    texfhand.write("\n")    
    texfhand.write("\\begin{document}\n")
    texfhand.write("\\maketitle \n")

    texfhand.write("\n")    
    texfhand.write("\\chapter{Input Files Parameters} \n")

    texfhand.write("\\section{LIST OF FASTRAN INPUTS} \n")

    texfhand.write("\\begin{center} \n")
    for SHOT_ID in lshots:
        if SHOT_ID == lshots[0]:
           tblcol = ""
           hdrtxt = ""
           if "dt"              in fastrandiff: hdrtxt += " & dt";                  tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & dt";                  tblcol += "| c "
           if "dt_relax_j"      in fastrandiff: hdrtxt += " & dt\_relax\_j";        tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & dt\_relax\_j";        tblcol += "| c "
           if "dt_step_relax_j" in fastrandiff: hdrtxt += " & dt\_step\_relax\_j";  tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & dt\_step\_relax\_j";  tblcol += "| c "
           if "iexb"            in fastrandiff: hdrtxt += " & iexb";                tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & iexb";                tblcol += "| c "
           if "irelax"          in fastrandiff: hdrtxt += " & irelax";              tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & irelax";              tblcol += "| c "
           if "isolver"         in fastrandiff: hdrtxt += " & isolver";             tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & isolver";             tblcol += "| c "
           if "relax_j"         in fastrandiff: hdrtxt += " & relax\_j";            tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & relax\_j";            tblcol += "| c "
           if "error"           in fastrandiff: hdrtxt += " & error";               tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & error";               tblcol += "| c "
           if "error_e"         in fastrandiff: hdrtxt += " & error\_e";            tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & error\_e";            tblcol += "| c "
           if "error_i"         in fastrandiff: hdrtxt += " & error\_i";            tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & error\_i";            tblcol += "| c "
           if "error_v"         in fastrandiff: hdrtxt += " & error\_v";            tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & error\_v";            tblcol += "| c "
           if "error_n"         in fastrandiff: hdrtxt += " & error\_n";            tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & error\_n";            tblcol += "| c "
           if "nrho"            in fastrandiff: hdrtxt += " & nrho";                tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & nrho";                tblcol += "| c "
           if "rho_bdry"        in fastrandiff: hdrtxt += " & rho\_bdry";           tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & rho\_bdry";           tblcol += "| c "
           if "include_nhe"     in fastrandiff: hdrtxt += " & include\_nhe";        tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & include\_nhe";        tblcol += "| c "
           if "maxiter_relax"   in fastrandiff: hdrtxt += " & maxiter\_relax";      tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & maxiter\_relax";      tblcol += "| c "
           if "nrho_transport"  in fastrandiff: hdrtxt += " & nrho\_transport";     tblcol += "| >{\columncolor{yellow}}c "
           else:                                hdrtxt += " & nrho\_transport";     tblcol += "| c "
           texfhand.write("\\begin{tabular}{| c " + tblcol + "|}\n")
           texfhand.write("\\hline \n")
           texfhand.write("FASTRAN" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        rectxt = ""
        if "dt"              in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['dt'][0])
        else:                                                         rectxt += " & - "
        if "dt_relax_j"      in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['dt_relax_j'][0])
        else:                                                         rectxt += " & - "
        if "dt_step_relax_j" in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['dt_step_relax_j'][0])
        else:                                                         rectxt += " & - "
        if "iexb"            in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['iexb'][0])
        else:                                                         rectxt += " & - "
        if "irelax"          in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['irelax'][0])
        else:                                                         rectxt += " & - "
        if "isolver"         in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['isolver'][0])
        else:                                                         rectxt += " & - "
        if "relax_j"         in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['relax_j'][0])
        else:                                                         rectxt += " & - "
        if "error"           in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['error'][0])
        else:                                                         rectxt += " & - "
        if "error_e"         in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['error_e'][0])
        else:                                                         rectxt += " & - "
        if "error_i"         in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['error_i'][0])
        else:                                                         rectxt += " & - "
        if "error_v"         in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['error_v'][0])
        else:                                                         rectxt += " & - "
        if "error_n"         in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['error_n'][0])
        else:                                                         rectxt += " & - "
        if "nrho"            in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['nrho'][0])
        else:                                                         rectxt += " & - "
        if "rho_bdry"        in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['rho_bdry'][0])
        else:                                                         rectxt += " & - "
        if "include_nhe"     in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['include_nhe'][0])
        else:                                                         rectxt += " & - "
        if "maxiter_relax"   in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['maxiter_relax'][0])
        else:                                                         rectxt += " & - "
        if "nrho_transport"  in fastran[SHOT_ID]['infastran'].keys(): rectxt += " & " + str(fastran[SHOT_ID]['infastran']['nrho_transport'][0])
        else:                                                         rectxt += " & - "
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{17}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "taup"            in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & taup";             rectxt += " & " + str(fastran[SHOT_ID]['infastran']['taup'][0])
        if "alpha_e"         in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & alpha\_e";         rectxt += " & " + str(fastran[SHOT_ID]['infastran']['alpha_e'][0])
        if "alpha_i"         in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & alpha\_i";         rectxt += " & " + str(fastran[SHOT_ID]['infastran']['alpha_i'][0])
        if "alpha_v"         in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & alpha\_v";         rectxt += " & " + str(fastran[SHOT_ID]['infastran']['alpha_v'][0])
        if "alpha_n"         in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & alpha\_n";         rectxt += " & " + str(fastran[SHOT_ID]['infastran']['alpha_n'][0])
        if "solve_ne"        in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & solve\_ne";        rectxt += " & " + str(fastran[SHOT_ID]['infastran']['solve_ne'][0])
        if "alpha_chi"       in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & alpha\_chi";       rectxt += " & " + str(fastran[SHOT_ID]['infastran']['alpha_chi'][0])
        if "solve_v"         in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & solve\_v";         rectxt += " & " + str(fastran[SHOT_ID]['infastran']['solve_v'][0])
        if "solve_j"         in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & solve\_j";         rectxt += " & " + str(fastran[SHOT_ID]['infastran']['solve_j'][0])
        if "solve_te"        in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & solve\_te";        rectxt += " & " + str(fastran[SHOT_ID]['infastran']['solve_te'][0])
        if "solve_ti"        in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & solve\_ti";        rectxt += " & " + str(fastran[SHOT_ID]['infastran']['solve_ti'][0])
        if "solve_mhd"       in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & solve\_mhd";       rectxt += " & " + str(fastran[SHOT_ID]['infastran']['solve_mhd'][0])
        if "model_chi"       in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & model\_chi";       rectxt += " & " + str(fastran[SHOT_ID]['infastran']['model_chi'][0])
        if "model_density"   in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & model\_density";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['model_density'][0])
        if "model_neoclass"  in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & model\_neoclass";  rectxt += " & " + str(fastran[SHOT_ID]['infastran']['model_neoclass'][0])
        if "model_bootstrap" in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & model\_bootstrap"; rectxt += " & " + str(fastran[SHOT_ID]['infastran']['model_bootstrap'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("FASTRAN" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{14}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "multi_neo_n"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & multi\_neo\_e";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['multi_neo_e'][0])
        if "multi_neo_i"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & multi\_neo\_i";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['multi_neo_i'][0])
        if "multi_neo_v"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & multi\_neo\_v";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['multi_neo_v'][0])
        if "multi_neo_n"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & multi\_neo\_n";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['multi_neo_n'][0])
        if "multi_chi_e"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & multi\_chi\_e";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['multi_chi_e'][0])
        if "multi_chi_i"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & multi\_chi\_i";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['multi_chi_i'][0])
        if "multi_chi_v"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & multi\_chi\_v";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['multi_chi_v'][0])
        if "multi_chi_n"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & multi\_chi\_n";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['multi_chi_n'][0])
        if "const_chi_e"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & const\_chi\_e";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['const_chi_e'][0])
        if "const_chi_i"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & const\_chi\_i";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['const_chi_i'][0])
        if "const_chi_v"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & const\_chi\_v";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['const_chi_v'][0])
        if "const_chi_n"     in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & const\_chi\_n";   rectxt += " & " + str(fastran[SHOT_ID]['infastran']['const_chi_n'][0])
        if "include_fusion"  in fastran[SHOT_ID]['infastran'].keys(): hdrtxt += " & include\_fusion"; rectxt += " & " + str(fastran[SHOT_ID]['infastran']['include_fusion'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("FASTRAN" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\section{LIST OF NUBEAM INPUTS} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{13}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "WGHTA"         in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & WGHTA";          rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['WGHTA'][0])
        if "NDEP0"         in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NDEP0";          rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NDEP0'][0])
        if "NSEED"         in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NSEED";          rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NSEED'][0])
        if "NZONES"        in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NZONES";         rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NZONES'][0])
        if "NZNBMA"        in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NZNBMA";         rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NZNBMA'][0])
        if "NZNBME"        in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NZNBME";         rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NZNBME'][0])
        if "NPTCLS"        in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NPTCLS";         rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NPTCLS'][0])
        if "NSDBGB"        in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NSDBGB";         rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NSDBGB'][0])
        if "NMCURB"        in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NMCURB";         rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NMCURB'][0])
        if "NSIGEXC"       in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NSIGEXC";        rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NSIGEXC'][0])
        if "NZONE_FB"      in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NZONE\_FB";      rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NZONE_FB'][0])
        if "NLTEST_OUTPUT" in nubeam[SHOT_ID]['NBI_INIT'].keys(): hdrtxt += " & NLTEST\_OUTPUT"; rectxt += " & " + str(nubeam[SHOT_ID]['NBI_INIT']['NLTEST_OUTPUT'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_INIT" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{6}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NKDIFB"   in nubeam[SHOT_ID]['NBI_MODEL'].keys(): hdrtxt += " & NKDIFB";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_MODEL']['NKDIFB'][0])
        if "DIFB_0"   in nubeam[SHOT_ID]['NBI_MODEL'].keys(): hdrtxt += " & DIFB\_0";   rectxt += " & " + str(nubeam[SHOT_ID]['NBI_MODEL']['DIFB_0'][0])
        if "DIFB_A"   in nubeam[SHOT_ID]['NBI_MODEL'].keys(): hdrtxt += " & DIFB\_A";   rectxt += " & " + str(nubeam[SHOT_ID]['NBI_MODEL']['DIFB_A'][0])
        if "DIFB_IN"  in nubeam[SHOT_ID]['NBI_MODEL'].keys(): hdrtxt += " & DIFB\_IN";  rectxt += " & " + str(nubeam[SHOT_ID]['NBI_MODEL']['DIFB_IN'][0])
        if "DIFB_OUT" in nubeam[SHOT_ID]['NBI_MODEL'].keys(): hdrtxt += " & DIFB\_OUT"; rectxt += " & " + str(nubeam[SHOT_ID]['NBI_MODEL']['DIFB_OUT'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_MODEL" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{6}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NBEAM"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys(): hdrtxt += " & NBEAM";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0])
        nbeam = nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0]
        
        if "ABEAMA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & ABEAMA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['ABEAMA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['ABEAMA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['ABEAMA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['ABEAMA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n" 

        if "XZBEAMA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & XZBEAMA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZBEAMA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZBEAMA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZBEAMA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZBEAMA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n" 

        if "NLCO"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & NLCO"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NLCO'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NLCO'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NLCO'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NLCO'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n" 

        if "NBSHAPA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & NBSHAPA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NBSHAPA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NBSHAPA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NBSHAPA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NBSHAPA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n" 

        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_CONFIG" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
        texfhand.write("\\hline \n")
        
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")


    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{6}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NBEAM"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys(): hdrtxt += " & NBEAM";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0])
        nbeam = nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0]

        if "BMWIDRA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & BMWIDRA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['BMWIDRA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['BMWIDRA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['BMWIDRA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['BMWIDRA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"               

        if "BMWIDZA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & BMWIDRA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['BMWIDZA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['BMWIDZA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['BMWIDZA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['BMWIDZA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"               

        if "FOCLRA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & FOCLRA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FOCLRA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FOCLRA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FOCLRA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FOCLRA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"               

        if "FOCLZA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & FOCLZA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FOCLZA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FOCLZA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FOCLZA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FOCLZA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"               

        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_CONFIG" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
        texfhand.write("\\hline \n")

    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{4}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NBEAM"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys(): hdrtxt += " & NBEAM";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0])
        nbeam = nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0]

        if "DIVRA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & DIVRA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['DIVRA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['DIVRA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['DIVRA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['DIVRA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"               

        if "DIVZA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & DIVZA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['DIVZA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['DIVZA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['DIVZA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['DIVZA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ "               

        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_CONFIG" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
        texfhand.write("\\hline \n")

    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")


    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{4}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NBEAM"   in  nubeam[SHOT_ID]['NBI_CONFIG'].keys(): hdrtxt += " & NBEAM";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0])
        nbeam = nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0]

        if "RTCENA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & RTCENA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['RTCENA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['RTCENA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['RTCENA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['RTCENA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"

        if "NBAPSHA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & NBAPSHA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NBAPSHA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NBAPSHA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NBAPSHA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['NBAPSHA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ "

        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_CONFIG" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
        texfhand.write("\\hline \n")

    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{4}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NBEAM"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys(): hdrtxt += " & NBEAM";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0])
        nbeam = nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0]

        if "XLBTNA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & XLBTNA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XLBTNA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XLBTNA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XLBTNA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XLBTNA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"

        if "XYBSCA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & XYBSCA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XYBSCA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XYBSCA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XYBSCA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XYBSCA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ "

        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_CONFIG" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
        texfhand.write("\\hline \n")

    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{4}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NBEAM"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys(): hdrtxt += " & NBEAM";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0])
        nbeam = nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0]

        if "XLBAPA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & XLBAPA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XLBAPA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XLBAPA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XLBAPA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XLBAPA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"

        if "XYBAPA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & XYBAPA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XYBAPA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XYBAPA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XYBAPA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XYBAPA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ "

        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_CONFIG" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
        texfhand.write("\\hline \n")

    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{4}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NBEAM"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys(): hdrtxt += " & NBEAM";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0])
        nbeam = nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0]

        if "RAPEDGA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & RAPEDGA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['RAPEDGA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['RAPEDGA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['RAPEDGA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['RAPEDGA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"

        if "XZPEDGA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & XZPEDGA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZPEDGA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZPEDGA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZPEDGA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZPEDGA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ "

        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_CONFIG" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
        texfhand.write("\\hline \n")

    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{4}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NBEAM"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys(): hdrtxt += " & NBEAM";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0])
        nbeam = nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0]

        if "XRAPOFFA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & XRAPOFFA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XRAPOFFA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XRAPOFFA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XRAPOFFA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XRAPOFFA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"

        if "XZAPOFFA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & XZAPOFFA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZAPOFFA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZAPOFFA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZAPOFFA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['XZAPOFFA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ "

        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_CONFIG" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
        texfhand.write("\\hline \n")

    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{5}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NBEAM"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys(): hdrtxt += " & NBEAM";    rectxt += " & " + str(nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0])
        nbeam = nubeam[SHOT_ID]['NBI_CONFIG']['NBEAM'][0]

        if "EINJA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & EINJA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['EINJA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['EINJA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['EINJA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['EINJA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"

        if "FFULLA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & FFULLA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FFULLA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FFULLA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FFULLA'][irec+2]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FFULLA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ \n"

        if "FHALFA"   in nubeam[SHOT_ID]['NBI_CONFIG'].keys():
           hdrtxt += " & FHALFA"
           rectxt += " & $\\begin{matrix} \n"
           for irec in range(0,nbeam,4):
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FHALFA'][irec+0]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FHALFA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FHALFA'][irec+1]) + " & "
               rectxt += str(nubeam[SHOT_ID]['NBI_CONFIG']['FHALFA'][irec+3]) + "\\\\ \n"
           rectxt += "\\end{matrix}$ "

        if SHOT_ID == lshots[0]:
           texfhand.write("NBI\_CONFIG" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
        texfhand.write("\\hline \n")

    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\section{LIST OF NFREYA INPUTS} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{19}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "RIN"     in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & rin";     rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['RIN'][0])
        if "ROUT"    in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & rout";    rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['ROUT'][0])
        if "BPTOR"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bptor";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BPTOR'][0])
        if "EBKEV"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & ebkev";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['EBKEV'][0])
        if "FBCUR"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & fbcur";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['FBCUR'][0])
        if "BLENI"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bleni";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BLENI'][0])
        if "BLENP"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & blenp";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BLENP'][0])
        if "NBEAMS"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & nbeams";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['NBEAMS'][0])
        if "RMAJOR"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & rmajor";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['RMAJOR'][0])
        if "RMINOR"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & rminor";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['RMINOR'][0])
        if "ANGLEV"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & anglev";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['ANGLEV'][0])
        if "ANGLEH"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & angleh";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['ANGLEH'][0])
        if "NSOURC"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & nsourc";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['NSOURC'][0])
        if "SFRAC1"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & sfrac1";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['SFRAC1'][0])
        if "RPIVOT"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & rpivot";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['RPIVOT'][0])
        if "ZPIVOT"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & zpivot";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['ZPIVOT'][0])
        if "BVOFSET" in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bvofset"; rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BVOFSET'][0])
        if "BHOFSET" in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bhofset"; rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BHOFSET'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("NFREYA" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{17}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "MF"               in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & mf";                rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['MF'][0])
        if "NPART"            in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & npart";             rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['NPART'][0])
        if "NAMEB"            in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & nameb";             rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['NAMEB'][0])
        if "BTIME"            in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & btime";             rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BTIME'][0])
        if "NPSKIP"           in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & npskip";            rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['NPSKIP'][0])
        if "IEXCIT"           in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & iexcit";            rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['IEXCIT'][0])
        if "IZSTRP"           in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & izstrp";            rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['IZSTRP'][0])
        if "RELNUB"           in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & relnub";            rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['RELNUB'][0])
        if "IBSLOW"           in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & ibslow";            rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['IBSLOW'][0])
        if "RTSTCX"           in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & rtstcx";            rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['RTSTCX'][0])
        if "BEAMON"           in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & beamon";            rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BEAMON'][0])
        if "FDBEAM"           in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & fdbeam";            rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['FDBEAM'][0])
        if "TIMBPLT"          in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & timbplt";           rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['TIMBPLT'][0])
        if "NBEAMTCX"         in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & nbeamtcx";          rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['NBEAMTCX'][0])
        if "ITERATE_BEAM"     in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & iterate\_beam";     rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['ITERATE_BEAM'][0])
        if "FAST_ION_TARGET"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & fast\_ion\_target"; rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['FAST_ION_TARGET'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("NFREYA" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{13}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "ALEN"    in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & alen";    rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['ALEN'][0])
        if "NAPTR"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & naptr";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['NAPTR'][0])
        if "BHFOC"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bhfoc";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BHFOC'][0])
        if "BVFOC"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bvfoc";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BVFOC'][0])
        if "BHDIV"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bhdiv";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BHDIV'][0])
        if "BVDIV"   in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bvdiv";   rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BVDIV'][0])
        if "AHEIGH"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & aheigh";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['AHEIGH'][0])
        if "AWIDTH"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & awidth";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['AWIDTH'][0])
        if "BHEIGH"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bheigh";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BHEIGH'][0])
        if "BWIDTH"  in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & bwidth";  rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['BWIDTH'][0])
        if "NASHAPE" in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & nashape"; rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['NASHAPE'][0])
        if "NBSHAPE" in nfreya[SHOT_ID]['innfreya'].keys(): hdrtxt += " & nbshape"; rectxt += " & " + str(nfreya[SHOT_ID]['innfreya']['NBSHAPE'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("NFREYA" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\section{LIST OF HCD INPUTS} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{9}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "PE"        in hcd[SHOT_ID]['inhcd'].keys(): hdrtxt += " & Pe";         rectxt += " & " + str(hcd[SHOT_ID]['inhcd']['PE'][0])
        if "PI"        in hcd[SHOT_ID]['inhcd'].keys(): hdrtxt += " & Pi";         rectxt += " & " + str(hcd[SHOT_ID]['inhcd']['PI'][0])
        if "NSRC"      in hcd[SHOT_ID]['inhcd'].keys(): hdrtxt += " & nsrc";       rectxt += " & " + str(hcd[SHOT_ID]['inhcd']['NSRC'][0])
        if "XMID"      in hcd[SHOT_ID]['inhcd'].keys(): hdrtxt += " & xmid";       rectxt += " & " + str(hcd[SHOT_ID]['inhcd']['XMID'][0])
        if "XWID"      in hcd[SHOT_ID]['inhcd'].keys(): hdrtxt += " & xwid";       rectxt += " & " + str(hcd[SHOT_ID]['inhcd']['XWID'][0])
        if "J0_SEED"   in hcd[SHOT_ID]['inhcd'].keys(): hdrtxt += " & j0\_seed";   rectxt += " & " + str(hcd[SHOT_ID]['inhcd']['J0_SEED'][0])
        if "X0_SEED"   in hcd[SHOT_ID]['inhcd'].keys(): hdrtxt += " & x0\_seed";   rectxt += " & " + str(hcd[SHOT_ID]['inhcd']['X0_SEED'][0])
        if "DRHO_SEED" in hcd[SHOT_ID]['inhcd'].keys(): hdrtxt += " & drho\_seed"; rectxt += " & " + str(hcd[SHOT_ID]['inhcd']['DRHO_SEED'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("HCD" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\section{LIST OF GENRAY\_HC INPUTS} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{8}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "B0"            in genrayhc[SHOT_ID]['GENR'].keys(): hdrtxt += " & B0";             rectxt += " & " + str(genrayhc[SHOT_ID]['GENR']['B0'][0])
        if "R0X"           in genrayhc[SHOT_ID]['GENR'].keys(): hdrtxt += " & R0X";            rectxt += " & " + str(genrayhc[SHOT_ID]['GENR']['R0X'][0])
        if "STAT"          in genrayhc[SHOT_ID]['GENR'].keys(): hdrtxt += " & STAT";           rectxt += " & " + str(genrayhc[SHOT_ID]['GENR']['STAT'][0])
        if "RAYOP"         in genrayhc[SHOT_ID]['GENR'].keys(): hdrtxt += " & RAYOP";          rectxt += " & " + str(genrayhc[SHOT_ID]['GENR']['RAYOP'][0])
        if "OUTDAT"        in genrayhc[SHOT_ID]['GENR'].keys(): hdrtxt += " & OUTDAT";         rectxt += " & " + str(genrayhc[SHOT_ID]['GENR']['OUTDAT'][0])
        if "MNEMONIC"      in genrayhc[SHOT_ID]['GENR'].keys(): hdrtxt += " & MNEMONIC";       rectxt += " & " + str(genrayhc[SHOT_ID]['GENR']['MNEMONIC'][0])
        if "DIELECTRIC_OP" in genrayhc[SHOT_ID]['GENR'].keys(): hdrtxt += " & DIELECTRIC\_OP"; rectxt += " & " + str(genrayhc[SHOT_ID]['GENR']['DIELECTRIC_OP'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("GENR" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{10}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "IPSI"     in genrayhc[SHOT_ID]['TOKAMAK'].keys(): hdrtxt += " & IPSI";      rectxt += " & " + str(genrayhc[SHOT_ID]['TOKAMAK']['IPSI'][0])
        if "NLOOP"    in genrayhc[SHOT_ID]['TOKAMAK'].keys(): hdrtxt += " & NLOOP";     rectxt += " & " + str(genrayhc[SHOT_ID]['TOKAMAK']['NLOOP'][0])
        if "IEFFIC"   in genrayhc[SHOT_ID]['TOKAMAK'].keys(): hdrtxt += " & IEFFIC";    rectxt += " & " + str(genrayhc[SHOT_ID]['TOKAMAK']['IEFFIC'][0])
        if "EQDSKIN"  in genrayhc[SHOT_ID]['TOKAMAK'].keys(): hdrtxt += " & EQDSKIN";   rectxt += " & " + str(genrayhc[SHOT_ID]['TOKAMAK']['EQDSKIN'][0])
        if "IONETWO"  in genrayhc[SHOT_ID]['TOKAMAK'].keys(): hdrtxt += " & IONETWO";   rectxt += " & " + str(genrayhc[SHOT_ID]['TOKAMAK']['IONETWO'][0])
        if "INDEXRHO" in genrayhc[SHOT_ID]['TOKAMAK'].keys(): hdrtxt += " & INDEXRHO";  rectxt += " & " + str(genrayhc[SHOT_ID]['TOKAMAK']['INDEXRHO'][0])
        if "PSIFACTR" in genrayhc[SHOT_ID]['TOKAMAK'].keys(): hdrtxt += " & PSIFACTR";  rectxt += " & " + str(genrayhc[SHOT_ID]['TOKAMAK']['PSIFACTR'][0])
        if "DELTRIPL" in genrayhc[SHOT_ID]['TOKAMAK'].keys(): hdrtxt += " & DELTRIPL";  rectxt += " & " + str(genrayhc[SHOT_ID]['TOKAMAK']['DELTRIPL'][0])
        if "I_RIPPLE" in genrayhc[SHOT_ID]['TOKAMAK'].keys(): hdrtxt += " & I\_RIPPLE"; rectxt += " & " + str(genrayhc[SHOT_ID]['TOKAMAK']['I_RIPPLE'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("TOKAMAK" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{11}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "IBW"         in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & IBW";           rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['IBW'][0])
        if "IOXM"        in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & IOXM";          rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['IOXM'][0])
        if "JWAVE"       in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & JWAVE";         rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['JWAVE'][0])
        if "ISTART"      in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & ISTART";        rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['ISTART'][0])
        if "FRQNCY"      in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & FRQNCY";        rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['FRQNCY'][0])
        if "IREFLM"      in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & IREFLM";        rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['IREFLM'][0])
        if "DELPWRMN"    in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & DELPWRMN";      rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['DELPWRMN'][0])
        if "I_VGR_INI"   in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & I\_VGR\_INI";   rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['I_VGR_INI'][0])
        if "POLDIST_MX"  in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & POLDIST\_MX";   rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['POLDIST_MX'][0])
        if "IOXM_N_NPAR" in genrayhc[SHOT_ID]['WAVE'].keys(): hdrtxt += " & IOXM\_N\_NPAR"; rectxt += " & " + str(genrayhc[SHOT_ID]['WAVE']['IOXM_N_NPAR'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("WAVE" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{16}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "IB"            in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & IB";             rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['IB'][0])
        if "ID"            in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & ID";             rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['ID'][0])
        if "JY_D"          in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & JY\_D";          rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['JY_D'][0])
        if "DEL_Y"         in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & DEL\_Y";         rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['DEL_Y'][0])
        if "IHERM"         in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & IHERM";          rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['IHERM'][0])
        if "IFLUX"         in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & IFLUX";          rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['IFLUX'][0])
        if "ISWITCH"       in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & ISWITCH";        rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['ISWITCH'][0])
        if "IABSORP"       in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & IABSORP";        rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['IABSORP'][0])
        if "IDSWITCH"      in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & IDSWITCH";       rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['IDSWITCH'][0])
        if "IABSWITCH"     in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & IABSWITCH";      rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['IABSWITCH'][0])
        if "I_IM_NPERP"    in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & I\_IM\_NPERP";   rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['I_IM_NPERP'][0])
        if "N_RELT_HARM"   in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & N\_RELT\_HARM";  rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['N_RELT_HARM'][0])
        if "N_RELT_INTGR"  in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & N\_RELT\_INTGR"; rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['N_RELT_INTGR'][0])
        if "I_GEOM_OPTIC"  in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & I\_GEOM\_OPTIC"; rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['I_GEOM_OPTIC'][0])
        if "RAY_DIRECTION" in genrayhc[SHOT_ID]['DISPERS'].keys(): hdrtxt += " & RAY\_DIRECTION"; rectxt += " & " + str(genrayhc[SHOT_ID]['DISPERS']['RAY_DIRECTION'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("DISPERS" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{13}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "IRKMETH"     in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & IRKMETH";      rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['IRKMETH'][0])
        if "NDIM1"       in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & NDIM1";        rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['NDIM1'][0])
        if "ISOLV"       in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & ISOLV";        rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['ISOLV'][0])
        if "IDIF"        in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & IDIF";         rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['IDIF'][0])
        if "NRELT"       in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & NRELT";        rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['NRELT'][0])
        if "PRMT1"       in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & PRMT1";        rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['PRMT1'][0])
        if "PRMT2"       in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & PRMT2";        rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['PRMT2'][0])
        if "PRMT3"       in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & PRMT3";        rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['PRMT3'][0])
        if "PRMT4"       in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & PRMT4";        rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['PRMT4'][0])
        if "PRMT6"       in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & PRMT6";        rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['PRMT6'][0])
        if "ICORRECT"    in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & ICORRECT";     rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['ICORRECT'][0])
        if "MAXSTEPS_RK" in genrayhc[SHOT_ID]['NUMERCL'].keys(): hdrtxt += " & MAXSTEPS\_RK"; rectxt += " & " + str(genrayhc[SHOT_ID]['NUMERCL']['MAXSTEPS_RK'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("NUMERCL" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{7}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "IWJ"      in genrayhc[SHOT_ID]['OUTPUT'].keys(): hdrtxt += " & IWJ";        rectxt += " & " + str(genrayhc[SHOT_ID]['OUTPUT']['IWJ'][0])
        if "IWCNTR"   in genrayhc[SHOT_ID]['OUTPUT'].keys(): hdrtxt += " & IWCNTR";     rectxt += " & " + str(genrayhc[SHOT_ID]['OUTPUT']['IWCNTR'][0])
        if "IWOPEN"   in genrayhc[SHOT_ID]['OUTPUT'].keys(): hdrtxt += " & IWOPEN";     rectxt += " & " + str(genrayhc[SHOT_ID]['OUTPUT']['IWOPEN'][0])
        if "ITOOLS"   in genrayhc[SHOT_ID]['OUTPUT'].keys(): hdrtxt += " & ITOOLS";     rectxt += " & " + str(genrayhc[SHOT_ID]['OUTPUT']['ITOOLS'][0])
        if "I_PLOT_B" in genrayhc[SHOT_ID]['OUTPUT'].keys(): hdrtxt += " & I\_PLOT\_D"; rectxt += " & " + str(genrayhc[SHOT_ID]['OUTPUT']['I_PLOT_B'][0])
        if "I_PLOT_D" in genrayhc[SHOT_ID]['OUTPUT'].keys(): hdrtxt += " & I\_PLOT\_D"; rectxt += " & " + str(genrayhc[SHOT_ID]['OUTPUT']['I_PLOT_D'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("OUTPUT" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{7}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NDENS"       in genrayhc[SHOT_ID]['PLASMA'].keys(): hdrtxt += " & NDENS";       rectxt += " & " + str(genrayhc[SHOT_ID]['PLASMA']['NDENS'][0])
        if "NBULK"       in genrayhc[SHOT_ID]['PLASMA'].keys(): hdrtxt += " & NBULK";       rectxt += " & " + str(genrayhc[SHOT_ID]['PLASMA']['NBULK'][0])
        if "IZEFF"       in genrayhc[SHOT_ID]['PLASMA'].keys(): hdrtxt += " & IZEFF";       rectxt += " & " + str(genrayhc[SHOT_ID]['PLASMA']['IZEFF'][0])
        if "IDENS"       in genrayhc[SHOT_ID]['PLASMA'].keys(): hdrtxt += " & IDENS";       rectxt += " & " + str(genrayhc[SHOT_ID]['PLASMA']['IDENS'][0])
        if "DEN_SCALE"   in genrayhc[SHOT_ID]['PLASMA'].keys(): hdrtxt += " & DEN\_SCALE";  rectxt += " & " + str(genrayhc[SHOT_ID]['PLASMA']['DEN_SCALE'][0])
        if "TEMP_SCALE"  in genrayhc[SHOT_ID]['PLASMA'].keys(): hdrtxt += " & TEMP\_SCALE"; rectxt += " & " + str(genrayhc[SHOT_ID]['PLASMA']['TEMP_SCALE'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("PLASMA" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")

    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{| *{18}{c |}}\n")
    texfhand.write("\\hline \n")
    for SHOT_ID in lshots:
        hdrtxt = ""
        rectxt = ""
        if "NTHIN"        in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & NTHIN";          rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['NTHIN'][0])
        if "ANMIN"        in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & ANMIN";          rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['ANMIN'][0])
        if "ANMAX"        in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & ANMAX";          rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['ANMAX'][0])
        if "NNKPAR"       in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & NNKPAR";         rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['NNKPAR'][0])
        if "POWERS"       in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & POWERS";         rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['POWERS'][0])
        if "NNKPOL"       in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & NNKPOL";         rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['NNKPOL'][0])
        if "NGRILL"       in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & NGRILL";         rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['NGRILL'][0])
        if "HEIGHT"       in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & HEIGHT";         rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['HEIGHT'][0])
        if "THGRILL"      in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & THGRILL";        rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['THGRILL'][0])
        if "RHOPSI0"      in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & RHOPSI0";        rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['RHOPSI0'][0])
        if "ANPOLMIN"     in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & ANPOLMIN";       rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['ANPOLMIN'][0])
        if "IGRILLPW"     in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & IGRILLPW";       rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['IGRILLPW'][0])
        if "PHIGRILL"     in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & PHIGRILL";       rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['PHIGRILL'][0])
        if "ANPOLMAX"     in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & ANPOLMAX";       rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['ANPOLMAX'][0])
        if "IGRILLTW"     in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & IGRILLTW";       rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['IGRILLTW'][0])
        if "I_N_POLOIDAL" in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & I\_N\_POLOIDAL"; rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['I_N_POLOIDAL'][0])
        if "I_RHO_CUTOFF" in genrayhc[SHOT_ID]['GRILL'].keys(): hdrtxt += " & I\_RHO\_CUTOFF"; rectxt += " & " + str(genrayhc[SHOT_ID]['GRILL']['I_RHO_CUTOFF'][0])
        if SHOT_ID == lshots[0]:
           texfhand.write("GRILL" + hdrtxt + "  \\\\ \n")
           texfhand.write("\\hline \n")
        texfhand.write(SHOT_ID + rectxt + "\\\\ \n")
    texfhand.write("\\hline \n")
    texfhand.write("\\end{tabular}\n")
    texfhand.write("\\end{center} \n")


    texfhand.write("\\clearpage \n")
    texfhand.write("\\section{FASTRAN PLOTS} \n")

    texfhand.write("\\begin{figure}[!h] \n")
    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{c} \n")
    texfhand.write("\\includegraphics[scale=0.80]{"+figurepath+"profiles.png} \n")
    texfhand.write("\\end{tabular} \n")
    texfhand.write("\\end{center} \n")
    texfhand.write("\\end{figure} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\begin{figure}[!h] \n")
    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{c} \n")
    texfhand.write("\\includegraphics[scale=0.80]{"+figurepath+"currents.png} \n")
    texfhand.write("\\end{tabular} \n")
    texfhand.write("\\end{center} \n")
    texfhand.write("\\end{figure} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\begin{figure}[!h] \n")
    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{c} \n")
    texfhand.write("\\includegraphics[scale=0.80]{"+figurepath+"geometry.png} \n")
    texfhand.write("\\end{tabular} \n")
    texfhand.write("\\end{center} \n")
    texfhand.write("\\end{figure} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\begin{figure}[!h] \n")
    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{c} \n")
    texfhand.write("\\includegraphics[scale=0.80]{"+figurepath+"mainheating.png} \n")
    texfhand.write("\\end{tabular} \n")
    texfhand.write("\\end{center} \n")
    texfhand.write("\\end{figure} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\begin{figure}[!h] \n")
    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{c} \n")
    texfhand.write("\\includegraphics[scale=0.80]{"+figurepath+"otherheating.png} \n")
    texfhand.write("\\end{tabular} \n")
    texfhand.write("\\end{center} \n")
    texfhand.write("\\end{figure} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\begin{figure}[!h] \n")
    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{c} \n")
    texfhand.write("\\includegraphics[scale=0.80]{"+figurepath+"heattransport.png} \n")
    texfhand.write("\\end{tabular} \n")
    texfhand.write("\\end{center} \n")
    texfhand.write("\\end{figure} \n")

    texfhand.write("\\clearpage \n")
    texfhand.write("\\begin{figure}[!h] \n")
    texfhand.write("\\begin{center} \n")
    texfhand.write("\\begin{tabular}{c} \n")
    texfhand.write("\\includegraphics[scale=0.80]{"+figurepath+"heatflux.png} \n")
    texfhand.write("\\end{tabular} \n")
    texfhand.write("\\end{center} \n")
    texfhand.write("\\end{figure} \n")

    texfhand.write("\n")    
    texfhand.write("\\end{document} \n")
    texfhand.close()

    print(CYELLOW + 'Processing TEX to PDF ...' + CEND)
   #latex2pdf_cmd = "pdflatex -interaction=batchmode -output-directory=" + reportpath + " " + reportpath + texfname)
    if 'debugtex' in reportparam and reportparam['debugtex']:
        latex2pdf_cmd = "pdflatex -output-directory=" + reportpath + " " + reportpath + texfname
    else:
        latex2pdf_cmd = "pdflatex -output-directory=" + reportpath + " " + reportpath+texfname + " > /dev/null"
    os.system(latex2pdf_cmd)
    print(CGREEN + 'FASTRAN Report Created SUCCESSFULLY' + CEND)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noplot',  '-noplot',  action='store_const',const=1,help='Do not Plot FASTRAN results before creating the report.')
    parser.add_argument('--debugtex','-debugtex',action='store_const',const=1,help='Show details while compiling TEX file to PDF.')
    parser.add_argument('inputs',nargs='*')

    if parser.parse_args():
        args = parser.parse_args()
        inputs   = args.inputs
        noplot   = args.noplot
        debugtex = args.debugtex

    reportparam = {}
    if noplot:
        reportparam['noplot'] = True
    if debugtex:
        reportparam['debugtex'] = True

    fastran_report(fastranfpath=inputs[:],reportparam=reportparam)

    sys.exit()



   #texfhand.write("\\multicolumn{2}{|c|}{General}\\\\ \n")
