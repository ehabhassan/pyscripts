import os
import sys
import glob

if len(sys.argv) > 2:
    foldername = sys.argv[1]
    shotnumber = sys.argv[2]

    filepath = foldername + "/work/plasma_state/"
    filelist = glob.glob(filepath+'*')

    for ofpath in filelist:
        nfpath = ofpath[:-2] + "%02d" % int(shotnumber)
       #if   len(shotnumber) == 2:
       #     nfpath = ofpath[:-2] + "%02d" % int(shotnumber)
       #elif len(shotnumber) == 1:
       #     nfpath = ofpath[:-1] + "%02d" % int(shotnumber)

        cmd = "mv %s %s" % (ofpath,nfpath)
        os.system(cmd)

