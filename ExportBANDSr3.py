from __future__ import division
import sys
import gdal
import numpy as np
import pandas as pd
import os

fileList =[]

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


def between(value, a, b):
    # Find and validate before-part.
    pos_a = value.find(a)
    if pos_a == -1: return ""
    # Find and validate after part.
    pos_b = value.rfind(b)
    if pos_b == -1: return ""
    # Return middle part.
    adjusted_pos_a = pos_a + len(a)
    if adjusted_pos_a >= pos_b: return ""
    return value[adjusted_pos_a:pos_b]

import glob

myfileList = []

createFolder('./split/')
createFolder('./split/images')
createFolder('./split/csv')

for filename in glob.glob("./*.tif"):
    myfileList.append(filename)

for file in myfileList:

    img_file = gdal.Open(file,gdal.GA_ReadOnly)
        
    if img_file is None:
        print 'Unable to open IMAGE:' .img_file
        sys.exit(1)

    driver = img_file.GetDriver()
        
    cols =  img_file.RasterXSize
    rows =  img_file.RasterYSize
    bands = img_file.RasterCount

    for band in range(15):
        band += 1

        srcband = img_file.GetRasterBand(band)
        
        Array = img_file.GetRasterBand(band).ReadAsArray()
        
        NDV = 0
        
        if srcband is None:
            continue

        bfname = between(file, "/", ".")
                 
        #print(bfname)
        
        fname = "./split/images/" + str(band) + "-" + bfname + ".tif"
        sname = "./split/csv/" + str(band) + "-" + bfname + ".csv"
        
        Array[np.isnan(Array)] = NDV
        
        outDs = driver.Create(fname,cols,rows,1,gdal.GDT_Float32)
        if outDs is None:
            print 'Could not create ' +fname
            sys.exit(1)
            
        outDs.GetRasterBand(1).WriteArray( Array )
        outDs.GetRasterBand(1).SetNoDataValue(NDV)
        
        outDs = None
               
        oDF = pd.DataFrame(Array)
        oDF.to_csv(sname,mode = 'w',index=False,header=False)

print("Images and csv for Bands exported")