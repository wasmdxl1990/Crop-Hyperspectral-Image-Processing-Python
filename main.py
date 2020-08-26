# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:08:28 2020

@author: ma125
"""


#from spectral import imshow
import spectral.io.envi as envi
import numpy as np
#import matplotlib.pyplot as plt
#from PIL import Image, ImageEnhance
#from scipy import ndimage as ndi
#import math
import cv2
import pandas as pd
#from skimage.measure import label, regionprops
import os.path 
from Imagefcs import ImageProcess


InputPath='E:/Hyperspectral_Imaging_Python/image/'
OutputPath='E:/Hyperspectral_Imaging_Python/ProcessedResults/'
threshold=7

imagep=ImageProcess(InputPath,OutputPath,threshold)

#path='E:/Hyperspectral_Imaging_Python/image/'
#file='PS32300.raw'

path, file_list, file_white=imagep.definepath()


rootfolder = os.path.basename(path[:-1])
for i in range(len(file_list)):
    
    file=file_list[i]

    img_raw, wavelengths, spectral=imagep.reshapeImage_modified(path,file)
    rgb=imagep.img2rgb(img_raw,wavelengths)
#    rgb.show()   
    mask=imagep.red_edge_segmentation(img_raw,wavelengths,threshold=7)
#    plt.imshow(mask)
#    plt.show()
        
    f_raw = path + file_white
    f_hdr=f_raw.replace('.raw', '.hdr')
    white_ref = np.array(envi.open(f_hdr,f_raw).load()) 
    
    m,n,o=img_raw.shape #match image size
    a,b,c=white_ref.shape
    
    if [m,n,o]!=[a,b,c]:
        whiteimg=imagep.padimage(white_ref,m,n,o)
    else:
        whiteimg=white_ref
    
#    refimg=np.divide(img_raw,whiteimg)
    refimg=img_raw/whiteimg
    calirgb= imagep.img2rgb(refimg,wavelengths)
#    image_cali_test=refimg[:,:,200]     
    DataCell=imagep.gettable(file,path,refimg,mask,wavelengths,threshold)
    
    
    datefolder = os.path.basename(path[:-1])
    if not os.path.exists(os.path.join(OutputPath, datefolder)):
        os.mkdir(os.path.join(OutputPath, datefolder))
    foldername=os.path.join(OutputPath, datefolder,file[:-4])
    if not os.path.exists(foldername):        
        os.mkdir(foldername)
    
    rgb.save(foldername+'/RGB.png')
    calirgb.save(foldername+'/caliRGB.png')
    cv2.imwrite((foldername+'/mask.png'), mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    #np.save(foldername, refimg)

    if i==0:
        DataTable=DataCell  
    else:
        DataTable = pd.concat([DataTable,DataCell])


tablename=OutputPath+'Table_'+rootfolder+'.csv'
DataTable.to_csv(tablename, index = False)