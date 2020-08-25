# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:19:35 2020

@author: ma125
"""

from spectral import imshow, view_cube
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi
import math
import cv2
import pandas as pd
from skimage.measure import label, regionprops
import os.path 
InputPath='E:/Hyperspectral_Imaging_Python/image/'
ReferencePath='E:/Hyperspectral_Imaging_Python/image/'
OutputPath='E:/Hyperspectral_Imaging_Python/image/'
threshold=7



path='E:/Hyperspectral_Imaging_Python/image/'
file='PS32300.raw'

#len(path)


def parseHdrInfo(path,file):
    file=file.replace('.raw', '.hdr')
    f = path + file
    tf = open(f,"r")

    spatial =0                     #Initialize spatial and spectral
    spectral=0
    waveFlag = 0                   #Flag changes to 1 when wavelengths vector is collected
    wavelengths = []    
    
    line=tf.readline()
    while True:
        if (line == '') and (waveFlag == 1):
            break
        else:
            ind = str.find(line, 'samples')    
            if (ind >= 0):
                spatial = float(line[(str.find(line, '=')+2):(len(line)-1)])
            else:
                ind = str.find(line, 'lines') 
                if (ind >= 0):
                   frames=float(line[(str.find(line, '=')+2):(len(line)-1)]) 
                else:
                    ind = str.find(line, 'bands')
                    if ind == 0:
                        spectral = float(line[(str.find(line, '=')+2):(len(line)-1)])
                    else:
                        ind = str.find(line, 'tint')
                        if ind == 0:
                            tint = float(line[(str.find(line, '=')+2):(len(line)-1)])
                        else:
                            ind = str.find(line, 'Wavelength')
                            if (ind >= 0):
                               line=tf.readline() 
                               for k in range(int(spectral)):
                                   wavelengths.append(float(line[:(len(line)-2)]))
                                   #wavelengths[k]=float(line[:(len(line)-2)])
                                   line=tf.readline()
                                   waveFlag =1
        
        line=tf.readline()
    wavelengths = np.sort(wavelengths)            
    return wavelengths, spatial, frames, spectral, tint        

 

def reshapeImage_modified(path,file):
    wavelengths, spatial, frames, spectral, tint =parseHdrInfo(path,file)
    f_raw = path + file
    f_hdr=f_raw.replace('.raw', '.hdr')
    img_raw = np.array(envi.open(f_hdr,f_raw).load())
    return img_raw, wavelengths, spectral


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def image_norm(image_r):
    image_r = (image_r-image_r.min())/(image_r.max()-image_r.min())
    return image_r



def img2rgb(img_raw,wavelengths):
    redband=690
    greenband=540
    blueband=460
    rindex=find_nearest(wavelengths, redband)    
    gindex=find_nearest(wavelengths, greenband) 
    bindex=find_nearest(wavelengths, blueband) 
    image_r=img_raw[:,:,rindex]
    image_g=img_raw[:,:,gindex]
    image_b=img_raw[:,:,bindex]    
    red=Image.fromarray((image_norm(image_r)*256).astype(np.uint8))
    green=Image.fromarray((image_norm(image_g)*256).astype(np.uint8))
    blue=Image.fromarray((image_norm(image_b)*256).astype(np.uint8))
    rgb=Image.merge("RGB",(red,green,blue))
    enhancer = ImageEnhance.Brightness(rgb)    #image brightness enhancer
    rgb_output = enhancer.enhance(factor=30)
#    rgb_output.show()
    return rgb_output


def red_edge_segmentation(img_raw,wavelengths,threshold):
    lin=np.arange(-20, 20, 1)
    m,n,o=img_raw.shape
    t=np.zeros((m,n))
    redband=680
    rindex=find_nearest(wavelengths, redband)  

    for i in range(m):
        t[i,:]=(np.matmul((lin.reshape((-1,1))).T,(np.squeeze(img_raw[i,:,rindex:rindex+40])).T))/np.sum((np.transpose(lin)*lin))
    mask = np.where(t<threshold, 0, t) 
    mask = np.where(mask>=threshold, 1, mask)
    label_objects, nb_labels = ndi.label(mask)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 150  #threshhold for noisy background
    mask_sizes[0] = 0
    seg_cleaned = mask_sizes[label_objects]
    mask=seg_cleaned*mask
    return mask.astype(int)



def padimage(img,m,n,o):
    a,b,c=img_raw.shape
    del1=m-a
    if del1>0:  #smaller than required image size
        if del1 % 2 ==0: 
            img=np.pad(img, (int((del1/2), int(del1/2)),(0,0)), 'edge')
        else:
            img=np.pad(img, ((math.floor(del1/2), math.floor(del1/2)),(0,0)), 'edge')
    elif del1<0: #larger than required image size
        img=img[math.ceil(abs(del1)/2):math.ceil(abs(del1)/2)+m-1,:,:]
        
    del2=n-b
    if del2>0:
        if del2 % 2 ==0:
            img=np.pad(img, ((0,0),(int(del2/2), int(del2/2))), 'edge')
        else:
            img=np.pad(img, ((0,0),(math.floor(del2/2), math.floor(del2/2))), 'edge')
    elif del2<0:
        img=img[math.ceil(abs(del2)/2):math.ceil(abs(del2)/2)+m-1,:,:]            
        
    if c!=o:
        print("Spectral dimension mismatch")
        
    return img




def feature_extraction(mask):
    label_img1= label(mask,connectivity=2,background=0)
    region1 = regionprops(label_img1)   
    N=len(region1)
    Area=[]
    if N==0:
        LeafProp={}
        LeafProp["Area"]=0
        LeafProp["Perimeter"] =0
        LeafProp["MinL"]=0
        LeafProp["MaxL"]=0
        LeafProp["Eccentricity"]=0
        LeafProp["Solidity"]=0
        LeafProp["Orientation"]=0
        LeafProp["Equivalent_diameter"]=0
        LeafProp["Convex_area"]=0
        LeafProp["Bbox_area"]=0        
    else:
        if N >=1:
            for i in range (N):
                Area.append(region1[i].area)                
                index=Area.index(max(Area))    
        LeafProp={}
        LeafProp["Area"]=region1[index].area
        LeafProp["Perimeter"] =region1[index].perimeter
        LeafProp["MinL"]=region1[index].minor_axis_length
        LeafProp["MaxL"]=region1[index].major_axis_length
        LeafProp["Eccentricity"]=region1[index].eccentricity
        LeafProp["Solidity"]=region1[index].solidity
        LeafProp["Orientation"]=region1[index].orientation
        LeafProp["Equivalent_diameter"]=region1[index].equivalent_diameter
        LeafProp["Convex_area"]=region1[index].convex_area
        LeafProp["Bbox_area"]=region1[index].bbox_area
    return LeafProp


def pixelNDVI(intensity,wavelengths):
    nirband=800
    redband=680
    rindex=find_nearest(wavelengths, redband)    
    nirindex=find_nearest(wavelengths, nirband) 
    intensity_r=intensity[rindex]
    intensity_nir=intensity[nirindex]
    NDVI=(intensity_nir-intensity_r)/(intensity_nir+intensity_r)
    return NDVI

def pixelSIPI(intensity,wavelengths):
    nirband=800
    redband=680
    blueband=445
    rindex=find_nearest(wavelengths, redband)    
    nirindex=find_nearest(wavelengths, nirband)
    bindex=find_nearest(wavelengths, blueband) 
    intensity_r=intensity[rindex]
    intensity_nir=intensity[nirindex]
    intensity_b=intensity[bindex]    
    SIPI=(intensity_nir-intensity_b)/(intensity_nir+intensity_r)
    return SIPI



def gettable(file,path,refimg,mask,wavelengths,threshold):
    rootfolder = os.path.basename(path[:-1]) 
    LeafProp=feature_extraction(mask)
    columnlist=['Folder','Filename','Threshold','LeafArea','NDVI','SIPI']+(wavelengths.tolist())
    # create a dictionary of column names and the value you want
    df_image = pd.DataFrame(np.zeros((1, len(columnlist))),columns=columnlist)
    df_image['Folder']= rootfolder      
    df_image['Filename']= file
    df_image['Threshold']= threshold
    df_image['LeafArea']= LeafProp["Area"]    
    intensity=[]
    maskindex = mask ==0
    refimg[maskindex]=0
    for i in range(len(wavelengths)):
        intensity.append(np.mean(refimg[:,:,i]))
    df_image['NDVI']= pixelNDVI(intensity,wavelengths) 
    df_image['SIPI']= pixelSIPI(intensity,wavelengths)      
    df_image.iloc[0,6:]=intensity
    return df_image





img_raw, wavelengths, spectral=reshapeImage_modified(path,file)

rgb=img2rgb(img_raw,wavelengths)
rgb.show()

mask=red_edge_segmentation(img_raw,wavelengths,threshold=7)
plt.imshow(mask)
plt.show()


f_raw = path + file
f_hdr=f_raw.replace('.raw', '.hdr')
white_ref = np.array(envi.open(f_hdr,f_raw).load())

m,n,o=img_raw.shape #match image size
a,b,c=img_raw.shape



if [m,n,o]==[a,b,c]:
    whiteimg=padimage(white_ref,m,n,o)
else:
    whiteimg=white_ref

refimg=np.divide(img_raw,whiteimg)

calirgb=img2rgb(refimg,wavelengths)    






## create Pillow image
#image2 = Image.fromarray(data)

rgbdata = np.asarray(rgb).copy()
maskindex = mask ==0 
rgbdata[maskindex] = 0
plt.imshow(rgbdata)




for i in range(np.size(InputPath)):
  
    dark_ref = envi.open('E:/Hyperspectral_Imaging_Python/image/dark_reference.raw')
    white_ref = envi.open('E:/Hyperspectral_Imaging_Python/image/white_reference.raw')
    data_ref = envi.open('E:/Hyperspectral_Imaging_Python/image/data_capture.raw')
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    white_nparr = np.array(white_ref.load())
    dark_nparr = np.array(dark_ref.load())
    data_nparr = np.array(data_ref.load()) 
    
    [m,n,o]=np.size(data_nparr)
    [a,b,c]=np.size(white_nparr)
    
    corrected_nparr = np.divide(
    np.subtract(data_nparr, dark_nparr),
    np.subtract(white_nparr, dark_nparr))
    
    