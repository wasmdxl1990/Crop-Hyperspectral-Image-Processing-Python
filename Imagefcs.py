# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:19:35 2020

@author: ma125
"""

from spectral import imshow
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi
import math
import cv2
import pandas as pd
from skimage.measure import label, regionprops
import os.path 


InputPath='F:/Phenotyping_Lab/HSI_Raw/'
OutputPath='F:/Phenotyping_Lab/HSI_Raw/ProcessedResults/'
threshold=7

#len(path)
class ImageProcess():
    def __init__(self, InputPath, OutputPath, threshold):
        self.InputPath = InputPath
        self.OutputPath = OutputPath
        self.threshold = threshold

    def definepath(self):
        file_list = [f for f in os.listdir(self.InputPath) if os.path.isfile(os.path.join(self.InputPath, f)) and f.endswith('.raw')]
        file_white = [f for f in os.listdir(self.InputPath+'white') if os.path.isfile(os.path.join(self.InputPath+'white', f)) and f.endswith('.raw')]    
        file_white_name=('white/'+file_white[0])
        return self.InputPath, file_list,file_white_name  

    
    def parseHdrInfo(self,path,file):
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


 

    def reshapeImage_modified(self,path,file):
        wavelengths, spatial, frames, spectral, tint =self.parseHdrInfo(path,file)
        f_raw = path + file
        f_hdr=f_raw.replace('.raw', '.hdr')
        img_raw = np.array(envi.open(f_hdr,f_raw).load())
        return img_raw, wavelengths, spectral


    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


    def image_norm(self,image_r):
        image_r = (image_r-image_r.min())/(image_r.max()-image_r.min())
        return image_r



    def img2rgb(self,img_raw,wavelengths):
        redband=690
        greenband=540
        blueband=460
        rindex=self.find_nearest(wavelengths, redband)    
        gindex=self.find_nearest(wavelengths, greenband) 
        bindex=self.find_nearest(wavelengths, blueband) 
        image_r=img_raw[:,:,rindex]
        image_g=img_raw[:,:,gindex]
        image_b=img_raw[:,:,bindex]    
        red=Image.fromarray((self.image_norm(image_r)*256).astype(np.uint8))
        green=Image.fromarray((self.image_norm(image_g)*256).astype(np.uint8))
        blue=Image.fromarray((self.image_norm(image_b)*256).astype(np.uint8))
        rgb=Image.merge("RGB",(red,green,blue))
        enhancer = ImageEnhance.Brightness(rgb)    #image brightness enhancer
        rgb_output = enhancer.enhance(factor=30)
    #    rgb_output.show()
        return rgb_output


    def red_edge_segmentation(self,img_raw,wavelengths,threshold):
        lin=np.arange(-20, 20, 1)
        m,n,o=img_raw.shape
        t=np.zeros((m,n))
        redband=680
        rindex=self.find_nearest(wavelengths, redband)  
    
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


    def padimage(self,img,m,n,o):
        a,b,c=img.shape
        del1=m-a
        if del1>0:  #smaller than required image size
            if del1 % 2 ==0: 
                img=np.pad(img, (int((del1/2), int(del1/2)),(0,0)), 'edge')
            else:
                img=np.pad(img, ((math.floor(del1/2), math.floor(del1/2)),(0,0)), 'edge')
        elif del1<0: #larger than required image size
            img=img[math.ceil(abs(del1)/2):math.ceil(abs(del1)/2)+m,:,:]
            
        del2=n-b
        if del2>0:
            if del2 % 2 ==0:
                img=np.pad(img, ((0,0),(int(del2/2), int(del2/2))), 'edge')
            else:
                img=np.pad(img, ((0,0),(math.floor(del2/2), math.floor(del2/2))), 'edge')
        elif del2<0:
            img=img[math.ceil(abs(del2)/2):math.ceil(abs(del2)/2)+m,:,:]            
            
        if c!=o:
            print("Spectral dimension mismatch")       
        return img


    def feature_extraction(self,mask):
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


    def pixelNDVI(self,intensity,wavelengths):
        nirband=800
        redband=680
        rindex=self.find_nearest(wavelengths, redband)    
        nirindex=self.find_nearest(wavelengths, nirband) 
        intensity_r=intensity[rindex]
        intensity_nir=intensity[nirindex]
        NDVI=(intensity_nir-intensity_r)/(intensity_nir+intensity_r)
        return NDVI
    
    def pixelSIPI(self,intensity,wavelengths):
        nirband=800
        redband=680
        blueband=445
        rindex=self.find_nearest(wavelengths, redband)    
        nirindex=self.find_nearest(wavelengths, nirband)
        bindex=self.find_nearest(wavelengths, blueband) 
        intensity_r=intensity[rindex]
        intensity_nir=intensity[nirindex]
        intensity_b=intensity[bindex]    
        SIPI=(intensity_nir-intensity_b)/(intensity_nir+intensity_r)
        return SIPI


    def gettable(self,file,path,refimg,mask,wavelengths,threshold):
        refimg1=refimg.copy()
        rootfolder = os.path.basename(path[:-1]) 
        LeafProp=self.feature_extraction(mask)
        columnlist=['Folder','Filename','Threshold','LeafArea','NDVI','SIPI']+(wavelengths.tolist())
        # create a dictionary of column names and the value you want
        df_image = pd.DataFrame(np.zeros((1, len(columnlist))),columns=columnlist)
        df_image['Folder']= rootfolder      
        df_image['Filename']= file
        df_image['Threshold']= threshold
        df_image['LeafArea']= LeafProp["Area"]    
        intensity=[]
        maskindex = mask ==0
        refimg1[maskindex]=0
        nonmaskindex=mask !=0
        for i in range(len(wavelengths)):
            intensity.append(np.mean(refimg1[nonmaskindex,i]))        
        df_image['NDVI']= self.pixelNDVI(intensity,wavelengths) 
        df_image['SIPI']= self.pixelSIPI(intensity,wavelengths)      
        df_image.iloc[0,6:]=intensity
        return df_image



   
if __name__ == "__main__":    
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



#plt.plot(DataCell.iloc[0,10:])




## create Pillow image
#image2 = Image.fromarray(data)
## creat numpy image
#rgbdata = np.asarray(rgb).copy()
#maskindex = mask ==0 
#rgbdata[maskindex] = 0
#plt.imshow(rgbdata)



  


    
    