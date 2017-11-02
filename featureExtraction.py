# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:21:51 2017

@author: Sujay
"""

import cv2
import glob
import numpy as np
from scipy.stats import skew
import scipy.io as spio
''' COLOR EXTRACTION '''
def colorExtraction(img):
    c,t1,t2,t3=[],[],[],[]
    for i in range(3):
        t1.append(np.mean(img[:,:,i]))
        t2.append(np.std(img[:,:,i]))
        t3.append(skew(skew(img[:,:,i])))
    for i in range(3):  
        c.append((t1[i]-np.min(t1))/(np.max(t1)-np.min(t1)))
        c.append((t2[i]-np.min(t2))/(np.max(t2)-np.min(t2)))
        c.append((t3[i]-np.min(t3))/(np.max(t3)-np.min(t3)))
    return c
def textureExtraction(img):
    filters = []
    ksize = [47,91,181,359]
    wMin=4/(2**0.5)
    for i in range(4):
        Lambda=(2**i)*wMin
        for theta in np.arange(0, np.pi, np.pi/6):
            kern = cv2.getGaborKernel((ksize[i], ksize[i]), 1.0, theta, Lambda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    
    accum,t1,t2=[],[],[]
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        t=np.mean(fimg)
        t1.append(t)
        t=np.std(fimg)
        t2.append(t)
    
    for i in range(24):
        accum.append((t1[i]-np.min(t1))/(np.max(t1)-np.min(t1)))
        accum.append((t2[i]-np.min(t2))/(np.max(t2)-np.min(t2)))
        
    return accum
 
def ehd(I):
    [m,n]=I.shape
    newimg=np.zeros((m,n))
    bm=int(m/4)
    bn=int(n/4)
    blocks=np.zeros((bm,bn,16))
    f=np.zeros((3,3,5))
    hist=np.zeros(85)
    cnt=-1
    for i in range(4):
        for j in range(4):
            cnt=cnt+1
            for k in range(bm):
                for l in range(bn):
                    blocks[k,l,cnt]=I[bm*i+k,bn*j+l]
                    newimg[bm*i+k,bn*j+l]=blocks[k,l,cnt]
    tcnt=-1
    f[:,:,0]=[[-1,0,1],[-2,0,2],[-1,0,1]]
    f[:,:,1]=[[1,2,1],[0,0,0],[-1,-2,-1]]
    f[:,:,2]=[[2,2,-1],[2,-1,-1],[-1,-1,-1]]
    f[:,:,3]=[[-1,2,2],[-1,-1,2],[-1,-1,-1]]
    f[:,:,4]=[[-1,0,1],[0,0,0],[1,0,-1]]
    for i in range(16):
        for j in range(5):
            tcnt=tcnt+1
            temp=cv2.filter2D(blocks[:,:,i],-1,f[:,:,j])
            edges=cv2.Canny((np.uint8(temp)),bm,bn)
            ecnt=0
            for k in range(bm):
                for l in range(bn):
                    if(edges[k,l]==255):
                        ecnt=ecnt+1
            hist[tcnt]=ecnt
    for j in range(5):
        temp=cv2.filter2D(I,-1,f[:,:,j])
        edges=cv2.Canny(np.uint8(temp),m,n)
        ecnt=0
        for k in range(m):
            for l in range(n):
                if(edges[k,l]==255):
                    ecnt=ecnt+1
        hist[tcnt+j]=ecnt       
    hist=(hist-np.min(hist))/(np.max(hist)-np.min(hist)) 
    return(hist)           

if __name__=="__main__":
    images=glob.glob('C:\\Users\\Sujay\\Desktop\\doc\\testImages\\*.jpg')
    dataset=[]
    for file in images:
        img=cv2.imread(file)
        print('\nExtracting features from',file,'\nImage resolution:',img.shape)
        dataset.append(np.concatenate((colorExtraction(img),textureExtraction(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)),ehd(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)))))
    dataset=np.reshape(dataset,(len(images),142))
    spio.savemat('dataset.mat',{'vect':dataset})
    spio.savemat('list.mat',{'vect':images})
