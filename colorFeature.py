# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:47:25 2017

@author: Sujay
"""
import cv2
import glob
import numpy as np
from scipy.stats import skew
'''
def colorExtraction(img):
    c,t1,t2,t3=[],[],[],[]
    for i in range(3):
        t1.append(np.mean(img[:,:,i]))
        print(t1)
        t2.append(np.std(img[:,:,i]))
        print(t2)
        t3.append(skew(skew(img[:,:,i])))
        print(t3)
        
    c.append((t1-np.min(t1))/(np.max(t1)-np.min(t1)))
    c.append((t2-np.min(t2))/(np.max(t2)-np.min(t2)))
    c.append((t3-np.min(t3))/(np.max(t3)-np.min(t3)))
    return c
'''
img=cv2.imread('C:\\Users\\Sujay\\Desktop\\doc\\testImages\\IMG_0652.jpg')
c,t1,t2,t3=[],[],[],[]
for i in range(3):
    t1.append(np.mean(img[:,:,i]))
    print(t1)
    t2.append(np.std(img[:,:,i]))
    print(t2)
    t3.append(skew(skew(img[:,:,i])))
    print(t3)
        
c.append((t1-np.min(t1))/(np.max(t1)-np.min(t1)))
c.append((t2-np.min(t2))/(np.max(t2)-np.min(t2)))
c.append((t3-np.min(t3))/(np.max(t3)-np.min(t3)))