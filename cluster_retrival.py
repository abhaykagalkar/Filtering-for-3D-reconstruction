# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:09:27 2017

@author: Sujay
"""

import scipy.io as spio
import cv2
import numpy as np
from featureExtraction import colorExtraction,textureExtraction,ehd
mat = spio.loadmat('Centroid_list.mat', squeeze_me=True)
C=mat['cen']
#mat = spio.loadmat('Label_list.mat', squeeze_me=True)
#L=mat['vect']
Qimg=cv2.imread('C:\\Users\\Sujay\\Desktop\\doc\\testImages\\IMG_0779.jpg')
q=np.concatenate((colorExtraction(Qimg),textureExtraction(cv2.cvtColor(Qimg,cv2.COLOR_BGR2GRAY)),ehd(cv2.cvtColor(Qimg,cv2.COLOR_BGR2GRAY))))
q=np.reshape(q,(1,142))
t=[]
for i in range(len(C)):
    t.append(C[i,:]-q)
