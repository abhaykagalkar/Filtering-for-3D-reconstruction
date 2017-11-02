# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:36:36 2017

@author: Sujay
"""

import numpy as np
import cv2
def build_filters(size):
    filters = []
    ksize = [47,91,181,359]
    wMin=4/(2**0.5)
    for i in range(4):
        Lambda=(2**i)*wMin
        for theta in np.arange(0, np.pi, np.pi/6):
            kern = cv2.getGaborKernel((ksize[i], ksize[i]), 1.0, theta, Lambda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return(filters)
    
def process(img, filters):
    accum=[]
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        t=np.mean(fimg)
        accum.append(t)
        t=np.std(fimg)
        accum.append(t)
    return accum
      
        
filters = build_filters(size)
res1 = process(img, filters)
        