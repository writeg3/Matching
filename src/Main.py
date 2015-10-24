'''
Created on Oct 11, 2015

@author: Robert Washbourne
'''
import scipy.misc
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal
from scipy.signal import argrelmax
from DistanceEstimator import normalizedCorrelationHorzShift2D

# Find relative horizontal distances between 2 images
#
#   1. find single global 2D horizontal shift
#      a. write 2D correlation as function of horizontal shift
#      b. properly handle edge effects
#      c. write unit test to ensure correctness
#      d. interpolate with parabolic maximum 
#      e. replace loop over column with dot product
#
#   2. for each row of the images, find the local shifts in small
#      window around the global shift
#      a. write 1D correlation as function of horizontal shift and input global offset
#      b. properly handle edge effects
#      c. write unit test to ensure correctness

# Design goals: 
#   Make this easy to understand for someone unfamiliar with the project
#   Make unit tests that guarantee the individual functions are correct


# Workflow
# 1. write small functions for 1a, 2a
# 2. write unit tests for functions
# 3. test on images

#main routine

#get the images
im0 = scipy.misc.imread("d0.png", 1)
im1 = scipy.misc.imread("d1.png", 1)

#resize the images
# im0_small = scipy.misc.imresize(im0, 0.5, interp='bicubic', mode=None).astype(float)
# im1_small = scipy.misc.imresize(im1, 0.5, interp='bicubic', mode=None).astype(float)
im0_small = im0.astype(float)
im1_small = im1.astype(float)

# im0_small = im0_small[200:500,0:800]
# im1_small = im1_small[200:500,0:800]

#find the means of the images
mean0 = np.mean(im0_small)
mean1 = np.mean(im1_small)
print("mean1,mean2; %+12.4f %+12.4f" % (mean0, mean1))

#subtract the means from the small images
im0_small -= mean0
im1_small -= mean1

maxLag = 40
# estimate the horizontal shift for the 2D images
array = normalizedCorrelationHorzShift2D(im0_small, im1_small, maxLag)
for shift in range(-maxLag, maxLag+1):
    print("shift,corr; %+3d %+12.6f" % (shift, array[shift + maxLag]))
    
array = list(array)
maxIndex = array.index(max(array)) - maxLag
print("Index of maximum ", maxIndex)       

# exit()

#plot them
image = [array]
# plt.subplot(1,3,1)
# plt.imshow(image, cmap = "binary", interpolation='nearest')
plt.subplot(1,2,1)
plt.imshow(im0_small, cmap = "binary", interpolation='nearest')
plt.subplot(1,2,2)
plt.imshow(im1_small, cmap = "binary", interpolation='nearest')
plt.show()



#///////////////////////END////////////////////////////////////
#/////////////////////////////////////////////////////////////

# subroutine for finding local shifts for pairs of rows from image1 and image2
def CorrsWithEdge(slit0, slit1, windowSize, lagDis):
    '''input two arrays and correlate using block matching, sized windowSize'''
    if (len(slit0.shape) != 1 or len(slit1.shape)!= 1):
        raise Exception("array1 and array2 are not 1D")
    
    if (len(slit0.shape) != len(slit1.shape)):
        raise Exception("array1 and array2 are not the same length")
    
    if (windowSize % 2 == 0):
        raise Exception("windowSize must be odd")
    
    length = len(slit0) #lag length
    
    sideBar = (windowSize - 1) / 2 #side bar to skip
    
    lag  = range(0-lagDis,lagDis)
    lagLength = len(lag)
    
    slitCorrelations = []
    
    for x in range(sideBar + (lagLength / 2), length - sideBar - (lagLength / 2)):
        box0 = slit0[(x - sideBar) : (x + sideBar)]
        
        correl = []
        for l in lag:
            box1 = slit1[(x - sideBar + l) : (x + sideBar + l)]
            
            sum11 = np.dot(box0,box0) #finding cross correlation
            sum22 = np.dot(box1,box1)
            sum12 = np.dot(box0,box1)
            
            corr = sum12 / (sum11*sum22)**0.5 #make correlation
            correl.append(corr) 
            
        slitCorrelations.append(correl.index(max(correl))-lagDis)
    return(slitCorrelations)             