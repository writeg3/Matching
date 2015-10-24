'''
Created on Oct 11, 2015

@author: Robert Washbourne
'''
import numpy as np
from wx.lib.pubsub.utils.misc import Enum
from numpy import ndarray
from Algorithm import Algorithm

# Find relative horizontal distances between 2 images
#python is column fast


#   1. find single global 2D horizontal shift
#      a. write 2D correlation as function of horizontal shift
#      b. properly handle edge effects
#      c. write unit test to ensure correctness
#
#   2. for each row of the images, find the local shifts in small
#      window around the global shift
#      a. write 1D correlation as function of horizontal shift and input global offset
#      b. properly handle edge effects
#      c. write unit test to ensure correctness
#
# Design goals: 
#   Make this easy to understand for someone unfamiliar with the project
#   Make unit tests that guarantee the individual functions are correct
#
# Workflow
# 1. write small functions for 1a, 2a
# 2. write unit tests for functions
# 3. test on images

# Normalized correlation function
#   cols are the fast dimension
#   rows are the slow dimension
def normalizedCorrelationHorzShift2D(signal1, signal2, maxLag, alg):
  '''signal1_1D is a 2D array that is shifted,
       signal2_1D is a 2D array, 
       maxLag is the maximum lag distance'''

  shape1 = signal1.shape
  shape2 = signal2.shape
  
  if (shape1 != shape2):
    raise Exception("signal1 and signal2 are not the same shape")
  
  ncol = signal1.shape[0]
  nrow = signal1.shape[1]

  output = np.zeros(2 * maxLag + 1)
  
#   signal1_T = np.transpose(signal1)
#   signal2_T = np.transpose(signal2)

  signal1_T = np.ndarray.transpose(signal1)
  signal2_T = np.ndarray.transpose(signal2)
  signal1_T = np.array(signal1_T)
  signal2_T = np.array(signal2_T)
  
  for shift in xrange(-maxLag, maxLag+1):
    
    # BRUTE_FORCE
    if alg == Algorithm.BRUTE_FORCE:
      sum11 = 0
      sum12 = 0
      sum22 = 0
      for krow in xrange(0, nrow):
        ksft = krow + shift
        if ksft >= 0 and ksft < nrow:
          for kcol in xrange(0, ncol):
            sum11 += signal1[kcol, ksft] * signal1[kcol, ksft]
            sum12 += signal1[kcol, ksft] * signal2[kcol, krow]
            sum22 += signal2[kcol, krow] * signal2[kcol, krow]
      denom = (sum11 * sum22)**0.5


    # INNER_1D
    elif alg == Algorithm.INNER_1D:
      sum11 = 0
      sum12 = 0
      sum22 = 0
      for krow in xrange(0, nrow):
        ksft = krow + shift
        if ksft >= 0 and ksft < nrow:
#           sum11 += np.inner(signal1_T[ksft], signal1_T[ksft])
#           sum12 += np.inner(signal1_T[ksft], signal2_T[krow])
#           sum22 += np.inner(signal2_T[krow], signal2_T[krow])
          sum11 += np.dot(signal1_T[ksft], signal1_T[ksft])
          sum12 += np.dot(signal1_T[ksft], signal2_T[krow])
          sum22 += np.dot(signal2_T[krow], signal2_T[krow])
      denom = (sum11 * sum22)**0.5
      
    # INNER_2D
    elif alg == Algorithm.INNER_2D:
      sum11 = 0
      sum12 = 0
      sum22 = 0
      ksft1 = max(0, 0 + shift)
      ksft2 = min(nrow - 1, (nrow - 1) + shift)
      krow1 = max(0, 0 - shift)
      krow2 = min(nrow - 1, (nrow - 1) - shift)
      s1 = np.array(signal1[0:ncol, ksft1:(ksft2 + 1)])
      s2 = np.array(signal2[0:ncol, krow1:(krow2 + 1)])
      sum11 = np.dot(s1, s1)
      sum22 = np.dot(s2, s2)
      sum12 = np.dot(s1, s2)
      denom = (sum11 * sum22)**0.5
 
    else:
      raise Exception("parameter 'alg' must be from 'borked enum' Algorithm")
 
    if denom > 0:
      output[shift + maxLag] = sum12 / (sum11 * sum22)**0.5
    
  return output
       