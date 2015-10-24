'''
Created on Oct 18, 2015
'''

from numpy.matlib import rand
from DistanceEstimator import normalizedCorrelationHorzShift2D
import numpy
from copy import deepcopy
import timeit
import unittest
from Algorithm import Algorithm

class TestStringMethods(unittest.TestCase):

#   def testMakeSureImNotCrazy(self):
#     x = range(0,10)
#     print(x[0:2])
    
#  @unittest.skip("demonstrating skipping")
  def testNormalizedCorrelationHorzShift2D_SameArray_AllAlgorithms(self):    
    nrow = 251
    ncol = 250
    maxLag = 3
    nloop = 4
    x1 = rand(ncol, nrow)
    x2 = deepcopy(x1)

    t1a = timeit.time.clock()
    for loop in xrange(0, nloop):
      arrayA = normalizedCorrelationHorzShift2D(x1, x2, maxLag, Algorithm.BRUTE_FORCE)
    t2a = timeit.time.clock()
    
    t1b = timeit.time.clock()
    for loop in xrange(0, nloop):
      arrayB = normalizedCorrelationHorzShift2D(x1, x2, maxLag, Algorithm.INNER_1D)
    t2b = timeit.time.clock()

    t1c = timeit.time.clock()
    for loop in xrange(0, nloop):
      arrayC = normalizedCorrelationHorzShift2D(x1, x2, maxLag, Algorithm.INNER_2D)
    t2c = timeit.time.clock()
    
    timeA = (t2a - t1a)
    timeB = (t2b - t1b)
    timeC = (t2c - t1c)
    
    print("\ntestSame_NormalizedCorrelationHorzShift2D")
    print("  Algorithm.BRUTE_FORCE -- time; %.6f sec -- ratio; %12.6f" % (timeA, (timeA / timeA)))
    print("  Algorithm.INNER_1D    -- time; %.6f sec -- ratio; %12.6f" % (timeB, (timeA / timeB)))
    print("  Algorithm.INNER_2D    -- time; %.6f sec -- ratio; %12.6f" % (timeC, (timeA / timeC)))

    print
    for shift in range(-maxLag, maxLag+1):
        print("shift,A,B,C; %+3d %+12.6f %+12.6f %+12.6f" % (shift, arrayA[shift + maxLag], arrayB[shift + maxLag], arrayC[shift + maxLag]))
    assert arrayA[maxLag] == 1
    assert arrayB[maxLag] == 1
    assert arrayC[maxLag] == 1
    arrayA = list(arrayA)
    arrayB = list(arrayB)
    arrayC = list(arrayC)
    maxIndexA = arrayA.index(max(arrayA)) - maxLag
    maxIndexB = arrayB.index(max(arrayB)) - maxLag
    maxIndexC = arrayC.index(max(arrayC)) - maxLag

    print
    print("Index of maximum A,B,C", maxIndexA, maxIndexB, maxIndexC)
    print("Correlation at maximum A,B,C", 
          arrayA[maxIndexA + maxLag], arrayB[maxIndexB + maxLag], arrayC[maxIndexC + maxLag])

  @unittest.skip("demonstrating skipping")
  def testNormalizedCorrelationHorzShift2D_ShiftedArray(self):
    nrow = 400
    ncol = 300
    maxLag = 5
    
    for shift in range(-maxLag, (maxLag + 1)):
      x1 = numpy.zeros((ncol, nrow))
      x2 = numpy.zeros((ncol, nrow))
      for krow in range(0, nrow):
        for kcol in range(0, ncol):
          x2[kcol,krow] = krow
          if (krow + shift) >= 0 and (krow + shift) < nrow:
            x1[kcol,krow + shift] = krow
      t1 = timeit.time.clock()
      array = normalizedCorrelationHorzShift2D(x1, x2, maxLag, Algorithm.INNER_2D)
      t2 = timeit.time.clock()
      print("\ntestShift_NormalizedCorrelationHorzShift2D -- time; %.6f sec" % (t2 - t1))
#       for kshift in range(-maxLag, maxLag+1):
#           print("kshift,corr; %+3d %+12.6f" % (kshift, array[kshift + maxLag]))
      array = list(array)
      maxIndex = array.index(max(array)) - maxLag
      print("Index of maximum ", maxIndex)
      print("shift", shift)        
      assert maxIndex == shift

if __name__ == '__main__':
    unittest.main()
