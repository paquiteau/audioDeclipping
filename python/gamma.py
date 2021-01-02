#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:54:20 2021

@author: pac
"""
import numpy as np

class Gamma(object):
    """Defines Gamma(y) the convex set of admissibles signal to declip y."""

    def __init__(self, y):
        self.y = y
        tauMax= np.max(y)
        tauMin = np.min(y)
        self.tauMax = tauMax
        self.tauMin = tauMin
        self.mLow = y == tauMin
        self.mHigh = y == tauMax
        self.mRight = np.logical_not(np.logical_or(self.mLow, self.mHigh))
        self.bLow = np.empty_like(y)
        self.bHigh = np.empty_like(y)
        # create the boundary vectors
        self.bLow[self.mLow] = -np.infty
        self.bLow[self.mRight] = y[self.mRight]
        self.bLow[self.mHigh] = tauMax

        self.bHigh[self.mLow] = tauMin 
        self.bHigh[self.mRight] = y[self.mRight]
        self.bHigh[self.mHigh] = +np.infty
    def __call__(self, x):
        """
        Perform the projection on the Gamma(y) set.
        """
        #return np.maximum(np.minimum(x, self.bLow), self.bHigh)
        return np.minimum(np.maximum(self.bLow,x),self.bHigh)
