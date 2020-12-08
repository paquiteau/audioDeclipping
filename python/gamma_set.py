
import numpy as np

class Gamma(object):
    def __init__(self, y):
        tau = np.max(y)
        tauMin = np.min(y)
        self.mLow = y == tauMin
        self.mHigh = y == tau
        self.mRight = not(self.mLow or self.mHigh )
        self.bLow = np.empty_like(y)
        self.bHigh = np.empty_like(y)
        # create the boundary vectors
        self.bLow[self.mRight] = y[self.mRight]
        self.bLow[self.mLow] = tauMin
        self.bLow[self.mHigh] = np.inf
        self.bHigh[self.mRight] = y[self.mRight]
        self.bHigh[self.mLow] = -np.inf
        self.bHigh[self.mHigh] = tau

    def __call__(self,x):
        return np.maximum(np.minimum(x,self.bLow),self.bHigh)
    
