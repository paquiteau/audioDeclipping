import numpy as np
import scipy.signal as sps

class GaborTransform(object):
    """
    Set the parameters for a Gabor transform, on a 1D signal
    """
    def __init__(self,windowType,windowWidth, overlap=0, fftpadding):
        self.windowType = windowType
        self.overlap = overlap
        self.windowWidth = windowWidth

    def __call__(self,x):
        return sps.stft(x,
                        window=self.windowType,
                        nperseg=self.windowWidth,
                        noverlap=self.overlap)
    def inverse(self,t,f,x):
        return sps.istft(Zxx,window=self.windowType,self.windowWidth,)