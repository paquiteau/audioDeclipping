import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

from . import Gamma
fs, x0 = wav.read('Sounds/quintet.wav')
tau = np.percentile(x0,75)


def hard_clip(x, th):
    y = x
    y[y > th] = th
    y[y < -th] = -th
    return y

def hard_zero(x, k):
    """ keep the K greatest values of x, set the rest to zero"""
    sort_idx = np.argsort(x)
    x[sort_idx[:-k]] = 0
    return x

y = hard_clip(x0,tau)
gammaSet = Gamma(y)
