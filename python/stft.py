#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:54:58 2021

@author: pac
"""
import scipy.signal as sps

class Stft(object):
    def __init__(self,
                 signal_length=-1,
                 fs=44100,
                 window="hann",
                 overlap_percent=0.5,
                 length=1024):
        self.fs= fs
        self.nperseg = length
        self.noverlap = int(overlap_percent * length)
        self.window = window
        self.signal_length = signal_length
    def stft(self,x):
        t,f,Zxx = sps.stft(x,self.fs,
                           nperseg=self.nperseg,
                           noverlap=self.noverlap,
                           window=self.window)
        return Zxx

    def invert(self):
        B = self.copy()
        B.__call__ = lambda x: self.__call__(x, invert=True)
        
        
    def istft(self,Zxx):
        t,x = sps.istft(Zxx,self.fs,
                        nperseg=self.nperseg,
                        noverlap=self.noverlap,
                        window=self.window)
        return x[:self.signal_length]


class AnalysisFrame(Stft):
    def __call__(self, x, invert=False):
        if invert:
            return self.istft(x)
        return self.stft(x)
    

class SynthesisFrame(Stft):
    def __call__(self, x, invert=False):
        if invert:
            return self.stft(x)
        return self.istft(x)
    
