#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:04:43 2021

@author: pac
"""
import numpy as np
import scipy as sp
from scipy.io import wavfile
from tqdm import tqdm

from gamma import Gamma

def hard_zero(x, k):
        """Keep the K greatest values of x, set the rest to zero."""
        if k >= x.size:
            return x
        sort_idx = np.argsort(np.abs(x).flat)
        x[np.unravel_index(sort_idx[:-k], x.shape)] = 0
        return x


class Declipper(object):
    """
    Contains everything needed to study the audio declipping signal
    """
    
    def __init__(self,y, x=np.ndarray([])):
        """ y - clipped signal to reconstruct
            x - initial signal
            """
        self.fs, self.y = Declipper.audio_format(y)
        _ , self.x = Declipper.audio_format(x)
        
        self.xhat = np.empty_like(x)
        self.gamma = Gamma(self.y)
        
    @staticmethod
    def audio_format(x):
        if type(x) is str:
            fs, x = wavfile.read(x)
            x =  x.astype(np.float32)
            return fs, x     
        else:
            return None,x
    
    @classmethod
    def clipper(cls,x, ptile=0, thres=None):
        """ clip a signal with a proposed method"""
        
        _, x = Declipper.audio_format(x)
        
        if thres is not None:
            if np.size(thres) == 1:
                thresMin, thresMax = -thres, thres
        else:
            if np.size(ptile) == 1:
                thresMin, thresMax = [np.percentile(x,ptile),
                                      np.percentile(x,100-ptile)]
            else:
                thresMin, thresMax = [np.percentile(x,ptile[0]),
                                      np.percentile(x,ptile[1])]        
                
        y = x.copy()
        y[x < thresMin] = thresMin
        y[x > thresMax] = thresMax        
        return cls(y,x), x
    @staticmethod
    def sdr(x,y):
        return 20 * np.log10(np.linalg.norm(x,2)/np.linalg.norm(x-y,2))
                             
                             
    def sdr_clipped(self):
        idx = np.logical_not(self.gamma.mRight)
        return Declipper.sdr(self.x[idx],self.y[idx])
    
    def sdr_process(self):
        idx = np.logical_not(self.gamma.mRight)
        return Declipper.sdr(self.x[idx],self.xhat[idx])
    
    
    def aSpade(self, A, k_step=1, k_init=1, iter_k=1, eps=0.1, iter_max=50, progress=True):
        # create the set of admissibles solutions.
        x = self.y.copy()
        Ax = A(x)
        z = np.zeros_like(Ax)
        u = np.zeros_like(z)
        k = k_init
        pbar = tqdm(range(iter_max),disable=not(progress))
        for i in pbar:
            xprev = x.copy()
            z = hard_zero(Ax + u, k)
            x = self.gamma(A(z - u, invert=True))
            Ax = A(x)
            u += Ax - z
            res = np.linalg.norm((Ax-z).flat,2)
            pbar.set_postfix({'res': res})

            if res < eps:
                break
            if i % iter_k == 0:
                k += k_step
        self.xhat = x
        return x

    def sSpade(self, D, k_step=1,k_init=1, iter_k=1, eps=0.1, iter_max=50, progress=True):
        zhat = D(self.y,invert=True)
        u = np.zeros_like(zhat)
        zbar = np.zeros_like(zhat)
        k = k_init
        pbar = tqdm(range(iter_max),disable=not(progress))
        for i in pbar:
            zbar = hard_zero(zhat+u,k)
            v = zbar -u 
            dv = D(v)
            zhat = v - D(dv-self.gamma(dv),invert=True)
            res = np.linalg.norm((zbar- zhat).flat,2)
            sdr = self.sdr
            pbar.set_postfix({'res': res})
            if res < eps:
                break
            if i % iter_k == 0:
                k += k_step
        self.xhat = D(zhat)
        return self.xhat
    
    
    def solve(self,method,frame, *args, **kwargs):
        method = method.upper()
        frame.signal_length=self.y.size
        if "BLOC" in method:
            method.remove("BLOC")
            return self.solve_bloc(method,frame, *args, **kwargs)
        if method == "ASPADE":
            return self.aSpade(frame, **kwargs)
        if method == "SSPADE":
            return self.sSpade(frame, **kwargs)
   
    @classmethod
    def sdr_study(cls,x, method, frame, ptiles = np.linspace(0,100,11),**kwargs):
        sdr_clipped = np.empty_like(ptiles)
        sdr_process = np.empty_like(ptiles)
        
        for i,p in enumerate(ptiles):
            prob, x = Declipper.clipper("quintet.wav",ptile=p)
            prob.solve(method,frame,**kwargs)
            sdr_clipped[i] = prob.sdr_clipped()
            sdr_process[i] = prob.sdr_process()
        return sdr_clipped, sdr_process
        

    def solve_block(self,method,frame,block_size, **kwargs):
        window = sp.signal.get_window("boxcar",block_size)
        
        y_bloc = np.array_split(self.y,self.y.size//block_size)
        xhatbloc = np.empty_like(y_bloc)
        for i,y in tqdm(enumerate(y_bloc)):
            window = sp.signal.get_window("boxcar",y.size)
            BlocDeclip = Declipper(y*window)
            xhatbloc[i] = BlocDeclip.solve(method,frame,**kwargs)
        self.xhat = xhatbloc.flatten()
        return self.xhat
        
      
        
      
        
      
