#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:07:48 2021

@author: rachid2
@ License: SQU

Base class used by other NMF-based methods
"""

import numpy as np
import scipy.linalg as LA
from scipy.stats import entropy
from sys import exit


class NMFMain():
    
    def __init__(self, X, rank = 2, **kwargs):
        '''
        X: input data matrix (dim, nsamples)
        '''
        self.X = X
        self._rank = rank
        self.dim = self.X.shape[0]
        self._nsamples = self.X.shape[1]
        
        
    def norm_frobenius(self):
        '''
        Euclidean distance (error) between X and WxH
        '''
        
        if hasattr(self, 'H') and hasattr(self, 'W'):
            error = LA.norm(self.X - np.dot(self.W, self.H))
        else:
            error = None
            
        return error
    
    
    def kl_divergence(self):
        '''
        KL divergence between X and WxH
        '''
        
        if hasattr(self, 'H') and hasattr(self, 'W'):
            V = np.dot(self.W, self.H)
            error = entropy(self.X, V).sum()
        else:
            error = None
            
        return error      
    
    
    def init_h(self):
        '''
        Initialize randomly h in [0,1]
        '''
        self.H = np.random.random((self._rank, self._nsamples))
        
        
    def init_w(self):
        '''
        Initialzie randomly w in [0,1]
        '''
        self.W = np.random.random((self.dim, self._rank))
        
        
    def update_h(self):
        '''
        override by the method of the subclass
        '''
        pass
    
    def update_w(self):
        '''
        override by the method of the subclass
        '''
        pass
        
    def check_positivity(self):
        if self.X.min() < 0:
            return 0
        else:
            return 1
    
    def compute_w_h(self, max_iter=100):
        if self.check_positivity():
            pass
        else:
            print('Data should be positive!! Please double check the values')
            exit()
            
        if not hasattr(self, 'H'):
            self.init_w()
        if not hasattr(self, 'H'):
            self.init_h()
        
        self.frob_error = np.zeros(max_iter)
        for i in range(max_iter):
            self.update_w()
            self.update_h()
            self.frob_error[i] = self.norm_frobenius()
    
    def normalize_w_h(self, norm='l2'):
        k = self.W.shape[1]
        normH = True
        if normH:
            if norm == 'max':
                for r in range(k):
                    h_max = np.max(self.H[r,:])
                    if h_max > 0:
                        self.H[r,:] = self.H[r,:] / h_max
                        self.W[:,r] = self.W[:,r] * h_max
            elif norm == 'l2':
                for r in range(k):
                    h_nrm = np.max((1E-15,np.sqrt(np.sum(self.H[r,:]**2))))
                    if h_nrm > 0:
                        self.H[r,:] = self.H[r,:] / h_nrm
                        self.W[:,r] = self.W[:,r] * h_nrm
        else:
            if norm == 'max':
                for r in range(k):
                    v_max = np.max(self.W[:,r])
                    if v_max > 0:
                        self.W[:,r] = self.W[:,r] / v_max
                        self.H[r,:] = self.H[r,:] * v_max
            elif norm == 'l2':
                for r in range(k):
                    w_nrm = np.max((1E-15,np.sqrt(np.sum(self.W[:,r]**2))))
                    if w_nrm > 0:
                        self.W[:,r] = self.W[:,r] / w_nrm
                        self.H[r,:] = self.H[r,:] * w_nrm   
    



