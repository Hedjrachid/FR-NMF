#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:31:59 2021

@author: rachid2
@Licence: SQU


FR-NMF: Feature Relationship Preservation NMF

Hedjam R, Abdesselam A, Melgani F. NMF with feature relationship preservation 
penalty term for clustering problems. Pattern Recognition. 2021 Apr 1;112:107814.
    
"""


import numpy as np
from sys import exit
from .nmfmainclass import NMFMain
import scipy.linalg as LA
from sys import float_info

class FRNMF(NMFMain):
    
    '''
    decomposes X into r components by FR-NMF
        
    Parameters:
        X: a (d x n)-array containing n observations in the columns
        k: number of components to extract
        threshold: relative error threshold of the iteration
        maxiter: maximum number of iterations
    Returns:
        W: (d x k)-array of non-negative basis images (components)
        H: (k x n)-array of weights, one column for each of the n observations
    '''
    
    def __init__(self, X, rank = 2, lmbd = 1, max_it = 1000):
        super().__init__(X, rank = rank)
        self.lmbd = lmbd
        self.max_it = max_it
        
    def norm_frobenius(self):
        '''
        Euclidean distance (error) between X and WxH
        '''
        
        if hasattr(self, 'H') and hasattr(self, 'W'):
            err1 = LA.norm(self.X - np.dot(self.W, self.H))
            err2 = LA.norm(np.dot(self.X, self.X.T) - self.lmbd*np.dot(self.W, self.W.T) )
            error = err1+err2
        else:
            error = None
            
        return error
    
    def compute_w_h(self, normWH = True, norm = 'l2'):
        
        if self.check_positivity():
            pass
        else:
            print("Data should be positive. Please double check the values")
            exit()
            
        if not hasattr(self,'W'):
            self.init_w()
            
        if not hasattr(self,'H'):
            self.init_h()
            
        self.frob_error = []
        error = self.norm_frobenius()
        converged = False
        it_no = 0        
        while (not converged) or it_no <= self.max_it:
            self.update_w()
            self.update_h()
            self.frob_error.append(self.norm_frobenius())
            
            error_new = self.norm_frobenius()
            converged = np.abs(error_new - error) <= 1E-5
            error = error_new
            it_no = it_no + 1
            
        if normWH:
            self.normalize_w_h(norm='l2') 

            
    def update_h(self):
        ''' performs the multiplicative non-negative matrix factorization updates
            
        Usage:
            W, H = seung_update(V, W, H)
        Parameters:
            X: a (d x n)-array containing n observations in the columns
            W: (d x k)-array of non-negative basis images (components)
            H: (k x n)-array of weights, one column for each of the n observations
        Returns:
            H: (k x n)-array of updated weights, one column for each of the n observations
        '''
        eps =  float_info.min
        WH = np.dot(self.W, self.H)
        self.H =  np.multiply(self.H, np.dot((self.X / WH).T, self.W).T)
        self.H[self.H <= 0] = eps
        self.H[np.isnan(self.H)] = eps
        
    def update_w(self):
        
        ''' performs the multiplicative non-negative matrix factorization updates for W
        
        Usage:
            W, H = seung_update(V, W, H)
        Parameters:
            X: a (d x n)-array containing n observations in the columns
            W: (d x k)-array of non-negative basis images (components)
            H: (k x n)-array of weights, one column for each of the n observations
        Returns:
            W: (d x k)-array of updated non-negative basis images (components)
        '''
      
        eps = float_info.min
        T1 = self.X.dot(self.H.T) + self.lmbd*np.dot(self.X.dot(self.X.T), self.W)
        T2 = (self.W.dot(self.H)).dot(self.H.T) + 0.5*self.lmbd**2*np.dot(self.W.dot(self.W.T), self.W) + eps
        self.W = self.W * T1/T2
        self.W = self.W / np.sum(self.W, axis=0, keepdims=True)  
        
        self.W[self.W <= 0] = eps
        self.W[np.isnan(self.W)] = eps


    
