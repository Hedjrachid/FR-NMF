#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 16:41:32 2021

@author: rachid2
"""



from sklearn.datasets import load_digits
from frnmf import FRNMF
import nimfa
import GNMFLIB as gnmflb
import tools 
import numpy as np
#
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning) 


# load data
data  = load_digits()
X = data.data
y = data.target
# sacel data
X = tools.featureScaling(X, scaler_name = 'minmax')

# parameters
# r:  rank
# max iteration: mi
    
r = len(np.unique(y))  # number of centers (classes)
mi = 500


# matrix factorization algorithms
# Feature Relationship Preservation NMF (FRNM)
frnmf = FRNMF(X.T, rank = r, lmbd =0.5, max_it = mi)
frnmf.compute_w_h()
Hfrnmf = frnmf.H
y_pred = np.argmax(Hfrnmf, axis=0)
tools.scoresList(y, y_pred, 'FRNMF')


# Basic NMF (NMF)
nmf = nimfa.Nmf(X.T, rank = r, max_iter = mi)
nmf_fit = nmf()
Hnmf = nmf_fit.coef()
y_pred = np.array(np.argmax(Hnmf, axis=0))[0]
tools.scoresList(y, y_pred, 'NMF')

#Probabilistic Sparse Matrix Factorization (PSNMF)
psmf = nimfa.Psmf(X.T, rank = r, max_iter=mi)
psmf_fit = psmf()
Hpsmf = psmf_fit.coef()
y_pred = np.array(np.argmax(Hpsmf, axis=0))[0]
tools.scoresList(y, y_pred, 'PSMF')

#Probabilistic Matrix Factorization (PMF)
pmf = nimfa.Pmf(X.T, rank = r, max_iter=mi)
pmf_fit = pmf()
Hpmf = pmf_fit.coef()
y_pred = np.array(np.argmax(Hpmf, axis=0))[0]
tools.scoresList(y, y_pred, 'PMF')

# Penalized Matrix Factorizartion for Constrained Clustering (PMFCC)
pmfcc = nimfa.Pmfcc(X.T, rank = r, max_iter=mi)
pmfcc_fit = pmfcc()
Hpmfcc = pmfcc_fit.coef()
y_pred = np.array(np.argmax(Hpmfcc, axis=0))[0]
tools.scoresList(y, y_pred, 'PMFCC')

#Spare NMF (SNMF) 
snmf = nimfa.Snmf(X.T, seed="random_c", rank = r, max_iter=mi)
snmf_fit = snmf()
Hsnmf = snmf_fit.coef()
y_pred = np.array(np.argmax(Hsnmf, axis=0))[0]
tools.scoresList(y, y_pred, 'SNMF')

#Graph Regularized NMF (GNMF)
from libnmf.gnmf import GNMF
gnmf = GNMF(X.T, rank = r)
gnmf.compute_factors(max_iter=mi, lmd= 10, weight_type='dot-weighting', param= 5)
Hgnmf= gnmf.H
y_pred = np.array(np.argmax(Hgnmf, axis=0))[0]
tools.scoresList(y, y_pred, 'GNMF')


print('GNMF')
_, Hgnmf2, _ = gnmflb.gnmf(X.T,  lambd=0.3, n_components=r, tol_nmf=1e-3, max_iter=mi, verbose=False)
y_pred = np.array(np.argmax(Hgnmf2, axis=0))
tools.scoresList(y, y_pred, 'GNMF2')






