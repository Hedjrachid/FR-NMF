#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:50:38 2021

@author: rachid2
"""

import sklearn.metrics as met
from sklearn.preprocessing import MinMaxScaler, Normalizer

def featureScaling(X, scaler_name = 'minmax'):
    
    if scaler_name.lower() =='none':
        return X
    
    elif scaler_name.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_name.lower() == 'normalizer':
        scaler = Normalizer(norm='l2')
       
    return scaler.fit_transform(X)


def scoresList(y_true, y_pred, meth):
    acc_ami = met.adjusted_mutual_info_score(y_true, y_pred) 
    acc_ari = met.adjusted_rand_score(y_true, y_pred) 
    acc_vms = met.v_measure_score(y_true, y_pred) 
    acc_nmi = met.normalized_mutual_info_score(y_true, y_pred) 
    print('----')
    print('Scores for', meth, 'are:', 
          '\n-> Adjusted Mutual Information (AMI) = %.2f'%acc_ami, 
          '\n-> Adjusted Random Information (ARI) = %.2f'%acc_ari,
          '\n-> V_Mesure_Score (VMS) = %.2f'%acc_vms, 
          '\n-> Normalized Mutual Information (NMI) = %.2f'%acc_nmi)
