#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:00:26 2018

@author: frederictheunissen
"""

import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from zeebeez3.models.acoustic_encoder_decoder import AcousticEncoderDecoder
from zeebeez3.models.spline_ridge import StagewiseSplineRidgeRegression


# Code to test the curve fitting
    
exp_name = 'GreBlu9508M'
agg_dir = '/Users/frederictheunissen/Documents/Data/mschachter/aggregate'
preproc_dir = '/Users/frederictheunissen/Documents/Data/mschachter/%s/preprocess' % exp_name
decoder_dir = '/Users/frederictheunissen/Documents/Data/mschachter/%s/decoders' % exp_name
seg_uname = 'Site4_Call1_L'
decomp = 'spike_rate'
plot_me = True
# decomp = 'full_psds'
    
# Input and output files
preproc_file = os.path.join(preproc_dir, 'preproc_%s_%s.h5' % (seg_uname, decomp))
output_file = os.path.join(decoder_dir, '_acoustic_encoder_decoder_%s_%s.h5' % (decomp, seg_uname))

# The object that has the encoding and decoding results for a particualr site
aed = AcousticEncoderDecoder()

# The following code is modified from aed.fit
zscore_response = True

# Reading the data
aed.read_preproc_file(preproc_file)
aed.model_type = 'linear'    # I don't think this is used for anything

# X is neural data, nsamp the number of syllables/stims
nsamps, nfeatures_neural = aed.X.shape

# S is stimulus features
nfeatures_stim = aed.S.shape[1]

# Make a stimulus matrix
# Note that the stimulus matrix does not need to be normalize - as this is done in sklearn routines
sound_amp = np.log(aed.S[:,aed.integer2prop.index(b'maxAmp')]).reshape(nsamps,1)
sound_dur = aed.S[:,aed.integer2prop.index(b'stdtime')].reshape(nsamps, 1)
sound_loud = (np.log(aed.S[:,aed.integer2prop.index(b'maxAmp')]*aed.S[:, aed.integer2prop.index(b'stdtime')])).reshape(nsamps,1)
sound_sal = aed.S[:,aed.integer2prop.index(b'sal')].reshape(nsamps, 1)
sound_meanspect = aed.S[:,aed.integer2prop.index(b'meanspect')].reshape(nsamps, 1)

S = np.hstack( (sound_amp, sound_dur, sound_loud, sound_sal, sound_meanspect) )
S_names = [b'maxAmp', b'stdtime', b'loudness', b'sal', b'meanspect']
nsamps,nfeatures = S.shape

    
# Y has information about each stimulus or syllables
assert aed.Y.shape[0] == nsamps

aed.good_encoders = list()

cv_indices = list(zip(*aed.bootstrap(25)))

# Base features to deat with sound intensity
base_features = [S_names.index(b'maxAmp'), S_names.index(b'stdtime'),
    S_names.index(b'loudness')]

# Get baseline features in spline coordinates for plotting
sr = StagewiseSplineRidgeRegression()
Sb = sr.spline_basis(S[:,base_features])
nb_features = Sb.shape[1]
            
# run an encoder for each neural feature, which could be the spike rate of a neuron or the LFP power
# at a given frequency

for k in range(nfeatures_neural):
    
    print(('\n----------------------------Spike %d--------------------------\n')% k)

    y = deepcopy(aed.X[:, k])

    if zscore_response:
        y -= y.mean()
        y /= y.std(ddof=1)
        y[np.isnan(y)] = 0.
        y[np.isinf(y)] = 0.


    sr = StagewiseSplineRidgeRegression()
    
    # Added value models returns the base model, the x models and the amodels
    bmodel, xmodels, amodels  = sr.fit_added_value(S, y, baseline_features=base_features, cv_indices=cv_indices, verbose=True,
                               feature_names=S_names)
    
    if plot_me:
        plt.figure()
        plt.subplot(1,len(base_features)+1, 1)
        plt.plot(y, bmodel['predict'], 'b+')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')        
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        
        plt.title('Rate')
        for ip in base_features:
            plt.subplot(1,len(base_features)+1, ip+2)
            
            plt.plot(S[:,ip], y, 'r+')
            
            # Plot the prediction
            si = ip*6;
            ei = si + 6
            
            # The stimulus keeping other features constant
            St = deepcopy(Sb)
            
            for i in range(si):
                mean_col = np.mean(St[:,i])
                St[:,i] = mean_col
                
            for i in range(ei,nb_features):
                mean_col = np.mean(St[:,i])
                St[:,i] = mean_col

            # get the encoder weights for this acoustic feature
            wj = bmodel['W']

            # get the prediction of the neural response based on the regression
            yhat = np.dot(St, wj) + bmodel['b']
            plt.plot(S[:,ip], yhat, 'k+')
            limits = plt.axis()
            plt.axis([limits[0], limits[1], -2, 2])
            plt.xlabel(S_names[ip])
            
        plt.show()
        
        # Now the added-value plot
        plt.figure()
        all_features = list(range(nfeatures))
        features_left = list(np.setdiff1d(all_features, base_features))
        for ip, ifeat in enumerate(features_left):
            plt.subplot(1,len(features_left), ip+1)
            
            if bmodel['r2'] > 2.0*bmodel['r2_std']:
                yres = y - bmodel['predict']
            else:
                yres = y
            
            plt.plot(S[:,ifeat], yres, 'r+')
            plt.plot(S[:,ifeat], amodels[ip]['predict'], 'k+')
            limits = plt.axis()
            plt.axis([limits[0], limits[1], -2, 2])
            plt.xlabel('%s|base' % S_names[ifeat].decode('UTF-8'))
            plt.ylabel('y|base')
            plt.title(('%s R2 = %.2f +- %.3f\n') % (S_names[ifeat].decode('UTF-8'), amodels[ip]['r2'], amodels[ip]['r2_std']) )
        plt.show()
