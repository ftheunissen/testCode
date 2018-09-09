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
sound_amp = np.log(aed.S[:,aed.integer2prop.index(b'maxAmp')]).reshape(nsamps,1)
sound_dur = aed.S[:,aed.integer2prop.index(b'stdtime')].reshape(nsamps, 1)
sound_loud = (np.log(aed.S[:,aed.integer2prop.index(b'maxAmp')]*aed.S[:, aed.integer2prop.index(b'stdtime')])).reshape(nsamps,1)
sound_sal = aed.S[:,aed.integer2prop.index(b'sal')].reshape(nsamps, 1)

# To test all features
# S = np.hstack( (aed.S, sound_loud))
# S_names = deepcopy(aed.integer2prop)
# S_names.append(b'loudness')

# Let's start simple:
S = np.hstack( (sound_amp, sound_dur, sound_loud, sound_sal) )
S_names = [b'maxAmp', b'stdtime', b'loudness', b'sal']

# Here we are using the same functions as in the fit to get the spline representation
sr = StagewiseSplineRidgeRegression()
Sb = sr.spline_basis(S)
nstim_features = Sb.shape[1]
    
# Y has information about each stimulus or syllables
assert aed.Y.shape[0] == nsamps

aed.good_encoders = list()

cv_indices = list(zip(*aed.bootstrap(25)))

# Base features to deat with sound intensity
base_features = [S_names.index(b'maxAmp'), S_names.index(b'stdtime'),
    S_names.index(b'loudness')]

# Z-score the acoustic features
#S = self.preprocess_acoustic_features(acoustic_props=encoder_acoustic_props)


            
# run an encoder for each neural feature, which could be the spike rate of a neuron or the LFP power
# at a given frequency
# for k in range(nfeatures_neural):
for k in [15]:
    y = deepcopy(aed.X[:, k])

    if zscore_response:
        y -= y.mean()
        y /= y.std(ddof=1)
        y[np.isnan(y)] = 0.
        y[np.isinf(y)] = 0.


    sr = StagewiseSplineRidgeRegression()
    edict = sr.fit(S, y, baseline_features=base_features, cv_indices=cv_indices, verbose=True,
                               feature_names=S_names)
    
    if plot_me:
        plt.figure()
        for ip in base_features:
            plt.subplot(1,len(base_features), ip+1)
            
            x = S[:,ip]
            plt.plot(S[:,ip], y, 'r+')
            
            # Plot the prediction
            si = ip*6;
            ei = si + 6
            
            # The stimulus keeping other features constant
            St = deepcopy(Sb)
            
            for i in range(si):
                mean_col = np.mean(St[:,i])
                St[:,i] = mean_col
                
            for i in range(ei,nstim_features):
                mean_col = np.mean(St[:,i])
                St[:,i] = mean_col

            # get the encoder weights for this acoustic feature
            wj = edict['W']

            # get the prediction of the neural response based on the regression
            yhat = np.dot(St, wj) + edict['b']
            plt.plot(S[:,ip], yhat, 'k+')
        plt.show()

    if edict is None:
        print('\tFeature %d is not predictable!' % k)
    else:
        bf = [S_names[f].decode('UTF-8') for f in edict['features']]
        print('\tFeature %d: best_props=%s, R2=%0.2f' % (k, ','.join(bf), edict['r2']))
        aed.good_encoders.append((k, edict))

# Load and fit data
# aed.fit(preproc_file, model_type='linear', encoder=True, decoder=True, zscore_response=True)
    
# Save file
#aed.save(output_file)
#aed = AcousticEncoderDecoder.load(output_file)
 
#aed.plot_tuning_curves(acoustic_prop=b'meanspect')
#plt.axis('tight')
#plt.show()