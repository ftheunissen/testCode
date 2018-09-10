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
import pandas as pd

from zeebeez3.models.acoustic_encoder_decoder import AcousticEncoderDecoder
from zeebeez3.models.spline_ridge import StagewiseSplineRidgeRegression


# Encoder wrapper for analysing how acoustical features predict neural response
    
# This is the stuff that the wrapper will read
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

# Some data processin flags
zscore_response = True     # The neural response is shown in zscore units
do_fund = True             # To remove rows that don't have fundamental info for full model

# Reading the data
aed.read_preproc_file(preproc_file)
aed.model_type = 'linear'    # I don't think this is used for anything

# Find entries that are valid for fundamental
sound_fund = aed.S[:,aed.integer2prop.index(b'fund')]
ind_fund = sound_fund != -1.0

# We remove the data here that does not have fundemental estimation
if do_fund:
    aed.S = aed.S[ind_fund,:]
    aed.X = aed.X[ind_fund,:]
    aed.Y = aed.Y[ind_fund,:]

# S is stimulus features
nsamps, nfeatures_stim = aed.S.shape
nfeatures_neural = aed.X.shape[1]

# Make a stimulus matrix
# Note that the stimulus matrix does not need to be normalize - as this is done in sklearn routines
# The names are extracted to simplify code

# Loudness features 
sound_amp = np.log(aed.S[:,aed.integer2prop.index(b'maxAmp')]).reshape(nsamps,1)
sound_dur = aed.S[:,aed.integer2prop.index(b'stdtime')].reshape(nsamps, 1)
sound_loud = (np.log(aed.S[:,aed.integer2prop.index(b'maxAmp')]*aed.S[:, aed.integer2prop.index(b'stdtime')])).reshape(nsamps,1)
sound_loudness = np.hstack( (sound_amp, sound_dur, sound_loud) )
names_loudness = [b'maxAmp', b'stdtime', b'loud']

# Mean spectrum
sound_meanS = aed.S[:,aed.integer2prop.index(b'meanspect')].reshape(nsamps, 1)
names_meanS = [b'meanspect']

# Spectral bandwidth
sound_stdS = aed.S[:,aed.integer2prop.index(b'meanspect')].reshape(nsamps, 1)
names_stdS = [b'stdspect']

# Spectral Shape
sound_skewS = aed.S[:, aed.integer2prop.index(b'skewspect')].reshape(nsamps, 1)
sound_kurtS = aed.S[:, aed.integer2prop.index(b'kurtosisspect')].reshape(nsamps, 1)
sound_shapeS = np.hstack( (sound_skewS, sound_kurtS) )
names_shapeS = [b'skewspect', b'kurtosisspect']

# Temporal Shape
sound_skewT = aed.S[:, aed.integer2prop.index(b'skewtime')].reshape(nsamps, 1)
sound_kurtT = aed.S[:, aed.integer2prop.index(b'kurtosistime')].reshape(nsamps, 1)
sound_shapeT = np.hstack( (sound_skewT, sound_kurtT) )
names_shapeT = [b'skewtime', b'kurtosistime' ]    

# Saliency    
sound_sal = aed.S[:,aed.integer2prop.index(b'sal')].reshape(nsamps, 1)
names_sal = [b'sal']

# Fundamental
sound_fund = aed.S[:,aed.integer2prop.index(b'fund')].reshape(nsamps, 1)
names_fund = [b'fund']

sound_cvfund = aed.S[:,aed.integer2prop.index(b'cvfund')].reshape(nsamps, 1)
names_cvfund = [b'cvfund']

# Full feature space
Sfull = np.hstack( (sound_loudness, sound_meanS, sound_stdS, sound_shapeS, sound_shapeT, sound_sal, sound_fund, sound_cvfund) )
Sfull_names = names_loudness + names_meanS + names_stdS + names_shapeS + names_shapeT + names_sal + names_fund + names_cvfund

# Full model vs Delete1
assert aed.Y.shape[0] == nsamps
# This is smart bootstraping to use different birds,etc
cv_indices = list(zip(*aed.bootstrap(25)))

feature_groups = [names_loudness, names_meanS, names_stdS, names_shapeS, names_shapeT, names_sal, names_fund, names_cvfund]

# Loop through the neural features (number of electrodes or single units)
resultsAll = list()

for k in range(nfeatures_neural):    
    print(('\n----------------------------Spike %d--------------------------\n')% k)

    # Get the response
    y = deepcopy(aed.X[:, k])
    if zscore_response:
        y -= y.mean()
        y /= y.std(ddof=1)
        y[np.isnan(y)] = 0.
        y[np.isinf(y)] = 0.


    sr = StagewiseSplineRidgeRegression()
    
    # Perform the firs nested model comparison to decide whether to go on
    del_features = [Sfull_names.index(strval) for strval in feature_groups[0]]
    fmodel, nmodel  = sr.fit_nested(Sfull, y, del_features=del_features, cv_indices=cv_indices, verbose=True,
                               feature_names=Sfull_names)
    # For comparison, I also ran the step-wise model using the command below.
    # It was always worse than the full model with ridge
    # smodel = sr.fit(Sfull, y, cv_indices=cv_indices, verbose=True, feature_names=Sfull_names)
  
    if (fmodel['r2'] < 2.0*fmodel['r2_std']):
        print('Full encoding model not significant for %s %s %s %d neural feature' % (exp_name, seg_uname, decomp, k ))
        continue
    
    # Save data
    fmodel['neural feat'] = k
    fmodel['group'] = -1
    nmodel['neural feat'] = k 
    nmodel['group'] = 0
    resultsAll.append(fmodel)
    resultsAll.append(nmodel)
    
        
    for ig, names_del_features in enumerate(feature_groups):
        if ig == 0:     # Done above
            continue
        del_features = [Sfull_names.index(strval) for strval in names_del_features]
        fmodel, nmodel  = sr.fit_nested(Sfull, y, del_features=del_features, cv_indices=cv_indices, verbose=True,
                               feature_names=Sfull_names)
        nmodel['neural feat'] = k 
        nmodel['group'] = ig
        resultsAll.append(nmodel)
        


resultsDataFrame = pd.DataFrame(data=resultsAll)
    
if plot_me:
    for k in range(nfeatures_neural):
        print(('\n----------------------------Spike %d--------------------------\n')% k)
                 
        # Get the full model information from the results df    
        mfull = resultsDataFrame.loc[(resultsDataFrame['name'] =='Full') & (resultsDataFrame['neural feat'] == k)]      
        if len(mfull) == 0:
            continue
              
        # Get the response for plotting a scatter plot
        y = deepcopy(aed.X[:, k])
        if zscore_response:
            y -= y.mean()
            y /= y.std(ddof=1)
            y[np.isnan(y)] = 0.
            y[np.isinf(y)] = 0.
            
        # Calculate R2 differences from nested models    
        r2_diff = np.zeros(len(feature_groups)) 
        r2_diff_std = np.zeros(len(feature_groups)) 
        for ig, names_del_features in enumerate(feature_groups):

            mnested = resultsDataFrame.loc[
                    (resultsDataFrame['name'] =='Nested') & 
                    (resultsDataFrame['neural feat'] == k) &
                    (resultsDataFrame['group'] == ig) ]
            r2_diff[ig] = mfull['r2'].iloc[0] - mnested['r2'].iloc[0]
            r2_diff_std[ig] = np.sqrt((mfull['r2_std'].iloc[0]**2 + mnested['r2_std'].iloc[0]**2)/2)
                 
            
        plt.figure()
        plt.subplot(1,2, 1)
        plt.plot(y, mfull['predict'].values[0], 'b+')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')        
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])        
        plt.title('Full R2 = %.2f' % mfull['r2'].iloc[0])
        
        plt.subplot(1,2, 2)
        plt.errorbar(range(len(feature_groups)), r2_diff, yerr = r2_diff_std*2.0, fmt='ko')
        plt.axis([-0.5, 7.5, -0.05, 0.25]) 
        plt.hlines(0, -1, 8, colors='k', linestyles='dashed')
        plt.xlabel('Group')
        plt.ylabel('R2 Diff')               
        plt.title('R2 Difference for Nested Model')
        
        plt.show()

        
    
# Y has information about each stimulus or syllables used for generating cross-validation sets
assert aed.Y.shape[0] == nsamps

# This is smart bootstraping to use different birds,etc
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
