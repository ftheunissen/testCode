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


#--------------------------------------- Global Variables --------------------------------
zscore_response = True     # The neural response is shown in zscore units

# Subset of stimulus features for added-value tuning curves
added_valuegroups = [[b'fund'], [b'meanspect'], [b'stdspect'], [b'sal']]
base_feature_names = [[b'maxAmp', b'stdtime', b'loud', b'meanspect',b'sal'],
                      [b'maxAmp', b'stdtime', b'loud'],
                      [b'maxAmp', b'stdtime', b'loud', b'meanspect'],
                      [b'maxAmp', b'stdtime', b'loud', b'meanspect']]

# Some useful functions
#-------------------------------------  Select sound features --------------------------
def select_sound_feat(aed):
    # Select and group some features from an AcousticEncoderDecoder class aed
    # aed must already be populated with 
    
    # S is stimulus features
    nsamps, nfeatures_stim = aed.S.shape
    
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
    return [sound_loudness, sound_meanS, sound_stdS, sound_shapeS, sound_shapeT, sound_sal, sound_fund, sound_cvfund],  \
        [names_loudness, names_meanS, names_stdS, names_shapeS, names_shapeT, names_sal, names_fund, names_cvfund]


def run_encoder(preproc_file):
# Runs all encoders using the data stored in the h5 preproc_file
# Returns a panda data frame with the results from all the encoding models of choice
 
    # Read the data
    aed = AcousticEncoderDecoder()
    aed.read_preproc_file(preproc_file)
    aed.model_type = 'linear'    # I don't think this is used for anything 
    
    # We are going to run full models only for the entries (rows) that have valid fundamental values
    sound_fund = aed.S[:,aed.integer2prop.index(b'fund')]
    ind_fund = sound_fund != -1.0

    # We remove the data here that does not have fundemental estimation
    # I am doing this in-line because the cross-validaiton index generation uses aed.y
    aed.S = aed.S[ind_fund,:]
    aed.X = aed.X[ind_fund,:]
    aed.Y = aed.Y[ind_fund,:]

    # S is stimulus features
    nsamps, nfeatures_stim = aed.S.shape
    nfeatures_neural = aed.X.shape[1]
    # Get index for cross-validations (Uses bird id)
    assert aed.Y.shape[0] == nsamps
    cv_indices = list(zip(*aed.bootstrap(25)))

    # Make a stimulus matrix
    # Note that the stimulus matrix does not need to be normalized - as this is done in sklearn routines
    features, feature_groups = select_sound_feat(aed) 

    # Full feature space
    Sfull = np.hstack(features)
    Sfull_names = [name for feature_grp_names in feature_groups for name in feature_grp_names ] 


    # For the added-value models, the one for the fundamental is done first because if it based on the restricted set
    ia = 0
    Sadded_names =  base_feature_names[ia] + added_valuegroups[ia]
    ind_added = [Sfull_names.index(strval) for strval in Sadded_names]
    Sadded = Sfull[:,ind_added]
    base_features = [Sadded_names.index(strval) for strval in base_feature_names[ia] ]

    # Full model vs Delete1 
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
        ig = 0
        del_features = [Sfull_names.index(strval) for strval in feature_groups[ig]]
        fmodel, nmodel  = sr.fit_nested(Sfull, y, del_features=del_features, cv_indices=cv_indices, verbose=True,
                               feature_names=Sfull_names)
        # For comparison, I also ran the step-wise model using the command below.
        # It was always worse than the full model with ridge
        # smodel = sr.fit(Sfull, y, cv_indices=cv_indices, verbose=True, feature_names=Sfull_names)
      
        # We always save the overall fit - even if not significant so that we can compare across cells
        # or neural types
        fmodel['neural feat'] = k
        fmodel['group'] = ['All']
        resultsAll.append(fmodel)
    
        if (fmodel['r2'] < 2.0*fmodel['r2_std']):
            print('Full encoding model not significant for %s %s %s %d neural feature' % (exp_name, seg_uname, decomp, k ))
            continue
    
        # Now save the nested model for ig=0
        nmodel['neural feat'] = k 
        nmodel['group'] = ig
        resultsAll.append(nmodel)
    
        # Perform all other nested models   
        for ig, names_del_features in enumerate(feature_groups):
            if ig == 0:     # Done above
                continue
            del_features = [Sfull_names.index(strval) for strval in names_del_features]
            fmodel, nmodel  = sr.fit_nested(Sfull, y, del_features=del_features, cv_indices=cv_indices, verbose=True,
                               feature_names=Sfull_names)
            nmodel['neural feat'] = k 
            nmodel['group'] = ig
            resultsAll.append(nmodel)
        
        # Perform fit for added value for fundamental tuning curve (ia=0)
        sr = StagewiseSplineRidgeRegression()
        bmodel, xmodels, amodel  = sr.fit_added_value(Sadded, y, baseline_features=base_features, cv_indices=cv_indices, verbose=True,
                                   feature_names=Sadded_names)
        # Store only significant base or added value models 
        if bmodel['r2'] > 2.0*bmodel['r2_std']:
            bmodel['neural feat'] = k
            bmodel['group'] = ia
            resultsAll.append(bmodel)
        
        if amodel['r2'] > 2.0*amodel['r2_std']:
            xmodels[0]['neural feat'] = k
            amodel['neural feat'] = k
            xmodels[0]['group'] = ia
            amodel['group'] = ia
            resultsAll.append(xmodels[0])
            resultsAll.append(amodel)
            
    
    # Do the added value models for meanspect, bandwidth and saliency using all of the data
    # Reading the data again to have it all
    aed.read_preproc_file(preproc_file)
    nsamps, nfeatures_stim = aed.S.shape
    nfeatures_neural = aed.X.shape[1]
    assert aed.Y.shape[0] == nsamps
    cv_indices = list(zip(*aed.bootstrap(25)))
    features, feature_groups = select_sound_feat(aed) 
    # Full feature space
    Sfull = np.hstack(features)
    Sfull_names = [name for feature_grp_names in feature_groups for name in feature_grp_names ] 

    for ia in range(len(added_valuegroups)):
        if ia == 0:
            continue        # this was done above with the restricted data set
    
        Sadded_names =  base_feature_names[ia] + added_valuegroups[ia]
        ind_added = [Sfull_names.index(strval) for strval in Sadded_names]
        Sadded = Sfull[:,ind_added]
        base_features = [Sadded_names.index(strval) for strval in base_feature_names[ia] ]
    
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
    

            bmodel, xmodels, amodel  = sr.fit_added_value(Sadded, y, baseline_features=base_features, cv_indices=cv_indices, verbose=True,
                                                          feature_names=Sadded_names)
            # Store significant  models        
            if bmodel['r2'] > 2.0*bmodel['r2_std']:
                bmodel['neural feat'] = k
                bmodel['group'] = ia
                resultsAll.append(bmodel)
        
            if amodel['r2'] > 2.0*amodel['r2_std']:
                xmodels[0]['neural feat'] = k
                amodel['neural feat'] = k
                xmodels[0]['group'] = ia
                amodel['group'] = ia
                resultsAll.append(xmodels[0])
                resultsAll.append(amodel)
            

    # Make a pandas data frame for the results
    resultsDataFrame = pd.DataFrame(data=resultsAll)
    
    # Return
    return resultsDataFrame
   

def plot_encoder(preproc_file, resultsDataFrame):
# Plotter for encoder tuning curves and the results of the encoding models for one site
# aed is the acoustic-encoder-decoder object that has already the S and Y data
# resultsDataFrame is the Pandas data frame generated by run_encoder
 
# Read the data
    aed = AcousticEncoderDecoder()
    aed.read_preproc_file(preproc_file)
    aed.model_type = 'linear'    # I don't think this is used for anything 
    
# Get the data parameters and shape/group stimulus as in run_encoder
    nsamps, nfeatures_stim = aed.S.shape
    nfeatures_neural = aed.X.shape[1]
    features, feature_groups = select_sound_feat(aed) 
    Sfull = np.hstack(features)
    Sfull_names = [name for feature_grp_names in feature_groups for name in feature_grp_names ] 

# Plot the results.  First the ones that involve the entire data set.
    for ia in range(len(added_valuegroups)):
        if ia == 0:
            continue  # The fundamental tuning curve is done with the restricted data set
        
        Sadded_names =  base_feature_names[ia] + added_valuegroups[ia]
        ind_added = [Sfull_names.index(strval) for strval in Sadded_names]
        Sadded = Sfull[:,ind_added]
        base_features = [Sadded_names.index(strval) for strval in base_feature_names[ia] ]

        # Get baseline features in spline coordinates for plotting
        sr = StagewiseSplineRidgeRegression()
        Sb = sr.spline_basis(Sadded[:,base_features])
        nb_features = Sb.shape[1]
    
        for k in range(nfeatures_neural):
            print(('\n----------------------------Spike %d--------------------------\n')% k)
                # Get the response for plotting a scatter plot
            y = deepcopy(aed.X[:, k])
            if zscore_response:
                y -= y.mean()
                y /= y.std(ddof=1)
                y[np.isnan(y)] = 0.
                y[np.isinf(y)] = 0.
                        
            mbase= resultsDataFrame.loc[(resultsDataFrame['name'] =='Baseline') & (resultsDataFrame['neural feat'] == k) & (resultsDataFrame['group'] == ia)] 
            if ia == 1:
                # Plot the response vs intensity graphs
                if len(mbase) == 1:
                    plt.figure()
                    plt.plot(y, mbase['predict'].iloc[0], 'b+')
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')        
                    plt.axis('square')
                    plt.axis([-2, 2, -2, 2])
            
                    plt.title('Base Model (Intensity)')
                    plt.show()
                    
                    plt.figure()
                    for ip in base_features:
                        plt.subplot(1,len(base_features), ip+1)
                
                        plt.plot(Sadded[:,ip], y, 'r+')
                
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
                        wj = mbase['W'].iloc[0]
    
                    # get the prediction of the neural response based on the regression
                        yhat = np.dot(St, wj) + mbase['b'].iloc[0]
                        plt.plot(Sadded[:,ip], yhat, 'k+')
                        limits = plt.axis()
                        plt.axis([limits[0], limits[1], -2, 2])
                        plt.xlabel(Sadded_names[ip])
                
                    plt.show()
                        
                else:
                    print('No Significant Response vs Intensity Relationship')
                    
            
            # Perform the added value plots
            madded = resultsDataFrame.loc[(resultsDataFrame['name'] =='Added') & (resultsDataFrame['neural feat'] == k) & (resultsDataFrame['group'] == ia)]  
            added_feature = Sadded_names.index(added_valuegroups[ia][0]) 
            
            if len(madded) == 1:
                if len(mbase) != 1:
                    yres = y
                else:
                    if mbase['r2'].iloc[0] > 2.0*mbase['r2_std'].iloc[0]:
                        yres = y - mbase['predict'].iloc[0]
                    else:
                        yres = y
                plt.figure()
                plt.plot(Sadded[:,added_feature], yres, 'r+')
                plt.plot(Sadded[:,added_feature], madded['predict'].iloc[0], 'k+')
                limits = plt.axis()
                plt.axis([limits[0], limits[1], -2, 2])
                plt.xlabel('%s|base' % Sadded_names[added_feature].decode('UTF-8'))
                plt.ylabel('y|base')
                plt.title(('R2 = %.2f +- %.3f\n') % (madded['r2'].iloc[0], madded['r2_std'].iloc[0]) )
                plt.show()
            elif len(madded) > 1:
                print('Error: Found more than one entry for added value model for %s' % added_valuegroups[ia][0].decode('UTF-8'))
            else:
                print('No tuning curve for %s' % added_valuegroups[ia][0].decode('UTF-8'))
  

    # Now restrict to data that has fundamental values
    sound_fund = aed.S[:,aed.integer2prop.index(b'fund')]
    ind_fund = sound_fund != -1.0
    # We remove the data here that does not have fundemental estimation
    aed.S = aed.S[ind_fund,:]
    aed.X = aed.X[ind_fund,:]
    aed.Y = aed.Y[ind_fund,:]
    
    features, feature_groups = select_sound_feat(aed) 
    Sfull = np.hstack(features)
    Sfull_names = [name for feature_grp_names in feature_groups for name in feature_grp_names ] 
           
    r2_full_all = np.full(nfeatures_neural, np.nan)
    r2_diff_all = np.full((nfeatures_neural, len(feature_groups)), np.nan)
    r2_reldiff_all = np.full((nfeatures_neural, len(feature_groups)), np.nan)
    
    for k in range(nfeatures_neural):
        print(('\n----------------------------Spike %d--------------------------\n')% k)
                 
        # Get the full model information from the results df    
        mfull = resultsDataFrame.loc[(resultsDataFrame['name'] =='Full') & (resultsDataFrame['neural feat'] == k)]      
        if len(mfull) == 0:
            continue
              
        r2_full_all[k] = mfull['r2'].iloc[0]
        
        if (mfull['r2'].iloc[0] < 2.0*mfull['r2_std'].iloc[0]):
            continue

        # Get the response for plotting a scatter plot
        y = deepcopy(aed.X[:, k])
        if zscore_response:
            y -= y.mean()
            y /= y.std(ddof=1)
            y[np.isnan(y)] = 0.
            y[np.isinf(y)] = 0.
            
        # Calculate R2 differences from nested models    
        r2_diff = np.full(len(feature_groups), np.nan) 
        r2_diff_std = np.full(len(feature_groups), np.nan) 
        
        for ig, names_del_features in enumerate(feature_groups):

            mnested = resultsDataFrame.loc[
                    (resultsDataFrame['name'] =='Nested') & 
                    (resultsDataFrame['neural feat'] == k) &
                    (resultsDataFrame['group'] == ig) ]
            if len(mnested) == 0:
                continue
            r2_diff[ig] = mfull['r2'].iloc[0] - mnested['r2'].iloc[0]
            r2_diff_std[ig] = np.sqrt((mfull['r2_std'].iloc[0]**2 + mnested['r2_std'].iloc[0]**2)/2)
            r2_diff_all[k,ig] = r2_diff[ig]
            r2_reldiff_all[k,ig] = r2_diff[ig]/mfull['r2'].iloc[0]
                             
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
        
        # Plot the added value Tuning Curves for the fundamental        
        ia = 0
        madded = resultsDataFrame.loc[(resultsDataFrame['name'] =='Added') & (resultsDataFrame['neural feat'] == k) & (resultsDataFrame['group'] == ia)]  
        
        if len(madded) == 1:
            if len(mbase) == 0:
                 yres = y
            else:
                if mbase['r2'].iloc[0] > 2.0*mbase['r2_std'].iloc[0]:
                    yres = y - mbase['predict'].iloc[0]
                else:
                    yres = y
                    
            Sadded_names =  base_feature_names[ia] + added_valuegroups[ia]
            ind_added = [Sfull_names.index(strval) for strval in Sadded_names]
            Sadded = Sfull[:,ind_added]
            base_features = [Sadded_names.index(strval) for strval in base_feature_names[ia] ]

            # Get baseline features in spline coordinates for plotting
            sr = StagewiseSplineRidgeRegression()
            Sb = sr.spline_basis(Sadded[:,base_features])
            nb_features = Sb.shape[1]
            added_feature = Sadded_names.index(added_valuegroups[ia][0])
               
            plt.plot(Sadded[:,added_feature], yres, 'r+')
            plt.plot(Sadded[:,added_feature], madded['predict'].iloc[0], 'k+')
            limits = plt.axis()
            plt.axis([limits[0], limits[1], -2, 2])
            plt.xlabel('%s|base' % Sadded_names[added_feature].decode('UTF-8'))
            plt.ylabel('y|base')
            plt.title(('R2 = %.2f +- %.3f\n') % (madded['r2'].iloc[0], madded['r2_std'].iloc[0]) )
            plt.show()
        elif (len(madded) > 1):
            print('Error: Found more than one entry for added value model for %s' % added_valuegroups[ia][0].decode('UTF-8') )
        else:
            print('No tuning curve for %s' % added_valuegroups[ia][0].decode('UTF-8'))

        
    print('--------------------- Summary -----------------------------------')
    plt.figure()
    plt.subplot(1,3,1)
    plt.hist(r2_full_all[np.logical_not(np.isnan(r2_full_all))])
    plt.xlabel('R2')
    plt.ylabel('Count')
    plt.title('Full Model')
    
    plt.subplot(1,3,2)
    plt.errorbar(range(len(feature_groups)), np.nanmean(r2_diff_all, axis=0), yerr = 2.0*(np.nanstd(r2_diff_all, axis=0, ddof=1))/np.sqrt(nfeatures_neural-sum(np.isnan(r2_diff_all))) , fmt='ko')
    plt.axis([-0.5, 7.5, -0.05, 0.25]) 
    plt.hlines(0, -1, 8, colors='k', linestyles='dashed')
    plt.xlabel('Group')
    plt.ylabel('R2 diff')
    plt.title('Raw')
    
    plt.subplot(1,3,3)
    plt.errorbar(range(len(feature_groups)), np.nanmean(r2_reldiff_all, axis=0), yerr = 2.0*(np.nanstd(r2_reldiff_all, axis=0, ddof=1))/np.sqrt(nfeatures_neural-sum(np.isnan(r2_reldiff_all))) , fmt='ko')
    plt.axis([-0.5, 7.5, -0.1, 1.0]) 
    plt.hlines(0, -1, 8, colors='k', linestyles='dashed')
    plt.xlabel('Group')
    plt.ylabel('R2 diff')
    plt.title('Relative')
    plt.show()
    
#-----------------------------------------------------------------------------

# Encoder wrapper for analysing how acoustical features predict neural response
batchflg = True

if batchflg:
    decomps = ['full_psds', 'spikes']
    exp_names = ['YelBlu6903F', 'BlaBro09xxF', 'GreBlu9508M', 'WhiWhi4522M' ]
    plot_me = False
else:
    decomps = ['full_psds']   
    exp_names = ['GreBlu9508M']
    plot_me = True
    

# This is the stuff that the wrapper will read

for exp_name in exp_names:

    if batchflg:
        preproc_dir = '/auto/tdrive/mschachter/data/%s/preprocess' % exp_name
        encoder_dir = '/auto/tdrive/mschachter/data/%s/encoder' % exp_name
    else:            
        preproc_dir = '/Users/frederictheunissen/Documents/Data/mschachter/%s/preprocess' % exp_name
        encoder_dir = '/Users/frederictheunissen/Documents/Data/mschachter/%s/encoder' % exp_name

    seg_list = []
    for fname in os.listdir(preproc_dir):
        for decomp in decomps:
            if fname.endswith('R_%s.h5' % decomp) and fname.beginswith('preproc_'):
                segname = fname.split('_')[1] + '_' +fname.split('_')[2] + '_' + fname.split('_')[3]
                seg_list.append(segname)
            if fname.endswith('L_%s.h5' % decomp) and fname.beginswith('preproc_'):
                segname = fname.split('_')[1] + '_' +fname.split('_')[2] + '_' + fname.split('_')[3]
                seg_list.append(segname)
                
    if batchflg:
        print('Exp %s:' % exp_name)
        print('\t %s' % seg_list)

#    for seg_uname in seg_list:
#        for decomp in decomps:
#            
#            preproc_file = os.path.join(preproc_dir, 'preproc_%s_%s.h5' % (seg_uname, decomp))
#            output_file = os.path.join(encoder_dir, 'encoder_%s_%s.h5' % (seg_uname, decomp))
#
#
#            # Running all the encoders
#            resultsDataFrame = run_encoder(preproc_file)
#
#            # Save the results
#            resultsDataFrame.to_pickle(output_file)
#
#            # Plotting all the results
#            if plot_me:
#                plot_encoder(preproc_file, resultsDataFrame)

