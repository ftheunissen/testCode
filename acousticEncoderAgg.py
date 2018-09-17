#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 10:29:39 2018

@author: frederictheunissen
"""
import os
import pandas as pd
from acousticEncoder import plot_encoder

#-----------------------------------------------------------------------------

# Encoder wrapper for analysing how acoustical features predict neural response
batchflg = False  # To run on cluster, otherwise on Frederic's laptop

if batchflg:
    decomps = ['full_psds', 'spikes']
    exp_names = ['YelBlu6903F', 'GreBlu9508M', 'WhiWhi4522M' ]
    agg_file = '/auto/tdrive/mschachter/data/aggregate/encoder_all.pkl'
    plot_me = False
else:
    decomps = ['full_psds']   
    exp_names = ['GreBlu9508M']
    agg_file = '/Users/frederictheunissen/Documents/Data/mschachter/aggregate/encoder_all.pkl'
    plot_me = True
    
iframe = 0
for exp_name in exp_names:

    if batchflg:
        preproc_dir = '/auto/tdrive/mschachter/data/%s/preprocess' % exp_name
        encoder_dir = '/auto/tdrive/mschachter/data/%s/encoder' % exp_name
    else:            
        preproc_dir = '/Users/frederictheunissen/Documents/Data/mschachter/%s/preprocess' % exp_name
        encoder_dir = '/Users/frederictheunissen/Documents/Data/mschachter/%s/encoder' % exp_name
    
    # Make the output directory if it does not exist.
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)
        
    seg_list = []
    for fname in os.listdir(preproc_dir):
        for decomp in decomps:
            if fname.endswith('R_%s.h5' % decomp) and fname.startswith('preproc_'):
                segname = fname.split('_')[1] + '_' +fname.split('_')[2] + '_' + fname.split('_')[3]
                seg_list.append(segname)
            if fname.endswith('L_%s.h5' % decomp) and fname.startswith('preproc_'):
                segname = fname.split('_')[1] + '_' +fname.split('_')[2] + '_' + fname.split('_')[3]
                seg_list.append(segname)
                

    for seg_uname in seg_list:
        for decomp in decomps:
            
            # Input and Output files of the decoder
            preproc_file = os.path.join(preproc_dir, 'preproc_%s_%s.h5' % (seg_uname, decomp))
            output_file = os.path.join(encoder_dir, 'encoder_%s_%s.pkl' % (seg_uname, decomp))

           # Read the results         
            resultsDataFrame = pd.read_pickle(output_file)
           
           # Add columns to specify origin of theresults  
            nrows = len(resultsDataFrame.index)
            resultsDataFrame['experiment'] = exp_name
            resultsDataFrame['site'] = seg_uname
            resultsDataFrame['decomp'] = decomp
            
           
            # Plotting all the results
            if plot_me:
                plot_encoder(preproc_file, resultsDataFrame)

            if iframe == 0:
                resultsAgg = resultsDataFrame
            else:
                resultsAgg = pd.concat([resultsAgg, resultsDataFrame], ignore_index = True )
            
            iframe += 1

print('%d dataframes were aggregated' % (iframe +1))

# Saving aggregate data file
resultsAgg.to_pickle(agg_file)