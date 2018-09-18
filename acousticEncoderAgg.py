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
    exp_names = ['YelBlu6903F', 'GreBlu9508M', 'WhiWhi4522M' ]
    agg_file = '/auto/tdrive/mschachter/data/aggregate/encoder_all.pkl'
    plot_me = False
else:
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
      
    # Find all the output files in the decoder folders.    
    seg_list = []
    dec_list = []
    fil_list = []
    for fname in os.listdir(encoder_dir):
        if fname.endswith('.pkl'):
            segname = fname.split('_')[1] + '_' + fname.split('_')[2] + '_' + fname.split('_')[3]
            decomp = fname.split('_')[4] + '_' + fname.split('_')[5]
            decomp = decomp.split('.')[0]
            fil_list.append(fname)
            seg_list.append(segname)
            dec_list.append(decomp)
                
    for i, fname in enumerate(fil_list):
        
        # Each output file corresponds to one experiment, one site and one neural decomposition
        segname = seg_list[i]
        decomp  = dec_list[i]
        
        # Input and Output files of the decoder
        preproc_file = os.path.join(preproc_dir, 'preproc_%s_%s.h5' % (segname, decomp))
        encoder_file = os.path.join(encoder_dir, '%s' % fname )

        # Read the results         
        resultsDataFrame = pd.read_pickle(encoder_file)
           
        # Add columns to specify origin of theresults  
        nrows = len(resultsDataFrame.index)
        resultsDataFrame['experiment'] = exp_name
        resultsDataFrame['site'] = segname
        resultsDataFrame['decomp'] = decomp
            
           
        # Plotting all the results
        if plot_me:
            plot_encoder(preproc_file, resultsDataFrame)

        if iframe == 0:
            resultsAgg = resultsDataFrame
        else:
            resultsAgg = pd.concat([resultsAgg, resultsDataFrame], ignore_index = True )
        iframe = iframe +1
            

print('%d dataframes were aggregated' % (iframe +1))

# Saving aggregate data file
resultsAgg.to_pickle(agg_file)