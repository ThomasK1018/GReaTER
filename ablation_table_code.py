# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:19:38 2024

@author: thoma
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


task_ids = [10005, 10006, 14584, 22100, 31941, 31996, 34382, 34975]
types = ['direct_combine', 'cdr_01', 'cdr_03', 'hierarchical', 'auto_mapping_reduction', 'replace_special_symbol_reduction']


average_results = np.zeros([len(types), 3])
max_results = np.zeros([len(types), 3])
min_results = np.zeros([len(types), 3])


for j in range(len(types)):
    results = np.zeros([len(task_ids), 3])

    thres = 0.333
    
    for i in range(len(task_ids)):
        dp = pd.read_csv(f"results/p_val_{task_ids[i]}_{types[j]}.csv")

        results[i, 0] = sum(dp['New'] - dp['Old'] >= thres)
        results[i, 2] = sum(dp['New'] - dp['Old'] <= -thres)
        results[i, 1] = sum((dp['New'] - dp['Old'] <= thres) & (dp['New'] - dp['Old'] >= -thres))
            
    average_results[j] = np.mean(results, axis = 0)
    max_results[j] = np.max(results, axis = 0)
    min_results[j] = np.min(results, axis = 0)





new_index = ['Direct Multiplication', 'CDR with Threshold 0.1', 'CDR with Threshold 0.3', 'Hierarchical Clustering', 'Auto Categorical Mapping with CDR', 'Replacing ^ with and']

average_results = pd.DataFrame(average_results, index = new_index, columns = ['Improved', 'No Change', 'Worsened'])
max_results = pd.DataFrame(max_results, index = new_index, columns = ['Improved', 'No Change', 'Worsened'])
min_results = pd.DataFrame(min_results, index = new_index, columns = ['Improved', 'No Change', 'Worsened'])


combined_df = pd.DataFrame()

for col in min_results.columns:
    combined_df[col] = min_results[col].combine(average_results[col], lambda x1, x2: [x1, x2])
    combined_df[col] = combined_df[col].combine(max_results[col], lambda x12, x3: x12 + [x3])
    
    
