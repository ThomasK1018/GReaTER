
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


task_ids = [10005, 10006, 14584, 22100, 31941, 31996, 34382, 34975]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

types = ['direct_combine', 'cdr_01', 'cdr_03', 'hierarchical']
new_index = ['Direct Multiplication', 'Correlation Threshold 0.1', 'Correlation Threshold 0.3', 'Hierarchical Clustering']



for j in range(len(types)):
    sum_dp = pd.read_csv(f"results/p_val_{task_ids[0]}_{types[j]}.csv", index_col = 0)

    for i in range(1, len(task_ids)):
        dp = pd.read_csv(f"results/p_val_{task_ids[i]}_{types[j]}.csv", index_col = 0)
        sum_dp += dp
        
            
    sum_dp = sum_dp.dropna()
    average_dp = sum_dp / len(task_ids)

    sns.ecdfplot(data = average_dp['New'], alpha = 0.5, ax = axes[1], label = new_index[j])  
    sns.kdeplot(data = average_dp['New'], clip = (0, 1), alpha = 0.5, ax = axes[0], label = new_index[j])  
    
sns.kdeplot(data = average_dp['Old'], clip = (0, 1), alpha = 0.5, ax = axes[0], label = 'Original DEREC') 
sns.ecdfplot(data = average_dp['Old'], alpha = 0.5, ax = axes[1], label = 'Original DEREC')  

lines = axes[0].get_lines()
for line in lines:
    if new_index[1] in line.get_label():
        line.set_linestyle('--') 
        line.set_alpha(1)
        
lines = axes[1].get_lines()
for line in lines:
    if new_index[1] in line.get_label():
        line.set_linestyle('--') 
        line.set_alpha(1)



handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol = 5, bbox_to_anchor=(0.5, 0.02))

axes[0].set_title("P-values Distribution Plot")
axes[0].set_xlabel("P-values")
axes[1].set_xlabel("P-values")
axes[0].set_ylabel("Density")
axes[1].set_ylabel("Cumulative Density")
plt.show()