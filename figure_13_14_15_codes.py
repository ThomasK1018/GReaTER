import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


task_ids = [10005, 10006, 14584, 22100, 31941, 31996, 34382, 34975]



choices = ['manual_mapping_reduction', 'auto_mapping_reduction', 'replace_special_symbol_reduction']

# Please choose respective index
choice = choices[0]

fig, axes = plt.subplots(2, 4, figsize=(12, 8))

for i in range(len(task_ids)):
    dp = pd.read_csv(f"results/p_val_{task_ids[i]}_{choice}.csv")
    sns.kdeplot(data = dp['Old'], clip = (0, 1), alpha = 0.2, ax = axes[i // 4, i % 4], label = 'Original DEREC')
    sns.kdeplot(data = dp['New'], clip = (0, 1), alpha = 1, ax = axes[i // 4, i % 4], label = 'New Method')
    axes[i // 4, i % 4].set_xlabel("P-Values")
    axes[i // 4, i % 4].set_title(f"{task_ids[i]}")
    
    
    
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol = 2, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
plt.show()