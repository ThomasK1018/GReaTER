
import pandas as pd
import numpy as np
import scipy
from Data_Clean_Room import data_clean_room
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.cluster import hierarchy


def cramer_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, _, _ = scipy.stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def is_numeric(x):
    try:
        pd.to_numeric(x)
        return True
    except ValueError:
        return False

task_id = 10005

d1 = pd.read_csv(f"datasets/task_id_{task_id}/feeds.csv")
d2 = pd.read_csv(f"datasets/task_id_{task_id}/ads.csv")

if 'log_id' in d2.columns:
    d2 = d2.drop('log_id', axis = 1)

dcr = data_clean_room(d1, d2, 'user_id')
dcr.derec()
dcr.sampling(200)

c1 = dcr.derec_child_1_small
c2 = dcr.derec_child_2_small

special_col_list = ['user_id']
normal_col_list = ['user_id']
c = pd.merge(c2, c1, left_on = 'user_id', right_on = 'user_id')



for col in c.columns:
    if col != 'user_id':
        if isinstance(c[col][0], str) and c[col].str.contains('^').any():
            special_col_list.append(col)
        elif isinstance(c[col][0], datetime):
            special_col_list.append(col)
        else:
            normal_col_list.append(col)
      


special_col = c[special_col_list].drop_duplicates()
normal_col = c[normal_col_list].drop_duplicates()


def plot_cor_heatmap(df):
    cor = np.zeros([len(df.columns), len(df.columns)])

    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            cor[i, j] = cramer_v(df.iloc[:, i], df.iloc[:, j])

    cor = pd.DataFrame(cor, index = df.columns, columns = df.columns)



    pdist = hierarchy.distance.pdist(cor)
    linkage = hierarchy.linkage(pdist, method='average')
    idx = hierarchy.fcluster(linkage, 0.5 * pdist.max(), 'distance')


    order = []
    subsets = {}
    for i in range(len(np.unique(idx))):
        order.extend(np.where(idx == i + 1)[0])
        subsets[f"Subset_{i + 1}"] = ['user_id'] + list(cor.columns[np.where(idx == i + 1)[0]])

    correlation_matrix_reordered = cor.iloc[order, order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_reordered, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Reordered Correlation Matrix')
    plt.show()
    return cor

cor_all = plot_cor_heatmap(c)
cor_cat = plot_cor_heatmap(normal_col)
    