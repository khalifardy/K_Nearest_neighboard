# %%
import pandas as pd
import numpy as np
from library.metode import KnearestNeighbors
# %%
dwn_url = 'https://drive.google.com/uc?id=' + '1C5-s_COuWrjn52wzs-QtEYc6QE_IlR2O'
df = pd.read_csv(dwn_url)
# %%
copy_data = df.copy()
copy_data = copy_data.drop("Id", axis=1)
# %%
copy_data = copy_data.sample(150).reset_index(drop=True)
# %%
data_fold1 = (copy_data[:50].reset_index(drop=True),
              copy_data[50:150].reset_index(drop=True))
data_fold2 = (copy_data[50:100].reset_index(drop=True), pd.concat(
    [copy_data[:50], copy_data[100:]]).reset_index(drop=True))
data_fold3 = (copy_data[100:].reset_index(drop=True),
              copy_data[:150].reset_index(drop=True))
# %%
knn_ = KnearestNeighbors(5)

# %%
folds = [data_fold1, data_fold2, data_fold3]
# %%
knn_.get_average_accuracy("euclidean", folds, "Species", cetak=True)
# %%
knn_.plot_evaluasi_k("euclidean", folds, "Species", 1, 10)

# %%
