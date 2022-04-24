#%% Display

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('output.csv', index_col=0)
print(len(data))
plt.plot([1, 21, 41, 61, 81], np.transpose(data.values))




# %%
plt.plot([1, 21, 41, 61, 81], np.mean(np.transpose(data.values), axis=1))
