import pandas as pd
import numpy as np

feature = np.load('param_wiii_ts.npy')
label = np.load('label_wiii_ts.npy')

df_dict = {}
df_dict['Convective Available Potential Energy'] = feature[:, 0]
df_dict['K Index'] = feature[:, 1]
df_dict['Cross totals index'] = feature[:, 2]
df_dict['Vertical totals index'] = feature[:, 3]
df_dict['Lifted index'] = feature[:, 4]
df_dict['Showalter index'] = feature[:, 5]
df_dict['Thunderstorm'] = label

new_df = pd.DataFrame(df_dict)
new_df.to_excel('ts_feature.xlsx')