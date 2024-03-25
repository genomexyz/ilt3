import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from metar import Metar

#setting
data_warning_filename = 'llwas/llwas_warning.csv'
data_wspd_filename = 'llwas/llwas_speeddir.csv'
dt_begin = datetime(2020, 2, 1, 0, 0)
dt_end = datetime(2020, 5, 12, 0, 0)
wspd_col = ['A1 - Kedaung speed[kt]', 'A1 - Kedaung dir [deg]',
       'A2 - Airnav Tower 710 speed[kt]', 'A2 - Airnav Tower 710 dir [deg]',
       'A3 - Airnav Tower 720 speed[kt]', 'A3 - Airnav Tower 720 dir [deg]',
       'A4 - Pintu Kapuk speed[kt]', 'A4 - Pintu Kapuk dir [deg]',
       'A5 - BMKG Old LLWAS Tower 2 speed[kt]',
       'A5 - BMKG Old LLWAS Tower 2 dir [deg]', 'A6 - Batu Ceper speed[kt]',
       'A6 - Batu Ceper dir [deg]', 'A7 - Benda speed[kt]',
       'A7 - Benda dir [deg]', 'A8 - Bayur speed[kt]', 'A8 - Bayur dir [deg]',
       'A9 - Rawa Burung speed[kt]', 'A9 - Rawa Burung dir [deg]',
       'A10 - Kelor speed[kt]', 'A10 - Kelor dir [deg]',
       'A11 - Comm Tower speed[kt]', 'A11 - Comm Tower dir [deg]',
       'A12 - BMKG Old LLWAS Tower 1 speed[kt]',
       'A12 - BMKG Old LLWAS Tower 1 dir [deg]']
warning_col = ['RWY 07LA', 'RWY 25RD', 'RWY 25RA', 'RWY 07LD',
       'RWY 07RA', 'RWY 25LD', 'RWY 25LA', 'RWY 07RD']

warning_df = pd.read_csv(data_warning_filename)
wspd_df = pd.read_csv(data_wspd_filename)
print(wspd_df)
print(wspd_df.keys())
print(warning_df.keys())



warning_time = list(warning_df['Time (UTC)'])
wspd_time = list(wspd_df['Time (UTC)'])
warning_param = warning_df.loc[:, warning_col]
wspd_param = wspd_df.loc[:, wspd_col]
print(wspd_time[0], wspd_time[-1])
print(wspd_param)
exit()

#18.04.2020 04:18:00
while dt_begin < dt_end:
    dt_str = dt_begin.strftime('%d.%m.%Y %H:%M:%S')
    try:
        dt_idx_warning = warning_time.index(dt_str)
        dt_idx_wspd = wspd_time.index(dt_str)
    except ValueError:
        dt_begin += timedelta(seconds=10)
        print('not found %s'%(dt_str))
        continue
    print(dt_idx_warning, dt_idx_wspd)
    dt_begin += timedelta(seconds=10)