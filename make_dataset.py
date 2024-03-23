import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from metar import Metar

#setting
data_dir = 'label_raw_wiii'
stasiun = 'WIII'
limit_sandi = ['TEMPO', 'BECMG']
start_date = datetime(2013,1,1)
end_date = datetime(2023,1,1)
flag_none = np.nan

def create_param(raw_data_split, first_time = True):
    temperature = flag_none
    vis = flag_none
    wind_dir = flag_none
    wind_speed = flag_none
    wx = flag_none
    pressure = flag_none
    dew = flag_none
    try:
        obs = Metar.Metar(raw_data_split)
    except Metar.ParserError:
        print('sandi %s invalid, continue...'%raw_data_split)
        if first_time:
            #recreate sandi string
            raw_data_split_cnt = raw_data_split.split()
            raw_data_split2 = []
            for iter_split in range(len(raw_data_split_cnt)):
                raw_data_split2.append(raw_data_split_cnt[iter_split])
                if raw_data_split_cnt[iter_split][0] == 'Q':
                    break
            raw_data_split2 = ' '.join(raw_data_split2)
            #print('SPLIT 2ND CHANCE', raw_data_split2)
            create_param(raw_data_split2, first_time = False)
        return None

    if obs.type != 'METAR':
        print('not METAR, continue...')
        return None

    try:
        temperature = obs.temp.value()
    except AttributeError:
        pass

    correction = obs.correction
    if correction is None:
        correction = flag_none
        
    station_code = obs.station_id
    if station_code is None:
        station_code = flag_none

    try:
        vis = obs.vis.value()
    except AttributeError:
        pass
        
    try:
        wind_dir = obs.wind_dir.value()
    except AttributeError:
        pass
        
    try:
        wind_speed = obs.wind_speed.value()
    except AttributeError:
        pass
        
    try:
        wind_gust = obs.wind_gust.value()
    except AttributeError:
        wind_gust = flag_none
        
    try:
        wx = obs.weather[0]
        if wx is None:
            wx = flag_none
        wx_cnt = []
        if wx != flag_none:
            for iter_wx in range(len(wx)):
                if wx[iter_wx] is None:
                    continue
                if wx[iter_wx] == '':
                    continue
                wx_cnt.append(wx[iter_wx])
            wx = ''.join(wx_cnt)
    except IndexError:
        pass
        
    try:
        pressure = obs.press.value()
    except AttributeError:
        pass
        
    try:
        dew = obs.dewpt.value()
    except AttributeError:
        pass

    dict_param = {}
    dict_param['temperature'] = temperature
    dict_param['vis'] = vis
    dict_param['wind_dir'] = wind_dir
    dict_param['wind_speed'] = wind_speed
    dict_param['wx'] = wx
    dict_param['pressure'] = pressure
    dict_param['dew'] = dew
    return dict_param
        

def make_array_from_raw(raw_str):
    array = raw_str.split('\n')
    if array[-1] == '':
        array = array[:-1]
    date_array = []
    sandi_array = []
    for i in range(len(array)):
        fragment = array[i].split('\t')
        #01/05/2014 03:30:00Z
        date_data = datetime.strptime(fragment[0].strip(), '%d/%m/%Y %H:%M:00Z')
        date_array.append(date_data)

        sandi = fragment[2].strip()
        for limit in limit_sandi:
            idx = sandi.find(limit)
            if idx != -1:
                break
        sandi = sandi[:idx]
        sandi_array.append(sandi)
    return date_array, sandi_array

def open_data(filename):
    with open(filename) as f:
        file_raw = f.read()
        date_array, sandi_array = make_array_from_raw(file_raw)
    return date_array, sandi_array


current_date = start_date
all_label = []
all_label_pandas = []

tahun_file = 1970
bulan_file = 0
datetime_arr = []
temperature_arr = []
dew_arr = []
vis_arr = []
winddir_arr = []
windspeed_arr = []
pressure_arr = []
wx_arr = []
while current_date < end_date:
    tahun_file_cur = current_date.year
    bulan_file_cur = current_date.month
    if tahun_file != tahun_file_cur or bulan_file != bulan_file_cur:
        try:
            date_arr, sandi_arr = open_data('%s/%s-%s%s'%(data_dir, stasiun, current_date.strftime('%Y'), current_date.strftime('%m')))
        except FileNotFoundError:
            single_temp = np.nan
            single_dew = np.nan
            single_vis = np.nan
            single_winddir = np.nan
            single_windspeed = np.nan
            single_pressure = np.nan
            single_wx = np.nan

            datetime_arr.append(current_date)
            temperature_arr.append(single_temp)
            dew_arr.append(single_dew)
            vis_arr.append(single_vis)
            winddir_arr.append(single_winddir)
            windspeed_arr.append(single_windspeed)
            pressure_arr.append(single_pressure)
            wx_arr.append(single_wx)
            current_date += timedelta(minutes=30)
            continue
        tahun_file = tahun_file_cur
        bulan_file = bulan_file_cur
    try:
        idx_sandi = date_arr.index(current_date)
        single_sandi = sandi_arr[idx_sandi]
        data_dict = create_param(single_sandi)
        print(single_sandi, data_dict)
        if data_dict is None:
            single_temp = np.nan
            single_dew = np.nan
            single_vis = np.nan
            single_winddir = np.nan
            single_windspeed = np.nan
            single_pressure = np.nan
            single_wx = np.nan
        else:
            single_temp = data_dict['temperature']
            single_dew = data_dict['dew']
            single_vis = data_dict['vis']
            single_winddir = data_dict['wind_dir']
            single_windspeed = data_dict['wind_speed']
            single_pressure = data_dict['pressure']
            single_wx = data_dict['wx']
        datetime_arr.append(current_date)
        temperature_arr.append(single_temp)
        dew_arr.append(single_dew)
        vis_arr.append(single_vis)
        winddir_arr.append(single_winddir)
        windspeed_arr.append(single_windspeed)
        pressure_arr.append(single_pressure)
        wx_arr.append(single_wx)

    except ValueError:
        single_temp = np.nan
        single_dew = np.nan
        single_vis = np.nan
        single_winddir = np.nan
        single_windspeed = np.nan
        single_pressure = np.nan
        single_wx = np.nan

        datetime_arr.append(current_date)
        temperature_arr.append(single_temp)
        dew_arr.append(single_dew)
        vis_arr.append(single_vis)
        winddir_arr.append(single_winddir)
        windspeed_arr.append(single_windspeed)
        pressure_arr.append(single_pressure)
        wx_arr.append(single_wx)
        current_date += timedelta(minutes=30)
        continue
    print(idx_sandi, current_date)
    current_date += timedelta(minutes=30)

temperature_arr = np.array(temperature_arr)
dew_arr = np.array(dew_arr)
vis_arr = np.array(vis_arr)
winddir_arr = np.array(winddir_arr)
windspeed_arr = np.array(windspeed_arr)
pressure_arr = np.array(pressure_arr)

#auto qc
temperature_arr[temperature_arr < 15] = np.nan
temperature_arr[temperature_arr > 40] = np.nan
dew_arr[dew_arr < 0] = np.nan
dew_arr[dew_arr > 40] = np.nan
vis_arr[vis_arr < 100] = np.nan
vis_arr[vis_arr > 10000] = np.nan
windspeed_arr[windspeed_arr < 0] = np.nan
windspeed_arr[windspeed_arr > 40] = np.nan
pressure_arr[pressure_arr < 900] = np.nan
pressure_arr[pressure_arr > 1020] = np.nan

df_dict = {}
df_dict['datetime'] = datetime_arr
df_dict['temperature'] = temperature_arr
df_dict['dew'] = dew_arr
df_dict['vis'] = vis_arr
df_dict['wind_direction'] = winddir_arr
df_dict['wind_speed'] = windspeed_arr
df_dict['pressure'] = pressure_arr
df_dict['wx'] = wx_arr

df_new = pd.DataFrame(df_dict)
df_new.to_excel('wiii_dset_qc.xlsx')