import numpy as np
import pandas as pd
import os

TEMPERATURE_FILE_PATH = './data/temperature_history.csv'
LOAD_FILE_PATH = './data/Load_history.csv'

def normalization(data, if_mean=True, if_std=True, if_log = False):
    if if_log:
        data = np.log(data + 1)
    df = pd.DataFrame(data=data)
    rst = df
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    if if_mean:
        rst = rst.sub(mean, axis=1)
    if if_std:
        rst = rst.div(std, axis=1)
    return np.array(mean), np.array(std), np.array(rst)

def de_normalization(mean,std,data,if_mean=True, if_std=True, if_log = False):
    df = pd.DataFrame(data=data,copy=True)
    rst = df
    if if_mean:
        rst = rst.mul(std, axis=1)
    if if_std:
        rst = rst.add(mean, axis=1)
    rst = np.array(rst)
    if if_log:
        scale_array = np.empty_like(rst)
        scale_array.fill(np.e)
        rst = np.power(scale_array, rst) - 1
    return rst


def log(data):
    df = pd.DataFrame(data=data)
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    return np.array(mean),np.array(std), \
           np.array(df.sub(mean,axis=1).div(std,axis=1))

def _read_load_from_raw(if_full=False):
    print 'Making load data from raw...'
    raw_load_file = LOAD_FILE_PATH
    if if_full:
        raw_load_file = './data/full_load_history.csv'
    raw_data = pd.read_csv(raw_load_file, sep=',')
    T = []
    L = []
    for name, group in raw_data.groupby('zone_id'):
        if name == 1:
            for row in group.iterrows():
                tstep = np.zeros((4, 1), dtype=float)
                for i in range(1, 4):
                    if i == 1:
                        tstep[i - 1][0] = row[1][i]
                    else:
                        tstep[i - 1][0] = row[1][i]
                for i in range(4, len(row[1])):
                    temp = np.zeros((20, 1), dtype=float)
                    if row[1][i] != np.nan:
                        tstep_temp = np.copy(tstep)
                        tstep_temp[3][0] = i - 3
                        if isinstance(row[1][i], str):
                            temp[0][0] = float(row[1][i].replace(',', ''))
                        else:
                            temp[0][0] = float(row[1][i])
                        T.append(tstep_temp)
                        L.append(temp)
        else:
            index = int(name)
            count = 0
            for row in group.iterrows():
                for i in range(4, len(row[1])):
                    if row[1][i] != np.nan:
                        if isinstance(row[1][i], str):
                            L[count][
                                index - 1][0] = \
                                float(row[1][i].replace(',', ''))
                        else:
                            L[count][index - 1][0] = float(row[1][i])
                        count += 1
    if len(L) != len(T):
        raise ValueError("TypeError")
    length = len(T)
    Time = np.array(T, dtype=np.float64)
    Load = np.array(L, dtype=np.float64)
    _load_to_csv(length, Time, Load, if_full)
    Time = Time.reshape(length, 1, 4)
    Load = Load.reshape(length, 1, 20)
    print 'Done'
    return length, Time, Load


def _load_to_csv(length, Time, Load, if_full=False):
    load_file = './data/load.csv'
    time_load_file = './data/time_load.csv'
    if if_full:
        load_file = './data/full_load.csv'
        time_load_file = './data/time_full_load.csv'
    time_df = pd.DataFrame(Time.reshape(length, 4))
    time_df.to_csv(time_load_file, header=False, index=False)
    load_df = pd.DataFrame(Load.reshape(length, 20))
    load_df.to_csv(load_file, header=False, index=False)


def _temp_to_csv(length, Time, Temp):
    time_df = pd.DataFrame(Time.reshape(length, 4))
    time_df.to_csv('./data/time_temp.csv', header=False, index=False)
    temp_df = pd.DataFrame(Temp.reshape(length, 11))
    temp_df.to_csv('./data/temp.csv', header=False, index=False)


def _read_temp_from_raw():
    print 'Making temperature data from raw...'
    raw_data = pd.read_csv(TEMPERATURE_FILE_PATH, sep=',')
    T = []
    TIME = []
    for name, group in raw_data.groupby('station_id'):
        if name == 1:
            for row in group.iterrows():
                tstep = np.zeros((4, 1), dtype=float)
                for i in range(1, 4):
                    if i == 1:
                        tstep[i - 1][0] = row[1][i]
                    else:
                        tstep[i - 1][0] = row[1][i]
                for i in range(4, len(row[1])):
                    temp = np.zeros((11, 1), dtype=float)
                    if row[1][i] != np.nan:
                        temp[0][0] = row[1][i]
                        tstep_temp = np.copy(tstep)
                        tstep_temp[3][0] = i - 3
                        TIME.append(tstep_temp)
                        T.append(temp)
        else:
            index = int(name)
            count = 0
            for row in group.iterrows():
                for i in range(4, len(row[1])):
                    if row[1][i] != np.nan:
                        T[count][index - 1][0] = row[1][i]
                        count += 1
    if len(TIME) != len(T):
        raise ValueError("TypeError")
    length = len(TIME)
    Time = np.array(TIME, dtype=np.float64)
    Temp = np.array(T, dtype=np.float64)
    _temp_to_csv(length, Time, Temp)
    Time = Time.reshape(length, 1, 4)
    Temp = Temp.reshape(length, 1, 11)
    print 'Done'
    return length, Time, Temp


def read_temperature_history():
    if os.path.exists('./data/temp.csv') and \
       os.path.exists('./data/time_temp.csv'):
        Time = np.array(pd.read_csv('./data/time_temp.csv', header=None))
        Temp = np.array(pd.read_csv('./data/temp.csv', header=None))
        if len(Time) != len(Temp):
            raise ValueError("TypeError")
        length = len(Time)
        Time = Time.reshape(length, 1, 4)
        Temp = Temp.reshape(length, 1, 11)
        return length, Time, Temp
    else:
        return _read_temp_from_raw()


def read_load_history(if_full=False):
    load_file = './data/load.csv'
    time_load_file = './data/time_load.csv'
    if if_full:
        load_file = './data/full_load.csv'
        time_load_file = './data/time_full_load.csv'
    if os.path.exists(load_file) and \
       os.path.exists(time_load_file):
        Load = np.array(pd.read_csv(load_file, header=None))
        Time = np.array(pd.read_csv(time_load_file, header=None))
        if len(Time) != len(Load):
            raise ValueError("TypeError")
        length = len(Load)
        Time = Time.reshape(length, 1, 4)
        Load = Load.reshape(length, 1, 20)
        return length, Time, Load
    else:
        return _read_load_from_raw(if_full)

def _read_benchmark_from_raw():
    if os.path.exists('./data/predict.csv'):
        raw_data = pd.read_csv('./data/predict.csv', header=None)
        L = []
        for name, group in raw_data.groupby(0):
            if name == 1:
                for row in group.iterrows():
                    for i in range(1, len(row[1])):
                        temp = np.zeros((21, 1), dtype=float)
                        if row[1][i] != np.nan:
                            temp[0][0] = row[1][i]
                            L.append(temp)
            else:
                index = int(name)
                count = 0
                for row in group.iterrows():
                    for i in range(1, len(row[1])):
                        if row[1][i] != np.nan:
                            L[count][index - 1][0] = row[1][i]
                            count += 1
        L = np.array(L)
        L_tf = pd.DataFrame(L.reshape((len(L),21)))
        L_tf.to_csv('./data/expected_result.csv', header=False, index=False)
        return L.reshape((len(L),1,21))

def read_benchmark():
    if os.path.exists('./data/expected_result.csv'):
        Bechmark = np.array(pd.read_csv('./data/expected_result.csv', header=None))
        return Bechmark.reshape(length, 1, 21)
    else:
        return _read_benchmark_from_raw()



def main():
    # length, Time, Temp = read_temperature_history()
    # # mean, std ,Temp= normalization(Temp.reshape(len(Temp),11))
    # print length, len(Time), len(Temp), len(Time[0][0]), len(Temp[0][0])
    length, Time, Load = read_load_history(if_full=True)
    print length, len(Time), len(Load), len(Time[0]), len(Load[0])
    # print len(read_benchmark())

if __name__ == "__main__":
    main()
