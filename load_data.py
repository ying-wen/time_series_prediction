import numpy as np
import pandas as pd
import os

TEMPERATURE_FILE_PATH = './data/temperature_history.csv'
LOAD_FILE_PATH = './data/Load_history.csv'


def _read_load_from_raw():
    print 'Making load data from raw...'
    raw_data = pd.read_csv(LOAD_FILE_PATH, sep=',')
    T = []
    L = []
    for name, group in raw_data.groupby('zone_id'):
        if name == 1:
            for row in group.iterrows():
                tstep = np.zeros((4, 1), dtype=float)
                for i in range(1, 4):
                    if i == 1:
                        tstep[i - 1][0] = row[1][i] - 2004
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
    Time = np.array(T, dtype=float)
    Load = np.array(L, dtype=float)
    _load_to_csv(length, Time, Load)
    Time = Time.reshape(length, 4, 1)
    Load = Load.reshape(length, 20, 1)
    print 'Done'
    return length, Time, Load


def _load_to_csv(length, Time, Load):
    time_df = pd.DataFrame(Time.reshape(length, 4))
    time_df.to_csv('./data/time_load.csv', header=False, index=False)
    load_df = pd.DataFrame(Load.reshape(length, 20))
    load_df.to_csv('./data/load.csv', header=False, index=False)


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
                        tstep[i - 1][0] = row[1][i] - 2004
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
    Time = np.array(TIME, dtype=float)
    Temp = np.array(T, dtype=float)
    _temp_to_csv(length, Time, Temp)
    Time = Time.reshape(length, 4, 1)
    Temp = Temp.reshape(length, 11, 1)
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


def read_load_history():
    if os.path.exists('./data/load.csv') and \
       os.path.exists('./data/time_load.csv'):
        Load = np.array(pd.read_csv('./data/load.csv', header=None))
        Time = np.array(pd.read_csv('./data/time_load.csv', header=None))
        if len(Time) != len(Load):
            raise ValueError("TypeError")
        length = len(Load)
        Time = Time.reshape(length, 4, 1)
        Load = Load.reshape(length, 20, 1)
        return length, Time, Load
    else:
        return _read_load_from_raw()


def main():
    length, Time, Temp = read_temperature_history()
    print length, len(Time), len(Temp), len(Time[0]), len(Temp[0])
    length, Time, Load = read_load_history()
    print length, len(Time), len(Load), len(Time[0]), len(Load[0])

if __name__ == "__main__":
    main()
