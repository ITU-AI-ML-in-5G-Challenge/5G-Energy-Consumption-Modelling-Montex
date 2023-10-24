import torch
import torch.utils.data as utils
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm


class Dataset:
    def __init__(self, data_path, eval_time_stamps, seq_len=1, eval_percentage=0.1):
        self.data = pd.read_csv(data_path)
        # print(self.data.head(1))

        eval_timestamps_number = 0
        self.data_x_series = []
        self.data_y_series = []
        self.validation_data_x_series = []
        self.validation_data_y_series = []
                                                 
        self.evaluation_data_x_series = []
        self.evaluation_datastamps = []
        bs = None
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
        cnt_else = 0

        np.random.seed(121)

        for index, row in tqdm(self.data.iterrows(), desc=f"Rec data prep progress", total=len(self.data)):
            row_np = row.to_numpy()
            row_x, row_y = row_np[2:-1], row_np[-1]
            # row_x, row_y = row_np[0:-1], row_np[-1]
            # print(row_x, row_y)
            if bs != row.BS:
                #  zmiana BS
                date = pd.Timestamp(row.Time)
                bs = row.BS
                my_deque = deque([row_x]*seq_len, maxlen=seq_len) #specjalna struktura danych
            elif (pd.Timestamp(row.Time) - date).total_seconds() / 3600 == 1:
                cnt1 += 1
                # kolejna probka dokladnie godzine do przodu
                date = pd.Timestamp(row.Time)
                my_deque.append(row_x)
            elif (pd.Timestamp(row.Time) - date).total_seconds() / 3600 == 2 and seq_len > 2:
                cnt2 += 1
                # kolejna probka 2 godziny do przodu
                date = pd.Timestamp(row.Time)
                my_deque.append(row_x) #moze by tu przykurwic srednia arytmetyczna?
                my_deque.append(row_x)
            elif (pd.Timestamp(row.Time) - date).total_seconds() / 3600 == 3 and seq_len > 3:
                cnt3 += 1
                # kolejna probka 3 godziny do przodu
                date = pd.Timestamp(row.Time)
                my_deque.append(row_x) #moze by tu przykurwic srednia arytmetyczna?
                my_deque.append(row_x)
                my_deque.append(row_x)
            elif (pd.Timestamp(row.Time) - date).total_seconds() / 3600 == 4 and seq_len > 4:
                cnt4 += 1
                # kolejna probka 4 godziny do przodu
                date = pd.Timestamp(row.Time)
                my_deque.append(row_x) #moze by tu przykurwic srednia arytmetyczna?
                my_deque.append(row_x)
                my_deque.append(row_x)
                my_deque.append(row_x)
            elif (pd.Timestamp(row.Time) - date).total_seconds() / 3600 == 5 and seq_len > 5:
                cnt5 += 1
                # kolejna probka 5 godziny do przodu
                date = pd.Timestamp(row.Time)
                my_deque.append(row_x) #moze by tu przykurwic srednia arytmetyczna?
                my_deque.append(row_x)
                my_deque.append(row_x)
                my_deque.append(row_x)
                my_deque.append(row_x)
            else:
                cnt_else+=1
                # wiecej niz piec godzin do przodu  
                my_deque = deque([row_x]*seq_len, maxlen=seq_len)

            #sprawdzanie czy probka pasuje do setu ewaluacyjnego
            result = (eval_time_stamps['Time'] == row.Time) & (eval_time_stamps['BS'] == row.BS)
            if result.any():
                # print("no nie gadaj ze to dziala")
                eval_timestamps_number += 1
                # eval_time_stamps.drop(eval_time_stamps[result].index, inplace=True)
                self.evaluation_data_x_series.append(np.array(my_deque))
                self.evaluation_datastamps.append((row.BS, row.Time))
                # print(row.BS, row.Time, row_y)
                if not np.isnan(row_y):
                    self.data_x_series.append(np.array(my_deque))
                    self.data_y_series.append(row_y)
            else:
                random_number = np.random.rand()
                if random_number < eval_percentage:
                    self.validation_data_x_series.append(np.array(my_deque))
                    self.validation_data_y_series.append(row_y)
                else:
                    self.data_x_series.append(np.array(my_deque))
                    self.data_y_series.append(row_y)
        
        print(f"{cnt1} probek 1h do przodu, {cnt2} probek 2h do przodu, {cnt3} probek 3h do przodu,{cnt4} probek 4h do przodu,{cnt5} probek 5h do przodu, {cnt_else} probek >5h do przodu")
        print(f"Udalo sie wypelnic {eval_timestamps_number}/{eval_time_stamps.shape[0]} probek eval")


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
    
    def get_validation_data(self):
        return np.array(self.validation_data_x_series, dtype=float), np.array(self.validation_data_y_series, dtype=float).reshape(-1,1)
    
    def get_eval_data(self):
        return np.array(self.evaluation_data_x_series, dtype=float), self.evaluation_datastamps
        # return np.array(self.evaluation_data_x_series), self.evaluation_datastamps

    def get_data(self):
        return np.array(self.data_x_series, dtype=float), np.array(self.data_y_series, dtype=float).reshape(-1,1)
    
# data = Dataset(data_path="./prepared_data/One_Cell_merged_all.csv", seq_len=3)
# data_x, data_y = data.get_data()
# data_x = np.array(data_x)
# data_y =  np.array(data_y)
# print(data_x.shape, data_y.shape)
