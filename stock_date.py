from datetime import *
import glo
from jqdatasdk import *
import numpy as np
import pandas as pd


class StockDate:
    def __init__(self):
        # np[str]
        self.date_list = np.array(
            pd.read_csv('Data/'+glo.stock_code.replace(".", "_") + ".csv")['Unnamed: 0'])
        # datetime
        self.date = datetime.strptime(self.date_list[0], "%Y-%m-%d %H:%M:%S")
        self.index = 0

    def set_date(self, date):
        # 输入datetime
        self.date = date
        self.index = self.find_index(0, self.date_list.size)

    def set_date_with_index(self, date, index):
        self.date = date
        self.index = index

    def next_date(self):
        frequency = int(glo.frequency[:-1])
        self.date = datetime.strptime(self.date_list[self.index + frequency], "%Y-%m-%d %H:%M:%S")
        self.index += frequency
        return self.date

    def get_date(self):
        return self.date

    def find_index(self, start, end):
        mid = int((start + end) / 2)
        if datetime.strptime(self.date_list[mid], "%Y-%m-%d %H:%M:%S").__gt__(self.date):
            return self.find_index(start, mid - 1)
        elif datetime.strptime(self.date_list[mid], "%Y-%m-%d %H:%M:%S").__lt__(self.date):
            return self.find_index(mid + 1, end)
        return mid
