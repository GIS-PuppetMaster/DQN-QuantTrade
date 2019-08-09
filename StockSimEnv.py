from jqdatasdk import *
import pandas as pd
import numpy as np
import glo
import time
import datetime
from pyalgotrade import *
from sklearn.preprocessing import *
import tushare as ts
from stock_date import *

# auth("13074581737", "trustno1")

gdate = StockDate()
data = []


def __init__():
    """
    初始化环境
    :return: None
    """
    global gdate
    global data

    gdate = StockDate()
    glo.__init__()
    data = pd.read_csv(glo.stock_code.replace(".", "_") + ".csv", index_col='Unnamed: 0')
    glo.price = get_stock_price(gdate.get_date())


def reshape3D(tensor):
    # 把一个2D张量改为3D张量，第三个维度只有原数据这一组
    tensor = np.array(tensor)
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], 1)
    return tensor.tolist()


def get_state(date):
    """
    stock_state = np.array(
        get_price(glo.stock_code, count=glo.count, frequency=glo.frequency,
                  end_date=date.strftime("%Y-%m-%d %H:%M:%S"),
                  skip_paused=True))[..., [0, 1, 2, 3]].transpose().tolist()
    """
    global gdate
    global data

    gdate.set_date(date)
    stock_state = []
    for i in range(0, int(glo.frequency[:-1]) * glo.count, int(glo.frequency[:-1])):
        line = np.array(data.loc[str(gdate.get_date())])[[0, 1, 2, 3, 4, 5]].transpose().tolist()
        gdate.next_date()
        stock_state.append(line)
    # stock_state=reshape3D(stock_state)
    # 返回状态前先进行归一化
    stock_state = scale(stock_state, axis=0)
    agent_state = [glo.money] + [glo.get_stock_total_value(glo.price)] + [
        glo.get_stock_amount()]
    agent_state = np.array(agent_state).reshape(-1, 1).tolist()
    agent_state = scale(agent_state, axis=0)
    return stock_state, agent_state


def get_stock_price(date):
    """
    return np.array(get_price(glo.stock_code, count=1, frequency=glo.frequency,
                              end_date=date.strftime("%Y-%m-%d %H:%M:%S"),
                              skip_paused=True)).tolist()[0][1]
    """
    global data
    global gdate
    gdate.set_date(date)
    return np.array(data.loc[str(gdate.get_date())]).tolist()[1]


def trade(stock, action, date):
    """action:买入股票,stock股票代码,quant卖出股数"""
    global gdate
    gdate.set_date(date)
    quant = 0
    flag = False
    action_0 = action[0]
    # 交易开关激活时
    if action[1] > 0:
        # 买入
        if action_0 > 0:
            # 按钱数百分比买入
            # 当前的钱购买多少手
            amount = int(glo.money / (100 * glo.price))
            # 实际买多少手
            quant = int(action_0 * amount)
            if quant == 0:
                print("钱数：" + str(glo.money) + "不足以购买一手股票")
                flag = True
        # 卖出
        elif action_0 < 0:
            # 当前手中有多少手
            amount = glo.get_stock_amount()
            if amount == 0:
                flag = True
            # 实际卖出多少手
            quant = int(action_0 * amount)
            if quant == 0 and action_0 != 0:
                flag = True
    price = glo.price
    # 钱数-=每股价格*100*交易手数
    money = glo.money - price * 100 * quant
    # [股价,手数]
    glo.stock_value.append([price, quant])
    glo.money = money
    print("实际交易量：" + str(quant))
    print("实际交易金额（收入为正卖出为负）：" + str(-price * 100 * quant))
    next_date = gdate.next_date()
    if flag:
        reward = -abs(action[0])
    else:
        reward = (glo.money + glo.get_stock_total_value(
            get_stock_price(next_date)) - glo.ori_money - glo.ori_value) / (glo.ori_money + glo.ori_value)
    # 返回glo.frequency之后的状态
    state = get_state(date=next_date)
    return state[0], state[1], reward
