import os
import tensorflow as tf
import numpy as np
import pandas as pd
from jqdatasdk import *
from pyalgotrade import *
import glo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知信息和警告信息
auth("13074581737", "trustno1")

state = []
glo.__init__()


def __init__():
    global state
    state = np.array(
        get_price(glo.get_value("stockCode"), count=1, frequency='1m', end_date='2019-7-19 10:00:00',
                  skip_paused=True)).tolist()[0] + [glo.get_value("money")] + [glo.get_value(
        "stockBuyInPrice")] + [glo.get_value("stockBuyInAmount")]


def sell_out(stock, quant):
    """action:卖出股票,stock股票代码,quant卖出金额"""


def buy_in(stock, quant):
    """action:买入股票,stock股票代码,quant卖出金额"""


def get_state():
    """获取当前环境状态"""
    global state
    state = np.array(
        get_price(glo.get_value("stockCode"), count=1, frequency='1m', end_date='2019-7-19 10:00:00',
                  skip_paused=True)).tolist()[0] + [glo.get_value("money")] + [glo.get_value(
        "stockBuyInPrice")] + [glo.get_value("stockBuyInAmount")]

__init__()

x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3]))
w2 = tf.Variable(tf.random_normal([3, 1]))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y, feed_dict={x: [[1, 2], [0.2, 0.4], [0.5, 0.4]]}))
