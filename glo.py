count = 32
stock_code = "000517.XSHE"
money = 1*pow(10, 5)
ori_money = money
# [[股价,股数]]
stock_value = []
frequency = '1m'
price = 0
ori_value = 0


def __init__():
    global count
    global stock_value
    global stock_code
    global money
    global ori_money
    global frequency
    global price
    global state_update_frequency
    global ori_value
    count = 32
    ori_value = 0
    stock_code = "000517.XSHE"
    money = pow(10, 5)
    ori_money = money
    # [[股价,股数]]
    stock_value = []
    frequency = '1m'
    price = 0


def get_stock_total_value(price):
    """
    计算股票总价值
    :return: 股票总价值
    """

    return price * 100 * get_stock_amount()


def get_stock_amount():
    amount = 0
    for l in stock_value:
        amount += l[1]
    return amount


def init_with_oristock(price,quant):
    global stock_value
    global ori_value
    if quant==-1:
        # 初始化时默认持股
        amount = int(money / (100 * price))
        # 实际买多少手
        quant = int(1 * amount)
    stock_value = [[price, quant]]
    ori_value = price * quant * 100
