_global_direct = {}


def __init__():
    global _global_direct
    _global_direct = {"stockCode": "000300.XSHG", "money": 10 ^ 5, "stockBuyInPrice": 0, "stockBuyInAmount": 0}


def set_value(key, value):
    _global_direct[key] = value


def get_value(key, defValue=None):
    try:
        return _global_direct[key]
    except KeyError:
        return defValue
