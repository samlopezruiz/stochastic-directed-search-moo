from functools import reduce  # forward compatibility for Python 3
import operator


def set_in_dict(dataDict, mapList, value):
    get_from_dict(dataDict, mapList[:-1])[mapList[-1]] = value


def get_from_dict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)
