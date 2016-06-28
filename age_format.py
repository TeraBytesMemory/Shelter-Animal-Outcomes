#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import numpy as np


def age_format(age):
    try:
        n, unit = age.split(' ')
    except ValueError:
        n, unit = age, 'day'
    n = int(n)

    if 'day' in unit:
        return n
    elif 'week'in unit:
        return 7 * n
    elif 'month' in unit:
        return 30 * n
    elif 'year' in unit:
        return 365 * n

def name_format(name):
    try:
        return ord(name[0]) - ord('A')
    except (IndexError, TypeError):
        return -1

def date_format(date):
    try:
        d = datetime.strptime(date[2:], '%y/%m/%d %H:%M')
    except:
        d = datetime.strptime(date[2:], '%y-%m-%d %H:%M:%S')

    return d.hour
