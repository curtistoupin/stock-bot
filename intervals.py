# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 21:42:48 2021

@author: curti
"""

#Interval class being introduced out of convenience for parsing yfinance data.
#The list of intervals is all accepted interval types when querying yfinance.
class Interval:
    def __init__(self, name, intervals_per_year):
        self.name = name
        self.intervals_per_year = intervals_per_year
        
one_min = Interval('1m', 525600)
two_min = Interval('2m', 262800)
five_min = Interval('5m', 105120)
fifteen_min = Interval('15m', 35040)
thirty_min = Interval('30m', 17520)
half_hour = thirty_min
sixty_min = Interval('60m', 8760)
ninety_min = Interval('90m', 5840)
one_hr = Interval('1h', 8760)
one_day = Interval('1d', 365)
five_day = Interval('5d', 73)
one_week = Interval('1wk', 52)
one_month = Interval('1mo', 12)
three_month = Interval('3no', 4)