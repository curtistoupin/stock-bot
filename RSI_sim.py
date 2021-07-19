# -*- coding: utf-8 -*-
"""
This file contains tools for analyzing the efficacy of RSI as a buy/sell indicator.
Created on Sun Feb  7 01:01:59 2021

@author: curti
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt

#Checks RSI of provided stock once per period. If no stock is held and RSI is more extreme than the buy limit
#we purchase 1000 and we sell the next time the RSI is more extreme than the sell limit. Returns total profit.
def rsi_sim(data, rsi_sell_lim = 70, rsi_buy_lim = 30):
    money = 0
    stonks = 0
    #This step allows the simulation to function properly for the reverse of conventional strategies (i.e. buy_lim > sell_lim)
    sign = np.sign(rsi_sell_lim - rsi_buy_lim)
    for i in range(len(data)):
        if data['RSI'][i] <= rsi_buy_lim and stonks == 0:
            stonks += 1000
            money -= stonks*data['Close'][i]
        elif data['RSI'][i] >= rsi_sell_lim and stonks > 0:
            money += stonks*data['Close'][i]
            stonks = 0
            
    return money
        
#Tests the rsi_sim simulation with every combination of sell limits between 1 and 50 and buy limits between 51 and 100.
#Reports on the best performing strategy, the worst performing strategy, and the performace of the conventional 30/70 strategy.
def run_the_gammit(data):
    gains = [None]*49*49
    gains_index = 0
    max_gain = None
    max_gain_sell = None
    max_gain_buy = None
    max_loss = None
    max_loss_sell = None
    max_loss_buy = None
    normal_gain_loss = None
    
    for j in range(1,50):
        for i in range(51,100):
            strat_gains = rsi_sim(data, rsi_sell_lim = i, rsi_buy_lim = j)
            gains[gains_index] = strat_gains
            gains_index += 1
            if max_gain is None or ( max_gain is not None and strat_gains >= max_gain ):
                max_gain = strat_gains
                max_gain_sell = i
                max_gain_buy = j
            if max_loss is None or ( max_gain is not None and strat_gains <= max_loss ):
                max_loss = strat_gains
                max_loss_sell = i
                max_loss_buy = j
            if i == 70 and j == 30:
                normal_gain_loss = strat_gains
            
    print('Max gain of $',max_gain, 'realized with',max_gain_buy,'/',max_gain_sell,'strategy.')
    print('Gain/loss of $', normal_gain_loss, 'realized with 30/70 strategy.')
    print('Max loss of $',max_loss, 'realized with',max_loss_buy,'/',max_loss_sell,'strategy.')
    
    return gains

#Generates and displays a plot of all resulting gains from strategies as well as a relative frequency plot of acheived gains
def do_the_thing(ticker_code):
    ticker = get_ticker(ticker_code)
    ticker_df = normalize_ticker_data(ticker)
    ticker_gains = run_the_gammit(ticker_df[-365:])
    plt.plot(ticker_gains)
    plt.show()
    
    p, x = np.histogram(ticker_gains, bins = 1000)
    x = x[:-1] + (x[1]-x[0])/2
    f = UnivariateSpline(x, p, s = 200)
    plt.plot(x, f(x))
    plt.show()
    
#Shortcut for fetching data with one day intervals and testing a particular buy/sell RSI strategy
def test_strat(ticker_code, sell_lim, buy_lim):
    ticker = get_ticker(ticker_code)
    ticker_df = normalize_ticker_data(ticker)
    return rsi_sim(ticker_df[-365:], sell_lim, buy_lim)

#Fetches data for a provided ticker and plots the RSI against future price difference a provided number of days in the future
#as well as the results of an OLS regression to determine the correlation between RSI and future price difference for that stock.
def plot_rsi_vs_diff(ticker_code, days_later):
    ticker = get_ticker(ticker_code)
    ticker_df = normalize_ticker_data(ticker)
    ticker_df['Shift'] = (ticker_df['Close'].shift(-days_later) - ticker_df['Close'])/ticker_df['Close']
    slope = ticker_df.cov()['RSI']['Shift']/ticker_df.cov()['RSI']['RSI']
    correlation = ticker_df.cov()['RSI']['Shift']/np.sqrt(ticker_df.cov()['RSI']['RSI'] * ticker_df.cov()['Shift']['Shift'])
    print('Average slope of', slope, 'and correlation coefficient', correlation)
    ticker_df = ticker_df[-730:]
    plt.plot(ticker_df['RSI'], ticker_df['Shift'])