# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:03:04 2021

@author: curti
"""
from intervals import *
import yfinance as yf
import pandas as pd

#Avg market growth per year is used as a baseline against which to determine how well a stock is truly performing
avg_market_growth_per_year = 0.1

def get_ticker(ticker_symbol):
    return yf.Ticker(ticker_symbol)

#Stock prices are normalized so as to not ignore stocks experiencing small changes due to having a low price point
def normalize_ticker_data(ticker, interval = one_day):
    ticker_df = pd.DataFrame(ticker.history(period='max', interval = interval.name))
    
    avg_market_growth_per_interval = avg_market_growth_per_year/interval.intervals_per_year
    
    #Create columns which report change in open, close, high, and low price between consecutive entries normalized as a percent of the former entry's
    ticker_df['OpenPcDiff'] = ((ticker_df['Open'] - ticker_df['Open'].shift(1))/ticker_df['Open']) - avg_market_growth_per_interval
    ticker_df['ClosePcDiff'] = ((ticker_df['Close'] - ticker_df['Close'].shift(1))/ticker_df['Close']) - avg_market_growth_per_interval
    ticker_df['HighPcDiff'] = ((ticker_df['High'] - ticker_df['High'].shift(1))/ticker_df['High']) - avg_market_growth_per_interval
    ticker_df['LowPcDiff'] = ((ticker_df['Low'] - ticker_df['Low'].shift(1))/ticker_df['Low']) - avg_market_growth_per_interval
    #Report on the transaction volume in each period normalized as a percent of the shares available for sale
    ticker_df['VolumePcOfFloat'] = ticker_df['Volume']/ticker.info['floatShares']
    #Report on the relative strength index of the stock during each period
    ticker_df['RSI'] = calculate_rsi(ticker_df['ClosePcDiff'])
    #Experimental indicator being tested as a potential indicator of stock/option price (chosen since options expire on Fridays)
    ticker_df['DaysUntilFriday'] = [4 - timestamp.weekday() for timestamp in ticker_df.index[:]]
    
    return ticker_df[14:]

#The RSI or relative strength index is often used as an indicator by amateur stock traders.
#Roughly speaking, the RSI can be intuitively understood as the "percent of total recent movement
#which was in an upward direction".
def calculate_rsi(price_list):
    rsi_list = [0.5]*len(price_list)
    for i in range(14, len(price_list)):
        obs = price_list[(i-14) : (i-1)]
        gains = [gain for gain in obs if gain > 0]
        losses = [loss for loss in obs if loss < 0]
        avg_gain = sum(gains)/14
        avg_loss = -sum(losses)/14
        current_gain = price_list[i] if price_list[i] > 0 else 0
        current_loss = -price_list[i] if price_list[i] < 0 else 0
        rel_strength = (avg_gain * 13 + current_gain)/(avg_loss * 13 + current_loss)
        rsi_list[i] = 100 - 100/(1+rel_strength)
    
    return rsi_list        
        