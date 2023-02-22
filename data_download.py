import yfinance as yf
from datetime import datetime
from datetime import date
from datetime import timedelta
import pandas as pd
import os
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

checker = 0
def download(tickers):
    for name in tickers:
        reps = os.walk('')
        print(reps) 
        today = date.today()
        yesterday =  str(today - timedelta(days=7))
        today = str(today)
        print(today, yesterday)
        data = yf.download(name, start=yesterday, end=today, interval='1m') #str(yesterday); str(today)
        #print (data, 'data')
        df = pd.DataFrame(data)
        df.to_csv('app_test/'+name+'.csv', mode='a', header=False)
        
def train_nn():
    print('nothigh here yet')