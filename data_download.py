import yfinance as yf
from datetime import date
from datetime import timedelta
import pandas as pd
import os
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

checker = 0
def download(tickers):
    """Downloads the newest data that are avaiilable for training 

    Args:
        tickers (string_list): the list of strings should contain stock tickers in all capital letters
    """
    
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
        if(len(df)>20):
            df.to_csv('app_test/'+name+'.csv', mode='a', header=True)
        
def train_nn():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import seaborn as sns
    import os
    from datetime import datetime
    from difflib import SequenceMatcher
    import warnings
    from sklearn.preprocessing import MinMaxScaler
    
    warnings.filterwarnings("ignore")
    
    linker = 'app_test'#'finised_files'
    names = os.listdir(linker)
    itemss = os.scandir()
    if os.path.exists('model.h5'):
        os.remove('model.h5')
    
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=512, return_sequences=True, input_shape=(512, 6)))
    model.add(keras.layers.LSTM(units=512))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Dropout(0.5))   
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    model.summary
    model.compile(optimizer='adam', loss='mean_squared_error')

    for imar in names:
        print(imar)
        data = pd.read_csv(linker + '/' + imar) 
        data.info()
    
        close_data = data.filter(['Close', 'Open', "High", "Low", "Volume", "Adj Close"])
        dataset = close_data.values
        dataset_close = close_data['Close'].values
        training = int(np.ceil(len(dataset)*.7))   

        scaler = MinMaxScaler(feature_range=(0, 1))
        scalar_close = MinMaxScaler(feature_range=(0, 1))
        dataset_close = dataset_close.reshape(-1,1)
        slop = scalar_close.fit_transform(dataset_close)
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:int(training), :]

        x_train = []
        y_train = []
    
        for i in range(512, len(train_data)):
            #print (i, '6', train_data[i-512:i, 0:6])
            x_train.append(train_data[i-512:i, 0:6])
            y_train.append(train_data[i, 0])
    
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
  
       
        hystory = model.fit(x_train, y_train, epochs=5)

        test_data = scaled_data[training - 512:, :]
        x_test = []
        y_test = dataset[training:, :]
        for i in range(512, len(test_data)):
            x_test.append(test_data[i-512:i, 0:6])
  
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 6))

        predictions = model.predict(x_test)
        predictions = scalar_close.inverse_transform(predictions)
  
    # evaluation metrics
        mse = np.mean(((predictions - y_test) ** 2))
    #print("MSE", mse)
        print("RMSE", np.sqrt(mse))

        train = data[:training]
        test = data[training:]
        test['Predictions'] = predictions
  
        
    model.save('model.h5')
    
def live_data(tickers):
    for name in tickers:
        today = date.today()
        yesterday =  today - timedelta(days=1)
        data = yf.download(name, start=str(yesterday), end=str(today), interval='1m')
        df = pd.DataFrame(data)
        df.to_csv('live_data/'+name+'.csv', mode='a', header=False)

