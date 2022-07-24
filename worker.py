'''
@author: Daniel S. Londoño
'''
####################################################################################################################################
####################################################################################################################################
###############      0. IMPORTS       ##############################################################################################
####################################################################################################################################
####################################################################################################################################

from ast import Raise
import pandas as pd
import json
import requests
import re
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, InputLayer, Embedding
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import deque
import random
import matplotlib.pyplot as plt

def model_build(coins: list, n_periods: int, SEQ_LEN: int, epochs: int, batch_size: int, pct: float, n_days: int):
    '''
    Trains, executes, and evaluates the Neural Network model. It is evaluated over data from the past n_days.
    @Parameters:
        - coins: List of coins to the model be trained with. (e.g. 'bitcoin', 'ethereum', 'dogecoin')
        - n_periods: Number of days to be looked into the future. 
        - SEQ_LEN: Length of the sequence data to be feeded into the model. 
        - epochs: Number of epochs to run the model under. 
        - batch_size: Size of the batch to run the model under. 
        - pct: Percentage of data to evaluate the model with. (e.g. 0.2 means 80% of the oldest data will be used to train the model)
        - n_days: Number of days to be looked into the past. (e.g. n_days=1000 means there will be 1000 records on the dataset, one for each day.)
    '''



    ####################################################################################################################################
    ####################################################################################################################################
    ###############      1. ETL LAYER       ############################################################################################
    ####################################################################################################################################
    ####################################################################################################################################
    
    
    coin_dfs = []
    for coin in coins:
        req = requests.get('https://api.coingecko.com/api/v3/coins/{c}/market_chart?vs_currency=USD&days={days}&interval=daily'.format(c=coin, days=n_days))

        market_cap_ = dict(req.json())['market_caps']
        prices_ = dict(req.json())['prices']
        total_volumes_ = dict(req.json())['total_volumes']

        price_s = [i[1] for i in prices_]
        volume_s = [i[0] for i in prices_]
        market_cap_irr_ = [i[1] for i in market_cap_]
        total_volumes_ = [i[1] for i in total_volumes_]

        df = pd.DataFrame()
        df['price_{c}'.format(c=coin)] = price_s
        df['volume'] = volume_s
        df['market_cap_{c}'.format(c=coin)] = market_cap_irr_
        df['total_volume_{c}'.format(c=coin)] = total_volumes_

        coin_dfs.append(df)
   
    main_df = coin_dfs[0]
    for i in range(1,len(coin_dfs)):
        main_df = main_df.merge(coin_dfs[i], on='volume')

    def classify(future, current):
        if future > current:
            return 1
        else:
            return 0
    
    # Creates the target label. 
    target_label = main_df.columns.to_list()[-3]
    main_df['future'] = main_df[target_label].shift(-n_periods)
    main_df['target'] = list(map(classify, main_df[target_label], main_df['future']))

    # Normalization of data

    cols = main_df.columns.to_list()

    # 1. Price normalization. The taken maximum is the entire dataframe's maximum. 

    regex_prices = re.compile('price')
    re_filt_prices = list(filter(regex_prices.match, cols))

    min_price = main_df[re_filt_prices].min().min()
    max_price = main_df[re_filt_prices].max().max()

    for c in re_filt_prices:
        main_df[c] = (main_df[c] - min_price)/(max_price-min_price)
    
    # 2. Volume normalization. Independent for each crypto. 
    regex_tot_volumes = re.compile('total_volume')
    re_filt_vols = list(filter(regex_tot_volumes.match, cols))

    for c in re_filt_vols:
        main_df[c] = (main_df[c] - main_df[c].min())/(main_df[c].max()-main_df[c].min())
    
    # 3. Market capitalization normalization. The taken maximum is the entire dataframe's maximum
    regex_marketcap = re.compile('market_cap')
    re_filt_marc = list(filter(regex_marketcap.match, cols))

    min_mc = main_df[re_filt_marc].min().min()
    max_mc = main_df[re_filt_marc].max().max()

    for c in re_filt_marc:
        main_df[c] = (main_df[c] - min_mc)/(max_mc - min_mc)
    
    # 4. Volume Normalization
    main_df['volume'] = (main_df['volume'] - main_df['volume'].min())/(main_df['volume'].max() - main_df['volume'].min())

    main_df = main_df.drop('future', 1)

    def preprocessDf(df):
        for col in df.columns:
            if col != 'target':
                df[col] = preprocessing.scale(df[col].values)
        df.dropna(inplace=True)
        sequential_data = []
        prev_days = deque(maxlen=SEQ_LEN)
        for i in df.values:
            prev_days.append([n for n in i[:-1]])
            if len(prev_days) == SEQ_LEN:
                sequential_data.append([np.array(prev_days), i[-1]])

        buys = []
        sells = []

        for seq, target in sequential_data:
            if target == 0:
                sells.append([seq, target])
            elif target == 1:
                buys.append([seq, target])
        

        lower = min(len(buys), len(sells))
        buys = buys[:lower]
        sells = sells[:lower]

        sequential_data = buys + sells
        random.shuffle(sequential_data)

        X = []
        Y = []
        for seq, tgt in sequential_data:
            X.append(seq)
            Y.append(tgt)
        
        return np.array(X), np.array(Y)

    times = sorted(main_df.index.values)
    last_pct = times[-int(pct*len(times))]

    train = main_df[(main_df.index < last_pct)]  
    validation = main_df[(main_df.index >= last_pct)]

    x_train, y_train = preprocessDf(train)
    x_test, y_test = preprocessDf(validation)


    ####################################################################################################################################
    ####################################################################################################################################
    ###############      2. MODEL LAYER       ##########################################################################################
    ####################################################################################################################################
    ####################################################################################################################################

    # Sequential model for timed data
    model = Sequential()

    # Rectified Linear Unit Activation Function
    model.add(LSTM(128, activation='tanh', input_shape=(x_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, activation='tanh', input_shape=(x_train.shape[1:]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128, activation='tanh', input_shape=(x_train.shape[1:])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    # Optimizer
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = opt,
        metrics = ['accuracy'] 
    )

    # Early stopping for model eficiency
    es = EarlyStopping(monitor='AUC', mode='min', verbose=1, patience=20)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es]
    )

    ####################################################################################################################################
    ####################################################################################################################################
    ###############      3. EVALUATION LAYER        ####################################################################################
    ####################################################################################################################################
    ####################################################################################################################################

    _, accuracy = model.evaluate(x_test, y_test)
    print('')
    print('[%] Accuracy from evaluate: %.2f' % (accuracy*100))

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    return history

def main():
    try:
        coins_list = input("[$] Enter the names of the coins (separated with commas , and no blank spaces) that you'd like to study. Note that the last one should be the one desired to predict.\n-> ")
        coins = coins_list.split(',')
        n_periods = input("[$] Great! Now input the number of periods (days) to look into the future.\n-> ")
        seq_len = input("[$] Now please input the sequence length. (Recommended: 60)\n-> ")
        epochs = input("[$] Please input the number of epochs.\n-> ")
        batch_size = input("[$] Now the batch size.\n-> ")
        pct = input("[$] Input the percentage of the data (float) to evaluate the model with.\n-> ")
        n_days = input("[$] Finally, input the number of day records to train the model with. \n-> ")
        
        model_build(
            coins=coins,
            n_periods=int(n_periods),
            SEQ_LEN=int(seq_len),
            epochs=int(epochs),
            batch_size=int(batch_size),
            pct=float(pct),
            n_days=int(n_days)
        )
    except:
        print('((!)) Error: Please read documentation. ')

if __name__ == '__main__':
    main()
