from utils import *
from attention import *
from transformer import *
from hParams import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np




def buildModel(train_subset, val_subset, test_subset, hParams):
    # print(np.array(test_data)[:, np.newaxis])
    # Build the model
    train_x, train_y, train_scaler = train_subset
    val_x, val_y, val_scaler = val_subset
    test_x, test_y, test_scaler = test_subset


    # LSTM/GRU/BiLSTM model from hParams
    model = tf.keras.Sequential()

    # LSTM/GRU/BiLSTM layers 
    if hParams.get("LSTMLayers") != None:
        for layer in hParams['LSTMLayers']:
            model.add(tf.keras.layers.LSTM(layer['LSTM_numLayers'], return_sequences=layer['return_sequences'], activation=layer['LSTM_act'], input_shape= (train_x.shape[1], 1)))
    elif hParams.get("GRULayers") != None:
        for layer in hParams['GRULayers']:
            model.add(tf.keras.layers.LSTM(layer['GRU_numLayers'], return_sequences=layer['return_sequences'], activation=layer['GRU_act'], input_shape= (train_x.shape[1], 1)))
    else:
        for layer in hParams['BiLSTMLayers']:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer['BiLSTM_numLayers'], return_sequences=layer['return_sequences'], activation=layer['BiLSTM_act'], input_shape= (train_x.shape[1], 1))))
    
    # Dense layers
    for layer in range(len(hParams['denseLayers'])):
        if layer < len(hParams['denseLayers']) - 1:
            model.add(tf.keras.layers.Dense(hParams['denseLayers'][layer], activation='relu'))
        else:
            model.add(tf.keras.layers.Dense(hParams['denseLayers'][layer]))
  

    # transformer model
    # inputs = tf.keras.layers.Input(shape=(hParams["look_back"], 1))
    # x = TransformerEncoder(num_layers=12, d_model=32, num_heads=12, dff=64, dropout_rate=0.1)(inputs)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(64, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)


    # model = tf.keras.Sequential([
    #     # tf.keras.layers.GRU(128, return_sequences=True, input_shape= (train_x.shape[1], 1)),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,activation='relu', input_shape= (train_x.shape[1], 1))),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(1)
    # ])



    model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mae'])
    # Train the model
    history = model.fit(train_x,train_y, epochs=hParams['epochs'], validation_data = (val_x,val_y), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=(10**-6), patience = 20)])
    # Evaluate the model on the test data
    test_loss = model.evaluate(test_x, test_y)
    # print(test_loss)
    # Make predictions on new data
    predictions = model.predict(test_x)
    print(test_loss)
    predictions = test_scaler.inverse_transform(predictions)
    model.save('stock.h5')

    # ========================= added code ========================= #
    writeExperimentalResults(hParams, history.history, test_loss)

    return predictions

def main(name=None):
    df = get_stock('AAPL',"2010-12-31","2023-03-20")

    hParams = getHParams(name)

    print(hParams)

    # hParams = {
    #     'test_prop': 0.1,
    #     'valid_prop':0.2,
    #     'look_back': 10,
    #     'epochs': 50,
    #     'LSTM': [128, 64],
    #     'Dense': [32, 32],
    #     'experimentName': 'Bidirectional',
    # }

    train_data,val_data, test_data = preprocessing(df,hParams)
    train_subset = generate_data(train_data, hParams['look_back'])
    val_subset =  generate_data(val_data, hParams['look_back'])
    test_subset =  generate_data(test_data, hParams['look_back'])
    predictions = buildModel(train_subset,val_subset,test_subset,hParams)
    plotPredictions(df,predictions,hParams)


# names of files of models tested
fileNames = [
    'LSTM_128_64_Dense_32_1',
    'LSTM_128_64_Dense_64_1',
    'LSTM_128_128_Dense_64_1',
    'LSTM_256_128_Dense_64_1',
    'LSTM_256_128_Dense_64_64_1',
    'LSTM_256_128_Dense_64_32_1',
    'LSTM_64_Dense_128_128_64_1',
    'LSTM_64_Dense_256_128_64_1',
    'LSTM_64_64_Dense_128_128_64_1',    
    'LSTM_64_32_Dense_128_128_64_1',
    'LSTM_128_64_Dense_256_128_64_1',
    'LSTM_128_Dense_128_128_64_1',
    'LSTM_128_64_64_Dense_256_128_64_1',
    'LSTM_128_Dense_512_256_128_64_1',    
    'LSTM_128_64_Dense_512_256_128_64_1'
]

fileNames = [
    'GRU_128_64_Dense_32_1',
    'GRU_128_64_Dense_64_1',
    'GRU_128_128_Dense_64_1',
    'GRU_256_128_Dense_64_1',
    'GRU_256_128_Dense_64_64_1',
    'GRU_256_128_Dense_64_32_1',
    'GRU_64_Dense_128_128_64_1',
    'GRU_64_Dense_256_128_64_1',
    'GRU_64_64_Dense_128_128_64_1',    
    'GRU_64_32_Dense_128_128_64_1',
    'GRU_128_64_Dense_256_128_64_1',
    'GRU_128_Dense_128_128_64_1',
    'GRU_128_64_64_Dense_256_128_64_1',
    'GRU_128_Dense_512_256_128_64_1',    
    'GRU_128_64_Dense_512_256_128_64_1'
]

fileNames = [
    'BiLSTM_128_64_Dense_32_1',
    'BiLSTM_128_64_Dense_64_1',
    'BiLSTM_128_128_Dense_64_1',
    'BiLSTM_256_128_Dense_64_1',
    'BiLSTM_256_128_Dense_64_64_1',
    'BiLSTM_256_128_Dense_64_32_1',
    'BiLSTM_64_Dense_128_128_64_1',
    'BiLSTM_64_Dense_256_128_64_1',
    'BiLSTM_64_64_Dense_128_128_64_1',    
    'BiLSTM_64_32_Dense_128_128_64_1',
    'BiLSTM_128_64_Dense_256_128_64_1',
    'BiLSTM_128_Dense_128_128_64_1',
    'BiLSTM_128_64_64_Dense_256_128_64_1',
    'BiLSTM_128_Dense_512_256_128_64_1',    
    'BiLSTM_128_64_Dense_512_256_128_64_1'
]

for name in fileNames:
    main(name) 

buildAccuracyPlot(fileNames, "BiLSTM_Dense")


