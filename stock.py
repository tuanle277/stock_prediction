import matplotlib.pyplot as plt
import numpy as np
from utils import *
from attention import *
from hParams import *
from models import LSTMModel, GRUModel, BiLSTMModel, TransformerModel

def build_and_train_model(train_subset, val_subset, test_subset, hParams, model_class):
    train_x, train_y, _ = train_subset
    val_x, val_y, _ = val_subset
    test_x, test_y, test_scaler = test_subset

    # Reshape input data to 3D for models
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    val_x = val_x.reshape((val_x.shape[0], val_x.shape[1], 1))
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

    model = model_class(hParams)
    model.build((train_x.shape[1], train_x.shape[2]))
    history = model.compile_and_train(train_x, train_y, val_x, val_y)
    test_loss, _ = model.evaluate_and_predict(test_x, test_y, test_scaler)
    model.save('stock.h5')

    writeExperimentalResults(hParams, history.history, test_loss)

def main(name=None):
    df = get_stock('AAPL', "2010-12-31", "2023-03-20")

    hParams = getHParams(name)

    train_data, val_data, test_data = preprocessing(df, hParams)
    train_subset = generate_data(train_data, hParams['look_back'])
    val_subset = generate_data(val_data, hParams['look_back'])
    test_subset = generate_data(test_data, hParams['look_back'])

    model_map = {
        'LSTM': LSTMModel,
        'GRU': GRUModel,
        'BiLSTM': BiLSTMModel,
        'Transformer': TransformerModel
    }
    model_type = hParams.get('model_type', 'LSTM')
    model_class = model_map.get(model_type, LSTMModel)

    predictions = build_and_train_model(train_subset, val_subset, test_subset, hParams, model_class)
    plotPredictions(df, predictions, hParams)

if __name__ == "__main__":
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

    for name in fileNames[0:1]:
        main(name)

    buildAccuracyPlot(fileNames, "BiLSTM_Dense")
