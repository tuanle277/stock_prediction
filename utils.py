import pandas_datareader as pdr
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json

def get_stock(ticker, startday, endday):
    # Specify the start and end dates for the data
    # Download the data using the Yahoo Finance API
    data = yf.download(ticker, start=startday, end=endday)

    # Save the data to a CSV file
    data.to_csv(f"data/{ticker}_data.csv")
    return data

# Plot the close price
def plot_close_price(df):
    plt.plot(df.Close)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close Price Over Time')
    plt.show()

def preprocessing(df, hParams):
    data = df.Close.values

    # Split the data into training and test sets
    # Calculate the number of data points in each set
    train_size = int(len(data) * (1-hParams['test_prop']-hParams['valid_prop']))
    val_size = int(len(data) * hParams['valid_prop'])
    test_size = int(len(data) * hParams['test_prop'])

    # Split the data into training, validation, and test sets
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[-test_size:]

    return train_data,val_data, test_data

def generate_data(stock_prices, lookback):
    stock_prices = np.reshape(stock_prices,(-1,1))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(stock_prices)
    X, Y = [], []
    for i in range(len(scaled_data) - lookback - 1): # cut time-series data into vectors of lookback size, the next element in the scaled_data is the label.
        x = scaled_data[i : (i + lookback)]
        X.append(x)
        Y.append(scaled_data[i + lookback])
    X, Y = np.array(X), np.array(Y)
    x_train = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print(X.shape, Y.shape)

    return x_train, Y, scaler

def plotPredictions(df, predictions, hParams):
    #plot the data
    # print(predictions)
    train = df['Close']
    train = df[:int(len(df)-len(predictions))]
    valid = df[int(len(df)-len(predictions)):]
    valid['Predictions'] = predictions
    #visualize the data
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig("graphs/predictions/" + hParams["model_type"] + "_predictions")
    # plt.show()

# ====================== added Utils functions ====================== #
def writeExperimentalResults(hParams, trainResults, testResults):
    # == open file == #
    f = open("results/" + hParams["model_type"] + ".txt", 'w')

    # == write in file == #
    f.write(str(hParams) + '\n\n')
    f.write(str(trainResults) + '\n\n')
    f.write(str(testResults))

    # == close file == #
    f.close()

def readExperimentalResults(fileName):
    f = open("results/" + fileName + ".txt",'r')

    # == read in file == #
    data = f.read().split('\n\n')

    # == process data to json-convertible == #
    data[0] = data[0].replace("\'", "\"")
    data[1] = data[1].replace("\'", "\"")
    data[2] = data[2].replace("\'", "\"")

    # == convert to json == #
    hParams = json.loads(data[0])
    trainResults = json.loads(data[1])
    testResults = json.loads(data[2])

    return hParams, trainResults, testResults

def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
    plt.figure()
    plt.scatter(xList,yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if pointLabels != []:
        for i, label in enumerate(pointLabels):
            plt.annotate(label, (xList[i], yList[i]))
    filepath = "results/" + filename + ".png"
    plt.savefig(filepath)
    print("Figure saved in", filepath)

def buildAccuracyPlot(fileNames, title):
    # == get hParams == #
    hParams = readExperimentalResults(fileNames[0])[0]

    # == plot points with xList being the parameter counts of all and yList being the test accuracies == #
    plotPoints(xList=[0 * x for x in range(len(fileNames))],
                yList=[readExperimentalResults(name)[2][0] for name in fileNames],
                pointLabels= [name for name in fileNames],
                xLabel='x',
                yLabel='Test set loss',
                title="Test set loss_" + title,
                filename="Test set loss_" + title)

