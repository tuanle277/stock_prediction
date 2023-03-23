from flask import Flask, render_template, request
from stock import *
from utils import *
import numpy as np
import base64
import io
from io import BytesIO
from keras.models import load_model
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
app.debug = True
training = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    startday = request.form['startday']
    endday = request.form['endday']
    hParams = {
        'test_prop': 0.1,
        'valid_prop':0.2,
        'look_back': 10,
        'epochs': 10,
        'LSTM': [128, 32],
        'Dense': [32,32]
    }

    hParams = getHParams('LSTM_64_Dense_128_128_64_1')
    df = get_stock(ticker,startday,endday)
    # print(df.head())
    train_data,val_data, test_data = preprocessing(df,hParams)
    train_subset = generate_data(train_data, hParams['look_back'])
    val_subset =  generate_data(val_data, hParams['look_back'])
    test_subset =  generate_data(test_data, hParams['look_back'])
    predictions = buildModel(train_subset,val_subset,test_subset,hParams)
    # model = load_model('stock.h5')
    # # Use your LSTM model to predict the stock prices for the given ticker
    _,_, test_data = preprocessing(df,hParams)

    test_x, test_y, test_scaler =  generate_data(test_data, hParams['look_back'])
    model = load_model('stock.h5')
    predicted_prices = model.predict(test_x)
    predicted_prices = test_scaler.inverse_transform(predicted_prices)
    
    # Get the plot image
    # Create a Matplotlib figure
    fig = Figure(figsize=(16, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Model')
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Close Price USD ($)', fontsize=18)
    train = df[:int(len(df)-len(predicted_prices))]
    valid = df[int(len(df)-len(predicted_prices)):]
    valid['Predictions'] = predicted_prices
    ax.plot(train['Close'])
    ax.plot(valid['Close'])
    ax.plot(valid['Predictions'])
    ax.legend(['Train', 'Val', 'Predictions'], loc='lower left')

    # Render the figure to a PNG image
    canvas = FigureCanvas(fig)
    img_buf = io.BytesIO()
    canvas.print_png(img_buf)
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    return render_template('prediction.html', ticker=ticker, startday=startday, endday=endday, image_data=img_base64)


if __name__ == '__main__':
    app.run()