from flask import Flask, render_template, request
from stock import build_and_train_model
from utils import get_stock, preprocessing, generate_data, plotPredictions
from models import LSTMModel, GRUModel, BiLSTMModel, TransformerModel
import numpy as np
import base64
import io
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
app.debug = True

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
        'valid_prop': 0.2,
        'look_back': 5,
        'epochs': 10,
        'LSTMLayers': [{'LSTM_numLayers': 64, 'return_sequences': True, 'LSTM_act': 'relu'}],
        'denseLayers': [128, 128, 64, 1],
        'model_type': 'LSTM'
    }

    df = get_stock(ticker, startday, endday)
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
    model_class = model_map.get(hParams['model_type'], None)
    
    build_and_train_model(train_subset, val_subset, test_subset, hParams, model_class)

    model = load_model('stock.h5')
    test_x, _, test_scaler = test_subset
    predicted_prices = model.predict(test_x).reshape(-1, 1)
    predicted_prices = test_scaler.inverse_transform(predicted_prices)
    
    # Create a Matplotlib figure
    fig = Figure(figsize=(16, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Model')
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Close Price USD ($)', fontsize=18)
    train = df[:int(len(df) - len(predicted_prices))]
    valid = df[int(len(df) - len(predicted_prices)):]
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
