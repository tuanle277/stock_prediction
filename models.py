import tensorflow as tf
from transformer import TransformerEncoder

class BaseModel:
    def __init__(self, hParams):
        self.hParams = hParams
        self.model = None

    def build(self, input_shape):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compile_and_train(self, train_x, train_y, val_x, val_y):
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        history = self.model.fit(train_x, train_y, epochs=self.hParams['epochs'], 
                                 validation_data=(val_x, val_y), 
                                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=(10**-6), patience=20)])
        return history

    def evaluate_and_predict(self, test_x, test_y, test_scaler):
        print(test_x.shape, test_y.shape)
        test_loss = self.model.evaluate(test_x, test_y)
        predictions = self.model.predict(test_x).reshape(-1, 1)
        print("predictions is ", predictions.shape)
        # predictions = test_scaler.inverse_transform(predictions.reshape(-1, 1))
        return test_loss, predictions

    def save(self, filename):
        self.model.save(filename)

class LSTMModel(BaseModel):
    def build(self, input_shape):
        self.model = tf.keras.Sequential()
        for layer in self.hParams['LSTMLayers']:
            self.model.add(tf.keras.layers.LSTM(layer['LSTM_numLayers'], 
                                                return_sequences=layer['return_sequences'], 
                                                activation=layer['LSTM_act'], 
                                                input_shape=input_shape))
        self._add_dense_layers()

    def _add_dense_layers(self):
        for layer in range(len(self.hParams['denseLayers'])):
            if layer < len(self.hParams['denseLayers']) - 1:
                self.model.add(tf.keras.layers.Dense(self.hParams['denseLayers'][layer], activation='relu'))
            else:
                self.model.add(tf.keras.layers.Dense(self.hParams['denseLayers'][layer]))

class GRUModel(BaseModel):
    def build(self, input_shape):
        self.model = tf.keras.Sequential()
        for layer in self.hParams['GRULayers']:
            self.model.add(tf.keras.layers.GRU(layer['GRU_numLayers'], 
                                               return_sequences=layer['return_sequences'], 
                                               activation=layer['GRU_act'], 
                                               input_shape=input_shape))
        self._add_dense_layers()

    def _add_dense_layers(self):
        for layer in range(len(self.hParams['denseLayers'])):
            if layer < len(self.hParams['denseLayers']) - 1:
                self.model.add(tf.keras.layers.Dense(self.hParams['denseLayers'][layer], activation='relu'))
            else:
                self.model.add(tf.keras.layers.Dense(self.hParams['denseLayers'][layer]))

class BiLSTMModel(BaseModel):
    def build(self, input_shape):
        self.model = tf.keras.Sequential()
        for layer in self.hParams['BiLSTMLayers']:
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer['BiLSTM_numLayers'], 
                                                                              return_sequences=layer['return_sequences'], 
                                                                              activation=layer['BiLSTM_act'], 
                                                                              input_shape=input_shape)))
        self._add_dense_layers()

    def _add_dense_layers(self):
        for layer in range(len(self.hParams['denseLayers'])):
            if layer < len(self.hParams['denseLayers']) - 1:
                self.model.add(tf.keras.layers.Dense(self.hParams['denseLayers'][layer], activation='relu'))
            else:
                self.model.add(tf.keras.layers.Dense(self.hParams['denseLayers'][layer]))

class TransformerModel(BaseModel):
    def build(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = TransformerEncoder(num_layers=12, d_model=32, num_heads=12, dff=64, dropout_rate=0.1)(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
