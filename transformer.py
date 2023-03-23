import tensorflow as tf
import numpy as np


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.embedding = tf.keras.layers.Dense(d_model, activation='relu')
        self.positional_encoding = self.positional_encoding(maximum_position_encoding=10000,
                                                             d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.enc_layers = [self.EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

    def call(self, x, training):
        seq_len = tf.shape(x)[1]

        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config

    def EncoderLayer(self, d_model, num_heads, dff, dropout_rate):
        inputs = tf.keras.Input(shape=(None, d_model))
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='EncoderLayer')

    def positional_encoding(self, maximum_position_encoding, d_model):
        angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads = np.arange(maximum_position_encoding)[:, np.newaxis] * angle_rates

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)