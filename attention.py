# The call method of the attention layer has to compute the alignment scores, weights, and context. 
# You can go through the details of these parameters in Stefania’s excellent article on The Attention Mechanism from Scratch. 
# You’ll implement the Bahdanau attention in your call() method.

# The good thing about inheriting a layer from the Keras Layer class and adding the weights via the add_weights() method is that weights are automatically tuned. 
# Keras does an equivalent of “reverse engineering” of the operations/computations of the call() method and calculates the gradients during training. 
# It is important to specify trainable=True when adding the weights. 
# You can also add a train_step() method to your custom layer and specify your own method for weight training if needed.

# Add attention layer to the deep learning network
from keras.layers import Layer
import keras.backend as K
from keras import Model

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context