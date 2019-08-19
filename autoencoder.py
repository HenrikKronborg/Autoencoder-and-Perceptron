import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from tensorflow import set_random_seed
import numpy as np
import os

# Set the seeds to give the same random numbers every run
def seed(s):
    np.random.seed(s)
    set_random_seed(s)

class Autoencoder:
    def __init__(self, encodingDimension):
        # Set the desired number of hidden nodes we're reducing to
        self.encodingDimension = encodingDimension
        # Input array, 1000 random numbers between 1 and 8
        self.input = np.array([[np.random.randint(1, 8)] for _ in range(1000)])

    # Encodes the input. Compresses it down to the desired number of hidden nodes
    def encoder(self):
        # Set input shapes to the same shape as the input array above, in this case (?,1)
        inputs = Input(shape=(self.input[0].shape))
        # Create the hidden nodes using TanhLayer activiation function, as asked for in the task description
        encoded = Dense(self.encodingDimension, activation='tanh')(inputs)
        # Create an encoder model based on inputs and the hidden nodes
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def decoder(self):
        # Set the input shape to the number of hidden nodes.
        # In the instance of 8 hidden nodes: (?, 8)
        inputs = Input(shape=(self.encodingDimension,))
        # Decode the hidden nodes into one output
        decoded = Dense(1)(inputs)
        # Create a decoder model based on number of hidden nodes and the encoded inputs
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encodeDecode(self):
        # Encoding model returned by the encoder() function
        encoderModel = self.encoder()
        # Decoding model returned by the decoder() function
        decoderModel = self.decoder()

        inputs = Input(self.input[0].shape)
        # Send the input shape to the encoder model
        encodedOut = encoderModel(inputs)
        # Send the encoded input to the decoder model
        decodedOut = decoderModel(encodedOut)
        # Put together the input shape and the decoded input into a final model
        model = Model(inputs, decodedOut)

        self.model = model
        return model

    def fit(self, batchSize, epochs):
        # Compile the model using stochastic gradient descent, with a loss of mean squared error
        self.model.compile(optimizer='sgd', loss='mse')
        # Display the loss
        tensorBoardCallback = keras.callbacks.TensorBoard(log_dir='./log/', histogram_freq=0, write_graph=True, write_images=True)
        # Setting the epochs and the batch size(continueEpochs)
        self.model.fit(self.input, self.input, epochs=epochs, batch_size=batchSize, callbacks=[tensorBoardCallback])

    # Save calculated data so it can be used by the test file
    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')


if __name__ == '__main__':
    seed(4)
    autoencoder = Autoencoder(8)
    autoencoder.encodeDecode()
    autoencoder.fit(10, 1000)
    autoencoder.save()