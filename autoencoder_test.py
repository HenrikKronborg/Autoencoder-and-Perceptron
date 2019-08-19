from keras.models import load_model
import numpy as np

encoder = load_model(r'./weights/encoder_weights.h5')
decoder = load_model(r'./weights/decoder_weights.h5')

inputs = np.array([-3,0.6,11])
encoderPredict = encoder.predict(inputs)
decoderPredict = decoder.predict(encoderPredict)

print('Input: {}'.format(inputs))
print('Encoded: {}'.format(encoderPredict))
print('Decoded: {}'.format(decoderPredict))