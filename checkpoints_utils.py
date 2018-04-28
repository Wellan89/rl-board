import keras
import numpy as np
import tensorflow as tf

import supervised


TENSOR_CODES = {
    'pol_fc1/W': 'pi/pol/fc1/kernel:0',
    'pol_fc1/b': 'pi/pol/fc1/bias:0',
    'pol_fc2/W': 'pi/pol/fc2/kernel:0',
    'pol_fc2/b': 'pi/pol/fc2/bias:0',
    'pol_final/W': 'pi/pol/final/kernel:0',
    'pol_final/b': 'pi/pol/final/bias:0',
}


def _read_weights_from_keras(filename):
    tensor_values = {}
    with tf.Graph().as_default():
        with tf.Session().as_default():
            model = keras.models.load_model(filename, compile=False,
                                            custom_objects={'EntropyLayer': supervised.EntropyLayer})
            for tensor_key, tensor_target in TENSOR_CODES.items():
                layer_weights = model.get_layer(tensor_key.split('/')[0]).get_weights()
                tensor_weights = layer_weights[0 if tensor_key.endswith('/W') else 1]
                tensor_values[tensor_target] = tensor_weights
    return tensor_values


def _read_weights_from_numpy(filename):
    data = np.load(filename)
    return {k: data[k] for k in data.files}


def read_weights(filename):
    if filename.endswith('.h5'):
        return _read_weights_from_keras(filename)
    elif filename.endswith('.npz'):
        return _read_weights_from_numpy(filename)
    raise ValueError('Unknown file type: {}'.format(filename))
