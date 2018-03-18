import os

import keras
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


TENSOR_CODES = {
    'dense0/W': 'trpo/actions-and-internals/layered-network/apply/dense0/apply/linear/apply/W',
    'dense0/b': 'trpo/actions-and-internals/layered-network/apply/dense0/apply/linear/apply/b',
    'dense1/W': 'trpo/actions-and-internals/layered-network/apply/dense1/apply/linear/apply/W',
    'dense1/b': 'trpo/actions-and-internals/layered-network/apply/dense1/apply/linear/apply/b',
    'alpha/W': 'trpo/actions-and-internals/beta/parameterize/alpha/apply/W',
    'alpha/b': 'trpo/actions-and-internals/beta/parameterize/alpha/apply/b',
    'beta/W': 'trpo/actions-and-internals/beta/parameterize/beta/apply/W',
    'beta/b': 'trpo/actions-and-internals/beta/parameterize/beta/apply/b',
}


def _replace_basename(tensor_name, replacement):
    return '{}/{}'.format(replacement, '/'.join(tensor_name.split('/')[1:]))


def _read_weights_from_tf_checkpoint(filename):
    if os.path.isdir(filename):
        filename = tf.train.latest_checkpoint(filename)
    reader = pywrap_tensorflow.NewCheckpointReader(filename)

    weights = {}
    available_tensors = set(reader.get_variable_to_shape_map().keys())
    for tensor_key, tensor_name in TENSOR_CODES.items():
        if tensor_name not in available_tensors:
            tensor_name = _replace_basename(tensor_name, 'trpo-ps')
        weights[tensor_key] = reader.get_tensor(tensor_name)
    return weights


def _read_weights_from_keras(filename):
    tensor_values = {}
    with tf.Graph().as_default():
        model = keras.models.load_model(filename, compile=False)
        for tensor_key in TENSOR_CODES.keys():
            layer_weights = model.get_layer(tensor_key.split('/')[0]).get_weights()
            tensor_weights = layer_weights[0 if tensor_key.endswith('/W') else 1]
            tensor_values[tensor_key] = tensor_weights
    return tensor_values


def read_weights(filename):
    print('Reading weights from:', filename)
    if filename.endswith('.h5'):
        weights = _read_weights_from_keras(filename)
    else:
        weights = _read_weights_from_tf_checkpoint(filename)
    return weights


def restore_agent(filename, agent, task_index):
    print('Restoring from: ', filename)
    tensor_values = read_weights(filename)

    with agent.model.graph.as_default():
        all_vars = tf.global_variables()
        for tensor_key, tensor_name in TENSOR_CODES.items():
            if task_index is None:
                tensor_name_variations = [tensor_name]
            else:
                tensor_name_variations = [_replace_basename(tensor_name, 'trpo-ps'),
                                          _replace_basename(tensor_name, 'trpo-worker{}'.format(task_index))]

            for tensor_name_var in tensor_name_variations:
                matching_vars = [var for var in all_vars if var.op.name == tensor_name_var]
                assert len(matching_vars) > 0
                for matching_var in matching_vars:
                    matching_var.load(tensor_values[tensor_key], agent.model.session)
