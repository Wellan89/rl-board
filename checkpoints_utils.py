import os

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import supervised


TENSOR_CODES = {
    'dense0/W': '[BASENAME]/actions-and-internals/layered-network/apply/dense0/apply/linear/apply/W',
    'dense0/b': '[BASENAME]/actions-and-internals/layered-network/apply/dense0/apply/linear/apply/b',
    'dense1/W': '[BASENAME]/actions-and-internals/layered-network/apply/dense1/apply/linear/apply/W',
    'dense1/b': '[BASENAME]/actions-and-internals/layered-network/apply/dense1/apply/linear/apply/b',
    'alpha/W': '[BASENAME]/actions-and-internals/beta/parameterize/alpha/apply/W',
    'alpha/b': '[BASENAME]/actions-and-internals/beta/parameterize/alpha/apply/b',
    'beta/W': '[BASENAME]/actions-and-internals/beta/parameterize/beta/apply/W',
    'beta/b': '[BASENAME]/actions-and-internals/beta/parameterize/beta/apply/b',
}


def _find_agent_basename(var_names):
    for var_name in var_names:
        if var_name.startswith('trpo'):
            return 'trpo'
        elif var_name.startswith('ppo'):
            return 'ppo'
    raise ValueError('Unknown agent basename')


def _replace_basename(tensor_name, replacement):
    return '{}/{}'.format(replacement, '/'.join(tensor_name.split('/')[1:]))


def _read_weights_from_tf_checkpoint(filename):
    if os.path.isdir(filename):
        filename = tf.train.latest_checkpoint(filename)

    print('Reading weights from:', filename)
    reader = pywrap_tensorflow.NewCheckpointReader(filename)

    weights = {}
    available_tensors = set(reader.get_variable_to_shape_map().keys())
    agent_basename = _find_agent_basename(list(available_tensors))
    for tensor_key, tensor_name in TENSOR_CODES.items():
        tensor_name = tensor_name.replace('[BASENAME]', agent_basename)
        if tensor_name not in available_tensors:
            tensor_name = _replace_basename(tensor_name, '{}-ps'.format(agent_basename))
        weights[tensor_key] = reader.get_tensor(tensor_name)
    return weights


def _read_weights_from_keras(filename):
    tensor_values = {}
    with tf.Graph().as_default():
        print('Reading weights from:', filename)
        model = keras.models.load_model(filename, compile=False,
                                        custom_objects={'EntropyLayer': supervised.EntropyLayer})
        for tensor_key in TENSOR_CODES.keys():
            layer_weights = model.get_layer(tensor_key.split('/')[0]).get_weights()
            tensor_weights = layer_weights[0 if tensor_key.endswith('/W') else 1]
            tensor_values[tensor_key] = tensor_weights
    return tensor_values


def _read_weights_from_numpy(filename):
    data = np.load(filename)
    return {k: data[k] for k in data.files}


def read_weights(filename):
    if filename.endswith('.h5'):
        weights = _read_weights_from_keras(filename)
    elif filename.endswith('.npz'):
        weights = _read_weights_from_numpy(filename)
    else:
        weights = _read_weights_from_tf_checkpoint(filename)
    return weights


def restore_agent(filename, agent, task_index):
    print('Restoring agent from:', filename)
    tensor_values = read_weights(filename)

    with agent.model.graph.as_default():
        all_vars = tf.global_variables()
        agent_basename = _find_agent_basename([var.op.name for var in all_vars])
        for tensor_key, tensor_name in TENSOR_CODES.items():
            tensor_name = tensor_name.replace('[BASENAME]', agent_basename)
            if task_index is None:
                tensor_name_variations = [tensor_name]
            else:
                tensor_name_variations = [
                    _replace_basename(tensor_name, '{}-ps'.format(agent_basename)),
                    _replace_basename(tensor_name, '{}-worker{}'.format(agent_basename, task_index))
                ]

            for tensor_name_var in tensor_name_variations:
                matching_vars = [var for var in all_vars if var.op.name == tensor_name_var]
                assert len(matching_vars) > 0
                for matching_var in matching_vars:
                    matching_var.load(tensor_values[tensor_key], agent.model.session)
