import argparse
import base64
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def read_weights(filename):
    tensor_codes = {
        'dense0/W': 'trpo/actions-and-internals/layered-network/apply/dense0/apply/linear/apply/W',
        'dense0/b': 'trpo/actions-and-internals/layered-network/apply/dense0/apply/linear/apply/b',
        'dense1/W': 'trpo/actions-and-internals/layered-network/apply/dense1/apply/linear/apply/W',
        'dense1/b': 'trpo/actions-and-internals/layered-network/apply/dense1/apply/linear/apply/b',
        'alpha/W': 'trpo/actions-and-internals/beta/parameterize/alpha/apply/W',
        'alpha/b': 'trpo/actions-and-internals/beta/parameterize/alpha/apply/b',
        'beta/W': 'trpo/actions-and-internals/beta/parameterize/beta/apply/W',
        'beta/b': 'trpo/actions-and-internals/beta/parameterize/beta/apply/b',
    }

    if os.path.isdir(filename):
        filename = tf.train.latest_checkpoint(filename)
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    print('Reading weights from: {}'.format(filename))

    variable_to_shape = reader.get_variable_to_shape_map()
    weights = {
        tensor_code: (variable_to_shape[tensor_name], base64.encodebytes(reader.get_tensor(tensor_name).astype(np.float16).tobytes()).decode())
        for tensor_code, tensor_name in tensor_codes.items()
    }
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l')
    args = parser.parse_args()

    weights = read_weights(args.l)

    os.makedirs('./submissions', exist_ok=True)
    with open('./submissions/csb.json', 'w') as f:
        json.dump(weights, f)


if __name__ == '__main__':
    main()
