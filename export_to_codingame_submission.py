import argparse
import base64
import json
import os

import numpy as np
from tensorflow.python import pywrap_tensorflow


def main():
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-l')
    args = parser.parse_args()

    reader = pywrap_tensorflow.NewCheckpointReader(args.l)
    variable_to_shape = reader.get_variable_to_shape_map()
    for tensor_code, tensor_name in tensor_codes.items():
        print('{}: {}'.format(tensor_code, variable_to_shape[tensor_name]))

    tensors = {
        tensor_code: (variable_to_shape[tensor_name], base64.encodebytes(reader.get_tensor(tensor_name).astype(np.float16).tobytes()).decode())
        for tensor_code, tensor_name in tensor_codes.items()
    }

    os.makedirs('./submissions', exist_ok=True)
    with open('./submissions/csb.json', 'w') as f:
        json.dump(tensors, f)


if __name__ == '__main__':
    main()
