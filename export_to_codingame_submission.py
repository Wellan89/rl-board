import argparse
import base64
import json
import os

import numpy as np

import checkpoints_utils


def _get_encoded_tensor_value(tensor):
    return base64.encodebytes(tensor.tobytes()).decode()


def _encode_tensor(tensor):
    tensor_encoding = _get_encoded_tensor_value(tensor)
    if len(tensor_encoding) < 2000:
        return tensor.shape, tensor.dtype.name, tensor_encoding

    tensor = tensor.astype(np.float16)
    return tensor.shape, tensor.dtype.name, _get_encoded_tensor_value(tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l')
    parser.add_argument('--vf', type=int, default=1)
    args = parser.parse_args()

    weights = checkpoints_utils.read_weights(args.l)
    weights = {tensor_code: _encode_tensor(tensor) for tensor_code, tensor in weights.items()
               if args.vf or '/vf/' not in tensor_code}
    print(json.dumps(weights))


if __name__ == '__main__':
    main()
