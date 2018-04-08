import argparse
import base64
import json
import os

import numpy as np

import checkpoints_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l')
    args = parser.parse_args()

    weights = checkpoints_utils.read_weights(args.l)
    weights = {
        tensor_code: (tensor.shape, base64.encodebytes(tensor.astype(np.float16).tobytes()).decode())
        for tensor_code, tensor in weights.items()
    }
    print(json.dumps(weights))


if __name__ == '__main__':
    main()
