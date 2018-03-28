import argparse
import json
import os

import keras
import numpy as np
from tensorforce.contrib.openai_gym import OpenAIGym

import envs


def _basename_no_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def _read_data(supervised_data_dir):
    data_files = [os.path.join(supervised_data_dir, data_file)
                  for data_file in sorted(os.listdir(supervised_data_dir))
                  if data_file.endswith('.npz')]

    x_shape = None
    y_shape = None
    for data_file in data_files:
        with np.load(data_file) as data:
            data_x_shape = data['x'].shape
            data_y_shape = data['y'].shape
            if x_shape is None:
                x_shape = data_x_shape
                y_shape = data_y_shape
            else:
                x_shape = (x_shape[0] + data_x_shape[0],) + tuple(x_shape[1:])
                y_shape = (y_shape[0] + data_y_shape[0],) + tuple(y_shape[1:])

    x = np.empty(shape=x_shape, dtype=np.float32)
    y = np.empty(shape=y_shape, dtype=np.float32)
    x_cur = 0
    y_cur = 0
    for i, data_file in enumerate(data_files):
        print('Reading', data_file)
        with np.load(data_file) as data:
            data_x = data['x']
            data_y = data['y']
            x[x_cur:(x_cur + data_x.shape[0]), ...] = data_x
            y[y_cur:(y_cur + data_y.shape[0]), ...] = data_y
            x_cur += data_x.shape[0]
            y_cur += data_y.shape[0]

    return x, y


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--monitor', default=None, help="Save results to this directory")
    parser.add_argument('--data-dir', default=None, help="Directory containing the supervised training data")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--loss', default='mean_absolute_error', help="Loss")

    args = parser.parse_args()

    if not args.monitor:
        args.monitor = 'supervised_{}_{}'.format(args.gym_id, _basename_no_ext(args.network))
    if not args.data_dir:
        args.data_dir = 'logs/supervised_data_{}'.format(args.gym_id)

    with open(args.network, 'r') as fp:
        network_def = json.load(fp=fp)

    environment = OpenAIGym(gym_id=args.gym_id)

    save_dir = 'logs/{}/checkpoints'.format(args.monitor)

    net_input = keras.layers.Input(shape=(environment.states['shape']))
    network = net_input
    for i, layer in enumerate(network_def):
        assert layer['type'] == 'dense'
        network = keras.layers.Dense(units=layer['size'], activation=layer['activation'], name='dense{}'.format(i))(network)

    alpha = keras.layers.Dense(units=environment.actions['shape'][0], name='alpha')(network)
    alpha = keras.layers.Activation('softplus')(alpha)
    alpha = keras.layers.Lambda(lambda x: x + 1)(alpha)

    beta = keras.layers.Dense(units=environment.actions['shape'][0], name='beta')(network)
    beta = keras.layers.Activation('softplus')(beta)
    beta = keras.layers.Lambda(lambda x: x + 1)(beta)

    network = keras.layers.Lambda(lambda x: x[1] / (x[0] + x[1]))([alpha, beta])

    model = keras.Model([net_input], [network])
    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr), loss=args.loss)

    x, y = _read_data(args.data_dir)

    model.fit(x, y, batch_size=512, epochs=args.epochs, validation_split=0.1)
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir + '/keras_model.h5')


if __name__ == '__main__':
    main()
