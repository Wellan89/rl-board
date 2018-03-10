import argparse
import json
import os

import keras
import numpy as np
from tensorforce.contrib.openai_gym import OpenAIGym

import envs


def _basename_no_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs")
    parser.add_argument('--monitor', help="Save results to this directory")

    args = parser.parse_args()

    if not args.monitor:
        args.monitor = 'supervised_{}_{}'.format(args.gym_id, _basename_no_ext(args.network))

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
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss='mean_squared_error')

    supervised_data_path = 'logs/supervised_data_{}/data.npz'.format(args.gym_id)
    with np.load(supervised_data_path) as data:
        x = data['x']
        y = data['y']
    model.fit(x, y, batch_size=512, epochs=args.epochs, validation_split=0.1)
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir + '/keras_model.h5')


if __name__ == '__main__':
    main()
