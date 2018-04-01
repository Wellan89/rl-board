import argparse
import json
import os

import keras
import numpy as np
from tensorforce.contrib.openai_gym import OpenAIGym
from tqdm import tqdm

import envs
from envs.csb.checkpoint import Checkpoint
from envs.csb.world import World


def _basename_no_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def _read_np_episodes(data_file):
    with np.load(data_file) as data:
        return data['x'], data['y']


def _read_txt_episode(data_file, env):
    x = []
    y = []
    with open(data_file) as f:
        world = World()
        world.nblaps, checkpoints_count = map(int, f.readline().split())
        world.circuit.cps = []
        for i in range(checkpoints_count):
            cp_x, cp_y = map(int, f.readline().split())
            world.circuit.cps.append(Checkpoint(i, cp_x, cp_y))

        turn_idx = 0
        while True:
            new_turn_idx = int(f.readline())
            if new_turn_idx <= turn_idx:
                # End of the game
                break
            turn_idx = new_turn_idx

            for p in world.pods:
                p.x, p.y, p.vx, p.vy, p.angle, p.ncpid, p.lap, p.timeout, p.shield, p.boost_available = \
                    map(float, f.readline().split())
                p.ncpid = int(p.ncpid)

            f.readline()  # skip evaluations

            actions = [list(map(float, f.readline().split())) for _ in range(2)]
            x.append(env.compute_custom_state(world, opponent_view=False))
            y.append(actions[0])
            x.append(env.compute_custom_state(world, opponent_view=True))
            y.append(actions[1])

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def _do_read_data(supervised_data_dir, extension, read_episode_func):
    data_files = [os.path.join(supervised_data_dir, data_file)
                  for data_file in sorted(os.listdir(supervised_data_dir))
                  if data_file.endswith(extension)]
    assert data_files
    data_files = data_files[:100]

    x_shape = None
    y_shape = None
    for data_file in tqdm(data_files):
        data_x, data_y = read_episode_func(data_file)
        data_x_shape = data_x.shape
        data_y_shape = data_y.shape
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
    for i, data_file in enumerate(tqdm(data_files)):
        data_x, data_y = read_episode_func(data_file)
        x[x_cur:(x_cur + data_x.shape[0]), ...] = data_x
        y[y_cur:(y_cur + data_y.shape[0]), ...] = data_y
        x_cur += data_x.shape[0]
        y_cur += data_y.shape[0]

    return x, y


def _read_data(supervised_data_dir, env):
    if any(data_file.endswith('.npz') for data_file in os.listdir(supervised_data_dir)):
        return _do_read_data(supervised_data_dir, '.npz', _read_np_episodes)
    else:
        return _do_read_data(supervised_data_dir, '.game', lambda e: _read_txt_episode(e, env))


def _softplus_layer(x):
    # Tensorflow, tensorforce and math needs to be imported here so that the saved model can be loaded again
    import math
    import tensorflow as tf
    from tensorforce import util
    log_eps = math.log(util.epsilon)
    x = tf.clip_by_value(t=x, clip_value_min=log_eps, clip_value_max=-log_eps)
    x = tf.log(x=(tf.exp(x=x) + 1.0)) + 1.0
    return x


def _entropy_layer(x):
    # Tensorflow needs to be imported here so that the saved model can be loaded again
    import tensorflow as tf
    alpha, beta, alpha_beta = x
    log_norm = tf.lgamma(x=alpha) + tf.lgamma(x=beta) - tf.lgamma(x=alpha_beta)
    entropy = log_norm - (beta - 1.0) * tf.digamma(x=beta) - (alpha - 1.0) * tf.digamma(x=alpha) \
        + (alpha_beta - 2.0) * tf.digamma(x=alpha_beta)
    return tf.reduce_mean(entropy)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--monitor', default=None, help="Save results to this directory")
    parser.add_argument('--data-dir', default=None, help="Directory containing the supervised training data")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--loss', default='mean_absolute_error', help="Loss")
    parser.add_argument('--entropy-loss-weight', type=float, default=0.05, help="Entropy loss weight")

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
        network = keras.layers.Dense(units=layer['size'], activation=layer['activation'],
                                     name='dense{}'.format(i))(network)

    alpha = keras.layers.Dense(units=environment.actions['shape'][0], name='alpha')(network)
    alpha = keras.layers.Lambda(_softplus_layer)(alpha)

    beta = keras.layers.Dense(units=environment.actions['shape'][0], name='beta')(network)
    beta = keras.layers.Lambda(_softplus_layer)(beta)

    alpha_beta = keras.layers.add([alpha, beta])
    action = keras.layers.Lambda(lambda x: x[0] / x[1], name='action')([beta, alpha_beta])

    outputs = [action]
    loss_weights = {'action': 1.0}
    loss = {'action': args.loss}

    if args.entropy_loss_weight:
        entropy = keras.layers.Lambda(_entropy_layer, name='entropy')([alpha, beta, alpha_beta])
        outputs.append(entropy)
        loss_weights['entropy'] = args.entropy_loss_weight
        loss['entropy'] = lambda y_true, y_pred: y_pred

    model = keras.Model([net_input], outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr), loss_weights=loss_weights, loss=loss)

    x, y = _read_data(args.data_dir, environment.gym.unwrapped)
    if args.entropy_loss_weight:
        y = [y, np.zeros(shape=y.shape[0])]

    model_name = 'keras_model_loss={}_lr={}_entropy={}.h5'.format(args.loss, args.lr, args.entropy_loss_weight)
    os.makedirs(save_dir, exist_ok=True)
    model.fit(x, y, batch_size=512, epochs=args.epochs, validation_split=0.1,
              callbacks=[keras.callbacks.ModelCheckpoint(os.path.join(save_dir, model_name), save_best_only=True)])


if __name__ == '__main__':
    main()
