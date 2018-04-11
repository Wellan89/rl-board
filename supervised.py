import argparse
import datetime
import functools
import json
import os
import sys

import keras
import numpy as np
import scipy.stats
from keras import backend as K
from tensorforce.contrib.openai_gym import OpenAIGym
from tqdm import tqdm

import envs
from envs.csb.checkpoint import Checkpoint
from envs.csb.world import World
from envs.csb.util import LIN, CLAMP


def _basename_no_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def _read_np_episodes(data_file):
    with np.load(data_file) as data:
        return data['x'], data['y']


def _read_np_episodes_shape(data_file):
    data_x, data_y = _read_np_episodes(data_file)
    return data_x.shape, data_y.shape


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

            # Transform targets from the genetic actions space to the agent actions space
            # (linear scales and thresholds are modified)
            for action in actions:
                for g in [0, 3]:
                    action[g] = LIN(CLAMP(action[g], 0.25, 0.75), 0.25, 0.1, 0.75, 0.9)
                for g in [1, 4]:
                    action[g] = LIN(CLAMP(action[g], 0.2, 0.8), 0.2, 0.1, 0.8, 0.9)
                for g in [2, 5]:
                    if action[g] < 0.05:
                        action[g] = 0.1
                    elif action[g] > 0.95:
                        action[g] = 0.9
                    else:
                        action[g] = 0.5
                assert all(0.0 <= v <= 1.0 for v in action)

            x.append(env.compute_custom_state(world, opponent_view=False))
            y.append(actions[0])
            x.append(env.compute_custom_state(world, opponent_view=True))
            y.append(actions[1])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError('NaN value in data file {}'.format(data_file))
    return x, y


def _read_txt_episodes_shape(data_file, env):
    x_shape = [0, env.observation_space.shape[0]]
    y_shape = [0, env.action_space.shape[0]]
    with open(data_file) as f:
        _, checkpoints_count = map(int, f.readline().split())
        for i in range(checkpoints_count):
            f.readline()  # checkpoints

        turn_idx = 0
        while True:
            new_turn_idx = int(f.readline())
            if new_turn_idx <= turn_idx:
                # End of the game
                break
            turn_idx = new_turn_idx

            # 4 pods + evaluations + 2 actions
            for _ in range(7):
                f.readline()  # pods

            x_shape[0] += 2
            y_shape[0] += 2

    return tuple(x_shape), tuple(y_shape)


def _do_read_data(supervised_data_dir, max_data_files, extension, read_episode_func, read_episode_shape_func):
    data_files = [os.path.join(supervised_data_dir, data_file)
                  for data_file in sorted(os.listdir(supervised_data_dir))
                  if data_file.endswith(extension)]
    if max_data_files is not None:
        data_files = data_files[:max_data_files]
    assert data_files

    x_shape = None
    y_shape = None
    ok_data_files = []
    for data_file in tqdm(data_files):
        try:
            data_x_shape, data_y_shape = read_episode_shape_func(data_file)
        except Exception:
            print('Warning: could not read data file:', data_file)
            continue
        ok_data_files.append(data_file)

        if x_shape is None:
            x_shape = data_x_shape
            y_shape = data_y_shape
        else:
            x_shape = (x_shape[0] + data_x_shape[0],) + tuple(x_shape[1:])
            y_shape = (y_shape[0] + data_y_shape[0],) + tuple(y_shape[1:])
    data_files = ok_data_files
    assert data_files

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


def _read_data(supervised_data_dir, max_data_files, env):
    if any(data_file.endswith('.npz') for data_file in os.listdir(supervised_data_dir)):
        return _do_read_data(supervised_data_dir, max_data_files, '.npz',
                             _read_np_episodes, _read_np_episodes_shape)
    else:
        return _do_read_data(supervised_data_dir, max_data_files, '.game',
                             functools.partial(_read_txt_episode, env=env),
                             functools.partial(_read_txt_episodes_shape, env=env))


def _softplus_layer(x):
    # Tensorflow, tensorforce and math needs to be imported here so that the saved model can be loaded again
    import math
    import tensorflow as tf
    from tensorforce import util
    log_eps = math.log(util.epsilon)
    x = tf.clip_by_value(t=x, clip_value_min=log_eps, clip_value_max=-log_eps)
    x = tf.log(x=(tf.exp(x=x) + 1.0)) + 1.0
    return x


class EntropyLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entropy_loss_bonus = K.variable(1.0)

    def call(self, inputs, **kwargs):
        # Tensorflow needs to be imported here so that the saved model can be loaded again
        import tensorflow as tf
        alpha, beta, alpha_beta = inputs
        log_norm = tf.lgamma(x=alpha) + tf.lgamma(x=beta) - tf.lgamma(x=alpha_beta)
        entropy = log_norm - (beta - 1.0) * tf.digamma(x=beta) - (alpha - 1.0) * tf.digamma(x=alpha) \
            + (alpha_beta - 2.0) * tf.digamma(x=alpha_beta)
        return tf.reduce_mean(entropy) * self.entropy_loss_bonus


class AdjustEntropyLossCallback(keras.callbacks.Callback):
    def __init__(self, entropy_loss_bonus, entropy_loss_weight, entropy_initial_weight):
        super().__init__()
        self.entropy_loss_bonus = entropy_loss_bonus
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_initial_weight = entropy_initial_weight

    def on_epoch_begin(self, epoch, logs=None):
        if self.entropy_loss_weight and self.entropy_initial_weight is not None \
                and self.entropy_loss_weight < self.entropy_initial_weight and epoch == 0:
            target_entropy_weight = self.entropy_initial_weight / self.entropy_loss_weight
        else:
            target_entropy_weight = 1.0
        K.set_value(self.entropy_loss_bonus, target_entropy_weight)


def _distribution_loss(y_true, y_pred):
    import tensorflow as tf
    alpha, beta = tf.split(y_pred, 2, axis=-1)
    log_norm = tf.lgamma(x=alpha) + tf.lgamma(x=beta) - tf.lgamma(x=alpha + beta)
    prob = tf.pow(x=y_true, y=alpha - 1.0) * tf.pow(x=1.0 - y_true, y=beta - 1.0) / tf.exp(log_norm)
    return -tf.reduce_mean(tf.log(prob + 1e-6))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-n', '--network', help="Network specification file")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs")
    parser.add_argument('--monitor', default=None, help="Save results to this directory")
    parser.add_argument('--data-dir', default=None, help="Directory containing the supervised training data")
    parser.add_argument('--max-data-files', type=int, default=None, help="Maximum number of data files to load")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=4096, help="Training batch size")
    parser.add_argument('--action-loss-weight', type=float, default=1.0, help="Action loss weight")
    parser.add_argument('--distribution-loss-weight', type=float, default=0.0, help="Distribution loss weight")
    parser.add_argument('--entropy-loss-weight', type=float, default=0.0, help="Entropy loss weight")
    parser.add_argument('--entropy-initial-weight', type=float, default=None, help="Entropy loss weight for the first epoch")
    parser.add_argument('--outliers-removal-frequency', type=int, default=None, help="Outliers removal frequency (epochs)")
    parser.add_argument('--outliers-removal-proportion', type=float, default=None, help="Outliers removal proportion")

    args = parser.parse_args()

    if not args.monitor:
        args.monitor = 'supervised_{}_{}_{}'.format(args.gym_id, _basename_no_ext(args.network), datetime.datetime.now().isoformat())
    if not args.data_dir:
        args.data_dir = 'logs/supervised_data_{}'.format(args.gym_id)

    with open(args.network, 'r') as fp:
        network_def = json.load(fp=fp)

    save_dir = 'logs/{}'.format(args.monitor)
    os.makedirs(save_dir, exist_ok=True)
    checkpoints_save_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoints_save_dir, exist_ok=True)

    environment = OpenAIGym(gym_id=args.gym_id)

    net_input = keras.layers.Input(shape=(environment.states['shape']))
    network = net_input
    for i, layer_def in enumerate(network_def):
        assert layer_def['type'] == 'dense'

        if layer_def.get('skip'):
            assert layer_def.get('size') is None
            units = int(network.shape[1])
        else:
            units = layer_def['size']

        layer = keras.layers.Dense(units=units, activation=layer_def['activation'],
                                   name='dense{}'.format(i))(network)

        if layer_def.get('skip'):
            layer = keras.layers.Dense(units=units, activation=None,
                                       name='dense{}-skip'.format(i))(layer)
            layer = keras.layers.add([network, layer])
            layer = keras.layers.Activation(activation=layer_def['activation'])(layer)

        network = layer

    alpha = keras.layers.Dense(units=environment.actions['shape'][0], name='alpha')(network)
    alpha = keras.layers.Lambda(_softplus_layer)(alpha)

    beta = keras.layers.Dense(units=environment.actions['shape'][0], name='beta')(network)
    beta = keras.layers.Lambda(_softplus_layer)(beta)

    alpha_beta = keras.layers.add([alpha, beta])

    x, y = _read_data(args.data_dir, args.max_data_files, environment.gym.unwrapped)

    outputs = []
    target_funcs = []
    loss_weights = {}
    loss = {}
    if args.action_loss_weight:
        action = keras.layers.Lambda(lambda x: x[0] / x[1], name='action')([beta, alpha_beta])
        outputs.append(action)
        target_funcs.append(lambda y: y)
        loss_weights['action'] = args.action_loss_weight
        loss['action'] = 'mean_absolute_error'

    if args.distribution_loss_weight:
        distribution = keras.layers.concatenate([alpha, beta], name='distribution')
        outputs.append(distribution)
        target_funcs.append(lambda y: y)
        loss_weights['distribution'] = args.distribution_loss_weight
        loss['distribution'] = _distribution_loss

    entropy_layer = EntropyLayer(name='entropy')
    if args.entropy_loss_weight:
        entropy = entropy_layer([alpha, beta, alpha_beta])
        outputs.append(entropy)
        target_funcs.append(lambda y: np.zeros(shape=y.shape[0]))
        loss_weights['entropy'] = args.entropy_loss_weight
        loss['entropy'] = lambda y_true, y_pred: y_pred

    assert outputs
    model = keras.Model([net_input], outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr), loss_weights=loss_weights, loss=loss)

    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(save_dir, 'training_log.csv')),
        keras.callbacks.ModelCheckpoint(os.path.join(checkpoints_save_dir, 'keras_model_best.h5'), save_best_only=True),
        keras.callbacks.ModelCheckpoint(os.path.join(checkpoints_save_dir, 'keras_model_latest.h5'), save_best_only=False),
        AdjustEntropyLossCallback(entropy_layer.entropy_loss_bonus, args.entropy_loss_weight, args.entropy_initial_weight),
    ]

    validation_split = 0.1
    split_at = int(x.shape[0] * (1.0 - validation_split))
    x, x_val = x[:split_at], x[split_at:]
    y, y_val = y[:split_at], y[split_at:]

    remove_outliers = args.outliers_removal_frequency and args.outliers_removal_proportion
    epoch_steps = args.outliers_removal_frequency if remove_outliers else args.epochs
    for epoch in range(0, args.epochs, epoch_steps):
        model.fit(x, [target_func(y) for target_func in target_funcs], batch_size=args.batch_size,
                  validation_data=(x_val, [target_func(y_val) for target_func in target_funcs]),
                  initial_epoch=epoch, epochs=epoch + epoch_steps, callbacks=callbacks)

        if not remove_outliers:
            break

        pred = model.predict(x)
        if len(outputs) > 1:
            pred = pred[0]
        losses = np.mean(np.abs(pred - y), axis=1)
        max_loss = np.percentile(losses, q=100.0 * (1.0 - args.outliers_removal_proportion))
        print('Losses:', scipy.stats.describe(losses))
        print('Max allowed loss:', max_loss)
        mask = losses < max_loss
        print(x.shape[0] - np.sum(mask, dtype=np.int), 'examples above threshold')
        x = x[mask]
        y = y[mask]

        if x.shape[0] < 100:
            print(x.shape[0], 'examples remaining: exiting')
            break


if __name__ == '__main__':
    main()
