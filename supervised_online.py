import argparse
import json
import multiprocessing
import os

import keras
import numpy as np
from tensorforce.contrib.openai_gym import OpenAIGym

import envs
from envs.csb.vincent_algo import VincentSalimInterface


def _basename_no_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def _generate_episode_data(episode_id, gym_id, monitor):
    if episode_id % 100 == 0:
        print('Computing game', episode_id)

    try:
        environment = OpenAIGym(
            gym_id=gym_id,
            monitor=monitor if episode_id == 0 else None,
            monitor_video=1 if episode_id == 0 else 0
        )
        state = environment.reset()
        world = environment.gym.unwrapped.world
        interface = VincentSalimInterface()
        interface.start(world)
        episode = []
        while True:
            interface.feed(world)
            actions = interface.get_moves(world, 0)
            action = [actions[0].g1, actions[0].g2, actions[0].g3, actions[1].g1, actions[1].g2, actions[1].g3]
            episode.append((state, action))
            state, terminal, step_reward = environment.execute(action)
            if terminal:
                break
        return episode
    except Exception:
        print('An exception occurred during game {} generation!'.format(episode_id))
        return []


def _generate_episodes_data(initial_episode, n_episodes, gym_id, monitor):
    with multiprocessing.Pool() as p:
        episodes = p.starmap(_generate_episode_data,
                             [(initial_episode + episode_id, gym_id, monitor) for episode_id in range(n_episodes)])

    x = np.array([state for episode in episodes for state, _ in episode])
    y = np.array([action for episode in episodes for _, action in episode])
    return x, y


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs")
    parser.add_argument('--episodes', type=int, default=100, help="Number of episodes per epoch")
    parser.add_argument('--episodes-repeat', type=int, default=1, help="Number of repetitions per episode")
    parser.add_argument('--monitor', help="Save results to this directory")

    args = parser.parse_args()

    if not args.monitor:
        args.monitor = 'supervised_{}_{}'.format(args.gym_id, _basename_no_ext(args.network))

    with open(args.network, 'r') as fp:
        network_def = json.load(fp=fp)

    environment = OpenAIGym(gym_id=args.gym_id)

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

    for epoch in range(args.epochs):
        x, y = _generate_episodes_data(epoch * args.episodes, args.episodes,
                                       args.gym_id, 'logs/{}/gym'.format(args.monitor))

        model_loss = model.evaluate(x, y, batch_size=4096)
        print('Model loss on new data: {:.5f}'.format(model_loss))

        model.fit(x, y, batch_size=512,
                  epochs=args.episodes_repeat, initial_epoch=epoch * args.episodes_repeat)

        save_dir = 'logs/{}/checkpoints'.format(args.monitor)
        os.makedirs(save_dir, exist_ok=True)
        model.save(save_dir + '/keras_model.h5')
        print('Model saved in: {}/keras_model.h5'.format(save_dir))


if __name__ == '__main__':
    main()
