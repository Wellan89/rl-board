import sys

from baselines.ppo1 import mlp_policy
import numpy as np

from envs.nph import nph

IS_CODINGAME = False


class CsbPolicy(mlp_policy.MlpPolicy):
    def __init__(self, name, ob_space, ac_space):
        super().__init__(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=96, num_hid_layers=2)

    @staticmethod
    def predict_from_weights(state, weights, deterministic=True):
        obz = np.asarray(state)
        state_mean = weights['pi/obfilter/runningsum:0'] / weights['pi/obfilter/count:0']
        state_std = np.sqrt(np.maximum(
            (weights['pi/obfilter/runningsumsq:0'] / weights['pi/obfilter/count:0']) - np.square(state_mean),
            1e-2
        ))
        obz = np.clip((obz - state_mean) / state_std, -5.0, 5.0)
        obz = np.asarray(obz, dtype=np.float32)

        action = np.tanh(nph.matmul_single(obz, weights['pi/pol/fc1/kernel:0']) + weights['pi/pol/fc1/bias:0'])
        action = np.tanh(nph.matmul_single(action, weights['pi/pol/fc2/kernel:0']) + weights['pi/pol/fc2/bias:0'])
        action = nph.matmul_single(action, weights['pi/pol/final/kernel:0']) + weights['pi/pol/final/bias:0']

        if IS_CODINGAME:
            vf = np.tanh(nph.matmul_single(obz, weights['pi/vf/fc1/kernel:0']) + weights['pi/vf/fc1/bias:0'])
            vf = np.tanh(nph.matmul_single(vf, weights['pi/vf/fc2/kernel:0']) + weights['pi/vf/fc2/bias:0'])
            vf = float(nph.matmul_single(vf, weights['pi/vf/final/kernel:0']) + weights['pi/vf/final/bias:0'])
            print(action, file=sys.stderr)
            print(np.exp(weights['pi/pol/logstd:0'][0]), file=sys.stderr)
            print('{:.2f}'.format(vf), file=sys.stderr)
            global pod_msgs
            pod_msgs = [' {:.2f}'.format(vf)] * 2

        if not deterministic:
            action = np.random.normal(action, np.exp(weights['pi/pol/logstd:0'][0]))

        return np.clip(action, 0.0, 1.0)
