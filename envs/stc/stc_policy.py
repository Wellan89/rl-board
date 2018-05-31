import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype


def _grid_cnn(x):
    x = tf.reshape(x, shape=[-1, 12, 6])
    x = tf.one_hot(x, depth=7)
    x = tf.nn.tanh(U.conv2d(x, 12, 'l1', [3, 3], [1, 1], pad='VALID'))
    x = tf.nn.tanh(U.conv2d(x, 12, 'l2', [3, 3], [1, 1], pad='VALID'))
    x = U.flattenallbut0(x)
    return x


class StcPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):
        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name='ob', dtype=tf.int32, shape=[sequence_length] + list(ob_space.shape))
        next_blocks, my_grid, opp_grid = tf.split(ob, [16, 12 * 6, 12 * 6], axis=1)

        with tf.variable_scope('next_blocks'):
            next_blocks = tf.one_hot(next_blocks, depth=5)
            next_blocks = U.flattenallbut0(next_blocks)
            next_blocks = tf.nn.tanh(tf.layers.dense(next_blocks, 12, name='l1', kernel_initializer=U.normc_initializer(1.0)))
            next_blocks = tf.nn.tanh(tf.layers.dense(next_blocks, 12, name='l2', kernel_initializer=U.normc_initializer(1.0)))

        with tf.variable_scope('grids', reuse=False):
            my_grid = _grid_cnn(my_grid)

        with tf.variable_scope('grids', reuse=True):
            opp_grid = _grid_cnn(opp_grid)

        x = tf.concat([next_blocks, my_grid, opp_grid], axis=1)
        x = tf.nn.tanh(tf.layers.dense(x, 64, name='lin', kernel_initializer=U.normc_initializer(1.0)))

        logits = tf.layers.dense(x, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=U.normc_initializer(1.0))[:,0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def batch_act(self, stochastic, obs):
        return self._act(stochastic, obs)
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

