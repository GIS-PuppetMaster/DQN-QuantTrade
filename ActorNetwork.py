import numpy as np
from keras.callbacks import *
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Flatten, Conv1D, Input, Concatenate, BatchNormalization, Activation
from keras.utils import plot_model
import glo
from math import *


class ActorNetwork(object):
    def __init__(self, sess, stock_state_size, agent_state_size, action_size, TAU, LEARNING_RATE):
        self.sess = sess
        self.stock_state_size = stock_state_size
        self.agent_state_size = agent_state_size
        self.action_size = action_size
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        self.model, self.weights, self.stock_state, self.agent_state = self.build_actor_network(stock_state_size,
                                                                                                agent_state_size)
        self.target_model, self.target_weights, self.target_stock_state, self.target_agent_state = self.build_actor_network(
            stock_state_size,
            agent_state_size)
        self.action_gradients = tf.placeholder(tf.float32, [None, action_size])
        self.params_gradients = tf.gradients(self.model.output, self.weights, -self.action_gradients)
        grad = zip(self.params_gradients, self.weights)
        self.global_step = tf.Variable(0, trainable=False)
        self.learn_rate = tf.train.exponential_decay(LEARNING_RATE, self.global_step, 20000, 0.9)
        self.optimize = tf.train.AdamOptimizer(self.learn_rate).apply_gradients(grad, global_step=self.global_step)
        self.sess.run(tf.initialize_all_variables())

    def train(self, stock_state, agent_state, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.stock_state: stock_state,
            self.agent_state: agent_state,
            self.action_gradients: action_grads
        })

    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.TAU * weights[i] + (1 - self.TAU) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def build_actor_network(self, stock_state_size, agent_state_size):
        """
           输入：state(stock,agent)
           输出：action
           loss：max(q)，即-tf.reduce_mean(q)
           :return:actor_net_model,weights,stock_state,agent_state
           """
        input_stock_state = Input(shape=(stock_state_size, glo.count))
        # input_stock_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_stock_state)
        input_agent_state = Input(shape=(agent_state_size,))
        # input_agent_state_ = BatchNormalization(epsilon=1e-4, scale=True, center=True)(input_agent_state)
        # x_stock_state = Conv1D(filters=25, kernel_size=2, padding='same')(input_stock_state_)
        # x_stock_state = BatchNormalization(axis=2, epsilon=1e-4, scale=True, center=True)(x_stock_state)
        x_stock_state = Flatten()(input_stock_state)
        # x_stock_state = Activation('tanh')(x_stock_state)
        dense01 = Dense(64)(x_stock_state)
        # dense01 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
        dense01 = Activation('tanh')(dense01)
        dense01 = Dense(8)(dense01)
        # dense01 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense01)
        dense01 = Activation('tanh')(dense01)
        merge_layer = Concatenate()([dense01, input_agent_state])
        dense02 = Dense(8)(merge_layer)
        # dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
        dense02 = Activation('tanh')(dense02)
        dense02 = Dense(4)(dense02)
        # dense02 = BatchNormalization(epsilon=1e-4, scale=True, center=True)(dense02)
        dense02 = Activation('tanh')(dense02)
        output = Dense(self.action_size, name='output', activation='tanh')(dense02)
        model = Model(inputs=[input_stock_state, input_agent_state], outputs=[output])
        plot_model(model, to_file='actor_net.png', show_shapes=True)
        return model, model.trainable_weights, input_stock_state, input_agent_state
