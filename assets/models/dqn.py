import tensorflow as tf
import numpy as np
import random
from collections import deque

HIDDEN = 24  # Number of hidden neurons
REPLAY_SIZE = 1_00_000  # SIze of experiences buffer
EPSILON = 1  # Initial
EPSILON_DECAY = 0.997  # Exponential Decay
EPSILON_MIN = 0.01  # Minimum annealing
LEARNING_RATE = 0.001  # For the optimizer
GAMMA = 0.96  # Discounter
BATCH_SIZE = 64  # Training experiences to be trained on
TAU = 0.01  # Soft Update parameter


class DQN:
    """The Brain of the project"""

    def __init__(self, action_space, observation_space, use_double=True, use_er=True,
                 hidden=HIDDEN,
                 replay_size=REPLAY_SIZE, batch_size=BATCH_SIZE,
                 epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                 lr=LEARNING_RATE,
                 gamma=GAMMA,
                 tau=TAU):
        self.action_space = action_space
        self.observation_space = observation_space
        self.hidden = hidden
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.network = self._make_model()
        self.use_double = use_double
        self.use_er = use_er
        if use_double:
            self.target_network = self._make_model()
        if use_er:
            self.experience_memory = deque(maxlen=replay_size)
            self.batch_size = batch_size

    def _make_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=self.hidden, activation='relu', input_dim=self.observation_space),
            tf.keras.layers.Dense(units=self.action_space, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr),
                      loss='mse')
        model.summary()
        return model

    def update_target_network(self):
        if not self.use_double:
            raise UnboundLocalError("Double DQN is disabled. Target Network not created")
        local_wts = np.array(self.network.get_weights())
        target_wts = np.array(self.target_network.get_weights())
        self.target_network.set_weights(target_wts + self.tau * (local_wts - target_wts))

    def anneal_hps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_action(self, state, evaluate=False):
        if evaluate:
            q = self.network.predict(state)
            action = np.argmax(q[0])
            return action
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            q = self.network.predict(state)
            action = np.argmax(q[0])
            return action

    def append_experience(self, state, action, reward, next_state, done):
        """
        Adds to memory
        :param state: State in
        :param action: Action performed
        :param reward: Reward gained
        :param next_state: Resultant State
        :param done: Terminal flag
        :return: None
        """
        if not self.use_er:
            raise UnboundLocalError("Experience Replay is not enabled. You cannot append experiences")
        self.experience_memory.append((state, action, reward, next_state, done))

    def train(self, experience=None):
        if self.use_er:
            batches = random.choices(self.experience_memory, k=self.batch_size)
        else:
            batches = experience


