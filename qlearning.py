import numpy as np
from tensorflow import keras
import tensorflow as tf


class QLearning():
    
    def __init__(self, learning_rate, discount_factor, eps_start, action_space, obs_shape, trained_model, hidden_layers, activation, seed=123, bandit=False): 
        self.discount_factor = discount_factor
        self.epsilon = eps_start  # Will linearly anneal
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.prev_obs = None
        self.bandit = bandit
        self.Q_values = None
        
        # Flatten obs_shape
        num_inputs = np.prod(obs_shape)
        
        if trained_model is not None:
            self.model = trained_model
            self.rng = np.random.default_rng(seed+1)  # So that the agent takes different actions this time around
        else:
            self.rng = np.random.default_rng(seed)
            
            # Using the functional API
            weight_init = keras.initializers.RandomNormal(seed=seed)
            bias_init = keras.initializers.Ones()
            inputs = keras.Input(shape=(num_inputs,))
            x = keras.layers.BatchNormalization()(inputs, training=False)
            for i in range(0, len(hidden_layers)):
                x = keras.layers.Dense(hidden_layers[i], activation=activation, kernel_initializer=weight_init,
                                       bias_initializer=bias_init)(x)
                x = keras.layers.BatchNormalization()(x, training=False)
            outputs = keras.layers.Dense(self.num_actions, kernel_initializer=weight_init, bias_initializer=bias_init)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95, clipvalue=1.0), loss=keras.losses.MeanSquaredError())
            self.model = model
        
    # Updates NN using SGD
    # Agent takes current_action, the environment steps and the agent receives both an observation and a reward
    def update(self, current_action, obs, reward, done, abort_update=False):
        
        # Should flatten obs in case it is 2D
        
        obs = obs.reshape(1, -1)  # Change so that it can handle a 2D obs
        
        if (not abort_update) and (current_action is not None):
            
            TD_target = reward
            if (not done) and (not self.bandit):
                TD_target += self.discount_factor * np.amax(self.model(obs).numpy())
            target = np.copy(self.Q_values)
            target[0][current_action] = TD_target

            # Runs a single gradient update on a single step
            self.model.train_on_batch(self.prev_obs, target)
        
    def policy(self, obs):
        obs = obs.reshape(1, -1)
        self.prev_obs = obs
        action_probs = np.ones(self.num_actions) * self.epsilon / self.num_actions
        self.Q_values = self.model(obs).numpy()
        best_action = np.argmax(self.Q_values[0])
        action_probs[best_action] += (1.0 - self.epsilon)
        return self.rng.choice(np.arange(self.num_actions), p=action_probs)