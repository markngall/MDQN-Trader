import numpy as np
from tensorflow import keras
import tensorflow as tf
from collections import deque
import random


def create_network(learning_rate, num_inputs, num_actions, hidden_layers, activation, momentum, dueling, seed):
    weight_init = keras.initializers.GlorotUniform(seed=seed)
    inputs = keras.Input(shape=(num_inputs,))
    x = keras.layers.BatchNormalization()(inputs, training=False)  # Keeping in training mode because of the small batch sizes
    if dueling:
        x = keras.layers.Dense(512, activation=activation, kernel_initializer=weight_init)(x)
        x = keras.layers.BatchNormalization()(x, training=False)
        x = keras.layers.Dense(512, activation=activation, kernel_initializer=weight_init)(x)
        x = keras.layers.BatchNormalization()(x, training=False)
        
        # Value stream
        value = keras.layers.Dense(512, activation=activation, kernel_initializer=weight_init)(x)
        value = keras.layers.BatchNormalization()(value, training=False)
        value = keras.layers.Dense(1, kernel_initializer=weight_init)(value)
        
        # Advantage stream
        adv = keras.layers.Dense(512, activation=activation, kernel_initializer=weight_init)(x)
        adv = keras.layers.BatchNormalization()(adv, training=False)
        adv = keras.layers.Dense(num_actions, kernel_initializer=weight_init)(adv)
        
        # Combine streams
        outputs = (value + (adv - tf.math.reduce_mean(adv, axis=1, keepdims=True)))
        
    else:
        x = keras.layers.Dense(hidden_layers[0], activation=activation, kernel_initializer=weight_init)(x)
        x = keras.layers.BatchNormalization()(x, training=False)
        for i in range(1, len(hidden_layers)):
            x = keras.layers.Dense(hidden_layers[i], activation=activation, kernel_initializer=weight_init)(x)
            x = keras.layers.BatchNormalization()(x, training=False)
        outputs = keras.layers.Dense(num_actions, kernel_initializer=weight_init)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum), loss=keras.losses.MeanSquaredError())
    return model

                  
class DQN():
    
    def __init__(self, learning_rate, discount_factor, eps_start, action_space, obs_shape, trained_model, experience, hidden_layers, activation, replay_memory_size, replay_start_size, C, batch_size, double, dueling, seed=1, bandit=False):
        self.discount_factor = discount_factor
        self.epsilon = eps_start  # Will linearly anneal
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.prev_obs = None
        self.bandit = bandit
        self.num_updates = 0
        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.C = C
        self.double = double
        self.Q_values = np.zeros((1, self.num_actions))

        # Flatten obs_shape
        num_inputs = np.prod(obs_shape)
        
        self.batch_size = batch_size
        self.X = np.zeros((self.batch_size, num_inputs))
        self.y = np.zeros((self.batch_size, self.num_actions))
        
        if trained_model is not None:
            self.model = trained_model
            self.target_model = trained_model
            self.D = experience
            self.rng = np.random.default_rng(seed+1)  # So that the agent takes different actions this time around
            random.seed(seed+1)
        else:
            self.rng = np.random.default_rng(seed=seed)
            random.seed(seed)
        
            # Initialise action-value function and target action-value function
            self.model = create_network(learning_rate, num_inputs, self.num_actions, hidden_layers, activation, 0.95, dueling, seed)    
            self.target_model = create_network(learning_rate, num_inputs, self.num_actions, hidden_layers, activation, 0.95, dueling, seed)
            self.target_model.set_weights(self.model.get_weights()) 
            
            self.D = deque(maxlen=self.replay_memory_size)
    
    def update(self, current_action, obs, reward, done, abort_update=False):
        
        obs = obs.reshape(1, -1)
        
        if (not abort_update) and (current_action is not None):  #and (self.prev_obs is not None):
        
            # Store transition in replay memory
            self.D.append((self.prev_obs, current_action, reward, obs, done))

            # Check if we have enough experience to train network
            if len(self.D) > self.replay_start_size:

                # Sample mini-batch
                batch = random.sample(self.D, self.batch_size)

                i = 0
                for prev_obs, action, reward, obs, done in batch:
                    TD_target = reward
                    if (not done) and (not self.bandit):

                        if self.double:
                            TD_target += self.discount_factor * self.target_model(obs).numpy()[0, np.argmax(self.model(obs).numpy()[0])]
                        else:
                            TD_target += self.discount_factor * np.amax(self.target_model(obs).numpy())

                    target = self.model(prev_obs).numpy()
                    target[0][action] = TD_target
                             
                    self.X[i] = prev_obs
                    self.y[i] = target
                    i += 1
                    
                self.model.train_on_batch(self.X, self.y)

                self.num_updates += 1
                if self.num_updates % self.C == 0: 
                    self.target_model.set_weights(self.model.get_weights())
                  
    def policy(self, obs):
        obs = obs.reshape(1, -1)
        self.prev_obs = obs
        if self.num_updates > 0 :  # Choose random actions until learning starts  
            action_probs = np.ones(self.num_actions) * self.epsilon / self.num_actions  
            self.Q_values = self.model(obs).numpy() 
            best_action = np.argmax(self.Q_values[0])
            action_probs[best_action] += (1.0 - self.epsilon)
            return self.rng.choice(np.arange(self.num_actions), p=action_probs)
        else:
            return self.rng.choice(np.arange(self.num_actions))