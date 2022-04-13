import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from IPython import display
from qlearning import QLearning
from dqn import DQN
import pickle
import pyfolio as pf
from collections import deque


class TradingSystem():
    
    def __init__(self, env, algo, load_model=False, **kwargs):
        self.env = env
        self.possible_agents = env.possible_agents
        self.num_agents = len(env.possible_agents)
        self.algo = algo
        self.load_model = load_model
        self.max_iter = 200

        models = {}
        experiences = {}
        self.agents = {}
        
        if load_model:
            for name in self.possible_agents:
                models[name] = keras.models.load_model('saved_models/' + name + '-' + algo + '.h5')
        else:
            for name in self.possible_agents:
                models[name] = None
            
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.0001
        discount_factor = kwargs['discount_factor'] if 'discount_factor' in kwargs else 0.99
        hidden_layers = kwargs['hidden_layers'] if 'hidden_layers' in kwargs else (80, 20)
        activation = kwargs['activation'] if 'activation' in kwargs else 'relu'
        
        eps_start = kwargs['eps_start'] if 'eps_start' in kwargs else 1
        self.decay_rate = kwargs['decay_rate'] if 'decay_rate' in kwargs else 0.99995

        if algo == 'QL':
            for name in self.possible_agents:
                bandit = False if name == 'SS' else True
                agent = QLearning(learning_rate, discount_factor, eps_start,
                                 np.arange(self.env.action_spaces[name].n), self.env.observation_spaces[name].shape,
                                 models[name], hidden_layers, activation, bandit=bandit
                                )
                self.agents[name] = agent

        elif algo == 'DQN':
            
            replay_memory_size = kwargs['replay_memory_size'] if 'replay_memory_size' in kwargs else 125000
            replay_start_size = kwargs['replay_start_size'] if 'replay_start_size' in kwargs else 6250
            C = kwargs['C'] if 'C' in kwargs else 1250
            batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 4
            double = kwargs['double'] if 'double' in kwargs else True
            dueling = kwargs['dueling'] if 'dueling' in kwargs else True
            
            if load_model:
               for name in self.possible_agents:
                   with open('data/experience-' + name + '.pickle', 'rb') as handle:
                       exp = pickle.load(handle)
                   experiences[name] = deque(list(exp), maxlen=replay_memory_size)
            else:
                for name in self.possible_agents:
                    experiences[name] = None
                
            for name in self.possible_agents:
                bandit = False if name == 'SS' else True
                agent = DQN(learning_rate, discount_factor, eps_start,
                           np.arange(self.env.action_spaces[name].n), self.env.observation_spaces[name].shape,
                           models[name], experiences[name], hidden_layers, activation, replay_memory_size,
                           replay_start_size, C, batch_size, double, dueling, bandit=bandit
                          )
                self.agents[name] = agent
    
        else:
            raise ValueError('Unavailable algorithm')
  
        self.actions = {agent: None for agent in self.possible_agents}
        
        # To store metrics
        self.rewards = {agent: [] for agent in self.possible_agents}
        self.Q_values = {}
          
    def train(self, num_episodes):
        self.env.set_mode('train')
        self.env.reset()
        
        # Initialise dictionary in which to store Q-values
        for agent in self.possible_agents:
            for j in range(self.env.action_spaces[agent].n):
                key = agent + ' ' + str(j)
                self.Q_values[key] = np.zeros(num_episodes)
                
        for i_episode in range(num_episodes):
            eps = self.agents['BS'].epsilon
            episode_rewards = {agent: 0 for agent in self.possible_agents}
            count = 0
            for agent in self.env.agent_iter():
                obs, reward, done, info = self.env.last()
                self.agents[agent].update(self.actions[agent], obs, reward, done, info['Abort Update'])
                episode_rewards[agent] += reward
                if done:
                    action = None  # Required by PettingZoo
                else:
                    action = self.agents[agent].policy(obs)
                    
                    # Store Q-values
                    for j in range(self.env.action_spaces[agent].n):
                        key = agent + ' ' + str(j)
                        self.Q_values[key][i_episode] = self.agents[agent].Q_values[0][j]
                        
                self.actions[agent] = action
                self.env.step(action)
                
                # Check number of iterations
                count += 1
                if count > self.max_iter: 
                    break
                    
            # Decay epsilon        
            for agent in self.agents.values():
                agent.epsilon *= self.decay_rate
                
            self.env.reset()
            
            display.clear_output(wait=True)
            print(f'Episode: {i_episode+1}')
            
            for key in self.rewards.keys():
                self.rewards[key].append(episode_rewards[key])
                
        # Save trained model
        for key, value in self.agents.items():
            keras.models.save_model(self.agents[key].model, 'saved_models/' + key + '-' + self.algo + '.h5')
            
        # Save experience
        if self.algo == 'DQN':
            for key, value in self.agents.items():
                with open('data/experience-' + key + '.pickle', 'wb')  as handle:
                    pickle.dump(self.agents[key].D, handle)
                
        # Save results
        results = pd.DataFrame.from_dict(self.rewards)
        results.index.rename('Episode', inplace=True)
        for agent in self.possible_agents:
            for j in range(self.env.action_spaces[agent].n):
                key1 = agent + ' Q-Value ' + str(j)
                key2 = agent + ' ' + str(j)
                results[key1] = self.Q_values[key2]

        # If we're loading a model, we don't want to override the results file
        if self.load_model:
            prev_results = pd.read_csv('data/results-'+ self.algo + '.csv')
            comb_results = prev_results.append(results, ignore_index=True)
            comb_results.to_csv('data/results-'+ self.algo + '.csv', index=False)
        else:
            results.to_csv('data/results-'+ self.algo + '.csv', index=False)
            
        print(f'Epsilon: {eps:.4f}')

    def test(self):
        
        for agent in self.agents.values():
            agent.epsilon = 0.05  # To prevent overfitting
            agent.num_updates = 1  # To ensure it takes greedy actions
        
        self.env.set_mode('test')
        self.env.reset()
        self.env.delta = self.env.price_history
        self.env.close = self.env.test_data['Close']
        current_day = self.env.delta-self.env.price_history
        returns = pd.Series(data=np.zeros(self.env.num_test_days), dtype=np.float64)
        while current_day < (self.env.num_test_days - 1):
            for agent in self.env.agent_iter():
                obs, _, done, _ = self.env.last()
                if done:
                    action = None
                else:
                    action = self.agents[agent].policy(obs)
                self.env.step(action)
                current_day = self.env.delta-self.env.price_history
                
                # Record returns
                if self.num_agents == 1:
                    if action == 1:
                        returns.iloc[current_day-self.env.holding_period:current_day] = self.env.close.pct_change().iloc[self.env.delta-self.env.holding_period:self.env.delta]
                elif self.num_agents == 2:
                    if action is not None:
                        if (agent == 'BS') and (action == 1):
                            returns.iloc[current_day] = (self.env.close.iloc[self.env.delta] - self.env.buy_price) / self.env.buy_price
                        elif (agent == 'SS') and (action == 1):
                            returns.iloc[current_day] = (self.env.close.iloc[self.env.delta] - self.env.close.iloc[self.env.delta-1]) / self.env.close.iloc[self.env.delta-1]
                elif self.num_agents == 4:
                    if action is not None:
                        if agent == 'BO':
                            if self.env.buy_price is not None:
                                returns.iloc[current_day] = (self.env.close.iloc[self.env.delta] - self.env.buy_price) / self.env.buy_price
                        elif agent == 'SS':
                            if action == 1:
                                returns.iloc[current_day] = (self.env.close.iloc[self.env.delta] - self.env.close.iloc[self.env.delta-1]) / self.env.close.iloc[self.env.delta-1]
                        elif agent == 'SO':
                            returns.iloc[current_day] = (self.env.sell_price - self.env.close.iloc[self.env.delta-1]) / self.env.close.iloc[self.env.delta-1]
                
                # Stop if at end of test data
                if current_day == (self.env.num_test_days - 1): 
                    break
                    
            self.env.reset()
        
        returns.index = self.env.test_dates
        pf.create_simple_tear_sheet(returns)
        
        return returns
    

# For use when stopping prematurely
def manual_save(system, num):

    # Save trained model
    for key, value in system.agents.items():
        keras.models.save_model(system.agents[key].model, 'saved_models/' + key + '-' + system.algo + '.h5')

    # Save experience
    if system.algo == 'DQN':
        for key, value in system.agents.items():
            with open('data/experience-' + key + '.pickle', 'wb')  as handle:
                pickle.dump(system.agents[key].D, handle)

    # Save results
    results = pd.DataFrame.from_dict(system.rewards)
    results.index.rename('Episode', inplace=True)
    for agent in system.possible_agents:
        for j in range(system.env.action_spaces[agent].n):
            key1 = agent + ' Q-Value ' + str(j)
            key2 = agent + ' ' + str(j)
            results[key1] = system.Q_values[key2][:num]

    # If we're loading a model, we don't want to override the results file
    if system.load_model:
        prev_results = pd.read_csv('data/results-'+ system.algo + '.csv')
        comb_results = prev_results.append(results, ignore_index=True)
        comb_results.to_csv('data/results-'+ system.algo + '.csv', index=False)
    else:
        results.to_csv('data/results-'+ system.algo + '.csv', index=False)