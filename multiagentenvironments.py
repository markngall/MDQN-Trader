from oneenv import *
from twoenv import *
from fourenv import *


class MQOne(OneEnv):
    
    def __init__(self, load=False):
        self.load = load
        super().__init__()
        super().MQ_data()
        self.observation_spaces = {'BS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape,), dtype=np.float64)}
        
    def observe(self, agent):
        signal_observations = self.signal_observations if self.mode == 'train' else self.signal_observations_test
        return signal_observations[0, self.delta]
        
        
class MQTwo(TwoEnv):
    
    def __init__(self, load=False):
        self.load = load
        super().__init__()
        super().MQ_data()
        self.observation_spaces = {'BS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape,), dtype=np.float64), 
                                   'SS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape+1,), dtype=np.float64)
                                  }
        
    def observe(self, agent):
        signal_observations = self.signal_observations if self.mode == 'train' else self.signal_observations_test
        if agent == 'BS':
            return signal_observations[0, self.delta]
        elif agent == 'SS':
            matrix = signal_observations[0, self.delta]
            profit = (self.close.iloc[self.delta] - self.buy_price) / self.buy_price
            return np.append(matrix, profit)
        
        
class MQFour(FourEnv):
    
    def __init__(self, load=False):
        self.load = load
        super().__init__()
        super().MQ_data()
        self.observation_spaces = {'BS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape,), dtype=np.float64), 
                                   'BO': spaces.Box(low=0, high=np.inf, shape=(self.order_obs_shape,), dtype=np.float64),
                                   'SS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape+1,), dtype=np.float64),
                                   'SO': spaces.Box(low=0, high=np.inf, shape=(self.order_obs_shape,), dtype=np.float64)
                                  }
    
    def observe(self, agent):
        
        signal_observations = self.signal_observations if self.mode == 'train' else self.signal_observations_test
        order_observations = self.order_observations if self.mode == 'train' else self.order_observations_test
        
        if agent == 'BS':
            return signal_observations[0, self.delta]
        elif (agent == 'BO') or (agent == 'SO'):
            return order_observations[0, self.delta]
        elif agent == 'SS':
            matrix = signal_observations[0, self.delta]
            profit = (self.close.iloc[self.delta] - self.buy_price) / self.buy_price
            return np.append(matrix, profit)
        
        
class MDQNOne(OneEnv):
    
    def __init__(self):
        super().__init__()
        super().MDQN_data()
        self.observation_spaces = {'BS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape,), dtype=np.float64)}
    
    def observe(self, agent):
        data = self.scaled_data if self.mode == 'train' else self.scaled_test_data
        return np.array(data['Close'].iloc[self.delta-self.price_history+1:self.delta+1]).flatten()
    
    
class MDQNTwo(TwoEnv):
    
    def __init__(self):
        super().__init__()
        super().MDQN_data()
        self.observation_spaces = {'BS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape,), dtype=np.float64), 
                                   'SS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape+1,), dtype=np.float64)
                                  }
        
    def observe(self, agent):
        data = self.scaled_data if self.mode == 'train' else self.scaled_test_data
        prices = np.array(data['Close'].iloc[self.delta-self.price_history+1:self.delta+1]).flatten()
        if agent == 'BS':
            return prices
        elif agent == 'SS':
            profit = (self.close.iloc[self.delta] - self.buy_price) / self.buy_price
            return np.hstack((prices, profit))
        

class MDQNFour(FourEnv):
    
    def __init__(self):
        super().__init__()
        super().MDQN_data()
        self.observation_spaces = {'BS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape,), dtype=np.float64), 
                                   'BO': spaces.Box(low=0, high=np.inf, shape=(self.order_obs_shape,), dtype=np.float64),
                                   'SS': spaces.Box(low=0, high=np.inf, shape=(self.signal_obs_shape+1,), dtype=np.float64),
                                   'SO': spaces.Box(low=0, high=np.inf, shape=(self.order_obs_shape,), dtype=np.float64),
                                  }
    
    def observe(self, agent):
        data = self.scaled_data if self.mode == 'train' else self.scaled_test_data
        if agent == 'BS':
            return np.array(data['Close'].iloc[self.delta-self.price_history+1:self.delta+1]).flatten()
        elif (agent == 'BO') or (agent == 'SO'):
            a1 = np.array(data[['Open', 'High', 'Low']].iloc[self.delta-4:self.delta+1]).flatten()
            a2 = np.array(data['Close'].iloc[self.delta-19:self.delta+1])
            return np.hstack((a1, a2))
        elif agent == 'SS':
            prices = np.array(data['Close'].iloc[self.delta-self.price_history+1:self.delta+1]).flatten()
            profit = (self.close.iloc[self.delta] - self.buy_price) / self.buy_price
            return np.hstack((prices, profit))