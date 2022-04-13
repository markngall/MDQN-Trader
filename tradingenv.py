import pandas as pd
import numpy as np
from gym import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from IPython import display


def get_turning_points(data):
    data = data.rolling(5).mean()
    data['First Order FD'] = data['Close'].shift(-1) - data['Close']
    data['First Order BD'] = data['Close'] - data['Close'].shift(1)
    data['Second Order'] = data['Close'].shift(-1) - 2*data['Close'] + data['Close'].shift(1)
    
    up_TP = data.copy()
    down_TP = data.copy()
    
    mask1 = data['Second Order'] > 0
    mask2 = data.shift(-1)['First Order BD'] > 0
    mask3 = data.shift(-2)['First Order BD'] > 0
    mask4 = data.shift(-3)['First Order BD'] > 0
    mask5 = data.shift(1)['First Order FD'] < 0
    mask6 = data.shift(2)['First Order FD'] < 0
    mask7 = data.shift(3)['First Order FD'] < 0
    
    up_TP.loc[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7] = 1
    up_TP.loc[(~mask1 | ~mask2 | ~mask3 | ~mask4 | ~mask5 | ~mask6 | ~mask7)] = 0
    
    mask1 = data['Second Order'] < 0
    mask2 = data.shift(-1)['First Order BD'] < 0
    mask3 = data.shift(-2)['First Order BD'] < 0
    mask4 = data.shift(-3)['First Order BD'] < 0
    mask5 = data.shift(1)['First Order FD'] > 0
    mask6 = data.shift(2)['First Order FD'] > 0
    mask7 = data.shift(3)['First Order FD'] > 0
    
    down_TP.loc[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7] = 1
    down_TP.loc[(~mask1 | ~mask2 | ~mask3 | ~mask4 | ~mask5 | ~mask6 | ~mask7)] = 0
    
    return up_TP, down_TP


def get_matrix(up_TP, down_TP, day, close):
    closing_price = close.iloc[day]
    
    obs = []
    fib = [0, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    for i in range(1, len(fib)):
        start = day - sum(fib[:i+1])
        end = day - sum(fib[:i])
        if up_TP.iloc[start:end].values.any() == 1:

            # Location of turning points
            indices = start + np.argwhere(np.array(up_TP.iloc[start:end]['Close']))

            ratios = []
            for index in indices:
                ratios.append(100 * (close.iloc[index[0]] - closing_price) / closing_price)
            for j in range(1, len(fib)):
                lower_limit = sum(fib[:j])
                upper_limit = 10**4 if j == len(fib)-1 else sum(fib[:j+1])
                present = False
                for ratio in ratios:
                    if ratio >= lower_limit and ratio < upper_limit:
                        present = True
                if present:
                    obs.append(1)
                else:
                    obs.append(0)
                present = False
                for ratio in ratios:
                    if ratio < -lower_limit and ratio >= -upper_limit:
                        present = True
                if present:
                    obs.append(1)
                else:
                    obs.append(0)
                    
        else:
            obs.extend([0]*18)
          
        if down_TP.iloc[start:end].values.any() == 1:

            # Location of turning points
            indices = start + np.argwhere(np.array(down_TP.iloc[start:end]['Close']))

            ratios = []
            for index in indices:
                ratios.append(100 * (close.iloc[index[0]] - closing_price) / closing_price)
            for j in range(1, len(fib)):
                lower_limit = sum(fib[:j])
                upper_limit = 10**4 if j == len(fib)-1 else sum(fib[:j+1])
                present = False
                for ratio in ratios:
                    if ratio >= lower_limit and ratio < upper_limit:
                        present = True
                if present:
                    obs.append(1)
                else:
                    obs.append(0)
                present = False
                for ratio in ratios:
                    if ratio < -lower_limit and ratio >= -upper_limit:
                        present = True
                if present:
                    obs.append(1)
                else:
                    obs.append(0)
        
        else:
            obs.extend([0]*18)

    return np.array(obs)


class TradingEnv(AECEnv):

    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(1)
        self.max_holding_period = 100
        self.mode = 'train'
        self.TC = 0.0045

        self.training_period = [pd.to_datetime('1994-01-01', format='%Y-%m-%d'), pd.to_datetime('2015-12-31', format='%Y-%m-%d')]
        self.test_period = [pd.to_datetime('2016-01-01', format='%Y-%m-%d'), pd.to_datetime('2021-01-01', format='%Y-%m-%d')]
        
        self.data = yf.Ticker('SPY').history(start=self.training_period[0], end=self.test_period[1]+pd.Timedelta('4D'), auto_adjust=True)
        self.data = self.data.loc[self.training_period[0]:self.test_period[1]]
        self.data = self.data[['Open', 'High', 'Low', 'Close']]

        self.num_days = len(self.data)
        self.num_training_days = len(self.data.loc[self.training_period[0]:self.training_period[1]])
        
        # Needed for testing
        self.test_dates = self.data.loc[self.test_period[0]:self.test_period[1]].index
        self.num_test_days = len(self.test_dates)
        
        # Replace missing values (forward fill up to 10 consecutive null values)
        self.data.fillna(method='ffill', inplace=True, limit=10)

        self.data.reset_index(drop=True, inplace=True)
        
        # If there are null values remaining, raise exception
        if self.data.isnull().values.any():
            raise ValueError('Dataset contains missing values')
        
        # If there are negative values, raise exception
        if (self.data.values < 0).any():
            raise ValueError('Dataset contains negative values')
            
    def MQ_data(self):
        self.price_history = 230
        
        # Functionality to import the observations
        if self.load:
            print('Loading environment...')
            a = np.loadtxt('data/signal_observations.txt')
            b = np.loadtxt('data/order_observations.txt')
            signal_observations = a.reshape(a.shape[0], a.shape[1] // 324, 324)
            order_observations = b.reshape(b.shape[0], b.shape[1] // 12, 12)
        else:
            print('Setting up environment...')

            # Obtains TP matrices for signal agents
            signal_observations = np.zeros((1, self.num_days, 324))  # 3D array, first dim is asset, second is day, third is indicator
            up_TP, down_TP = get_turning_points(self.data)
            for day in range(self.num_days):
                signal_observations[0, day] = get_matrix(up_TP, down_TP, day, self.data['Close'])

            # Obtains state representation for order agents
            order_observations = np.zeros((1, self.num_days, 12)) 
            close = self.data['Close']
            ma20 = close.rolling(20).mean()
            ma10 = close.rolling(10).mean()
            ma5 = close.rolling(5).mean()
            g20 = (ma20 - ma20.shift(1)) / ma20.shift(1)
            g10 = (ma10 - ma10.shift(1)) / ma10.shift(1)
            g5 = (ma5 - ma5.shift(1)) / ma5.shift(1)
            d20 = (close - ma20) / ma20
            d10 = (close - ma10) / ma10
            d5 = (close - ma5) / ma5
            u = (self.data['High'] - self.data['Open'].combine(close, np.maximum)) / self.data['Open'].combine(close, np.maximum)
            l = (self.data['Open'].combine(close, np.minimum) - self.data['Low']) / self.data['Open'].combine(close, np.minimum)
            b1 = (close - self.data['Open']) / self.data['Open']
            q1 = (close - close.shift(1)) / close.shift(1)
            b2 = (close.shift(1) - self.data['Open'].shift(1)) / self.data['Open'].shift(1)
            q2 = (close.shift(1) - close.shift(2)) / close.shift(2)
            order_observations[0] = np.array([g20, g10, g5, d20, d10, d5, u, l, b1, q1, b2, q2]).transpose()

            # Save the NumPy arrays
            a = signal_observations.reshape(signal_observations.shape[0], -1)
            b = order_observations.reshape(order_observations.shape[0], -1)
            np.savetxt('data/signal_observations.txt', a)
            np.savetxt('data/order_observations.txt', b)
            
        display.clear_output(wait=False)
            
        # Training data
        self.signal_observations = signal_observations[:, :self.num_training_days, :]
        self.order_observations = order_observations[:, :self.num_training_days, :]

        # Test data
        self.signal_observations_test = signal_observations[:, self.num_training_days-self.price_history:, :]
        self.order_observations_test = order_observations[:, self.num_training_days-self.price_history:, :]
        
        self.test_data = self.data.iloc[self.num_training_days-self.price_history:]
        self.data = self.data.iloc[:self.num_training_days]
        self.close = self.data['Close']
        
        self.signal_obs_shape = 324
        self.order_obs_shape = 12
        
    def MDQN_data(self):
        self.price_history = 252
            
        self.test_data = self.data.iloc[self.num_training_days-self.price_history:]
        self.data = self.data.iloc[:self.num_training_days]
        self.close = self.data['Close']
        
        # Normalisation
        scaler = StandardScaler()
        scaler.fit(self.data)  # Only fit scaler to training data
        self.scaled_data = pd.DataFrame(data=scaler.transform(self.data), columns=self.data.columns)
        self.scaled_test_data = pd.DataFrame(data=scaler.transform(self.test_data), columns=self.test_data.columns)
        
        self.signal_obs_shape = self.price_history
        self.order_obs_shape = 3 * 5 + 20
            
    def add_agent(self, agent):
        self.agents.append(agent)
        self.rewards[agent] = 0
        self._cumulative_rewards[agent] = 0
        self.dones[agent] = False
        self.infos[agent] = {'Abort Update': False}
        
    def reset(self):
        self.agents = ['BS']
        self.agent_selection = self.agents[0]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {'Abort Update': False} for agent in self.agents}
        self.buy_price = None

    def set_mode(self, mode):
        self.mode = mode