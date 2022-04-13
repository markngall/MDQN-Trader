from tradingenv import *

class TwoEnv(TradingEnv):
    
    def __init__(self):
        super().__init__()
        
        self.possible_agents = ['BS', 'SS']
        self.action_spaces = {'BS': spaces.Discrete(2), 
                              'SS': spaces.Discrete(2)
                             }

    def reset(self):
        super().reset()
        if self.mode == 'train':
            self.delta = self.rng.integers(self.price_history, self.num_training_days-self.max_holding_period-1)
            
    def step(self, action):
        
        # Performs step() for done agents
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        
        agent = self.agent_selection
        
        # The agent's cumulative reward has just been accounted for by last()
        self._cumulative_rewards[agent] = 0

        if agent == 'BS':
            if action == 0:  # Do not buy
                self.dones = {agent: True for agent in self.agents}
                if self.mode == 'test': 
                    self.delta += 1
            elif action == 1:  # Buy
                self.buy_price = self.close.iloc[self.delta] * (1+self.TC)
                self.days_held = 0
                self.delta += 1
                self.add_agent('SS')
                self.agent_selection = 'SS'
                
        elif agent == 'SS':
            if (action == 0) or (self.days_held == self.max_holding_period):  # Sell
                self.sell_price = self.close.iloc[self.delta] * (1-self.TC)
                ret = (self.sell_price - self.buy_price) / self.buy_price
                self.rewards = {'BS': ret, 'SS': 0}
                self.dones = {agent: True for agent in self.agents}
                if self.mode == 'test': 
                    self.delta += 1
            elif action == 1:  # Hold
                self.delta += 1
                self.days_held += 1
                q = (self.close.iloc[self.delta] - self.close.iloc[self.delta-1]) / self.close.iloc[self.delta-1]
                self.rewards = {'BS': 0, 'SS': q}
                
        self._accumulate_rewards()    