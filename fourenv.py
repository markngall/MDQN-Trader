from tradingenv import *

class FourEnv(TradingEnv):
    
    def __init__(self):
        super().__init__()
        self.N = 5
        
        self.buy_betas = [0.88, 0.93, 0.95, 0.99, 1.01, 1.03, 1.07, 1.12]
        self.sell_betas = [0.88, 0.93, 0.95, 0.99, 1.01, 1.03, 1.07, 1.12]
        self.possible_agents = ['BS', 'BO', 'SS', 'SO']
        self.action_spaces = {'BS': spaces.Discrete(2), 
                              'BO': spaces.Discrete(len(self.buy_betas)),
                              'SS': spaces.Discrete(2), 
                              'SO': spaces.Discrete(len(self.sell_betas))
                             }
        
    def reset(self):
        super().reset()
        if self.mode == 'train':
            self.delta = self.rng.integers(self.price_history, self.num_training_days-self.max_holding_period-2)
            
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
                self.add_agent('BO')
                self.agent_selection = 'BO'
                
        elif agent == 'BO':
            self.delta += 1
            MA = np.mean(self.close.iloc[self.delta-self.N:self.delta])
            lowest_price = self.data.iloc[self.delta]['Low']
            if self.buy_betas[-1] * MA >= lowest_price:
                BP = self.buy_betas[action] * MA
                if BP >= lowest_price:  # Can execute at desired price
                    self.buy_price = BP
                    self.days_held = 0
                    self.rewards = {'BS': 0, 'BO': np.exp(-100*(self.buy_price-lowest_price)/lowest_price)}                
                    self.dones = {'BS': False, 'BO': True} 
                    self.add_agent('SS')
                    self.add_agent('SO')
                    self.agent_selection = 'SS'
                else:  # If testing, cannot retry
                    if self.mode == 'test':
                        self.dones = {agent: True for agent in self.agents}
            else:  # Can't make a purchase
                self.infos['BS']['Abort Update'] = True
                self.infos['BO']['Abort Update'] = True
                self.dones = {agent: True for agent in self.agents}
                    
        elif agent == 'SS':
            if (action == 0) or (self.days_held == self.max_holding_period):  # Sell
                q = 0
                self.dones = {'BS': False, 'BO': True, 'SS': True, 'SO': False}
                self.agent_selection = 'SO'
            elif action == 1:  # Hold
                self.delta += 1
                self.days_held += 1
                q = (self.close.iloc[self.delta] - self.close.iloc[self.delta-1]) / self.close.iloc[self.delta-1]
            self.rewards = {'BS': 0, 'BO': 0, 'SS': q, 'SO': 0}
                
        elif agent == 'SO':
            self.delta += 1
            MA = np.mean(self.close.iloc[self.delta-self.N:self.delta])
            highest_price = self.data.iloc[self.delta]['High']
            if (self.sell_betas[0] * MA) > highest_price: 
                self.sell_price = self.close.iloc[self.delta]
                ret = (self.sell_price * (1-self.TC) - self.buy_price * (1+self.TC)) / (self.buy_price * (1+self.TC))
                self.rewards = {'BS': ret, 'BO': 0, 'SS': 0, 'SO': 0}
                self.infos['SO']['Abort Update'] = True
                self.dones = {agent: True for agent in self.agents}
            else:
                SP = self.sell_betas[action] * MA
                if SP <= highest_price: 
                    self.sell_price = SP
                    ret = (self.sell_price * (1-self.TC) - self.buy_price * (1+self.TC)) / (self.buy_price * (1+self.TC))
                    self.rewards = {'BS': ret, 'BO': 0, 'SS': 0, 'SO': np.exp(-100*(highest_price-self.sell_price)/self.sell_price)}
                    self.dones = {agent: True for agent in self.agents}
                else: 
                    self.rewards = {agent: 0 for agent in self.agents}
                    if self.mode == 'test':  # If testing, cannot retry
                        self.sell_price = self.close.iloc[self.delta]  
                        ret = (self.sell_price * (1-self.TC) - self.buy_price * (1+self.TC)) / (self.buy_price * (1+self.TC))
                        self.rewards = {'BS': ret, 'BO': 0, 'SS': 0, 'SO': 0}
                        self.dones = {agent: True for agent in self.agents}
                
        self._accumulate_rewards()