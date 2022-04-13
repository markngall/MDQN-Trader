from tradingenv import *

class OneEnv(TradingEnv):
    
    def __init__(self):
        super().__init__()
        self.holding_period = 10
        
        self.possible_agents = ['BS']
        self.action_spaces = {'BS': spaces.Discrete(2)}
        
    def reset(self):
        super().reset()
        if self.mode == 'train':
            self.delta = self.rng.integers(self.price_history, self.num_training_days-self.holding_period)

    def step(self, action):
        
        # Performs step() for done agents
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        
        agent = self.agent_selection
        
        # The agent's cumulative reward has just been accounted for by last()
        self._cumulative_rewards[agent] = 0

        if action == 0:  # Do not buy
            if self.mode == 'test': 
                self.delta += 1
        elif action == 1:  # Buy
            self.buy_price = self.close.iloc[self.delta] * (1+self.TC)
            self.sell_price = self.close.iloc[self.delta+self.holding_period] * (1-self.TC)
            self.rewards['BS'] = (self.sell_price - self.buy_price) / self.buy_price
            if self.mode == 'test':
                self.delta += self.holding_period + 1
        
        self.dones = {agent: True for agent in self.agents}
                
        self._accumulate_rewards()