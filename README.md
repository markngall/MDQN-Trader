# MDQN-Trader
MDQN-Trader is a fully-automated trading system consisting of cooperative deep reinforcement learning (DRL) agents. 

I developed the MDQN-Trader for my dissertation (MSc Mathematical Finance, University of Warwick). The class structure is designed to be interacted with at a high-level. The user first instantiates the environment object and passes it as an argument into the 'TradingSystem' class' constructor. One must also specify the desired algorithm – 'QL' or 'DQN'. Training and testing the system can then be completed using the corresponding methods. Only four lines of code are needed. For example:
```python
env = MDQNFour()
system = TradingSystem(env, "DQN")
system.train(500000)
returns = system.test()
```

## Abstract
Multi-agent reinforcement learning (MARL) is the problem of multiple autonomous agents learning to make optimal decisions in a common environment. Our aim was to investigate the performance of MARL-based trading systems. To do so, we reproduced the MQ-Trader, a system developed by Lee et al. that uses a 'team' of four Q-learning (QL) agents. Each agent specialises in a distinct sub-problem, such as signal generation or trade execution, and can have a state representation better suited to its goal. The MQ-Trader was then extended to include the double deep Q-network (DDQN) algorithm, state-of-the-art deep learning (DL) techniques (e.g. batch normalisation) and a dueling architecture. Both the MQ-Trader and the improved MDQN-Trader were used to trade SPY at a daily frequency over a five-year test period (04/01/2016–31/12/2020). During testing, the systems remained neutral due to poor execution. There was a flaw in their centralised designs: the agents only gained experience if they performed well. This result demonstrated important drawbacks of MARL. For example, each agent interacts with a non-stationary environment that contains other agents. Other studies demonstrate its benefits, such as scalability and the ability to transfer knowledge between agents. To compare multi-agent and single-agent systems, we also tested one and two-agent variants. Despite strong training performance, they failed to beat SPY. This was likely due to overfitting, which is common when applying neural networks to financial datasets. The DL techniques were beneficial. For example, gradient clipping helped prevent 'exploding gradients' and the Adam optimiser improved convergence stability. However, due to time and computational constraints, it could not be concluded whether the use of DDQN and a dueling architecture led to policy improvement. It is clear that MARL-based trading systems have high potential, but only if care is taken to design the training procedure, tackle overfitting and reduce the computational cost.

## Implementation Details
Six trading systems were created. The MQ-Trader is a reproduction of the system created by Lee et al. It consists of four cooperative QL agents: the buy and sell signal agents, which generate signals from pricing trends, and the buy and sell order agents, which execute optimally. 1Q-Trader and 2Q-Trader are one and two-agent variants that were proposed by Lee et al. The 1DQN-Trader, 2DQN-Trader and MDQN-Trader are improved versions of these systems that incorporate recent developments in the fields of DL and DRL. 

The most important improvements were:
- QL was replaced with DDQN and a dueling architecture.
- Raw pricing data was used because deep neural networks can automatically extract informative features.
- The networks' gradients were clipped to [-1, 1], thereby preventing 'exploding gradients'.
- Batch normalisation was used to standardise the inputs to each neural network layer. This ensures that the distribution of inputs to each layer stays constant throughout training, which allows larger learning rates and avoids the need for careful parameter initialisation.
- Glorot initialisation was used to speed up convergence. This method selects weights so that the variance of the activations is constant across layers.
- The Adam optimiser was used. Adam maintains a distinct learning rate for each parameter and updates each using the gradient’s first and second moments. Benefits include greater computational efficiency and a straightforward implementation. 
- The learning rate was decreased from 0.3 to 0.0001 to strike a better balance between convergence speed and stability.
- Epsilon (which controls the exploration-exploitation trade-off) decays exponentially because exploration is more important the less information an agent has about its environment.
- Momentum was added to accelerate training. It does so by adding inertia; if recent updates have been in the same direction, updates will continue in that direction.

### State Representation
Lee et al. used a compact state representation for the signal agents – the turning point (TP) matrix. A TP is defined as a local extremum of the five-day moving average of the closing price. TPs convey the positions of support and resistance lines. The TP matrix compactly summarises when these TPs occur and how the prices at these points relate to the asset’s current price. For the order agents, Lee et al. used 12 technical indicators.

In the DDQN systems, raw pricing data was used because deep neural networks can automatically extract informative features. The signal agents observe 252 trading days (i.e. a year) of closing prices. The order agents observe 5 trading days of OHLC prices and an additional 15 days of closing prices.

### Data
The dataset was scraped from Yahoo Finance (using yfinance) and contains split and dividend-adjusted OHLC data spanning 27 years, or 6799 trading days, from 03/01/1994 to 31/12/2020. The first 22 years (03/01/1994-31/12/2015) was used for training and the remainder (04/01/2016-31/12/2021) was used for testing.

### Environment
We created environments using the PettingZoo API, which was introduced by Terry et al. in 2020. It is based on the Agent-Environment Cycle (AEC), which is a sequential version of the stochastic game. One by one, the agents observe their environment, take an action and receive a reward, which can be attributed both to its previous action and the actions of other agents. The model is highly flexible – stepping can occur in any order, the common environment can be inserted between any two agents and agents can be added/removed at any time. 

Custom PettingZoo environments inherit from the AECEnv base class and implement several key methods. 'reset()' resets the environment at the start of each episode and 'observe(agent)' provides an agent with its current observation. 'step(action)' receives an agent’s action, steps forward the environment and selects the next action to act. The environment can be interacted with as follows:
```python
for i_episode in range(num_episodes):
  env.reset()
  for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    action = policy(observation, agent) env.step(action)
```
The environment’s current agent is obtained using 'agent iter()', which returns an iterator. Finally, 'last()' returns an observation and a reward for the current agent. In doing so, it calls 'observe(agent)', which is described above. Note that the reward is the cumulative reward since the agent last acted.

Each of the six systems required a different PettingZoo environment class. We used inheritance to reduce repetition. The diagram below shows the inheritance structure.

![inheritance](https://user-images.githubusercontent.com/69372349/163196885-2a36f755-7a75-42a8-9cd3-5ccdf90f1882.png)

The base class, 'TradingEnv', implements common functionality, such as data acquisition, data cleaning and resetting the environment at the start of each episode. We built the data acquisition and preprocessing steps into the environments’ constructors. Thus, the implementation was end-to-end, simplifying its use and improving its portability between versions. The 'OneEnv', 'TwoEnv' and 'FourEnv' classes inherit from this base class and each implement a unique 'step(action)' method that controls the dynamics of the agent-environment interaction. For each of these three classes, there are two derived classes that implement the data preprocessing steps required for the QL and DDQN state respresentations. The derived classes also implement the 'observe(agent)' method, which is different for all six environments. In practice, there is no need to instantiate the parent classes.

### Data Acquisition, Cleaning and Transformation
The yfinance module was used to scrape historical split and dividend-adjusted price data from Yahoo Finance. The data was returned as a Pandas DataFrame with OHLC column names. The data was then cleaned. In particular, forward filling was used to replace up to ten consecutive missing values. To improve the code’s robustness and maintain data integrity, exceptions were used to flag remaining missing values or invalid values (e.g. negative prices).
In the QL environments, the data was then transformed into a 2D NumPy array. NumPy provides support for multi-dimensional arrays. It is vectorised, and thus can perform mathematical functions on multi-dimensional arrays in a highly efficient manner. The array contained each day’s state representation (i.e. flattened TP matrices). Thus, the entire set of possible observations was processed prior to training. The environment could simply index this array, improving the efficiency of training.

In the DDQN environments, the raw prices were normalised using the ‘StandardScaler’ class provided by scikit-learn. This ensured that each field (e.g. closing price) had a mean of 0 and a standard deviation of 1. This is a common preprocessing step because a neural network's input should have values close to zero. Importantly, the 'StandardScaler' object was fit using only the training data, thereby preventing data leakage.

### TradingSystem
The 'TradingSystem' class manages the interaction between the agents and the environment. It instantiates the agents ('QLearning' or 'DQN' objects) and consists of two methods, 'train(num episodes)' and 'test()'. The 'QLearning'and 'DQN' classes each have two methods, 'update(action, obs,
reward, done, abort update)', which updates the agent’s Q-values, and 'policy(obs)', which outputs epsilon-greedy actions.

The class structure is designed to be interacted with at a high-level. The user first instantiates the environment object and passes it as an argument into the 'TradingSystem' class' constructor. One must also specify the desired algorithm – 'QL' or 'DQN'. Training and testing the system can then be completed using the corresponding methods. Only four lines of code are needed. For example:
```python
env = MDQNFour()
system = TradingSystem(env, "DQN")
system.train(500000)
returns = system.test()
```
Optional keyword arguments ('kwargs') in the 'TradingSystem' constructor allow the system’s hyperparameters and neural network configuration to be altered. Note that the 'TradingSystem' class creates a variable number of agents depending on the environment it is given.

To allow the training process to be broken up into shorter chunks, saving and loading functionality was added to the 'TradingSystem' class. This included network weights, replay memories and recorded results (e.g. average rewards). The optional 'load model' kwarg is a Boolean that controls this behaviour. The MQ-Trader environments also have a 'load' kwarg that determines whether or not the environment creates the TP matrices from scratch or loads them from the 'data' directory. This reduces the set-up time when resuming training.

The method 'train(num episode)' controls the interaction between the agents and the environment with a loop similar to that presented in the Environment section. The 'test()' method runs the environment in test mode rather than training mode. To do so, it switches the environment’s 'mode' attribute from 'train' to 'test'.

### References

Jae Won Lee, Jonghun Park, Jangmin O, Jongwoo Lee, and Euyseok Hong. A Multiagent Approach to Q-Learning for Daily Stock Trading. IEEE Transactions on Systems, Man, and Cybernetics - Part A: Systems and Humans, 37(6):864–877, 2007. doi: 10.1109/tsmca.2007.904825.
