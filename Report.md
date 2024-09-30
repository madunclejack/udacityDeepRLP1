# Project 1 Navigation using Double Deep Q-Network Implementation
The goal of this project was to design a Reinforcement Learning (RL) based agent that could navigate a closed space to collect yellow bananas while avoiding blue bananas. The environment is based on the Unity-ML Agents Banana Collector Environment. The agent for this project was originally a Deep Q-Network (DQN) model. However, the learning rate was unsatisfactory with DQN, so the agent was modified into a Double Deep Q-Network (DDQN) model which greatly reduced training time. Both models are discussed in Section 1. Section 2 discussed the hyperparameters available, and which values were chosen. Section 3 shows the resutls of training, and compares the DQN to the DDQN model. Section 4 contains the conclusion and recommendation for future work.

# I. Agent Architecture
The original (DQN) model was revolutionary because it was able to train and play Atari games with exceptional skill, as discussed in the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602). 
talk about DQN architecture, math equations, etc.
The problem is to find the optimal action-value function, from which one can define the policy that maximizes the return for each state. From [1], it is defined as $`q^*`$ for a state action pair $(s,a)$ provided by the optimal policy $\pi$ such that(pg 258 and pg 260):  
  
$`$q^*(s,a) = \text{max}_{\pi} \mathbb{E}_{\pi} \left[G_t | S_t = s, A_t = a\right]\,\,  \forall s \in S, \forall a \in A(s)$`$  
  
That is, the optimal policy $\pi$ will maximize the expected return $G_t$ of $`q^*`$ over the state space $S$ and action space $A$. $G_t$ is the discounted return, defined as:  
  
$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-1} R_T $  
  
The $R$ terms are the return from each time step, from t+1 to T. Future returns are weighted by $\gamma$, the discount factor because future returns as less useful than returns in the current state for making decisions. Additionally, future returns have higher variance because they are known with less certainty than the return the agent recently received. We can estimate the optimal policy using nonlinear function approximation AKA a neural network.  

There are several ways to estimate $`q^*`$ which is the "target" for Q-Learning. This project will use the Temporal Difference (TD) Target because it is commonly used and often has success. As the agent steps throught the environment, it gets rewards for taking actions. The TD method combines the reward from the next state action pair with the discounted future reward. There is a temporal difference between the current reward and the future rewards. Over time, as the Q-function is updated, the estimated future reward should converge to the actual reward value. The TD method uses an off-policy target that approximates a greedy policy because it always chooses the maximizing action. Off-policy indicates that the agent might not always choose the greedy action depending on the exploration/exploitation parameter $\epsilon$ discussed in Section II. From [1], the TD-Target is formulated as:  
  
$`$y_i = R_{t+1} + \gamma \text{max}_{a} \left[Q(S_{t+1}, a; \theta_i)\right]$`$  
  
where $`\theta_i`$ are the weights of the neural network used to parameterize Q for each layer i, and $S_{t+1}$ is the next state, and $a$ is the action that maximizes the return in the next state. 
Thus we can estimate the unknown function $`q^*`$ with the Q-Learning TD target:  
  
$`$q^*(s,a) \approx \text{max}_{\pi} \mathbb{E}_{\pi} \left[r + \gamma \text{ max}_{a^{'}}\,Q^*(s^{'}, a^{'})\right]$`$  
  
The $s'$ and $a'$ indicate they are the state and action from the next time step respetively.  

The network is updated gradually to minimize the expected error between the true optimal action-value function $`$q^*$`$  and the  neural network estimated action-value function Q. The error is calcluated using a Mean Squared Error loss function $L_i(\theta_i)$:  
  
$L_i(\theta_i) = \mathbb{E}_{s, a} [(q^* (s,a) - Q(s,a; \theta_i))^2]$  

Substitue the TD Target for $`q^*`$ into the loss function:  
  
$`$L_i(\theta_i) = \mathbb{E}_{s, a}[(r + \gamma \text{ max}_{a^{'}} \,Q^*(s^{'}, a^{'}) - Q(s,a; \theta_i))^2]$`$  
  
and take the gradient of $L$ with respect to the weights $\theta_i$ to obtain the network update function/objective function:  
  
$`$\nabla L(\theta_i) = \mathbb{E}_{s,a,r,s^{'}} \left[(r + \gamma \text{max}_{a^{'}} Q(s^{'}, a^{'}; \theta_i) - Q(s, a; \theta_i)) \nabla_{\theta_i} Q(s, a; \theta_i)  )\right]$`$  
  
The DQN algorithm uses a copy of the network that is updated less frequently as a represenentatio of the Target (the off-line network), and a copy that is updated every step the agent takes which represents the current estimate of the action-value function (the on-line network).
To update the on-line network, the agent takes several steps to create a mini-batch of samples. After a set number of steps, the algorithm computes the error between the off-line TD Target network and the on-line network and updates the on-line network weights to minimize the error. Mini-batches are important because updating after every step would have high varaince, and make it difficult for the network to converge on a solution. The steps are also assumed to be from a stationary,  Independent Identically Distributed (IID)  distribution. The samples cannot be taken from a distribution that is changing, and the samples are assumed to be independent from each other yet taken from an identical distribution. A mini-batch approach helps to re-create that assumption since there are multiple steps taken before the off-line network gets updated and changes the sampled distribution.  


then talk about what things I added to original DQN to make this one (replay buffer, fixed q-targets)

pg 52 seems to say s' means next state. whatabout a'?

The 
The agent is a (DDQN) implementation, which is an improvement on the original DQN model
# II. Improvements to  DQN
# III. Hyperparameters
# IV. Training Results
# V. Conclusion and Future Work
