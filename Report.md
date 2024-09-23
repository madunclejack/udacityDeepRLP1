# Project 1 Navigation using Double Deep Q-Network Implementation
The goal of this project was to design a Reinforcement Learning (RL) based agent that could navigate a closed space to collect yellow bananas while avoiding blue bananas. The environment is based on the Unity-ML Agents Banana Collector Environment. The agent for this project was originally a Deep Q-Network (DQN) model. However, the learning rate was unsatisfactory with DQN, so the agent was modified into a Double Deep Q-Network (DDQN) model which greatly reduced training time. Both models are discussed in Section 1. Section 2 discussed the hyperparameters available, and which values were chosen. Section 3 shows the resutls of training, and compares the DQN to the DDQN model. Section 4 contains the conclusion and recommendation for future work.

# I. Agent Architecture
The original (DQN) model was revolutionary because it was able to train and play Atari games with exceptional skill, as discussed in the paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602). 
talk about DQN architecture, math equations, etc.
The problem is to find the optimal action-value function, from which one can define the policy that maximizes the return for each state. It is defined as $`q^*`$ for a state action pair, $(s, a)$ such that(pg 258 and pg 260)
$`$q^*(s,a) = \mathbb{E}_{(s,a)} \left[r + \gamma \text{ max}_{a^{'}}\,Q^*(s^{'}, a^{'})\right]$`$  
$`$q^*(s,a) = \text{max}_{\pi} \mathbb{E}_{\pi} \left[r + \gamma \text{ max}_{a^{'}}\,Q^*(s^{'}, a^{'})\right]$`$  
The network is designed to minimize the expected error between the true optimal action-value function $`$q^*$`$ (known as the target) and the  neural network estimated action-value function Q. The loss function $L_i(\theta_i)$ is given as:  
$$L_i(\theta_i) = \mathbb{E}_{s, a} [(q^* (s,a) - Q(s,a; \theta_i))^2]$$  
where $`\theta_i`$ are the weights of the neural network used to parameterize Q for each layer i, and (s, a) are a sampled state-action pair. Substitue the defintion of the target $q^*$ into the loss function, and take the gradient of $L$ with respect to the weights $\theta_i$ to obtain the network update function/objective function:  
$`$\nabla L(\theta_i) = \mathbb{E}_{s,a,r,s^{'}} \left[(r + \gamma \text{max}_{a^{'}} Q(s^{'}, a^{'}; \theta_i) - Q(s, a; \theta_i)) \nabla_{\theta_i} Q(s, a; \theta_i)  )\right]$`$  
then talk about what things I added to original DQN to make this one (replay buffer, fixed q-targets)

pg 52 seems to say s' means next state. whatabout a'?

The 
The agent is a (DDQN) implementation, which is an improvement on the original DQN model

# II. Hyperparameters
# III. Training Results
# IV. Conclusion and Future Work
