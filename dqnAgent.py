
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

from networkModel import agentQNetwork

# Global Parameters
""" Alpha, or Learning Rate """
LEARN_RATE = 5e-4
""" Discount factor for rewards """
GAMMA = 0.99
""" Size of Replay Buffer """
BUFFER_SIZE = int(1e5)
""" Size of the batchs to fetch from the Replay Buffer """
BATCH_SIZE = 32
""" Parameter to control how many steps between updating the Neural Net """
LEARN_EVERY = 5
""" Parameter to control how much to update the target network with the local network"""
""" Bigger means it updates the target network with more of the local network """
TAU = 0.25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''
    Structure that contains the mechanics for sampling and learning from the environment
    
    Methods
    step: take the next step in the environment after choosing an action
    act: take an action based on the current state
    learn: apply the rewards gained to the Neural Net
    '''
    
    def __init__(self, state_size, action_size, seed):
        
        # Size of the state space
        self.stateSize = state_size
        
        # Size of the action space
        self.actionSize = action_size
        
        # PRNG seed
        self.seed = seed
        random.seed(seed)
        
        # Varaible to keep track of steps in between each NN learning Update
        self.stepNum = 0
        
        # Establish the Target and the Local Q-Networks for Fixed Q-Targets
        # use .to(device) to specify the data type of the torch Tensor (for CPU or GPU)
        self.QNet_Target = agentQNetwork(self.stateSize, self.actionSize, self.seed).to(device)
        self.QNet_Local = agentQNetwork(self.stateSize, self.actionSize, self.seed).to(device)
        
        # Set up the Optimizer for training the networks. Can set learning rate (i.e. gradient step size) 
        # and apply momentum if desired
        self.optim = optim.Adam(self.QNet_Local.parameters(), lr=LEARN_RATE)
        
        # Establish the Replay Buffer
        self.replayMem = ReplayBuffer(self.actionSize, seed)
    
    def step(self, state, action, reward, next_state, done):
        
        # Add step to the Replay Buffer
        self.replayMem.addMem(state, action, reward, next_state, done)
        
        # After LEARN_EVERY number of steps, update the NN
        self.stepNum = (self.stepNum + 1) % LEARN_EVERY
        
        if self.stepNum == 0:
            # Perform learning Action if there are enough samples to make up a batch (set by BATCH_SIZE)
            if len(self.replayMem) > BATCH_SIZE:
                experienceSample = self.replayMem.memSample()
                self.learn(experienceSample)
    
    def act(self, state, epsilon):
        '''
        Returns the best guess of action values based on the current state of the NN
        '''
        # Convert state to PyTorch tensor, make sure all values are a float
        # .unsqueeze will reformat to dimensions with 1 at the specified dim (0 - row, 1 - column)
        # .to(device) converts the tensor to the proper type for the device (CPU or GPU)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Turn off Dropout Layers, Batch layers, etc. so we can run the network straight-through/without
        # training mechanisms applied. Called eval mode
        self.QNet_Local.eval()
        
        with torch.no_grad():
            actionVals = self.QNet_Local(state)
        
        # Turn training mode back on
        self.QNet_Local.train()
        
        # Epsilon-Greedy action selection
        if random.random() > epsilon:
            # Return index of action value with highest score
            # .cpu() returns a copy of the tensor in CPU memory
            # .data is a deprecated method that returns the underlying Tensor from a Variable object
            # in PyTorch. The new way is to use .detach. A Variable is a Tensor object with history tracking
            # for autograd method.
            # .numpy returns as Numpy ndarray, added a "astype" on 7/18/2024
            return np.argmax(actionVals.cpu().data.numpy()).astype(np.int_)
        else:
            # Return a random action choice
            return random.choice(np.arange(self.actionSize)).astype(np.int_)
    
    def learn(self, experiences):
        '''
        Function to update the neural net with a batch update of experience tuples
        Uses Double DQN Approach
        '''
        states, actions, rewards, next_states, dones = experiences
        
        # Get network output for the next state. Note that nn.Module class inherently calls the forward method
        # .max acts on the Tensor, arguemnt is the dim. Returns (value, indices) where value is the max val in each
        #     row and indices are the location of each max val in the row. With the dim argument, max will
        #     peform torch.squeeze along that dim. .squeeze removes all dims with size = 1, or just the specified dim.
        #     that is, with dim = 1, (Ax1xB) -> (AxB)
        #     .unsqueeze will reformat to dimensions with 1 at the specified dim (0 - row, 1 - column)
        
        #Q_Targets_Next = self.QNet_Target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # For DDQN, use the Local network to evalute the action values of the next state
        max_Q_Locals_Next = self.QNet_Local(next_states).max(1)[1]
        # Get Q-Values from the Target network
        Q_Targets_Next = self.QNet_Target(next_states).detach()
        # Get the Q-Values of the target network, indexed by the argmax index from the local network
        Q_Targets_Next = Q_Targets_Next[np.arange(len(dones)), max_Q_Locals_Next].unsqueeze(1)
        
        # Calculate the reward for the current net performance with the current states
        # rewards = R + (Gamma * maximizing action reward * (1-dones)
        # the end part will go to 0 if the max action brought us to the done state,
        # so the reward will just be the reward received.
        Q_Targets = rewards + (GAMMA * Q_Targets_Next * (1 - dones))
        
        # Get the expected Q-values from the local model using the current states
        # .gather(dim, index) selects values from each row (or column) of a Tensor. Row/Column specified by dim.
        #     index indicates which element of the row/column to collect into the output array
        Q_Expected = self.QNet_Local(states).gather(1, actions)
        
        # Calculate the loss
        # Note: nn.NLLLoss expects a torch.LongTensor as the Target type
        # using nn.LLLoss caused an error: multi-target not supported at /pytorch/aten/src/THNN/generic/ClassNLLCriterion.
        #loss = self.QNet_Target.criterion(Q_Expected, Q_Targets.long())
        loss = F.mse_loss(Q_Expected, Q_Targets)
        # Zero out the gradients in the optimizer
        self.optim.zero_grad()
        # Use the loss function to calculate the gradient of the loss Tensor. As a note,
        # a PyTorch Tensor by default will store the information used to compute it, so it has the steps
        # needed stored to perform the backpropagation gradient
        loss.backward()
        # Use results of backprop to update weights in network with the chosen optimizer.
        self.optim.step()
        
        # Update Target Network
        self.softUpdate(self.QNet_Local, self.QNet_Target)


    def softUpdate(self, qNetLocal, qNetTarget):
        """
        Soft-update equation
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Copies the parameters of the local network into the target network
        """
        for targetParams, localParams in zip(qNetTarget.parameters(), qNetLocal.parameters()):
            targetParams.data.copy_(TAU*localParams.data + (1.0-TAU)*targetParams.data)
       
        
class ReplayBuffer():
    '''
    Structure to keep track of previous state, action pairs for Experience Replay
    
    Methods
    memAdd: store a memory in the buffer
    memSample: sample a memory from the buffer
    '''
    
    def __init__(self, action_size, seed):
        self.actionSize = action_size
        self.memBuff = deque(maxlen = BUFFER_SIZE)
        self.batchSize = BATCH_SIZE
        self.experience = namedtuple("Experience", field_names = \
                                        ["state", "action", "reward", "nextState", "done"])
        
    def addMem(self, state, action, reward, nextState, done):
        # Store the data in an experience tuple
        e = self.experience(state, action, reward, nextState, done)
        
        # Add the tuple to the buffer
        self.memBuff.append(e)
    
    def memSample(self):
        # Choose a random sample of memory from the Replay Buffer
        memorySample = random.sample(self.memBuff, k=self.batchSize)
        
        states = torch.from_numpy(np.vstack([e.state for e in memorySample if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in memorySample if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in memorySample if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.nextState for e in memorySample if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in memorySample if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        # Convenience function for returning the length of the buffer
        return len(self.memBuff)
            
            
            
                                                   