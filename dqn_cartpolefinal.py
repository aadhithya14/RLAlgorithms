
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy


# In[2]:


batch_size=128
alpha=0.01
gamma=0.9
epsilon=0.9
MEMORY_CAPACITY=2000
Q_NETWORK_ITERATION=100

env=gym.make("CartPole-v0")
env=env.unwrapped
num_actions=env.action_space.n
num_states=env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


# In[3]:


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1=nn.Linear(num_states,50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2=nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3=nn.Linear(30,num_actions)
        self.fc3.weight.data.normal_(0,0.1)
    
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        action_prob=self.fc3(x)
        return action_prob


# In[4]:


class DQN():
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = net(), net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, num_states * 2 + 2))
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=alpha)
        self.loss_func = nn.MSELoss()     
        
    def choose_action(self,state):
        state=torch.unsqueeze(torch.FloatTensor(state),0)
        if np.random.randn()<=epsilon:
            action_value=self.eval_net.forward(state)
            action=torch.max(action_value,1)[1].data.numpy()
            if ENV_A_SHAPE==0:
                action=action[0]
            else:
                action=action.reshape(ENV_A_SHAPE)
        else:
            action=np.random.randint(0,num_actions)
            if ENV_A_SHAPE==0:
                action=action
            else:
                action=action.reshape(ENV_A_SHAPE)
        return action
    
    def store_transition(self,state,action,reward,next_state):
        transition=np.hstack((state,[action,reward],next_state))
        index=self.memory_counter%MEMORY_CAPACITY
        self.memory[index,:]=transition
        self.memory_counter+=1
        
        
    def learn(self):
        
        
        if self.learn_step_counter%Q_NETWORK_ITERATION==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1
        
        
        sample_index=np.random.choice(MEMORY_CAPACITY,batch_size)
        batch_memory=self.memory[sample_index,:]
        batch_state=torch.FloatTensor(batch_memory[:,:num_states])
        batch_action=torch.LongTensor(batch_memory[:,num_states:num_states+1].astype(int))
        batch_reward=torch.FloatTensor(batch_memory[:,num_states+1:num_states+2])
        batch_next_state=torch.FloatTensor(batch_memory[:,-num_states:])
        
        
        q_val=self.eval_net(batch_state).gather(1,batch_action)
        q_next=self.target_net(batch_next_state).detach()
        q_target=batch_reward+gamma*q_next.max(1)[0].view(batch_size,1)
        loss=self.loss_func(q_val,q_target)
        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
def reward_func(env,x,x_dot,theta,theta_dot):
    r1=(env.x_threshold-abs(x))/(env.x_threshold)-0.5
    r2=(env.theta_threshold_radians-abs(theta))/(env.theta_threshold_radians)-0.5
    reward=r1+r2
    return reward


# In[5]:


dqn=DQN()
env=gym.make("CartPole-v0")
episodes=400
print("collecting experience")
reward_list=[]
sum_reward=0
for i in range(episodes):
    state=env.reset()
    ep_reward=0
    done=False
    while not done:
        env.render()
        action=dqn.choose_action(state)
        next_state, _ , done, info = env.step(action)
        x,x_dot,theta,theta_dot=next_state
        reward=reward_func(env,x,x_dot,theta,theta_dot)
        dqn.store_transition(state,action,reward,next_state)
        ep_reward+=reward
        if dqn.memory_counter>=MEMORY_CAPACITY:
            dqn.learn()
        state = next_state
    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
    r = copy.copy(ep_reward)
    reward_list.append(r)
    sum_reward+=r
    avg_reward=(sum_reward)/200
print(avg_reward)
    


# In[6]:


env=gym.make("CartPole-v0")
done = False
state = env.reset()
tot_reward = 0
while not done:
    action = dqn.choose_action(state)
    next_state, reward, done, _ = env.step(action)
    tot_reward += reward
    state = next_state
    
print(tot_reward)

