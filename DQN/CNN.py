
# coding: utf-8

# In[3]:


import torch.nn.functional as F
import gym
import cv2
import torch 
import torch.nn.functional as F
import torch.nn as nn
import math ,random
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[5]:


env=gym.make('CartPole-v0')


# In[2]:


import sys
print(sys.path)
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


# In[6]:


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen():
    
    screen = env.render(mode='rgb_array')
    
    screen_height, screen_width,_ = screen.shape
    screen = screen[int(screen_height*0.4):int(screen_height * 0.8),:]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    
    
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[ :, slice_range,:]
    
    width = int(screen.shape[1] * 30 / 100)
    height = int(screen.shape[0] * 30 / 100)
    dim = (width, height)
    # resize image
    state = cv2.resize(screen, dim, interpolation = cv2.INTER_AREA) 
    screen=state.transpose((2, 0, 1))
    
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen=np.expand_dims(screen,0)
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    
    return screen


# In[7]:


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        self.device='cuda' if T.cuda.is_available() else 'cpu'
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x=x.to(self.device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            
            q_value = self.forward(state)
            action  = torch.argmax(q_value).item()
        else:
            action = random.randrange(env.action_space.n)
        return action


# In[8]:


from collections import deque
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


# In[9]:


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 3000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


# In[10]:


device= 'cuda' if torch.cuda.is_available() else 'cpu'
model=CnnDQN([3, 48, 108],2)
model=model.to(device)
model_target=CnnDQN([3, 48, 108],2)
model_target=model_target.to(device)
model_target.load_state_dict(model.state_dict())
model_target.eval()
optimizer = torch.optim.Adam(model.parameters())
replay_buffer = ReplayBuffer(10000)
crit=nn.MSELoss()
gamma=0.999


# In[11]:


def compute_loss(batch_size):
    state,action,reward,next_state,done=replay_buffer.sample(batch_size)
    state=torch.FloatTensor(state)
    action=torch.tensor(action).to(device)
    reward=torch.tensor(reward).to(device)
    next_state=torch.FloatTensor(next_state)
    done=torch.tensor(done).float().to(device)

    q_values      = model(state)
    next_q_values = model_target(next_state).detach()

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = crit(q_value,expected_q_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


# In[12]:


def plot(episode, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (episode, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


# In[14]:


num_frames = 2000
batch_size = 128


losses = []
all_rewards = []
es=0
for episode in range(1, num_frames + 1):
    done =0
    env.reset()
    last_screen=get_screen()
    current_screen=get_screen()
    state=current_screen-last_screen
    episode_reward = 0
    while not done:
        epsilon = epsilon_by_frame(es)
        es+=1
        action = model.act(state, epsilon)

        _, reward, done, _ = env.step(action)
        last_screen=current_screen
        current_screen=get_screen()
        next_state=current_screen-last_screen

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        if len(replay_buffer) > batch_size:
            loss = compute_loss(batch_size)
            losses.append(loss.item())
            
    all_rewards.append(episode_reward)
    
    if episode % 10==0:
        model_target.load_state_dict(model.state_dict())
        
    plot(episode, all_rewards, losses)

