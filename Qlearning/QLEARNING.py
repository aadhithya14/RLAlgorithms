
# coding: utf-8

# In[49]:


import import_ipynb
import custom_environment
import pickle
import numpy as np
import cv2
import random


# In[48]:




NUM_ITERATIONS = 25000
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 1000
DISPLAY_EVERY = 500
SIZE = 10
LEARNING_RATE = 0.1
DISCOUNT = 0.95
total_reward = 0


def get_q_table(start_q_table=None, size=10, action=4):

    if start_q_table is None:
        q_table = np.random.randn(size,size,action)

    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    return q_table


def improve_q_table(env, q, parameters, verbose=True):
    reward = 0
    for episode in range(parameters.NUM_ITERATIONS):
        state, reward, done = env.startover(newpos=parameters.random_start) 
        epsilon=parameters.epsilon
        while not done:
            current_q = q[state[0],state[1],:]
            if random.random() > epsilon:
                action = np.argmax(current_q)
            else:
                action = env.sample_action()

            next_state ,(next_reward, done)= env.step(action)
            reward += next_reward
            future_q = q[next_state[0],next_state[1],:]
            q[state[0],state[1],action] = (1 - parameters.alpha) * current_q[action] + parameters.alpha * ( next_reward + parameters.gamma * max(future_q) - current_q[action])
            if done and next_reward == 100:
                q[state[0],state[1] :] = 0

   
            if episode%parameters.render_every == 0:
                env.render(100)

            state = next_state

        parameters.decay_epsilon()
        cv2.destroyAllWindows()

        if episode%parameters.print_every == 0 and print :
            print('Episode: ',episode,'state:',state,'| Total Average Reward:', total_reward/500,'| Epsilon:', epsilon)
            total_reward= 0

    return q


class Parameters():

    def __init__(self,NUM_ITERATIONS=25000,random_start=True,epsilon=1, EPS_DECAY=.998, RENDER_EVERY=1000, PRINT_EVERY=500, LEARNING_RATE=.1,DISCOUNT=.95):
        self.NUM_ITERATIONS =NUM_ITERATIONS
        self.random_start = random_start
        self.epsilon = epsilon
        self.EPS_DECAY = EPS_DECAY
        self.gamma = DISCOUNT
        self.alpha = LEARNING_RATE
        self.render_every = SHOW_EVERY
        self.print_every = DISPLAY_EVERY
    def decay_epsilon(self):
            self.epsilon*=self.EPS_DECAY
    


# In[ ]:




