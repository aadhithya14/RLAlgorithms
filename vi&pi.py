
# coding: utf-8

# In[9]:


import numpy as np
import gym
import time
import  import_ipynb
from lake_envs import *
import random

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""
def isover(V,V_new,tol):
    if np.all(np.abs(V - V_new) < tol) :    #np.sum(np.sqrt(np.square(V_new-V))) < tol
        return 1
    return 0

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    V = np.zeros(nS)
    V_new=V.copy()
    i=0
    max_iteration=40
    for i in range(max_iteration):
        V=V_new.copy()
        V_new = np.zeros(nS, dtype=float)
        for state in range(nS):
            for probability, nextstate, reward, terminal in P[state][policy[state]]:
                V_new[state] += probability * (reward + gamma * V[nextstate])
        if isover(V,V_new,tol) :
            break
    return V_new

	    

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    P_new = np.zeros(nS, dtype=int)
    for state in range(nS):
        B=np.zeros(nA,dtype=float)
        q=0
        for action in range(nA):
            for probability, nextstate, reward, terminal in P[state][action]:
                B[action] += probability * (reward + gamma * value_from_policy[nextstate])
            if(B[action]>q):
                q=B[action]
                P_new[state]=action
            elif q == B[action]:
                 if random.random() < 0.5:
                        P_new[state]=action
    return P_new                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        


	############################



def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    V = np.zeros(nS,dtype=float)
    policy = np.zeros(nS, dtype=int)
    max_iteration=40
    for s in range(nS):
        policy[s]=s%nA
    for i in range(max_iteration):
        V_new=policy_evaluation(P, nS, nA, policy, gamma)
        policy_new=policy_improvement(P, nS, nA, V_new, policy, gamma)
        if isover(V,V_new,tol) :
            break
        V=V_new.copy()
        policy=policy_new.copy()

    return V, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    max_iteration=40
    V = np.zeros(nS,dtype=float)
    policy = np.zeros(nS, dtype=int)
    for i in range(max_iteration):
        V_next=np.zeros(nS,dtype=float)
        for s in range(nS):
            for a in range(nA):
                q=0
                for probability, nextstate, reward, terminal in P[s][a]:
                    q += probability * (reward + gamma * V[nextstate])
                if V_next[s] < q:
                    V_next[s] = q
        if isover(V,V_next,tol):
             break
        V = V_next.copy()
    policy=policy_improvement(P, nS, nA, V_next, policy, gamma)

    return V_next, policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
              break
    env.render();
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

	# comment/uncomment these lines to switch between deterministic/stochastic environments
	#env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")

	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, 100)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, 100)


