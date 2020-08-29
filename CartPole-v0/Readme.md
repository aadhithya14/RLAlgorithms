# CARTPOLE-V0

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.

I have implemented the following algorithms on the environment  and the results have been compared.

### DQN 

![alt text](https://github.com/aadhithya14/RLprojects/blob/master/CartPole-v0/DQN/Results/result.png)


### REINFORCE 

![alt text](https://github.com/aadhithya14/RLprojects/blob/master/CartPole-v0/POLICY GRADIENTS/REINFORCE/Results/result.png)


### ADVANTAGE-ACTOR-CRITIC

![alt text](https://github.com/aadhithya14/RLprojects/blob/master/CartPole-v0/POLICY%20GRADIENTS/ACTOR-CRITIC/RESULTS/RESULT.png)
