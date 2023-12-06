
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from DQN import DQN

# 创建环境
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
dqn_agent = DQN(state_size, action_size)

# 训练智能体
num_episodes = 1000
for i_episode in range(num_episodes):
    # state = torch.FloatTensor([env.reset()])
    # state = torch.zeros(4, 64, dtype=float)
    state = torch.zeros(1, 4)
    env.reset()
    for t in range(500):
        _, action = dqn_agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        reward = reward if not done else -10
        next_state = torch.FloatTensor([next_state])
        dqn_agent.remember(state, action, reward, next_state, done)
        state = next_state
        # if reward != 1:
        #     print(f'Reward = {reward}')
            
        if done:
            print("Episode: {}/{}, Score: {}".format(i_episode, num_episodes, t))
            break
        if len(dqn_agent.memory) > 32:
            dqn_agent.replay(32)


torch.save(dqn_agent.model.state_dict(), 'saved/saved_model.pkl') # save model and tensor             