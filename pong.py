import sys
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from Policy_Model import PolicyModel

#discount distant future rewards if 0 < DISCOUNT < 1, weight future rewards equally if DISCOUNT = 1.0
DISCOUNT = 0.9 
def update_policy(model, rewards, log_probs):
    discounted_rewards = []

    #compute total rewards for each action taken
    for t in range(len(rewards)):
        G = 0 
        for i, r in enumerate(rewards[t:]):
            G += DISCOUNT**i * r
        discounted_rewards.append(G)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    model.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    model.optimizer.step()


def main():
    env = gym.make("Pong-v4")
    print(env.unwrapped.get_action_meanings())
    policy_net = PolicyModel()
    # policy_net.load_state_dict(torch.load('last_cart_pole.pt'))
    max_episode_num = 10
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []

        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), 'last_cart_pole.pt')

        for steps in range(max_steps):
            env.render()

            stateTensor = torch.from_numpy(np.swapaxes(np.array([state]), 1, 3))
            action, log_prob = policy_net.get_action(stateTensor)
            # print(action)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
                break
            
            state = new_state
    torch.save(policy_net.state_dict(), 'last_cart_pole.pt')
    # plt.plot(numsteps)
    # plt.plot(avg_numsteps)
    # plt.xlabel('Episode')
    # plt.show()

main()
# env = gym.make("Pong-v4")
# observation = env.reset()
# done = False
# # while not done:
# #     env.render()
    
# #     action = random.choice([0, 1, 2, 5]) # take a random action
# #     observation, reward, done, info = env.step(action)

# model = PolicyModel()
# # plt.imshow(observation, interpolation='nearest')
# # plt.show()
# tensor = torch.from_numpy(np.swapaxes(np.array([observation]), 1, 3))
# print(tensor.size())
# output = model.forward(tensor)
# print(output)