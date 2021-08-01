import sys
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from Policy_Model import PolicyModel
from collections import Counter
np.set_printoptions(threshold=sys.maxsize)
#discount distant future rewards if 0 < DISCOUNT < 1, weight future rewards equally if DISCOUNT = 1.0
GAMMA = 0.5 

def mean_normalize(observation):
    tensor = torch.from_numpy(np.swapaxes(np.array([observation]), 1, 3))[:, :, :, 35:193]
    tensor.type(torch.DoubleTensor)
    tensor = torch.div(tensor, 255.0)
    channel_means = torch.mean(tensor, dim = [0, 2, 3])
    channel_stds = torch.std(tensor, dim = [0, 2, 3])
    normal_tensor = (tensor - channel_means[None, :, None, None]) / channel_stds[None, :, None, None]
    return normal_tensor

def update_policy(model, rewards, log_probs):
    discounted_rewards = []

    #compute total rewards for each action taken
    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(round(Gt, 4))
    
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
    # print(discounted_rewards.numpy())

    policy_gradient = []
    for log_prob, G in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * G)
    
    model.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    model.optimizer.step()


def main():
    env = gym.make("Pong-v4")
    print(env.unwrapped.get_action_meanings())
    policy_net = PolicyModel()
    # policy_net.load_state_dict(torch.load('last_pong_without_points.pt'))
    max_episode_num = 1000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []
    scores = []
    total_rewards = []
    canUpdate = True

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []
        actions = []
        steps_before_point = 0
        score = 0
        total_rewards_in_episode = 0

        if (episode + 1) % 10 == 0:
            torch.save(policy_net.state_dict(), 'last_pong_without_points.pt')
        for steps in range(max_steps):
            env.render()
            stateTensor = mean_normalize(state)
            action, log_prob = policy_net.get_action(stateTensor)
            # print(action)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            actions.append(action)
            steps_before_point += 1
            # if reward == -1 and steps_before_point > 57:
            #     rewards[-1] = 0.5 

            if reward == 1:
                # rewards[-1] += 2
                # update_policy(policy_net, rewards, log_probs)
                score += 1
            if done:
                # rewards[-1] = (steps_before_point - 47) / 10
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))

                total_rewards_in_episode += np.sum(rewards)
                steps_before_point = 0
                rewards = []
                log_probs = []
                actions = []

                scores.append(score)
                print(scores)
                print(all_rewards)
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