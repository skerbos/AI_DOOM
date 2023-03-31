import random
import os
from collections import deque
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm, trange
import skimage


# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

class Actor(nn.Module):
    def __init__(self, available_actions_count) -> None:
        super().__init__()
        self.initial_layer = nn.Sequential(nn.Conv2d(1,3,3,1,1), nn.ReLU())
        self.model = torchvision.models.efficientnet_b0(weights =  torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = nn.Sequential(
            self.initial_layer,
            self.model,
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(),
            nn.Linear(256, out_features=available_actions_count),
            # nn.Softmax(),
            )
    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.initial_layer = nn.Sequential(nn.Conv2d(1,3,3,1,1), nn.ReLU())
        self.model = torchvision.models.efficientnet_b0(weights =  torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = nn.Sequential(
            self.initial_layer,
            self.model,
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            # nn.Linear(in_features=256, out_features=128),
            # nn.ReLU(),
            nn.Linear(256, out_features=1),
            # nn.Tanh(),
            )
    def forward(self, x):
        return self.model(x)

def preprocess(img, resolution):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def unison_shuffled_copies(a, b, c, d):
    assert len(a) == len(b)
    assert len(a) == len(c)
    assert len(a) == len(d)
    # assert len(a) == len(e)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]
    
class Actor_Critic_Agent():
    def __init__(self, actions, game) -> None:
        self.init_hyperparameters()
        self.game = game
        self.actions = actions
        self.action_size = len(actions)

        self.actor = Actor(self.action_size).to(DEVICE)
        self.critic = Critic().to(DEVICE)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.critic_criterion = nn.MSELoss()
          # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.action_size,), fill_value=0.5)
        
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)
    def learn(self, total_time_steps):
        curr_t = 0
        while curr_t < total_time_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            # Calculate how many timesteps we collected this batch   
            curr_t += np.sum(batch_lens)

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            # ALG STEP 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach().cpu()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate Actor and critic loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = self.critic_criterion(V, batch_rtgs)
                print("Actor_Loss:", actor_loss.item())
                print("Critic_Loss:", critic_loss.item())

                # Calculate gradients and perform backward propagation for actor 
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()

        pass
    def rollout(self):
        # Batch Data
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rtgs = []
        batch_lens = []
        t = 0
        train_scores = []
        # Start rollout
        pbar = tqdm(total = self.timesteps_per_batch)
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            self.game.new_episode()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                pbar.update(1)
                # Collect observation
                obs = preprocess(self.game.get_state().screen_buffer, self.resolution)
                batch_obs.append(obs)
                action, log_prob, curr_action = self.get_action(obs)
                # print(action)
                # print(log_prob)
                reward = self.game.make_action(self.actions[curr_action], self.frame_repeat)
                done = self.game.is_episode_finished()
            
                # Collect reward, action, and log prob
                ep_rews.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    train_scores.append(self.game.get_total_reward())
                    break
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rewards.append(ep_rews)
        pbar.close()
        train_scores = np.array(train_scores)
        print(
                "Results: mean: {:.1f} +/- {:.1f},".format(
                    train_scores.mean(), train_scores.std()
                ),
                "min: %.1f," % train_scores.min(),
                "max: %.1f," % train_scores.max(),
            )
        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rewards)
        # Return the batch data
        batch_obs, batch_acts, batch_log_probs, batch_rtgs = unison_shuffled_copies(batch_obs, batch_acts, batch_log_probs, batch_rtgs)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens 
    

    def get_action(self, obs):
        obs = torch.tensor(obs.astype(np.float32)).reshape((1,1,self.resolution[0],self.resolution[1]))
        mean = self.actor(obs.to(DEVICE)).cpu()
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        # action = np.random.choice(self.action_size, p=action)
        log_prob = dist.log_prob(action)
        curr_action = torch.argmax(action[0])
        return action.detach().numpy(), log_prob.detach(), curr_action
    
    def init_hyperparameters(self):
        self.name = "ACNagent"
        self.gamma = 0.95
        self.actor_lr = 0.00005
        self.critic_lr = 0.01
        self.timesteps_per_batch = 300 #2000 
        self.max_timesteps_per_episode = 300
        self.frame_repeat = 12
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # As recommended by the paper
        self.test_episodes_per_epoch = 100
        self.ckpt_dir = "./ckpt/"
        self.resolution = (32,48)
        pass
    
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        # print(batch_obs.shape)
        batch_obs = batch_obs.reshape((batch_obs.size(0),1,self.resolution[0],self.resolution[1]))

        # obs.unsqueeze(0)
        V = self.critic(batch_obs.to(DEVICE)).squeeze().cpu()

        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs.to(DEVICE)).cpu()
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs

    def save_model(self, max_timesteps):
        # Specify a path to save to
        PATH = os.path.join(self.ckpt_dir,f"model-doom-{self.name}-{self.actor_lr}-{self.critic_lr}-{max_timesteps}-{self.resolution}.pth")

        torch.save({
                    'Actor_state_dict': self.actor.state_dict(),
                    'Critic_state_dict': self.critic.state_dict()
                    }, PATH)
        
if __name__ =="__main__":
    model = Actor(9)
    summary(model, (1,1,480,640))