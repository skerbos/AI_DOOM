import random
import os
from collections import deque
import time

import numpy as np
import vizdoom as vzd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm, trange
import skimage
# from AI_DOOM.rewards import hit_reward

from rewards import dist_reward, kill_reward, hit_reward, ammo_reward, dist_fixed_reward, health_reward

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

class Actor(nn.Module):
    def __init__(self, available_actions_count) -> None:
        super().__init__()
        # self.initial_layer = nn.Sequential(nn.Conv2d(1,3,3,1,1), nn.ReLU())
        self.model = torchvision.models.resnet.resnet18(weights='DEFAULT') 
        # model = torchvision.models.efficientnet_b0(weights =  torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        for param in self.model.parameters():
                param.requires_grad = False
        count = 0
        for i in self.model.children():
            count += 1
            if count == 8:
                nc = 0
                for j in i.children():
                    nc +=1
                    if nc == 2:
                        for param in j.parameters():
                            param.requires_grad = True
            if count >8:
                for param in i.parameters():
                    param.requires_grad = True
        self.model = nn.Sequential(
            # self.initial_layer,
            self.model,
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(128, out_features=available_actions_count),
            nn.Sigmoid(),
            )
    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.initial_layer = nn.Sequential(nn.Conv2d(1,3,3,1,1), nn.ReLU())
        self.model = torchvision.models.resnet.resnet18(weights='DEFAULT') 
        # model = torchvision.models.efficientnet_b0(weights =  torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        for param in self.model.parameters():
                param.requires_grad = False
        count = 0
        for i in self.model.children():
            count += 1
            if count == 8:
                nc = 0
                for j in i.children():
                    nc +=1
                    if nc == 2:
                        for param in j.parameters():
                            param.requires_grad = True
            if count >8:
                for param in i.parameters():
                    param.requires_grad = True
        self.model = nn.Sequential(
            # self.initial_layer,
            self.model,
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(128, out_features=1),
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

def stack_frames(stacked_frames, state, is_new_episode, maxlen = 3, resize = (64, 96)):
    
    # Preprocess frame
    frame = preprocess(state, resize)
    frame = torch.tensor(frame)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([frame[None] for i in range(maxlen)], maxlen=maxlen) 
        # Stack the frames
        stacked_state = torch.cat(tuple(stacked_frames), dim = 1)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame[None]) 
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = torch.cat(tuple(stacked_frames), dim = 1)
    return stacked_state, stacked_frames



def unison_shuffled_copies(a, b, c, d):
    assert len(a) == len(b)
    assert len(a) == len(c)
    assert len(a) == len(d)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]

def unison_sample(a,b,c,d, num_samples):
    assert len(a) == len(b)
    assert len(a) == len(c)
    assert len(a) == len(d)
    idx = np.random.choice(np.arange(len(a)), num_samples, replace=False)
    a_sample = a[idx]
    b_sample = b[idx]
    c_sample = c[idx]
    d_sample = d[idx]
    return a_sample, b_sample, c_sample, d_sample

    
class Actor_Critic_Agent():
    def __init__(self, action_size, game, load_model = "", start_time = 0) -> None:
        self.init_hyperparameters()
        self.game = game
        self.action_size = action_size
        self.actor = Actor(self.action_size).to(DEVICE)
        self.critic = Critic().to(DEVICE)
        self.start_time = start_time

        if load_model != "":
            checkpoint = torch.load(load_model)
            self.actor.load_state_dict(checkpoint['Actor_state_dict'])
            self.critic.load_state_dict(checkpoint['Critic_state_dict'])
            

        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.critic_criterion = nn.MSELoss()

    def learn(self, total_time_steps):
        curr_t = 0
        epoch = 0
        while curr_t < total_time_steps:
            self.actor.eval()
            self.critic.eval()
            full_batch_obs, full_batch_acts, full_batch_log_probs, full_batch_rtgs, full_batch_lens = self.rollout()
            # Calculate how many timesteps we collected this batch   
            curr_t += np.sum(full_batch_lens)
            epoch +=1
            for i in range(self.num_minibatches):
                batch_obs, batch_acts, batch_log_probs, batch_rtgs = unison_sample(full_batch_obs, full_batch_acts, full_batch_log_probs, full_batch_rtgs, self.mini_batch_size)

                # Calculate V_{phi, k}
                V, _ = self.evaluate(batch_obs, batch_acts)
                # ALG STEP 5
                # Calculate advantage
                A_k = batch_rtgs - V.detach().cpu()

                # Normalize advantages
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                self.actor.train()
                self.critic.train()

                for _ in range(self.n_updates_per_iteration):

                    # Calculate pi_theta(a_t | s_t)
                    V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                    ratios = torch.exp(curr_log_probs - batch_log_probs)
                    # Calculate surrogate losses
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    # Calculate Actor and critic loss
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = self.critic_criterion(V, batch_rtgs)

                    # Calculate gradients and perform backward propagation for actor 
                    # network
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network    
                    self.critic_optim.zero_grad()    
                    critic_loss.backward()    
                    self.critic_optim.step()
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                else:
                    self.epsilon = self.epsilon_min
            if epoch % 100 == 1:
                self.save_model(epoch, epoch = True)
    
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
            total_rew = 0
            new = True
            # state = self.game.get_state().screen_buffer
            stacked_frames = deque([torch.zeros(self.resolution, dtype=torch.int) for i in range(self.stack_size)], maxlen = self.stack_size)
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                pbar.update(1)
                # Collect observation
                state = self.game.get_state().screen_buffer
                if new:
                    state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size, self.resolution)
                    new = False
                else:
                    state, stacked_frames = stack_frames(stacked_frames, state, False, self.stack_size, self.resolution)
                # obs = preprocess(state, self.resolution)
                batch_obs.append(state)
                action, log_prob = self.get_action(state)
                kill_num = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                # hit_num = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
                AMMO_num = self.game.get_game_variable(vzd.GameVariable.AMMO2)
                health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
                reward = self.game.make_action(action, self.frame_repeat)
                reward += ammo_reward(self.game, 0.1, AMMO_num) + health_reward(self.game,0.1,health) #+ kill_reward(self.game, 1, kill_num)
                done = self.game.is_episode_finished()
                total_rew += reward
            
                # Collect reward, action, and log prob
                ep_rews.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    train_scores.append(total_rew)
                    break
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rewards.append(ep_rews)
        pbar.close()
        train_scores = np.array(train_scores)
        print(train_scores)
        print(
                "Results: mean: {:.1f} +/- {:.1f},".format(
                    train_scores.mean(), train_scores.std()
                ),
                "min: %.1f," % train_scores.min(),
                "max: %.1f," % train_scores.max(),
            )
        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.cat(tuple(batch_obs), dim = 0)
        batch_acts = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rewards)
        # Return the batch data
        batch_obs, batch_acts, batch_log_probs, batch_rtgs = unison_shuffled_copies(batch_obs, batch_acts, batch_log_probs, batch_rtgs)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens 
    

    def get_action(self, obs):
        if np.random.uniform() < self.epsilon:
            mean = torch.ones((self.action_size))/2 #self.action_size
            dist = Bernoulli(mean)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_prob = log_prob.sum()
            return action.detach().numpy(), log_prob.detach()
        else:
            mean = self.actor(obs.to(DEVICE)).cpu()
            dist = Bernoulli(mean)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_prob = log_prob.sum()
            return action.detach().numpy()[0], log_prob.detach()
    
    def init_hyperparameters(self):
        self.name = "ACNagent-defend_center"
        self.gamma = 0.95
        self.actor_lr = 5e-5 #1e-3
        self.critic_lr = 5e-5 #1e-3
        self.timesteps_per_batch = 1000 #2000 
        self.max_timesteps_per_episode = 600
        self.frame_repeat = 2
        self.n_updates_per_iteration = 1
        self.clip = 0.2 # As recommended by the paper
        self.test_episodes_per_epoch = 25
        self.ckpt_dir = "./ckpt_defend_ctr/"
        self.resolution = (64,96)
        self.num_minibatches = 12
        self.mini_batch_size = 100
        self.epsilon = 1
        self.epsilon_decay = 0.9996
        self.epsilon_min = 0.1
        self.min_rew = -15
        self.max_rew = 15
        self.stack_size = 3
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
        batch_obs = batch_obs.reshape((batch_obs.size(0),3,self.resolution[0],self.resolution[1]))

        # obs.unsqueeze(0)
        V = self.critic(batch_obs.to(DEVICE)).squeeze().cpu()


        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs.to(DEVICE)).cpu()
        dist = Bernoulli(mean)
        log_probs = dist.log_prob(batch_acts)
        log_probs = log_probs.sum(dim = 1)
        # Return predicted values V and log probs log_probs
        return V, log_probs

    def save_model(self, max_timesteps, epoch = False):
        # Specify a path to save to
        if not epoch:
            PATH = os.path.join(self.ckpt_dir,f"model-doom-{self.name}-{self.actor_lr}-{self.critic_lr}-{self.start_time+max_timesteps}-{self.resolution}.pth")
        else:
            PATH = os.path.join(self.ckpt_dir,f"model-doom-{self.name}-{self.actor_lr}-{self.critic_lr}-epoch-{max_timesteps}-{self.resolution}.pth")

        torch.save({
                    'Actor_state_dict': self.actor.state_dict(),
                    'Critic_state_dict': self.critic.state_dict()
                    }, PATH)
        
if __name__ =="__main__":
    model = Actor(9)
    summary(model, input_size=(1,3,64,96))
