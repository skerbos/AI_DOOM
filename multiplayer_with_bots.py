#!/usr/bin/env python3

#####################################################################
# This script presents how to play a deathmatch game with built-in bots.
#####################################################################

import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
import vizdoom as vzd
from tqdm import trange

from Agents.DQN import DQNAgent
import rewards

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 1
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4
resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = True
load_model = False
skip_learning = False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

# class DuelQNet(nn.Module):
#     """
#     This is Duel DQN architecture.
#     see https://arxiv.org/abs/1511.06581 for more information.
#     """

#     def __init__(self, available_actions_count):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#         )

#         self.conv4 = nn.Sequential(
#             nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#         )

#         self.state_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))

#         self.advantage_fc = nn.Sequential(
#             nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, available_actions_count)
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = x.view(-1, 192)
#         x1 = x[:, :96]  # input for the net to calculate the state value
#         x2 = x[:, 96:]  # relative advantage of actions in the state
#         state_value = self.state_fc(x1).reshape(-1, 1)
#         advantage_values = self.advantage_fc(x2)
#         x = state_value + (
#             advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
#         )

#         return x

# class DQNAgent:
#     def __init__(
#         self,
#         action_size,
#         memory_size,
#         batch_size,
#         discount_factor,
#         lr,
#         load_model,
#         epsilon=1,
#         epsilon_decay=0.9996,
#         epsilon_min=0.1,
#     ):
#         self.action_size = action_size
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.epsilon_min = epsilon_min
#         self.batch_size = batch_size
#         self.discount = discount_factor
#         self.lr = lr
#         self.memory = deque(maxlen=memory_size)
#         self.criterion = nn.MSELoss()

#         if load_model:
#             print("Loading model from: ", model_savefile)
#             self.q_net = torch.load(model_savefile)
#             self.target_net = torch.load(model_savefile)
#             self.epsilon = self.epsilon_min

#         else:
#             print("Initializing new model")
#             self.q_net = DuelQNet(action_size).to(DEVICE)
#             self.target_net = DuelQNet(action_size).to(DEVICE)

#         self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

#     def get_action(self, state):
#         if np.random.uniform() < self.epsilon:
#             return random.choice(range(self.action_size))
#         else:
#             state = np.expand_dims(state, axis=0)
#             state = torch.from_numpy(state).float().to(DEVICE)
#             action = torch.argmax(self.q_net(state)).item()
#             return action

#     def update_target_net(self):
#         self.target_net.load_state_dict(self.q_net.state_dict())

#     def append_memory(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def train(self):
#         batch = random.sample(self.memory, self.batch_size)
#         batch = np.array(batch, dtype=object)

#         states = np.stack(batch[:, 0]).astype(float)
#         actions = batch[:, 1].astype(int)
#         rewards = batch[:, 2].astype(float)
#         next_states = np.stack(batch[:, 3]).astype(float)
#         dones = batch[:, 4].astype(bool)
#         not_dones = ~dones

#         row_idx = np.arange(self.batch_size)  # used for indexing the batch

#         # value of the next states with double q learning
#         # see https://arxiv.org/abs/1509.06461 for more information on double q learning
#         with torch.no_grad():
#             next_states = torch.from_numpy(next_states).float().to(DEVICE)
#             idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
#             next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
#             next_state_values = next_state_values[not_dones]

#         # this defines y = r + discount * max_a q(s', a)
#         q_targets = rewards.copy()
#         q_targets[not_dones] += self.discount * next_state_values
#         q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

#         # this selects only the q values of the actions taken
#         idx = row_idx, actions
#         states = torch.from_numpy(states).float().to(DEVICE)
#         action_values = self.q_net(states)[idx].float().to(DEVICE)

#         self.opt.zero_grad()
#         td_error = self.criterion(q_targets, action_values)
#         td_error.backward()
#         self.opt.step()

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#         else:
#             self.epsilon = self.epsilon_min

def run(game, agent, actions, episodes, bots, num_epochs, frame_repeat, steps_per_epoch):

    last_frags = 0
    prev_item = 0
    train_scores = []

    for i in range(episodes):

        print("Episode #" + str(i + 1))

        # Add specific number of bots
        # (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
        # edit this file to adjust bots).
        game.send_game_command("removebots")
        for i in range(bots):
            game.send_game_command("addbot")

        # Play until the game (episode) is over.

        # Initialize new reward
        reward = 0
        while not game.is_episode_finished():

            # Get the state.
            state = game.get_state()

            # Analyze the state.

            # Make your action.
            #game.make_action(choice(actions))
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            reward = game.make_action(actions[best_action_index], frame_repeat)

            reward += rewards.frag_reward(game, 100, last_frags)
            prev_frag = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)

            reward += rewards.item_reward(game, 20, prev_item)
            prev_item  = game.get_game_variable(vzd.GameVariable.ITEMCOUNT)

            next_state = preprocess(game.get_state().screen_buffer)
            # Update agent rewards
            agent.append_memory(state, actions, reward, next_state, game.is_episode_finished())

            frags = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
            if frags != last_frags:
                last_frags = frags
                print("Player has " + str(frags) + " frags.")

            items = game.get_game_variable(vzd.GameVariable.ITEMCOUNT)
            if items != prev_item:
                prev_item = items
                print("Player has " + str(items) + " items.")

            # Check if player is dead
            if game.is_player_dead():
                print("Player died.")
                # Use this to respawn immediately after death, new state will be available.
                game.respawn_player()

        print("Episode finished.")
        print("************************")

        print("Results:")
        server_state = game.get_server_state()
        for i in range(len(server_state.players_in_game)):
            if server_state.players_in_game[i]:
                print(
                    server_state.players_names[i]
                    + ": "
                    + str(server_state.players_frags[i])
                )
        print("************************")

        # Starts a new episode. All players have to call new_episode() in multiplayer mode.
        train_scores.append(game.get_total_reward() + reward)
        game.new_episode()

        agent.update_target_net()

        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

    if save_model:
        print("Saving the network weights to:", model_savefile)
        torch.save(agent.q_net, model_savefile)

    return agent, game

if __name__ == "__main__":
    game = vzd.DoomGame()

    # Use CIG example config or your own.
    game.load_config(os.path.join(vzd.scenarios_path, "multiplayer_with_bots.cfg"))

    game.set_doom_map("map01")  # Limited deathmatch.
    #game.set_doom_map("map02")  # Full deathmatch.

    # Start multiplayer game only with your AI
    # (with options that will be used in the competition, details in cig_mutliplayer_host.py example).
    game.add_game_args(
        "-host 1 -deathmatch +timelimit 10.0 "
        "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
        "+viz_respawn_delay 10 +viz_nocheat 1"
    )

    # Bots are loaded from file, that by default is bots.cfg located in the same dir as ViZDoom exe
    # Other location of bots configuration can be specified by passing this argument
    game.add_game_args("+viz_bots_path ../../scenarios/perfect_bots.cfg")

    # Name your agent and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game.add_game_args("+name AI +colorset 0")

    game.set_mode(vzd.Mode.PLAYER)
    game.set_console_enabled(True)

    # game.set_window_visible(False)

    game.init()

    # Three example sample actions
    actions = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0]
    ]
    last_frags = 0

    # Play with this many bots
    bots = 7

    # Run this many episodes
    episodes = 10

    # Instantiate DQNAgent
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
    )

    agent, game = run(
        game,
        agent,
        actions,
        episodes,
        bots,
        5,
        frame_repeat,
        25200
    )

    game.close()
