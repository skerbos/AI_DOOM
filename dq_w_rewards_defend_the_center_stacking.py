#!/usr/bin/env python3

# E. Culurciello, L. Mueller, Z. Boztoprak
# December 2020

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
import pandas as pd
from datetime import datetime
from tqdm import trange
from torchinfo import summary


# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs =  5
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 10

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 20
exp_name = "dq_w_rewards_defend_the_center_double_epochs"
model_savefile = 'C:/Users/nicho/Desktop/SUTD Term 6/50.021/Project/AI_DOOM/pth/' + exp_name + ".pth"
save_model = True
load_model = False
skip_learning = False
set_additional_rewards = True

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "defend_the_center.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")

# Uses GPU if available``
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
        # print(stacked_state)
        # print(stacked_state.shape)
        # assert False
    return stacked_state, stacked_frames


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    # game.set_episode_timeout(2000)
    game.init()
    print("Doom initialized.")

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(agent.test_episodes_per_epoch, leave=False):
        agent.game.new_episode()
        stacked_frames = deque([torch.zeros(agent.resolution, dtype=torch.int) for i in range(agent.stack_size)], maxlen = agent.stack_size)
        new = True
        while not agent.game.is_episode_finished():
            state = agent.game.get_state().screen_buffer
            if new:
                state, stacked_frames = stack_frames(stacked_frames, state, True, agent.stack_size, agent.resolution)
                new = False
            else:
                state, stacked_frames = stack_frames(stacked_frames, state, False, agent.stack_size, agent.resolution)
            best_action,_ = agent.get_action(state)
            agent.game.make_action(best_action, agent.frame_repeat)
        r = agent.game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    result = (
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )
    return result


def additional_rewards(game, reward, variable, var_count, amt=100):
    # print(game.get_game_variable(variable))
    # TODO: Modify base on state
    if game.get_game_variable(variable) > var_count:
        reward = reward + (game.get_game_variable(variable) - var_count) * amt
        var_count = game.get_game_variable(variable)
        
        # print(game.get_game_variable(vzd.GameVariable.HEALTH),reward)
    else:
        reward = reward
    return reward, var_count

def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    train_lines = []
    test_lines = []
    result_lines = []
    reward_ls = [f'KILLCOUNT: 5', f'HITS_TAKEN: -2', f'DEATH_PENALTY: 1']

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        dist_set = []
        kill_count = 0
        item_count = 0
        health_count = 0
        armor_count = 0
        hits_taken_count = 0
        target_ls = [(531,-3142,0), (1505, -2500, 0), (2104, -2690, 0), (2905,-2813, 0), (3007, -3962, 0), (3017, -4401, 0), (2915, -4827,0)]

        print("\nEpoch #" + str(epoch + 1))
        if epoch == num_epochs - 1:
            game.set_window_visible(False)

        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            # reward, kill_count = additional_rewards(game, reward, vzd.GameVariable.KILLCOUNT, kill_count, amt=5)
            # # reward, item_count = additional_rewards(game, reward, vzd.GameVariable.ITEMCOUNT, item_count, amt=1)
            # # reward, health_count = additional_rewards(game, reward, vzd.GameVariable.HEALTH, health_count, amt=1)
            # # reward, armor_count = additional_rewards(game, reward, vzd.GameVariable.ARMOR, armor_count, amt=5)
            # reward, hits_taken_count = additional_rewards(game, reward, vzd.GameVariable.HITS_TAKEN, hits_taken_count, amt=-2)
            # (reward, dist_set) = distance_reward(game, dist_set, reward, target_ls)
            done = game.is_episode_finished()
            target_ls = [(531,-3142,0), (1505, -2500, 0), (2104, -2690, 0), (2905,-2813, 0), (3007, -3962, 0), (3017, -4401, 0), (2915, -4827,0)]

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, resolution[0], resolution[1])).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)

        train_result = (
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        train_lines.append(train_result)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test_result = test(game, agent)
        if save_model:
            model_savefile_new = model_savefile[:-4] + (datetime.now().strftime("_%m_%d_%Y_%H_%M_%S")) + '.pth'
            print("Saving the network weights to:", model_savefile_new)
            torch.save(agent.q_net, model_savefile_new)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))
        test_lines.append(test_result)


    game.close()
    result_lines = [train_lines, test_lines]
    return agent, game, result_lines, reward_ls


class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, available_actions_count):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # self.pooling = nn.MaxPool2d(2,2)
        self.state_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))

        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, available_actions_count)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.pooling(x)
        x = self.conv4(x)
        # x = self.pooling(x)
        x = x.view(-1, 192)
        x1 = x[:, :96]  # input for the net to calculate the state value
        x2 = x[:, 96:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
        )

        return x


class DQNAgent:
    def __init__(
        self,
        action_size,
        memory_size,
        batch_size,
        discount_factor,
        lr,
        load_model,
        epsilon=1,
        epsilon_decay=0.9996,
        epsilon_min=0.1,
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


if __name__ == "__main__":
    # Logging
    print('This experiments: ', exp_name)
    exp_name = '_%s_%s' % (datetime.now().strftime('%m%d'), exp_name) 
    temp_df = pd.DataFrame([])

    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game, results, reward_ls = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    game.close()

    print(learning_rate)
    # Save log file
    temp_df['frame_repeat'] = [frame_repeat for x in range(len(results[0]))]
    temp_df['learning_rate'] = [learning_rate for x in range(len(results[0]))]
    temp_df['discount_factor'] = [discount_factor for x in range(len(results[0]))]
    temp_df['train_epochs'] = [train_epochs for x in range(len(results[0]))]
    temp_df['learning_steps_per_epoch'] = [learning_steps_per_epoch for x in range(len(results[0]))]
    temp_df['replay_memory_size'] = [replay_memory_size for x in range(len(results[0]))]
    temp_df['batch_size'] = [batch_size for x in range(len(results[0]))]
    temp_df['test_episodes_per_epoch'] = [test_episodes_per_epoch for x in range(len(results[0]))]
    temp_df['rewards'] = [reward_ls for x in range(len(results[0]))]
    temp_df['train_results'] = results[0]
    temp_df['test_results'] = results[1]
    temp_df.to_csv('C:/Users/nicho/Desktop/SUTD Term 6/50.021/Project/AI_DOOM/logs/' + exp_name + '.csv')

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)