from collections import deque
import itertools as it
import os
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import vizdoom as vzd
from tqdm import trange

from Agents.ACN import Actor_Critic_Agent, preprocess, stack_frames
from rewards import dist_reward, dist_fixed_reward

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "Single_player.cfg")\

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")
# resolution = (30, 45)


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_doom_map("E1M1")
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game


def test(agent):
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
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )

if __name__ == "__main__":
    # Initialize game and actions
    start_time = time()
    game = create_simple_game()
    n = game.get_available_buttons_size()
    load_model = ".\ckpt\model-doom-ACNagent-stacked-unfreeze-E1M1-distfixed-ckpt2-0.001-0.001-120000-(64, 96).pth"
    start_timestep = 120000
    actions = [list(a) for a in it.product([0, 1], repeat=n)]   
    save_model = True
    skip_learning = False
    episodes_to_watch = 3

    # Initialize our agent with the set parameters
    agent = Actor_Critic_Agent(action_size= n, game = game,
                                load_model=load_model, start_time=start_timestep)
    agent.epsilon= 0.6
    # Run the training for the set number of epochs
    if not skip_learning:
        max_timesteps = 100000
        agent.learn(max_timesteps)
        if save_model:
            agent.save_model(max_timesteps)
        print(time())
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))
        print("======================================")
        print("Training finished. It's time to watch!")
    agent.critic.eval()
    agent.actor.eval()
    test(agent)

    # Reinitialize the game with window visible
    agent.game.close()
    agent.game.set_window_visible(True)
    agent.game.init()

    for _ in range(episodes_to_watch):
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
            best_action_index,_= agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            agent.game.set_action(best_action_index)
            for _ in range(4):
                agent.game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = agent.game.get_total_reward()
        print("Total score: ", score)