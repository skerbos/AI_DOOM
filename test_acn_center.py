from collections import deque
import itertools as it
import os
from time import sleep, time
import sys
import numpy as np
import skimage.transform
import torch
import vizdoom as vzd
from tqdm import trange
import moviepy.editor as mpy
import matplotlib.pyplot as plt

# from Agents.ACN import Actor_Critic_Agent, preprocess
from Agents.ACN_center import Actor_Critic_Agent, preprocess, stack_frames
from rewards import dist_reward, dist_fixed_reward

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "defend_the_center.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")
# resolution = (30, 45)
def make_gif(images, fname, fps=50):
    """
    Description
    ---------------
    Makes gifs from list of images
    
    Parameters
    ---------------
    images  : list, contains all images used to creates a gif
    fname   : str, name used to save the gif
    
    """
    def make_frame(t):
        try: x = images[int(fps*t)]
        except: x = images[-1]
        return x.astype(np.uint8)
    clip = mpy.VideoClip(make_frame, duration=len(images)/fps)
    clip.size = (640, 480)
    clip.fps = fps
    clip.write_gif(fname, program='ffmpeg', fuzz=50, verbose=False)

def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    # game.set_doom_map("E1M1")
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
    return test_scores.mean()

if __name__ == "__main__":
    # Initialize game and actions
    try:
        load_model = sys.argv[1]
    except:
        sys.exit("Please provide path to a Actor Critic model Checkpoint as an argument (python test_acn_center.py <ckpt_name>) without quotes.")
    start_time = time()
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    episodes_to_watch = 3

    # Initialize our agent with the set parameters
    agent = Actor_Critic_Agent(action_size= n, game = game, load_model=load_model)
    agent.epsilon= agent.epsilon_min 
    agent.frame_repeat = 2

    agent.critic.eval()
    agent.actor.eval()
    test_score = test(agent)
    # Reinitialize the game with window visible
    agent.game.close()
    agent.game.set_window_visible(True)
    agent.game.init()

    for i in range(episodes_to_watch):
        agent.game.new_episode("./episodes/" + str(i) + "_rec.lmp")
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
            for _ in range(agent.frame_repeat):
                agent.game.advance_action()
        # Sleep between episodes
        sleep(1.0)
        score = agent.game.get_total_reward()
        print("Total score: ", score)
    agent.game.close()
    agent.game.set_screen_format(vzd.ScreenFormat.CRCGCB)
    agent.game.set_mode(vzd.Mode.PLAYER)
    agent.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    agent.game.init()
    for i in range(episodes_to_watch):
        episode_frames = []
        agent.game.replay_episode("./episodes/" + str(i) + "_rec.lmp")
        while not agent.game.is_episode_finished():
            s = agent.game.get_state()
            # print(s.screen_buffer.shape)
            # assert False
            episode_frames.append(s.screen_buffer.transpose((1,2,0)))

            # Use advance_action instead of make_action.
            agent.game.advance_action()

        print("Saving episode GIF..")
        # images = np.array(episode_frames)
        gif_file = os.path.join("./gif",agent.name+str(i+1)+".gif")
        make_gif(episode_frames, gif_file, fps=60)
        print("Done")
    agent.game.close()