import vizdoom as vzd
import math

def kill_reward(game, reward_per_kill, prev_KILL):
    # print("Kill:",(game.get_game_variable(vzd.GameVariable.KILLCOUNT)-prev_KILL)*reward_per_kill)
    return (game.get_game_variable(vzd.GameVariable.KILLCOUNT)-prev_KILL)*reward_per_kill

def dist_reward(game, reward_factor, x_end, y_end, z_end):
    state = game.get_state()
    # print(state.game_variables)
    x_player = state.game_variables[0]
    y_player = state.game_variables[1]
    z_player = state.game_variables[2]
    # print("dist:", -math.sqrt((x_end- x_player)**2+(y_end- y_player)**2+(z_end- z_player)**2)*reward_factor)
    return -((x_end- x_player)**2+(y_end- y_player)**2+(z_end- z_player)**2)*reward_factor

def ammo_reward(game, reward_factor, prev_ammo):
    # print("ammo:", -(prev_ammo-game.get_game_variable(vzd.GameVariable.AMMO1))*reward_factor)
    return -(prev_ammo-game.get_game_variable(vzd.GameVariable.AMMO1))*reward_factor

def hit_reward(game, reward_factor, prev_hit):
    # print("hit :", (game.get_game_variable(vzd.GameVariable.HITCOUNT)- prev_hit)*reward_factor)
    return (game.get_game_variable(vzd.GameVariable.HITCOUNT)- prev_hit)*reward_factor
