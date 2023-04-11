import vizdoom as vzd
import math

def frag_reward(game, reward_factor, prev_frag):
    return (game.get_game_variable(vzd.GameVariable.FRAGCOUNT)- prev_frag)*reward_factor

def kill_reward(game, reward_per_kill, prev_KILL):
    # print("Kill:",(game.get_game_variable(vzd.GameVariable.KILLCOUNT)-prev_KILL)*reward_per_kill)
    return (game.get_game_variable(vzd.GameVariable.KILLCOUNT)-prev_KILL)*reward_per_kill

def item_reward(game, reward_factor, prev_item):
    return (game.get_game_variable(vzd.GameVariable.ITEMCOUNT)- prev_item)*reward_factor

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

def dist_fixed_reward(game, reward_factor, x_end, y_end, z_end, x_prev, y_prev, z_prev):
    state = game.get_state()
    # print(state.game_variables)
    x_player = state.game_variables[0]
    y_player = state.game_variables[1]
    z_player = state.game_variables[2]
    dist_prev = ((x_end- x_prev)**2+(y_end- y_prev)**2+(z_end- z_prev)**2)
    dist_curr = ((x_end- x_player)**2+(y_end- y_player)**2+(z_end- z_player)**2)
    if dist_prev > dist_curr:
        return 1*reward_factor
    else:
        return -1.5*reward_factor
    # print("dist:", -math.sqrt((x_end- x_player)**2+(y_end- y_player)**2+(z_end- z_player)**2)*reward_factor)