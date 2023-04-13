import vizdoom as vzd
import math

def kill_reward(game, reward_factor, prev_KILL):
    # print("Kill:",(game.get_game_variable(vzd.GameVariable.KILLCOUNT)-prev_KILL)*reward_per_kill)
    d_kill = prev_KILL-game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    if d_kill >0:
        return reward_factor
    else:
        return 0

def dist_reward(game, reward_factor, x_end, y_end, z_end):
    state = game.get_state()
    # print(state.game_variables)
    x_player = state.game_variables[0]
    y_player = state.game_variables[1]
    z_player = state.game_variables[2]
    # print("dist:", -math.sqrt((x_end- x_player)**2+(y_end- y_player)**2+(z_end- z_player)**2)*reward_factor)
    return -((x_end- x_player)**2+(y_end- y_player)**2+(z_end- z_player)**2)*reward_factor

def ammo_reward(game, reward_factor, prev_ammo):
    # print("ammo:", -(prev_ammo-game.get_game_variable(vzd.GameVariable.AMMO2))*reward_factor)
    d_ammo = prev_ammo-game.get_game_variable(vzd.GameVariable.AMMO2)
    # d_ammo = 1-0 =1 >0
    if d_ammo >0:
        return -1* reward_factor
    else:
        return 0
    # return -()*reward_factor

def hit_reward(game, reward_factor, prev_hit):
    # print("hit :", (game.get_game_variable(vzd.GameVariable.HITCOUNT)- prev_hit)*reward_factor)
    d_hit = prev_hit-game.get_game_variable(vzd.GameVariable.HITCOUNT)
    if d_hit >0:
        return reward_factor
    else:
        return 0

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

def health_reward(game, reward_factor, prev_health):
    new_health = game.get_game_variable(vzd.GameVariable.HEALTH)
    if new_health<prev_health:
        return -1*reward_factor
    else:
        return 0
