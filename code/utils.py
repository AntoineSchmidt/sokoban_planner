import os
import re
import sys
import copy
import random
import gym
import gym_sokoban
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from gym_sokoban.envs.sokoban_env import SokobanEnv

from config import *


# create sokoban environments
class Sokoban:
    def reward(value, info):
        ''' modify reward value '''
        if Config.custom: # custom reward function
            if info.get('all_boxes_on_target', False) or value == 0.9: return 1.0 # box pushed on target
            if value == -1.1: return -1.0 # box pushed off target
            if not info['action.moved_player']: return -1.0 # action has no effect
            return -0.1 # general step punishment
        return value / 2


    def create(size=(7, 13), steps=50, boxes=None):
        if boxes is None: boxes = np.random.randint(1, 4)

        room_size = (np.random.randint(size[0], size[1] + 1), np.random.randint(size[0], size[1] + 1))

        while True:
            env = SokobanEnv(dim_room=room_size, num_gen_steps=steps, max_steps=-1, num_boxes=boxes)
            env_solution = solve(env)
            if env_solution: break
            print("no solution found / timeout")

        return env, env_solution


    def predict(states):
        ''' calculates manhattan like heuristic value, using compatible interface '''
        result = []
        for state in states:
            heuristic = 0
            for box in np.argwhere(state[:, :, 1]):
                manhattan = np.inf
                for target in np.argwhere(state[:, :, 3]):
                    value = np.sum(np.abs(box - target))
                    if value < manhattan: manhattan = value
                heuristic += manhattan
            result.append([heuristic])
        return result


    def __init__(self, size=(7, 13), steps=50, boxes=None, count=10):
        self.all_envs = []
        for _ in tqdm(range(count)):
            self.all_envs.append(Sokoban.create(size, steps, boxes))


    def get(self, index=None, min_index=0):
        if index is None:
            assert(min_index < len(self.all_envs))
            index = np.random.randint(min_index, len(self.all_envs))
        else: index %= len(self.all_envs)
        return copy.deepcopy(self.all_envs[index]) + (index,)


# action dictionary for fast forward results
action_dict = {
    'dir-up': 0,
    'dir-right': 1,
    'dir-down': 2,
    'dir-left': 3,
}

# action translator (my encoding -> gym sokoban encoding)
# for easier data augmentation
action_translator = (1, 4, 2, 3)


# calculate class weights for length
def weights(length):
    weights = {}
    counts = np.unique(length, return_counts=True)[1]
    counts = max(counts) / counts
    counts /= max(counts)
    for i in range(len(counts)): weights[i+1] = counts[i]
    return weights


# calculates softmax values
def softmax(values, tau=1.0):
    values = np.exp((values - max(values)) / tau)
    values /= np.sum(values)
    return values


# solves problem with fastforward
def solve(env):
    room = _shrink(strip(env))

    # generate pddl problem file
    with open('pddl/task.pddl', 'w') as f:
        f.write('(define (problem task) (:domain sokoban)\n')

        # objects
        f.write('(:objects\n')
        f.write('dir-down - direction\n')
        f.write('dir-left - direction\n')
        f.write('dir-right - direction\n')
        f.write('dir-up - direction\n')
        f.write('player - thing\n')
        stone_counter = 0
        for x in range(room.shape[0]):
            for y in range(room.shape[1]):
                f.write('pos-{}-{} - location\n'.format(x, y))
                if room[x, y] == 4 or room[x, y] == 3:
                    f.write('stone{} - thing\n'.format(stone_counter))
                    stone_counter += 1
        f.write(')\n')

        # goal
        f.write('(:goal (and\n')
        for i in range(stone_counter):
            f.write('(at-goal stone{})\n'.format(i))
        f.write('))\n')

        # init
        f.write('(:init\n')
        f.write('(move dir-down)\n')
        f.write('(move dir-left)\n')
        f.write('(move dir-right)\n')
        f.write('(move dir-up)\n')
        stone_counter = 0
        for x in range(room.shape[0]):
            for y in range(room.shape[1]):
                if room[x, y]: # not a wall
                    if room[x, y] == 1 or room[x, y] == 2: # position can be moved to
                        f.write('(clear pos-{}-{})\n'.format(x, y))
                    elif room[x, y] == 5 or room[x, y] == 6: # player position
                        f.write('(at player pos-{}-{})\n'.format(x, y))
                        f.write('(is-player player)\n')
                    elif room[x, y] == 3 or room[x, y] == 4: # stone position
                        f.write('(at stone{} pos-{}-{})\n'.format(stone_counter, x, y))
                        f.write('(is-stone stone{})\n'.format(stone_counter))
                        if room[x, y] == 3: f.write('(at-goal stone{})\n'.format(stone_counter))
                        stone_counter += 1

                    if room[x, y] == 2 or room[x, y] == 3 or room[x, y] == 6: # goal position
                        f.write('(is-goal pos-{}-{})\n'.format(x, y))
                    else: f.write('(is-nongoal pos-{}-{})\n'.format(x, y))

                    # check possible moving directions, match to array visualisation (also gym-sokoban way)
                    if y-1 >= 0 and room[x, y-1] != 0:
                        f.write('(move-dir pos-{}-{} pos-{}-{} dir-left)\n'.format(x, y, x, y-1))
                    if y+1 < room.shape[1] and room[x, y+1] != 0:
                        f.write('(move-dir pos-{}-{} pos-{}-{} dir-right)\n'.format(x, y, x, y+1))
                    if x-1 >= 0 and room[x-1, y] != 0:
                        f.write('(move-dir pos-{}-{} pos-{}-{} dir-up)\n'.format(x, y, x-1, y))
                    if x+1 < room.shape[0] and room[x+1, y] != 0:
                        f.write('(move-dir pos-{}-{} pos-{}-{} dir-down)\n'.format(x, y, x+1, y))
        f.write(')')

        f.write(')')

    # run ff
    for _ in range(3):
        try:
            timeout_cmd = "gtimeout" if sys.platform == "darwin" else "timeout"
            cmd_str = "{} {} {} -o {} -f {}".format(timeout_cmd, Config.timeout, "ff-v2.3/ff", "pddl/sokoban.pddl", "pddl/task.pddl")
            if Config.wsl: cmd_str = "wsl " + cmd_str # run ff in wsl
            output = subprocess.getoutput(cmd_str)
            break
        except KeyboardInterrupt: raise
        except: # sometimes problems executing subprocess
            print("ff retry")
            output = "unsolvable"

    # evaluate result
    if "goal can be simplified to FALSE" in output or "unsolvable" in output: return []

    plan = re.findall(r"\d+?: (.+)", output.lower())
    return [action_dict[step.split()[-1]] for step in plan] if plan else []


# returns pure sokoban board
def strip(env):
    room = np.array(env.room_state)
    if env.room_fixed is not None:
        # player on a target
        room[(room == 5) & (env.room_fixed == 2)] = 6
    return room

# remove pure multiple wall borders
def _shrink(room):
    non_walls = np.argwhere(room)
    top_left = non_walls.min(axis=0)
    bottom_right = non_walls.max(axis=0)
    return room[top_left[0]-1:bottom_right[0]+2, top_left[1]-1:bottom_right[1]+2] # keep single wall border


# returns state padded and binary encoded
# every position gets decomposed into [player, stone, floor, goal]
def binary(room, shrink=False, random=False, size=(19, 19), placement=None):
    assert type(room) is np.ndarray

    if shrink: room = _shrink(room)

    # insert into max-size problem
    xpad = (size[0] - room.shape[0])//2
    ypad = (size[1] - room.shape[1])//2
    if placement is not None:
        xpad, ypad = placement
    elif random:
        xpad = np.random.randint(2*xpad+1)
        ypad = np.random.randint(2*ypad+1)
    room_padded = np.zeros(size)
    room_padded[xpad:xpad+room.shape[0], ypad:ypad+room.shape[1]] = room

    wall = [0, 0, 0, 0]
    floor = [0, 0, 1, 0]
    stone_target = [0, 0, 1, 1]
    stone_on_target = [0, 1, 1, 1]
    stone = [0, 1, 1, 0]
    player = [1, 0, 1, 0]
    player_on_target = [1, 0, 1, 1]

    surfaces = [wall, floor, stone_target, stone_on_target, stone, player, player_on_target]

    # assemble binary encoding
    encoded = np.zeros(size+(4,))
    for x in range(size[0]):
        for y in range(size[1]):
            encoded[x, y, :] = surfaces[int(room_padded[x, y])]

    return encoded, (xpad, ypad)


# augments single transition randomly, rotate and mirror
def augment(state, action=None, next_state=None, r=None):
    if r is None: r = random.randrange(8)
    else: assert(0 <= r < 8)

    # rotate
    rotations = r % 4
    state = np.rot90(state, rotations)
    if action is not None: action = np.roll(action, -rotations)
    if next_state is not None: next_state = np.rot90(next_state, rotations)

    # mirror
    flip = r // 4
    if flip:
        state = np.flip(state, flip)
        if action is not None:
            swap = action[flip]
            action[flip] = action[flip + 2]
            action[flip + 2] = swap
        if next_state is not None: next_state = np.flip(next_state, flip)

    return state, action, next_state


# evalute model greedily, ignore discount
# returns average reward, coverage, average quality from solved
def evaluate(model, problems, index=None, stats=None):
    total_envs = np.zeros((3,))
    total_reward = np.zeros((3,))
    total_coverage = np.zeros((3,))
    total_quality = np.zeros((3,))

    for i in range(len(problems.all_envs) if index is None else index):
        env, plan, _ = problems.get(i)
        total_envs[env.num_boxes - 1] += 1

        for j in range(2 * len(plan)):
            env_b = binary(strip(env))[0]
            action = np.argmax(model.predict(np.expand_dims(env_b, axis=0))[0]) # best predicted action
            _, reward, done, info = env.step(action_translator[action], 'tiny_rgb_array')
            total_reward[env.num_boxes - 1] += Sokoban.reward(reward, info)

            if done:
                if stats is not None: stats.append(i)
                total_coverage[env.num_boxes - 1] += 1
                total_quality[env.num_boxes - 1] += (j + 1) / len(plan)
                break

    return (total_reward / total_envs).tolist(),\
           (total_coverage / total_envs).tolist(),\
           (total_quality / total_coverage).tolist()


def plot_stats(stats_val, file=None):
    x = np.arange(len(stats_val))
    label = ['Average\nreward', 'Coverage', 'Average\nlength']

    index = 0
    for i in range(3): # measurement
        for j in range(3): # boxes
            index += 1
            values = [y[i][j] for y in stats_val]
            plt.subplot(3, 3, index)
            plt.plot(x, values, color='green')
            plt.plot(x, [stats_val[0][i][j]] * len(x), color='blue') # baseline
            plt.xticks([])

            if i == 0:
                plt.ylim(-5.0, 3.0)
                plt.title('1 Box' if j == 0 else '{} Boxes'.format(j + 1))
            if i == 1:
                plt.ylim(0.3, 1.1)
            if i == 2:
                plt.ylim(1.0, 0.70)

            if j > 0: plt.yticks([])
            else: plt.ylabel(label[i])

    if file is not None:
        plt.savefig(file, bbox_inches='tight', dpi=200)
    plt.show()


# draw sokoban environment
def draw(env, file=None):
    plt.imshow(env.get_image(mode='rgb_array'))
    plt.axis('off')
    if file is not None:
        plt.savefig(file, bbox_inches='tight', dpi=200)
    plt.show()


# Unit Test
if __name__ == "__main__":
    import pickle

    # create trivial Sokoban image
    env = SokobanEnv(dim_room=(5, 5), num_gen_steps=40, max_steps=-1, num_boxes=1)
    env.room_fixed = None

    new_state = np.zeros((3, 5), dtype=np.uint8)
    new_state[1, 1:4] = [5, 4, 2]

    env.room_state = new_state
    draw(env, 'images/trivial.png')


    # plot training stats
    training_stats = ['il_action_exploration', 'rl_qaction_dqn', 'rl_action_ppo']
    for training_stat in training_stats:
        with open('models/{}.pkl'.format(training_stat), 'rb') as f:
            stats = pickle.load(f)

            length = len(stats)
            print(training_stat, length)
            if length > 180:
                index = [0] + np.random.choice(length, 179, replace=False).tolist()
                index.sort()
                stats = [stats[i] for i in index]

            plot_stats(stats, 'models/{}.png'.format(training_stat))

            for j in range(3):
                values = [y[1][j] for y in stats]
                values.reverse()
                index = len(values) - np.argmax(values) - 1
                print('Box', j + 1, max(values), index, stats[index][1])