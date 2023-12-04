import gym
import copy
import random
import numpy as np

from tqdm import tqdm

from utils import *
from config import *
from rollout import *
from exploration import UCB1


# Monte-Carlo tree search
# model dict with 'policy', 'heuristic', 'value'
def search_mcts(env, model={}, cutoff=False, prune=True):
    class State:
        ACTIONS = 4

        def __init__(self, env, parent=None, terminal=False, ID=None):
            self.ID = hash(str(strip(env))) if ID is None else ID

            self.env = env
            self.parent = parent
            self.terminal = terminal
            if not terminal:
                self.counter = [1] * State.ACTIONS
                self.actions = [a for a in range(State.ACTIONS)]
                self.rewards = [None] * State.ACTIONS # instant reward for action leading to child
                self.returns = [None] * State.ACTIONS
                self.children = [None] * State.ACTIONS


        def action_values(self):
            assert(None not in self.children)
            return [self.returns[i] / max(self.counter[i], 1) for i in range(len(self.children))]


        def prune(self, index):
            self.actions.pop(index)
            self.rewards.pop(index)
            self.returns.pop(index)
            self.counter.pop(index)
            self.children.pop(index)


        def update(self, path): # expansion, selection, simulation and backpropagation
            if self.terminal:
                path.append(True)
                return 0

            # expand
            while None in self.children:
                env = copy.deepcopy(self.env)

                index_none = [i for i, x in enumerate(self.children) if x is None]
                if 'policy' in model: # expand best predicted child first
                    predicted = model['policy'].predict(np.expand_dims(binary(strip(env))[0], axis=0))[0]
                    index_none.sort(key = lambda index: predicted[self.actions[index]])
                    index = index_none[-1]
                else: index = random.choice(index_none) # expand random child

                _, reward, done, info = env.step(action_translator[self.actions[index]], 'tiny_rgb_array')
                reward = Sokoban.reward(reward, info)

                env_id = hash(str(strip(env)))
                if prune: # prune if action has no effect or action leading back
                    if self.ID == env_id or (self.parent is not None and self.parent.ID == env_id):
                        self.prune(index)
                        continue # expand other child

                state = State(env, self, done, env_id)
                value = 0 if state.terminal else rollout(state.env, model=model)[0]
                self.rewards[index] = reward
                self.returns[index] = reward + Config.discount_tree * value
                self.children[index] = state

                if done: path.append(True)
                path.append(self.actions[index])
                return self.returns[index]

            while True:
                if len(self.children) == 0: return False # all childrens pruned

                # select
                action = UCB1.calculate(self.action_values(), self.counter)[0]
                value = self.children[action].update(path)
                if type(value) != bool: # backpropagation
                    path.append(self.actions[action])

                    value = self.rewards[action] + Config.discount_tree * value
                    self.returns[action] += value
                    self.counter[action] += 1
                    return value
                else: self.prune(action) # has no children

    # build tree
    iterations = 0
    start_node = State(copy.deepcopy(env))
    while True:
        iterations += 1
        if cutoff and iterations > cutoff: break

        update_path = []
        value = start_node.update(update_path)
        if type(update_path[0]) == bool: # solution found
            update_path.reverse()
            return update_path[:-1]

    return False


# Unit Test
if __name__ == "__main__":
    # create envs to test
    generated_envs = Sokoban(boxes=2, count=10, size=(7, 7))

    for i in range(len(generated_envs.all_envs)):
        env, ff, _ = generated_envs.get(i)
        #draw(env)

        path = search_mcts(env)
        for action in path:
            _, reward, done, _ = env.step(action_translator[action], 'tiny_rgb_array')
            #draw(env)
        assert(done)
        print('Length:', len(path) / len(ff), len(path))