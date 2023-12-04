import gym
import copy
import random
import numpy as np

from tqdm import tqdm

from utils import *
from config import *
from rollout import *
from exploration import UCB1


# Monte-Carlo tree search improved for classical planning
# model dict with 'policy', 'heuristic', 'value'
def search_mcts_improved(env, model={}, cutoff=False, lr=1.0, prune=True):
    class State:
        ACTIONS = 4
        REGISTRY = {}

        def __init__(self, env, parent, reward, terminal=False, regID=None):
            self.env = env
            self.parent = parent
            self.terminal = terminal

            self.visits = 1
            self.value_backward = 0
            self.value_forward = reward # path reward to this state

            if not terminal:
                self.reward = [None] * State.ACTIONS # instant reward for action leading to child
                self.children = [None] * State.ACTIONS
                self.actions = [a for a in range(State.ACTIONS)]
                self.actions_selected = [True] * State.ACTIONS # action selectable on select

                if 'policy' in model: self.actions_prob = (model['policy'].predict(np.expand_dims(binary(strip(self.env))[0], axis=0))[0]).tolist()
                else: self.actions_prob = [1] * State.ACTIONS

                if 'heuristic' in model: self.heuristic = model['heuristic'].predict(np.expand_dims(binary(strip(self.env))[0], axis=0))[0][0]
                else: self.heuristic = 20 # default rollout depth

            # add to registry for unique states
            self.regID = hash(str(strip(env))) if regID is None else regID
            State.REGISTRY[self.regID] = self


        def prune(self, index):
            self.reward.pop(index)
            self.children.pop(index)
            self.actions.pop(index)
            self.actions_prob.pop(index)
            self.actions_selected.pop(index)

        def deselect(self, index):
            self.actions_selected[index] = False


        def action_values(self, k=0.0):
            assert(None not in self.children)
            return [k * self.reward[i] + Config.discount_tree * self.children[i].value_backward for i, x in enumerate(self.actions_selected) if x]

        def action_probs(self, e=0.25):
            assert(None not in self.children)
            values = [self.actions_prob[i] for i, x in enumerate(self.actions_selected) if x]
            values = softmax(np.array(values)) # normalize probs to remaining actions
            #if 'policy' in model: values = (1 - e) * values + e * np.random.dirichlet([10 / len(values)] * len(values)) # add dirichlet noise
            return values

        def action_count(self):
            assert(None not in self.children)
            return [self.children[i].visits for i, x in enumerate(self.actions_selected) if x]


    def select(node, back_list):
        assert(not node.terminal)

        # create all children
        if None in node.children:
            while None in node.children:
                new_state, action = expand(node, back_list)
                if new_state:
                    if new_state.terminal: return action
                    else:
                        #new_state.value_backward, rollout_path = rollout(new_state.env, depth=int(1.5 * new_state.heuristic), model=model)
                        #if rollout_path and Config.early_stop:
                        #    return [node.actions[action]] + rollout_path
                        if 'value' in model:
                            new_state.value_backward = max(model['value'].predict(np.expand_dims(binary(strip(new_state.env))[0], axis=0))[0])
                            #new_state.value_backward /= 2
                        new_state.value_backward = 1 / new_state.heuristic
            back_list.append(node)
            return False

        # all children disabled or pruned away
        if sum(node.actions_selected) == 0:
            back_list.append(node)
            return False

        # select action using UCT
        c = np.sqrt(2)
        c = 5 * node.action_probs()
        return [i for i, x in enumerate(node.actions_selected) if x][UCB1.calculate(node.action_values(), node.action_count(), c)[0]]


    def expand(node, back_list):
        index_none = [i for i, x in enumerate(node.children) if x is None]
        if 'policy' in model: # expand best predicted child first
            index_none.sort(key = lambda index: node.actions_prob[index])
            index = index_none[-1]
        else: index = random.choice(index_none) # expand random child

        env = copy.deepcopy(node.env)
        _, reward, done, info = env.step(action_translator[node.actions[index]], 'tiny_rgb_array')
        reward = Sokoban.reward(reward, info)
        node.reward[index] = reward

        newID = hash(str(strip(env)))
        if newID != node.regID or not prune: # no direct loop / no action without effect
            if newID in State.REGISTRY: # state already seen
                state = State.REGISTRY[newID]
                node.children[index] = state
                if (node.value_forward + reward) > state.value_forward: # lower path cost found
                    move(state, node, index, back_list)
                else: # disable action from current state
                    node.deselect(index)
            else:
                state = State(env, node, node.value_forward + reward, done, newID)
                node.children[index] = state
                return state, index
        else: node.prune(index) # prune direct loop
        return False, index


    # moves subtree to better node
    def move(root, dest, dest_action, back_list):
        # remove state from worse path
        assert(not root.parent.regID == dest.regID)
        for i, x in enumerate(root.parent.children):
            if x.regID == root.regID: root.parent.deselect(i)
        if root.parent not in back_list:
            back_list.append(root.parent) # add to list for backpropagation

        # move state to current path
        root.parent = dest
        root.value_forward = dest.value_forward + dest.reward[dest_action]

        # re-evaluate moved tree
        reevaluate(root, back_list)


    # update state path costs and re-evaluate de-selected actions
    def reevaluate(node, back_list):
        if None not in node.children:
            for i in range(len(node.children)):
                if (node.value_forward + node.reward[i]) > node.children[i].value_forward: # better path found
                    if not node.children[i].parent.regID == node.regID: # child in different branch
                        assert(not node.actions_selected[i])
                        node.actions_selected[i] = True # re-select
                        move(node.children[i], node, i, back_list)
                    else:
                        node.actions_selected[i] = True # re-select in case de-selected by backpropagation
                        node.children[i].value_forward = node.value_forward + node.reward[i]
                        reevaluate(node.children[i], back_list)

        else: assert(node.children.count(None) == len(node.children))


    def backpropagate(node):
        assert(not node.terminal)
        if sum(node.actions_selected) == 0:
            # disable action as all successors are worse than from other pathes
            for i, x in enumerate(node.parent.children):
                if x.regID == node.regID: node.parent.deselect(i)
        else:
            node.value_backward += lr * (max(node.action_values()) - node.value_backward)
            node.visits = sum(node.action_count())
        return node.parent


    iterations = 0
    start_node = State(copy.deepcopy(env), None, 0)
    path = rollout(start_node.env, depth=int(1.5 * start_node.heuristic), model=model)[1]
    if path: return path#, State.REGISTRY

    while True:
        iterations += 1
        if cutoff and iterations > cutoff: break

        path = []
        back_list = []

        node = start_node
        while True: # selection, expansion, simulation
            action = select(node, back_list)

            if type(action) == bool: break
            elif type(action) == list: # early stop, rollout solution
                return path + action#, State.REGISTRY

            path.append(node.actions[action])
            assert(node.actions_selected[action])
            node = node.children[action]
            if node.terminal: return path#, State.REGISTRY

        while back_list: # backpropagate
            node = back_list.pop()
            while node:
                node = backpropagate(node)
                if node in back_list: break # prohibit double backup

    return False#, State.REGISTRY


# Unit Test
if __name__ == "__main__":
    Config.early_stop = False


    # chain loop test https://arxiv.org/pdf/1805.09218.pdf
    if False: # needs changes in mcts id function
        class ChainLoop:
            def __init__(self, length=10):
                self.state = 0
                self.length = length

            def step(self, action):
                assert(action == 0 or action == 1)
                if action == 0:
                    self.state = 0 # jump to init state
                else:
                    self.state += 1
                    if self.state >= self.length:
                        return None, 1, True, None
                return None, 0, False, None

        State.ACTIONS = 2
        for i in range(1, 200): # max tree depth
            assert(search_mcts_improved(ChainLoop(i), cutoff=i)[1] == 1) # tested and passed

    # full integrity test
    if False: # needs additional State.REGISTRY return
        generated_envs = Sokoban(boxes=3, count=10, size=(7, 7))
        for i in range(len(generated_envs.all_envs)):
            env = generated_envs.get(i)[0]
            path, tree = search_mcts_improved(env, prune=True)

            print(path, len(tree))
            for key, value in tree.items():
                if not value.terminal and None not in value.children:
                    for index in range(len(value.actions)):
                        if value.actions_selected[index]: assert((value.value_forward + value.reward[index]) == value.children[index].value_forward)
                        else: assert((value.value_forward + value.reward[index]) <= value.children[index].value_forward)

                        env_c = copy.deepcopy(value.env)
                        _, reward, done, info = env_c.step(action_translator[value.actions[index]], 'tiny_rgb_array')
                        assert(Sokoban.reward(reward, info) == value.reward[index])
                        assert(hash(str(strip(env_c))) == value.children[index].regID)


    # create envs to test
    generated_envs = Sokoban(boxes=2, count=10, size=(7, 7))

    for i in range(len(generated_envs.all_envs)):
        env, ff, _ = generated_envs.get(i)
        #draw(env)

        path = search_mcts_improved(env, prune=False)
        for action in path:
            _, reward, done, _ = env.step(action_translator[action], 'tiny_rgb_array')
            #draw(env)
        assert(done)
        print('Length:', len(path) / len(ff), len(path))