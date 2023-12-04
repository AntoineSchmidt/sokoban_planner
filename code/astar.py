import gym
import copy
import numpy as np

from utils import *
from config import *


# A* tree search
def search_astar(env, model={}, cutoff=False, w=1.0):
    heuristic = 0
    if 'heuristic' in model:
        env_b = np.expand_dims(binary(strip(env))[0], axis=0)
        heuristic = model['heuristic'].predict(env_b)[0][0]

    start = hash(str(strip(env)))
    best = {start: [env, False, [], heuristic]} # env, done, best path, heuristic
    queue = [start] # search queue

    def add_state(child, path, done):
        child_hash = hash(str(strip(child)))
        if child_hash not in best: # new state
            heuristic = 0
            if 'heuristic' in model:
                child_b = np.expand_dims(binary(strip(child))[0], axis=0)
                heuristic = model['heuristic'].predict(child_b)[0][0]

            best[child_hash] = [child, done, path, heuristic]
            queue.append(child_hash)
            return True
        else: # already seen
            best_path = best[child_hash][2]
            if len(path) < len(best_path): # improve path
                for j in best.keys():
                    if np.all(best_path == best[j][2][:len(best_path)]): # shorten all matching longer pathes
                        best[j][2] = path + best[j][2][len(best_path):]
        return False

    iterations = 0
    while queue:
        iterations += 1
        if cutoff and iterations > cutoff: break

        queue.sort(key = lambda entry: len(best[entry][2]) + w * best[entry][3])
        parent = best[queue.pop(0)]

        if parent[1]: return parent[2] # goal reached

        if 'policy' in model: # greedy policy exploration
            path = parent[2].copy()
            child = copy.deepcopy(parent[0])

            for _ in range(int(1.5 * parent[3])):
                child_b = np.expand_dims(binary(strip(child))[0], axis=0)
                policy = model['policy'].predict(child_b)[0]
                action = np.argmax(policy) # best predicted action
                done = child.step(action_translator[action], 'tiny_rgb_array')[2]
                path += [action]
                if done and Config.early_stop: return path
                if not add_state(copy.deepcopy(child), path.copy(), done): break # state already seen
                #if -sum(policy * np.log(policy + 1e-10)) > 0.8: break # entropy as certainty measurement

        for i in range(4): # add unseen child states to queue
            child = copy.deepcopy(parent[0])
            done = child.step(action_translator[i], 'tiny_rgb_array')[2]

            path = parent[2].copy() + [i]
            if done and Config.early_stop: return path
            add_state(child, path, done)

    return False


# Unit Test
if __name__ == "__main__":
    Config.early_stop = False

    # create envs to test
    generated_envs = Sokoban(boxes=3, count=10, size=(7, 7))

    for i in range(len(generated_envs.all_envs)):
        env, ff, _ = generated_envs.get(i)
        #draw(env)

        path = search_astar(env, {'heuristic': Sokoban})
        for action in path:
            _, reward, done, _ = env.step(action_translator[action], 'tiny_rgb_array')
            #draw(env)
        assert(done and len(path) / len(ff) <= 1)
        print('Length:', len(path) / len(ff), len(path))