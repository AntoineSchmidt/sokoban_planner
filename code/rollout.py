import copy

from utils import *
from config import *


def rollout(env, repeat=1, depth=20, model={}, actions=4):
    if len(model) == 0: return 0, False # unguided rollout too noisy, worsening the mcts performance

    rollout_rewards = []
    for i in range(repeat):
        path = []
        trace = []
        rollout_rewards.append(0)
        env_c = copy.deepcopy(env)

        for j in range(depth):
            env_h = hash(str(strip(env_c)))
            if env_h in trace: break
            else: trace.append(env_h)

            if 'policy' in model or 'value' in model:
                env_b = binary(strip(env_c))[0]
                probs = np.zeros((actions,))

                if 'policy' in model: probs += model['policy'].predict(np.expand_dims(env_b, axis=0))[0]
                if 'value' in model: probs += softmax(model['value'].predict(np.expand_dims(env_b, axis=0))[0])

                if i > 0:
                    probs /= sum(probs) # gpu rounding accuracy and possibly both networks
                    action = np.argmax(np.random.choice(probs, p=probs) == probs) # sample action
                else: action = np.argmax(probs) # greedy
            else: action = np.random.randint(actions) # random action

            _, reward, done, info = env_c.step(action_translator[action], 'tiny_rgb_array')
            path.append(action)
            rollout_rewards[-1] += Sokoban.reward(reward, info) * (Config.discount**j)
            if done: return rollout_rewards[-1], path

    #return np.average(rollout_rewards), False
    return 0, False