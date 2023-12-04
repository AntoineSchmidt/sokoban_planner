import numpy as np

from utils import *
from config import *
from network import *
from exploration import *


class DQNAgent:
    def __init__(self, exploration=Boltzmann(tau=0.25)):
        self.Q, self.Q_masked = create(activation_out='linear', loss='mean_squared_error', lr=1e-4)
        self.Q_target, _ = create(activation_out='linear')
        approximate(self.Q, self.Q_target, 1.0) # clone weights

        self.exploration = exploration


    # sample action for state
    def act(self, state, deterministic=False):
        q_values = self.Q.predict(np.expand_dims(state, axis=0))[0]
        action = np.argmax(q_values) if deterministic else self.exploration.act(state, q_values)
        return action, q_values


    # update network
    def update(self, data):
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = data

        q_actions = np.argmax(self.Q.predict(batch_next_states), axis=1) # best action in next_state
        q_values = self.Q_target.predict(batch_next_states)[np.arange(len(batch_states)), q_actions] # action value next_state
        batch_rewards[np.logical_not(batch_dones)] += Config.discount * q_values[np.logical_not(batch_dones)]
        batch_rewards = (batch_actions.T * batch_rewards).T # one hot encode (loss is masked)

        # update network and target network
        loss = self.Q_masked.train_on_batch([batch_states, batch_actions], batch_rewards)
        approximate(self.Q, self.Q_target)

        return loss


    # load trained model
    def load(self, path):
        self.Q.load_weights(path)
        approximate(self.Q, self.Q_target, 1.0) # clone weights

    # save trained model
    def save(self, path):
        self.Q.save_weights(path)


# runs a dqn episode
def dqn_episode(env, agent, replay, path=None, evaluation=False, train=True, max_steps=100):
    if path is not None: max_steps = len(path)

    total_loss = 0
    total_reward = 0
    state_stripped = strip(env)
    for step in range(max_steps):
        if path is not None:
            action = path[step]
        else:
            action, q_values = agent.act(binary(state_stripped)[0], evaluation) # predict on unchanged state
            if evaluation and not Config.silent: print(q_values, list(action_dict.keys())[np.argmax(q_values)])

        _, reward, done, info = env.step(action_translator[action], 'tiny_rgb_array')
        reward = Sokoban.reward(reward, info)

        state_stripped_next = strip(env)

        if not evaluation:
            action_hot = np.eye(4)[action]
            if replay.augment: replay.add(state_stripped, action_hot, state_stripped_next, reward, done)
            else: replay.add(binary(state_stripped)[0], action_hot, binary(state_stripped_next)[0], reward, done)
            if train: total_loss += agent.update(replay.sample())

        total_reward += reward
        state_stripped = state_stripped_next

        if done:
            #if not Config.silent: print('solved')
            break

    if path is not None and not done: print('not solution path')
    return total_reward, done, step+1, total_loss / (step + 1)


# Unit Test
if __name__ == "__main__":
    from replay import *
    Config.silent = False

    generated_envs = Sokoban(count=1, boxes=1, size=(5, 5))
    env, plan, _ = generated_envs.get()
    draw(env, 'dqn.png')

    agent = DQNAgent()
    replay = Replay()

    episode = 0
    while True:
        evaluation = (episode % 10 == 0)
        steps = 2 * len(plan) if evaluation else int(len(plan) * 1.5)

        total_reward, done, steps, total_loss = dqn_episode(copy.deepcopy(env), agent, replay, evaluation=evaluation, max_steps=steps)

        # add ff transitions
        if not done and evaluation: dqn_episode(copy.deepcopy(env), agent, replay, path=plan)

        if evaluation: print(episode, steps, total_reward, steps / len(plan), replay.full())
        episode += 1