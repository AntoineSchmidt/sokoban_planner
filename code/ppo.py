import numpy as np

from keras import backend as K

from utils import *
from config import *
from network import *


# Loss function for ppo
# Plus additional changes to be able to reuse the network create function and load the imitation weights
# mask and y_true are one hot encoded for the used action
# mask additionally contains the old probabilities
# y_true additionally contains the advantage
def ppo_loss(mask):
    def loss(y_true, y_pred):
        old_prob = K.sum(mask, axis=-1)
        prob = K.sum(y_pred, axis=-1) / old_prob # reverse mask multiplication
        ratio = prob / old_prob
        advantage = K.sum(y_true, axis=-1)
        return -K.mean(K.minimum(ratio * advantage, K.clip(ratio, min_value=0.8, max_value=1.2) * advantage) - 5e-3 * (prob * K.log(prob + 1e-10)))
    return loss


class PPOAgent:
    def __init__(self):
        self.actor, self.actor_masked = create(activation_out='softmax', loss=ppo_loss, lr=1e-4)
        self.critic, _ = create(shape_out=1, activation_out='linear', loss='mean_squared_error', lr=5e-5)
        self.critic_fixed = False


    def setCritic(self, critic):
        self.critic = critic
        self.critic_fixed = True


    # sample action for state
    def act(self, state, deterministic=False):
        probs = self.actor.predict(np.expand_dims(state, axis=0))[0]
        action = np.argmax(probs) if deterministic else np.argmax(np.random.choice(probs, p=probs) == probs)
        return action, probs


    # update network
    def update(self, data):
        batch_states, batch_actions, batch_probabilities, batch_rewards, _ = data

        loss_value = 0
        if self.critic: # calculate advantage
            if not self.critic_fixed: loss_value = self.critic.train_on_batch(batch_states, batch_rewards) # update value predictor
            batch_rewards -= np.average(self.critic.predict(batch_states), axis=-1).reshape((len(batch_states),))

        # one hot transformation
        probabilities_hot = batch_actions * batch_probabilities
        advantage_hot = (batch_actions.T * batch_rewards).T

        # update policy network
        loss_policy = self.actor_masked.train_on_batch([batch_states, probabilities_hot], advantage_hot)

        return loss_value, loss_policy


    # load trained model
    def load(self, path):
        self.actor.load_weights(path)

    # save trained model
    def save(self, path):
        self.actor.save_weights(path)


# runs a ppo episode
def ppo_episode(env, agent, replay, evaluation=False, max_steps=100):
    assert not replay.augment # on-policy

    states = []
    actions = []
    actions_probs = []
    rewards = []

    augmented = not evaluation

    r = np.random.randint(8) if augmented else 0
    state, action_augmented, _ = augment(strip(env), np.arange(4), r=r)
    state_binary, placement = binary(state, shrink=augmented, random=augmented)

    for step in range(max_steps):
        action, probs = agent.act(state_binary, evaluation)
        if evaluation and not Config.silent: print(probs, list(action_dict.keys())[action])

        _, reward, done, info = env.step(action_translator[action_augmented[action]], 'tiny_rgb_array')
        reward = Sokoban.reward(reward, info)

        # save observations
        states.append(state_binary)
        actions.append(np.eye(4)[action])
        actions_probs.append(probs)
        rewards.append(reward)

        if done:
            #if not Config.silent: print('solved')
            break

        state = augment(strip(env), r=r)[0]
        state_binary = binary(state, shrink=augmented, placement=placement)[0]

    # bracktrack rewards and save to replay
    if not evaluation:
        for s in range(len(rewards)):
            reward = sum([r*(Config.discount**j) for j,r in enumerate(rewards[s:])])
            replay.add(states[s], actions[s], actions_probs[s], reward, done)

    return sum(rewards), done, step + 1


# Unit Test
if __name__ == "__main__":
    from replay import *
    Config.silent = False

    generated_envs = Sokoban(count=1, boxes=1, size=(5, 5))
    env, plan, _ = generated_envs.get()
    draw(env, 'ppo.png')

    agent = PPOAgent()
    replay = Replay(augment=False)

    episode = 0
    while True:
        evaluation = (episode % 10 == 0)
        steps = 2 * len(plan) if evaluation else int(len(plan) * 1.5)

        total_reward, done, steps = ppo_episode(copy.deepcopy(env), agent, replay, evaluation=evaluation, max_steps=steps)

        if episode > 0 and episode % 5 == 0: # update policy
            for _ in range(5): # batches
                data = replay.sample()
                loss_value, loss_policy = agent.update(data)
            replay.reset()

        if evaluation: print(episode, steps, total_reward, steps / len(plan))
        episode += 1