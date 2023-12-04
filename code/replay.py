import numpy as np

from utils import *


# handels the observed tuples
class Replay:
    # setup buffer and maximum size
    def __init__(self, size=5e4, augment=True):
        self.size = size
        self.augment = augment
        self.reset()


    def reset(self):
        self.index = 0
        self.states = []
        self.actions = []
        self.next_states=[]
        self.rewards = []
        self.dones = []


    def full(self):
        return len(self.states) / self.size


    # add observed transition to buffer
    def add(self, state, action, next_state=None, reward=None, done=None):
        if int(self.full()): # fifo
            self.states[self.index] = state
            self.actions[self.index] = action
            self.next_states[self.index] = next_state
            self.rewards[self.index] = reward
            self.dones[self.index] = done
            self.index = int((self.index + 1) % self.size)
        else:
            self.states.append(state)
            self.actions.append(action)
            self.next_states.append(next_state)
            self.rewards.append(reward)
            self.dones.append(done)


    # samples a random batch of transitions
    def sample(self, batch_size=64):
        if batch_size < 0: # take all
            batch_indices = np.arange(len(self.states))
            np.random.shuffle(batch_indices)
        else: batch_indices = np.random.choice(len(self.states), batch_size)

        if self.augment:
            batch_states = []
            batch_actions = []
            batch_next_states = []
            for i in batch_indices:
                state, action, next_state = augment(self.states[i], self.actions[i], self.next_states[i])
                state, placement = binary(state, shrink=True, random=True)
                if next_state is not None: next_state = binary(next_state, shrink=True, placement=placement)[0]
                batch_states.append(state)
                batch_actions.append(action)
                batch_next_states.append(next_state)
        else:
            batch_states = [self.states[i] for i in batch_indices]
            batch_actions = [self.actions[i] for i in batch_indices]
            batch_next_states = [self.next_states[i] for i in batch_indices]
        batch_rewards = [self.rewards[i] for i in batch_indices]
        batch_dones = [self.dones[i] for i in batch_indices]

        return np.array(batch_states), np.array(batch_actions), np.array(batch_next_states),\
               np.array(batch_rewards).astype(np.float32), np.array(batch_dones)