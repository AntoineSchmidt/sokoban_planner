import numpy as np

from utils import *


# different exploration strategies

# e-greedy exploration
class EGreedy:
    def __init__(self, epsilon=1.0, epsilon_min=0.1, epsilon_decay=2e-3):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def reset(self):
        pass

    def act(self, state, prediction):
        r = np.random.uniform()
        return np.argmax(prediction) if r > self.epsilon else np.random.randint(0, len(prediction))

    # anneal exploration parameter
    def anneal(self, e=0):
        self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay)) # linear
        #self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-self.epsilon_decay * e))

    def info(self):
        return self.epsilon


# selects action following probabilities
class Boltzmann:
    def __init__(self, tau=1.0, tau_min=0.1, tau_decay=2e-3):
        self.tau = tau
        self.tau_min = tau_min
        self.tau_decay = tau_decay

    def reset(self):
        pass

    def act(self, state, prediction):
        probs = softmax(prediction, self.tau)
        return np.argmax(np.random.choice(probs, p=probs) == probs)

    # anneal exploration parameter
    def anneal(self, e=0):
        self.tau = max(self.tau_min, self.tau * (1 - self.tau_decay)) # linear
        #self.tau = max(self.tau_min, self.tau * np.exp(-self.tau_decay * e))

    def info(self):
        return self.tau


# selects action following UCB1
class UCB1:
    def calculate(values, count, c=np.sqrt(2)):
        values = np.array(values)
        count = np.array(count)

        if 0 in count: # enforce at least one execution
            values[count == 0] = np.inf
        else:
            values = (values - min(values)) / (max(values) - min(values) + 1e-10) # normalize values to [0-1]
            values += c * np.sqrt(np.log(sum(count)) / count)

        action = np.random.choice(np.arange(len(values))[values == max(values)]) # randomly select max value
        return action, values

    def __init__(self, c=np.sqrt(2)):
        self.c = c
        self.reset()

    def reset(self):
        self.states = []
        self.counter = []

    def act(self, state, prediction):
        state_hash = hash(str(state))
        state_index = len(self.states)
        if state_hash not in self.states:
            self.states.append(state_hash)
            self.counter.append([1] * len(prediction))
        else:
            state_index = self.states.index(state_hash)

        action = UCB1.calculate(prediction, self.counter[state_index], self.c)[0]
        self.counter[state_index][action] += 1
        return action

    def anneal(self, e=0):
        pass

    def info(self):
        return len(self.states)