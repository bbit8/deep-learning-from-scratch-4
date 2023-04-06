import numpy as np
import matplotlib.pyplot as plt
from bandit import Agent

class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)  # Add noise
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha
        self.n = np.ones(actions)

        self.ts_alpha = np.ones(actions)
        self.ts_beta = np.ones(actions)
        self.actions = actions


    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha
        if reward > 0:
            self.ts_alpha[action] += 1
        else:
            self.ts_beta[action] += 1

    def get_action_(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)

    def get_action(self):
        ucbs = self.Qs + np.sqrt(2 * np.log(np.sum(self.n)) / self.n)
        action = np.argmax(ucbs)
        self.n[action] += 1
        return action

    def get_action_ts(self):
        samples = [np.random.beta(self.ts_alpha[i] + 1, self.ts_beta[i] + 1) for i in range(self.actions)]
        action = np.argmax(samples)
        return action


runs = 200
steps = 1000
epsilon = 0.1
alpha = 0.8
agent_types = ['sample average', 'alpha const update']
results = {}

for agent_type in agent_types:
    all_rates = np.zeros((runs, steps))  # (200, 1000)

    for run in range(runs):
        if agent_type == 'sample average':
            agent = Agent(epsilon)
        else:
            agent = AlphaAgent(epsilon, alpha)

        bandit = NonStatBandit()
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)
    results[agent_type] = avg_rates

# plot
plt.figure()
plt.ylabel('Average Rates')
plt.xlabel('Steps')
for key, avg_rates in results.items():
    plt.plot(avg_rates, label=key)
plt.legend()
plt.show()