import numpy as np


class ArmedBandit:
    def __init__(self, k, epsilon, true_means):
        """__init__ Instantiates a stationary, normally distributed
         k armed bandit problem

        Parameters
        ----------
        k : int
            the number of arms (or actions) available in the problem
        epsilon : float
            the probability of choosing a random action instead of
            greedy selection
        true_means : array[float]
            the true mean of each action's normal distribution
        iterations : int
            the number of iterations
        """
        self.k = k
        self.epsilon = epsilon
        self.option_estimates = np.zeros(k)
        self.true_means = true_means
        self.step_counters = np.zeros(k)

    def get_action(self):
        rand = np.random.uniform()
        if rand > self.epsilon:
            return np.argmax(self.option_estimates)
        else:
            return np.random.choice(self.k)

    def get_reward(self, action):
        print(action)
        return np.random.normal(self.true_means[action])

    def iterate(self, iterations):
        """iterate: iterates the specified number of times over the
        bandit problem and returns the accumulated reward for each
        time step

        Parameters
        ----------
        iterations: int
            number of iterations to be carried out

        Returns
        -------
        array[float]
            array representation of the total reward earned for
            each step of the problem
        """
        total_reward = []
        accumulated_reward = 0
        while iterations > 0:
            action = self.get_action()
            reward = self.get_reward(action)
            accumulated_reward += reward
            total_reward.append(accumulated_reward)
            self.step_counters[action] += 1
            old_estimate = self.option_estimates[action]
            error = (reward - old_estimate)/self.step_counters[action]
            self.option_estimates[action] = old_estimate + error
            iterations -= 1
        return total_reward
