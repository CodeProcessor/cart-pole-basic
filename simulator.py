import gym

class Simulator:
    def __init__(self, render = True):
        self.env = gym.make("CartPole-v0").env
        self.observation = self.env.reset()
        self.fitness = 0
        self.done = False
        self.render =render

    def reset(self):
        self.observation = self.env.reset()
        self.fitness = 0
        self.done = False

    def get_obs(self):
        return self.observation

    def get_status(self):
        return self.done

    def next_step(self, action):
        self.fitness += 1
        if self.render:
            self.env.render()
        self.observation, reward, self.done, info = self.env.step(action)

    def get_fitness(self):
        return self.fitness