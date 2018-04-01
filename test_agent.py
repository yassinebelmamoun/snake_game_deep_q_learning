import sys
import numpy as np

class Agent_Test:
    ''' Test the agent '''

    def __init__(self, snake, agent, param_snake, param_agent, N, iters):
        ''' Initialization '''

        self.iters      = iters
        self.N          = N
        self.agent      = agent
        self.snake      = snake

        self.snakeTable = []
        for i in range(N):
            self.snakeTable[len(self.snakeTable):] = [snake(**param_snake)]

        self.agentTable = []
        for i in range(N):
            self.agentTable[len(self.agentTable):] = [agent(**param_agent)]

        #TODO: Replace with # field value - initial size
        self.optimal = 18 # 4*5-2 maximum number of fruits the snake can eat


    def oneStep(self, param_train, epsilon, gamma):
        ''' N number of agents to test simulatenosly '''

        eaten_fruits    = np.zeros(self.N) # Count of eaten fruits
        error_train     = np.zeros(self.N) # Error in the train
        optimality      = np.zeros(self.N) # Optimality
        games_won       = np.zeros(self.N) # Count games won (Eat all fruits)

        for i in range(self.N):

            jeu     = self.snakeTable[i]
            agent   = self.agentTable[i]

            eaten_fruits[i], error_train[i], games_won[i] = agent.train(jeu, epsilon, gamma, **param_train)
            optimality[i] = eaten_fruits[i] / self.optimal

        return eaten_fruits.mean(), optimality.mean() * 100, error_train.mean(), games_won.sum()


    def test(self, param_train, epsilon, gamma):
        ''' Test the agent '''

        # Make epsilon decrease over time
        delta_epsilon   = 5/4 * (epsilon[1] - epsilon[0]) / self.iters
        last_epsilon    = epsilon[1]
        eps             = epsilon[0]

        # Make gamma increase
        delta_gamma     = 5/4 * (gamma[1] - gamma[0]) / self.iters
        last_gamma      = epsilon[1]
        gam             = gamma[0]

        # Mean values for KPI
        reward_mean         = np.zeros(self.iters)
        optimality_mean     = np.zeros(self.iters)
        error_mean          = np.zeros(self.iters)
        total_of_games_won  = np.zeros(self.iters)

        for i in range(self.iters):
            reward_mean[i], optimality_mean[i], error_mean[i], total_of_games_won[i] = self.oneStep(param_train, eps, gam)
            display = '\r-- Epoch: {0}\n -- Average_reward: {1}\n -- optimality_average: {2}\n -- training_error: {3}\n -- number_of_games_won: {4}\n'
            sys.stdout.write(display.format(i, reward_mean[i], optimality_mean[i], error_mean[i], total_of_games_won[i]))
            sys.stdout.flush()

            if eps > last_epsilon:
                eps -= delta_epsilon
            if gam < last_gamma:
                gam += delta_gamma

        return reward_mean, optimality_mean, error_mean, total_of_games_won
