'''
Snake game with Deep Q learning
@author: Yassine Belmamoun & Mohamed Amine Sekkat
'''

from keras.models import Sequential
from keras import backend as Kbackend
from keras.layers import *
from keras.optimizers import *
from test_agent import Agent_Test
from train_agent import Agent_Q_learning_reduced
from environment import Snake_Walls
import matplotlib.pyplot as plt
Kbackend.set_image_dim_ordering('th')


field = 6
length_snake = 2
batch_size = 6
depth_training_network = 80
epsilon = [0.0, 1.0]
N = 10 # number of simultaneosly playing agents
iters = 2000 # number of games played by each agent
nb_hidden_layers = 4
alpha = 0.1
gamma = [0.8, 0.9]
param_snake = {
    'field': field,
    'length_snake' : length_snake,
}
param_train = {
    'depth_training_network': depth_training_network,
    'alpha': alpha
}

# Neural Network
neural_network = Sequential()

# Inputs of the network : last states for the current game
neural_network.add(Conv1D(nb_hidden_layers, 3, activation='relu', input_shape=(batch_size, 6)))
neural_network.add(Flatten())
neural_network.add(Dense(256, activation='relu'))
neural_network.add(Dense(4))
neural_network.compile(RMSprop(), 'MSE')
# Output : 4 , one for each possible action

param_agent_simplifie = {
    'model':neural_network,
    'batch_size':batch_size
}
tester = Agent_Test(
    Snake_Walls,
    Agent_Q_learning_reduced,
    param_snake,
    param_agent_simplifie,
    N=N,
    iters=iters
)
reward_mean, optimality_mean, error_mean, total_of_games_won\
        = tester.test(param_train, epsilon, gamma)

print('========> Number of games won : {} / {}'.format(total_of_games_won.sum(), N * iters))


# Plot: # Eaten fruits
plt.figure(1)
plt.plot(reward_mean)
plt.xlabel('Epoch')
plt.ylabel('Reward Mean')

# Plot: % success
plt.figure(2)
plt.xlabel('Epoch')
plt.ylabel('Optimality')
plt.plot(optimality_mean)
plt.show()

# Plot: Error of training
plt.figure(3)
plt.xlabel('Epoch')
plt.ylabel('Error Mean')
plt.plot(error_mean)
plt.show()

















