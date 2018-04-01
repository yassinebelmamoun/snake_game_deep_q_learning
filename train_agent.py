import numpy as np
from random import sample

class Agent_Q_learning_reduced:
    ''' Q Learning Reduced/Simple State '''

    def __init__(self, model, batch_size=6):
        ''' Initialization '''
         self.memory         = []
         self.model          = model
         self.batch_size     = batch_size
         self.states         = None
         self.shape_of_input = (batch_size, 6)
         # Size of the reduced state space , batch_size is the number of frames of the game composing one single state


    def refresh_states(self, game):
        ''' Refresh the states '''
        current_state = game.simple_state()

        if self.states is None:
            self.states = [current_state] * self.batch_size
        else:
            self.states.append(current_state)
            self.states.pop(0)

        return np.expand_dims(self.states, axis=0)


    def refresh_NN(self, model, depth_training_network, gamma=0.9,alpha=0.9, ):
        ''' Function to train the Neural Network on a certain amount of episodes '''

        # If depth_training_network not reached yet, use all previous episodes
        if len(self.memory) < depth_training_network:
            depth_training_network = len(self.memory)

        # If depth_training_network exceeded, we select randomly "depth_training_network" amongst the history to test the NN
        epsd = np.array(sample(self.memory, depth_training_network))
        dim = np.prod(self.shape_of_input)

        # For each episod, we keep the state , action and reward corresponding
        action      = epsd[:, dim]
        state       = epsd[:, 0 : dim]
        reward      = epsd[:, dim + 1]
        next_state  = epsd[:, dim + 2 : 2 * dim + 2]
        game_over   = epsd[:, 2 * dim + 2]
        reward      = reward.repeat(4).reshape((depth_training_network, 4))
        game_over   = game_over.repeat(4).reshape((depth_training_network, 4))
        state       = state.reshape((depth_training_network, ) + self.shape_of_input)
        next_state  = next_state.reshape((depth_training_network, ) + self.shape_of_input)

        X = np.concatenate([state, next_state], axis=0) # Matrix of old and new states
        Y = model.predict(X) # Matrix of the different predictions of the next states

        # We keep the highest q value, and the action that leads to it
        max_q_value_suivante    = np.max(Y[depth_training_network:], 1).repeat(4).reshape((depth_training_network, 4))
        choosenaction           = np.zeros((depth_training_network, 4))
        action                  = np.cast['int'](action)
        choosenaction[np.arange(depth_training_network), action] = 1

        # Q-Learning Formula: off-policy TD control
        Q_Value = (1 - choosenaction) * Y[:depth_training_network] + \
                  choosenaction * \
                ( \
                 Y[:depth_training_network] + alpha * \
                 (reward + gamma * (1 - game_over) * max_q_value_suivante - Y[:depth_training_network]) \
                )

        error_train = float(self.model.train_on_batch(state, Q_Value))

        return error_train


    def train(self, game, epsilon=0.1, gamma=0.5, depth_training_network=80, alpha=0.1):
        ''' Train the agent until the end: Win or Lose in the limit of 2000 moves '''

        error_train     = 0.
        eaten_fruits    = 0
        games_won       = 0
        game.reset_game()

        self.reset_states()
        game_over   = False
        state       = self.refresh_states(game)

        i = 0
        while (not game_over) and i <= 2000:

            # Exploration
            if np.random.random() < epsilon:
                action = int(np.random.randint(4))

            # Exploitation
            else:
                q       = self.model.predict(state)
                action  = np.argmax(q[0])

            # Get the reward value
            reward = game.play_game(action)
            if reward == 20:
                eaten_fruits += 1

            # Update states
            next_state = self.refresh_states(game)

            # Check if Game is over
            game_over = game.game_over()

            if game.win():
                games_won += 1

            self.memory.append(np.concatenate([state.flatten(), np.array(action).flatten(), np.array(reward).flatten(), next_state.flatten(),
                                   1 * np.array(game_over).flatten()]))
            state       = next_state
            error_train += self.refresh_NN(model=self.model,  depth_training_network=depth_training_network,gamma=gamma,alpha=alpha)
            i += 1

        return eaten_fruits, error_train, games_won


    def reset_states(self):
        self.states = None
