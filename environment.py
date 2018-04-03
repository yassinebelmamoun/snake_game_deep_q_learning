import numpy as np


actions = {
    0: 'left',
    1: 'right',
    2: 'up',
    3: 'down'
}

# forbiden moves (ex: if going to the right, can not go the left)
forbiden = [
    (2, 3),
    (3, 2),
    (0, 1),
    (1, 0)
]


class Snake_Walls:
    ''' Snake Walls '''

    def __init__(self, field=6, length_snake=2):
        ''' Initialization '''

        self.field = field
        self.length_snake = length_snake
        self.reset_game()


    def play_game(self, action):
        ''' Make the snake move once and reward it '''

        # One move for the snake
        self.move_snake(action)

        # POSITIVE REWARD: The fruit is eaten, then make another one appear on the field
        if self.fruit == self.snake[0]:
            reward = 20
            self.new_fruit()

        # NEGATIVE REWARD: The snake hits his tail or the walls
        elif self.game_over():
            reward = -20

        # (small) NEGATIVE REWARD: Nothing happens & This to incentivize the snake to eat faster
        else:
            reward = -1

        return reward


    def new_fruit(self):
        ''' Make a new fruit appear '''
        if len(self.snake) >= (self.field - 2) ** 2:
            self.fruit = (0, 0)
        else:
            # Make sure the fruit appear out of the snake body
            while True:
                fruit = np.random.randint(1, self.field - 1, 2)
                fruit = (fruit[0], fruit[1])
                if fruit in self.snake:
                    # Case 1: The snake occupy the position
                    continue
                else:
                    # Case 2: The position is free
                    self.fruit = fruit
                    break

    def move_snake(self, action):
        ''' Make the snake move in one of the possible directions, and keep going if the direction is forbiden '''
        if (action, self.prev_action) in forbiden:
            action = self.prev_action
        else:
            self.prev_action = action

        head = self.snake[0]

        # Go left:
        if action == 0:
            p = (head[0] - 1, head[1])
        # Go right:
        elif action == 1:
            p = (head[0] + 1, head[1])
        # Go down:
        elif action == 2:
            p = (head[0], head[1] - 1)
        # Go down:
        elif action == 3:
            p = (head[0], head[1] + 1)

        self.snake.insert(0, p)

        if self.fruit != self.snake[0]:
            self.snake.pop()


    def win(self):
        ''' Win scenario : the snake fills all the fields '''
        return len(self.snake) == (self.field - 2) ** 2


    def game_over(self):
        ''' Lost (game over) scenario : The snake hits a wall or his tail '''
        is_hit_wall = self.snake[0] in self.walls
        is_hit_tail = (len(self.snake) > len(set(self.snake)))
        return (is_hit_wall or is_hit_tail)


    def reset_game(self):
        ''' Reset game '''
        field = self.field
        length_snake = self.length_snake
        head_x = (field - length_snake) // 2

        # TODO: Improve the initialization of the snake position
        self.snake = [(x,field // 2) for x in range(head_x + 1, head_x + length_snake + 1)]
        self.new_fruit()

        if 2 + np.random.randint(2) == 2:
            self.prev_action = 2
        else:
            self.prev_action = 3
            self.snake.reverse()

        self.walls = []
        for z in range(field):
            self.walls += [(z, 0), (z, field - 1), (0, z), (field - 1, z)]


    def full_state(self):
        ''' Drawing the snake/wall with numpy Array '''

        position = np.ones((self.field,) * 2)
        position[1:-1, 1:-1] = 0

        # Snake position
        for e in self.snake:
            position[e[0], e[1]] = 8

        # Fruit Position
        position[self.fruit[0], self.fruit[1]] = 4
        print(position)
        return position


    def simple_state(self):
        ''' Return the reduced state with relative values '''

        # Relative position to the Wall:
        relative_wall_left  = 0
        relative_wall_right = 0
        relative_wall_up    = 0
        relative_wall_down  = 0

        head        = self.snake[0]
        fruit       = self.fruit
        position    = self.full_state()

        if (head[0] not in (0, self.field - 1)) and (head[1] not in (0, self.field - 1)):

            # Measure the distance from the head to left wall
            i = 1
            while position[head[0] - i, head[1]] not in (1, 8):
                i += 1
                relative_wall_up += 1

            # Measure the distance from the head to right wall
            j = 1
            while position[head[0] + j, head[1]] not in (1, 8):
                j += 1
                relative_wall_down += 1

            # Measure the distance from the head to bottom wall
            k = 1
            while position[head[0], head[1] - k] not in (1, 8):
                k += 1
                relative_wall_left += 1

            # Measure the distance from the head to top wall
            l = 1
            while position[head[0], head[1] + l] not in (1, 8):
                l += 1
                relative_wall_right += 1
        # TODO: else Raise error / If no error raised: Remove the if condition

        relative_fruit_x = fruit[0] - head[0]
        relative_fruit_y = fruit[1] - head[1]

        # TODO: Fix the confusion between up+down and left+right (Currently, this work, DO NOT TOUCH!)
        state = [relative_wall_up, relative_wall_down, relative_wall_left, relative_wall_right, relative_fruit_x, relative_fruit_y]

        return np.array(state)
