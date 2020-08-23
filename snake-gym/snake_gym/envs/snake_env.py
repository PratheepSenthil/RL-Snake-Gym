

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import math
from collections import deque


class SnakeAction(object):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class BoardColor(object):
    HEAD_COLOR = np.array([255, 0, 0], dtype=np.uint8)
    BODY_COLOR = np.array([0, 0, 0], dtype=np.uint8)
    FOOD_COLOR = np.array([0, 255, 0], dtype=np.uint8)
    SPACE_COLOR = np.array([255,255,255], dtype=np.uint8)

def getVectors(observation,n):
    # print(len(observation))
    # print(observation)
    head = {}
    food = {}
    vector = []
    hor = 1
    ver = 1
    flag = False
    for i in range(len(observation)):   
        for j in range(len(observation[i])):
            if(observation[i][j]==1):
                head["x"] = j
                head["y"] = i
                flag = True
            elif(observation[i][j]==3):
                food["x"] = j
                food["y"] = i
    
    mat = [[4 for x in range(2*n+1)] for y in range(2*n+1)] 

    if(flag):
        x1 = head["x"]-n
        # x2 = head["x"]+3
        y1 = head["y"]-n
        # y2 = head["y"]+3 
        x = x1
        for i in range(len(mat)):
            if(x<0 or x>len(observation)-1):
                x+=1
            else:
                y = y1
                for j in range(len(mat[i])):
                    if(y<0 or y>len(observation[i])-1):
                        y+=1
                    else:
                        mat[j][i] = observation[y][x]
                        y+=1
                x+=1
        hor = head["x"]-food["x"]
        ver = head["y"]-food["y"]
    
    else:
        hor = -21
        ver = -21
    # print(hor,ver)
    state1 = np.array(mat).flatten()
    state1 = np.append(state1, [hor,ver])
    # state1 = np.array(observation).flatten()
    # print(state1)
    return state1    



class SnakeEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 50
    }

    def __init__(self, observation_mode=None, energy_consum=False):
        self.observation_mode = observation_mode
        self.energy_consum = energy_consum
        self.width = 10
        self.height = 10
        self.vision = 3

        self.action_space = spaces.Discrete(4)

        if observation_mode == 'rgb':
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.width * 20, self.height * 20, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=2, shape=(self.width, self.height, 1), dtype=np.uint8)

        self.snake = Snake()
        self.foods = []
        self.n_foods = 1
        self.viewer = None
        self.np_random = np.random
    
    

    def set_foods(self, n):
        self.n_foods = n

    def reset(self):
        self.snake.body.clear()
        self.foods.clear()
        empty_cells = self.get_empty_cells()
        empty_cells = self.snake.init(empty_cells, self.np_random)
        self.foods = [empty_cells[i] for i in self.np_random.choice(len(empty_cells), self.n_foods, replace=False)]
        return self.get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        snake_tail = self.snake.step(action)
        
        self.snake.reward = 0.

        if self.energy_consum:
            self.snake.reward -= 0.01

        if self.snake.head in self.foods:
            self.snake.reward += 1.
            self.snake.body.append(snake_tail)
            self.foods.remove(self.snake.head)
            empty_cells = self.get_empty_cells()
            food = empty_cells[self.np_random.choice(len(empty_cells))]
            self.foods.append(food)
        
        #snake collided wall
        if self.is_collided_wall(self.snake.head):
            self.snake.reward -= 1.
            self.snake.done = True
        
        #snake bite itself 
        if self.snake.head in list(self.snake.body)[1:]:
            self.snake.reward -= 1.
            self.snake.done = True
        
        self.snake.reward = np.clip(self.snake.reward, -1., 1.)
        obs = self.get_observation()
        #   print(obs)
        sum = obs[-1]**2 + obs[-2]**2
        # print(obs[-1],obs[-2],sum)
        if self.snake.done:
            self.snake.reward = -1000
        elif self.snake.reward == 1:
            self.snake.reward = 100
        else: 
            self.snake.reward = int(-1*math.sqrt(sum))

        return obs, self.snake.reward, self.snake.done, {}

    def get_observation(self):
        if self.observation_mode == 'rgb':
            return self.get_image()
        else:
            observation = np.zeros((self.width, self.height), dtype=np.uint8)

            for x, y in self.snake.body:
                try:
                    observation[x][y] = 2
                except:
                    pass

            x, y = self.snake.head
            try:
                observation[x][y] = 1
            except:
                pass
            
            for food in self.foods:
                x, y = food
                observation[x][y] = 3
            # print("1------------:",observation[:,:,None])
            # print("2------------:",observation[:,:])
            # print(len(observation[0]))
            # return observation[:, :, None]
            return getVectors(observation[:,:],self.vision)

    def get_image(self):
        board_width = 20 * self.width
        board_height = 20 * self.height
        cell_size = 20

        board = Board(board_height, board_width)
        for x, y in self.snake.body:
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.BODY_COLOR)
        
        x,y = self.snake.head
        board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.HEAD_COLOR)

        for food in self.foods:
            x, y = food
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.FOOD_COLOR)
        return board.board

    def get_empty_cells(self):
        empty_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        for cell in self.snake.body:
            if cell in empty_cells:
                empty_cells.remove(cell)
        for food in self.foods:
            if food in empty_cells:
                empty_cells.remove(food)
        return empty_cells

    def is_collided_wall(self, head):
        x, y = head
        if x < 0 or x > (self.width - 1) or y < 0 or y > (self.height - 1):
            return True
        return False

    def render(self, mode='human'):
        img = self.get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class Snake(object):

    def __init__(self):
        self.body = deque()
        self.prev_act = None
        self.done = False
        self.reward = 0.
        
    def step(self, action):
        if not self.done:
            if not self.is_valid_action(action):
                action = self.prev_act
            self.prev_act = action
            x, y = self.head
            if action == SnakeAction.LEFT:
                self.body.appendleft((x, y - 1))
            if action == SnakeAction.RIGHT:
                self.body.appendleft((x, y + 1))
            if action == SnakeAction.UP:
                self.body.appendleft((x - 1, y))
            if action == SnakeAction.DOWN:
                self.body.appendleft((x + 1, y))
            return self.body.pop()

    @property
    def head(self):
        return self.body[0]

    def is_valid_action(self, action):
        if len(self.body) == 1:
            return True
        
        horizontal_actions = [SnakeAction.LEFT, SnakeAction.RIGHT]
        vertical_actions = [SnakeAction.UP, SnakeAction.DOWN]

        if self.prev_act in horizontal_actions:
            return action in vertical_actions
        return action in horizontal_actions

    def init(self, empty_cells, np_random):
        self.body.clear()
        self.done = False
        self.reward = 0.
        self.prev_act = None
        start_head = empty_cells[np_random.choice(len(empty_cells))]
        self.body.appendleft(start_head)
        empty_cells.remove(start_head)
        return empty_cells


class Board(object):

    def __init__(self, height, weight):
        self.board = np.empty((height, weight, 3), dtype=np.uint8)
        self.board[:, :, :] = BoardColor.SPACE_COLOR

    def fill_cell(self, vertex, cell_size, color):
        x, y = vertex
        self.board[x:x+cell_size, y:y+cell_size, :] = color