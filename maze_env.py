import numpy as np
import time
import sys

#If using python 2.7 or say python 2
#Then "import Tkinter as tk"
#importing the appropriate tkinter package based on your python version
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

#Defining a unit of 40 pixels
UNIT = 40 
#Defining maze height and maze width 
MAZE_H = 6  
MAZE_W = 6 

class Maze():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('maze')
        self.window.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.build_grid()

    def build_grid(self):
        #For creating the maze first we need to create a canvas
        self.canvas = tk.Canvas(self.window, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        #Creating grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin point(It is the center of the first cell in the first row)
        origin = np.array([20, 20])

        #Creating 2 hell points
        #The block point(or sya the hell point)
        #Hell 1 or block 1
        #Two unit to the right and one unit to down
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        #Hell 2
        #Two unit to the down and one unit to right
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        #Setting the goal
        #Creating the oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        #Creating the red rect(the agent)
        #The explorer setting
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        #Packing all the widgets together
        self.canvas.pack()

    def render(self):
        time.sleep(0.1)
        self.window.update()
    #Reset the explorer agent at the origin position,return the canvas with the explorer agent at the origin position
    def reset(self):
        #To initially withdraw the widgets
        self.window.update()
        time.sleep(0.5)
        #Deleting the red rectangle
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        #Creating the rectangle again
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        #Returning observation
        return self.canvas.coords(self.rect)
    #Implementing the function to get the next state and reward
    def get_state_reward(self, action):
        #Getting the current coordinate of explorer
        #State is basically the coordinate of red rectangle
        #Current state
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        #For moving up
        if action == 0: 
            if s[1] > UNIT:
                base_action[1] -= UNIT
        #For moving down
        elif action == 1: 
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        #For moving right
        elif action == 2:  
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        #For moving left
        elif action == 3:  
            if s[0] > UNIT:
                base_action[0] -= UNIT
        #Now moving the red rectangle
        self.canvas.move(self.rect, base_action[0], base_action[1])
        #Getting the coordinate of next state
        s_ = self.canvas.coords(self.rect) 

        #Now as we got our new state we can compute the reward based on new state
        if s_ == self.canvas.coords(self.oval):
            #Reward is 1 if we get to goal position(and also we are done with our goal here)
            reward = 1 
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            #Reward is -1 if we get into hell(also a terminating condition)
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            #Reward 0 for all other position other than goal and hell positions
            reward = 0 
            done = False

        return s_, reward, done
#This is all what we have in our Maze class
