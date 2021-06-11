%%writefile submission.py

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import random
import numpy as np
from collections import defaultdict
import queue

directions = {0:'EAST', 1:'NORTH', 2:'WEST', 3:'SOUTH', 'EAST':0, 'NORTH':1, 'WEST':2, 'SOUTH':3}

"""
def move(loc, direction):
    global directions
    direction = directions[direction]
    new_loc = []
    if direction == 'EAST':
        new_loc.append(int(11*(loc[0]//11)  + (loc[0]%11 + 1)%11))
    elif direction == 'WEST':
        new_loc.append(int(11*(loc[0]//11) + (loc[0]%11 + 10)%11))
    elif direction == 'NORTH':
        new_loc.append(int(11*((loc[0]//11 + 6)%7) + loc[0]%11))
    else:
        new_loc.append(int(11*((loc[0]//11 + 1)%7) + loc[0]%11))
    if len(loc) == 1:
        return new_loc
    print(new_loc + loc[:-1])
    return new_loc + loc[:-1]
    

def greedy_choose(head, board):
    move_queue = []
    visited = [[[100, 'NA'] for _ in range(11)] for l in range(7)]
    visited[head//11][head%11][0] = 0
    
    for i in range(4):
        move_queue.append([head, [i]])
    
    while len(move_queue) > 0:
        now_move = move_queue.pop(0)
        
        next_step = move([now_move[0]], now_move[1][-1])[0]
        
        if board[next_step//11][next_step%11] < 0:
            continue
        
        if len(now_move[1]) < visited[next_step//11][next_step%11][0]:
            visited[next_step//11][next_step%11][0] = len(now_move[1])
            visited[next_step//11][next_step%11][1] = now_move[1][0]
            for i in range(4):
                move_queue.append([next_step, now_move[1] + [i]])
        
        if board[next_step//11][next_step%11] > 0:
            return now_move[1][0]
    return random.randint(0,3)
"""

class state:
    def __init__(self, obs):
        #foodの管理
        self.foods = obs.food
        self.step = obs.food
        self.index = obs.index
        #当たり判定
        self.bodyDict = defaultdict(int)
        #geeseの管理、高速化のためqueue
        self.enemies = []  
        for ind, geese in enumerate(obs.geese):
            geese.reverse()
            if ind == self.index:
                mygeese = queue.Queue()
                for s in geese:
                    self.bodyDict[s] = 1
                    mygeese.put(s)
                self.myGeese = mygeese
            else:
                enemy = queue.Queue()
                for s in geese:
                    self.bodyDict[s] = 1
                    enemy.put(s)
                self.enemies.append(enemy)
                
    def checkSegment(self):#step40ごとにsegmentを1削除
        return 0
    def deleteGeese(self): #geese削除を管理
        return 0
    def next(self, action): #TODO これ大丈夫か？actionは４手一緒に行う
        return 0
    def legalActions(self): #合法手(動けるアクション)を選択
        return 0
    def isDone(self): #ゲーム終了かどうか
        #TODO終了条件について考える必要あり、foodの条件によって勝ち筋が異なる
        return 0
    def isWin(self): #勝利管理
        return 0
    def isLose(self): #敗北
        return 0

def mcts_action(state):
    class Node:
        def __init__(self, state):
            self.value = 0
            self.w = 0
            
            
    
def agent(obs, conf):
    global directions
    
    obs = Observation(obs)
    conf = Configuration(conf)
    
            
                
    """
    board = np.zeros((7, 11), dtype=int)
    print(obs)
    #Obstacle-ize your opponents
    for ind, goose in enumerate(obs.geese):
        if ind == obs.index or len(goose) == 0:
            continue
        for direction in range(4):
            moved = move(goose, direction)
            for part in moved:
                board[part//11][part%11] -= 1
    
    #Obstacle-ize your body, except the last part
    if len(obs.geese[obs.index]) > 1:
        for k in obs.geese[obs.index][:-1]:
            board[k//11][k%11] -= 1
    
    #Count food only if there's no chance an opponent will meet you there
    for f in obs.food: 
        board[f//11][f%11] += (board[f//11][f%11] == 0)
    k = greedy_choose(obs.geese[obs.index][0], board)
    print(k)
    """
    return directions[k]