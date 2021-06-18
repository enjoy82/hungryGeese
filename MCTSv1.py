%%writefile submission.py

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import random
import numpy as np
from collections import defaultdict, deque

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

def playout(state):
    #報酬値を管理する -> 一番ネック
    return playout(state.next(randomActions(state)))

def randomActions(state):
    #すべてのエージェントの合法手をランダムで選択する
    return 1

#######################　hyper params  ############################
direct = ["EAST", "WEST", "SOUTH", "NORTH"]
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]
READSTEPS = 8
###################################################################

def get1vec(x, y):
    return x + y * 11

def get2vec(s):
    return s % 11, s // 11

def nextPos(s): #4方向の配列を返す,方角も 0 <= x <= 10 0 <= y <= 6
    lis = []
    x, y = get2vec(s)
    for d, xx, yy in zip(direct, dx, dy):
        npos = get1vec((x+xx+11)% 11, (y+yy+7)%7)
        lis.append({"pos":npos, "direct":d})
    return lis


class State:
    def __init__(self, obs):
        #foodの管理
        self.foods = obs.food
        self.step = obs.step
        self.index = obs.index
        self.deletion = [False, False, False, False]
        #当たり判定
        self.bodyDict = defaultdict(int)
        #geeseの管理、高速化のためdeque top head
        self.geeses = []
        for _, geese in enumerate(obs.geese):
            deq = deque()
            for s in geese:
                self.bodyDict[s] = 1
                deq.append(s)
            self.geeses.append(deq)
                
    def checkSegment(self):#step40ごとにsegmentを1削除
        if self.step % 40 == 0:
            for geese in self.geeses:
                geese.pop()
        return
    def checkDeleteGeese(self): #geese削除を管理
        for ind, geese in enumerate(self.geeses):
            if self.deletion == True:
                continue
            else:
                geeseHead = geese[0]
                if self.bodyDict[geeseHead] >= 2:
                    self.deletion[ind] = True
                    for _ in range(len(geese)): #盤面削除
                        self.bodyDict[geese.pop()] -= 1

        return 0
    def next(self, action): #TODO これ大丈夫か？actionは４手一緒に行う
        return 0
    def legalActions(self, ind): #indで指定した合法手(動けるアクション)を取得(もちろん生きているもののみ)
        geeseHead = self.geeses[0]
        nextP = nextPos(geeseHead)
        nextLegalActions = []
        for p in nextP:
            np = p["pos"]
            if self.bodyDict[np] == 0: #TODO 本当はしっぽいきたい
                nextLegalActions.append(p)
        return nextLegalActions
    def isDone(self):
        return 0
    def getReward(self): #勝利管理
        return 0
    def isLose(self): #敗北
        if self.deletion[self.index] == True:
            return True
        return False

def mcts_action(state):
    class Node:
        def __init__(self, state):
            self.value = 0
            self.w = 0
        def evaluate():
            return 0
        def expand():
            return 0
        def next_child_node():
            return 0
            
    
def agent(obs, conf):
    global directions
    
    obs = Observation(obs)
    conf = Configuration(conf)
    state = State(obs)
            
                
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