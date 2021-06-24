%%writefile submission.py

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import random
import numpy as np
from collections import defaultdict, deque
import copy
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


#######################　hyper params  ############################
directdict = {"EAST" : (0, 1), "WEST" : (0, -1), "SOUTH" : (-1, 0), "NORTH" : (1, 0)}
direct = ["EAST", "WEST", "SOUTH", "NORTH"]
dx = [0, 0, -1, 1]
dy = [1, -1, 0, 0]
READSTEPS = 8
NOACTION = "NOACTION"
###################################################################

def get1vec(x, y):
    return x * 11 + y

def get2vec(s):
    #return x, y
    return s // 11, s % 11


def playout(state): #TODO　これだと同時着手ゲームに対応できない。自分の手の報酬を管理したい
    if state.isLose():
        return -1
    if state.getReward() != -1:
        return 0
    actionlist = []
    #get randomaction
    for ind in range(4):
        actionlist.append(state.randomAction(ind))
    return playout(state.next(actionlist))

class State:
    def __init__(self, obs):
        #foodの管理
        self.foods = obs.food
        self.step = obs.step
        self.count = 0
        self.index = obs.index #自分のgeeseのindex
        self.deletion = [False, False, False, False]
        #当たり判定　配列で
        self.board = [[0 for i in range(11)] for l in range(7)]
        #geeseの管理、高速化のためdeque top head
        self.geeses = []
        for _, geese in enumerate(obs.geese):
            deq = deque()
            for s in geese:
                x, y = get2vec(s)
                self.board[x][y] = 1
                deq.append((x,y))
            self.geeses.append(deq)
                
    def checkSegment(self):#step40ごとにsegmentを1削除
        if self.step % 40 == 0:
            for ind, geese in enumerate(self.geeses):
                if self.deletion[ind] == True:
                    continue
                geese.pop()
        return
    def checkDeleteGeese(self): #geese削除を管理
        for ind, geese in enumerate(self.geeses):
            if self.deletion[ind] == True:
                continue
            if len(geese) == 0:
                self.deletion[ind] = True
                continue
            else:
                geeseHeadx, geeseHeady = geese[0]
                if self.board[geeseHeadx][geeseHeady] >= 2: #重複削除
                    self.deletion[ind] = True
                    for _ in range(len(geese)): #盤面削除
                        x, y = geese.pop()
                        self.board[x][y] -= 1

        return
        
    def next(self, actions): #TODO これ大丈夫か？actionは４手一緒に行う 次の盤面を用意する。TODO #directで管理
        for ind, action in enumerate(actions):
            if self.deletion[ind] == True:
                continue
            if action == NOACTION:
                action = directdict[direct[np.random.randint(0, len(4))]] #ランダム行動
            #TODO write this func!! 頭に入れる！！ けつをカット！！
        self.checkSegment()
        self.checkDeleteGeese()
        return 0

    def legalActions(self, ind): #indで指定した合法手(動けるアクション)を取得(もちろん生きているもののみ)
        if self.deletion[ind] == True:
            return []
        geeseHeadx, geeseHeady = self.geeses[0]
        nextLegalActions = []
        for dir, xx, yy in zip(direct, dx, dy):
            nx = geeseHeadx + xx
            ny = geeseHeady + yy
            if self.board[nx][ny] != 1:
                nextLegalActions.append(dir)
            else:
                for ind, geese in enumerate(self.geeses): #しっぽは最強手
                    if self.deletion[ind] == True:
                        continue
                    geeseBackx, geeseBacky = geese[-1]
                    if nx == geeseBackx and ny == geeseBacky:
                        nextLegalActions.append(dir)
                        break
        return nextLegalActions

    def randomAction(self, ind):
        nextLegalActions = self.legalActions(ind)
        if len(nextLegalActions) == 0:
            return NOACTION
        return nextLegalActions[np.random.randint(0, len(nextLegalActions))]

    def isDraw(self):
        return 0
        
    def getReward(self): #勝利管理 8手先読み
        if self.count == READSTEPS:
            if self.deletion[self.index] == False:
                return len(self.geeses[self.index]) #ひとまずTODO
        else:
            return -1

    def isLose(self): #敗北管理
        if self.deletion[self.index] == True:
            return True
        return False

def mcts_action(state):
    class Node:
        def __init__(self, state):
            self.state = state
            self.value = 0
            self.w = 0
            self.child_nodes = None
        def evaluate():
            return 0
        def expand():
            return 0
        def next_child_node():
            return 0
    
    root_node = Node(state)
    root_node.expand()
            
    
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