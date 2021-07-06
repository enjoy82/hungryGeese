#%%writefile submission.py

#from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import random
import numpy as np
from collections import defaultdict, deque
import copy
import math
directions = {0:'EAST', 1:'NORTH', 2:'WEST', 3:'SOUTH', 'EAST':0, 'NORTH':1, 'WEST':2, 'SOUTH':3}

#TODO DUCTに変更？
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
EXPANDCOUNT = 1000 #ノード展開の数
SIMULATECOUNT = 10000 #シミュレーション数
###################################################################

def displayBoard(board):
    for row in board:
        print(row)
    return

def get1vec(x, y):
    return x * 11 + y

def get2vec(s):
    #return x, y
    return s // 11, s % 11


def playout(state): #TODO　DUCT
    reward = []
    if state.count == READSTEPS:
        for ind in range(4):
            reward.append(state.getReward(ind))
        return reward
    actionlist = []
    #get randomaction
    for ind in range(4):
        actionlist.append(state.randomAction(ind))
    return playout(state.next(actionlist))

class State:
    def __init__(self, obs):
        #foodの管理
        self.foods = obs["food"]
        self.step = obs["step"]
        self.count = 0
        self.index = obs["index"] #自分のgeeseのindex
        self.deletion = [False, False, False, False]
        #当たり判定　配列で
        self.board = [[0 for i in range(11)] for l in range(7)]
        #geeseの管理、高速化のためdeque top head
        self.geeses = []
        for _, geese in enumerate(obs["geese"]):
            deq = deque()
            for s in geese:
                x, y = get2vec(s)
                self.board[x][y] = 1
                deq.append((x,y))
            self.geeses.append(deq)
        displayBoard(self.board)
    def checkSegment(self):#step40ごとにsegmentを1削除
        if self.step % 40 == 0:
            for ind in len(self.geeses):
                if self.deletion[ind] == True:
                    continue
                self.geeses[ind].pop()
        return
    def checkDeleteGeese(self): #geese削除を管理
        for ind in len(self.geeses):
            if self.deletion[ind] == True:
                continue
            if len(self.geeses[ind]) == 0:
                self.deletion[ind] = True
                continue
            else:
                geeseHeadx, geeseHeady = self.geeses[ind][0]
                if self.board[geeseHeadx][geeseHeady] >= 2: #重複削除
                    self.deletion[ind] = True
                    for _ in range(len(self.geeses[ind])): #盤面削除
                        x, y = self.geeses[ind].pop()
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
        #count増やす
        self.count += 1
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

    #TODO たぶんいらん
    def isDraw(self):
        return 0
        
    def getReward(self, ind): #勝利管理 8手先読み
        if self.deletion[ind] == False:
            return len(self.geeses[self.index]) #ひとまずTODO
        else:
            return -1

    def isLose(self, ind): #敗北管理
        if self.deletion[self.index] == True:
            return True
        return False
    
    def isDone(self, ind): #終了->価値, else-> -2
        if (self.isLose() == False) or (self.getReward() == -1):
            return -2
        if self.isLose() == True:
            return -1
        return self.getReward()


def mcts_action(state):
    class Node:
        def __init__(self, state):
            self.state = state
            self.n = 0
            self.w = 0
            self.child_nodes = None
        def evaluate(self):
            value = self.state.isDone()
            #ゲーム終了
            if value != -2:
                self.w += value
                self.n += 1
                return value
            if not self.child_nodes:
                #TODO DUCT
                value = playout(self.state)
                self.w += value
                self.n += 1
                if self.n == EXPANDCOUNT:
                    self.expand()
                return value
            else:
                value = -self.next_child_node().evaluate()

                self.w += value
                self.n += 1
                return value
        
        def expand(self):
            #TODO index
            legal_actions = self.state.legalActions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))
        def next_child_node(self): #TODO change to DUCT!!
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w / child_node.n + (2*math.log(t)/child_node.n)**0.5)
            return self.child_nodes[np.argmax(ucb1_values)]
    root_node = Node(state)
    root_node.expand()

    for _ in range(SIMULATECOUNT):
        root_node.evaluate()
    
    #試行回数が最大のものを選ぶ
    legal_actions = state.legalActions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[np.argmax(n_list)]
            
    
def agent(obs, conf):
    global directions
    
    #TODO delete
    #obs = Observation(obs)
    #conf = Configuration(conf)
    state = State(obs)

    #TODO よくわからん    
    k = mcts_action(state)  
    return directions[k]

if __name__ == '__main__':
    obs = {'remainingOverageTime': 60, 'step': 0, 'geese': [[16], [30], [76], [56]], 'food': [24, 38], 'index': 0}
    agent(obs, " ")