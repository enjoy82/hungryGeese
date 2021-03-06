#%%writefile submissionMCTS.py

#from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import sys

sys.setrecursionlimit(1000000)

import random
import numpy as np
from collections import defaultdict, deque
import math
import time
import copy
import gc
directions = {0:'EAST', 1:'NORTH', 2:'WEST', 3:'SOUTH', 'EAST':0, 'NORTH':1, 'WEST':2, 'SOUTH':3}
try:
    import cPickle as pickle
except:
    import pickle

#######################　hyper params  ############################
directdict = {"EAST" : (0, 1), "WEST" : (0, -1), "SOUTH" : (1, 0), "NORTH" : (-1, 0)}
direct = ["EAST", "WEST", "SOUTH", "NORTH"]
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
READSTEPS = 8 #先読み手数
NOACTION = "NOACTION"
EXPANDCOUNT = 20 #ノード展開の数
#SIMULATECOUNT = EXPANDCOUNT * 2 #シミュレーション数 使わないでいく
STARTTIME = time.time()
###################################################################
Globalpreactions = [NOACTION, NOACTION, NOACTION, NOACTION]
Globalpregeesehead = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
copycount = 0
def displayBoard(board):
    for row in board:
        print(row)
    return

def get1vec(x, y):
    return x * 11 + y

def get2vec(s):
    #return x, y
    return s // 11, s % 11

def getLegalPos(x,y):
    return (x + 7) % 7, (y + 11) % 11

def oppositeAction(s):
    if s == "EAST":
        return "WEST"
    elif s == "WEST":
        return "EAST"
    elif s == "SOUTH":
        return "NORTH"
    elif s == "NORTH":
        return "SOUTH"
    return s

def playout(state):
    reward = []
    if state.count == READSTEPS or state.isLose == True:
        for ind in range(4):
            reward.append(state.getReward(ind))
        return reward
    actionlist = []
    #get randomaction
    for ind in range(4):
        actionlist.append(state.randomAction(ind))
    return playout(state.next(actionlist, fromPlayout = True))

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
        for ind, geese in enumerate(obs["geese"]):
            deq = deque()
            if len(geese) == 0:
                self.deletion[ind] = True
            for s in geese:
                x, y = get2vec(s)
                self.board[x][y] = 1
                deq.append((x,y))
            self.geeses.append(deq)
        self.preaction = copy.deepcopy(Globalpreactions)
        #displayBoard(self.board)
        self.copyflag = 1 #cooyflag 高速化
    def checkSegment(self):#step40ごとにsegmentを1削除
        if self.step != 0 and (self.step + self.count) % 40 == 0:
            for ind in range(len(self.geeses)):
                if self.deletion[ind] == True:
                    continue
                nextGeeseTalex, nextGeeseTaley = self.geeses[ind].pop()
                self.board[nextGeeseTalex][nextGeeseTaley] -= 1

        return
    def checkDeleteGeese(self): #geese削除を管理
        headcheckflag = [False, False, False, False]
        for ind in range(len(self.geeses)):
            if self.deletion[ind] == True:
                continue
            if len(self.geeses[ind]) == 0:
                self.deletion[ind] = True
                continue
            else:
                headcheckflag[ind] = True
        #盤面削除
        deletecheckflag = [False, False, False, False]
        for ind, headflag in enumerate(headcheckflag):
            if headflag == True:
                geeseHeadx, geeseHeady = self.geeses[ind][0]
                if self.board[geeseHeadx][geeseHeady] >= 2: #重複確認
                    deletecheckflag[ind] = True
        for ind, deleteflag in enumerate(deletecheckflag):
            if deleteflag == True:
                self.deletion[ind] = True
                for _ in range(len(self.geeses[ind])): #盤面削除を同時に行う
                    x, y = self.geeses[ind].pop()
                    self.board[x][y] -= 1
        return
        
    def next(self, actions, fromPlayout = False):
        global copycount
        if self.copyflag == 1:
            statecopy = copy.deepcopy(self)
            copycount += 1
            if fromPlayout == True:
                statecopy.copyflag = 0
        else:
            statecopy = self
        for ind in range(len(actions)):
            if statecopy.deletion[ind] == True:
                continue
            if actions[ind] == NOACTION:
                actions[ind] = direct[np.random.randint(0, 4)] #ランダム行動
            geeseHeadx, geeseHeady = statecopy.geeses[ind][0]
            ddx, ddy = directdict[actions[ind]]
            statecopy.preaction[ind] = actions[ind]
            nextGeeseHeadx, nextGeeseHeady = getLegalPos(geeseHeadx + ddx, geeseHeady + ddy)
            statecopy.board[nextGeeseHeadx][nextGeeseHeady] += 1
            statecopy.geeses[ind].appendleft((nextGeeseHeadx, nextGeeseHeady))
            headvec1 = get1vec(nextGeeseHeadx, nextGeeseHeady)
            if headvec1 in statecopy.foods:
                statecopy.foods.remove(headvec1)
            else:
                nextGeeseTalex, nextGeeseTaley = statecopy.geeses[ind].pop()
                statecopy.board[nextGeeseTalex][nextGeeseTaley] -= 1

        statecopy.count += 1            
        statecopy.checkSegment()
        statecopy.checkDeleteGeese()
        return statecopy

    def legalActions(self, ind): #indで指定した合法手(動けるアクション)を取得(もちろん生きているもののみ)
        #delete oposite action
        if self.deletion[ind] == True:
            return []
        geeseHeadx, geeseHeady = self.geeses[ind][0]
        forbidaction = oppositeAction(self.preaction[ind])
        nextLegalActions = []
        for dir, xx, yy in zip(direct, dx, dy):
            nx = geeseHeadx + xx
            ny = geeseHeady + yy
            nx, ny = getLegalPos(nx, ny)
            if self.board[nx][ny] != 1 and dir != forbidaction:
                nextLegalActions.append(dir)
            else:
                for ind, geese in enumerate(self.geeses): #しっぽは最強手
                    if self.deletion[ind] == True:
                        continue
                    geeseBackx, geeseBacky = geese[-1]
                    if nx == geeseBackx and ny == geeseBacky and dir != forbidaction:
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
            maxGeeseLen = -1
            for geese in self.geeses:
                maxGeeseLen = max(maxGeeseLen, len(geese))
            return len(self.geeses[self.index]) / maxGeeseLen #TODO 正規化する
        else:
            return -1

    def isLose(self): #敗北管理
        #print(self.deletion)
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
            self.w = [0 for _ in range(4)]
            self.child_nodes = None
        def evaluate(self):
            #ゲーム終了 -> 広げる意味がない TODO 打ち切り条件書く
            if self.state.isLose() == True:
                value = []
                for ind in range(4):
                    value.append(self.state.getReward(ind))
                self.n += 1
                return value
            if not self.child_nodes:
                value = playout(self.state)
                for ind, v in enumerate(value):
                    self.w[ind] += v
                self.n += 1
                if self.n >= EXPANDCOUNT:
                    self.expand()
                return value
            else:
                value = self.next_child_node().evaluate()
                for ind, v in enumerate(value):
                    self.w[ind] += v
                self.n += 1
                return value
        
        def expand(self):
            #DUCT 3**4 だけ展開する
            legal_actions = []
            for ind in range(4):
                legal_action = self.state.legalActions(ind)
                if len(legal_action) == 0:
                    legal_action.append(NOACTION)
                legal_actions.append(legal_action)
            #print(legal_actions)
            self.child_nodes = [[[[None for d in range(len(legal_actions[3]))] for c in range(len(legal_actions[2]))] for b in range(len(legal_actions[1]))] for a in range(len(legal_actions[0]))]
            for ind1, legal_action1 in enumerate(legal_actions[0]):
                for ind2, legal_action2 in enumerate(legal_actions[1]):
                    for ind3, legal_action3 in enumerate(legal_actions[2]):
                        for ind4, legal_action4 in enumerate(legal_actions[3]):
                            actionlist = [legal_action1, legal_action2, legal_action3, legal_action4]
                            self.child_nodes[ind1][ind2][ind3][ind4] = Node(self.state.next(actionlist))

        def next_child_node(self): #ucb1使う
            ucbTables = []
            t = 0
            actionSize = [len(self.child_nodes), len(self.child_nodes[0]), len(self.child_nodes[0][0]), len(self.child_nodes[0][0][0])]
            for acs in actionSize:
                ucbTable = [[0, 0] for i in range(acs)]
                ucbTables.append(ucbTable)
            for ind1 in range(actionSize[0]):
                for ind2 in range(actionSize[1]):
                    for ind3 in range(actionSize[2]):
                        for ind4 in range(actionSize[3]):
                            t += self.child_nodes[ind1][ind2][ind3][ind4].n
                            if self.child_nodes[ind1][ind2][ind3][ind4].n == 0:
                                return self.child_nodes[ind1][ind2][ind3][ind4]
                            li = [ind1, ind2, ind3, ind4]
                            cwlist = []
                            cn = self.child_nodes[ind1][ind2][ind3][ind4].n
                            for cw in self.child_nodes[ind1][ind2][ind3][ind4].w:
                                cwlist.append(cw)
                            for ind, ac, cw in zip(range(4), li, cwlist):
                                ucbTables[ind][ac][0] += cn
                                ucbTables[ind][ac][1] += cw
            selectac = [0,0,0,0]
            for ind in range(len(ucbTables)):
                bestac = 0
                bestucb = -1e9
                for ac in range(len(ucbTables[ind])):
                    sumn = ucbTables[ind][ac][0]
                    sumw = ucbTables[ind][ac][1]
                    ucb = sumw / sumn + (2*math.log(t)/sumn)**0.5
                    if bestucb < ucb:
                        bestucb = ucb
                        bestac = ac
                selectac[ind] = bestac
            return self.child_nodes[selectac[0]][selectac[1]][selectac[2]][selectac[3]]

        def delete_node(self):
            if self.child_nodes == None:
                del self
                return
            actionSize = [len(self.child_nodes), len(self.child_nodes[0]), len(self.child_nodes[0][0]), len(root_node.child_nodes[0][0][0])]
            for ind1 in range(actionSize[0]):
                for ind2 in range(actionSize[1]):
                    for ind3 in range(actionSize[2]):
                        for ind4 in range(actionSize[3]):
                            self.child_nodes[ind1][ind2][ind3][ind4].delete_node
            del self
            return

    root_node = Node(state)
    root_node.expand()
    c = 0
    while(time.time() - STARTTIME < 0.95):
        c += 1
        root_node.evaluate()
    global copycount
    print("selectionnum : ", c)
    print("copycount : ", copycount)
    #試行回数が最大のものを選ぶ
    legal_actions = root_node.state.legalActions(root_node.state.index)
    #print(root_node.state.deletion)
    actionSize = [len(root_node.child_nodes), len(root_node.child_nodes[0]), len(root_node.child_nodes[0][0]), len(root_node.child_nodes[0][0][0])]
    n_list = [0 for _ in range(max(1, len(legal_actions)))]
    for ind1 in range(actionSize[0]):
        for ind2 in range(actionSize[1]):
            for ind3 in range(actionSize[2]):
                for ind4 in range(actionSize[3]):
                    #print(ind1, ind2, ind3, ind4)
                    if root_node.state.index == 0:
                        n_list[ind1] += root_node.child_nodes[ind1][ind2][ind3][ind4].n
                    elif root_node.state.index == 1:
                        n_list[ind2] += root_node.child_nodes[ind1][ind2][ind3][ind4].n
                    elif root_node.state.index == 2:
                        n_list[ind3] += root_node.child_nodes[ind1][ind2][ind3][ind4].n
                    elif root_node.state.index == 3:
                        n_list[ind4] += root_node.child_nodes[ind1][ind2][ind3][ind4].n
    root_node.delete_node()
    del root_node
    gc.collect()
    #print(type(root_node))
    return legal_actions[np.argmax(n_list)]



def reloadPreActions(obs):
    geeses = obs["geese"]
    global Globalpreactions
    global Globalpregeesehead
    for ind, geese in enumerate(geeses):
        if len(geese) == 0:
            continue
        nowgeeseheadx, nowgeeseheady = get2vec(geese[0])
        pregeeseheadx, pregeeseheady = Globalpregeesehead[ind]
        Globalpregeesehead[ind] = (nowgeeseheadx, nowgeeseheady)
        if pregeeseheadx == -1 and pregeeseheady == -1:
            continue
        for dir, ddx, ddy in zip(direct, dx, dy):
            nx, ny = getLegalPos(pregeeseheadx + ddx, pregeeseheady + ddy)
            if nowgeeseheadx == nx and nowgeeseheady == ny:
                Globalpreactions[ind] = dir
                break

def agent(obs, conf):
    global STARTTIME
    STARTTIME = time.time()
    global directions
    #TODO delete
    #obs = Observation(obs)
    #conf = Configuration(conf)
    reloadPreActions(obs)
    print(Globalpregeesehead)
    state = State(obs)
    best_action = mcts_action(state)
    print(best_action)
    #print(time.time() - STARTTIME)
    return best_action 

if __name__ == '__main__':
    obs = {'remainingOverageTime': 60, 'step': 0, 'geese': [[16], [30], [76], [56]], 'food': [24, 38], 'index': 0}
    agent(obs, " ")

# %%
