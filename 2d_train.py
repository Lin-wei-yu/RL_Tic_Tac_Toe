import numpy as np
from collections import defaultdict
import json
# use Q-learning or Sarsa to train TIC-TAC-TOE
class Agent:
    def __init__(self, gamma, alpha, epsilon, num_action, name):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_action = num_action
        self.Q_table = defaultdict(lambda: np.zeros(num_action))
        self.name = name
        self.trajectory = {}
       
    def select_action(self, state, sym):
        """
        use epsilon-greedy policy to select action.
        @param
        sym : -1, 1 stand for 'x' and 'o'
        """
        state = self.preprocess_state(state, sym)
        possible_action_idxs = [a_idx for a_idx in range(0, self.num_action) if state[a_idx]==0]

        # epsilon greedy
        if (np.random.rand() < self.epsilon):
            # explore
            action_idx = np.random.choice(possible_action_idxs)
            action = (int(action_idx/3), action_idx%3)
        else :
            # choose greedy
            # [1, 2, 3] -> Q [23, 12, 23]
            max_Q = max(self.Q_table[state][possible_action_idxs]) # 23
            maxQ_idxs = np.where( self.Q_table[state][possible_action_idxs] == max_Q)[0] # [0, 2]
            action_idx = possible_action_idxs[np.random.choice(maxQ_idxs)]
            action = (int(action_idx/3), action_idx%3)
        
        return action
        
    def update(self):
        """
        Q learning update rule. (MC)
        @param
        state -> action -> state_prime (got reward).
        """
        terminal = True
        for (state, action), reward in reversed(self.trajectory.items()):
            if (terminal == True):
                self.Q_table[state][action] += self.alpha * (reward - self.Q_table[state][action])
                terminal = False
            else:
                self.Q_table[state][action] += self.alpha *\
                    (reward + self.gamma*max(self.Q_table[state_prime]) - self.Q_table[state][action])
            state_prime = state
        
        self.trajectory = {}

    def preprocess_state(self, state, sym):
        """
        preprocess state from 3*3 grid to 1d array. In addition, use sym to normalize state.
        """
        state = state.flatten()
        state = [s*sym for s in state]
        return tuple(state)

    def save_state(self, state, action, reward, sym):
        """
        save state, action, reward pair to trajectory
        """
        state = self.preprocess_state(state, sym)
        action = action[0]*3 + action[1]
        self.trajectory[(state, action)] = reward

    def read_Q(self, Q_path):
        with open(Q_path, 'r') as f:
            Q = {
                eval(k):np.array(v) for k, v in json.load(f).items()
            }
        for k, v in Q.items():
            self.Q_table[k] = v

    def save_Q(self, Q_path):
        with open(Q_path, 'w') as json_file:
            json.dump({
                str(k):v.tolist() for k, v in self.Q_table.items()
            }, json_file)



# training
class Game:
    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2
        self.board = np.zeros(shape=(3, 3), dtype=int)

    def start(self, mode="a2a"):
        if (mode == "a2a"):
             self.agent_2_agent()
        else :
            self.agent_2_user()

    def agent_2_user(self):
        pass
        # sym1 = 1
        # sym2 = -1
        # reply = input("would u like to go first (y/n)? ") 

        # if (reply == "y"):
        #     user_cmd = input("enter position: ")
        #     action1 = self.parse_cmd(user_cmd)
        #     self.update_board(sym1, action1)
        #     self.visulize(self.board)

        # while True:
        #     # agent turn
        #     state = self.board.copy()
        #     action2 = self.agent.select_action(state, sym2)
        #     state_p = self.update_board(sym2, action2)
        #     self.visulize(self.board)
        #     if (self.win() == True): # agent win
        #         reward = 1
        #         break
        #     elif (self.end() == True): #draw
        #         reward = 0
        #         break
        #     else:
        #         reward = 0
        #         self.agent.update(state, action2, reward, state_p, sym2)

        #     # user turn.
        #     user_cmd = input("enter position: ")
        #     action1 = self.parse_cmd(user_cmd)

        #     while (self.board[action1[0]][action1[1]] != 0):
        #         user_cmd = input("invalid, enter a new position: ")
        #         action1 = self.parse_cmd(user_cmd)

        #     self.update_board(sym1, action1)
        #     self.visulize(self.board)

        #     if (self.win() == True): # user win
        #         reward = -1 
        #         break
        #     if (self.end() == True): # draw
        #         reward = 0
        #         break

        # self.agent.update(state, action2, reward, state_p, sym2)

    def agent_2_agent(self):
        sym1 = 1
        sym2 = -1
        while True:
            # agent 1 turn
            state1 = self.board.copy()
            action1 = self.agent1.select_action(state1, sym1)
            self.update_board(sym1, action1)

            if (self.win() == True): # win
                reward1 = 1
                reward2 = -1
                break
            elif (self.end() == True): # draw
                reward1 = reward2 = 0
                break
            else: # continue
                reward1 = 0

            self.agent1.save_state(state1, action1, reward1, sym1)
            
            state2 = self.board.copy()
            action2 = self.agent2.select_action(state2, sym2)
            self.update_board(sym2, action2)
            # self.visulize(state2_p)
            if (self.win() == True):
                reward2 = 1
                reward1 = -1
                break
            elif (self.end() == True):
                reward1 = reward2 = 0
                break
            else:
                reward2 = 0

            self.agent2.save_state(state2, action2, reward2, sym2)

        self.agent1.save_state(state1, action1, reward1, sym1)
        self.agent2.save_state(state2, action2, reward2, sym2)
        
        self.agent1.update()
        self.agent2.update()

    def parse_cmd(self,user_cmd):
        r, c = map(int, user_cmd.split())
        return (r, c)

    def update_board(self, sym, action):
        """
        """
        self.board[action[0]][action[1]] = sym

    def win(self):
        if (self.board[0][0] == self.board[0][1] and
            self.board[0][1] == self.board[0][2] and
            self.board[0][0] != 0):
            return True
        elif (self.board[1][0] == self.board[1][1] and
            self.board[1][1] == self.board[1][2] and
            self.board[1][0] != 0):
            return True
        elif (self.board[2][0] == self.board[2][1] and
            self.board[2][1] == self.board[2][2] and
            self.board[2][0] != 0):
            return True
        elif (self.board[0][0] == self.board[1][0] and
            self.board[1][0] == self.board[2][0] and
            self.board[0][0] != 0):
            return True
        elif (self.board[0][1] == self.board[1][1] and
            self.board[1][1] == self.board[2][1] and
            self.board[0][1] != 0):
            return True
        elif (self.board[0][2] == self.board[1][2] and
            self.board[1][2] == self.board[2][2] and
            self.board[0][2] != 0):
            return True
        elif (self.board[0][0] == self.board[1][1] and
            self.board[1][1] == self.board[2][2] and
            self.board[0][0] != 0):
            return True        
        elif (self.board[0][2] == self.board[1][1] and
            self.board[1][1] == self.board[2][0] and
            self.board[0][2] != 0):
            return True
        else:
            return False

    def end(self):
        if (self.win() == True):
            return True
        if (self.board[0][0] != 0 and self.board[0][1] != 0 and self.board[0][2] != 0 and
            self.board[1][0] != 0 and self.board[1][1] != 0 and self.board[1][2] != 0 and
            self.board[2][0] != 0 and self.board[2][1] != 0 and self.board[2][2] != 0 ):
            return True
        return False

    def visulize(self, state):
        sym=['x',' ','o']
        b = state
        for i in range(0,3):
            print("------")
            print("%c|%c|%c" %(sym[b[i][0]+1],sym[b[i][1]+1],sym[b[i][2]+1]))
        print("------")


if __name__ == "__main__":
    agent1 = Agent(gamma=0.9, alpha=0.5, epsilon=0.1, num_action=3*3, name="a1")
    agent2 = Agent(gamma=0.9, alpha=0.5, epsilon=0.1, num_action=3*3, name="a2")
    agent1.read_Q(agent1.name + "_data.json")
    agent2.read_Q(agent2.name + "_data.json")

    episode = 1
    ep = 0
    while ep < episode:
        # print("ep %d start:" %ep)
        if (ep%(episode/10) == 0): print("ep %d :" %ep)
        game = Game(agent1, agent2)
        game.start(mode="a2a")
        ep += 1

    agent1.save_Q(agent1.name + "_data.json")
    agent2.save_Q(agent2.name + "_data.json")
