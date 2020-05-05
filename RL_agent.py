import numpy as np
import pandas as pd

class QLearningTable:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  # number of of possible actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        #Creating a q table in constructor
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        #Adding this observation to the table
        self.add_state(observation)
        #Action selection based on greedy policy
        if np.random.uniform() < self.epsilon:
            #Choosing a random action
            #Choose a random action number from the number of possible actions
            action = np.random.choice(self.actions)

        else:
            state_action = self.q_table.loc[observation, :]
            #Some actions have same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            #Our action is state action with max value
            action = state_action.idxmax()
        return action

    def learn(self, s, a, r, s_):
        #Adding the next observation (s_) to the table
        #Adding the new state to q table
        self.add_state(s_)

        #Choosing the best q-value for the given pair of (s, a)
        #Lookup for the record s in column a
        q_predict = self.q_table.loc[s, a]

        #Checking if the next state is a terminal state or not and get the expected q value
        #If the next state is a terminal state then only there will be a updation(significant) otherwise not
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r 

        #Updating q-value  in the table
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
    #Now creating a method to add states to q table
    def add_state(self, state):
        if state not in self.q_table.index:
            #Appending new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    #State of action initialized with zero
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )
