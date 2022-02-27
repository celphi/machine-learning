import numpy as np
import math
import pdb



class AI:

    def __init__(self, turns: int, learning_rate: int, discount_factor: int, actions: list, q_values: list):
        '''
        turns: max number of turns an agent can take,
        learning_rate: the rate in which an agent should learn,
        discount_factor: the decayed reward amount
        actions: the actions which the agent can take,
        q_values: a mapping of probabilities which suggests which action should be taken at any given state

        history_cs: state - number of cs built
        history_ci: state - number of ci built (buildings)
        '''

        # default values
        self.state = 0
        self.cs = 0
        self.buildings = 0
        self.max_buildings = 0
        self.history_cs = []
        self.history_ci = []

        self.turns = turns
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions
        self.q_values = q_values


    def reset(self):
        ''' Resets the default values back to their original values '''
        self.state = 0
        self.cs = 0
        self.buildings = 0
        self.history_cs = []
        self.history_ci = []


    def get_reward(self) -> int:
        ''' The reward will be based on the number of buildings created '''
        return self.buildings 

    def is_game_over(self) -> bool:
        ''' Determines if all turns have been used '''
        return self.state == self.turns


    def get_bpt(self, cs: int) -> int:
        ''' Determines the current buildings per turn '''
        return (math.floor(cs/4)) + 5


    def get_next_action(self, epsilon: float) -> int:
        '''
        Returns the most likely successful action with some probability that an inferior action may happen occasionally.
        '''
        if np.random.random() < epsilon:
            return np.argmax(self.q_values[self.state])
        else:
            return np.random.randint(2)


    def get_next_state(self, action_index: int) -> int:
        ''' Executes next action and returns the next state '''
        if self.actions[action_index] == "build ci":
            new_buildings = self.get_bpt(self.cs)
            self.buildings += new_buildings
            self.history_ci.append({self.state: new_buildings})

        elif self.actions[action_index] == "build cs":
            self.cs += 1
            self.history_cs.append({self.state : 1})
 
        self.state += 1
        return self.state

    def print_best_path(self):
        self.reset()
        while not ai.is_game_over():
            action_index = self.get_next_action(1.)
            if action_index == 0:
                print(f"build ci")
            else:
                print(f"build cs")
            self.get_next_state(action_index)
        print(f"total construction sites: {self.cs}")
        print(f"total buildings: {self.buildings}")

      
TURNS = 25

ai = AI(turns=TURNS,
        learning_rate=0.9,
        discount_factor=0.9,
        actions=["build ci", "build cs"],
        q_values=np.zeros((TURNS+1, 1, 2)))


for episode in range(100000):

    ai.reset()

    action_index = None

    while not ai.is_game_over():
        action_index = ai.get_next_action(.9)
        old_state = ai.state
        next_state = ai.get_next_state(action_index) 
        if ai.buildings < ai.max_buildings:
            reward = -10
        else:
            reward = -1 

        old_q_value = ai.q_values[old_state, 0, action_index]
        temporal_difference = reward + (ai.discount_factor * np.max(ai.q_values[next_state])) - old_q_value
        new_q_value = old_q_value + (ai.learning_rate * temporal_difference)
        ai.q_values[old_state, 0, action_index] = new_q_value

    if ai.buildings > ai.max_buildings:
        ai.max_buildings = ai.buildings
        print(f"\nepisode: {episode}")
        print(ai.history_cs)
        print(ai.history_ci)
        print(f"total construction sites: {ai.cs}")
        print(f"total buildings: {ai.buildings}")
        #if ai.buildings == 126:
        #    print(ai.q_values)


    #pdb.set_trace()

#ai.print_best_path()
