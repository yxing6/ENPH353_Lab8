import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # discount constant
        self.gamma = gamma  # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        # https://stackoverflow.com/questions/35067957/how-to-read-pickle-file
        with open(filename + '.pickle', 'rb') as f:
            self.q = pickle.load(f)

        print("Loaded file: {}".format(filename + ".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        with open(filename + '.pickle', 'wb') as f:
            pickle.dump(self.q, f)

        print("Wrote to file: {}".format(filename + ".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action
        # explo
        if random.random() < self.epsilon:
            if return_q:
                return random.choice(self.actions), True
            else:
                return random.choice(self.actions)
        else:
            max_q = max([self.getQ(state, a) for a in self.actions])
            max_q_actions = [a for a in self.actions if self.getQ(state, a) == max_q]
            if return_q:
                return random.choice(max_q_actions), max_q
            else:
                return random.choice(max_q_actions)

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # Find Q for current (state1, action1)
        q_11 = self.getQ(state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        max_q12 = max([self.getQ(state2, a) for a in self.actions])
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)
        # if the pair is in the dictionary, update the value
        if q_11 != False:
            self.q[(state1, action1)] += self.alpha * (reward + self.gamma * max_q12 - self.q[(state1, action1)])
        # if the pair is not in the dictionary, enter the new pair
        else:
            self.q[(state1, action1)] = self.alpha * (reward + self.gamma * max_q12)

