import numpy as np
from QValueTree import QValueTree


class Qlearning_TDControl():
    def __init__(self,
             actions_size,random_actions_probabilities,
             gamma=1,
             lr_v=0.01):
        """
        Calculates optimal policy using in-policy Temporal Difference control
        Evaluates Q-value for (S,A) pairs, using one-step updates.
        """
        # the discount factor
        self.gamma = gamma
        # size of system
        self.actions_size = actions_size
        self.random_actions_probabilities = random_actions_probabilities

        # the learning rate
        self.lr_v = lr_v

        # where to save returns
        self.Qvalues = QValueTree()
    # -------------------
    def single_step_update(self, s, a, r, new_s, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Employs the EXPERIENCED action in the new state  <- Q(S_new, A_new).
        """
        if done:
            # in TD(0) it was
            # delta = (r + 0 - self.values[s])
            deltaQ = (r + 0 - self.Qvalues.get_state_action_pair_value(s, a) )
        else:
            # in TD(0) it was
            # delta = (r + gamma*self.values[new_s] - self.values[s])
            # Notice that I evaluate Qvalue at the new_s only for the action new_a that I really took!
            maxQ_over_actions = self.Qvalues.find_best_qvalue(new_s )

            deltaQ = (r +
                      self.gamma * maxQ_over_actions
                                 - self.Qvalues.get_state_action_pair_value( s,a))

        self.Qvalues.update_state_action_pair(s, a, self.lr_v * deltaQ)

    # ---------------------
    def get_action_epsilon_greedy(self, s, eps):
        """
        Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        """
        ran = np.random.rand()

        if (ran < eps):
            # probability is uniform for all actions!
            prob_actions = self.random_actions_probabilities

        else:
            prob_actions = self.Qvalues.find_best_qvalue_actions(s, self.actions_size,self.random_actions_probabilities )

        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.actions_size, p=prob_actions)
        return a

    def greedy_policy(self):
        greedy_pol = np.argmax(self.Qvalues, axis = 2)
        return greedy_pol
