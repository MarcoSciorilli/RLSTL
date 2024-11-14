import numpy as np
from QValueTree import QValueTree


class EXPECTED_SARSA_TDControl():
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
    def single_step_update(self, s, a, r, new_s, done, eps):
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

            # Notice that I evaluate the EXPECTED Qvalue for the new_s weighted by the probability of taking
            # the action (i.e. the policy) !
            all_qvalues, all_actions = self.Qvalues.get_all_qvalues(new_s)
            deltaQ = (r +
                      self.gamma * ( all_qvalues * self.policy(new_s, eps)[all_actions])
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
            prob_actions = self.Qvalues.find_best_qvalue_actions_probabilities(s, self.actions_size,self.random_actions_probabilities )

        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.actions_size, p=prob_actions)
        return a


    def policy(self, s, eps):
        """
        Probabilities from an epsilon-greedy policy wrt the current Q(s,a).
        """
        # Uniform (epsilon) probability for all actions...
        policy = np.ones(self.actions_size) / self.actions_size * eps
        # ... plus 1-epsilon probabilities for best actions:
        # First I find the best values
        # There could be actions with equal value!
        # This mask is 1 if the value is equal to the best (tie)
        # or 0 if the action is suboptimal
        best_actions =  self.Qvalues.find_best_qvalue_actions(s)
        if len(best_actions)>0:
            policy[best_actions] += 1 / len(best_actions) * (1 - eps)
        return policy
