import numpy as np
import numpy.random as rnd

class QValueNode(object):
    def __init__(self):
        self.value = 0
        self.children = {}

class QValueTree(object):
    def __init__(self):
        self.root = QValueNode()


    def add_state_action_pair(self, node, action):
        node = self.find_node(node)
        node.children[action] = QValueNode()

    def find_node(self, state):
        if not state:
            return self.root
        elif state[0] not in self.root.children:
            self.root.children[state[0]] = QValueNode()
            return self.root.children[state[0]]
        else:
            root = self.root.children[state[0]]
            for index in range(1,len(state)):
                node = root.children[state[index]]
                root  = node
            return root

    def get_state_action_pair_value(self, state, action):
        node = self.find_node(state)

        if action not in node.children:
            self.add_state_action_pair(state, action)
        return node.children[action].value

    def update_state_action_pair(self, state, action, delta):
        node = self.find_node(state)
        node.children[action].value += delta

    def find_best_qvalue(self, state, action_number, random_actions_probabilities):
        best_qvalue = 0
        best_actions = []
        if not state:
            return random_actions_probabilities
        node = self.find_node(state)
        if not node.children:
            prob_actions = random_actions_probabilities
        else:
            for child in  node.children:
                if node.children[child].value > best_qvalue:
                    best_qvalue = node.children[child].value

            for child in  node.children:
                if node.children[child].value == best_qvalue:
                    best_actions.append(child)

            prob_actions = np.zeros(action_number)
            prob_actions[best_actions] = 1/len(best_actions)

        return prob_actions


class SARSA_TDControl():
    def __init__(self,
                 actions_size, random_actions_probabilities,
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
    def single_step_update(self, s, a, r, new_s, new_a, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Employs the EXPERIENCED action in the new state  <- Q(S_new, A_new).
        """
        if done:
            # in TD(0) it was
            # delta = (r + 0 - self.values[s])
            deltaQ = (r + 0 - self.Qvalues.get_state_action_pair_value(s, a))
        else:
            # in TD(0) it was
            # delta = (r + gamma*self.values[new_s] - self.values[s])
            # Notice that I evaluate Qvalue at the new_s only for the action new_a that I really took!
            deltaQ = (r +
                      self.gamma * self.Qvalues.get_state_action_pair_value(new_s, new_a)
                      - self.Qvalues.get_state_action_pair_value(s, a))

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
            prob_actions = self.Qvalues.find_best_qvalue(s, self.actions_size, self.random_actions_probabilities)

        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.actions_size, p=prob_actions)
        return a

    def greedy_policy(self):
        greedy_pol = np.argmax(self.Qvalues, axis=2)
        return greedy_pol