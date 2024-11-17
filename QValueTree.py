import numpy as np

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

    def find_best_qvalue_actions_probabilities(self, state, action_number, random_actions_probabilities):
        best_qvalue = 0
        best_actions = []

        node = self.find_node(state)
        if not node.children:
            prob_actions = random_actions_probabilities
        else:
            for child in node.children:
                if node.children[child].value >= best_qvalue:
                    best_qvalue = node.children[child].value

            for child in  node.children:
                if node.children[child].value == best_qvalue:
                    best_actions.append(child)

            prob_actions = np.zeros(action_number)
            prob_actions[best_actions] = 1/len(best_actions)
        return prob_actions


    def find_best_qvalue_actions(self, state):
        best_qvalue = 0
        best_actions = []
        if not state:
            return best_actions
        node = self.find_node(state)
        if not node.children:
            return best_actions
        else:
            for child in  node.children:
                if node.children[child].value > best_qvalue:
                    best_qvalue = node.children[child].value

            for child in  node.children:
                if node.children[child].value == best_qvalue:
                    best_actions.append(child)
            return best_actions

    def get_all_qvalues(self, state):
        qvalues = []
        actions = []
        node = self.find_node(state)
        if not node.children:
            qvalues.append(0)
            actions.append(0)
            return qvalues, actions
        else:
            for child in  node.children:
                qvalues.append(node.children[child].value)
                actions.append(child)
            return qvalues, actions


    def find_best_qvalue(self, state):
        best_qvalue = 0
        if not state:
            return 0
        node = self.find_node(state)
        if not node.children:
            return 0
        else:
            for child in  node.children:
                if node.children[child].value > best_qvalue:
                    best_qvalue = node.children[child].value
        return best_qvalue