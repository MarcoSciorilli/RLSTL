import torch
import random
import src.stl as stl
from src.kernel import StlKernel
from src.traj_measure import BaseMeasure
import GPUtil
import time
import random

class STLTreeNode(object):
    def __init__(self, args):
        """
        Data Structure of a Node of an STLTree. Each node is defined by its children, and the type of STL operator
        it is. Given that some operator can be implemented only when its children are already define, the STL operator
        store in the node is actually initialized only when all the conditions are met
        :param args: tuple of STL operator type (as a string), and a dictionary with the information needed to
                    that specific type of node.
        """
        self.left = None
        self.right = None
        self.node_type = args[0]
        self.node_data = args[1]
        if self.node_type =='atomic_proposition':

            self.is_branch_complete = True
            self.STLnode = self.choose_atomic_node(self.node_data['variable'],self.node_data['quartile'], self.node_data['lte'])
        elif self.node_type =='always' or self.node_type =='eventually' or self.node_type =='until':
            self.left_bound = self.node_data.get('left_bound', None)
            self.right_bound = self.node_data.get('right_bound', None)
            self.is_branch_complete = False
            self.STLnode = None
        else:
            self.is_branch_complete = False
            self.STLnode = None

    def choose_atomic_node(self, variable, quartile, lte):
        return stl.Atom(variable, quartile, lte)

    def update_node(self):
        """
        Method initialising the STL operator given all the needed information. choose_atomic_node and
        _get_temporal_parameters are ancillary method for it.
        """
        if self.node_type == 'not':
            self.STLnode = stl.Not(self.left.STLnode)
        elif self.node_type == 'and':
            self.STLnode = stl.And(self.left.STLnode, self.right.STLnode)
        elif self.node_type == 'or':
            self.STLnode = stl.Or(self.left.STLnode, self.right.STLnode)
        elif self.node_type == 'always':
            unbound, right_unbound, left_time_bound, right_time_bound = self._get_temporal_parameters(self.left_bound, self.right_bound)
            self.STLnode = stl.Globally(
                self.left.STLnode, unbound, right_unbound, left_time_bound, right_time_bound, True
            )
        elif self.node_type == 'eventually':
            unbound, right_unbound, left_time_bound, right_time_bound = self._get_temporal_parameters(self.left_bound, self.right_bound)
            self.STLnode = stl.Eventually(
                self.left.STLnode, unbound, right_unbound, left_time_bound, right_time_bound, True
            )
        elif self.node_type == 'until':
            unbound, right_unbound, left_time_bound, right_time_bound = self._get_temporal_parameters(self.left_bound, self.right_bound)
            self.STLnode = stl.Until(
                self.left.STLnode, self.right.STLnode, unbound, right_unbound, left_time_bound, right_time_bound
            )

    def _get_temporal_parameters(self, left_bound=None, right_bound=None):
        if left_bound is None and right_bound is None:
            return True, False, 0, 0
        elif left_bound is None and right_bound is not None:
            return False, True, right_bound, 1
        else:
            return False, False, left_bound, right_bound



class STLTree(object):
    def __init__(self):
        """
        Class implement an STL formula as a binary datastructure. It implements all standard method needed to
        use a binary tree: finding a node, adding a node, traversing the tree, adding a root, and updating the
        tree. Un update method is implemented to check if the current state of the tree translate to a meaningful STL
        formula.
        """
        self.root = None

    def _add_tree_root_(self,new_tree_root):
        if self.root is None:
            self.root = new_tree_root
        else:
            new_tree_root.left = self.root
            self.root = new_tree_root


    def _inorderTraversal_(self, root):
        if root.is_branch_complete is False :
            if root.left is not None:
                if root.left.is_branch_complete is True:
                    if root.right is not None:
                        return self._inorderTraversal_(root.right)
                    else:
                        return root
                else:
                    return self._inorderTraversal_(root.left)
            else:

                return root

    def _add_tree_node_(self, new_tree_node):
        if (self.root is None) or (self.root.is_branch_complete is True):
            self._add_tree_root_(new_tree_node)
        else:
            sub_root =self._inorderTraversal_(self.root)
            if sub_root.left is None:
                sub_root.left = new_tree_node
            elif sub_root.right is None:
                sub_root.right = new_tree_node
            self._update_tree_(sub_root)
        self._update_tree_(self.root)

    def _update_tree_(self, root):
        if root.is_branch_complete is True:
            return
        else:
            if (root.left is None) and (root.right is None):
                return
            elif (root.node_type =='not') or (root.node_type =='always' ) or (root.node_type =='eventually'):
                if root.left.is_branch_complete is True:
                    root.is_branch_complete = True
                    root.update_node()
                else:
                    self._update_tree_(root.left)
                    if root.left.is_branch_complete is True:
                        root.is_branch_complete = True
                        root.update_node()

            else:
                if root.right is None:
                    if root.left.is_branch_complete is False:
                        self._update_tree_(root.left)

                elif (root.left.is_branch_complete is True) and (root.right.is_branch_complete is True):
                    root.is_branch_complete = True
                    root.update_node()
                else:
                    self._update_tree_(root.right)
                    self._update_tree_(root.left)
                    if (root.left.is_branch_complete is True) and (root.right.is_branch_complete is True):
                        root.is_branch_complete = True
                        root.update_node()




    def add_node(self, args):
        self._add_tree_node_( STLTreeNode(args))

    def check_completeness(self):
        return self.root.is_branch_complete

    def get_STL_formula(self):
        if self.check_completeness():
            return self.root.STLnode
        else:
            return


class STLTreeEnv(object):
    def __init__(self, target_embedding, max_nodes_number):
        """
        Defines as an environment the space of possible STL formulas trees. As the rewards are given by the distance
        to a point in the enviroment space, this point has to be provided through the target_embedding variable. Given that
        we want to generate formulas of a maximum lenght, the maximum number of nodes has to be provided to correctly
        encode the coffin state.
        :target_embeggin target STL formula that we want to get close to, as a SLT formula
        :max_nodes_number maximum number of nodes in the SLT formula tree that we want to generate
        """

        # Reads the position of start and end
        self.max_nodes_number = max_nodes_number
        self.start = STLTree()
        self.end = target_embedding

        # Keeps track of current state
        self.current_state = self.start
        self.current_similarity = 0
        # Keeps track of terminal state
        self.done = False

    def get_first_free_gpu(self, min_memory=3000, max_load=1, wait=True):
        '''
        Just an hacky function to parallelize over GPUs
        :param min_memory:
        :param max_load:
        :param wait:
        :return:
        '''
        while True:
            GPUs = GPUtil.getGPUs()
            random.shuffle(GPUs)
            for gpu in GPUs:
                if gpu.memoryFree >= min_memory and gpu.load <= max_load:
                    return gpu.id
            if not wait:
                return None
            print("No available GPU found. Waiting...")
            time.sleep(1)

    def get_distance_from_target(self, current_embedding):
        """
        The reward is defined as the similiarity between the embedding of the currente state and the target embedding.
        If the current state does not correspond to a complete STL formula, the reward is 0.
        If the current state corresponds to a complete STL formula that does not make sense, the reward is -10
        :param current_embedding: current embedding as an STL tree formula
        :return: The reward
        """
        if current_embedding.check_completeness():
            device_id = self.get_first_free_gpu()
            device = f"cuda:{device_id}"
            initial_std = 1.0  # standard deviation of normal distribution of initial state
            total_var_std = 1.0
            n_var=3



            try:
                mu0 = BaseMeasure(sigma0=initial_std, sigma1=total_var_std, q=0.1, device=device)
                kernel = StlKernel(mu0, samples=10000, sigma2=0.44, varn=n_var)
                similarity = kernel.compute_bag_bag([current_embedding.get_STL_formula()], [self.end]).cpu().numpy()[0][0]
            except torch.cuda.OutOfMemoryError as e:
                try:
                    mu0 = BaseMeasure(sigma0=initial_std, sigma1=total_var_std, q=0.1, device='cpu')
                    kernel = StlKernel(mu0, samples=10000, sigma2=0.44, varn=n_var)
                    similarity = \
                        kernel.compute_bag_bag([current_embedding.get_STL_formula()], [self.end]).cpu().numpy()[0][
                            0]
                except Exception as a:
                    similarity = - 10
            except Exception as e:
                similarity = - 10

            torch.cuda.empty_cache()

            return similarity
        else:
            return 0

    def reset(self):
        """
        Resets the environment and similarity to the starting position.
        """
        # Reset the environment to initial state
        self.current_state = STLTree()
        self.done = False
        self.current_similarity =0

    def step(self, A, nodes_number):
        """
        Evolves the environment given action A and current state.
        """

        self.current_state.add_node(A)

        similarity = self.get_distance_from_target(self.current_state)
        # If instead of getting closer I go more far away, i get penalized
        if similarity !=0:
            reward = similarity - self.current_similarity
        else:
            reward = 0
        if similarity>self.current_similarity:
            self.current_similarity = similarity
        # If I'm close enought, I reached my target
        if self.get_distance_from_target(self.current_state) >0.8:
            self.done = True
        # If when I used all available nodes, I still have no complete STL formula, I get a big negative reward.
        if (nodes_number == self.max_nodes_number) and (self.current_similarity ==0):
            reward = - 0.8*self.max_nodes_number

        return self.current_state.get_STL_formula(), reward, self.done, self.current_similarity
