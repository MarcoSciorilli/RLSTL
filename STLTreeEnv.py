import src.stl as stl
from src.kernel import StlKernel
from src.traj_measure import BaseMeasure

class STLTreeNode(object):
    def __init__(self, args):
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
        Defines a GridWorld with start and end sites.
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

    def get_distance_from_target(self, current_embedding):
        if current_embedding.check_completeness():
            # device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            initial_std = 1.0  # standard deviation of normal distribution of initial state
            total_var_std = 1.0
            n_var=3
            mu0 = BaseMeasure( sigma0=initial_std, sigma1=total_var_std, q=0.1)
            kernel = StlKernel(mu0, samples=10000, sigma2=0.44, varn=n_var)
            similarity = kernel.compute_bag_bag([current_embedding.get_STL_formula()], [self.end]).cpu().numpy()[0][0]

            return similarity
        else:
            return 0

    def reset(self):
        """
        Resets the GridWorld to the starting position.
        """
        # Reset the environment to initial state
        self.current_state = STLTree()
        self.done = False
        self.current_similarity =0

    def step(self, A, nodes_number):
        """
        Evolves the environment given action A and current state.
        """
        # Check if action A is in proper set
        self.current_state.add_node(A)

        similarity = self.get_distance_from_target(self.current_state)
        if similarity !=0:
            reward = similarity - self.current_similarity
        else:
            reward = 0
        if similarity>self.current_similarity:
            self.current_similarity = similarity
        # If I fall over the ridge, I go back to the start and get a penalty
        if self.get_distance_from_target(self.current_state) >0.8:
            self.done = True
        if (nodes_number == self.max_nodes_number) and (self.current_similarity ==0):
            reward = - 0.8*self.max_nodes_number

        return self.current_state.get_STL_formula(), reward, self.done, self.current_similarity