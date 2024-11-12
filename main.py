import pickle
import sys
import scipy.stats as stats
import os
import multiprocessing as mp
from STLTreeEnv import STLTreeEnv
from SARSA_TDControl import SARSA_TDControl
from src.phis_generator import StlGenerator
import numpy as np
sys.setrecursionlimit(3000)



def single_instance_SARSA(target ):
    np.random.seed(target)
    atom_threshold_sd = 1.0

    ### Create the Actions space
    node_types = ['atomic_proposition', 'not', 'and', 'or', 'always', 'eventually', 'until']
    num_variables = 3
    max_right_bound = 10
    max_timespan = 20

    # Percentiles of a standard normal distribution
    percentiles = {p: stats.norm.ppf(p / 100) for p in range(1, 100)}
    quartiles = [percentiles[i] * atom_threshold_sd + 0 for i in range(3, 100, 5)]

    Actions = []
    atomic_probability = 0
    temporal_probability = 0
    for node_type in node_types:
        if node_type == 'atomic_proposition':
            for variable in range(num_variables):
                for quartile in quartiles:
                    for lte in [True, False]:
                        atomic_probability += 1
                        Actions.append((node_type, {'variable': variable, 'quartile': quartile, 'lte': lte}))
        elif node_type == 'always' or node_type == 'eventually' or node_type == 'until':
            for right_bound in range(1, max_right_bound):
                for left_bound in [None] + list(range(1, right_bound)):
                    temporal_probability += 1
                    Actions.append((node_type, {'left_bound': left_bound, 'right_bound': right_bound}))
            atomic_probability += 1
            Actions.append((node_type, {'left_bound': None, 'right_bound': None}))
        else:
            Actions.append((node_type, None))
    random_actions_probabilities = [1 / (atomic_probability * 7)] * atomic_probability + [1 / 7] * 3 + [
        3 / (7 * temporal_probability)] * temporal_probability

    prob_unbound_time_operator = 0.1  # probability of a temporal operator to have a time bound of the type [0,infty]
    prob_right_unbound_time_operator = 0.1  # probability of a temporal operator to have a time bound of the type [a, infty]
    leaf_probability = 0.5  # probability of generating a leaf (always zero for root)
    # Generate a target random STL formula
    # set environment!-----------------------
    # ----------------- Start   End
    target_embedding = StlGenerator(leaf_prob=leaf_probability, time_bound_max_range=max_right_bound,
                           unbound_prob=prob_unbound_time_operator, right_unbound_prob=prob_right_unbound_time_operator,
                           threshold_sd=atom_threshold_sd, inner_node_prob=[0.2, 0.2, 0.2, 0.2, 0.2, 0],
                           max_timespan=max_timespan).sample(num_variables)
    print(target_embedding)

    # Initialize the hyperparameters, SARSA and the enviroment
    gamma = 1.0
    # learning rate
    lr_v = 0.1
    n_episodes = 2000
    epsilon = 0.15
    rewards_data = {}
    rewards_data[target] = []
    max_nodes_number = 30

    env = STLTreeEnv(target_embedding,max_nodes_number)

    SARSA = SARSA_TDControl( actions_size=len(Actions),random_actions_probabilities = random_actions_probabilities, gamma=gamma, lr_v=lr_v)
    atomic_predicates= []
    # RUN OVER EPISODES
    for i in range(n_episodes):
        done = False
        env.reset()
        s = []
        a = SARSA.get_action_epsilon_greedy(s, epsilon)
        act = Actions[a]
        new_s = s.copy()
        rewards=0
        current_similarity = 0
        for steps in range(max_nodes_number+1):
            # Evolve one step|
            formula, r, done, current_similarity = env.step(act, steps)
            SARSA.Qvalues.add_state_action_pair(s, a)
            rewards+=r
            # Save observed reward
            new_s += [a]
            # Choose new action index
            new_a = SARSA.get_action_epsilon_greedy(new_s, epsilon)
            if r>0:
                print(r)
            SARSA.single_step_update(s, a, r, new_s, new_a, done)
            act = Actions[new_a]
            a = new_a
            s = new_s.copy()
            if done:
                atomic_predicates.append(formula)
                break

            # if count > tstar:
            #     # UPDATE OF LEARNING
            #     SARSA.lr_v = lr_v_0/(1 + 0.003*(count - tstar)**0.75)
            #     # UPDATE OF EPSILON
            #     epsilon = epsilon_0/(1. + 0.005*(count - tstar)**1.05)
        rewards_data[target].append([rewards, current_similarity])
        with open(f'SARSA_instance_{target}_rewards_data', 'wb') as file:
                pickle.dump(rewards_data, file)
        if i%100==0:
            with open(f'SARSA_Q_{target}_data', 'wb') as file:
                pickle.dump(SARSA, file)

if __name__ == "__main__":  # pragma: no cover


    def error_callback_function(error):
        print(f'Error: {error}')
    process_number = len(os.sched_getaffinity(0))
    pool = mp.Pool(process_number)
    results = [pool.apply_async(single_instance_SARSA, (seed,), error_callback=error_callback_function) for seed in
            list(range(36))]
    pool.close()
    pool.join()