import pickle
import sys
import scipy.stats as stats
import os
import multiprocessing as mp

from EXPECTED_SARSA_TDControl import EXPECTED_SARSA_TDControl
from Qlearning_TDControl import Qlearning_TDControl
from STLTreeEnv import STLTreeEnv
from SARSA_TDControl import SARSA_TDControl
from src.phis_generator import StlGenerator
import numpy as np
sys.setrecursionlimit(3000)


def single_instance_random(target):
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
    target_embedding = StlGenerator(leaf_prob=leaf_probability, time_bound_max_range=max_right_bound,
                           unbound_prob=prob_unbound_time_operator, right_unbound_prob=prob_right_unbound_time_operator,
                           threshold_sd=atom_threshold_sd, inner_node_prob=[0.2, 0.2, 0.2, 0.2, 0.2, 0],
                           max_timespan=max_timespan, seed=target).sample(num_variables)
    print(target_embedding)

    n_episodes = 2000

    rewards_data = []
    max_nodes_number = 30
    # Set enviroment
    env = STLTreeEnv(target_embedding,max_nodes_number)

    atomic_predicates= []
    # RUN OVER EPISODES
    for i in range(n_episodes):
        done = False
        env.reset()
        rewards=0
        current_similarity = 0
        for steps in range(max_nodes_number+1):
            # Evolve one step|
            formula, r, done, current_similarity = env.step(Actions[np.random.choice(len(Actions), p=random_actions_probabilities)], steps)
            rewards+=r
            if r>0:
                print(r)
            if done:
                atomic_predicates.append(formula)
                with open(f'random_instance_{target}_formulas', 'wb') as file:
                    pickle.dump(atomic_predicates, file)
                break
        print(f"instance: {target}  episode: {i}  similarity:{current_similarity}" )
        rewards_data.append([rewards, current_similarity])
        with open(f'random_instance_{target}_rewards_data', 'wb') as file:
                pickle.dump(rewards_data, file)


def single_instance_SARSA(target, epsilon_0 ):
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
    # ----------------- Start   End
    target_embedding = StlGenerator(leaf_prob=leaf_probability, time_bound_max_range=max_right_bound,
                           unbound_prob=prob_unbound_time_operator, right_unbound_prob=prob_right_unbound_time_operator,
                           threshold_sd=atom_threshold_sd, inner_node_prob=[0.2, 0.2, 0.2, 0.2, 0.2, 0],
                           max_timespan=max_timespan, seed=target).sample(num_variables)
    print(target_embedding)

    # Initialize the hyperparameters, SARSA and the enviroment
    gamma = 1.0
    # learning rate
    lr_v_0 = 0.5
    lr_v = lr_v_0
    n_episodes = 2000

    rewards_data = []
    max_nodes_number = 30
    # set environment!-----------------------

    env = STLTreeEnv(target_embedding,max_nodes_number)
    epsilon = epsilon_0
    SARSA = SARSA_TDControl( actions_size=len(Actions),random_actions_probabilities = random_actions_probabilities, gamma=gamma, lr_v=lr_v)
    atomic_predicates= []
    count = 0

    # RUN OVER EPISODES
    for i in range(n_episodes):
        done = False
        env.reset()
        s = []
        a = SARSA.get_action_epsilon_greedy(s, epsilon)
        SARSA.Qvalues.add_state_action_pair(s, a)

        act = Actions[a]
        new_s = s.copy()
        rewards=0
        current_similarity = 0
        for steps in range(max_nodes_number+1):
            count +=1
            # Evolve one step|
            formula, r, done, current_similarity = env.step(act, steps)
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
            SARSA.Qvalues.add_state_action_pair(s, a)

            if done:
                atomic_predicates.append(formula)
                with open(f'SARSA_instance_{target}_formulas_{epsilon_0}_constant', 'wb') as file:
                    pickle.dump(atomic_predicates, file)
                break

            # UPDATE OF LEARNING
            SARSA.lr_v = lr_v_0/(1 + 0.003*(count)**0.75)
            # UPDATE OF EPSILON
            # epsilon = epsilon_0/(1. + 0.005*(count)**1.05)
        print(f"instance: {target}  eps:{epsilon_0}  episode: {i}  similarity:{current_similarity}_constant" )
        rewards_data.append([rewards, current_similarity])
        with open(f'SARSA_instance_{target}_rewards_data_{epsilon_0}_constant', 'wb') as file:
                pickle.dump(rewards_data, file)
        if i%100==0:
            with open(f'SARSA_Q_{target}_data_{epsilon_0}', 'wb') as file:
                pickle.dump(SARSA, file)


def single_instance_Qlearning(target, epsilon_0 ):
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
                           max_timespan=max_timespan, seed=target).sample(num_variables)
    print(target_embedding)

    # Initialize the hyperparameters, SARSA and the enviroment
    gamma = 1.0
    lr_v_0 = 0.5
    # learning rate
    lr_v = lr_v_0
    n_episodes = 2000

    rewards_data = []
    max_nodes_number = 30
    epsilon = epsilon_0
    env = STLTreeEnv(target_embedding,max_nodes_number)

    Qlearning = Qlearning_TDControl( actions_size=len(Actions),random_actions_probabilities = random_actions_probabilities, gamma=gamma, lr_v=lr_v)
    atomic_predicates= []
    count =0
    # RUN OVER EPISODES
    for i in range(n_episodes):
        done = False
        env.reset()
        s = []
        a = Qlearning.get_action_epsilon_greedy(s, epsilon)
        Qlearning.Qvalues.add_state_action_pair(s, a)
        act = Actions[a]
        new_s = s.copy()
        rewards=0
        current_similarity = 0
        for steps in range(max_nodes_number+1):
            count +=1
            # Evolve one step|
            formula, r, done, current_similarity = env.step(act, steps)
            rewards+=r
            # Save observed reward
            new_s += [a]
            # Choose new action index
            new_a = Qlearning.get_action_epsilon_greedy(new_s, epsilon)
            if r>0:
                print(r)
            Qlearning.single_step_update(s, a, r, new_s, done)
            act = Actions[new_a]
            a = new_a
            s = new_s.copy()
            Qlearning.Qvalues.add_state_action_pair(s, a)

            if done:
                atomic_predicates.append(formula)
                with open(f'Qlearning_instance_{target}_formulas_{epsilon_0}_constant', 'wb') as file:
                    pickle.dump(atomic_predicates, file)
                break


            # UPDATE OF LEARNING
            Qlearning.lr_v = lr_v_0/(1 + 0.003*(count )**0.75)
            # UPDATE OF EPSILON
            # epsilon = epsilon_0/(1. + 0.005*(count)**1.05)
        print(f"instance: {target}  eps:{epsilon_0}  episode: {i}  similarity:{current_similarity}" )
        rewards_data.append([rewards, current_similarity])
        with open(f'Qlearning_instance_{target}_rewards_data_{epsilon_0}_constant', 'wb') as file:
                pickle.dump(rewards_data, file)
        if i%100==0:
            with open(f'Qlearning_Q_{target}_data_{epsilon_0}_constant', 'wb') as file:
                pickle.dump(Qlearning, file)

def single_instance_EXPECTED_SARSA(target, epsilon_0 ):
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
                           max_timespan=max_timespan, seed=target).sample(num_variables)
    print(target_embedding)

    # Initialize the hyperparameters, SARSA and the enviroment
    gamma = 1.0
    # learning rate
    lr_v = 0.5
    n_episodes = 2000
    lr_v_0 = lr_v
    epsilon = epsilon_0
    rewards_data = []
    max_nodes_number = 30

    env = STLTreeEnv(target_embedding,max_nodes_number)

    EXPECTED_SARSA = EXPECTED_SARSA_TDControl( actions_size=len(Actions),random_actions_probabilities = random_actions_probabilities, gamma=gamma, lr_v=lr_v)
    atomic_predicates= []
    count = 0
    # RUN OVER EPISODES
    for i in range(n_episodes):
        done = False
        env.reset()
        s = []
        a = EXPECTED_SARSA.get_action_epsilon_greedy(s, epsilon)
        EXPECTED_SARSA.Qvalues.add_state_action_pair(s, a)
        act = Actions[a]
        new_s = s.copy()
        rewards=0
        current_similarity = 0
        for steps in range(max_nodes_number+1):
            count +=1
            # Evolve one step|
            formula, r, done, current_similarity = env.step(act, steps)
            rewards+=r
            # Save observed reward
            new_s += [a]
            # Choose new action index
            new_a = EXPECTED_SARSA.get_action_epsilon_greedy(new_s, epsilon)
            # if r>0:
            #     print(r)
            EXPECTED_SARSA.single_step_update(s, a, r, new_s, done, epsilon)
            act = Actions[new_a]
            a = new_a
            s = new_s.copy()
            EXPECTED_SARSA.Qvalues.add_state_action_pair(s, a)
            if done:
                atomic_predicates.append(formula)
                with open(f'EXPECTED_SARSA_instance_{target}_formulas_{epsilon_0}_constant', 'wb') as file:
                    pickle.dump(atomic_predicates, file)
                break

                # UPDATE OF LEARNING
            EXPECTED_SARSA.lr_v = lr_v_0/(1 + 0.003*(count)**0.75)
            # UPDATE OF EPSILON
            # epsilon = epsilon_0/(1. + 0.005*(count)**1.05)
        print(f"instance: {target}  eps:{epsilon}  episode: {i}  similarity:{current_similarity}" )
        rewards_data.append([rewards, current_similarity])
        with open(f'EXPECTED_SARSA_instance_{target}_rewards_data_{epsilon_0}_constant', 'wb') as file:
                pickle.dump(rewards_data, file)
        if i%100==0:
            with open(f'EXPECTED_SARSA_Q_{target}_data_{epsilon_0}_constant', 'wb') as file:
                pickle.dump(EXPECTED_SARSA, file)





if __name__ == "__main__":  # pragma: no cover
    # single_instance_EXPECTED_SARSA(0, 0.01)

    def error_callback_function(error):
        print(f'Error: {error}')
    process_number = len(os.sched_getaffinity(0))
    pool = mp.Pool(process_number)
    results = [pool.apply_async(single_instance_EXPECTED_SARSA, (seed,epsilon), error_callback=error_callback_function) for seed in
            list(range(10)) for epsilon in [0.1,0.3,0.5]]
    pool.close()
    pool.join()
    #
    # def error_callback_function(error):
    #     print(f'Error: {error}')
    # process_number = len(os.sched_getaffinity(0))
    # pool = mp.Pool(process_number)
    # results = [pool.apply_async(single_instance_SARSA, (seed,epsilon), error_callback=error_callback_function) for seed in
    #         list(range(10)) for epsilon in [0.1,0.3,0.5]]
    # pool.close()
    # pool.join()
    #
    # def error_callback_function(error):
    #     print(f'Error: {error}')
    # process_number = len(os.sched_getaffinity(0))
    # pool = mp.Pool(process_number)
    # results = [pool.apply_async(single_instance_Qlearning, (seed,epsilon), error_callback=error_callback_function) for seed in
    #         list(range(10)) for epsilon in [0.1,0.3,0.5]]
    # pool.close()
    # pool.join()