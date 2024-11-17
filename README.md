Repository of my material and final project on the course of Reinforcement Learning by Prof. Antonio Celani at ICTP in the year 2021/2022.

### Content

#### Reinforcement_Learning_Lectures2021 

The notebooks provided during the practical exercise lessons carried on during the course. 

#### Final Project

###### src

Directory containing the source code implementation of the methods described in the paper:https://link.springer.com/chapter/10.1007/978-3-030-99524-9_15

In particular, the relevant modules for the project are:


- **stl.py**: Implements STL operators and their recursive semantics, both Boolean and quantitative.
- **phis_generator.py**: Implements a random sampling algorithm for STL formulas as described in the paper. It recursively constructs the syntax tree, assigning a probability p_leaf that the current node is an atomic proposition (a leaf) and uniform probability to other nodes. Thresholds for atomic propositions are sampled from percentiles of a standard normal distribution.
- **traj_measure.py**: Provides a distribution to sample trajectories from the measure μ0 as described in the paper. Adjusting the parameter q changes the number of derivative changes (resulting in more or less oscillating trajectories), while varying σ1 alters the overall variation (the magnitude of "jumps" between trajectory points).
- **kernel.py**: Implements the kernel functions. To generate embeddings (Gram matrix) and get the similarity between formulas:
  - **compute_bag**: Calculates similarity among a list of formulas.
  - **compute_one_bag**: Computes similarity between a single formula and a list of formulas.
  - **compute_bag_bag**: Computes similarity between two lists of formulas.

###### SARSA_TDControl/EXPECTED_SARSA_TDControl/Qlearning_TDControl

Classes implementing the corresponding Reinforcement Learning algorithms, using the method interfaces of the QValueTree class.

###### QValueTree

Class implementing a memory efficient data structure to keep track of the QValues. It mirrors the Tree structure of the environment, storing a new state-action pair only when it is visited.

###### STLTreeEnv

Environment class. An STL formula is modeled as a binary tree. A state in this environment is a specific instance of a tree. The rewards are defined as the similarity between the current state and the state of the target STL. In the case in which some branches are not complete, so a meaningful STL formula cannot be retrieved from it, the corresponding reward is 0.

###### Main

Python script containing functions running a full simulation of the above RL algorithms applied to the task of finding a SLT formula similar to a target one picked at random. Each function initialize the Actions space,  generate a random formula, and run a simulation of one of the algorithms. In this context an action is a choice of an STL operator (allowed by my syntax). An STL formula is constructed populating a binary tree, picking for each node an operator, once at a time. 

