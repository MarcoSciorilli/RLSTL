#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import torch
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import Ridge
import numpy as np
import pickle

from src.kernel import StlKernel, KernelRegression
from src.traj_measure import BaseMeasure
from src.phis_generator import StlGenerator
from src.utils import dump_pickle, load_pickle

# device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {device}".format(device=device))
# stl generator parameters
prob_unbound_time_operator = 0.1  # probability of a temporal operator to have a time bound of the type [0,infty]
prob_right_unbound_time_operator = 0.1  # probability of a temporal operator to have a time bound of the type [a, infty]
leaf_probability = 0.5  # probability of generating a leaf (always zero for root)
atom_threshold_sd = 1.0  # std for the normal distribution of the thresholds of atoms
# TODO: the following should be adapted on test case (for test formulae!!!)
time_bound_max_range = 29  # maximum value of time span of a temporal operator
max_timespan = 29

sampler = StlGenerator(leaf_prob=leaf_probability, time_bound_max_range=time_bound_max_range,
                       unbound_prob=prob_unbound_time_operator, right_unbound_prob=prob_right_unbound_time_operator,
                       threshold_sd=atom_threshold_sd, inner_node_prob=[0.2, 0.2, 0.2, 0.2, 0.2, 0],
                       max_timespan=max_timespan)

# trajectory sample parameters
initial_std = 1.0  # standard deviation of normal distribution of initial state
total_var_std = 1.0  # standard deviation of normal distribution of total variation

mu0 = BaseMeasure(device=device, sigma0=initial_std, sigma1=total_var_std, q=0.1)

# experiment parameters and dataset generation
# or load something like
# torch.from_numpy(load_pickle(os.getcwd() + os.path.sep + 'trajectories', 'sirs_trajectories.pickle'))
test_traj = None
task_idx = 0
task_dict = {0: 'average_robustness', 1: 'satisfaction_probability', 2: 'single_trajectory'}
folder_name = None  # or some custom name (e.g. the trajectory distribution name + the task name)
base_folder = os.getcwd()
sub_folder = task_dict[task_idx] if folder_name is None else folder_name
# creating subdirectory
folder = base_folder + os.path.sep + sub_folder
if not os.path.isdir(folder):
    os.mkdir(folder)
os.chdir(folder)
n_experiments = 30
resample_formulae = True  # whether to run independent experiments
# whether to use normalized robustness or standard robustness
norm_rob = [True] if task_idx != 1 else [False]
train_sizes = [1000]  # , 2000, 3000, 4000, 5000]
val_size = 250
test_size = 1000
n_vars = [3]
n_traj_test = 1000
n_traj_points = 100
n_detailed_exps = 10
# cv parameters
alpha_min = -3
alpha_max = 1
cv_steps = 10

# parameters for statistical tables
quantiles = torch.tensor([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
# for satisfaction probability
logit = lambda x: torch.max(torch.min(x, torch.ones(len(x))), torch.zeros(len(x)))
do_kpca = False  # whether to do prediction with reduced dimensionality embeddings

for n_var in n_vars:
    exp_folder = f'{n_var}_var'
    # creating subdirectory
    sub_folder = folder + os.path.sep + exp_folder
    if not os.path.isdir(sub_folder):
        os.mkdir(sub_folder)
    os.chdir(sub_folder)
    for train_size in train_sizes:
        if do_kpca:
            step = 50 if train_size < 2000 else 100
            n_comp = np.arange(100, train_size + 1, step)
            xai_case = 1 + 2 * n_var
            n_comp = np.append(xai_case, n_comp)
        train_folder = f'train_size_{train_size}'
        train_sub_folder = sub_folder + os.path.sep + train_folder
        if not os.path.isdir(train_sub_folder):
            os.mkdir(train_sub_folder)
        os.chdir(train_sub_folder)
        # containers to store results and statistics
        kreg_MRE, kreg_MSE, kreg_MAE, kreg_MRE_norm, kreg_MSE_norm, kreg_MAE_norm = \
            [np.zeros(n_experiments) for k in range(6)]
        kreg_stat_MRE, kreg_stat_MAE, kreg_stat_MSE, kreg_stat_MRE_norm, kreg_stat_MAE_norm, kreg_stat_MSE_norm = \
            [np.zeros((n_experiments, len(quantiles))) for i in range(6)]
        if do_kpca:
            kpca_MRE, kpca_MSE, kpca_MAE, kpca_MRE_norm, kpca_MSE_norm, kpca_MAE_norm = \
                [np.zeros((n_experiments, len(n_comp))) for k in range(6)]
            kpca_stat_MRE, kpca_stat_MAE, kpca_stat_MSE, kpca_stat_MRE_norm, kpca_stat_MAE_norm, kpca_stat_MSE_norm = \
                [np.zeros((len(n_comp), n_experiments, len(quantiles))) for i in range(6)]

        for i in range(n_experiments):
            if i < n_detailed_exps:
                # creating subdirectory
                cur_exp_folder = f"exp_{i}"
                sub_sub_folder = train_sub_folder + os.path.sep + cur_exp_folder
                if not os.path.isdir(sub_sub_folder):
                    os.mkdir(sub_sub_folder)
                os.chdir(sub_sub_folder)
            print("\nNumber of variables: ", n_var, "Training Size: ", train_size, "Experiment number: ", i)
            print("Initializing...")
            init_start = time.time()
            # test trajectories of current experiment  samples=n_traj_samples
            n_traj_samples = n_traj_test if task_idx < 2 else 1
            traj = mu0.sample(samples=n_traj_samples, varn=n_var, points=n_traj_points)
            if test_traj is not None:
                idx = torch.randperm(test_traj.shape[0], device=device)[:n_traj_samples].to(device)
                traj = test_traj[idx, :, :].to(device)

            if i == 0 or (i > 0 and resample_formulae):
                # instantiate the kernel
                kernel = StlKernel(mu0, samples=10000, sigma2=0.44, varn=n_var)
                regressor = KernelRegression(kernel, cross_validate=True)

                # generating the dataset
                phi_train = sampler.bag_sample(bag_size=train_size, nvars=n_var)
                phi_test = sampler.bag_sample(bag_size=test_size, nvars=n_var)
                phi_val = sampler.bag_sample(bag_size=val_size, nvars=n_var)

                if i < n_detailed_exps:
                    with open(f'test_phis.pickle', 'wb') as f:
                        pickle.dump(phi_test, f)
                    with open(f'train_phis.pickle', 'wb') as f:
                        pickle.dump(phi_train, f)

            init_end = time.time()
            print("..initialization done in {time:.5f} s".format(time=init_end - init_start))

            for norm in norm_rob:
                print("Computing robustness (norm={norm})..".format(norm=norm))
                data_start = time.time()
                obs_train = torch.zeros(len(phi_train))
                for j, phi in enumerate(phi_train):
                    if task_idx != 1:
                        obs_train[j] = torch.mean(phi.quantitative(traj, norm))
                    else:
                        obs_train[j] = (torch.sum(phi.boolean(traj), dtype=torch.float32) + 1) / (n_traj_samples + 2)

                obs_test = torch.zeros(len(phi_test))
                for j, phi in enumerate(phi_test):
                    if task_idx != 1:
                        obs_test[j] = torch.mean(phi.quantitative(traj, norm))
                    else:
                        obs_test[j] = (torch.sum(phi.boolean(traj), dtype=torch.float32) + 1) / (n_traj_samples + 2)

                if i < n_detailed_exps:
                    # saving ground truth labels for test set
                    with open(f"obs_test_norm={norm}.pickle", "wb") as f:
                        pickle.dump(obs_test.numpy(), f)
                    with open(f'obs_train_norm={norm}.pickle', 'wb') as f:
                        pickle.dump(obs_train.numpy(), f)

                obs_val = torch.zeros(len(phi_val))
                for j, phi in enumerate(phi_val):
                    if task_idx != 1:
                        obs_val[j] = torch.mean(phi.quantitative(traj, norm))
                    else:
                        obs_val[j] = (torch.sum(phi.boolean(traj), dtype=torch.float32) + 1) / (n_traj_samples + 2)

                data_end = time.time()
                print("..finished after {time:.5f} s".format(time=data_end - data_start))

                # train the regressor
                print("Training and testing kernel ridge regression..")
                kreg_start = time.time()
                regressor.train(phi_train, obs_train, phi_val, obs_val)

                # test the kernel regressor on current dataset
                kreg_mae, kreg_mse, p = regressor.test(phi_test, obs_test)
                kreg_ae = torch.abs(p - obs_test)
                kreg_re = torch.abs(p - obs_test) / torch.abs(obs_test)
                kreg_se = (p - obs_test) * (p - obs_test)
                kreg_mre = torch.mean(kreg_re).numpy()

                kreg_end = time.time()
                print("..done in {time:.5f}".format(time=kreg_end - kreg_start))
                if i < n_detailed_exps:
                    # saving kernel regressor's predictions
                    with open(f"kernel_regression_norm={norm}.pickle", "wb") as f:
                        pickle.dump(p.numpy(), f)

                if norm:
                    kreg_MRE_norm[i], kreg_MAE_norm[i], kreg_MSE_norm[i] = [kreg_mre, kreg_mae, kreg_mse]
                    kreg_stat_MRE_norm[i, :], kreg_stat_MAE_norm[i, :], kreg_stat_MSE_norm[i, :] = \
                        [torch.quantile(mes, quantiles).numpy() for mes in [kreg_re, kreg_ae, kreg_se]]
                else:
                    kreg_MRE[i], kreg_MAE[i], kreg_MSE[i] = [kreg_mre, kreg_mae, kreg_mse]
                    kreg_stat_MRE[i, :], kreg_stat_MAE[i, :], kreg_stat_MSE[i, :] = \
                        [torch.quantile(mes, quantiles).numpy() for mes in [kreg_re, kreg_ae, kreg_se]]

                if do_kpca:
                    # evaluate kernel
                    gram_train = kernel.compute_bag_bag(phi_train, phi_train).cpu().numpy()
                    gram_val = kernel.compute_bag_bag(phi_val, phi_train).cpu().numpy()
                    gram_test = kernel.compute_bag_bag(phi_test, phi_train).cpu().numpy()

                    if do_kpca:
                        print("Computing kPCA with all components (to be reduced later)...")
                        kpca_start = time.time()
                        # instantiate sklearn kPCA
                        kpca = KernelPCA(n_components=train_size, kernel='precomputed')
                        kpca.fit(gram_train)
                        reduced_train_kpca = kpca.transform(gram_train)
                        reduced_val_kpca = kpca.transform(gram_val)
                        reduced_test_kpca = kpca.transform(gram_test)
                        kpca_end = time.time()
                        print("..done in {time:.5f}".format(time=kpca_end - kpca_start))
                    print("kPCA on different number of components..")
                    # cv utilities
                    alphas = np.linspace(alpha_min, alpha_max, cv_steps)
                    if do_kpca:
                        chosen_alphas_kpca = np.zeros(len(n_comp))
                    # range over number of components
                    print("Different number of components..")
                    comp_start = time.time()
                    for w, n_c in enumerate(n_comp):
                        if do_kpca:
                            # cross-validation
                            current_loss_kpca = np.zeros(cv_steps)
                            for k, alpha in enumerate(alphas):
                                current_ridge_kpca = Ridge(alpha=abs(pow(10, alpha)), fit_intercept=True)
                                current_ridge_kpca.fit(reduced_train_kpca[:, :n_c], obs_train.cpu().numpy())
                                val_predicted_kpca = current_ridge_kpca.predict(reduced_val_kpca[:, :n_c])
                                current_loss_kpca[k] = np.mean((val_predicted_kpca - obs_val.cpu().numpy()) *
                                                               (val_predicted_kpca - obs_val.cpu().numpy()))
                            cv_min_idx_kpca = np.argmin(current_loss_kpca)
                            chosen_alphas_kpca[w] = alphas[cv_min_idx_kpca]
                            # instantiate sklearn ridge regression model
                            ridge_kpca = Ridge(alpha=abs(pow(10, chosen_alphas_kpca[w])), fit_intercept=True)
                            # training
                            ridge_kpca.fit(reduced_train_kpca[:, :n_c], obs_train.cpu().numpy())
                            # testing
                            y_predicted_kpca = ridge_kpca.predict(reduced_test_kpca[:, :n_c])
                            if i < n_detailed_exps:
                                # saving kpca predictions for this experiment
                                with open(f"kpca_norm={norm}_components={n_c}.pickle", "wb") as f:
                                    pickle.dump(y_predicted_kpca, f)
                            # measuring performance
                            kpca_ae = np.abs(obs_test.cpu().numpy() - y_predicted_kpca)
                            mae_kpca = np.mean(kpca_ae)
                            kpca_se = np.power(y_predicted_kpca - obs_test.cpu().numpy(), 2)
                            mse_kpca = np.mean(kpca_se)
                            kpca_re = np.abs(y_predicted_kpca - obs_test.cpu().numpy()) / np.abs(obs_test.cpu().numpy())
                            mre_kpca = np.mean(kpca_re)

                        # save results
                        if norm:
                            if do_kpca:
                                kpca_MAE_norm[i, w], kpca_MRE_norm[i, w], kpca_MSE_norm[i, w] = \
                                    [mae_kpca, mre_kpca, mse_kpca]
                                kpca_stat_MRE_norm[w, i, :], kpca_stat_MAE_norm[w, i, :], kpca_stat_MSE_norm[w, i, :] \
                                    = [np.quantile(mes, quantiles.numpy()) for mes in [kpca_re, kpca_ae, kpca_se]]
                        else:
                            if do_kpca:
                                kpca_MAE[i, w], kpca_MRE[i, w], kpca_MSE[i, w] = [mae_kpca, mre_kpca, mse_kpca]
                                kpca_stat_MRE[w, i, :], kpca_stat_MAE[w, i, :], kpca_stat_MSE[w, i, :] = \
                                    [np.quantile(mes, quantiles.numpy()) for mes in [kpca_re, kpca_ae, kpca_se]]

                    comp_end = time.time()
                    print("..all components done in {time:.5f}".format(time=comp_end - comp_start))
        os.chdir(train_sub_folder)
        files = ['kreg_mae', 'kreg_mse', 'kreg_mre', 'kreg_mae_norm', 'kreg_mse_norm', 'kreg_mre_norm',
                 'kreg_mae_stat', 'kreg_mre_stat', 'kreg_mse_stat', 'kreg_mae_stat_norm', 'kreg_mre_stat_norm',
                 'kreg_mse_stat_norm']
        things = [kreg_MAE, kreg_MSE, kreg_MRE, kreg_MAE_norm, kreg_MSE_norm, kreg_MRE_norm, kreg_stat_MAE,
                  kreg_stat_MRE, kreg_stat_MSE, kreg_stat_MAE_norm, kreg_stat_MRE_norm, kreg_stat_MSE_norm]
        if do_kpca:
            kpca_files = ['kpca_mae', 'kpca_mse', 'kpca_mre', 'kpca_mae_norm', 'kpca_mse_norm', 'kpca_mre_norm',
                          'kpca_mae_stat', 'kpca_mre_stat', 'kpca_mse_stat', 'kpca_mae_stat_norm', 'kpca_mre_stat_norm',
                          'kpca_mse_stat_norm']
            kpca_things = [kpca_MAE, kpca_MSE, kpca_MRE, kpca_MAE_norm, kpca_MSE_norm, kpca_MRE_norm, kpca_stat_MAE,
                           kpca_stat_MRE, kpca_stat_MSE, kpca_stat_MAE_norm, kpca_stat_MRE_norm, kpca_stat_MSE_norm]
            files = files + kpca_files
            things = things + kpca_things
        for file, thing in zip(files, things):
            dump_pickle(file, thing)
