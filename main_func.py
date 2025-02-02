import shutil
import tempfile
import time
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
import torch
from utils import set_seed, get_model, generate_all_data, get_latent_rep, draw_number_of_outliers_inliers
from experiments.utils import create_command, run_command, get_params_for_one_seed_command, check_for_job_state
from algo import get_rejections_indices, analyze_performance, get_calibration_set


def run_comparison(seed=42, n_seeds=100, mu_outlier=4, initial_cal=0,
                   n_cal=10000, p_cal=0.05,
                   n_test=10000, p_test=0.05,
                   n_train=10000, p_train=0.05,
                   level=0.1, p_trim=0.1,
                   model=None,
                   dataset="shuttle", outlr_dataset=None, dataset_version=1,
                   n_features=100, max_samples="auto",
                   n_estimators=10, kernel_svm="rbf", args_dict=None,
                   device="cpu",
                   dataset_path=None, distribute=True, save_path=None, slurm=True, 
                   exact_clean=False):
    if not distribute or n_seeds == 1:
        return run_one_comparison(seed=seed, n_seeds=n_seeds, mu_outlier=mu_outlier, initial_cal=initial_cal,
                   n_cal=n_cal, p_cal=p_cal,
                   n_test=n_test, p_test=p_test,
                   n_train=n_train, p_train=p_train,
                   level=level, p_trim=p_trim,
                   model=model,
                   dataset=dataset, outlr_dataset=outlr_dataset, dataset_version=dataset_version,
                   n_features=n_features, max_samples=max_samples,
                   n_estimators=n_estimators, kernel_svm=kernel_svm, args_dict=args_dict,
                   device=device,
                   dataset_path=dataset_path,
                   exact_clean=exact_clean)
    results = pd.DataFrame({})
    all_process = []
    all_commands = []
    all_result_paths = []
    set_seed(seed)
    # create tmp results dir in save_path
    tmp_files_path = save_path + '/tmp_results/'
    try:
        os.makedirs(tmp_files_path)
    except:
        pass
    seed_list = random.sample(range(1, 999999), n_seeds)
    for seed_ in tqdm(seed_list):
        # create tmp result dir for this specific run
        tmp_dir_path = tempfile.mkdtemp(dir=tmp_files_path)
        all_result_paths.append(tmp_dir_path)
        params, flag_params = get_params_for_one_seed_command(seed=seed_, args_dict=args_dict)
        command = create_command(tmp_dir_path + '/', params, flag_params)
        if slurm:
            command = f"bash ./create_tmp_empty.sh \"{command}\""
        process = run_command(command, read_out=True)
        all_process.append(process)
        all_commands.append(command)
    # wait for all processes to finish
    if slurm:
        job_ids = []
        for process in all_process:
            output, _ = process.communicate()
            job_id = str(output).split(' ')[-1].strip()
            job_ids.append(job_id)
        all_job_ids = job_ids.copy()
        while len(job_ids):
            state = check_for_job_state(job_ids[0])
            if state == 'complete':
                del job_ids[0]
            else:
                time.sleep(30)
    while len(all_process):
        all_process[0].communicate()
        # load results
        if os.path.isfile(all_result_paths[0] + '/results/results.pkl'):
            curr_result = pd.read_pickle(all_result_paths[0] + '/results/results.pkl')
            results = pd.concat([results, curr_result])
            # delete tmp dir results for this run
            shutil.rmtree(all_result_paths[0])
            del all_process[0]
            del all_commands[0]
            del all_result_paths[0]
            if slurm:
                del all_job_ids[0]
        else:
            shutil.rmtree(tmp_files_path, ignore_errors=True)
            job_desc = f'Failed job id {all_job_ids[0]}\n' if slurm else ''
            raise ValueError(f'{job_desc}Deleting tmp results dir...\nThe following run falied to save results file:\n{all_commands[0]}')
    # delete tmp results folder
    shutil.rmtree(tmp_files_path, ignore_errors=True)
    return results


def run_one_comparison(seed=42, n_seeds=100, mu_outlier=4, initial_cal=0,
                       n_cal=10000, p_cal=0.05,
                       n_test=10000, p_test=0.05,
                       n_train=10000, p_train=0.05,
                       level=0.1, p_trim=0.1,
                       model=None,
                       dataset="shuttle", outlr_dataset=None, dataset_version=1,
                       n_features=100, max_samples="auto",
                       n_estimators=10, kernel_svm="rbf", args_dict=None,
                       device="cpu",
                       dataset_path=None,
                       exact_clean=False):
    results = pd.DataFrame({})
    methods = ['Naive', 'NT', 'Oracle']
    # update params in args_dict
    if initial_cal >= 0:
        methods.append('Clean')
        methods.append('LT')
    set_seed(seed)
    if n_seeds == 1:
        seed_list = [seed]
    else:
        seed_list = random.sample(range(1, 999999), n_seeds)
    for seed_ in tqdm(seed_list):
        set_seed(seed_)
        # random hypotheses - draw number of samples
        cal_n_inliers, cal_n_outliers, test_n_inliers, test_n_outliers, data_n_inliers, data_n_outliers, \
        train_n_inliers, train_n_outliers = draw_number_of_outliers_inliers(n_cal, p_cal, n_test, p_test,
                                                           n_train, p_train, initial_cal, random=False)
        p_outliers_cal = float(cal_n_outliers / (cal_n_outliers + cal_n_inliers))
        p_outliers_test = float(test_n_outliers /
                                (test_n_outliers + test_n_inliers))
        # generate data
        calib_dataset, test_dataset, data_dataset, train_dataset, initial_calib = \
            generate_all_data(cal_n_inliers=cal_n_inliers, cal_n_outliers=cal_n_outliers,
                              test_n_inliers=test_n_inliers, test_n_outliers=test_n_outliers,
                              initial_cal=initial_cal,
                              train_n_inliers=train_n_inliers, train_n_outliers=train_n_outliers,
                              dataset=dataset, outlr_dataset=outlr_dataset,
                              dataset_version=dataset_version,
                              dataset_path=dataset_path, exact_clean=exact_clean)
        # model
        if not dataset.startswith('scores_'):
            model_ = get_model(model, n_estimators, max_samples, kernel_svm, device=device)
            # train model
            train_set, _ = get_latent_rep(train_dataset)
            if len(train_set.shape) == 1 or train_set.shape[1] == 1:
                train_set = train_set.reshape((-1,1))
            model_.fit(train_set)
        else:
            model_ = None
        # transform all data to scores
        calib_set, calib_y = get_latent_rep(calib_dataset)
        test_set, test_y = get_latent_rep(test_dataset)
        if initial_cal > 0:
            initial_calib_set, _ = get_latent_rep(initial_calib)
            data, labels = get_latent_rep(data_dataset)
        else:
            initial_calib_set = np.array([])
            data, labels = np.array([]), np.array([])
        if len(calib_set.shape) == 1 or calib_set.shape[1] == 1:
            calib_set = calib_set.reshape((-1,1))
            test_set = test_set.reshape((-1,1))
            if len(data):
                data = data.reshape((-1,1))
                initial_calib_set = initial_calib_set.reshape((-1,1))

        if dataset.startswith('scores_'):
            calib_set = -1 * calib_set.numpy()
            test_set = -1 * test_set.numpy()
            if initial_cal > 0:
                initial_calib_set = -1 * initial_calib_set.numpy()
                data = -1 * data.numpy()
        else:
            calib_set = -1 * model_.decision_function(calib_set)
            test_set = -1 * model_.decision_function(test_set)
            if initial_cal > 0:
                initial_calib_set = -1 * model_.decision_function(initial_calib_set)
                data = -1 * model_.decision_function(data)

        if torch.is_tensor(calib_set):
            calib_set, test_set = calib_set.numpy(), test_set.numpy()
            if initial_cal > 0:
                initial_calib_set = initial_calib_set.numpy()
                data = data.numpy()

        noise_level = (10**-15)
        calib_set = calib_set.reshape((-1,))
        noise = np.random.normal(0, 1, size=calib_set.shape) * noise_level
        calib_set += noise
        test_set = test_set.reshape((-1,))
        noise = np.random.normal(0, 1, size=test_set.shape) * noise_level
        test_set += noise
        if len(initial_calib_set):
            initial_calib_set = initial_calib_set.reshape((-1,))
            noise = np.random.normal(0, 1, size=initial_calib_set.shape) * noise_level
            initial_calib_set += noise

        for curr_method in methods:
            curr_calib_set, curr_calib_y, trimmed_info = get_calibration_set(curr_method, initial_cal,
                                                                             initial_calib_set, calib_set, calib_y,
                                                                             p_trim)
            if isinstance(level, list):
                levels = level
            else:
                levels = [level]
            for curr_level in levels:
                curr_level = float(curr_level)

                if curr_calib_set is not None and len(curr_calib_set):
                    curr_rejections, curr_threshold = get_rejections_indices(curr_calib_set, test_set, level=curr_level)
                    curr_power, curr_type1 = analyze_performance(curr_rejections, test_y)
                else:
                    curr_calib_set, curr_calib_y = np.array([]), np.array([])
                    curr_rejections, curr_threshold = [], np.inf
                    curr_power, curr_type1 = 0, 0
                curr_results_dict = {'Type': curr_method, 'Power': curr_power,
                                     'Type-1-Error': curr_type1, 'Threshold': curr_threshold,
                                     'Rejections': len(curr_rejections), 'Seed': seed_,
                                     'Signal amplitude': mu_outlier,
                                     'Outliers proportion (cal)': p_cal,
                                     'Outliers proportion (test)': p_test,
                                     'Actual Outliers proportion (cal)': p_outliers_cal,
                                     'Actual Outliers proportion (test)': p_outliers_test,
                                     'Clean calibration size': initial_cal,
                                     'Calibration size': len(curr_calib_set),
                                     'Original calibration size': len(calib_set),
                                     'Calibration inliers': np.sum(curr_calib_y == 0),
                                     'Calibration outliers': np.sum(curr_calib_y == 1),
                                     'Calibration outliers proportion': np.sum(curr_calib_y == 1) / len(curr_calib_y),
                                     'Labeled data inliers': np.sum(labels == 0),
                                     'Labeled data outliers': np.sum(labels == 1),
                                     'level': curr_level}
                if trimmed_info is not None:
                    n_trim, trim_label_samples, m_th = trimmed_info
                    curr_results_dict['Trimmed'] = n_trim
                    curr_results_dict['Trimmed Inliers'] = np.sum(trim_label_samples == 0)
                    curr_results_dict['Trimmed Outliers (proportion)'] = np.sum(trim_label_samples == 1) / (cal_n_outliers)
                    curr_results_dict['Trimmed Outliers'] = np.sum(trim_label_samples == 1)
                    curr_results_dict['Model-Threshold'] = m_th
                curr_result = pd.DataFrame(curr_results_dict, index=[0])
                results = pd.concat([results, curr_result])

    # add all args to results
    if args_dict is not None:
        for k,v in args_dict.items():
            if isinstance(v, list):
                continue
            results[k] = v
    return results


def run_exp(exp_type, n_cal, n_test, p_cal, p_test, level,
            mu_outlier=4, seed=42, initial_cal=0, n_seeds=100,
            model=None,
            n_train=0, p_train=None, dataset='shuttle', outlr_dataset=None,
            dataset_version=1, n_features=100, max_samples="auto",
            n_estimators=10, kernel_svm="rbf", args_dict=None,
            exp_params=[], device="cpu",
            dataset_path=None, save_path=None, distribute=True, slurm=True, 
            exact_clean=False):
    all_results = pd.DataFrame({})
    curr_args_dict = args_dict.copy()
    if exp_type == 'outliers_calib':
        if len(exp_params):
            prop = exp_params
        else:
            prop = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        for o_prop in prop:
            curr_p_cal = o_prop
            curr_args_dict['p_cal'] = o_prop
            if p_train is None:
                curr_p_train = o_prop
            else:
                curr_p_train = p_train
            if p_test is None:
                curr_p_test = o_prop
            else:
                curr_p_test = p_test
            results = run_comparison(seed=seed, n_seeds=n_seeds, mu_outlier=mu_outlier,
                           n_cal=n_cal, p_cal=curr_p_cal,
                           n_test=n_test, p_test=curr_p_test,
                           n_train=n_train, p_train=curr_p_train,
                           level=level, p_trim=o_prop, initial_cal=initial_cal,
                           model=model,
                           dataset=dataset, outlr_dataset=outlr_dataset, dataset_version=dataset_version,
                           n_features=n_features, max_samples=max_samples,
                           n_estimators=n_estimators, kernel_svm=kernel_svm, args_dict=curr_args_dict, 
                           device=device, dataset_path=dataset_path, save_path=save_path, distribute=distribute,
                           slurm=slurm, exact_clean=exact_clean)
            all_results = pd.concat([all_results, results])
    elif exp_type == 'labeling_budget' or exp_type == 'calib_size':
        if len(exp_params):
            calib_size_list = [int(p) for p in exp_params]
        else:
            calib_size_list = [100, 200, 300]
        if p_test is None:
            p_test = p_cal
        if p_train is None:
            p_train = p_cal
        for calib_size_ in calib_size_list:
            if exp_type == 'labeling_budget':
                initial_cal = calib_size_
                curr_args_dict['initial_labeled'] = calib_size_
            else:
                n_cal = calib_size_
                curr_args_dict['n_cal'] = calib_size_
            results = run_comparison(seed=seed, n_seeds=n_seeds, mu_outlier=mu_outlier,
                           n_cal=n_cal, p_cal=p_cal,
                           n_test=n_test, p_test=p_test,
                           n_train=n_train, p_train=p_train,
                           level=level, p_trim=p_cal, initial_cal=initial_cal,
                           model=model,
                           dataset=dataset, outlr_dataset=outlr_dataset, dataset_version=dataset_version,
                           n_features=n_features, max_samples=max_samples,
                           n_estimators=n_estimators, kernel_svm=kernel_svm, args_dict=curr_args_dict,
                           device=device, dataset_path=dataset_path, save_path=save_path, distribute=distribute,
                           slurm=slurm, exact_clean=exact_clean)
            all_results = pd.concat([all_results, results])
    elif exp_type == 'signal_amp':
        if len(exp_params):
            mu_list = exp_params
        else:
            mu_list = [1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4]
        if p_test is None:
            p_test = p_cal
        if p_train is None:
            p_train = p_cal
        for mu_ in mu_list:
            curr_args_dict['mu_outlier'] = mu_
            results = run_comparison(seed=seed, n_seeds=n_seeds, mu_outlier=mu_,
                           n_cal=n_cal, p_cal=p_cal,
                           n_test=n_test, p_test=p_test,
                           n_train=n_train, p_train=p_train,
                           level=level, p_trim=p_cal, initial_cal=initial_cal,
                           model=model,
                           dataset=dataset, outlr_dataset=outlr_dataset, dataset_version=dataset_version,
                           n_features=n_features, max_samples=max_samples,
                           n_estimators=n_estimators, kernel_svm=kernel_svm, args_dict=curr_args_dict,
                           device=device, dataset_path=dataset_path, save_path=save_path, distribute=distribute,
                           slurm=slurm, exact_clean=exact_clean)
            all_results = pd.concat([all_results, results])
    elif exp_type == 'levels':
        if len(exp_params):
            level_list = exp_params
        else:
            level_list = ['0.01', '0.02', '0.03', '0.04', '0.05']
        level = " ".join(level_list)
        curr_args_dict['level'] = level
        all_results = run_comparison(seed=seed, n_seeds=n_seeds, mu_outlier=mu_outlier,
                           n_cal=n_cal, p_cal=p_cal,
                           n_test=n_test, p_test=p_test,
                           n_train=n_train, p_train=p_train,
                           level=level, p_trim=p_cal, initial_cal=initial_cal,
                           model=model,
                           dataset=dataset, outlr_dataset=outlr_dataset, dataset_version=dataset_version,
                           n_features=n_features, max_samples=max_samples,
                           n_estimators=n_estimators, kernel_svm=kernel_svm, args_dict=curr_args_dict,
                           device=device, dataset_path=dataset_path, save_path=save_path, distribute=distribute,
                           slurm=slurm, exact_clean=exact_clean)
    else:
        raise ValueError(f'The following experiment type is not supported - {exp_type}')
    return all_results
