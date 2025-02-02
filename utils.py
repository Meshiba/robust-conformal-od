import random
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import fetch_openml
import torch
from torch.utils.data import Subset, ConcatDataset
from datasets import cDataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def concat(x1, x2):
    if torch.is_tensor(x1):
        x = torch.cat((x1, x2), dim=0)
    else:
        x = np.concatenate([x1, x2], axis=0)
    return x


def none_or_else(value):
    if value == 'None':
        return None
    return value


def get_run_description(args):
    file_name = ''
    file_name += str(args.dataset.replace(' ', '-')) + '_' + str(args.dataset_version)
    if args.model is not None:
        file_name += 'model_' + args.model
        if args.model == 'IF':
            file_name += '_e_' + str(args.n_estimators) + '_s_' + str(args.max_samples)
        elif args.model == 'OC-SVM':
            file_name += '_k_' + str(args.kernel_svm)
        elif args.model == 'LOF':
            file_name += '_n_' + str(args.n_estimators)
    if args.model is not None or args.dataset.startswith("scores_"):
        file_name += '_train_' + str(args.n_train) + '_'
    if args.p_train is not None:
        file_name += 'p_' + str(args.p_train) + '_'
    if args.exp_type is not None:
        file_name += 'exp_' + args.exp_type + '_'
    if args.dataset.startswith("scores_"):
        file_name += args.bb_postprocess + '_' + args.postprocess + '_' + str(args.outlr_dataset.replace(' ', '-'))
    file_name += '_initial_' + \
                 str(args.initial_labeled) + '_cal_' + str(args.n_cal) + '_p_' + str(args.p_cal) + '_test_' + \
                 str(args.n_test) + '_p_' + str(args.p_test) + '_q_' + str("-".join(args.level))
    if args.exact_clean:
        file_name += '_exact_clean'
    if args.n_seeds != 100:
        file_name += '_n_seeds_' + str(args.n_seeds)
    return file_name


def generate_data(n_inliers, n_outliers, dataset="shuttle", outlr_dataset=None,
                  dataset_version=1, dataset_path=None):
    if dataset.startswith('scores_'):
        inlr_path = os.path.join(dataset_path, dataset.removeprefix('scores_') + '_test.npy')
        outlr_path = os.path.join(dataset_path, outlr_dataset + '.npy')
        inlr = np.load(inlr_path)[1].reshape((-1,))
        outlr = np.load(outlr_path, allow_pickle=True)[1].reshape((-1,))
    else:
        X, y = fetch_openml(name=dataset, version=dataset_version, as_frame=False, return_X_y=True)
        if dataset == 'musk':
            X = X[:,1:]
        # process labels
        normal_label = {'shuttle': '1', 'KDDCup99': 'normal', 'mammography': '-1'}
        if dataset in normal_label.keys():
            if dataset == 'KDDCup99' and str(dataset_version) == '5':
                normal_label[dataset] = 'normal.'
            y_ = np.zeros(y.shape)
            y_[y == normal_label[dataset]] = 0
            y_[y != normal_label[dataset]] = 1
            y = y_
        y = y.astype(float)
        outlr, inlr = X[y == 1], X[y == 0]
    if len(outlr) < n_outliers:
        raise ValueError(f'Dataset {dataset} contains {len(outlr)} outliers but {n_outliers} required.')
    if len(inlr) < n_inliers:
        raise ValueError(f'Dataset {dataset} contains {len(inlr)} inliers but {n_inliers} required.')
    randomized = np.random.permutation(inlr.shape[0])
    inlr = inlr[randomized]
    randomized = np.random.permutation(outlr.shape[0])
    outlr = outlr[randomized]
    outliers, inliers = outlr[:n_outliers], inlr[:n_inliers]
    inlr_dataset = cDataset(inliers, np.zeros((inliers.shape[0],)))
    outlr_dataset = cDataset(outliers, np.ones((outliers.shape[0],)))
    return inlr_dataset, outlr_dataset


def generate_all_data(cal_n_inliers, cal_n_outliers,
                      test_n_inliers, test_n_outliers,
                      initial_cal=0,
                      train_n_inliers=0, train_n_outliers=0,
                      dataset="shuttle", outlr_dataset=None,
                      dataset_version=1, dataset_path=None,
                      exact_clean=False):
    # NOTE -  this function does not provide support to different outliers' types
    n_inliers = cal_n_inliers + test_n_inliers + train_n_inliers
    n_outliers = cal_n_outliers + test_n_outliers + train_n_outliers
    inlr_dataset, outlr_dataset = generate_data(n_inliers=n_inliers, n_outliers=n_outliers,
                                                dataset=dataset, outlr_dataset=outlr_dataset,
                                                dataset_version=dataset_version,
                                                dataset_path=dataset_path)
    inlr_indices = torch.randperm(len(inlr_dataset)).tolist()
    outlr_indices = torch.randperm(len(outlr_dataset)).tolist()
    # split indices data
    alloc_inlr = 0
    calib_data_inlr = inlr_indices[alloc_inlr:alloc_inlr + cal_n_inliers]
    alloc_inlr += cal_n_inliers
    test_inlr = inlr_indices[alloc_inlr:alloc_inlr + test_n_inliers]
    alloc_inlr += test_n_inliers
    train_inlr = inlr_indices[alloc_inlr:alloc_inlr + train_n_inliers]
    alloc_outlr = 0
    calib_data_outlr = outlr_indices[alloc_outlr:alloc_outlr + cal_n_outliers]
    alloc_outlr += cal_n_outliers
    test_outlr = outlr_indices[alloc_outlr:alloc_outlr + test_n_outliers]
    alloc_outlr += test_n_outliers
    train_outlr = outlr_indices[alloc_outlr:alloc_outlr + train_n_outliers]

    calib_data_y = torch.cat((torch.zeros((len(calib_data_inlr),)), torch.ones((len(calib_data_outlr),))), dim=0)
    calib_data_set = calib_data_inlr + calib_data_outlr
    randomized = np.random.permutation(len(calib_data_set))
    calib_data_set, calib_data_y = np.array(calib_data_set)[randomized], np.array(calib_data_y)[randomized]
    # sample data from calib_data
    if initial_cal > 0:
        if exact_clean:
            data_idx = random.sample(list(np.argwhere(calib_data_y == 0)), initial_cal)
        else:
            data_idx = random.sample(range(len(calib_data_set)), initial_cal)
    calib_set_inlr, calib_set_outlr = calib_data_set[calib_data_y == 0], calib_data_set[calib_data_y == 1]
    calib_dataset = ConcatDataset([Subset(inlr_dataset, calib_set_inlr), Subset(outlr_dataset, calib_set_outlr)])
    if initial_cal > 0:
        data, data_y = calib_data_set[data_idx], calib_data_y[data_idx]
        data_inlr, data_outlr = data[data_y == 0], data[data_y == 1]
        data_dataset = ConcatDataset([Subset(inlr_dataset, data_inlr), Subset(outlr_dataset, data_outlr)])
        initial_calib = Subset(inlr_dataset, data_inlr)
    else:
        data_dataset, initial_calib = [], []
    if exact_clean:
        assert len(data_inlr) == initial_cal

    test_dataset = ConcatDataset([Subset(inlr_dataset, test_inlr), Subset(outlr_dataset, test_outlr)])
    train_dataset = ConcatDataset([Subset(inlr_dataset, train_inlr), Subset(outlr_dataset, train_outlr)])
    return calib_dataset, test_dataset, data_dataset, train_dataset, initial_calib


def get_model(model, n_estimators, max_samples, kernel_svm, device="cpu"):
    if model == 'IF':
        return IsolationForest(n_estimators=n_estimators,
                               max_samples=max_samples)
    elif model == 'OC-SVM':
        return OneClassSVM(kernel=kernel_svm)
    elif model == 'LOF':
        return LocalOutlierFactor(n_neighbors=n_estimators, novelty=True)
    else:
        raise ValueError(f'the following model is not supported - {model}')


def get_latent_rep(dataset_):
    # load all data
    data_loader = torch.utils.data.DataLoader(dataset_, batch_size=len(dataset_),
                                               num_workers=2, shuffle=True, drop_last=False)
    data, labels = next(iter(data_loader))
    return data, labels.numpy()


def draw_number_of_outliers_inliers(n_cal, p_cal, n_test, p_test, n_train, p_train, initial_cal, random=False):
    if random:
        cal_n_outliers = np.random.binomial(n_cal, float(p_cal))
        train_n_outliers = np.random.binomial(n_train, float(p_train))
        test_n_outliers = np.random.binomial(n_test, float(p_test))
    else:
        cal_n_outliers = int(n_cal * float(p_cal))
        train_n_outliers = int(n_train * float(p_train))
        test_n_outliers = int(n_test * float(p_test))
    if initial_cal > 0:
        real_p_cal = float(cal_n_outliers / n_cal)
        data_n_outliers = np.random.binomial(initial_cal, float(real_p_cal))
        data_n_inliers = initial_cal - data_n_outliers
    else:
        real_p_cal = float(cal_n_outliers / n_cal)
        np.random.binomial(10, float(real_p_cal))
        data_n_outliers, data_n_inliers = 0, 0
    cal_n_inliers = n_cal - cal_n_outliers
    train_n_inliers = n_train - train_n_outliers
    test_n_inliers = n_test - test_n_outliers
    return cal_n_inliers, cal_n_outliers, test_n_inliers, test_n_outliers, data_n_inliers, \
           data_n_outliers, train_n_inliers, train_n_outliers




