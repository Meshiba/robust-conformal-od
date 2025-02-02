import os
import errno
import argparse
import torch
from main_func import run_exp, run_comparison
from utils import get_run_description, none_or_else
from utils_plot import plot, plot_xy


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.full_save_path:
        save_path = args.save_path + '/'
    else:
        save_path = args.save_path + '/' + get_run_description(args)
    if not os.path.exists(save_path + '/results/'):
        try:
            os.makedirs(save_path + '/results/', exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if args.plot:
        if not os.path.exists(save_path + '/plots/'):
            try:
                os.makedirs(save_path + '/plots/', exist_ok=True)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
    if args.dataset.startswith('scores_'):
        args.n_train = 0
    if args.exp_type is not None:
        results = run_exp(seed=args.seed, exp_type=args.exp_type, n_cal=args.n_cal,
                           n_test=args.n_test, p_cal=args.p_cal,
                           p_test=args.p_test, level=args.level,
                           mu_outlier=args.mu_outlier,
                           initial_cal=args.initial_labeled, n_seeds=args.n_seeds,
                           model=args.model,
                           n_train=args.n_train, p_train=args.p_train, dataset=args.dataset,
                           outlr_dataset=args.outlr_dataset, dataset_version=args.dataset_version,
                           n_features=args.n_features,
                           max_samples=args.max_samples,
                           n_estimators=args.n_estimators, kernel_svm=args.kernel_svm,
                           args_dict=dict(vars(args)),
                           exp_params=args.exp_params,
                           device=device, dataset_path=args.dataset_path, save_path=save_path,
                           distribute=not args.no_distribute, slurm=not args.local,
                           exact_clean=args.exact_clean)
    else:
        if args.p_test is None:
            args.p_test = args.p_cal
        if args.p_train is None:
            args.p_train = args.p_cal
        results = run_comparison(seed=args.seed, n_seeds=args.n_seeds, mu_outlier=args.mu_outlier,
                                  n_cal=args.n_cal, p_cal=args.p_cal,
                                  n_test=args.n_test, p_test=args.p_test,
                                  n_train=args.n_train, p_train=args.p_train,
                                  initial_cal=args.initial_labeled,
                                  level=args.level,
                                  p_trim=args.p_cal,
                                  model=args.model,
                                  dataset=args.dataset,
                                  outlr_dataset=args.outlr_dataset, dataset_version=args.dataset_version,
                                  n_features=args.n_features,
                                  max_samples=args.max_samples,
                                  n_estimators=args.n_estimators, kernel_svm=args.kernel_svm,
                                  args_dict=dict(vars(args)),
                                  device=device,
                                  dataset_path=args.dataset_path, save_path=save_path,
                                  distribute=not args.no_distribute, slurm=not args.local,
                                  exact_clean=args.exact_clean)

    # save raw results
    results.to_pickle(save_path + '/results/results.pkl')
    # save plots
    if args.plot:
        if args.exp_type is not None:
            desc = {'outliers_calib': 'Outliers proportion (cal)',
                    'calib_size': 'Original calibration size', 'labeling_budget': 'Clean calibration size',
                    'signal_amp': 'Signal amplitude', 'levels': 'level'}
            plot_xy(results, x=desc[args.exp_type], level=args.level, save_path=(save_path + '/plots/'),
                    extract_data=True, exp=True)
        else:
            plot(results=results, level=args.level, save_path=(save_path + '/plots/'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_save_path', action='store_true', help='Do not add intermediate folder with run description.')
    parser.add_argument('--local', action='store_true', help='Run local experiments (not via SLURM).')
    parser.add_argument('--exp_type', type=none_or_else, default=None, choices=[None, 'outliers_calib', 'calib_size',
                                                                       'labeling_budget', 'signal_amp', 'levels'])
    parser.add_argument('--exp_params', type=none_or_else, default=[], nargs='+', help='List of the parameters for the '
                                                                                'experiment.')
    parser.add_argument('--plot', action='store_true', help='Create plot figures, save and present them.')
    parser.add_argument('--no_distribute', action='store_true', help='Do not distribute. Run all seeds in the same '
                                                                     'process.')
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--seed', type=int, default=42, help='Initial seed.')
    parser.add_argument('--n_seeds', type=int, default=100, help='Number of runs. Each run correspond to different '
                                                                 'seed.')

    parser.add_argument('--initial_labeled', type=int, default=0, help='Number of labeled samples.')
    parser.add_argument('--exact_clean', action='store_true', help='Draw inital_labeled clean samples.')
    parser.add_argument('--n_cal', type=int, default=1000, help='Number of calibration samples.')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples.')
    parser.add_argument('--p_cal', type=float, default=0.05, help='Proportion of outliers in the calibration-set.')
    parser.add_argument('--p_test', type=none_or_else, default=None, help='Proportion of outliers in the test-set (is None - same proportion as in the calibration set).')

    parser.add_argument('--level', type=none_or_else, default=['0.01'], nargs='+', help='Significant level to control.')
    # data parameters
    parser.add_argument('--mu_outlier', type=float, default=2.5, help='')
    # model
    parser.add_argument('--model', type=none_or_else, default=None, choices=['OC-SVM', 'IF', 'LOF', None])
    parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples (same outliers '
                                                                  'proportion as in the calibration set unless otherwise specified).')
    parser.add_argument('--p_train', type=none_or_else, default=None, help='Proportion of outliers in the train-set.')
    parser.add_argument('--dataset', type=str, default='shuttle')
    parser.add_argument('--dataset_path', type=none_or_else, default=None, help='Path to dataset - only relevant for scores datasets.')
    parser.add_argument('--outlr_dataset', type=none_or_else, default=None)
    parser.add_argument('--dataset_version', default=1)
    parser.add_argument('--n_features', type=int, default=1, help='Number of features (only relevant for synthetic '
                                                                    'dataset)')
    parser.add_argument('--max_samples', default="auto", help='IF - The number of samples to draw from X to train each '
                                                              'base estimator.')
    parser.add_argument('--n_estimators', type=int, default=100, help='IF - The number of base estimators in the '
                                                                      'ensemble')
    parser.add_argument('--kernel_svm', type=str, default='rbf', help='OC-SVM - kernel type')
    args = parser.parse_args()
    if args.model is not None and args.n_train == 0:
        raise ValueError('When enabling predictive model, n_train parameter must be > 0.')
    if args.model is None and args.n_features != 1:
        raise ValueError('When predictive model is disabled, n_features must be 1.')
    if args.dataset.startswith('scores_') and (args.dataset_path is None or not os.path.exists(args.dataset_path)):
        raise ValueError('For scores dataset, dataset path must be specified and exists.')
    # add postprocess and bb to args
    if args.dataset.startswith('scores_'):
        args.postprocess = os.path.basename(os.path.normpath(args.dataset_path))
        args.bb_postprocess = os.path.basename(os.path.split(os.path.normpath(args.dataset_path))[0])
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
