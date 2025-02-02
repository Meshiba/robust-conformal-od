import argparse
import os
from experiments.utils import run_all_commands, load_config


def main(config_path, save_path, dataset=None, dataset_ver=None, slurm=True):
    params, flag_params = load_config(config_path)
    if dataset is not None:
        params['dataset'] = dataset
    if dataset_ver is not None:
        params['dataset_version'] = dataset_ver
    run_all_commands(save_path, params, flag_params, slurm=slurm)


def get_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("-c", "--config_path", help="Path to configuration YAML file")
    parser.add_argument("-s", "--save_path", required=True, help="Path to save results")
    parser.add_argument("-d", "--dataset", help="Dataset name")
    parser.add_argument("-v", "--dataset_ver", help="Dataset version")
    parser.add_argument('--local', action='store_true', help='Run local experiments (not via SLURM).')

    args = parser.parse_args()
    if args.config_path is not None and not os.path.exists(args.config_path):
        raise ValueError('Config file does not exist.')
    return args


if __name__ == "__main__":
    args = get_args()
    main(args.config_path, args.save_path, args.dataset, args.dataset_ver, slurm=not args.local)
