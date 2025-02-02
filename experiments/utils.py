import subprocess
import yaml


def run_command(command, read_out=False):
    print('Running --->')
    print(command)
    print('------------')
    if read_out:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    else:
        process = subprocess.Popen(command, shell=True)
    # process.communicate()
    return process


def load_config(config_path, base_config_path='./experiments/base_parameters.yml'):
    base_config = load_yml_config(base_config_path)
    if config_path is None:
        config = {}
    else:
        config = load_yml_config(config_path)
    params_type = ['exp_params', 'data_params', 'model_params']
    flag_params = base_config['flag_params']
    if 'flag_params' in config.keys():
        flag_params.update(config['flag_params'])
    params = {}
    for params_t in params_type:
        curr_params = base_config[params_t]
        params.update(curr_params)
        if params_t in config.keys():
            params.update(config[params_t])
    return params, flag_params


def load_yml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_list_params(params):
    list_params = []
    for k, v in params.items():
        if isinstance(v, list):
            list_params.append(k)
    return list_params


def create_command(save_path, params, flag_params):
    additional_params = ''
    for k, v in flag_params.items():
        if v:
            additional_params += f' --{k}'

    base_command = (
        f"python ./main.py --exp_type {params['exp_type']} "
        f"--n_seeds {params['n_seeds']} --level {params['level']} "
        f"--model {params['model']} "
        f"--n_cal {params['n_cal']} --p_cal {params['p_cal']} --initial_labeled {params['initial_labeled']} "
        f"--n_train {params['n_train']} --p_train {params['p_train']} "
        f"--n_test {params['n_test']} --p_test {params['p_test']} "
        f"--dataset {params['dataset']} --dataset_version {params['dataset_version']} "
        f"--dataset_path {params['dataset_path']} --outlr_dataset {params['outlr_dataset']} "
        f"--mu_outlier {params['mu_outlier']} "
        f"--n_features {params['n_features']}  --save_path {save_path} "
        f" {additional_params}"
    )

    additional_args = ['seed', 'exp_params', 'max_samples', 'n_estimators', 'kernel_svm']
    for arg in additional_args:
        if arg in params.keys():
            base_command += f" --{arg} {params[arg]}"
    return base_command


def run_all_commands(save_path, params, flag_params, slurm=True):
    list_params = get_list_params(params)
    if len(list_params) == 0:
        command = create_command(save_path, params, flag_params)
        if slurm:
            command = f"bash ./create_tmp_empty.sh \"{command}\""
        run_command(command)
        return
    param = list_params[0]
    param_values = params[param]
    curr_params = params.copy()
    for v in param_values:
        curr_params[param] = v
        run_all_commands(save_path, curr_params, flag_params, slurm=slurm)
    return


def get_params_for_one_seed_command(seed, args_dict):
    flag_params = {}
    params = args_dict.copy()
    params['seed'] = seed
    params['n_seeds'] = 1
    # a single run does not run exp
    params['exp_type'] = None
    if 'exp_params' in params.keys():
        del params['exp_params']
    for arg in list(params.keys()):
        if isinstance(params[arg], bool):
            flag_params[arg] = params[arg]
            del params[arg]
    flag_params['no_distribute'] = True
    flag_params['full_save_path'] = True
    if isinstance(params['level'], list):
        if len(params['level']) == 1:
            params['level'] = params['level'][0]
        else:
            params['level'] = " ".join(params['level'])
    return params, flag_params


def check_for_job_state(job_id):
    process = run_command(f'squeue --nohead --format %F', read_out=True)
    output, err = process.communicate()
    running_jobs = output.split('\n')
    if err != '':
        raise ValueError(err)
    if str(job_id) in running_jobs:
        state = 'incomplete'
    else:
        state = 'complete'
    return state


def create_plot_command(params, flag_params):
    additional_params = ''
    for k, v in flag_params.items():
        if v:
            additional_params += f' --{k}'

    base_command = (
        f"python ./plot_main.py "
        f"--result_dir {params['result_dir']}  --plot_dir {params['plot_dir']} {additional_params}"
    )

    additional_args = ['x', 'y', 'filter_k', 'filter_v', 'file_desc', 'combine_to_type']
    for arg in additional_args:
        if arg in params.keys():
            base_command += f" --{arg} {params[arg]}"
    return base_command


def run_all_plot_commands(params_dict_list, slurm=True):
    for all_params in params_dict_list:
        command = create_plot_command(all_params['params'], all_params['flag_params'])
        if slurm:
            command = f"bash ./create_tmp_empty.sh \"{command}\""
        run_command(command)

