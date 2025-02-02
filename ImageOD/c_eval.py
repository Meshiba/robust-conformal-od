######################################################################################################################
# This file contains code adapted from https://github.com/Jingkang50/OpenOOD
#                   by "OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection" (Zhang et. al 2024)
#
# Additional implementations have been added to the file
######################################################################################################################
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Callable, List, Type
from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor
from openood.networks.react_net import ReactNet
from openood.evaluation_api.datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from openood.evaluation_api.postprocessor import get_postprocessor
from openood.evaluation_api.preprocessor import get_default_preprocessor

from openood.evaluation_api import Evaluator


class cEvaluator(Evaluator):
    def __init__(
        self,
        net: nn.Module,
        id_name: str,
        data_root: str = './data',
        config_root: str = './configs',
        preprocessor: Callable = None,
        postprocessor_name: str = None,
        postprocessor: Type[BasePostprocessor] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
        n_train: int = 2000,
        p_train: float = 0.03,
        curr_ood: str = 'texture',
    ) -> None:
        # check the arguments
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('Please pass postprocessor_name or postprocessor')
        if postprocessor_name is not None and postprocessor is not None:
            print(
                'Postprocessor_name is ignored because postprocessor is passed'
            )
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # load data
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor, **loader_kwargs)

        # get postprocessor
        if postprocessor_name == 'react':
            postprocessor = get_postprocessor(config_root, postprocessor_name,
                                              id_name)
            net = ReactNet(net)
        else:
            raise ValueError(f'{postprocessor_name} postprocessor is not supported.')

        # postprocessor setup
        n_outliers = int(n_train * p_train)
        n_inliers = n_train - n_outliers
        postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'], n_inliers=n_inliers,
                            n_outliers=n_outliers, curr_ood=curr_ood)

        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                {k: None
                 for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }
        # perform hyperparameter search if have not done so
        if (self.postprocessor.APS_mode
                and not self.postprocessor.hyperparam_search_done):
            self.postprocessor.set_hyperparam([90])

        self.net.eval()

    def eval_ood(self, fsood: bool = False, progress: bool = True, curr_ood: str = 'texture'):
        id_name = 'id' if not fsood else 'csid'
        task = 'ood' if not fsood else 'fsood'
        if self.metrics[task] is None:
            self.net.eval()

            # id score
            if self.scores['id']['test'] is None:
                print(f'Performing inference on {self.id_name} test set...',
                      flush=True)
                id_pred, id_conf, id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict['id']['test'], progress)
                self.scores['id']['test'] = [id_pred, id_conf, id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores['id']['test']

            if fsood:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores['csid'].keys()):
                    if self.scores['csid'][dataset_name] is None:
                        print(
                            f'Performing inference on {self.id_name} '
                            f'(cs) test set [{i+1}]: {dataset_name}...',
                            flush=True)
                        temp_pred, temp_conf, temp_gt = \
                            self.postprocessor.inference(
                                self.net,
                                self.dataloader_dict['csid'][dataset_name],
                                progress)
                        self.scores['csid'][dataset_name] = [
                            temp_pred, temp_conf, temp_gt
                        ]

                    csid_pred.append(self.scores['csid'][dataset_name][0])
                    csid_conf.append(self.scores['csid'][dataset_name][1])
                    csid_gt.append(self.scores['csid'][dataset_name][2])

                csid_pred = np.concatenate(csid_pred)
                csid_conf = np.concatenate(csid_conf)
                csid_gt = np.concatenate(csid_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            # load nearood data and compute ood metrics
            near_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                          ood_split='near',
                                          progress=progress, curr_ood=curr_ood)
            # load farood data and compute ood metrics
            far_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                         ood_split='far',
                                         progress=progress, curr_ood=curr_ood)

            if self.metrics[f'{id_name}_acc'] is None:
                self.eval_acc(id_name)
        else:
            print('Evaluation has already been done!')

        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.float_format',
                '{:,.2f}'.format):  # more options can be specified also
            print(self.metrics[task])

        return self.metrics[task]

    def _eval_ood(self,
                  id_list: List[np.ndarray],
                  ood_split: str = 'near',
                  progress: bool = True,
                  curr_ood: str = 'texture'):
        print(f'Processing {ood_split} ood...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict['ood'][
                ood_split].items():
            print(dataset_name)
            if dataset_name == curr_ood:
                if self.scores['ood'][ood_split][dataset_name] is None:
                    print(f'Performing inference on {dataset_name} dataset...',
                          flush=True)
                    ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                        self.net, ood_dl, progress)
                    self.scores['ood'][ood_split][dataset_name] = [
                        ood_pred, ood_conf, ood_gt
                    ]
                else:
                    print(
                        'Inference has been performed on '
                        f'{dataset_name} dataset...',
                        flush=True)
                    [ood_pred, ood_conf,
                     ood_gt] = self.scores['ood'][ood_split][dataset_name]

                ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
                pred = np.concatenate([id_pred, ood_pred])
                conf = np.concatenate([id_conf, ood_conf])
                label = np.concatenate([id_gt, ood_gt])

                print(f'Computing metrics on {dataset_name} dataset...')
                ood_metrics = compute_all_metrics(conf, label, pred)
                metrics_list.append(ood_metrics)
                self._print_metrics(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        return np.concatenate([metrics_list, metrics_mean], axis=0) * 100

