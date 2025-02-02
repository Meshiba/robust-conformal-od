######################################################################################################################
#
#                          This file has been modified to support training with contaminated data.
#
######################################################################################################################
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class ReactPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ReactPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict, n_inliers, n_outliers, curr_ood):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                inliers, outliers = n_inliers, n_outliers
                if inliers > len(id_loader_dict['train'].dataset):
                    raise ValueError(f'Not enough inlier samples: {inliers} are required, but only '
                                     f'{len(id_loader_dict["train"])} is available.')
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    if inliers <= 0:
                        break
                    data = batch['data'].cuda()
                    data = data.float()
                    if len(data) > inliers:
                        data = data[:inliers]
                    inliers -= len(data)

                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

                ood_type = 'near' if curr_ood in ood_loader_dict['near'].keys() else 'far'
                if outliers > len(ood_loader_dict[ood_type][curr_ood].dataset):
                    raise ValueError(f'Not enough outlier samples: {outliers} are required, '
                                     f'but only {len(ood_loader_dict)} is available.')
                outlr_indices = []
                for batch in tqdm(ood_loader_dict[ood_type][curr_ood],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    if outliers <= 0:
                        break
                    data = batch['data'].cuda()
                    data = data.float()
                    indices = batch['index']
                    if len(data) > outliers:
                        data = data[:outliers]
                        indices = indices[:outliers]
                    outliers -= len(data)
                    outlr_indices.extend(indices)

                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.setup_flag = True
            # remove outlr_indices from dataloader
            outlr_imgs = [ood_loader_dict[ood_type][curr_ood].dataset.imglist[idx] for idx in outlr_indices]
            for img in outlr_imgs:
                ood_loader_dict[ood_type][curr_ood].dataset.imglist.remove(img)
        else:
            pass

        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net.forward_threshold(data, self.threshold)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))

    def get_hyperparam(self):
        return self.percentile
