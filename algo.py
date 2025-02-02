import numpy as np


def analyze_performance(rejections, test_y):
    n_outliers = np.sum(test_y == 1)
    n_false_discoveris = np.sum(test_y[rejections] == 0)
    n_true_discoveries = np.sum(test_y[rejections] == 1)
    assert n_true_discoveries + n_false_discoveris == len(rejections)
    power = n_true_discoveries / n_outliers if n_outliers else 1
    type1 = n_false_discoveris / (len(test_y) - n_outliers)
    return power, type1


def get_rejections_indices(calibration_scores, test_scores, level):
    calibration_scores = calibration_scores.reshape((-1,))
    test_scores = test_scores.reshape((-1,))
    # compute the (1-level) quantile of the calibration scores
    q_ind = int(np.ceil((1-level) * (calibration_scores.shape[0] + 1)))
    # sort calibration scores
    calibration_scores_sorted = np.sort(calibration_scores, axis=0)
    if q_ind > len(calibration_scores):
        threshold = np.inf
    else:
        threshold = calibration_scores_sorted[q_ind-1]
    rejections_indices = np.argwhere(test_scores > threshold)
    return rejections_indices, threshold


def get_naive_trimmed_calibration_set(calib_set, calib_y, trim=0.05):
    calib_set_sorted = np.sort(calib_set, axis=0)
    if int(len(calib_set) * trim) > 0:
        model_threshold = calib_set_sorted[-1 * int(len(calib_set) * trim)]
    else:
        model_threshold = np.inf
    our_calib_set = calib_set[calib_set < model_threshold]
    our_calib_y = calib_y[calib_set < model_threshold]
    trimmed_label_samples = calib_y[calib_set >= model_threshold]
    n_trimmed = len(calib_set) - len(our_calib_set)
    return our_calib_set, our_calib_y, trimmed_label_samples, n_trimmed, model_threshold


def get_calibration_set(method, initial_cal, initial_calib_set, calib_set, calib_y, p_trim):
    curr_calib_set, curr_calib_y, curr_trimmed_info = None, None, None
    if method == 'Clean':
        curr_calib_set = initial_calib_set
        curr_calib_y = np.zeros(len(initial_calib_set))
    elif method == 'Naive':
        curr_calib_set = calib_set
        curr_calib_y = calib_y
    elif method == 'NT':
        _calib_set = calib_set
        _calib_y = calib_y
        curr_calib_set, curr_calib_y, trimmed_label_samples, n_trimmed, naive_m_th = get_naive_trimmed_calibration_set(
                                                                                                           _calib_set,
                                                                                                           _calib_y,
                                                                                                           trim=p_trim)
        curr_trimmed_info = (n_trimmed, trimmed_label_samples, naive_m_th)
    elif method == 'Oracle':
        curr_calib_set = calib_set[calib_y == 0]
        curr_calib_y = np.zeros(len(curr_calib_set))
    elif method == 'LT':
        if initial_cal == 0:
            curr_calib_set = calib_set
            curr_calib_y = calib_y
            curr_trimmed_info = (0, [], np.inf)
        else:
            all_scores = calib_set.reshape((-1,1))
            all_labels = calib_y.reshape((-1,1))
            all_scores_labels = np.concatenate([all_scores, all_labels], axis=1)
            sorted_scores_labels = all_scores_labels[np.argsort(all_scores_labels[:,0])]
            sorted_scores = sorted_scores_labels[:,0]
            sorted_labels = sorted_scores_labels[:,1]
            # trim the outliers in the top #initial_cal scores
            cand_set = sorted_scores[-1 * initial_cal:]
            cand_y = sorted_labels[-1 * initial_cal:]
            cand_inlier = cand_set[cand_y == 0]
            if len(cand_inlier):
                curr_calib_set = np.concatenate([sorted_scores[:-1 * initial_cal], cand_inlier], axis=0)
                curr_calib_y = np.concatenate([sorted_labels[:-1 * initial_cal], np.zeros((len(cand_inlier),))], axis=0)
                trimmed_label_samples = np.ones((initial_cal - len(cand_inlier),))
                n_trimmed = initial_cal - len(cand_inlier)
            else:
                curr_calib_set = sorted_scores[:-1 * initial_cal]
                curr_calib_y = sorted_labels[:-1 * initial_cal]
                trimmed_label_samples = np.ones((initial_cal,))
                n_trimmed = initial_cal
            curr_trimmed_info = (n_trimmed, trimmed_label_samples, sorted_scores[-1 * initial_cal])
    return curr_calib_set, curr_calib_y, curr_trimmed_info

