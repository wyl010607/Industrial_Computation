import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import iqr, rankdata

from .recorder import symbol, multi
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, roc_curve

def get_mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)
    Formula: MAE = mean(|y_true - y_pred|)
    """
    return np.mean(np.abs(y_true - y_pred))


def get_mse(y_true, y_pred):
    """
    Mean Squared Error (MSE)
    Formula: MSE = mean((y_true - y_pred)^2)
    """
    return ((y_true - y_pred) ** 2).mean()


def get_rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)
    Formula: RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def get_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE)
    Formula: MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    """
    non_zero_mask = y_true != 0
    y_true_masked = y_true[non_zero_mask]
    y_pred_masked = y_pred[non_zero_mask]
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    return mape


def get_rmspe(y_true, y_pred):
    """
    Root Mean Square Percentage Error (RMSPE)
    Formula: RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2)) * 100
    """
    non_zero_mask = y_true != 0
    rmspe = (
        np.sqrt(
            np.mean(
                np.square(
                    (y_true[non_zero_mask] - y_pred[non_zero_mask])
                    / y_true[non_zero_mask]
                )
            )
        )
        * 100
    )
    return rmspe


def get_bce(y_true, y_pred):
    """
    Binary Cross-Entropy (BCE)
    Formula: BCE = -y_true*log(y_pred) - (1 - y_true)*log(1 - y_pred)
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


def get_cce(y_true, y_pred):
    """
    Categorical Cross-Entropy (CCE)
    Formula: CCE = -sum(y_true*log(y_pred))
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred), axis=1).mean()


def get_hinge(y_true, y_pred):
    """
    Hinge Loss
    Formula: Hinge = mean(max(0, 1 - y_true*y_pred))
    """
    return np.mean(np.maximum(1 - y_true * y_pred, 0))

#===================USAD========================
def ROC(y_test, y_pred):
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr - (1 - fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot(fpr, 1 - fpr, 'r:')
    plt.plot(fpr[idx], tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]


def confusion_matrix(target, predicted, perc=False):
    data = {'y_Actual': target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted', 'y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=['Predicted'], colnames=['Actual'])

    if perc:
        sns.heatmap(confusion_matrix / np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()


# =============================GDN======================================
def get_err_median_and_iqr(predicted, groundtruth):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))
    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)
    return err_median, err_iqr


def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res
    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    epsilon = 1e-2
    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])
    return smoothed_err_scores


def get_full_err_scores(self, test_result, val_result):
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)
    all_scores = None
    all_normals = None
    feature_num = np_test_result.shape[-1]
    labels = np_test_result[2, :, 0].tolist()
    for i in range(feature_num):
        test_re_list = np_test_result[:2, :, i]
        val_re_list = np_val_result[:2, :, i]
        scores = self.get_err_scores(test_re_list, val_re_list)
        normal_dist = self.get_err_scores(val_re_list, val_re_list)
        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))
    return all_scores, all_normals


def eval_scores(scores, true_scores, th_steps, return_thresold=False):
    padding_list = [0]*(len(true_scores) - len(scores))
    if len(padding_list) > 0:
        scores = padding_list + scores
    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)
        fmeas[i] = f1_score(true_scores, cur_pred)
        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
        thresholds[i] = scores[score_index]
    if return_thresold:
        return fmeas, thresholds
    return fmeas


def get_best_performance_data(total_err_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)
    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])
    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)
    return max(final_topk_fmeas), pre, rec, auc_score

# ==================DCdetector====================
def get_f_score(prec, rec):
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec)
    return f_score

# DCdetector源代码中的模型性能评价部分过于繁琐，此处为自己实现
def get_performance(pred, gt_labels):
    prec = precision_score(gt_labels, pred)
    rec = recall_score(gt_labels, pred)
    auc_score = roc_auc_score(gt_labels, pred)
    f1 = get_f_score(prec, rec)
    return f1, prec, rec, auc_score


def smooth_score(x, smooth_base):
        a, b, c, d = x
        smooth_result_a = round(a, smooth_base)
        smooth_result_b = round(b, smooth_base)
        smooth_result_c = round(c, smooth_base)
        smooth_result_d = round(d, smooth_base)

        return smooth_result_a, smooth_result_b, smooth_result_c, smooth_result_d
