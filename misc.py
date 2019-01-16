import numpy as np
import os

import pydensecrf.densecrf as dcrf


class AvgMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        """
        初始化相关参数
        
        :return:
        :rtype:
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        将每一个batch的损失进行积累, 最终可以得到训练集上完整的总的损失
        
        :param val: batch_mean_loss
        :type val:
        :param n: batch_size
        :type n:
        :return:
        :rtype:
        """
        self.val = val
        self.sum += val * n  # 累加总的损失
        self.count += n  # 累加样本数
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def cal_precision_recall_mae_fm(prediction, gt):
    """
    计算一张图MAE以及动态阈值下的P&R
    
    - mae 这里没有对真值进行二值化
    - precision, recall 这里使用动态阈值二值化预测值, 和使用0.5阈值来二值化真值, 进而计算P&R
    
    :param prediction:
    :type prediction:
    :param gt:
    :type gt:
    :return: precision, recall, mae, fm
    :rtype:
    """
    
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape
    
    eps = 1e-4
    
    # prediction = prediction / 255.
    # gt = gt / 255.
    
    # 按照@zengyu学姐的代码更改
    prediction = (prediction - prediction.min()) \
                 / (prediction.max() - prediction.min() + eps)
    gt[gt != 0] = 1
    
    # mae这里对真值进行二值化
    mae = np.mean(np.abs(prediction - gt))
    # threshold fm
    binary = np.zeros_like(prediction)
    threshold_fm = 2 * prediction.mean()
    if threshold_fm > 1:
        threshold_fm = 1
    binary[prediction >= threshold_fm] = 1
    tp = (binary * gt).sum()
    pre = tp / (binary.sum() + eps)
    rec = tp / (gt.sum() + eps)
    fm = 1.3 * pre * rec / (0.3 * pre + rec + eps)
    
    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)
    
    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.
        
        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1
        
        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)
        
        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))
    
    return precision, recall, mae, fm


def cal_fmeasure(precision, recall):
    """
    计算给定P&R时对应的Fm
    
    :param precision:
    :type precision:
    :param recall:
    :type recall:
    :return:
    :rtype:
    """
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    epsilon = 1e-5
    max_fmeasure = max(
        [(1 + beta_square) * p * r / (beta_square * p + r + epsilon)
         for p, r in zip(precision, recall)])
    
    return max_fmeasure


# codes of this function are borrowed from https://github.com/Andrew-Qibin/dss_crf
def crf_refine(img, annos):
    """
    使用条件随机场对结果调整
    
    :param img:
    :type img:
    :param annos:
    :type annos:
    :return:
    :rtype:
    """
    
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape
    
    # img and annos should be np array with data type uint8
    
    EPSILON = 1e-8
    
    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)
    
    anno_norm = annos / 255.
    
    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (
        tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))
    
    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()
    
    d.setUnaryEnergy(U)
    
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)
    
    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]
    
    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')
