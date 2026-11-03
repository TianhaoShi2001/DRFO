# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
workspace = '/data/shith/drfo-release/workspace'
import random
from typing import Dict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names,combined_dnn_input
from deepctr_torch.layers import DNN
from deepctr_torch.models.deepfm import *
import time
import argparse
import math
import itertools
from deepctr_torch.layers.utils import slice_arrays
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ray import tune, train
from mybasemodels import *
from torch.autograd import Variable
import json
import os
import logging
import copy

def smooth_abs_torch(x1, x2, beta = 1):
    y = 0
    if torch.abs(x1 - x2) < beta:
        y = 0.5 * (x1 - x2) ** 2
    else:
        y = torch.abs(x1 - x2) - 0.5
    return y
def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method = 1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def evaluation_methods(df, metrics):
    """
    Generate evaluation scores
    :param df:
    :param metrics:
    :return:
    """
    evaluations = []
    data_df = df.copy(deep=True)
    data_df["q*s"] = data_df['q'] * data_df['score']
    for metric in metrics:
        k = int(metric.split('@')[-1])
        tmp_df = data_df.sort_values(by='q*s', ascending=False, ignore_index=True)
        df_group = tmp_df.groupby('user_id')
        if metric.startswith('ndcg@'):
            ndcgs = []
            for uid, group in df_group:
                ndcgs.append(ndcg_at_k(group['label'].tolist()[:k], k=k, method=1))
            evaluations.append(np.average(ndcgs))
        elif metric.startswith('hit@'):
            hits = []
            for uid, group in df_group:
                hits.append(int(np.sum(group['label'][:k]) > 0))
            evaluations.append(np.average(hits))
        elif metric.startswith('precision@'):
            precisions = []
            for uid, group in df_group:
                if len(group['label'].tolist()) < k:
                    print(group)
                    print(uid)
                precisions.append(precision_at_k(group['label'].tolist()[:k], k=k))
            evaluations.append(np.average(precisions))
        elif metric.startswith('recall@'):
            recalls = []
            for uid, group in df_group:
                if np.sum(group['label']) == 0:
                    continue
                recalls.append(1.0 * np.sum(group['label'][:k]) / np.sum(group['label']))
            evaluations.append(np.average(recalls))
        elif metric.startswith('f1@'):
            f1 = []
            for uid, group in df_group:
                if np.sum(group['label']) == 0:
                    continue
                f1.append(2 * np.sum(group['label'][:k]) / (np.sum(group['label']) + k))
            evaluations.append(np.average(f1))
    return evaluations


def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.
    
      min ||x - u||_2 s.t. ||u||_1 <= eps
    
    Inspired by the corresponding numpy version by Adrien Gaidon.
    
    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU
      
    eps: float
      radius of l-1 ball to project onto
    
    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original
    
    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.
    
    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)

def project_simplex(v, z=1.0, axis=-1,):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
    :param v: A torch tensor, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = (torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1.).long()
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)


def project_onto_l1_ball_my(v, z = 1.0, axis = -1):
  if torch.norm(v, p = 1) <= z:
    w = v
  else:
    v_sign = torch.sign(v)
    v_abs = torch.abs(v)
    w_abs = project_simplex(v_abs, z, axis)
    w = torch.mul(w_abs, v_sign)
  return w

class MyFM(MybaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns,  use_fm=True,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                  task='binary', device='cpu', gpus=None,use_linear = False, lambda_ = 0, explicit = True, use_thre = False):

        super(MyFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        self.lambda_ = lambda_
        self.use_fm = use_fm
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.use_linear = use_linear
        self.sigmoid = torch.nn.Sigmoid
        self.explicit = explicit
        self.use_thre = use_thre
        if use_fm :
            self.fm = FM()
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        if(self.use_linear): 
            logit = self.linear_model(X)
        else:
            logit = 0
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)
        if self.explicit and not self.use_thre:
            y_pred = logit + self.bias# torch.sigmoid(logit+self.bias)
        else:
            y_pred = torch.sigmoid(logit+self.bias)
        return y_pred


    def train_one_batch_average_rating_reg(self, x, y, optim, model, loss_func, reg_kind = 0,):
        items = x[:, -2]
        batch_sensitive_attr = x[:, -1]
        y_pred = model(x).squeeze()
        optim.zero_grad()
        fair_reg_loss = 0
        if self.lambda_ != 0:
            fair_reg_loss = self.get_fair_reg_loss(y_pred, batch_sensitive_attr, reg_kind)
        loss = loss_func(y_pred, y.squeeze(),reduction = 'mean')
        reg_loss = self.get_regularization_loss()
        total_loss = loss + reg_loss + self.lambda_ * fair_reg_loss
        total_loss.backward()
        optim.step()
        batch_train_logs = {}
        batch_train_logs['loss'] = loss.item()
        if self.lambda_ != 0:
            batch_train_logs['fair_reg_loss'] = fair_reg_loss.item()
        batch_train_logs['batch_num'] = y.shape[0]
        return batch_train_logs
    
    def get_beyond_parity_reg_loss(self, items, y, y_pred, batch_senstive_attr, reg_kind = 0):
        # reg_kind = 0: calculate value unfairness of NIPS-2017-beyond-parity-fairness-objectives
        # reg_kind = 1: calculate absolute unfairness
        # reg_kind = 2: caculate value unfairness which is not conditioned on items
        if reg_kind == 0:
            uabs = 0
            n_items = list(set(items.clone().detach().cpu().numpy().squeeze().tolist()))
            sensitive_0 = (batch_senstive_attr == 0)
            sensitive_1 = (batch_senstive_attr == 1)
            i = 0
            for item in n_items:
                item_index = (items == item)
                s_0_index = sensitive_0 * item_index
                s_1_index = sensitive_1 * item_index
                if torch.any(s_0_index) and torch.any(s_1_index):
                    uabs += torch.abs(torch.abs(torch.mean(y[s_0_index]) - torch.mean(y_pred[s_0_index]))\
                        - torch.abs(torch.mean(y[s_1_index]) - torch.mean(y_pred[s_1_index] )))
                    i += 1
            uabs /= i
            return uabs

        if reg_kind == 1:
            # value unfairness in beyond parity
            uval = 0
            n_items = list(set(items.clone().detach().cpu().numpy().squeeze().tolist()))
            sensitive_0 = (batch_senstive_attr == 0)
            sensitive_1 = (batch_senstive_attr == 1)
            i = 0
            for item in n_items:
                item_index = (items == item)
                s_0_index = sensitive_0 * item_index
                s_1_index = sensitive_1 * item_index
                if torch.any(s_0_index) and torch.any(s_1_index):
                    uval += torch.abs((torch.mean(y[s_0_index]) - torch.mean(y_pred[s_0_index]))\
                        - (torch.mean(y[s_1_index]) - torch.mean(y_pred[s_1_index] )))
                    i += 1
            uval /= i
            return uval
        
        if reg_kind == 2:
            uabs = 0
            # reg for uabs_v1 
            s_0_index = batch_senstive_attr == 0
            s_1_index = batch_senstive_attr == 1
            uabs += torch.abs(torch.abs(torch.mean(y[s_0_index]) - torch.mean(y_pred[s_0_index]))\
                - torch.abs(torch.mean(y[s_1_index]) - torch.mean(y_pred[s_1_index] )))
            return uabs

        if reg_kind == 3:
            uabs = 0
            n_items = list(set(items.clone().detach().cpu().numpy().squeeze().tolist()))
            sensitive_0 = (batch_senstive_attr == 0)
            sensitive_1 = (batch_senstive_attr == 1)
            i = 0
            for item in n_items:
                item_index = (items == item)
                s_0_index = sensitive_0 * item_index
                s_1_index = sensitive_1 * item_index
                if torch.any(s_0_index) and torch.any(s_1_index):
                    uabs += smooth_abs_torch(torch.abs(torch.mean(y[s_0_index]) - torch.mean(y_pred[s_0_index])),
                        torch.abs(torch.mean(y[s_1_index]) - torch.mean(y_pred[s_1_index] )))
                    i += 1
            uabs /= i
            return uabs



    def get_fair_reg_loss(self, y_pred, batch_sensitive_attr, reg_kind = 0):
        # reg_kind 0: abs of discripancy of average ratings of advantaged groups and disadvantaged groups
        # reg_kind 1, 2, 3: independent reg in
        if reg_kind == 0:
            fair_reg_loss = (torch.abs(torch.mean(y_pred[batch_sensitive_attr == 0]) - torch.mean(y_pred[batch_sensitive_attr == 1]))).squeeze()
        if reg_kind == 1:
            fair_reg_loss = (torch.square(torch.mean(y_pred[batch_sensitive_attr == 0]) - torch.mean(y_pred[batch_sensitive_attr == 1]))).squeeze()
        if reg_kind == 2:
            b0 = 2 * 10e-24
            a0 = 10e-8
            q0 = torch.sum(torch.square(y_pred[batch_sensitive_attr == 0])).squeeze()
            s0 = torch.sum(y_pred[batch_sensitive_attr == 0]).squeeze()
            n0 = y_pred[batch_sensitive_attr == 0].shape[0]
            v0 = (b0 + q0 - s0 * s0 / n0) / (a0 + n0)
            q1 = torch.sum(torch.square(y_pred[batch_sensitive_attr == 1])).squeeze()
            s1 = torch.sum(y_pred[batch_sensitive_attr == 1]).squeeze()
            n1 = y_pred[batch_sensitive_attr == 1].shape[0]
            v1 = (b0 + q1 - s1 * s1 / n1) / (a0 + n1)
            fair_reg_loss = (- 1/2 * torch.log(2 * torch.sqrt(v0 * v1) / (v0 + v1)) + torch.square(s0 / n0 - s1 / n1) / (v0 + v1) / 4 ).squeeze()
        if reg_kind == 3:
            b0 = 2 * 10e-24
            a0 = 10e-8
            r0 = y_pred[batch_sensitive_attr == 0]
            q0 = torch.sum(torch.square(r0)).squeeze()
            s0 = torch.sum(r0).squeeze()
            n0 = r0.shape[0]
            v0 = (b0 + q0 - s0 * s0 / n0) / (a0 + n0)
            r1 = y_pred[batch_sensitive_attr == 1]
            q1 = torch.sum(torch.square(r1)).squeeze()
            s1 = torch.sum(r1).squeeze()
            n1 = r1.shape[0]
            v1 = (b0 + q1 - s1 * s1 / n1) / (a0 + n1)
            v = (2 * b0 + q0 + q1 - torch.square(s0 + s1) / y_pred.shape[0]) / (2 * a0 + y_pred.shape[0])
            fair_reg_loss = 1 / 2 * torch.log(2 * 3.1415926 * 2.71828 * v) - 1 / 2 * torch.log(2 * 3.1415926 * 2.71828 * v1) * (n1 / y_pred.shape[0]) \
                - 1 / 2 * torch.log(2 * 3.1415926 * 2.71828 * v0) * (n0 / y_pred.shape[0])
        
        if reg_kind == 100:
            fair_reg_loss = (torch.mean(y_pred[batch_sensitive_attr == 0]) + torch.mean(y_pred[batch_sensitive_attr == 1]) - torch.mean(y_pred) * 2).squeeze()
        return fair_reg_loss

    def fit_average_rating_reg(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, 
            save_models = True, ref_metric = None, reg_kind = 1, tune_mode = 'min',
              performance_mode = 'min', performance_scale = 0.9,
              watch_performance_metrics = None, test_data = None):
        train_tensor_data, val_x, val_y, do_validation, test_x, test_y, do_test = self.process_train_data(x, y, validation_data,
         validation_split,  test_data = test_data)
        if batch_size is None:
            batch_size = 256
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle = shuffle, batch_size = batch_size, num_workers = 4)  
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        if ref_metric is None:
            ref_metric = "val_auc"
        stopers = myearlystopping(refer_metric = ref_metric, mode = tune_mode, performance_satisfied_stop_condition=130)
        report_dict = {}
        if do_validation:
            eval_result = self.evaluate(val_x, val_y, batch_size)
            for name, result in eval_result.items():
                if name == self.performance_metric:
                    self.best_performance_value = result
        del_metrics = []
        # # ['average_rating' + '_unknow' + '_sensitive_0', 'average_rating' + '_unknow' + '_sensitive_1',
        save_metrics = self.metrics.copy()
        run_metrics = {}
        for name, metric in save_metrics.items():
            if name not in del_metrics:
                run_metrics[name] = metric
        self.metrics = run_metrics.copy()
        num_batches = int(sample_num / batch_size)
        train_iter = iter(train_loader)
        train_iter_cycle = itertools.cycle(train_iter)
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}
            loss_epoch = 0
            reg_loss_epoch = 0
            train_logs = {} # my own api
            begin_time = time.time()
            for batch_idx in range(num_batches):
                x_train, y_train = next(itertools.islice(train_iter_cycle, batch_idx, batch_idx + 1))
                print('load batch data, cost', time.time() - begin_time)
                begin_time = time.time()
                x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                y = y_train.to(self.device).float()
                batch_train_logs = self.train_one_batch_average_rating_reg(x = x, y = y, optim = optim, model = model,
                        loss_func = loss_func, reg_kind = reg_kind)
                if train_logs == {}:
                    train_logs = batch_train_logs
                else:
                    for name, result in batch_train_logs.items():
                        if name != 'batch_num':
                            train_logs[name] += result * train_logs['batch_num']
                        else:
                            train_logs[name] += result
                
                print('train finished, cost', time.time() - begin_time)
                begin_time = time.time()
            # except KeyboardInterrupt:
            #     t.close()
            #     raise
            # t.close()
            for name, result in train_logs.items():
                if name != 'batch_num':
                    train_logs[name] = result / train_logs['batch_num']
            epoch_logs["loss"] = loss_epoch / steps_per_epoch
            
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            
            
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            
            
            if do_test:
                test_eval_result = self.evaluate(test_x, test_y, batch_size)
                for name, result in test_eval_result.items():
                    epoch_logs["test_" + name] = result
            print('evaluate valid and test, cost', time.time() - begin_time)
            begin_time = time.time()
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                if do_validation:
                    for name in epoch_logs:
                        if('val' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])

                if do_test:
                    for name in epoch_logs:
                        if('test' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])  
                print(eval_str)
            
            if self.performance_metric is not None:
                performance_metric = 'val_' + self.performance_metric
                performance_satisfied = ((epoch_logs[performance_metric] > performance_scale * self.best_performance_value) and (performance_mode == 'max')) \
                    or ((epoch_logs[performance_metric]*performance_scale < self.best_performance_value) and (performance_mode == 'min'))
                print(epoch_logs[performance_metric] ,self.best_performance_value ,performance_satisfied )
            else:
                performance_satisfied = True

                
            need_saving = stopers.update_and_isbest(epoch_logs, epoch, performance_satisfied = performance_satisfied)
            if need_saving:
                epoch_logs['val_dp_search'] = epoch_logs['val_dp']
                epoch_logs['val_dp_true_search'] = epoch_logs['val_dp_true']
            for metric in watch_performance_metrics:
                if performance_satisfied:
                    epoch_logs[metric + '_performance_satisfied'] = epoch_logs[metric]
                    try:
                        epoch_logs[metric + '_performance_satisfied_top'] = stopers.best_eval_result[metric]
                    except:
                        epoch_logs[metric + '_performance_satisfied_top'] = 2.5
                else:
                    epoch_logs[metric + '_performance_satisfied'] = 2.5
                    try:
                        epoch_logs[metric + '_performance_satisfied_top'] = stopers.best_eval_result[metric]
                    except:
                        epoch_logs[metric + '_performance_satisfied_top'] = 2.5
                
            self.my_tune_report(epoch_logs, report_dict, epoch, train_logs = train_logs)
            need_stopping = stopers.is_stop()
            self.metrics = save_metrics.copy()
            self.my_save_model(epoch_logs = epoch_logs, need_saving = need_saving, save_models = save_models, ref_metric = ref_metric, epoch = epoch, mode = tune_mode)
            self.metrics = run_metrics.copy()
            print('save and report, cost', time.time() - begin_time)
            begin_time = time.time()
            if need_stopping:
                print("early stop.......")
                self.metrics = save_metrics.copy()

                break
        self.metrics = save_metrics.copy()

        return self.history

    def fit(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, 
            save_models = True, ref_metric = None, mode = 'min', test_data = None):

        train_tensor_data, val_x, val_y, do_validation, test_x, test_y, do_test = self.process_train_data(x, y
            , validation_data, validation_split, test_data = test_data)
        if batch_size is None:
            batch_size = 256
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle = shuffle, batch_size = batch_size, num_workers = 4)  
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        if ref_metric is None:
            ref_metric = "val_auc"
        stopers = myearlystopping(refer_metric = ref_metric, mode = tune_mode, performance_satisfied_stop_condition=130)
        report_dict = {}
        train_iter = iter(train_loader)
        train_iter_cycle = itertools.cycle(train_iter)
        num_batches = int(sample_num / batch_size)
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}
            for batch_idx in range(num_batches):
                x_train, y_train = next(itertools.islice(train_iter_cycle, batch_idx, batch_idx + 1))
                x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                y = y_train.to(self.device).float()
                self.train_one_batch(x = x, y = y, optim = optim, model = model, loss_func = loss_func)
                       
                print('train finished')

            # except KeyboardInterrupt:
            #     t.close()
            #     raise
            # t.close()

            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                if do_validation:
                    for name in epoch_logs:
                        if('val' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])
                print(eval_str)
            ######tune.report
            self.my_tune_report(epoch_logs, report_dict, epoch)

            if self.performance_metric is not None:
                performance_metric = 'val_' + self.performance_metric
                performance_satisfied = epoch_logs[performance_metric] > 0.9 * self.best_performance_value
            else:
                performance_satisfied = True
            need_saving = stopers.update_and_isbest(epoch_logs, epoch, performance_satisfied)
            need_stopping = stopers.is_stop()
            self.my_save_model(epoch_logs = epoch_logs, need_saving = need_saving, save_models = save_models, ref_metric = ref_metric, epoch = epoch, mode = mode)

            if need_stopping:
                print("early stop.......")
                # not stop here 
                break
        return self.history

    def process_train_data(self, x, y, validation_data, validation_split, test_data = None):
        if isinstance(x, dict):
            x_temp = [x[feature] for feature in self.feature_index]
            x_temp.append(x['item_id'])
            x_temp.append(x['sensitive_attribute'])
            x = x_temp
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x_temp = [val_x[feature] for feature in self.feature_index]
                val_x = val_x_temp

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        if test_data is not None:
            do_test = True
            if len(test_data) == 2:
                test_x, test_y = test_data
                val_sample_weight = None
            if isinstance(test_x, dict):
                test_x_temp = [test_x[feature] for feature in self.feature_index]
                test_x = test_x_temp
            
        else:
            test_x = None
            test_y = None
            do_test = False

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
            if x[i].dtype == 'object':
                x[i] = x[i].astype(int)
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))

        return train_tensor_data, val_x, val_y, do_validation, test_x, test_y, do_test


class DRFOBase(MybaseModel):
    # dro under equal rating constraint
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns,user_feat_column,obeserved_group_distributions, sensitive_num = 2, gammas = [0.1,0.1],
                  init_lambdas = 0.1, lambda_lr = 0.1, 
                  use_fm=True, l2_reg_linear=0.00001, l2_reg_embedding = 0.00001,  init_std = 0.0001, seed = 1024,
                  task='binary', device='cpu', gpus=None, use_linear = True, explicit = True, use_thre = False):

        
        super().__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        self.explicit = explicit
        self.use_thre = use_thre
        self.sensitive_num = sensitive_num
        sample_num = obeserved_group_distributions['0'].shape[0]
        self.gammas = gammas
        self.user_feat_column = user_feat_column
        self.user_feature_index = build_input_features(user_feat_column)
        self.lambda_lr = lambda_lr
        self.user_embedding_dict_dro = nn.ModuleDict({})
        self.user_embedding_dict_observed_noisy = nn.ModuleDict({})
        self.lambda_dict = nn.ParameterDict({str(i):nn.Parameter(torch.ones(1) * init_lambdas) for i in range(sensitive_num)})
        # observed_group_distribution should be sorted in order for user 0, 1, 2, 3, 4 ...
        for i in range(sensitive_num):
            # get dro_probability and save as user_embedding
            embedding_dict = nn.ModuleDict({})
            user_embedding = torch.from_numpy(obeserved_group_distributions[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict['user_id'] = nn.Embedding.from_pretrained(user_embedding, freeze = False)
            self.user_embedding_dict_dro[str(i)] = embedding_dict
            # get obser_probability
            embedding_dict2 = nn.ModuleDict({})
            user_embedding2 = torch.from_numpy(obeserved_group_distributions[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict2['user_id'] = nn.Embedding.from_pretrained(user_embedding2, freeze = True)
            self.user_embedding_dict_observed_noisy[str(i)] = embedding_dict2 # user_embedding.to(device)
            
        self.use_fm = use_fm
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.use_linear = use_linear
        self.sigmoid = torch.nn.Sigmoid
        if use_fm :
            self.fm = FM()
        self.to(device)

    def my_input_from_feature_columns(self, X, feature_columns, embedding_dict, feature_index,  support_dense=True, ):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")
        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns] #error

        sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
                                                      varlen_sparse_feature_columns)

        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def get_dro_distribution_probability(self, U, i):
        sparse_embedding_list, dense_value_list = self.my_input_from_feature_columns(U, self.user_feat_column,
                                                                                  self.user_embedding_dict_dro[str(i)], self.user_feature_index)                                                            
        return torch.cat(sparse_embedding_list,dim = 1).squeeze()

    def get_dro_distribution_probability_noisy(self, U, i):
        sparse_embedding_list, dense_value_list = self.my_input_from_feature_columns(U, self.user_feat_column,
                                                                                  self.user_embedding_dict_observed_noisy[str(i)], self.user_feature_index)                                                            
        return torch.cat(sparse_embedding_list,dim = 1).squeeze()

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        
        if(self.use_linear): 
            logit = self.linear_model(X)
        else:
            logit = 0
        
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)
        y_pred = torch.sigmoid(logit + self.bias)

        return y_pred
    
    def process_train_data(self, x, y, validation_data, validation_split, test_data = None):
        if isinstance(x, dict):
            x_temp = [x[feature] for feature in self.feature_index]
            x_temp.append(x['item_id'])
            x_temp.append(x['sensitive_attribute'])
            u = [x['user_id']]
            x = x_temp
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x_temp = [val_x[feature] for feature in self.feature_index]
                val_x = val_x_temp

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        if test_data is not None:
            do_test = True
            if len(test_data) == 2:
                test_x, test_y = test_data
                val_sample_weight = None
            if isinstance(test_x, dict):
                test_x_temp = [test_x[feature] for feature in self.feature_index]
                test_x = test_x_temp
            
        else:
            test_x = None
            test_y = None
            do_test = False

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
            if x[i].dtype == 'object':
                x[i] = x[i].astype(int)
        for i in range(len(u)):
            if len(u[i].shape) == 1:
                u[i] = np.expand_dims(u[i], axis=1)
            if u[i].dtype == 'object':
                u[i] = u[i].astype(int)
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(u, axis=-1)),
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
            
        return train_tensor_data, val_x, val_y, do_validation, test_x, test_y, do_test

    def get_beyond_parity_constraint(self, items, y, y_pred, group_i_prob, reg_kind = 0):
        # reg_kind = 0: calculate value unfairness of NIPS-2017-beyond-parity-fairness-objectives
        # reg_kind = 1: calculate absolute unfairness
        if reg_kind == 0:
            uabs = 0
            n_items = list(set(items.clone().detach().cpu().numpy().squeeze().tolist()))
            for item in n_items:
                with torch.no_grad():
                    item_index = (items == item)
                uabs += torch.abs(torch.abs(torch.mean(y[item_index]) - torch.mean(y_pred[item_index]))\
                     - torch.abs(torch.mean(torch.mul(y, group_i_prob)[item_index]) - torch.mean(torch.mul(y_pred, group_i_prob)[item_index] )))
            uabs /= items.shape[0]
            return uabs
        
    def get_constraint(self, y_pred, group_i_prob, reg_kind, epsilon = 0, label_weight = 1, indices = None):
        if indices is None:
            if reg_kind == 0:
                constraint = torch.mean(y_pred).squeeze() - torch.sum(torch.mul(y_pred, group_i_prob)).squeeze()
            if reg_kind == 1:
                constraint = torch.abs(torch.mean(y_pred).squeeze() - torch.sum(torch.mul(y_pred, group_i_prob)).squeeze()) - epsilon
        else:
            def scale_vector(x, indices, scale_factor):
                # scale_weight = scale_factor * torch.sum(x[indices])
                # x[indices] *= scale_factor
                # zero_mask = (x == 0)
                # zero_count = torch.sum(zero_mask)
                # x[zero_mask] = scale_weight / zero_count if zero_count > 0 else 0
                # x /= torch.sum(x)
                nonzero_mask = (x != 0)
                # Scale the elements at the specified indices by the scaling factor
                x[indices] *= scale_factor
                # Compute the sum of the elements that were scaled
                scaled_sum = torch.sum(x[indices])
                # Replace non-zero elements with scaled values
                x[nonzero_mask] *= torch.sum(x[nonzero_mask]) / (torch.sum(x[nonzero_mask]) + scaled_sum)
                # Normalize the entire vector so that its elements sum to 1
                x /= torch.sum(x)
                return x
            weights = torch.ones_like(y_pred, dtype=torch.float64, device = self.device)  # set all values to 1 by default
            weights[indices] = label_weight
            
            # group_i_prob_scale = scale_vector(group_i_prob, indices, label_weight)
            if reg_kind == 0:
                constraint = torch.mean(y_pred).squeeze() - torch.sum(torch.mul(y_pred, group_i_prob)).squeeze()
            if reg_kind == 1:
                constraint = torch.abs(torch.mean(torch.mul(weights, y_pred)).squeeze() \
                    - torch.sum(torch.mul(y_pred, torch.mul(weights, group_i_prob))).squeeze()) - epsilon
        return constraint


    def train_one_batch(self, u, x, y, optim1, optim2, model,
     loss_func, thetas, reg_kind = 0, lambda_max = 1, lambda_min = 0, epsilon = 0, update_k_times = 1,
     train_label_users = None, label_weight = 1, train_nolabel_users = None, forbid_users = None):
        # optim1 use for update dro
        # optim2 use for update theta
        # sensitive_attribute = x[:,-1]
        # _:batch_num
        # items = x[:, -1]
        
        if train_label_users is not None:
            is_label = torch.isin(u.squeeze(), torch.tensor(train_label_users, device = self.device))
            label_indices = torch.where(is_label)
            # weights[is_label] = label_weight
        else:
            label_indices = None
        begin_time = time.time()
        batch_train_logs = {}
        batch_sensitive_attr = x[:, -1]
        y_pred = model(x).squeeze()
        optim1.zero_grad()
        optim2.zero_grad()
        loss = loss_func(y_pred, y.squeeze(),reduction='mean')
        reg_loss = self.get_regularization_loss()
        fair_constraints = {}
        sum_fair_constraint = 0
        iter_0 = 0
        iter_1 = 0
        eps = 1e-5
        sensitive_num = self.sensitive_num
        old_embedding = self.user_embedding_dict_dro[str(0)]['user_id'].weight.data.clone().detach()


        for i in range(sensitive_num):
            group_i_prob = self.get_dro_distribution_probability(u, i).squeeze() # * (1 + self.get_dro_distribution_probability(u, i).squeeze())
            print('sum_prob', torch.sum(group_i_prob))
            fair_constraints[str(i)] = torch.abs(torch.mean(y_pred).squeeze() - torch.sum(torch.mul(y_pred, group_i_prob)).squeeze())
            sum_fair_constraint += self.lambda_dict[str(i)] * fair_constraints[str(i)]

        (-sum_fair_constraint).backward(inputs = list(self.user_embedding_dict_dro.parameters()), retain_graph = True)
        (loss + reg_loss + sum_fair_constraint).backward(inputs = thetas)

        optim1.step()
        optim2.step()

        with torch.no_grad():
            for i in range(sensitive_num):
                u_sort,_ = torch.sort(u.squeeze())
                u_sort = u_sort.reshape(-1,1)
                p_dro = self.get_dro_distribution_probability(u_sort, i).squeeze()
                p_noisy = self.get_dro_distribution_probability_noisy(u_sort, i).squeeze()
                u_sort_df = pd.Series(u_sort.squeeze().clone().detach().cpu().numpy())
                cnt = pd.value_counts(u_sort_df)
                indices = []
                index = 0
                for j in cnt.index.unique().sort_values().tolist():
                    indices.append(index)
                    index += cnt[j]
                unique_index = torch.unique(u_sort.squeeze()).long()
 
                p_proj_l1 = (project_onto_l1_ball_my(p_dro - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()

                while True:
                    p_proj_simplex = project_simplex(p_proj_l1).squeeze()
                    p_proj_l1 = (project_onto_l1_ball_my(p_proj_simplex - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()
                    if i == 0:
                        iter_0 += 1
                    if i == 1:
                        iter_1 += 1
                    if torch.norm(p_proj_l1 - p_proj_simplex,p = 1) < eps:
                        p_proj = p_proj_l1.reshape(-1, 1)
                        break
                    if (i == 0 and iter_0 >= 50) or (i == 1 and iter_1 >= 50):
                        p_proj = p_proj_l1.reshape(-1, 1)
                        eps /= 0.9
                        break
                self.user_embedding_dict_dro[str(i)]['user_id'].weight.data[unique_index] = p_proj[indices]
            
        with torch.no_grad():
            for i in range(sensitive_num):
                self.lambda_dict[str(i)].data += fair_constraints[str(i)] * self.lambda_lr
                # clamp lambda
                self.lambda_dict[str(i)] = torch.clamp(self.lambda_dict[str(i)], min = lambda_min, max = lambda_max)
            fair_reg_loss_not_abs = (torch.mean(y_pred[batch_sensitive_attr == 0]) - torch.mean(y_pred[batch_sensitive_attr == 1])).squeeze()
            fair_reg_loss = torch.abs(torch.mean(y_pred[batch_sensitive_attr == 0]) - torch.mean(y_pred[batch_sensitive_attr == 1])).squeeze()
            if reg_kind == 0:
                fair_constraints_0_true = (-torch.mean(y_pred[batch_sensitive_attr == 0]) + torch.mean(y_pred))
                fair_constraints_1_true = (-torch.mean(y_pred[batch_sensitive_attr == 1]) + torch.mean(y_pred))
            if reg_kind == 1:
                fair_constraints_0_true = torch.abs((-torch.mean(y_pred[batch_sensitive_attr == 0]) + torch.mean(y_pred)))
                fair_constraints_1_true = torch.abs((-torch.mean(y_pred[batch_sensitive_attr == 1]) + torch.mean(y_pred)))
        print('update dro prob cost', time.time() - begin_time)
        begin_time = time.time()
        new_embedding = self.user_embedding_dict_dro[str(0)]['user_id'].weight.data.clone().detach()
        batch_train_logs['embedding_distance'] = torch.norm(old_embedding - new_embedding, p = 1).squeeze().item()
        p_dro = self.get_dro_distribution_probability(u_sort, 0).squeeze()
        p_noisy = self.get_dro_distribution_probability_noisy(u_sort, 0).squeeze()
        batch_train_logs['prob_distance_0'] = torch.norm(p_dro - p_noisy, p = 1).squeeze().item() 
        p_dro = self.get_dro_distribution_probability(u_sort, 1).squeeze()
        p_noisy = self.get_dro_distribution_probability_noisy(u_sort, 1).squeeze()
        batch_train_logs['pro_distance_1'] = torch.norm(p_dro - p_noisy, p = 1).squeeze().item() 
        batch_train_logs['loss'] = loss.item()
        batch_train_logs['fair_reg_loss_not_abs'] = fair_reg_loss_not_abs.item()
        batch_train_logs['fair_reg_loss'] = fair_reg_loss.item()
        batch_train_logs['fair_constraint_0_dro'] = fair_constraints[str(0)].item()
        batch_train_logs['fair_constraint_1_dro'] = fair_constraints[str(1)].item()
        batch_train_logs['fair_constraint_0_true'] = fair_constraints_0_true.item()
        batch_train_logs['fair_constraint_1_true'] = fair_constraints_1_true.item()
        batch_train_logs['fair_constraint_0_dro-true'] =  fair_constraints[str(0)].item() - fair_constraints_0_true.item()
        batch_train_logs['fair_constraint_1_dro-true'] =  fair_constraints[str(1)].item() - fair_constraints_1_true.item()
        batch_train_logs['batch_num'] = y.shape[0]
        batch_train_logs['iter_0'] = iter_0
        batch_train_logs['iter_1'] = iter_1
        batch_train_logs['lambda_0'] = self.lambda_dict[str(0)].data.squeeze().item()
        batch_train_logs['lambda_1'] = self.lambda_dict[str(1)].data.squeeze().item()
        return batch_train_logs


    def fit(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, 
            save_models = True, ref_metric = None, tune_mode = 'min', 
            performance_mode = 'min', reg_kind = 0, performance_scale = 0.9,
            watch_performance_metrics = None, test_data = None, lambda_max = 1, lambda_min = 0, epsilon = 0, update_k_times = 1,
            train_label_users = None, label_weight = 1
            ):
        train_tensor_data, val_x, val_y, do_validation, test_x, test_y, do_test = self.process_train_data(x, y, validation_data,
         validation_split,  test_data = test_data)
        if batch_size is None:
            batch_size = 256
        model = self.train()
        loss_func = self.loss_func
        # optim = self.optim
        optim1 = self.optim1
        optim2 = self.optim2
        # if
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle = shuffle, batch_size = batch_size, num_workers = 4)  
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        if ref_metric is None:
            ref_metric = "val_auc"
        stopers = myearlystopping(refer_metric = ref_metric, mode = tune_mode, performance_satisfied_stop_condition=130)
        report_dict = {}

        # get model parameters
        thetas = []
        for name, param in self.named_parameters():
            if 'user_embedding' not in name and 'lambda' not in name:
                thetas.append(param)

        if do_validation:
            eval_result = self.evaluate(val_x, val_y, batch_size)
            for name, result in eval_result.items():
                if name == self.performance_metric:
                    self.best_performance_value = result
        del_metrics = []
        # del_metrics = ['average_rating' + '_unknow' + '_sensitive_0', 'average_rating' + '_unknow' + '_sensitive_1',
        save_metrics = self.metrics.copy()
        run_metrics = {}
        for name, metric in save_metrics.items():
            if name not in del_metrics:
                run_metrics[name] = metric
        self.metrics = run_metrics.copy()
        num_batches = int(sample_num / batch_size)
        train_iter = iter(train_loader)
        train_iter_cycle = itertools.cycle(train_iter)
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}
            loss_epoch = 0
            reg_loss_epoch = 0
            train_logs = {}
            begin_time = time.time()
            for batch_idx in range(num_batches):
                u_train, x_train, y_train = next(itertools.islice(train_iter_cycle, batch_idx, batch_idx + 1))
                    # for _, (u_train, x_train, y_train) in t:
                print('load_data, cost', time.time()-begin_time)
                begin_time = time.time()
                # begin_time = time.time()
                
                u = u_train.to(self.device).float()
                x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                y = y_train.to(self.device).float()
                print('trans to gpu, cost', time.time()-begin_time)
                begin_time = time.time()
                batch_train_logs = self.train_one_batch(u = u, x = x, y = y, optim1 = optim1, optim2 = optim2, model = model,
                    loss_func = loss_func, thetas = thetas, reg_kind = reg_kind, lambda_max = lambda_max, lambda_min = lambda_min,
                    epsilon = epsilon, update_k_times = update_k_times, train_label_users = train_label_users, label_weight = label_weight)
                print('train one batch, cost', time.time()-begin_time)
                # params = self.user_embedding_dict_dro['0']['user_id'].state_dict()

                begin_time = time.time()
                if train_logs == {}:
                    # train_logs[name] = batch_train_logs['batch_num']
                    for name, result in batch_train_logs.items():
                        if name != 'batch_num':
                            train_logs[name] = result * batch_train_logs['batch_num']
                            
                        else:
                            train_logs[name] = result
                else:
                    for name, result in batch_train_logs.items():
                        if name != 'batch_num':
                            train_logs[name] += result * batch_train_logs['batch_num']
                            
                        else:
                            train_logs[name] += result
                print('train finished, cost', time.time()-begin_time)
                begin_time = time.time()
            # except KeyboardInterrupt:
            #     t.close()
            #     raise
            # t.close()
            for name, result in train_logs.items():
                if name != 'batch_num':
                    train_logs[name] = result / train_logs['batch_num']
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose

            if do_test:
                test_eval_result = self.evaluate(test_x, test_y, batch_size)
                for name, result in test_eval_result.items():
                    epoch_logs["test_" + name] = result
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                if do_validation:
                    for name in epoch_logs:
                        if('val' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])
                                    
                if do_test:
                    for name in epoch_logs:
                        if('test' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])  
                print(eval_str)
            print('evlaute valid and test, cost', time.time()-begin_time)
            begin_time = time.time()
            if self.performance_metric is not None:
                performance_metric = 'val_' + self.performance_metric
                performance_satisfied = ((epoch_logs[performance_metric] > performance_scale * self.best_performance_value) and (performance_mode == 'max')) \
                    or ((epoch_logs[performance_metric]*performance_scale < self.best_performance_value) and (performance_mode == 'min'))
                print(epoch_logs[performance_metric] ,self.best_performance_value ,performance_satisfied )
            else:
                performance_satisfied = True
            self.metrics = save_metrics
            need_saving = stopers.update_and_isbest(epoch_logs, epoch, performance_satisfied = performance_satisfied)
            if need_saving and performance_satisfied:
                epoch_logs['val_dp_search'] = epoch_logs['val_dp']
                epoch_logs['val_dp_true_search'] = epoch_logs['val_dp_true']
            self.metrics = run_metrics
            for metric in watch_performance_metrics:
                if performance_satisfied:
                    epoch_logs[metric + '_performance_satisfied'] = epoch_logs[metric]
                    try:
                        epoch_logs[metric + '_performance_satisfied_top'] = stopers.best_eval_result[metric]
                    except:
                        epoch_logs[metric + '_performance_satisfied_top'] = 2.5
                else:
                    epoch_logs[metric + '_performance_satisfied'] = 2.5
                    try:
                        epoch_logs[metric + '_performance_satisfied_top'] = stopers.best_eval_result[metric]
                    except:
                        epoch_logs[metric + '_performance_satisfied_top'] = 2.5
                
            self.my_tune_report(epoch_logs, report_dict, epoch, train_logs = train_logs)
            need_stopping = stopers.is_stop()
            self.metrics = save_metrics.copy()
            self.my_save_model(epoch_logs = epoch_logs, need_saving = need_saving, save_models = save_models, ref_metric = ref_metric, epoch = epoch, mode = tune_mode)
            self.metrics = run_metrics.copy()
            if need_stopping:
                print("early stop.......")
                self.metrics = save_metrics.copy()
                break
        self.metrics = save_metrics.copy()
        return self.history            

    



class DRFO(DRFOBase):
    # dro under equal rating constraint
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns,user_feat_column,obeserved_group_distributions, sensitive_num = 2, gammas = [0.1,0.1],
                  init_lambdas = 0.1, lambda_lr = 0.1, 
                  use_fm=True, l2_reg_linear=0.00001, l2_reg_embedding = 0.00001,  init_std = 0.0001, seed = 1024,
                  task='binary', device='cpu', gpus=None, use_linear = True,
                   explicit = True, use_thre = False ):

        
        MybaseModel.__init__(self,linear_feature_columns = linear_feature_columns, dnn_feature_columns = dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        self.explicit = explicit
        self.use_thre = use_thre
        self.sensitive_num = sensitive_num
        sample_num = obeserved_group_distributions['0'].shape[0]
        self.gammas = gammas
        self.user_feat_column = user_feat_column
        self.user_feature_index = build_input_features(user_feat_column)
        self.lambda_lr = lambda_lr
        self.user_embedding_dict_dro = nn.ModuleDict({})
        self.user_embedding_dict_observed_noisy = nn.ModuleDict({})
        self.lambda_dict = nn.ParameterDict({str(i):nn.Parameter(torch.ones(1) * init_lambdas) for i in range(sensitive_num)})
        # observed_group_distribution should be sorted in order for user 0, 1, 2, 3, 4 ...
        for i in range(sensitive_num):
            # get dro_probability and save as user_embedding
            embedding_dict = nn.ModuleDict({})
            user_embedding = torch.from_numpy(obeserved_group_distributions[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict['user_id'] = nn.Embedding.from_pretrained(user_embedding, freeze = False)
            self.user_embedding_dict_dro[str(i)] = embedding_dict
            # get obser_probability
            embedding_dict2 = nn.ModuleDict({})
            user_embedding2 = torch.from_numpy(obeserved_group_distributions[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict2['user_id'] = nn.Embedding.from_pretrained(user_embedding2, freeze = True)
            self.user_embedding_dict_observed_noisy[str(i)] = embedding_dict2 # user_embedding.to(device)
            
        self.use_fm = use_fm
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.use_linear = use_linear
        self.sigmoid = torch.nn.Sigmoid
        if use_fm :
            self.fm = FM()
        self.to(device)

    def fit_drfo(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, 
            save_models = True, ref_metric = None, tune_mode = 'min', 
            performance_mode = 'min', reg_kind = 0, performance_scale = 0.9,
            watch_performance_metrics = None, test_data = None, lambda_max = 1, lambda_min = 0, epsilon = 0, update_k_times = 1,
            train_label_users = None, label_weight = 1, train_nolabel_users= None, etas = None, forbid_users = None, unknow_weight = 0,
            epoch_multiplier = 1, epoch_multiplier2 = 1
            ):
        self.etas = etas
        train_tensor_data, val_x, val_y, do_validation, test_x, test_y, do_test = self.process_train_data(x, y, validation_data,
         validation_split,  test_data = test_data)
        if batch_size is None:
            batch_size = 256
        model = self.train()
        loss_func = self.loss_func
        # optim = self.optim
        optim1 = self.optim1
        optim2 = self.optim2
        # if
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle = shuffle, batch_size = batch_size, num_workers = 4)  
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        if ref_metric is None:
            ref_metric = "val_auc"
        stopers = myearlystopping(refer_metric = ref_metric, mode = tune_mode, performance_satisfied_stop_condition=130)
        report_dict = {}

        # get model parameters
        thetas = []
        for name, param in self.named_parameters():
            if 'user_embedding' not in name and 'lambda' not in name:
                thetas.append(param)

        if do_validation:
            eval_result = self.evaluate(val_x, val_y, batch_size)
            for name, result in eval_result.items():
                if name == self.performance_metric:
                    self.best_performance_value = result
        del_metrics = []
        # del_metrics = ['average_rating' + '_unknow' + '_sensitive_0', 'average_rating' + '_unknow' + '_sensitive_1',
        save_metrics = self.metrics.copy()
        run_metrics = {}
        for name, metric in save_metrics.items():
            if name not in del_metrics:
                run_metrics[name] = metric
        self.metrics = run_metrics.copy()
        num_batches = int(sample_num / batch_size)
        train_iter = iter(train_loader)
        train_iter_cycle = itertools.cycle(train_iter)
        old_epoch_logs = {}
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}
            loss_epoch = 0
            reg_loss_epoch = 0
            train_logs = {}
            begin_time = time.time()
            for batch_idx in range(num_batches):
                u_train, x_train, y_train = next(itertools.islice(train_iter_cycle, batch_idx, batch_idx + 1))
                    # for _, (u_train, x_train, y_train) in t:
                print('load_data, cost', time.time()-begin_time)
                begin_time = time.time()
                # begin_time = time.time()
                
                u = u_train.to(self.device).float()
                x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                y = y_train.to(self.device).float()
                print('trans to gpu, cost', time.time()-begin_time)
                begin_time = time.time()
                batch_train_logs = self.train_one_batch_v2(u = u, x = x, y = y, optim1 = optim1, optim2 = optim2, model = model,
                    loss_func = loss_func, thetas = thetas, reg_kind = reg_kind, lambda_max = lambda_max, lambda_min = lambda_min,
                    epsilon = epsilon, update_k_times = update_k_times, train_label_users = train_label_users, label_weight = label_weight,
                    train_nolabel_users = train_nolabel_users, forbid_users = forbid_users, unknow_weight = unknow_weight)
                # for param_name, param in self.lambda_dict.items():
                #     param.data.mul_(epoch_multiplier)
                # new_lr = self.optim1.param_groups[0]['lr'] * epoch_multiplier2
                
                print('train one batch, cost', time.time()-begin_time)
                # params = self.user_embedding_dict_dro['0']['user_id'].state_dict()

                begin_time = time.time()
                if train_logs == {}:
                    # train_logs[name] = batch_train_logs['batch_num']
                    for name, result in batch_train_logs.items():
                        if name != 'batch_num':
                            train_logs[name] = result * batch_train_logs['batch_num']
                            
                        else:
                            train_logs[name] = result
                else:
                    for name, result in batch_train_logs.items():
                        if name != 'batch_num':
                            train_logs[name] += result * batch_train_logs['batch_num']
                            
                        else:
                            train_logs[name] += result
                print('train finished, cost', time.time()-begin_time)
                begin_time = time.time()
            # except KeyboardInterrupt:
            #     t.close()
            #     raise
            # t.close()
            for name, result in train_logs.items():
                if name != 'batch_num':
                    train_logs[name] = result / train_logs['batch_num']
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose

            if do_test:
                test_eval_result = self.evaluate(test_x, test_y, batch_size)
                for name, result in test_eval_result.items():
                    epoch_logs["test_" + name] = result
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                if do_validation:
                    for name in epoch_logs:
                        if('val' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])
                                    
                if do_test:
                    for name in epoch_logs:
                        if('test' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])  
                print(eval_str)
            print('evlaute valid and test, cost', time.time()-begin_time)
            begin_time = time.time()
            if self.performance_metric is not None:
                performance_metric = 'val_' + self.performance_metric
                performance_satisfied = ((epoch_logs[performance_metric] > performance_scale * self.best_performance_value) and (performance_mode == 'max')) \
                    or ((epoch_logs[performance_metric]*performance_scale < self.best_performance_value) and (performance_mode == 'min'))
                print(epoch_logs[performance_metric] ,self.best_performance_value ,performance_satisfied )
            else:
                performance_satisfied = True
            self.metrics = save_metrics
            need_saving = stopers.update_and_isbest(epoch_logs, epoch, performance_satisfied = performance_satisfied)
            

            self.metrics = run_metrics
            for metric in watch_performance_metrics:
                if performance_satisfied:
                    epoch_logs[metric + '_performance_satisfied'] = epoch_logs[metric]
                    try:
                        epoch_logs[metric + '_performance_satisfied_top'] = stopers.best_eval_result[metric]
                    except:
                        epoch_logs[metric + '_performance_satisfied_top'] = 2.5
                else:
                    epoch_logs[metric + '_performance_satisfied'] = 2.5
                    try:
                        epoch_logs[metric + '_performance_satisfied_top'] = stopers.best_eval_result[metric]
                    except:
                        epoch_logs[metric + '_performance_satisfied_top'] = 2.5
                
            self.my_tune_report(epoch_logs, report_dict, epoch, train_logs = train_logs)
            if need_saving and performance_satisfied:
                logs = epoch_logs.copy()
                logs['val_dp_search'] = epoch_logs['val_dp']
                logs['val_dp_true_search'] = epoch_logs['val_dp_true']
                best_model_search = copy.deepcopy(self)
            need_stopping = stopers.is_stop()

            if need_stopping or (epoch == list(range(initial_epoch, epochs))[-1]):
                self.metrics = save_metrics.copy()
                self.my_save_model(epoch_logs = logs, need_saving = True, save_models = save_models, ref_metric = ref_metric, epoch = epoch, mode = tune_mode, model = best_model_search)
                self.metrics = run_metrics.copy()
                print("early stop.......")
                self.metrics = save_metrics.copy()
                break
        self.metrics = save_metrics.copy()
        return self.history      



    def train_one_batch(self, u, x, y, optim1, optim2, model,
     loss_func, thetas, reg_kind = 0, lambda_max = 1, lambda_min = 0, epsilon = 0, update_k_times = 1,
     train_label_users = None, label_weight = 1, train_nolabel_users = None, forbid_users = None):
        # optim1 use for update dro
        # optim2 use for update theta
        # sensitive_attribute = x[:,-1]
        # _:batch_num
        # items = x[:, -1]
        
        if train_label_users is not None:
            is_label = torch.isin(u.squeeze(), torch.tensor(train_label_users, device = self.device))
            label_indices = torch.where(is_label)
            # weights[is_label] = label_weight
        else:
            label_indices = None
        begin_time = time.time()
        batch_train_logs = {}
        batch_sensitive_attr = x[:, -1]
        y_pred = model(x).squeeze()
        optim1.zero_grad()
        optim2.zero_grad()
        loss = loss_func(y_pred, y.squeeze(),reduction='mean')
        reg_loss = self.get_regularization_loss()
        fair_constraints = {}
        sum_fair_constraint = 0
        iter_0 = 0
        iter_1 = 0
        eps = 1e-5
        sensitive_num = self.sensitive_num
        old_embedding = self.user_embedding_dict_dro[str(0)]['user_id'].weight.data.clone().detach()

        label_user_index = torch.isin(u.squeeze(), torch.tensor(train_label_users, device = self.device))
        nolabel_user_index = torch.isin(u.squeeze(), torch.tensor(train_nolabel_users, device = self.device))
        for i in range(sensitive_num):
            group_i_prob = self.get_dro_distribution_probability(u, i).squeeze() # * (1 + self.get_dro_distribution_probability(u, i).squeeze())
            print('sum_prob', torch.sum(group_i_prob))
            if i == 0:
                fair_constraints[str(i)] = torch.abs(torch.mean(y_pred).squeeze() -\
                self.etas['unknow_0'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze() -\
                self.etas['know_0'] * torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index]))
            if i == 1:
                fair_constraints[str(i)] = torch.abs(torch.mean(y_pred).squeeze() -\
                self.etas['unknow_1'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze() -\
                self.etas['know_1'] * torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index]))
            sum_fair_constraint += self.lambda_dict[str(i)] * fair_constraints[str(i)]
        (-sum_fair_constraint).backward(inputs = list(self.user_embedding_dict_dro.parameters()), retain_graph = True)
        (loss + reg_loss + sum_fair_constraint).backward(inputs = thetas)

        optim1.step()
        optim2.step()
        nolabel_user_tensor = torch.tensor(train_nolabel_users, device = self.device)
        with torch.no_grad():
            # set to 0
            for i in range(sensitive_num):
                mask = torch.zeros_like(self.user_embedding_dict_dro[str(i)]['user_id'].weight.data)
                mask.scatter_(dim=0, index=nolabel_user_tensor.unsqueeze(1), value=1)
                self.user_embedding_dict_dro[str(i)]['user_id'].weight.data *= mask 

            for i in range(sensitive_num):
                u_sort,_ = torch.sort(u.squeeze())
                no_label_index = torch.isin(u_sort, torch.tensor(train_nolabel_users, device = self.device))
                u_sort = u_sort[no_label_index].reshape(-1,1)
                p_dro = self.get_dro_distribution_probability(u_sort, i).squeeze()
                p_noisy = self.get_dro_distribution_probability_noisy(u_sort, i).squeeze()
                u_sort_df = pd.Series(u_sort.squeeze().clone().detach().cpu().numpy())
                cnt = pd.value_counts(u_sort_df)
                indices = []
                index = 0
                for j in cnt.index.unique().sort_values().tolist():
                    indices.append(index)
                    index += cnt[j]
                unique_index = torch.unique(u_sort.squeeze()).long()
 
                p_proj_l1 = (project_onto_l1_ball_my(p_dro - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()

                while True:
                    p_proj_simplex = project_simplex(p_proj_l1).squeeze()
                    p_proj_l1 = (project_onto_l1_ball_my(p_proj_simplex - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()
                    if i == 0:
                        iter_0 += 1
                    if i == 1:
                        iter_1 += 1
                    if torch.norm(p_proj_l1 - p_proj_simplex,p = 1) < eps:
                        p_proj = p_proj_l1.reshape(-1, 1)
                        break
                    if (i == 0 and iter_0 >= 50) or (i == 1 and iter_1 >= 50):
                        p_proj = p_proj_l1.reshape(-1, 1)
                        eps /= 0.9
                        break
                
                self.user_embedding_dict_dro[str(i)]['user_id'].weight.data[unique_index] = p_proj[indices]

        """
        if update_k_times > 1:
            for _ in range(update_k_times - 1):
                y_pred = model(x).squeeze()
                optim1.zero_grad()
                sum_fair_constraint = 0
                for i in range(sensitive_num):
                    group_i_prob = self.get_dro_distribution_probability(u, i).squeeze()
                    fair_constraints[str(i)] = self.get_constraint(y_pred, group_i_prob, reg_kind = reg_kind,
                     epsilon = epsilon, label_weight = label_weight, indices = label_indices)
                    sum_fair_constraint += self.lambda_dict[str(i)] * fair_constraints[str(i)]
                (-sum_fair_constraint).backward(inputs = list(self.user_embedding_dict_dro.parameters()))
                optim1.step()
                with torch.no_grad():
                    for i in range(sensitive_num):
                        u_sort,_ = torch.sort(u.squeeze())
                        u_sort = u_sort.reshape(-1,1)
                        p_dro = self.get_dro_distribution_probability(u_sort, i).squeeze()
                        p_noisy = self.get_dro_distribution_probability_noisy(u_sort, i).squeeze()
                        u_sort_df = pd.Series(u_sort.squeeze().clone().detach().cpu().numpy())
                        cnt = pd.value_counts(u_sort_df)
                        indices = []
                        index = 0
                        for j in cnt.index.unique().sort_values().tolist():
                            indices.append(index)
                            index += cnt[j]
                        unique_index = torch.unique(u_sort.squeeze()).long()
                        p_proj_l1 = (project_onto_l1_ball_my(p_dro - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()
                        begin_time = time.time()
                        iter_0 = 0
                        iter_1 = 0
                        while True:
                            p_proj_simplex = project_simplex(p_proj_l1).squeeze()
                            p_proj_l1 = (project_onto_l1_ball_my(p_proj_simplex - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()
                            if i == 0:
                                iter_0 += 1
                            if i == 1:
                                iter_1 += 1
                            if torch.norm(p_proj_l1 - p_proj_simplex,p = 1) < eps:
                                p_proj = p_proj_l1.reshape(-1, 1)
                                break
                            if (i == 0 and iter_0 >= 50) or (i == 1 and iter_1 >= 50):
                                p_proj = p_proj_l1.reshape(-1, 1)
                                eps /= 0.9
                                break
                        self.user_embedding_dict_dro[str(i)]['user_id'].weight.data[unique_index] = p_proj[indices]
        """
        with torch.no_grad():
            for i in range(sensitive_num):
                self.lambda_dict[str(i)].data += fair_constraints[str(i)] * self.lambda_lr
                # clamp lambda
                self.lambda_dict[str(i)] = torch.clamp(self.lambda_dict[str(i)], min = lambda_min, max = lambda_max)
            fair_reg_loss_not_abs = (torch.mean(y_pred[batch_sensitive_attr == 0]) - torch.mean(y_pred[batch_sensitive_attr == 1])).squeeze()
            fair_reg_loss = torch.abs(torch.mean(y_pred[batch_sensitive_attr == 0]) - torch.mean(y_pred[batch_sensitive_attr == 1])).squeeze()
            if reg_kind == 0:
                fair_constraints_0_true = (-torch.mean(y_pred[batch_sensitive_attr == 0]) + torch.mean(y_pred))
                fair_constraints_1_true = (-torch.mean(y_pred[batch_sensitive_attr == 1]) + torch.mean(y_pred))
            if reg_kind == 1:
                fair_constraints_0_true = torch.abs((-torch.mean(y_pred[batch_sensitive_attr == 0]) + torch.mean(y_pred)))
                fair_constraints_1_true = torch.abs((-torch.mean(y_pred[batch_sensitive_attr == 1]) + torch.mean(y_pred)))
        print('update dro prob cost', time.time() - begin_time)
        begin_time = time.time()
        new_embedding = self.user_embedding_dict_dro[str(0)]['user_id'].weight.data.clone().detach()
        batch_train_logs['embedding_distance'] = torch.norm(old_embedding - new_embedding, p = 1).squeeze().item()
        p_dro = self.get_dro_distribution_probability(u_sort, 0).squeeze()
        p_noisy = self.get_dro_distribution_probability_noisy(u_sort, 0).squeeze()
        batch_train_logs['prob_distance_0'] = torch.norm(p_dro - p_noisy, p = 1).squeeze().item() 
        p_dro = self.get_dro_distribution_probability(u_sort, 1).squeeze()
        p_noisy = self.get_dro_distribution_probability_noisy(u_sort, 1).squeeze()
        batch_train_logs['pro_distance_1'] = torch.norm(p_dro - p_noisy, p = 1).squeeze().item() 
        batch_train_logs['loss'] = loss.item()
        batch_train_logs['fair_reg_loss_not_abs'] = fair_reg_loss_not_abs.item()
        batch_train_logs['fair_reg_loss'] = fair_reg_loss.item()
        batch_train_logs['fair_constraint_0_dro'] = fair_constraints[str(0)].item()
        batch_train_logs['fair_constraint_1_dro'] = fair_constraints[str(1)].item()
        batch_train_logs['fair_constraint_0_true'] = fair_constraints_0_true.item()
        batch_train_logs['fair_constraint_1_true'] = fair_constraints_1_true.item()
        batch_train_logs['fair_constraint_0_dro-true'] =  fair_constraints[str(0)].item() - fair_constraints_0_true.item()
        batch_train_logs['fair_constraint_1_dro-true'] =  fair_constraints[str(1)].item() - fair_constraints_1_true.item()
        batch_train_logs['batch_num'] = y.shape[0]
        batch_train_logs['iter_0'] = iter_0
        batch_train_logs['iter_1'] = iter_1
        batch_train_logs['lambda_0'] = self.lambda_dict[str(0)].data.squeeze().item()
        batch_train_logs['lambda_1'] = self.lambda_dict[str(1)].data.squeeze().item()
        batch_train_logs['train_dro_dp'] = fair_constraints[str(0)].item() + fair_constraints[str(1)].item()
        return batch_train_logs

    def train_one_batch_v2(self, u, x, y, optim1, optim2, model,
     loss_func, thetas, reg_kind = 0, lambda_max = 1, lambda_min = 0, epsilon = 0, update_k_times = 1,
     train_label_users = None, label_weight = 1, train_nolabel_users = None, forbid_users = None, know_weight = 0, unknow_weight = 1):
        # optim1 use for update dro
        # optim2 use for update theta
        # sensitive_attribute = x[:,-1]
        # _:batch_num
        # items = x[:, -1]
        
        if train_label_users is not None:
            is_label = torch.isin(u.squeeze(), torch.tensor(train_label_users, device = self.device))
            label_indices = torch.where(is_label)
            # weights[is_label] = label_weight
        else:
            label_indices = None
        begin_time = time.time()
        batch_train_logs = {}
        batch_sensitive_attr = x[:, -1]
        y_pred = model(x).squeeze()
        optim1.zero_grad()
        optim2.zero_grad()
        loss = loss_func(y_pred, y.squeeze(),reduction='mean')
        reg_loss = self.get_regularization_loss()
        fair_constraints = {}
        sum_fair_constraint = 0
        iter_0 = 0
        iter_1 = 0
        eps = 1e-5
        sensitive_num = 1 # self.sensitive_num 
        sensitive_range = [1]
        old_embedding = self.user_embedding_dict_dro[str(0)]['user_id'].weight.data.clone().detach()

        label_user_index = torch.isin(u.squeeze(), torch.tensor(train_label_users, device = self.device))
        nolabel_user_index = torch.isin(u.squeeze(), torch.tensor(train_nolabel_users, device = self.device))
        for i in sensitive_range: # range(sensitive_num):
            group_i_prob = self.get_dro_distribution_probability(u, i).squeeze() # * (1 + self.get_dro_distribution_probability(u, i).squeeze())
            print('sum_prob', torch.sum(group_i_prob))
            if i == 0:
                fair_constraints[str(i)] = torch.abs(torch.mean(y_pred).squeeze() -\
                self.etas['unknow_0'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze() -\
                self.etas['know_0'] * torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index])) 
                # fair_constraints[str(i)] = torch.abs((torch.sum(y_pred.squeeze()[label_user_index]).squeeze() + torch.sum(y_pred.squeeze()[nolabel_user_index]).squeeze() * unknow_weight)/y_pred.shape[0] -\
                # torch.abs(torch.mean(y_pred).squeeze() -\
                # +\
                # unknow_weight * torch.abs(torch.mean(y_pred).squeeze() -\
                # torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze())
                # torch.abs(torch.mean(y_pred).squeeze() -\
                # torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index]))
                # fair_constraints[str(i)] = self.etas['know_0'] * torch.abs(torch.mean(y_pred).squeeze() -\
                # torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index])) +\
                # torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze())

            if i == 1:
                fair_constraints[str(i)] = torch.abs(torch.mean(y_pred).squeeze() -\
                self.etas['unknow_1'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze() -\
                self.etas['know_1'] * torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index])) 
                # +unknow_weight * torch.abs(torch.mean(y_pred).squeeze() -\
                # torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze())
                # torch.abs(torch.mean(y_pred).squeeze() -\
                # torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index]))
                # fair_constraints[str(i)] =self.etas['know_1'] * torch.abs(torch.mean(y_pred).squeeze() -\
                # torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index])) +\
                # torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index], group_i_prob[nolabel_user_index])).squeeze())
            sum_fair_constraint += self.lambda_dict[str(i)] * fair_constraints[str(i)]
        (-sum_fair_constraint).backward(inputs = list(self.user_embedding_dict_dro.parameters()), retain_graph = True)
        (loss + reg_loss + sum_fair_constraint).backward(inputs = thetas)

        optim1.step()
        optim2.step()
        nolabel_user_tensor = torch.tensor(train_nolabel_users, device = self.device)
        with torch.no_grad():
            # set to 0
            for i in sensitive_range: #range(sensitive_num):
                mask = torch.zeros_like(self.user_embedding_dict_dro[str(i)]['user_id'].weight.data)
                mask.scatter_(dim=0, index=nolabel_user_tensor.unsqueeze(1), value=1)
                self.user_embedding_dict_dro[str(i)]['user_id'].weight.data *= mask 

            for i in sensitive_range: #range(sensitive_num):
                u_sort,_ = torch.sort(u.squeeze())
                no_label_index = torch.isin(u_sort, torch.tensor(train_nolabel_users, device = self.device))
                u_sort = u_sort[no_label_index].reshape(-1,1)
                p_dro = self.get_dro_distribution_probability(u_sort, i).squeeze()
                p_noisy = self.get_dro_distribution_probability_noisy(u_sort, i).squeeze()
                u_sort_df = pd.Series(u_sort.squeeze().clone().detach().cpu().numpy())
                cnt = pd.value_counts(u_sort_df)
                indices = []
                index = 0
                for j in cnt.index.unique().sort_values().tolist():
                    indices.append(index)
                    index += cnt[j]
                unique_index = torch.unique(u_sort.squeeze()).long()
 
                p_proj_l1 = (project_onto_l1_ball_my(p_dro - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()

                while True:
                    p_proj_simplex = project_simplex(p_proj_l1).squeeze()
                    p_proj_l1 = (project_onto_l1_ball_my(p_proj_simplex - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()
                    if i == 0:
                        iter_0 += 1
                    if i == 1:
                        iter_1 += 1
                    if torch.norm(p_proj_l1 - p_proj_simplex,p = 1) < eps:
                        p_proj = p_proj_l1.reshape(-1, 1)
                        break
                    if (i == 0 and iter_0 >= 50) or (i == 1 and iter_1 >= 50):
                        p_proj = p_proj_l1.reshape(-1, 1)
                        eps /= 0.9
                        break
                self.user_embedding_dict_dro[str(i)]['user_id'].weight.data[unique_index] = p_proj[indices]


        
        # begin_time = time.time()
        # new_embedding = self.user_embedding_dict_dro[str(0)]['user_id'].weight.data.clone().detach()
        # batch_train_logs['embedding_distance'] = torch.norm(old_embedding - new_embedding, p = 1).squeeze().item()
        # p_dro = self.get_dro_distribution_probability(u_sort, 0).squeeze()
        # p_noisy = self.get_dro_distribution_probability_noisy(u_sort, 0).squeeze()
        # batch_train_logs['prob_distance_0'] = torch.norm(p_dro - p_noisy, p = 1).squeeze().item() 
        # p_dro = self.get_dro_distribution_probability(u_sort, 1).squeeze()
        # p_noisy = self.get_dro_distribution_probability_noisy(u_sort, 1).squeeze()
        # batch_train_logs['pro_distance_1'] = torch.norm(p_dro - p_noisy, p = 1).squeeze().item() 
        # batch_train_logs['loss'] = loss.item()
        # batch_train_logs['fair_reg_loss_not_abs'] = fair_reg_loss_not_abs.item()
        # batch_train_logs['fair_reg_loss'] = fair_reg_loss.item()
        # batch_train_logs['fair_constraint_0_dro'] = fair_constraints[str(0)].item()
        # batch_train_logs['fair_constraint_1_dro'] = fair_constraints[str(1)].item()
        # batch_train_logs['fair_constraint_0_true'] = fair_constraints_0_true.item()
        # batch_train_logs['fair_constraint_1_true'] = fair_constraints_1_true.item()
        # batch_train_logs['fair_constraint_0_dro-true'] =  fair_constraints[str(0)].item() - fair_constraints_0_true.item()
        # batch_train_logs['fair_constraint_1_dro-true'] =  fair_constraints[str(1)].item() - fair_constraints_1_true.item()
        # batch_train_logs['batch_num'] = y.shape[0]
        # batch_train_logs['iter_0'] = iter_0
        # batch_train_logs['iter_1'] = iter_1
        # batch_train_logs['lambda_0'] = self.lambda_dict[str(0)].data.squeeze().item()
        # batch_train_logs['lambda_1'] = self.lambda_dict[str(1)].data.squeeze().item()
        
        # batch_train_logs['train_dro_dp'] = fair_constraints[str(0)].item() + fair_constraints[str(1)].item()
        return batch_train_logs



class DRFOWithRefusal(DRFOBase):
    # dro under equal rating constraint
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns,user_feat_column,group_distribution_profile, group_distribution_lack_profile
                 , sensitive_num = 2, gammas = [0.1,0.1],
                  init_lambdas = 0.1, lambda_lr = 0.1, 
                  use_fm=True, l2_reg_linear=0.00001, l2_reg_embedding = 0.00001,  init_std = 0.0001, seed = 1024,
                  task='binary', device='cpu', gpus=None, use_linear = True,
                   explicit = True, use_thre = False ):

        
        MybaseModel.__init__(self,linear_feature_columns = linear_feature_columns, dnn_feature_columns = dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        self.explicit = explicit
        self.use_thre = use_thre
        self.sensitive_num = sensitive_num
        sample_num = group_distribution_profile['0'].shape[0]
        self.gammas = gammas
        self.user_feat_column = user_feat_column
        self.user_feature_index = build_input_features(user_feat_column)
        self.lambda_lr = lambda_lr
        self.user_embedding_dict_dro_profile = nn.ModuleDict({})
        self.user_embedding_dict_observed_noisy_profile = nn.ModuleDict({})
        self.user_embedding_dict_dro_lack_profile = nn.ModuleDict({})
        self.user_embedding_dict_observed_noisy_lack_profile = nn.ModuleDict({})
        self.lambda_dict = nn.ParameterDict({str(i):nn.Parameter(torch.ones(1) * init_lambdas) for i in range(sensitive_num)})
        # observed_group_distribution should be sorted in order for user 0, 1, 2, 3, 4 ...
        for i in range(sensitive_num):
            # get dro_probability and save as user_embedding
            embedding_dict = nn.ModuleDict({})
            user_embedding = torch.from_numpy(group_distribution_profile[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict['user_id'] = nn.Embedding.from_pretrained(user_embedding, freeze = False)
            self.user_embedding_dict_dro_profile[str(i)] = embedding_dict
            # get obser_probability
            embedding_dict2 = nn.ModuleDict({})
            user_embedding2 = torch.from_numpy(group_distribution_profile[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict2['user_id'] = nn.Embedding.from_pretrained(user_embedding2, freeze = True)
            self.user_embedding_dict_observed_noisy_profile[str(i)] = embedding_dict2 # user_embedding.to(device)

            # get dro_probability_lack_profile
            embedding_dict3 = nn.ModuleDict({})
            user_embedding3 = torch.from_numpy(group_distribution_lack_profile[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict3['user_id'] = nn.Embedding.from_pretrained(user_embedding3, freeze = False)
            self.user_embedding_dict_dro_lack_profile[str(i)] = embedding_dict3
            # get obser_probability
            embedding_dict4 = nn.ModuleDict({})
            user_embedding4 = torch.from_numpy(group_distribution_lack_profile[str(i)]).unsqueeze(1).type(torch.float64)
            embedding_dict4['user_id'] = nn.Embedding.from_pretrained(user_embedding4, freeze = True)
            self.user_embedding_dict_observed_noisy_lack_profile[str(i)] = embedding_dict4

        self.use_fm = use_fm
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.use_linear = use_linear
        self.sigmoid = torch.nn.Sigmoid
        if use_fm :
            self.fm = FM()
        self.to(device)

    def get_dro_distribution_probability_profile(self, U, i):
        sparse_embedding_list, dense_value_list = self.my_input_from_feature_columns(U, self.user_feat_column,
                                                                                  self.user_embedding_dict_dro_profile[str(i)], self.user_feature_index)                                                            
        return torch.cat(sparse_embedding_list,dim = 1).squeeze()

    def get_dro_distribution_probability_noisy_profile(self, U, i):
        sparse_embedding_list, dense_value_list = self.my_input_from_feature_columns(U, self.user_feat_column,
                                                                                  self.user_embedding_dict_observed_noisy_profile[str(i)], self.user_feature_index)                                                            
        return torch.cat(sparse_embedding_list,dim = 1).squeeze()

    def get_dro_distribution_probability_lack_profile(self, U, i):
        sparse_embedding_list, dense_value_list = self.my_input_from_feature_columns(U, self.user_feat_column,
                                                                                  self.user_embedding_dict_dro_lack_profile[str(i)], self.user_feature_index)                                                            
        return torch.cat(sparse_embedding_list,dim = 1).squeeze()

    def get_dro_distribution_probability_noisy_lack_profile(self, U, i):
        sparse_embedding_list, dense_value_list = self.my_input_from_feature_columns(U, self.user_feat_column,
                                                                                  self.user_embedding_dict_observed_noisy_lack_profile[str(i)], self.user_feature_index)                                                            
        return torch.cat(sparse_embedding_list,dim = 1).squeeze()

    def fit_drfo(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, 
            save_models = True, ref_metric = None, tune_mode = 'min', 
            performance_mode = 'min', reg_kind = 0, performance_scale = 0.9,
            watch_performance_metrics = None, test_data = None, lambda_max = 1, lambda_min = 0, epsilon = 0, update_k_times = 1,
            train_label_users = None, label_weight = 1, train_nolabel_users= None, etas = None, 
            train_nolabel_users_profile = [],  train_nolabel_users_lack_profile = []
           , epoch_multiplier = 1            ):
        self.etas = etas
        train_tensor_data, val_x, val_y, do_validation, test_x, test_y, do_test = self.process_train_data(x, y, validation_data,
         validation_split,  test_data = test_data)
        if batch_size is None:
            batch_size = 256
        model = self.train()
        loss_func = self.loss_func
        # optim = self.optim
        optim1 = self.optim1
        optim2 = self.optim2
        # if
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle = shuffle, batch_size = batch_size, num_workers = 4)  
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        if ref_metric is None:
            ref_metric = "val_auc"
        stopers = myearlystopping(refer_metric = ref_metric, mode = tune_mode, performance_satisfied_stop_condition=130)
        report_dict = {}

        # get model parameters
        thetas = []
        for name, param in self.named_parameters():
            if 'user_embedding' not in name and 'lambda' not in name:
                thetas.append(param)

        if do_validation:
            eval_result = self.evaluate(val_x, val_y, batch_size)
            for name, result in eval_result.items():
                if name == self.performance_metric:
                    self.best_performance_value = result
        del_metrics = []
        # del_metrics = ['average_rating' + '_unknow' + '_sensitive_0', 'average_rating' + '_unknow' + '_sensitive_1',
        save_metrics = self.metrics.copy()
        run_metrics = {}
        for name, metric in save_metrics.items():
            if name not in del_metrics:
                run_metrics[name] = metric
        self.metrics = run_metrics.copy()
        num_batches = int(sample_num / batch_size)
        train_iter = iter(train_loader)
        train_iter_cycle = itertools.cycle(train_iter)
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}
            loss_epoch = 0
            reg_loss_epoch = 0
            train_logs = {}
            begin_time = time.time()
            for batch_idx in range(num_batches):
                u_train, x_train, y_train = next(itertools.islice(train_iter_cycle, batch_idx, batch_idx + 1))
                    # for _, (u_train, x_train, y_train) in t:
                print('load_data, cost', time.time()-begin_time)
                begin_time = time.time()
                # begin_time = time.time()
                
                u = u_train.to(self.device).float()
                x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                y = y_train.to(self.device).float()
                print('trans to gpu, cost', time.time()-begin_time)
                begin_time = time.time()
                batch_train_logs = self.train_one_batch(u = u, x = x, y = y, optim1 = optim1, optim2 = optim2, model = model,
                    loss_func = loss_func, thetas = thetas, reg_kind = reg_kind, lambda_max = lambda_max, lambda_min = lambda_min,
                    epsilon = epsilon, update_k_times = update_k_times, train_label_users = train_label_users, label_weight = label_weight,
                    train_nolabel_users = train_nolabel_users, train_nolabel_users_profile = train_nolabel_users_profile, 
                     train_nolabel_users_lack_profile = train_nolabel_users_lack_profile)
                # for param_name, param in self.lambda_dict.items():
                #     param.data.mul_(epoch_multiplier)
                print('train one batch, cost', time.time()-begin_time)
                # params = self.user_embedding_dict_dro['0']['user_id'].state_dict()

                begin_time = time.time()
                if train_logs == {}:
                    # train_logs[name] = batch_train_logs['batch_num']
                    for name, result in batch_train_logs.items():
                        if name != 'batch_num':
                            train_logs[name] = result * batch_train_logs['batch_num']
                            
                        else:
                            train_logs[name] = result
                else:
                    for name, result in batch_train_logs.items():
                        if name != 'batch_num':
                            train_logs[name] += result * batch_train_logs['batch_num']
                            
                        else:
                            train_logs[name] += result
                print('train finished, cost', time.time()-begin_time)
                begin_time = time.time()
            # except KeyboardInterrupt:
            #     t.close()
            #     raise
            # t.close()
            for name, result in train_logs.items():
                if name != 'batch_num':
                    train_logs[name] = result / train_logs['batch_num']
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose

            if do_test:
                test_eval_result = self.evaluate(test_x, test_y, batch_size)
                for name, result in test_eval_result.items():
                    epoch_logs["test_" + name] = result
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                if do_validation:
                    for name in epoch_logs:
                        if('val' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])
                                    
                if do_test:
                    for name in epoch_logs:
                        if('test' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])  
                print(eval_str)
            print('evlaute valid and test, cost', time.time()-begin_time)
            begin_time = time.time()
            if self.performance_metric is not None:
                performance_metric = 'val_' + self.performance_metric
                performance_satisfied = ((epoch_logs[performance_metric] > performance_scale * self.best_performance_value) and (performance_mode == 'max')) \
                    or ((epoch_logs[performance_metric]*performance_scale < self.best_performance_value) and (performance_mode == 'min'))
                print(epoch_logs[performance_metric] ,self.best_performance_value ,performance_satisfied )
            else:
                performance_satisfied = True
            self.metrics = save_metrics
            need_saving = stopers.update_and_isbest(epoch_logs, epoch, performance_satisfied = performance_satisfied)

            self.metrics = run_metrics
            for metric in watch_performance_metrics:
                if performance_satisfied:
                    epoch_logs[metric + '_performance_satisfied'] = epoch_logs[metric]
                    try:
                        epoch_logs[metric + '_performance_satisfied_top'] = stopers.best_eval_result[metric]
                    except:
                        epoch_logs[metric + '_performance_satisfied_top'] = 2.5
                else:
                    epoch_logs[metric + '_performance_satisfied'] = 2.5
                    try:
                        epoch_logs[metric + '_performance_satisfied_top'] = stopers.best_eval_result[metric]
                    except:
                        epoch_logs[metric + '_performance_satisfied_top'] = 2.5
                
            self.my_tune_report(epoch_logs, report_dict, epoch, train_logs = train_logs)
            if need_saving and performance_satisfied:
                logs = epoch_logs.copy()
                logs['val_dp_search'] = epoch_logs['val_dp']
                logs['val_dp_true_search'] = epoch_logs['val_dp_true']
                best_model_search = copy.deepcopy(self)
            need_stopping = stopers.is_stop()

            if need_stopping or (epoch == list(range(initial_epoch, epochs))[-1]):
                self.metrics = save_metrics.copy()
                self.my_save_model(epoch_logs = logs, need_saving = True, save_models = save_models, ref_metric = ref_metric, epoch = epoch, mode = tune_mode, model = best_model_search)
                self.metrics = run_metrics.copy()
                print("early stop.......")
                self.metrics = save_metrics.copy()
                break
        self.metrics = save_metrics.copy()
        return self.history      
    
    def train_one_batch(self, u, x, y, optim1, optim2, model,
     loss_func, thetas, reg_kind = 0, lambda_max = 1, lambda_min = 0, epsilon = 0, update_k_times = 1,
     train_label_users = None, label_weight = 1, train_nolabel_users = None,
     train_nolabel_users_lack_profile = [], train_nolabel_users_profile = []):
        # optim1 use for update dro
        # optim2 use for update theta
        # sensitive_attribute = x[:,-1]
        # _:batch_num
        # items = x[:, -1]
        
        # if train_label_users is not None:
            # is_label = torch.isin(u.squeeze(), torch.tensor(train_label_users, device = self.device))
            # label_indices = torch.where(is_label)
            # weights[is_label] = label_weight
            # label_indices = None
        begin_time = time.time()
        batch_train_logs = {}
        batch_sensitive_attr = x[:, -1]
        y_pred = model(x).squeeze()
        optim1.zero_grad()
        optim2.zero_grad()
        loss = loss_func(y_pred, y.squeeze(),reduction='mean')
        reg_loss = self.get_regularization_loss()
        fair_constraints = {}
        sum_fair_constraint = 0
        iter_0 = 0
        iter_1 = 0
        eps = 1e-5
        sensitive_num = 1 # self.sensitive_num
        sensitive_range = [1]
        label_user_index = torch.isin(u.squeeze(), torch.tensor(train_label_users, device = self.device))
        if len(train_nolabel_users_lack_profile):
            nolabel_user_index_lack_profile = torch.isin(u.squeeze(), torch.tensor(train_nolabel_users_lack_profile,
             device = self.device))
        else:
            nolabel_user_index_lack_profile = torch.zeros_like(u.squeeze(), dtype=torch.bool)
        if len(train_nolabel_users_profile):
            nolabel_user_index_profile = torch.isin(u.squeeze(), torch.tensor(train_nolabel_users_profile,
             device = self.device))
        else:
            nolabel_user_index_profile = torch.zeros_like(u.squeeze(), dtype=torch.bool)

        # nolabel_user_index = torch.isin(u.squeeze(), torch.tensor(train_nolabel_users, device = self.device))
        for i in sensitive_range: # range(sensitive_num):
            # group_i_prob = self.get_dro_distribution_probability(u, i).squeeze() # * (1 + self.get_dro_distribution_probability(u, i).squeeze())
            group_i_prob_profile = self.get_dro_distribution_probability_profile(u, i).squeeze()
            group_i_prob_lack_profile = self.get_dro_distribution_probability_lack_profile(u, i).squeeze()
            print('sum_prob_file', torch.sum(group_i_prob_profile))
            if i == 0:
                if len(train_nolabel_users_profile):
                    fair_constraints[str(i)] = torch.abs(torch.mean(y_pred).squeeze() -\
                    self.etas['unknow_0_profile'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index_profile],
                    group_i_prob_profile[nolabel_user_index_profile])).squeeze() -\
                    self.etas['unknow_0_lack_profile'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index_lack_profile],
                    group_i_prob_lack_profile[nolabel_user_index_lack_profile])).squeeze() -\
                    self.etas['know_0'] * torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index]))
                    
                else:
                    fair_constraints[str(i)] = torch.abs(torch.mean(y_pred).squeeze() -\
                    self.etas['unknow_0_lack_profile'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index_lack_profile],
                    group_i_prob_lack_profile[nolabel_user_index_lack_profile])).squeeze() -\
                    self.etas['know_0'] * torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index]))
            


            if i == 1:
                if len(train_nolabel_users_profile):
                    fair_constraints[str(i)] = torch.abs(torch.mean(y_pred).squeeze() -\
                    self.etas['unknow_1_profile'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index_profile] ,
                    group_i_prob_profile[nolabel_user_index_profile])).squeeze() -\
                    self.etas['unknow_1_lack_profile'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index_lack_profile],
                    group_i_prob_lack_profile[nolabel_user_index_lack_profile])).squeeze() -\
                    self.etas['know_1'] * torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index]))
                else:
                    fair_constraints[str(i)] = torch.abs(torch.mean(y_pred).squeeze() -\
                    self.etas['unknow_1_lack_profile'] * torch.sum(torch.mul(y_pred.squeeze()[nolabel_user_index_lack_profile],
                    group_i_prob_lack_profile[nolabel_user_index_lack_profile])).squeeze() -\
                    self.etas['know_1'] * torch.mean(y_pred.squeeze()[(batch_sensitive_attr == i) & label_user_index]))    

            sum_fair_constraint += self.lambda_dict[str(i)] * fair_constraints[str(i)]

        (-sum_fair_constraint).backward(inputs = list(itertools.chain(self.user_embedding_dict_dro_profile.parameters(),
            self.user_embedding_dict_dro_lack_profile.parameters())), retain_graph = True)
        (loss + reg_loss + sum_fair_constraint).backward(inputs = thetas)

        optim1.step()
        optim2.step()
        # nolabel_user_tensor = torch.tensor(train_nolabel_users, device = self.device)
        if len(train_nolabel_users_profile):
            nolabel_user_tensor_profile = torch.tensor(train_nolabel_users_profile, device = self.device)
            with torch.no_grad():
                # set to 0
                for i in sensitive_range: # range(sensitive_num):
                    mask = torch.zeros_like(self.user_embedding_dict_dro_profile[str(i)]['user_id'].weight.data)
                    mask.scatter_(dim=0, index=nolabel_user_tensor_profile.unsqueeze(1), value=1)
                    self.user_embedding_dict_dro_profile[str(i)]['user_id'].weight.data *= mask 

                for i in sensitive_range: # range(sensitive_num):
                    u_sort,_ = torch.sort(u.squeeze())
                    no_label_index = torch.isin(u_sort, torch.tensor(train_nolabel_users_profile, device = self.device))
                    u_sort = u_sort[no_label_index].reshape(-1,1)
                    p_dro = self.get_dro_distribution_probability_profile(u_sort, i).squeeze()
                    p_noisy = self.get_dro_distribution_probability_noisy_profile(u_sort, i).squeeze()
                    u_sort_df = pd.Series(u_sort.squeeze().clone().detach().cpu().numpy())
                    cnt = pd.value_counts(u_sort_df)
                    indices = []
                    index = 0
                    for j in cnt.index.unique().sort_values().tolist():
                        indices.append(index)
                        index += cnt[j]
                    unique_index = torch.unique(u_sort.squeeze()).long()
    
                    p_proj_l1 = (project_onto_l1_ball_my(p_dro - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()

                    while True:
                        p_proj_simplex = project_simplex(p_proj_l1).squeeze()
                        p_proj_l1 = (project_onto_l1_ball_my(p_proj_simplex - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()
                        if i == 0:
                            iter_0 += 1
                        if i == 1:
                            iter_1 += 1
                        if torch.norm(p_proj_l1 - p_proj_simplex,p = 1) < eps:
                            p_proj = p_proj_l1.reshape(-1, 1)
                            break
                        if (i == 0 and iter_0 >= 50) or (i == 1 and iter_1 >= 50):
                            p_proj = p_proj_l1.reshape(-1, 1)
                            eps /= 0.9
                            break
                    self.user_embedding_dict_dro_profile[str(i)]['user_id'].weight.data[unique_index] = p_proj[indices]
        else:
            with torch.no_grad():
                # set to 0
                for i in sensitive_range:
                    mask = torch.zeros_like(self.user_embedding_dict_dro_profile[str(i)]['user_id'].weight.data)
                    self.user_embedding_dict_dro_profile[str(i)]['user_id'].weight.data *= mask 


        if len(train_nolabel_users_lack_profile):
            nolabel_user_tensor_lack_profile = torch.tensor(train_nolabel_users_lack_profile, device = self.device)
            with torch.no_grad():
                # set to 0
                for i in sensitive_range: # range(sensitive_num):
                    mask = torch.zeros_like(self.user_embedding_dict_dro_lack_profile[str(i)]['user_id'].weight.data)
                    mask.scatter_(dim=0, index=nolabel_user_tensor_lack_profile.unsqueeze(1), value=1)
                    self.user_embedding_dict_dro_lack_profile[str(i)]['user_id'].weight.data *= mask 

                for i in sensitive_range: # range(sensitive_num):
                    u_sort,_ = torch.sort(u.squeeze())
                    no_label_index = torch.isin(u_sort, torch.tensor(train_nolabel_users_lack_profile, device = self.device))
                    u_sort = u_sort[no_label_index].reshape(-1,1)
                    p_dro = self.get_dro_distribution_probability_lack_profile(u_sort, i).squeeze()
                    p_noisy = self.get_dro_distribution_probability_noisy_lack_profile(u_sort, i).squeeze()
                    u_sort_df = pd.Series(u_sort.squeeze().clone().detach().cpu().numpy())
                    cnt = pd.value_counts(u_sort_df)
                    indices = []
                    index = 0
                    for j in cnt.index.unique().sort_values().tolist():
                        indices.append(index)
                        index += cnt[j]
                    unique_index = torch.unique(u_sort.squeeze()).long()
    
                    p_proj_l1 = (project_onto_l1_ball_my(p_dro - p_noisy, 2 * self.gammas2[i])  + p_noisy).squeeze()

                    while True:
                        p_proj_simplex = project_simplex(p_proj_l1).squeeze()
                        p_proj_l1 = (project_onto_l1_ball_my(p_proj_simplex - p_noisy, 2 * self.gammas2[i])  + p_noisy).squeeze()
                        if i == 0:
                            iter_0 += 1
                        if i == 1:
                            iter_1 += 1
                        if torch.norm(p_proj_l1 - p_proj_simplex,p = 1) < eps:
                            p_proj = p_proj_l1.reshape(-1, 1)
                            break
                        if (i == 0 and iter_0 >= 50) or (i == 1 and iter_1 >= 50):
                            p_proj = p_proj_l1.reshape(-1, 1)
                            eps /= 0.9
                            break
                    self.user_embedding_dict_dro_lack_profile[str(i)]['user_id'].weight.data[unique_index] = p_proj[indices]
        else:
            with torch.no_grad():
                # set to 0
                for i in sensitive_range: # range(sensitive_num):
                    mask = torch.zeros_like(self.user_embedding_dict_dro_lack_profile[str(i)]['user_id'].weight.data)
                    self.user_embedding_dict_dro_lack_profile[str(i)]['user_id'].weight.data *= mask 
            
        """
        if update_k_times > 1:
            for _ in range(update_k_times - 1):
                y_pred = model(x).squeeze()
                optim1.zero_grad()
                sum_fair_constraint = 0
                for i in range(sensitive_num):
                    group_i_prob = self.get_dro_distribution_probability(u, i).squeeze()
                    fair_constraints[str(i)] = self.get_constraint(y_pred, group_i_prob, reg_kind = reg_kind,
                     epsilon = epsilon, label_weight = label_weight, indices = label_indices)
                    sum_fair_constraint += self.lambda_dict[str(i)] * fair_constraints[str(i)]
                (-sum_fair_constraint).backward(inputs = list(self.user_embedding_dict_dro.parameters()))
                optim1.step()
                with torch.no_grad():
                    for i in range(sensitive_num):
                        u_sort,_ = torch.sort(u.squeeze())
                        u_sort = u_sort.reshape(-1,1)
                        p_dro = self.get_dro_distribution_probability(u_sort, i).squeeze()
                        p_noisy = self.get_dro_distribution_probability_noisy(u_sort, i).squeeze()
                        u_sort_df = pd.Series(u_sort.squeeze().clone().detach().cpu().numpy())
                        cnt = pd.value_counts(u_sort_df)
                        indices = []
                        index = 0
                        for j in cnt.index.unique().sort_values().tolist():
                            indices.append(index)
                            index += cnt[j]
                        unique_index = torch.unique(u_sort.squeeze()).long()
                        p_proj_l1 = (project_onto_l1_ball_my(p_dro - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()
                        begin_time = time.time()
                        iter_0 = 0
                        iter_1 = 0
                        while True:
                            p_proj_simplex = project_simplex(p_proj_l1).squeeze()
                            p_proj_l1 = (project_onto_l1_ball_my(p_proj_simplex - p_noisy, 2 * self.gammas[i])  + p_noisy).squeeze()
                            if i == 0:
                                iter_0 += 1
                            if i == 1:
                                iter_1 += 1
                            if torch.norm(p_proj_l1 - p_proj_simplex,p = 1) < eps:
                                p_proj = p_proj_l1.reshape(-1, 1)
                                break
                            if (i == 0 and iter_0 >= 50) or (i == 1 and iter_1 >= 50):
                                p_proj = p_proj_l1.reshape(-1, 1)
                                eps /= 0.9
                                break
                        self.user_embedding_dict_dro[str(i)]['user_id'].weight.data[unique_index] = p_proj[indices]
        """
        # begin_time = time.time()
        # new_embedding = self.user_embedding_dict_dro[str(0)]['user_id'].weight.data.clone().detach()
        # batch_train_logs['embedding_distance'] = torch.norm(old_embedding - new_embedding, p = 1).squeeze().item()
        # p_dro = self.get_dro_distribution_probability(u_sort, 0).squeeze()
        # p_noisy = self.get_dro_distribution_probability_noisy(u_sort, 0).squeeze()
        # batch_train_logs['prob_distance_0'] = torch.norm(p_dro - p_noisy, p = 1).squeeze().item() 
        # p_dro = self.get_dro_distribution_probability(u_sort, 1).squeeze()
        # p_noisy = self.get_dro_distribution_probability_noisy(u_sort, 1).squeeze()
        # batch_train_logs['pro_distance_1'] = torch.norm(p_dro - p_noisy, p = 1).squeeze().item() 
        # batch_train_logs['loss'] = loss.item()
        # batch_train_logs['fair_reg_loss_not_abs'] = fair_reg_loss_not_abs.item()
        # batch_train_logs['fair_reg_loss'] = fair_reg_loss.item()
        # batch_train_logs['fair_constraint_0_dro'] = fair_constraints[str(0)].item()
        # batch_train_logs['fair_constraint_1_dro'] = fair_constraints[str(1)].item()
        # batch_train_logs['fair_constraint_0_true'] = fair_constraints_0_true.item()
        # batch_train_logs['fair_constraint_1_true'] = fair_constraints_1_true.item()
        # batch_train_logs['fair_constraint_0_dro-true'] =  fair_constraints[str(0)].item() - fair_constraints_0_true.item()
        # batch_train_logs['fair_constraint_1_dro-true'] =  fair_constraints[str(1)].item() - fair_constraints_1_true.item()
        # batch_train_logs['batch_num'] = y.shape[0]
        # batch_train_logs['iter_0'] = iter_0
        # batch_train_logs['iter_1'] = iter_1
        # batch_train_logs['lambda_0'] = self.lambda_dict[str(0)].data.squeeze().item()
        # batch_train_logs['lambda_1'] = self.lambda_dict[str(1)].data.squeeze().item()
        return batch_train_logs


class MyLR(MybaseModel):
    # used for predict sensitive attributes
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, init_std=0.0001, seed=1024,
                  task='binary', device='cpu', gpus=None):
        nn.Module.__init__(self)
        self.dnn_feature_columns = dnn_feature_columns
        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")
        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.regularization_weight = []
        self.out = PredictionLayer(task, )
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()
        self.linear_model1 = torch.nn.Linear(len(linear_feature_columns),1, bias = True) 
        self.add_regularization_weight(self.linear_model1.parameters(), l2=l2_reg_linear)
        self.to(device)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, X):
        logit1 = self.linear_model1(X) # bs * 1
        # logit2 = self.linear_model2(X)
        # logits = torch.cat([logit1, logit2], dim = 1)
        # y_pred = self.softmax(logits)
        # y_pred = torch.sigmoid(logit+self.bias)
        return torch.sigmoid(logit1) # y_pred[:,0]

    def fit(self, x=None, y=None,  batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, user_id = None, save_models = True, ref_metric = None,
            weight_scale = 1, need_weight_value = 0, use_classifier_stage = False, classifier_stage = 250):
        
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(x.values.astype(float)),
            torch.from_numpy(y.astype(float)))
        
        # except:
        #     train_tensor_data = Data.TensorDataset(
        #     torch.from_numpy(x.astype(float)),
        #     torch.from_numpy(y.astype(float)))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size, num_workers = 4)  

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        ref_metric = "val_auc"
        stopers = myearlystopping(refer_metric = ref_metric, mode = tune_mode, performance_satisfied_stop_condition=130)
        report_dict = {}
        # stoper = my_early_stoper(stop_condition = 8,refer_metric='val_aver_auc')
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            if use_classifier_stage and epoch >= classifier_stage:
                break
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                        y = y_train.to(self.device).float()
                        y_pred = model(x).squeeze()
                        optim.zero_grad()
                        weight = torch.ones_like(y_pred)
                        weight[y.squeeze() == need_weight_value] *= weight_scale # reweight for female samples
                        loss = loss_func(y_pred, y.squeeze(),weight, reduction='mean')
                        reg_loss = self.get_regularization_loss()
                        total_loss = loss + reg_loss + self.aux_loss
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if(name == 'uauc' or name == 'gauc'):
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                print('train finished')

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size,user_id)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                
                for name in self.metrics:
                    if(name == 'gauc' or name == 'uauc'):
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in epoch_logs:
                        if('val' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])
                print(eval_str)
            
            for name in self.metrics:
                if ('val_' + name + '_top') in list(report_dict.keys()):
                    if name in ['auc', 'acc', 'gauc']:
                        if epoch_logs['val_' + name] > report_dict['val_' + name + '_top']:
                            report_dict['val_' + name + '_top'] = epoch_logs['val_' + name]
                    else:
                        if epoch_logs['val_' + name] < report_dict['val_' + name + '_top']:
                            report_dict['val_' + name + '_top'] = epoch_logs['val_' + name]
                else:
                    report_dict['val_' + name + '_top'] = epoch_logs['val_' + name]
                report_dict['val_' + name] = epoch_logs['val_' + name]
                report_dict[name] = epoch_logs[name]
            tune.report(**report_dict, training_iteraction = epoch) # train.report(**report_dict, training_iteraction=epoch)

            need_saving = stopers.update_and_isbest(epoch_logs, epoch)
            need_stopping = stopers.is_stop()
            # if self.trial_name is not None:
            if not use_classifier_stage:
                mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name, self.trial_name)
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_result.pt') 
            else:
                mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name, self.trial_name + str(classifier_stage))
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name
                    , self.trial_name + str(classifier_stage), 'best_result.pt')
            #     mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name)
            #     mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
            #         'best_result.pt') 
            if not os.path.exists(mycheckpoint_dir):
                os.makedirs(mycheckpoint_dir)
            if not os.path.exists(mycheckpoint_path):
                best_result = 0
            else:
                loop_time = time.time()
                while True:
                    if time.time()-loop_time >=100:
                        break
                    try:
                        best_result = torch.load(mycheckpoint_path)['result']
                        break
                    except:
                        time.sleep(0.01)
            if need_saving and save_models and best_result <= epoch_logs[ref_metric]:
                best_result = epoch_logs[ref_metric]
                # save best results
                # if self.trial_name is not None:
                if not use_classifier_stage:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_result.pt')
                else:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name
                    , self.trial_name + str(classifier_stage), 'best_result.pt')
                #     mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
                #     'best_result.pt') 
                mycheckpoint = {
                    'result':epoch_logs[ref_metric],
                    'model_config':self.model_config_name
                }
                loop_time = time.time()
                while True:
                    if time.time()-loop_time >=100:
                        break
                    try:
                        torch.save(mycheckpoint,mycheckpoint_path)
                        break
                    except:
                        time.sleep(0.01)
                # save the model
                mycheckpoint = {
                    'epoch':epoch,
                    'model':self
                }
                if not use_classifier_stage:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_model.pt') # use folder name to distinguish val_env
                else:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name
                    , self.trial_name + str(classifier_stage), 'best_model.pt')
                loop_time = time.time()
                while True:
                    if time.time()-loop_time >=100:
                        print('timeout...break')
                        break
                    try:
                        torch.save(mycheckpoint,mycheckpoint_path)
                        break
                    except:
                        time.sleep(0.01)
            if need_stopping:
                print("early stop.......")
                break
        return self.history


    def evaluate(self, x, y, batch_size=256,user_id = None):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        if isinstance(y, pd.DataFrame):
            y = y.values
        if isinstance(x, pd.DataFrame):
            x = x.values

        pred_ans = self.predict(x, batch_size)
        #y = y[:int(y.shape[0]/batch_size)*batch_size] #drop last
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            if(name == 'gauc' or name == 'uauc'):
                eval_result[name] = metric_fun(y, pred_ans,user_id)
            else:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result
        
    def predict(self, x, batch_size=256):
        if isinstance(x, pd.DataFrame):
            x = x.values
        model = self.eval()
        tensor_data = Data.TensorDataset(
            torch.from_numpy(x))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)
        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x_ = x_test[0].to(self.device).float()
                y_pred = model(x_)
                y_pred = y_pred.cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")



class MySVM(MybaseModel):
    # used for predict sensitive attributes
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, init_std=0.0001, seed=1024,
                  task='binary', device='cpu', gpus=None):
        nn.Module.__init__(self)
        self.dnn_feature_columns = dnn_feature_columns
        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")
        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.regularization_weight = []
        self.out = PredictionLayer(task, )
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()
        self.linear_model1 = torch.nn.Linear(len(linear_feature_columns),1, bias = True) 
        self.add_regularization_weight(self.linear_model1.parameters(), l2=l2_reg_linear)
        self.to(device)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, X):
        logit1 = self.linear_model1(X) # bs * 1
        # logit2 = self.linear_model2(X)
        # logits = torch.cat([logit1, logit2], dim = 1)
        # y_pred = self.softmax(logits)
        # y_pred = torch.sigmoid(logit+self.bias)
        return logit1 # torch.sigmoid(logit1) # y_pred[:,0]

    def hinge_loss(self, outputs, targets):
        losses = 1 - targets * outputs.squeeze()
        losses = torch.clamp(losses, min=0)
        loss = torch.mean(losses)
        return loss

    def fit(self, x=None, y=None,  batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, user_id = None, save_models = True, ref_metric = None,
            weight_scale = 1, need_weight_value = 0, use_classifier_stage = False, classifier_stage = 250):
        
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(x.values.astype(float)),
            torch.from_numpy(y.astype(float)))
        
        # except:
        #     train_tensor_data = Data.TensorDataset(
        #     torch.from_numpy(x.astype(float)),
        #     torch.from_numpy(y.astype(float)))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size, num_workers = 4)  

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        ref_metric = "val_auc"
        stopers = myearlystopping(refer_metric = ref_metric, mode = tune_mode, performance_satisfied_stop_condition=130)
        report_dict = {}
        # stoper = my_early_stoper(stop_condition = 8,refer_metric='val_aver_auc')
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            if use_classifier_stage and epoch >= classifier_stage:
                break
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                        y = y_train.to(self.device).float()
                        y_pred = model(x).squeeze()
                        optim.zero_grad()
                        # weight = torch.ones_like(y_pred)
                        # weight[y.squeeze() == need_weight_value] *= weight_scale # reweight for female samples
                        
                        
                        # loss = loss_func(y_pred, y.squeeze(),weight, reduction='mean')
                        loss = self.hinge_loss(y_pred, y.squeeze())
                        reg_loss = self.get_regularization_loss()
                        total_loss = loss + reg_loss + self.aux_loss
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if(name == 'uauc' or name == 'gauc'):
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                print('train finished')

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size,user_id)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                
                for name in self.metrics:
                    if(name == 'gauc' or name == 'uauc'):
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in epoch_logs:
                        if('val' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])
                print(eval_str)
            
            for name in self.metrics:
                if ('val_' + name + '_top') in list(report_dict.keys()):
                    if name in ['auc', 'acc', 'gauc']:
                        if epoch_logs['val_' + name] > report_dict['val_' + name + '_top']:
                            report_dict['val_' + name + '_top'] = epoch_logs['val_' + name]
                    else:
                        if epoch_logs['val_' + name] < report_dict['val_' + name + '_top']:
                            report_dict['val_' + name + '_top'] = epoch_logs['val_' + name]
                else:
                    report_dict['val_' + name + '_top'] = epoch_logs['val_' + name]
                report_dict['val_' + name] = epoch_logs['val_' + name]
                report_dict[name] = epoch_logs[name]
            tune.report(**report_dict, training_iteraction = epoch) # train.report(**report_dict, training_iteraction=epoch)

            need_saving = stopers.update_and_isbest(epoch_logs, epoch)
            need_stopping = stopers.is_stop()
            # if self.trial_name is not None:
            if not use_classifier_stage:
                mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name, self.trial_name)
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_result.pt') 
            else:
                mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name, self.trial_name + str(classifier_stage))
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name
                    , self.trial_name + str(classifier_stage), 'best_result.pt')
            #     mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name)
            #     mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
            #         'best_result.pt') 
            if not os.path.exists(mycheckpoint_dir):
                os.makedirs(mycheckpoint_dir)
            if not os.path.exists(mycheckpoint_path):
                best_result = 0
            else:
                loop_time = time.time()
                while True:
                    if time.time()-loop_time >=100:
                        break
                    try:
                        best_result = torch.load(mycheckpoint_path)['result']
                        break
                    except:
                        time.sleep(0.01)
            if need_saving and save_models and best_result <= epoch_logs[ref_metric]:
                best_result = epoch_logs[ref_metric]
                # save best results
                # if self.trial_name is not None:
                if not use_classifier_stage:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_result.pt')
                else:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name
                    , self.trial_name + str(classifier_stage), 'best_result.pt')
                #     mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
                #     'best_result.pt') 
                mycheckpoint = {
                    'result':epoch_logs[ref_metric],
                    'model_config':self.model_config_name
                }
                loop_time = time.time()
                while True:
                    if time.time()-loop_time >=100:
                        break
                    try:
                        torch.save(mycheckpoint,mycheckpoint_path)
                        break
                    except:
                        time.sleep(0.01)
                # save the model
                mycheckpoint = {
                    'epoch':epoch,
                    'model':self
                }
                if not use_classifier_stage:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_model.pt') # use folder name to distinguish val_env
                else:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name
                    , self.trial_name + str(classifier_stage), 'best_model.pt')
                loop_time = time.time()
                while True:
                    if time.time()-loop_time >=100:
                        print('timeout...break')
                        break
                    try:
                        torch.save(mycheckpoint,mycheckpoint_path)
                        break
                    except:
                        time.sleep(0.01)
            if need_stopping:
                print("early stop.......")
                break
        return self.history


    def evaluate(self, x, y, batch_size=256,user_id = None):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        if isinstance(y, pd.DataFrame):
            y = y.values
        if isinstance(x, pd.DataFrame):
            x = x.values

        pred_ans = self.predict(x, batch_size)
        #y = y[:int(y.shape[0]/batch_size)*batch_size] #drop last
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            if(name == 'gauc' or name == 'uauc'):
                eval_result[name] = metric_fun(y, pred_ans,user_id)
            else:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result
        
    def predict(self, x, batch_size=256):
        if isinstance(x, pd.DataFrame):
            x = x.values
        model = self.eval()
        tensor_data = Data.TensorDataset(
            torch.from_numpy(x))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)
        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x_ = x_test[0].to(self.device).float()
                y_pred = torch.round(torch.sigmoid(model(x_)))
                y_pred = y_pred.cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")



class UGF(MybaseModel):
    def __init__(self, data_loader, k, eval_metric_list, fairness_metric='f1',
                 epsilon=0.05, model_name='', group_name='', train_mode = 'noisy'):
        """
        Train fairness model
        :param data_loader: Dataloader object
        :param k: k for top-K number of items to be selected from the entire list
        :param eval_metric_list: a list contains all the metrics to report
        :param fairness_metric: a string, the metric used for fairness constraint, default='f1'
        :param epsilon: the upper bound for the difference between two groups scores
        :param logger: logger for logging info
        """
        self.data_loader = data_loader
        self.k = k
        self.eval_metric_list = eval_metric_list
        self.fairness_metric = fairness_metric
        self.epsilon = epsilon
        self.model_name = model_name
        self.group_name = group_name
        self.train_mode = train_mode

    @staticmethod
    def _check_df_format(df):
        """
        check if the input dataframe contains all the necessary columns
        :return: None
        """
        expected_columns = ['user_id', 'item_id', 'score', 'label', 'q']
        for c in expected_columns:
            if c not in df.columns:
                raise KeyError('Missing column ' + c)

    @staticmethod
    def _build_fairness_optimizer(group_df_list, k, metric, name='UGF'):
        """
        Use Gurobi to build faireness optimizer
        :param group_df_list: a list contains dataframes from two groups
        :param k: an integer for the length of top-K list
        :param metric: the metric string for fairness constraint. e.g. 'f1', 'recall', 'precision'
        :param name: a string which is the name of this optimizer
        :return: the Gurobi model, a list of Qi*Si for obj function, a list of two group metric
        """
        try:
            # Create a new model
            m = gp.Model(name)
            var_score_list = []                 # store the Qi * Si
            metric_list = []                    # store the averaged metric scores for two groups
            # Create variables
           
            for group_index, df in enumerate(group_df_list):
                df_group = df.groupby('user_id')
                tmp_metric_list = []            # store the metric calculation for each user in current user group
                tmp_var_score_list = []         # store var * score for object function use
                for uid, group in df_group:
                    tmp_var_list = []           # store variables for sum(Qi) == k use
                    tmp_var_label_list = []     # store var * label for recall calculation use
                    score_list = group['score'].tolist()
                    label_list = group['label'].tolist()
                    item_list = group['item_id'].tolist()

                    for i in range(len(item_list)):
                        var_name = str(uid) + '_' + str(item_list[i])   # variable name is "uid_iid"
                        v = m.addVar(vtype=GRB.BINARY, name=var_name)
                        tmp_var_list.append(v)
                        if group_index == 1:
                            tmp_var_score_list.append(score_list[i] * v) #
                        else:
                            tmp_var_score_list.append(v)
                        tmp_var_label_list.append(label_list[i] * v)
                    # Add first constraint: Sum(Qi)==k
                    m.addConstr(gp.quicksum(tmp_var_list) == k)
                    # calculate the corresponding measures
                    if group_index != 2:
                        if group['label'].sum() == 0:
                            continue
                        if metric == 'recall':
                            tmp_metric_list.append(gp.quicksum(tmp_var_label_list) / group['label'].sum())
                        elif metric == 'precision':
                            tmp_metric_list.append(gp.quicksum(tmp_var_label_list) / k)
                        elif metric == 'f1':
                            f1 = 2 * gp.quicksum(tmp_var_label_list) / (group['label'].sum() + k)
                            tmp_metric_list.append(f1)
                        elif metric == 'ndcg':
                            pass
                            # sorted_scores = sorted(tmp_var_score_list, reverse = True)
                            # weight = [1 / math.log2(i + 2) for i in range(len(tmp_var_score_list))]
                            # dcg = gp.quicksum(weight[i] * sorted_scores [i] for i in range(len(tmp_var_score_list)))
                            # tmp_metric_list.append(dcg)
                        else:
                            raise ValueError('Unknown metric for optimizer building.')
                if group_index != 2:
                    metric_list.append(gp.quicksum(tmp_metric_list) / len(tmp_metric_list))
                var_score_list.extend(tmp_var_score_list)
            m.update()
            return m, var_score_list, metric_list

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
        except AttributeError:
            print('Encountered an attribute error')
        pass

    @staticmethod
    def _format_result(model, df):
        """
        format the gurobi results to dataframe.
        :param model: optimized gurobi model
        :param df: the pandas dataframe to add the optimized results into
        :return: None
        """
        for v in model.getVars():
            v_s = v.varName.split('_')
            uid = int(v_s[0])
            iid = int(v_s[1])
            df.loc[(df['user_id'] == uid) & (df['item_id'] == iid), 'q'] = int(v.x)

    def _print_metrics(self, df, metrics, message='metric scores'):
        """
        Print out evaluation scores
        :param df: the dataframe contains the data for evaluation
        :param metrics: a list, contains the metrics to report
        :param message: a string, for print message
        :return: None
        """
        results = evaluation_methods(df, metrics=metrics)
        r_string = ""
        for i in range(len(metrics)):
            r_string = r_string + metrics[i] + "=" + '{:.4f}'.format(results[i]) + " "
        print(message + ": " + r_string)
        # write the message into the log file
        return results

    def train(self):
        """
        Train fairness model
        """
        # model = read('gurobi_model.mps')

        # Prepare data
        all_df = self.data_loader.rank_df.copy(deep=True)    # the dataframe with entire test data
        self._check_df_format(all_df)   # check the dataframe format
        group_df_list_test = [self.data_loader.g1_df.copy(deep=True),
                         self.data_loader.g2_df.copy(deep=True)]  # group 1 (active), group 2 (inactive)
        if self.train_mode == 'fake' or self.train_mode == 'partial':
            group_df_list_train = [self.data_loader.g1obe_df.copy(deep=True),
                         self.data_loader.g2obe_df.copy(deep=True), self.data_loader.g3obe_df.copy(deep=True)]
        else:
            group_df_list_train = [self.data_loader.g1obe_df.copy(deep=True),
                         self.data_loader.g2obe_df.copy(deep=True)]

        r0 = self._print_metrics(all_df, self.eval_metric_list, 'Before optimization overall scores           ')
        te00 = self._print_metrics(group_df_list_test[0], self.eval_metric_list, 'Before optimization group 0 (female) scores  ')
        te10 = self._print_metrics(group_df_list_test[1], self.eval_metric_list, 'Before optimization group 1 (male) scores')
        tr00 = self._print_metrics(group_df_list_train[0], self.eval_metric_list, 'Before optimization group 0 (female) scores  training ')
        tr10 = self._print_metrics(group_df_list_train[1], self.eval_metric_list, 'Before optimization group 1 (male) scores  training')
        # build optimizer
        m, var_score_list, metric_list = \
            self._build_fairness_optimizer(group_df_list_train, self.k, metric=self.fairness_metric, name='UGF_f1')
    
        # |group_1_recall - group_2_recall| <= epsilon
        m.addConstr(metric_list[1] - metric_list[0] <= self.epsilon)
        m.addConstr(metric_list[0] - metric_list[1] <= self.epsilon)
        

        # Set objective function
        m.setObjective(gp.quicksum(var_score_list), GRB.MAXIMIZE)

        # Optimize model
        m.optimize()

        # m.write('gurobi_model.mps')

        # Format the output results and update q column of the dataframe
        cols = ['user_id', 'item_id', 'score', 'label', 'true_group', 'fake_group', 'noisy_group']
        self._format_result(m, all_df)
        group_df_list_train[0].drop(columns=['q'], inplace=True)
        group_df_list_train[0] = pd.merge(group_df_list_train[0], all_df, on = cols, how='left')
        group_df_list_train[1].drop(columns=['q'], inplace=True)
        group_df_list_train[1] = pd.merge(group_df_list_train[1], all_df, on = cols, how='left')
        group_df_list_test[0].drop(columns=['q'], inplace=True)
        group_df_list_test[0] = pd.merge(group_df_list_test[0], all_df, on = cols, how='left')
        group_df_list_test[1].drop(columns=['q'], inplace=True)
        group_df_list_test[1] = pd.merge(group_df_list_test[1], all_df, on = cols, how='left')
        r1 = self._print_metrics(all_df, self.eval_metric_list, 'After optimization overall metric scores     ')
        te01 = self._print_metrics(group_df_list_test[0], self.eval_metric_list, 'After optimization group 0 (female) scores   ')
        te11 = self._print_metrics(group_df_list_test[1], self.eval_metric_list, 'After optimization group 1 (male) scores ')
        tr01 = self._print_metrics(group_df_list_train[0], self.eval_metric_list, 'After optimization group 0 (female) scores  training ')
        tr11 = self._print_metrics(group_df_list_train[1], self.eval_metric_list, 'After optimization group 1 (male) scores  training')
        print(r1[0] - r0[0], 'overall improvement after optimization')
        print(te01[0] - te00[0], 'group0 test improvement after optimization')
        print(te11[0] - te10[0], 'group1 test improvement after optimization')
        print(tr01[0] - tr00[0], 'group0 train improvement after optimization')
        print(tr11[0] - tr10[0], 'group1 train improvement after optimization')
        print(-abs(te11[0] - te01[0]) + abs(te10[0] - te00[0]), 'test fairness_improvement')
        print(-abs(tr11[0] - tr01[0]) + abs(tr10[0] - tr00[0]), 'train fairness_improvement')
        return r1[0] - r0[0], -abs(te11[0] - te01[0]) + abs(te10[0] - te00[0]), -abs(tr11[0] - tr01[0]) + abs(tr10[0] - tr00[0])


UtilityConstrainedEarlyStopping = myearlystopping
