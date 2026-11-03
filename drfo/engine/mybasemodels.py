# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../'))

workspace = '/data/shith/drfo-release/workspace'
from itertools import chain
import random
from typing import Dict
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch._C import device
from torch.nn import init
from torch.nn.modules import linear
from torch.nn.parameter import Parameter
from tqdm import tqdm
from sklearn.metrics import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names,combined_dnn_input
from deepctr_torch.layers import DNN
from deepctr_torch.models.deepfm import *
# from basemodel_ours import *
from deepctr_torch.callbacks import EarlyStopping
import time
from torch.utils.data import DataLoader
import torch.utils.data as Data
from tensorflow.python.keras.callbacks import CallbackList
from deepctr_torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from sklearn.metrics import roc_auc_score, mean_absolute_error
from deepctr_torch.layers.utils import slice_arrays
import torch.nn.functional as F
from collections import defaultdict
from deepctr_torch.layers import PredictionLayer
from deepctr_torch.layers.utils import slice_arrays
from deepctr_torch.callbacks import History
# from layers_ours import *
from torch.utils.tensorboard import SummaryWriter
from ray import tune,train
import functools
import math
# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
class my_early_stoper(object):
    def __init__(self,refer_metric='val_auc',stop_condition = 50, performance_satisfied_stop_condition = 50, mode = 'min'):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric
        self.performance_not_satisfied = 0
        self.performance_satisfied_stop_condition = performance_satisfied_stop_condition
        self.mode = mode

    def update_and_isbest(self, eval_metric, epoch, performance_satisfied = True):
        if self.init_flag:
            self.best_epoch = epoch
            if performance_satisfied:
                self.init_flag = False
                self.best_eval_result = eval_metric
                self.performance_not_satisfied = 0
            else:
                self.performance_not_satisfied += 1
            return performance_satisfied
        elif self.mode == 'min':
            if eval_metric[self.refer_metric] < self.best_eval_result[self.refer_metric] : # update the best results
                if performance_satisfied:
                    self.best_eval_result = eval_metric
                    self.not_change = 0
                    self.performance_not_satisfied = 0
                    self.best_epoch = epoch
                    return True              # best
                else:
                    self.performance_not_satisfied += 1
                    self.not_change = 0
                    return False
            else:                        # add one to the maker for not_change information 
                self.not_change += 1     # not best
                if performance_satisfied:
                    self.performance_not_satisfied = 0
                else:
                    self.performance_not_satisfied += 1
                return False
        elif self.mode == 'max':
            if eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric] : # update the best results
                if performance_satisfied:
                    self.best_eval_result = eval_metric
                    self.not_change = 0
                    self.performance_not_satisfied = 0
                    self.best_epoch = epoch
                    return True              # best
                else:
                    self.performance_not_satisfied += 1
                    self.not_change = 0
                    return False
            else:                        # add one to the maker for not_change information 
                self.not_change += 1     # not best
                if performance_satisfied:
                    self.performance_not_satisfied = 0
                else:
                    self.performance_not_satisfied += 1
                return False

    def is_stop(self):
        if self.not_change > self.stop_condition:
            return True
        else:
            if self.performance_not_satisfied > self.performance_satisfied_stop_condition:
                return True
            else:
                return False

class EarlyStopping(object):
    def __init__(self,refer_metric='val_auc',stop_condition = 50, performance_satisfied_stop_condition = 50, mode = 'min',delta=1e-5):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric
        self.performance_not_satisfied = 0
        self.performance_satisfied_stop_condition = performance_satisfied_stop_condition
        self.mode = mode
        self.delta = delta

    def update_and_isbest(self, eval_metric, epoch,  performance_satisfied = True):
        if self.init_flag:
            self.best_epoch = epoch
            if performance_satisfied:
                self.init_flag = False
                self.best_eval_result = eval_metric
                self.performance_not_satisfied = 0
            else:
                self.performance_not_satisfied += 1
            return performance_satisfied
        
        elif self.mode == 'min':
            change = np.abs(eval_metric[self.refer_metric] - self.best_eval_result[self.refer_metric])
            if change > self.delta:
            # eval_metric[self.refer_metric] < self.best_eval_result[self.refer_metric] : # update the best results
                if performance_satisfied:
                    self.best_eval_result = eval_metric
                    self.not_change = 0
                    self.performance_not_satisfied = 0
                    self.best_epoch = epoch
                    return True              # best
                else:
                    self.performance_not_satisfied += 1
                    self.not_change = 0
                    return False
            else:                        # add one to the maker for not_change information 
                self.not_change += 1     # not best
                if performance_satisfied:
                    self.performance_not_satisfied = 0
                else:
                    self.performance_not_satisfied += 1
                return False
            
        elif self.mode == 'max':
            change = np.abs(eval_metric[self.refer_metric] - self.best_eval_result[self.refer_metric])
            if change > self.delta:
            # eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric] : # update the best results
                if performance_satisfied:
                    self.best_eval_result = eval_metric
                    self.not_change = 0
                    self.performance_not_satisfied = 0
                    self.best_epoch = epoch
                    return True              # best
                else:
                    self.performance_not_satisfied += 1
                    self.not_change = 0
                    return False
            else:                        # add one to the maker for not_change information 
                self.not_change += 1     # not best
                if performance_satisfied:
                    self.performance_not_satisfied = 0
                else:
                    self.performance_not_satisfied += 1
                return False

    def is_stop(self):
        if self.not_change > self.stop_condition:
            return True
        else:
            if self.performance_not_satisfied > self.performance_satisfied_stop_condition:
                return True
            else:
                return False


class myearlystopping(object):
    def __init__(self,refer_metric='val_auc', performance_satisfied_stop_condition = 50, mode = 'min'):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.init_flag = True
        self.refer_metric = refer_metric
        self.performance_not_satisfied = 0
        self.performance_satisfied_stop_condition = performance_satisfied_stop_condition
        self.mode = mode

    def update_and_isbest(self, eval_metric, epoch,  performance_satisfied = True):
        if self.init_flag:
            self.best_epoch = epoch
            if performance_satisfied:
                self.init_flag = False
                self.best_eval_result = eval_metric
                self.performance_not_satisfied = 0
            else:
                self.performance_not_satisfied += 1
            return performance_satisfied
        
        else:
            if performance_satisfied:
                self.best_eval_result = eval_metric
                self.performance_not_satisfied = 0
                self.best_epoch = epoch
                return True              # best
            else:
                self.performance_not_satisfied += 1
                return False
            
    def is_stop(self):

        if self.performance_not_satisfied > self.performance_satisfied_stop_condition:
            return True
        else:
            return False


                
class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(sparse_embedding_list[0].device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


def cal_acc(y_true, y_pred, threshold = 0.5):
    return accuracy_score(y_true, np.where(y_pred > threshold, 1, 0))


def cal_ndcg(list1, list2, at = 10, explicit = True, thre = 3.5):
    
    rank = np.argsort(-np.array(list1),).tolist()
    dcg = 0
    if explicit:
        for i in range(at):
            if list2[rank[i]] >= thre:
                dcg += 1 / math.log2(i + 2)
        idcg = 0
        rank = np.argsort(-np.array(list2)).tolist()
        for i in range(at):
            if list2[rank[i]] >= thre:
                idcg += 1 / math.log2(i + 2)
        ndcg = dcg / idcg if idcg != 0 else 0
    else:
        for i in range(at):
            if list2[rank[i]] == 1:
                dcg += 1 / math.log2(i + 2)
        
        idcg = 0
        list2 = list2[:at]
        rank = np.argsort(-np.array(list2)).tolist()
        for i in range(at):
            if list2[rank[i]] == 1:
                idcg += 1 / math.log2(i + 2)
        ndcg = dcg / idcg if idcg != 0 else 0
    return ndcg

def cal_precision(list1, list2, at = 10, explicit = True, thre = 3.5):
    rank = np.argsort(-np.array(list1),).tolist()
    s = 0
    if explicit:
        for i in range(at):
            if list2[rank[i]] >= thre:
                s += 1
    else:
        for i in range(at):
            if list2[rank[i]] == 1:
                s += 1
    return s / at

def cal_recall(list1, list2, at = 10,  explicit = True, thre = 3.5):
    rank = np.argsort(-np.array(list1)).tolist()
    s = 0
    if explicit:
        for i in range(at):
            if list2[rank[i]] >= thre:
                s += 1
        n = 0
        for i in list2:
            if i >= thre:
                n += 1
    else:
        for i in range(at):
            if list2[rank[i]] == 1:
                s += 1
        n = 0
        for i in list2:
            if i == 1:
                n += 1
    return s / n



def rooted_mean_squared_error(x):
    y_pred = np.array(x['predict_ratings'])
    y_true = np.array(x['ratings'])
    return np.sqrt(mean_squared_error(y_true, y_pred))

def cal_mean_mae(x, ):
    maes = []
    data_groups = x.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        list2 = list(df['ratings'].tolist())
        maes.append(mean_absolute_error(list2, list1))
    # for idx in x.index:
    #     list1 = x.at[idx, 'predict_rating_list']
    #     list2 = x.at[idx, 'rating_list']
    #     maes.append(mean_absolute_error(list2, list1))
    return sum(maes) / len(maes)

def cal_group_mae(x, sensitive_value = 0):
    # x: dataframe, cols = (user_id, item_list, rating_list, predict_rating_list, sensitive_attribute)
    recalls = []
    x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
    data_groups = x_sensitive.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        list2 = list(df['ratings'].tolist())
        recalls.append(mean_absolute_error(list2, list1))
    return sum(recalls) / len(recalls)

def cal_mean_uabs(x):
    uabs = 0
    data_groupby_items = x.groupby('item_id')
    n = 0
    for item, df in data_groupby_items:
        # uabs += np.abs(np.mean(df['ratings']) - np.mean(df['predict_ratings']))
        df_0 = df.loc[df['sensitive_attribute'] == 0,:]
        df_1 = df.loc[df['sensitive_attribute'] == 1,:]
        uabs_iter = np.abs(np.abs(np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
                - np.abs(np.mean(df_1['ratings']) - np.mean(df_1['predict_ratings'])))
        if pd.isna(uabs_iter):
            continue
        else:
            uabs += uabs_iter
            n += 1 
        # np.abs(np.abs(np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
    uabs /= n
    return uabs

def cal_mean_uabs_true(x):
    # used to evaluate validation data when training to watch the discripancy between uabs calculated by true labels and noisy laebls
    uabs = 0
    data_groupby_items = x.groupby('item_id')
    n = 0
    for item, df in data_groupby_items:
        # n += 1 
        # uabs += np.abs(np.mean(df['ratings']) - np.mean(df['predict_ratings']))
        df_0 = df.loc[df['sensitive_attribute_true'] == 0,:]
        df_1 = df.loc[df['sensitive_attribute_true'] == 1,:]
        uabs_iter = np.abs(np.abs(np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
                - np.abs(np.mean(df_1['ratings']) - np.mean(df_1['predict_ratings'])))
        if pd.isna(uabs_iter):
            continue
        else:
            uabs += uabs_iter
            n += 1 
        # uabs += np.abs(np.abs(np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
    uabs /= n
    return uabs

def cal_mean_uabs_v1(x):
    # not condition on items
    # uabs = 0
    df_0 = x.loc[x['sensitive_attribute'] == 0, :]
    df_1 = x.loc[x['sensitive_attribute'] == 1, :]
    uabs = np.abs(np.abs(np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
                - np.abs(np.mean(df_1['ratings']) - np.mean(df_1['predict_ratings'])))
    # data_groupby_items = x.groupby('item_id')
    # n = 0
    # for item, df in data_groupby_items:
        
    #     # uabs += np.abs(np.mean(df['ratings']) - np.mean(df['predict_ratings']))
    #     df_0 = df.loc[df['sensitive_attribute'] == 0,:]
    #     df_1 = df.loc[df['sensitive_attribute'] == 1,:]
    #     uabs_iter = np.abs(np.abs(np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
    #     if pd.isna(uabs_iter):
    #         continue
    #         uabs += uabs_iter
    #         n += 1 
    #     # np.abs(np.abs(np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
    #             # - np.abs(np.mean(df_1['ratings']) - np.mean(df_1['predict_ratings'])))
    # uabs /= n
    return uabs

def cal_mean_uabs_v1_true(x):
    # not condition on items
    # uabs = 0
    df_0 = x.loc[x['sensitive_attribute_true'] == 0, :]
    df_1 = x.loc[x['sensitive_attribute_true'] == 1, :]
    uabs = np.abs(np.abs(np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
                - np.abs(np.mean(df_1['ratings']) - np.mean(df_1['predict_ratings'])))
    return uabs

def cal_mean_uval(x):
    uabs = 0
    data_groupby_items = x.groupby('item_id')
    n = 0
    for item, df in data_groupby_items:
        n += 1 
        # uabs += np.abs(np.mean(df['ratings']) - np.mean(df['predict_ratings']))
        df_0 = df.loc[df['sensitive_attribute'] == 0,:]
        df_1 = df.loc[df['sensitive_attribute'] == 1,:]
        uabs += np.abs((np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
                - (np.mean(df_1['ratings']) - np.mean(df_1['predict_ratings'])))
    uabs /= n
    return uabs

def cal_mean_uval_true(x):
    # used to evaluate validation data when training to watch the discripancy between uabs calculated by true labels and noisy laebls
    uabs = 0
    data_groupby_items = x.groupby('item_id')
    n = 0
    for item, df in data_groupby_items:
        n += 1 
        # uabs += np.abs(np.mean(df['ratings']) - np.mean(df['predict_ratings']))
        df_0 = df.loc[df['sensitive_attribute_true'] == 0,:]
        df_1 = df.loc[df['sensitive_attribute_true'] == 1,:]
        uabs += np.abs((np.mean(df_0['ratings']) - np.mean(df_0['predict_ratings'])) \
                - (np.mean(df_1['ratings']) - np.mean(df_1['predict_ratings'])))
    uabs /= n
    return uabs

def cal_group_uabs(x, sensitive_value = 0):
    uabs = 0
    data_groupby_items = x.groupby('item_id')
    n = 0
    for item, df in data_groupby_items:
        n += 1 
        df = df.loc[df['sensitive_attribute'] == sensitive_value, :]
        uabs += np.abs(np.mean(df['ratings']) - np.mean(df['predict_ratings']))
    uabs /= n
    return uabs

def cal_group_uabs_v1(x, sensitive_value = 0):
    uabs = 0
    # data_groupby_items = x.groupby('item_id')
    # n = 0
    # for item, df in data_groupby_items:
        # n += 1 
    df = x.loc[x['sensitive_attribute'] == sensitive_value, :]
    uabs += np.abs(np.mean(df['ratings']) - np.mean(df['predict_ratings']))
    # uabs /= n
    return uabs

def cal_group_uabs_v1_correct(x, sensitive_value = 0):
    uabs = 2.5
    df = x.loc[x['sensitive_attribute'] == x['sensitive_attribute_true'], :]
    try:
        df = df.loc[df['sensitive_attribute'] == sensitive_value, :]
        uabs = np.abs(np.mean(df['ratings']) - np.mean(df['predict_ratings']))
    except:
        uabs = 2.5
    return uabs

def cal_group_uabs_v1_false(x, sensitive_value = 0):
    uabs = 2.5
    # calculate true group uabs v1 for those whose predicted sensitive labels are false
    df = x.loc[x['sensitive_attribute'] != x['sensitive_attribute_true'], :]
    try:
        df = df.loc[df['sensitive_attribute_true'] == sensitive_value, :]
        uabs = np.abs(np.mean(df['ratings']) - np.mean(df['predict_ratings']))
    except:
        uabs = 2.5
    return uabs

def cal_group_rating_correct(x, sensitive_value = 0):
    uabs = 1
    df = x.loc[x['sensitive_attribute'] == x['sensitive_attribute_true'], :]
    try:
        df = df.loc[df['sensitive_attribute'] == sensitive_value, :]
        uabs = np.mean(df['predict_ratings'])
    except:
        uabs = 1
    return uabs

def cal_group_rating_false(x, sensitive_value = 0):
    uabs = 1
    # calculate true group uabs v1 for those whose predicted sensitive labels are false
    df = x.loc[x['sensitive_attribute'] != x['sensitive_attribute_true'], :]
    try:
        df = df.loc[df['sensitive_attribute_true'] == sensitive_value, :]
        uabs = np.mean(df['predict_ratings'])
    except:
        uabs = 1
    return uabs

def cal_group_rating_correct_know(x, sensitive_value = 0):
    uabs = 1
    df = x.loc[x['sensitive_attribute_noisy'] == x['sensitive_attribute_true'], :]
    try:
        df = df.loc[df['sensitive_attribute'] == sensitive_value, :]
        uabs = np.mean(df['predict_ratings'])
    except:
        uabs = 1
    return uabs

def cal_group_rating_correct_unknow(x, sensitive_value = 0):
    uabs = 1
    # calculate true group uabs v1 for those whose predicted sensitive labels are false
    df = x.loc[x['sensitive_attribute_noisy'] == x['sensitive_attribute_true'], :]
    try:
        df = df.loc[df['sensitive_attribute_true'] == sensitive_value, :]
        df = df.loc[df['sensitive_attribute'] != sensitive_value, :]
        uabs = np.mean(df['predict_ratings'])
    except:
        uabs = 1
    return uabs


def cal_group_rating_know(x, sensitive_value = 0):
    uabs = 1
    df = x.loc[x['sensitive_attribute'] != 1000, :]
    try:
        df = df.loc[df['sensitive_attribute_true'] == sensitive_value, :]
        uabs = np.mean(df['predict_ratings'])
    except:
        uabs = 1
    return uabs

def cal_group_rating_unknow(x, sensitive_value = 0):
    uabs = 1
    # calculate true group uabs v1 for those whose predicted sensitive labels are false
    df = x.loc[x['sensitive_attribute'] == 1000, :]
    try:
        df = df.loc[df['sensitive_attribute_true'] == sensitive_value, :]
        uabs = np.mean(df['predict_ratings'])
    except:
        uabs = 1
    return uabs


def cal_mean_recall(x, at = 10, explicit = True, thre = 3.5):
    # x: dataframe, cols = (user_id, item_list, rating_list, predict_rating_list, sensitive_attribute)
    recalls = []
    data_groups = x.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        list2 = list(df['ratings'].tolist())
        recalls.append(cal_recall(list1, list2, at, explicit, thre ))
    return sum(recalls) / len(recalls)

def cal_mean_precision(x, at = 10, explicit = True, thre = 3.5):
    # x: dataframe, cols = (user_id, item_list, rating_list, predict_rating_list, sensitive_attribute)
    precisions = []
    data_groups = x.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        list2 = list(df['ratings'].tolist())
        precisions.append(cal_precision(list1, list2, at, explicit, thre))
    return sum(precisions) / len(precisions)



def cal_mean_ndcg(x, at = 10,  explicit = True, thre = 3.5):
    # x: dataframe, cols = (user_id, item_list, rating_list, predict_rating_list, sensitive_attribute)
    ndcgs = []
    data_groups = x.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        list2 = list(df['ratings'].tolist())
        ndcgs.append(cal_ndcg(list1, list2, at, explicit, thre))
    return sum(ndcgs) / len(ndcgs)



def cal_average_rating(x):
    ratings = []
    data_groups = x.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        ratings.extend(list1)
    return sum(ratings) / len(ratings)


def cal_mean_dp_v1(x):
    dp = []
    all_ratings = []
    for sensitive_value in [0,1,1000]:    
        try: # no 1000 when noisy and true
            ratings = []
            x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
            data_groups = x_sensitive.groupby('user_id')
            for user, df in data_groups:
                list1 = list(df['predict_ratings'].tolist())
                ratings.extend(list1)
            all_ratings.extend(ratings)
            dp.append(sum(ratings) / len(ratings))
        except:
            pass
    aver_all = sum(all_ratings) / len(all_ratings)
    dp_v1 = abs(dp[0] - aver_all) + abs(dp[1] - aver_all)
    return dp_v1

def cal_mean_dp_v2(x):
    dp = []
    all_ratings = []
    for sensitive_value in [0,1,1000]:    
        try: # no 1000 when noisy and true
            ratings = []
            x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
            data_groups = x_sensitive.groupby('user_id')
            for user, df in data_groups:
                list1 = list(df['predict_ratings'].tolist())
                ratings.extend(list1)
            all_ratings.extend(ratings)
            dp.append(sum(ratings) / len(ratings))
        except:
            pass
    aver_all = sum(all_ratings) / len(all_ratings)
    try:
        dp_v2 = abs(dp[0] - aver_all) + abs(dp[1] - aver_all) +  abs(dp[2] - aver_all)
    except:
        dp_v2 = abs(dp[0] - aver_all) + abs(dp[1] - aver_all)
    return dp_v2


def cal_mean_dp_v3(x):
    try:
        dp = []
        all_ratings = []
        for sensitive_value in [0,1]:   
            ratings = []
            x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
            data_groups = x_sensitive.groupby('user_id')
            for user, df in data_groups:
                list1 = list(df['predict_ratings'].tolist())
                ratings.extend(list1)
            all_ratings.extend(ratings)
            dp.append(sum(ratings) / len(ratings))
        for sensitive_value in [1000]:
            # pass
            x = x.loc[x['sensitive_attribute'] == sensitive_value, :]
            for sensitive_value_noisy in [0,1]:   
                x_sensitive_noisy = x.loc[x['sensitive_attribute_noisy'] == sensitive_value_noisy, :]
                ratings = []
            # x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
                data_groups = x_sensitive_noisy.groupby('user_id')
                for user, df in data_groups:
                    list1 = list(df['predict_ratings'].tolist())
                    ratings.extend(list1)
                all_ratings.extend(ratings)
                dp.append(sum(ratings) / len(ratings))
        aver_all = sum(all_ratings) / len(all_ratings)
        dp_v3 = abs(dp[0] - aver_all) + abs(dp[1] - aver_all) +  abs(dp[2] - aver_all) + abs(dp[3] - aver_all)
    except:
        dp_v3 = cal_mean_dp_v2(x)
    return dp_v3


def cal_mean_dp_v4(x):
    try:
        dp = []
        all_ratings = []
        for sensitive_value in [0,1]:   
            ratings = []
            x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
            data_groups = x_sensitive.groupby('user_id')
            for user, df in data_groups:
                list1 = list(df['predict_ratings'].tolist())
                ratings.extend(list1)
            all_ratings.extend(ratings)
            dp.append(sum(ratings) / len(ratings))
        for sensitive_value in [1000]:
            # pass
            x = x.loc[x['sensitive_attribute'] == sensitive_value, :]
            for sensitive_value_noisy in [0,1]:   
                x_sensitive_noisy = x.loc[x['sensitive_attribute_noisy'] == sensitive_value_noisy, :]
                ratings = []
            # x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
                data_groups = x_sensitive_noisy.groupby('user_id')
                for user, df in data_groups:
                    list1 = list(df['predict_ratings'].tolist())
                    ratings.extend(list1)
                all_ratings.extend(ratings)
                dp.append(sum(ratings) / len(ratings))
        aver_all = sum(all_ratings) / len(all_ratings)
        dp_v4 = abs(dp[2] - dp[3])
    except:
        dp_v4 = cal_mean_dp_v2(x)
    return dp_v4


def cal_mean_dp(x):
    dp = []
    for sensitive_value in (0,1):    
        ratings = []
        x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
        data_groups = x_sensitive.groupby('user_id')
        for user, df in data_groups:
            list1 = list(df['predict_ratings'].tolist())
            ratings.extend(list1)
        dp.append(sum(ratings) / len(ratings))
    return abs(dp[0]-dp[1])

def cal_mean_dp_true(x):
    dp = []
    for sensitive_value in (0,1):    
        ratings = []
        x_sensitive = x.loc[x['sensitive_attribute_true'] == sensitive_value, :]
        data_groups = x_sensitive.groupby('user_id')
        for user, df in data_groups:
            list1 = list(df['predict_ratings'].tolist())
            ratings.extend(list1)
        dp.append(sum(ratings) / len(ratings))
    return abs(dp[0]-dp[1])

def cal_mean_dp_noisy(x):
    try:
        dp = []
        for sensitive_value in (0,1):    
            ratings = []
            x_sensitive = x.loc[x['sensitive_attribute_noisy'] == sensitive_value, :]
            data_groups = x_sensitive.groupby('user_id')
            for user, df in data_groups:
                list1 = list(df['predict_ratings'].tolist())
                ratings.extend(list1)
            dp.append(sum(ratings) / len(ratings))
        return abs(dp[0]-dp[1])
    except:
        return 1

def cal_mean_dp_fake(x):
    try:
        dp = []
        for sensitive_value in (0,1):    
            ratings = []
            x_sensitive = x.loc[x['sensitive_attribute_fake'] == sensitive_value, :]
            data_groups = x_sensitive.groupby('user_id')
            for user, df in data_groups:
                list1 = list(df['predict_ratings'].tolist())
                ratings.extend(list1)
            dp.append(sum(ratings) / len(ratings))
        return abs(dp[0]-dp[1])
    except:
        return 1
    
def cal_group_rating(x, sensitive_value = 0, explicit = True, thre = 3.5):
    ratings = []
    x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
    data_groups = x_sensitive.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        ratings.extend(list1)
    return sum(ratings) / len(ratings)

def cal_group_recall(x, at = 10, sensitive_value = 0, explicit = True, thre = 3.5):
    # x: dataframe, cols = (user_id, item_list, rating_list, predict_rating_list, sensitive_attribute)
    recalls = []
    x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
    data_groups = x_sensitive.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        list2 = list(df['ratings'].tolist())
        recalls.append(cal_recall(list1, list2, at, explicit, thre))
    return sum(recalls) / len(recalls)

def cal_group_precision(x, at = 10, sensitive_value = 0, explicit = True, thre = 3.5):
    # x: dataframe, cols = (user_id, item_list, rating_list, predict_rating_list, sensitive_attribute)
    precisions = []
    x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
    data_groups = x_sensitive.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        list2 = list(df['ratings'].tolist())
        precisions.append(cal_precision(list1, list2, at, explicit, thre))
    return sum(precisions) / len(precisions)

def cal_group_ndcg(x, at = 10, sensitive_value = 0, explicit = True, thre = 3.5):
    # x: dataframe, cols = (user_id, item_list, rating_list, predict_rating_list, sensitive_attribute)
    ndcgs = []
    x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
    data_groups = x_sensitive.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        list2 = list(df['ratings'].tolist())
        ndcgs.append(cal_ndcg(list1, list2, at, explicit, thre))
    return sum(ndcgs) / len(ndcgs)

class MybaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):

        super().__init__()
        #torch.manual_seed(seed)
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
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()
    
    def train_one_batch(self, x, y, optim, model, loss_func):
        y_pred = model(x).squeeze()
        optim.zero_grad()
        loss = loss_func(y_pred, y.squeeze(),reduction='mean')
        reg_loss = self.get_regularization_loss()
        total_loss = loss + reg_loss + self.aux_loss
        total_loss.backward()
        optim.step()
        
    def fit(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, 
            save_models = True, ref_metric = None, mode = 'min'):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`
        :param irm_optimize_params:None, 'inter_weight', 'inter_embedding','all' are legal
        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        """
        if isinstance(x, dict):
            x_temp = [x[feature] for feature in self.feature_index]
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
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
            if x[i].dtype == 'object':
                x[i] = x[i].astype(int)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        """
        train_tensor_data, val_x, val_y, do_validation = self.process_train_data(x, y, validation_data, validation_split)
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
        stopers = my_early_stoper(refer_metric = ref_metric, stop_condition = 15)
        report_dict = {}
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                        y = y_train.to(self.device).float()
                        self.train_one_batch(x = x, y = y, optim = optim, model = model, loss_func = loss_func)
                        """
                        y_pred = model(x).squeeze()
                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(),reduction='mean')
                        reg_loss = self.get_regularization_loss()
                        total_loss = loss + reg_loss + self.aux_loss
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()
                        """
                        """
                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if(name == 'uauc' or name == 'gauc'):
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                        """
                print('train finished')

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
 
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

            
            need_saving = stopers.update_and_isbest(epoch_logs, epoch) # and (epoch_logs[performance_metric] > 0.9 * self.best_performance_value)
            need_stopping = stopers.is_stop()
            self.my_save_model(epoch_logs = epoch_logs, need_saving = need_saving, save_models = save_models, ref_metric = ref_metric, epoch = epoch, mode = mode)

            if need_stopping:
                print("early stop.......")
                break
        return self.history

    def process_train_data(self, x, y, validation_data, validation_split):
        if isinstance(x, dict):
            x_temp = [x[feature] for feature in self.feature_index]
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
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
            if x[i].dtype == 'object':
                x[i] = x[i].astype(int)
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        return train_tensor_data, val_x, val_y, do_validation
        
    def my_tune_report(self, epoch_logs, report_dict, epoch, train_logs = {}, sum_dict =None):
        if train_logs is not None and train_logs is not {}:
            for name, result in train_logs.items():
                if name != 'batch_num':
                    report_dict[name] = result
        for name in self.metrics:
            if ('val_' + name + '_top') in list(report_dict.keys()):
                if name in ['auc', 'acc', 'gauc'] or 'ndcg' in name:
                    if epoch_logs['val_' + name] > report_dict['val_' + name + '_top']:
                        report_dict['val_' + name + '_top'] = epoch_logs['val_' + name]
                else:
                    if epoch_logs['val_' + name] < report_dict['val_' + name + '_top']:
                        report_dict['val_' + name + '_top'] = epoch_logs['val_' + name]
            else:
                report_dict['val_' + name + '_top'] = epoch_logs['val_' + name]
            report_dict['val_' + name] = epoch_logs['val_' + name]
            # report_dict[name] = epoch_logs[name]
            if 'test_' + name in list(epoch_logs.keys()):
                report_dict['test_' + name] = epoch_logs['test_' + name]

        for name in epoch_logs.keys():
            if 'discripancy' in name:
                if (name + 'top') in list(report_dict.keys()):
                    if epoch_logs[name] < report_dict[name + 'top']:
                        report_dict[name + 'top'] = epoch_logs[name]
                    else:
                        pass
                else:
                    report_dict[name + 'top'] = epoch_logs[name]
                report_dict[name] = epoch_logs[name]
            if name == 'loss' or name == 'reg_loss' or 'search' in name or 'performok' in name:
                report_dict[name] = epoch_logs[name]

        for name in epoch_logs.keys():
            if 'performance_satisfied' in name:
                report_dict[name] = epoch_logs[name]

        
        report_names = ['val_dp_v1', 'val_dp_v2','val_dp_v3','train_dro_dp','val_dp_true', 'val_dp_true_search' ,
                        'val_dp', 'val_dp_top','val_dp_true_top', 'val_rmse', 'val_dp_search', 'val_dp_v2_performok', 'val_dp_v3_performok'
                        ,'val_dp_performance_satisfied_top', 'val_dp_true_performok', 'val_dp_performok', 'val_dp_v1_performok', 'val_dp_v4',
                        'val_dp_v4_performok','val_dp_noisy','val_dp_fake','val_dp_noisy_performok','val_dp_fake_noisy_performok']
        
        new_report_dict = {key:value for key,value in report_dict.items() if key in report_names}
        # new_report_dict['val_dp_v5'] = new_report_dict['val_dp_v4'] + new_report_dict['val_dp']

        if (sum_dict == {})or (sum_dict is None):
            sum_dict = new_report_dict.copy()
        else:
            for key in new_report_dict.keys():
                sum_dict[key] += new_report_dict[key]

        for metric in ['val_dp', 'val_dp_v1', 'val_dp_v2', 'val_dp_v3', 'val_dp_v4', 'val_dp_v5','val_dp_noisy','val_dp_fake']:
            if metric in list(new_report_dict.keys()) and 'val_dp_true' in list(new_report_dict.keys()):
                new_report_dict[metric + '-true'] = abs(new_report_dict[metric] - new_report_dict['val_dp_true'])
                # if sum_dict == {} or sum_dict is None:
                try:
                    sum_dict[metric + '-true'] = new_report_dict[metric + '-true']
                    new_report_dict[metric + '-true_aver'] = new_report_dict[metric + '-true']
                except:
                    sum_dict[metric + '-true'] += new_report_dict[metric + '-true']
                    new_report_dict[metric + '-true_aver'] = sum_dict[metric + '-true'] / (epoch + 1 )
        
        try:
            train.report(new_report_dict) # tune.report(**report_dict, training_iteraction = epoch) # train.report(report_dict) # tune.report(**report_dict, training_iteraction = epoch)
        except:
            tune.report(**new_report_dict, training_iteraction = epoch)
        return new_report_dict, sum_dict

    def my_save_model(self, epoch_logs, need_saving, save_models, ref_metric, epoch, mode = 'min', model = None):
        if self.trial_name is not None:
            mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name, self.trial_name)
            mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,'best_result.pt') 
        else:
            mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name)
            mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,'best_result.pt') 
        if not os.path.exists(mycheckpoint_dir):
            os.makedirs(mycheckpoint_dir)
        if not os.path.exists(mycheckpoint_path):
            if mode == 'min':
                best_result = 1000000
            else:
                best_result = -1000000
        else:
            best_result = torch.load(mycheckpoint_path)['result']
        #  and compare_mark
        compare_mark = ((mode == 'min') and (best_result >= epoch_logs[ref_metric])) or ((mode == 'max') and (best_result <= epoch_logs[ref_metric]))
        if need_saving and save_models and compare_mark:
            best_result = epoch_logs[ref_metric]
            # save best results
            if self.trial_name is not None:
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,'best_result.pt') 
            else:
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,'best_result.pt') 
            mycheckpoint = {
                'result':epoch_logs[ref_metric],
                'model_config':self.model_config_name}
            torch.save(mycheckpoint,mycheckpoint_path) 
            # save the model
            if model is None:
                mycheckpoint = {
                    'epoch':epoch,
                    'model':self}
            else:
                mycheckpoint = {
                    'epoch':epoch,
                    'model':model}                
            if self.trial_name is not None:
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,'best_model.pt') 
            else:
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,'best_model.pt') 
            torch.save(mycheckpoint,mycheckpoint_path)




    def get_model_info(self, data_name, model_name, configs, trial_name = None):
        model_config_name = model_name
        for key,value in configs.items():
            model_config_name += (',' + key + ':' + str(value))
        self.data_name = data_name
        self.model_name = model_name
        self.trial_name = trial_name
        self.model_config_name = model_config_name
        self.model_config_dict = configs


    def evaluate(self, x, y, batch_size=256, user_id = None, task = 'topk'):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        if task == 'topk':
            # evaluate need : 
            # input x : dataframe <user_id, item_id, ratings>
            # generate:dataframe: user_id, y_trues, y_preds, sensitive_attribute
            eval_result = {}
            # x['predict_rating_list'] = pd.Series()
            # x['predict_rating_list'] = x['predict_rating_list'].astype('object')
            
            # for idx in x.index:
            #     user_id = x.loc[idx,'user_id']
            #     user_input = {}
            #     item_list = x.loc[idx, 'item_list']
            #     user_input['user_id'] = (np.ones([len(item_list)]) * user_id).astype(int)
            #     user_input['item_id'] = np.array(item_list).astype(int)
            #     user_pred = self.predict(user_input, batch_size).squeeze().tolist()
            #     x.at[idx, 'predict_rating_list'] = user_pred
            input = {}
            input['user_id'] = np.array(x['user_id']).astype(int)
            input['item_id'] = np.array(x['item_id']).astype(int)
            predict_ratings = self.predict(input, batch_size)
            x['predict_ratings'] = predict_ratings
            for name, metric_fun in self.metrics.items():
                eval_result[name] = metric_fun(x)
        else:
            if isinstance(x, dict):
                x_temp = [x[feature] for feature in self.feature_index]
                x = x_temp

            if isinstance(y, pd.DataFrame):
                y = y.values
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)
                    
            x = np.concatenate(x, axis=-1)
            pred_ans = self.predict(x, batch_size)
            #y = y[:int(y.shape[0]/batch_size)*batch_size] #drop last
            eval_result = {}
            for name, metric_fun in self.metrics.items():
                if(name == 'gauc' or name == 'uauc'):
                    eval_result[name] = metric_fun(y, pred_ans,user_id)
                else:
                    eval_result[name] = metric_fun(y, pred_ans)


        return eval_result

    def generate_basic_rankings(self, x, y, batch_size=256, ):
        eval_result = {}
        # x['predict_rating_list'] = pd.Series()
        # x['predict_rating_list'] = x['predict_rating_list'].astype('object')
        basic_rankings = pd.DataFrame(columns = ['user_id', 'item_id', 'score'])
        basic_rankings_idx = 0
        for idx in x.index:
            user_id = x.loc[idx,'user_id']
            user_input = {}
            item_list = x.loc[idx, 'item_list']
            user_input['user_id'] = (np.ones([len(item_list)]) * user_id).astype(int)
            user_input['item_id'] = np.array(item_list).astype(int)
            user_pred = self.predict(user_input, batch_size).squeeze()
            for i in range(user_pred.shape[0]):
                basic_rankings.at[basic_rankings_idx + i, 'user_id'] = user_id 
                basic_rankings.at[basic_rankings_idx + i, 'item_id'] = item_list[i]
                basic_rankings.at[basic_rankings_idx + i, 'score'] = user_pred[i]
            basic_rankings_idx += user_pred.shape[0]
            # x.at[idx, 'predict_rating_list'] = user_pred
        print(basic_rankings)
        return basic_rankings
    
    def predict(self, x, batch_size=256):
        """
        
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()

        #copy this part to function evaluate to get the 'env' information and add them to logs
        if isinstance(x, dict):
            x_temp = [x[feature] for feature in self.feature_index]
            x = x_temp
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)
            x = np.concatenate(x, axis=-1)
        
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
    


    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

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
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns] #error

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                threshold = 0.5,
                sensitive = True, sensitive_num = 2, cal_discripancy_metrics = None, fairness_metric = None,
                performance_metric = None, 
                best_performance_value = 0, 
                sensitive_attr_values = (0,1), 
                explicit = True, 
                thre = 3.5
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics, threshold = threshold, sensitive = sensitive, 
        sensitive_num = sensitive_num, explicit = explicit, thre = thre)
        self.cal_discripancy_metrics = cal_discripancy_metrics
        self.best_performance_value = best_performance_value
        self.performance_metric = performance_metric
        self.fairness_metric = fairness_metric
        self.sensitive_attr_values = sensitive_attr_values

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)
    
    def _log_loss_with_logits(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        loss_func = F.binary_cross_entropy_with_logits
        with torch.no_grad():
            y_true = torch.from_numpy(y_true).float()# .reshape(-1,1)
            y_pred = torch.from_numpy(y_pred).float()# .reshape(-1,1)
            loss = loss_func(y_pred.reshape(-1,1),y_true.reshape(-1,1),reduction='mean')
        return float(loss.clone().detach().numpy().tolist())

    def _get_metrics(self, metrics, set_eps=False, threshold = 0.5, sensitive = True, sensitive_num = 2, 
        explicit = True, thre = 3.5):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                # if metric == "binary_crossentropy" or metric == "logloss":
                #     metrics_[metric] = self._log_loss_with_logits
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "rmse":
                    metrics_[metric] = rooted_mean_squared_error 
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] =  functools.partial(cal_acc, threshold = threshold)
                    #cal_acc# lambda y_true, y_pred: accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))
                if metric =='gauc' or metric == 'uauc':
                    metrics_[metric] = self.cal_group_auc
                if 'ndcg' in metric:
                    (_, __, at) = metric.partition('myat')
                    at = int(at)
                    metrics_[metric] =  functools.partial(cal_mean_ndcg, at = at, explicit = explicit, thre = thre)
                if 'recall' in metric:
                    (_, __, at) = metric.partition('myat')
                    at = int(at)
                    metrics_[metric] =  functools.partial(cal_mean_recall, at = at, explicit = explicit, thre = thre)
                if 'precision' in metric:
                    (_, __, at) = metric.partition('myat')
                    at = int(at)
                    metrics_[metric] =  functools.partial(cal_mean_precision, at = at, explicit = explicit, thre = thre)
                if metric == 'mae':
                    metrics_[metric] = cal_mean_mae
                if metric == 'average_rating':
                    metrics_[metric] = cal_average_rating
                if metric == 'uabs':
                    metrics_[metric] = cal_mean_uabs
                if metric == 'uabs_true':
                    metrics_[metric] = cal_mean_uabs_true
                if metric == 'uabs_v1':
                    metrics_[metric] = cal_mean_uabs_v1
                if metric == 'uabs_v1_true':
                    metrics_[metric] = cal_mean_uabs_v1_true    
                if metric == 'uval':
                    metrics_[metric] = cal_mean_uval
                if metric == 'uval_true':
                    metrics_[metric] = cal_mean_uval_true
                if metric == 'dp':
                    metrics_[metric] = cal_mean_dp
                if metric == 'dp_true':
                    metrics_[metric] = cal_mean_dp_true
                if metric == 'dp_noisy':
                    metrics_[metric] = cal_mean_dp_noisy
                if metric == 'dp_fake':
                    metrics_[metric] = cal_mean_dp_fake

                self.metrics_names.append(metric)

        return metrics_
    
    
    def cal_group_auc(self,labels, preds, user_id_list):
        """Calculate group auc"""
        
        if len(user_id_list) != len(labels):
            raise ValueError(
                "impression id num should equal to the sample num," \
                "impression id num is {0}".format(len(user_id_list)))
        group_score = defaultdict(lambda: [])
        group_truth = defaultdict(lambda: [])
        for idx, truth in enumerate(labels):
            user_id = user_id_list[idx]
            score = preds[idx]
            truth = labels[idx]
            group_score[user_id].append(score)
            group_truth[user_id].append(truth)
        group_flag = defaultdict(lambda: False)
        for user_id in set(user_id_list):
            truths = group_truth[user_id]
            flag = False
            for i in range(len(truths) - 1):
                if truths[i] != truths[i + 1]:
                    flag = True
                    break
            group_flag[user_id] = flag
        impression_total = 0
        total_auc = 0
        for user_id in group_flag:
            if group_flag[user_id]:
                auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
                total_auc += auc * len(group_truth[user_id])
                impression_total += len(group_truth[user_id])
        group_auc = float(total_auc) / impression_total
        group_auc = round(group_auc, 4)
        return group_auc

    

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]