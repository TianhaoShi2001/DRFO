
import sys
import os
from tabnanny import check
sys.path.insert(1, os.path.join(sys.path[0], '../'))
workspace = '/data/shith/DRFO/workspace'
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names,combined_dnn_input
from deepctr_torch.layers import DNN
from deepctr_torch.models.deepfm import *
import time
from torch.utils.data import DataLoader
import torch.utils.data as Data
from deepctr_torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from sklearn.metrics import roc_auc_score, mean_absolute_error
from deepctr_torch.layers.utils import slice_arrays
import torch.nn.functional as F
from collections import defaultdict
from deepctr_torch.layers import PredictionLayer
from deepctr_torch.layers.utils import slice_arrays
from deepctr_torch.callbacks import History
from ray import tune,train
import torch.nn as nn



                
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
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit

class EarlyStopping(object):
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

class my_early_stoper_mf(object):
    def __init__(self,refer_metric='val_auc',stop_condition = 50, mode = 'min'):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric
        self.mode = mode

    def update_and_isbest(self, eval_metric, epoch):
        if self.init_flag:
            self.best_epoch = epoch
            self.init_flag = False
            self.best_eval_result = eval_metric
            return True
        elif self.mode == 'min':
            if eval_metric[self.refer_metric] < self.best_eval_result[self.refer_metric] : # update the best results
                self.best_eval_result = eval_metric
                self.not_change = 0
                self.best_epoch = epoch
                return True              # best
            else:                        # add one to the maker for not_change information 
                self.not_change += 1     # not best
                return False
        elif self.mode == 'max':
            if eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric] : # update the best results
                self.best_eval_result = eval_metric
                self.not_change = 0
                self.best_epoch = epoch
                return True              # best
            else:                        # add one to the maker for not_change information 
                self.not_change += 1     # not best
                return False

    def is_stop(self):
        if self.not_change > self.stop_condition:
            return True
        else:
            return False

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
    return sum(maes) / len(maes)

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

    
def cal_group_rating(x, sensitive_value = 0, explicit = True, thre = 3.5):
    ratings = []
    x_sensitive = x.loc[x['sensitive_attribute'] == sensitive_value, :]
    data_groups = x_sensitive.groupby('user_id')
    for user, df in data_groups:
        list1 = list(df['predict_ratings'].tolist())
        ratings.extend(list1)
    return sum(ratings) / len(ratings)



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
            if 'test_' + name in list(epoch_logs.keys()):
                report_dict['test_' + name] = epoch_logs['test_' + name]
        for name in epoch_logs.keys():
            if 'search' in name :
                report_dict[name] = epoch_logs[name]

        # different ray tune version
        try:
            train.report(report_dict) 
        except:
            tune.report(**report_dict, training_iteraction = epoch)
        return report_dict, sum_dict

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
            #  not save 0 at epoch 0 to avoid bug
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

    
    def predict(self, x, batch_size=256):
        model = self.eval()
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
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics,)
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

    def _get_metrics(self, metrics, set_eps=False, ):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss

                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "rmse":
                    metrics_[metric] = rooted_mean_squared_error 

                if metric == 'mae':
                    metrics_[metric] = cal_mean_mae
                if metric == 'dp':
                    metrics_[metric] = cal_mean_dp
                if metric == 'dp_true':
                    metrics_[metric] = cal_mean_dp_true

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