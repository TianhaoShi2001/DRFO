import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
workspace = './workspace'
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models.deepfm import *
import time
import itertools
from ray import tune, train
from mybasemodels import *
import os
import copy
import torch.nn as nn



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



class MyMF(MybaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns,  use_fm=True,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                  task='binary', device='cpu', gpus=None,use_linear = False, lambda_ = 0, explicit = True, use_thre = False):

        super(MyMF, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        self.lambda_ = lambda_
        self.use_fm = use_fm
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.use_linear = use_linear
        self.sigmoid = torch.nn.Sigmoid
        self.explicit = explicit
        self.use_thre = use_thre
        if use_fm : # only id embedding, mf
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
            y_pred = logit + self.bias
        else:
            y_pred = torch.sigmoid(logit+self.bias)
        return y_pred

    def fit(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, 
            save_models = True, ref_metric = None, mode = 'min'):

        train_tensor_data, val_x, val_y, do_validation = self.process_train_data(x, y
            , validation_data, validation_split)
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
        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        if ref_metric is None:
            ref_metric = "val_auc"
        stopers = my_early_stoper_mf(refer_metric = ref_metric, stop_condition = 200, mode = mode)
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
                x = x_train.to(self.device).float() 
                y = y_train.to(self.device).float()
                self.train_one_batch(x = x, y = y, optim = optim, model = model, loss_func = loss_func)
                       
                print('train finished')
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
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
            ####### save models
            if self.performance_metric is not None:
                performance_metric = 'val_' + self.performance_metric
                performance_satisfied = epoch_logs[performance_metric] > 0.9 * self.best_performance_value
            else:
                performance_satisfied = True
            need_saving = stopers.update_and_isbest(epoch_logs, epoch)
            need_stopping = stopers.is_stop()
            self.my_save_model(epoch_logs = epoch_logs, need_saving = need_saving, save_models = save_models, ref_metric = ref_metric, epoch = epoch, mode = mode)
            if need_stopping:
                print("early stop.......")
                # not stop here 
                break
        return self.history








class MF_with_reconstructed(MybaseModel):
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
    



    



class DRFO(MF_with_reconstructed):
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


    def fit(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True,  
            save_models = True, ref_metric = None, tune_mode = 'min', 
            performance_mode = 'min', performance_scale = 0.9,
            train_label_users = None,  train_nolabel_users= None, etas = None, 
            ):
        self.etas = etas
        train_tensor_data, val_x, val_y, do_validation = self.process_train_data(x, y, validation_data,
         validation_split)
        if batch_size is None:
            batch_size = 256
        model = self.train()
        loss_func = self.loss_func
        optim1 = self.optim1
        optim2 = self.optim2
        # if
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus) 
        else:
            print(self.device)
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle = shuffle, batch_size = batch_size, num_workers = 4)  
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        if ref_metric is None:
            ref_metric = "val_auc"
        stopers = EarlyStopping(refer_metric = ref_metric, mode = tune_mode, performance_satisfied_stop_condition=130)
        report_dict = {}

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
            train_logs = {}
            # generate base performance 
            begin_time = time.time()
            for batch_idx in range(num_batches):
                u_train, x_train, y_train = next(itertools.islice(train_iter_cycle, batch_idx, batch_idx + 1))
                print('load_data, cost', time.time()-begin_time)
                begin_time = time.time()
                u = u_train.to(self.device).float()
                x = x_train.to(self.device).float() 
                y = y_train.to(self.device).float()
                print('trans to gpu, cost', time.time()-begin_time)
                begin_time = time.time()
                batch_train_logs = self.train_one_batch(u = u, x = x, y = y, optim1 = optim1, optim2 = optim2, model = model,
                    loss_func = loss_func, thetas = thetas,  train_label_users = train_label_users, train_nolabel_users = train_nolabel_users, )
                print('train one batch, cost', time.time()-begin_time)
                begin_time = time.time()
                if train_logs == {}:
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
     loss_func, thetas,  train_label_users = None, train_nolabel_users = None,):
        
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
        sensitive_range = [1]
        label_user_index = torch.isin(u.squeeze(), torch.tensor(train_label_users, device = self.device))
        nolabel_user_index = torch.isin(u.squeeze(), torch.tensor(train_nolabel_users, device = self.device))
        for i in sensitive_range: 
            group_i_prob = self.get_dro_distribution_probability(u, i).squeeze() 
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
            for i in sensitive_range: 
                mask = torch.zeros_like(self.user_embedding_dict_dro[str(i)]['user_id'].weight.data)
                mask.scatter_(dim=0, index=nolabel_user_tensor.unsqueeze(1), value=1)
                self.user_embedding_dict_dro[str(i)]['user_id'].weight.data *= mask 
            for i in sensitive_range: 
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
        return batch_train_logs



class MF_DRFO_extension(MF_with_reconstructed):
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

    def fit(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, 
            save_models = True, ref_metric = None, tune_mode = 'min', 
            performance_mode = 'min', performance_scale = 0.9,
            train_label_users = None,  etas = None, 
            train_nolabel_users_profile = [],  train_nolabel_users_lack_profile = []):
        self.etas = etas
        train_tensor_data, val_x, val_y, do_validation = self.process_train_data(x, y, validation_data,
         validation_split)
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
        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        if ref_metric is None:
            ref_metric = "val_auc"
        stopers = EarlyStopping(refer_metric = ref_metric, mode = tune_mode,
                                 performance_satisfied_stop_condition=130)
        report_dict = {}
        # get model parameters
        thetas = []
        for name, param in self.named_parameters():
            if 'user_embedding' not in name and 'lambda' not in name:
                thetas.append(param)
        # generate base performance 
        if do_validation:
            eval_result = self.evaluate(val_x, val_y, batch_size)
            for name, result in eval_result.items():
                if name == self.performance_metric:
                    self.best_performance_value = result
        del_metrics = []
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
            train_logs = {}
            begin_time = time.time()
            for batch_idx in range(num_batches):
                u_train, x_train, y_train = next(itertools.islice(train_iter_cycle, batch_idx, batch_idx + 1))
                print('load_data, cost', time.time()-begin_time)
                begin_time = time.time()
                u = u_train.to(self.device).float()
                x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                y = y_train.to(self.device).float()
                print('trans to gpu, cost', time.time()-begin_time)
                begin_time = time.time()
                batch_train_logs = self.train_one_batch(u = u, x = x, y = y, optim1 = optim1, optim2 = optim2, model = model,
                    loss_func = loss_func, thetas = thetas, train_label_users = train_label_users, 
                     train_nolabel_users_profile = train_nolabel_users_profile, 
                     train_nolabel_users_lack_profile = train_nolabel_users_lack_profile)
                print('train one batch, cost', time.time()-begin_time)

                begin_time = time.time()
                if train_logs == {}:
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
     loss_func, thetas,     train_label_users = None,
       train_nolabel_users_lack_profile = [], train_nolabel_users_profile = []):
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

        for i in sensitive_range: 
            group_i_prob_profile = self.get_dro_distribution_probability_profile(u, i).squeeze()
            group_i_prob_lack_profile = self.get_dro_distribution_probability_lack_profile(u, i).squeeze()
            print('sum_prob_file', torch.sum(group_i_prob_profile))
            print('sum_lack_prob_file', torch.sum(group_i_prob_lack_profile))
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
        if len(train_nolabel_users_profile):
            nolabel_user_tensor_profile = torch.tensor(train_nolabel_users_profile, device = self.device)
            with torch.no_grad():
                # set to 0
                for i in sensitive_range: 
                    mask = torch.zeros_like(self.user_embedding_dict_dro_profile[str(i)]['user_id'].weight.data)
                    mask.scatter_(dim=0, index=nolabel_user_tensor_profile.unsqueeze(1), value=1)
                    self.user_embedding_dict_dro_profile[str(i)]['user_id'].weight.data *= mask 

                for i in sensitive_range: 
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
                for i in sensitive_range: 
                    mask = torch.zeros_like(self.user_embedding_dict_dro_lack_profile[str(i)]['user_id'].weight.data)
                    mask.scatter_(dim=0, index=nolabel_user_tensor_lack_profile.unsqueeze(1), value=1)
                    self.user_embedding_dict_dro_lack_profile[str(i)]['user_id'].weight.data *= mask 

                for i in sensitive_range: 
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
                for i in sensitive_range: 
                    mask = torch.zeros_like(self.user_embedding_dict_dro_lack_profile[str(i)]['user_id'].weight.data)
                    self.user_embedding_dict_dro_lack_profile[str(i)]['user_id'].weight.data *= mask 

        return batch_train_logs




