import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from Dataset import *
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from mymodels import MyFM
import time
import argparse
import random
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from load_models import *
from sklearn.model_selection import train_test_split
from experiment_config import experiment_config
def arg_para():
    parser = argparse.ArgumentParser(description='IRM-Feature-Interaction-Selection.')
    parser.add_argument('--k',type=int,default=32,help = 'dim of hidden layer of feature interaction')
    parser.add_argument('--data_name',type= str,default='ml-100k',help = 'name of the dataset')
    parser.add_argument('--model_name',type= str,default='MF',help = 'name of model(you cannot assign models through this arg)')
    parser.add_argument('--trial_name',type= str,default='0421-run-base-model',help = 'name of trial')
    parser.add_argument('--seed',type=int,default=2004,help = 'dim of hidden layer of feature interaction')
    parser.add_argument('--know_size',type=float,default= 0.5,help = 'propotion of users which have gender labels')
    parser.add_argument('--use_classifier_stage',type = bool,default = True,help = '11')
    parser.add_argument('--classifier_stage',type = int,default = 1000,help = '11')
    parser.add_argument('--only_unknown',type = bool,default = False,help = '11')
    parser.add_argument('--cuda',type = str,default = '7',help = '11')
    parser.add_argument('--num_per_gpu',type = float,default = 1,help = '11')
    parser.add_argument('--is_save',type = bool,default = True, help = '11')
    return parser.parse_args()

args = arg_para()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
def trainable(config, checkpoint_dir=None):
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    k = args.k
    data_name = args.data_name
    know_size = args.know_size# 0.5# experiment_config['know_size']
    explicit = experiment_config['isexplicit']
    test_inter_num = experiment_config['test_inter_num']
    test_propotion = experiment_config['test_propotion']
    thre = experiment_config['thre']
    use_thre = experiment_config['use_thre']
    only_unknown = args.only_unknown
    classifier_stage = args.classifier_stage
    use_classifier_stage = args.use_classifier_stage
    file_name  = 'noisy_label_data_{}_knowsize_{}_seed_{}_isexplicit_{}_testinternum_{}_testpropotion_{}_thre_{}_usethre_{}_use_classifier_stage_{}_classifier_stage_{}.csv'\
.format(data_name, str(know_size), str(seed), str(explicit), str(test_inter_num), str(test_propotion), str(thre), str(use_thre),str(use_classifier_stage),str(classifier_stage) )

    # file_name  = 'noisy_label_data_{}_knowsize_{}_seed_{}_isexplicit_{}_testinternum_{}_testpropotion_{}_thre_{}_usethre_{}.csv'.format(data_name, 
    # str(know_size), str(seed), str(explicit), str(test_inter_num), str(test_propotion), str(thre), str(use_thre))
    noisy_label_path = os.path.join(workspace, data_name, file_name)
    Dataset = Dataset_for_MF_with_reconstructed(data_name, k, know_size, data = None, seed = seed, add_sensitive = 'True',
    thre = thre, use_thre = use_thre, test_propotion = test_propotion, test_inter_num = test_inter_num, explicit = explicit, 
    noisy_label_path = noisy_label_path, sample = False, only_unknown = only_unknown)
    fixlen_feature_columns = Dataset.fixlen_feature_columns
    varlen_feature_columns = Dataset.varlen_feature_columns 
    model_input = Dataset.model_input
    val_model_input = Dataset.val_model_input
    target = Dataset.target
    train_target = Dataset.train_target
    linear_feature_columns =  varlen_feature_columns + fixlen_feature_columns
    dnn_feature_columns =  varlen_feature_columns + fixlen_feature_columns
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    if explicit and not use_thre:
        loss_fn = F.mse_loss
    else:
        loss_fn = F.binary_cross_entropy
    model = MyFM(linear_feature_columns, dnn_feature_columns,device = device,use_linear = True,l2_reg_embedding=config['l2_w'], 
    explicit = explicit, use_thre = use_thre)
    opt = _get_optim(model,config['opt'],config['lr'])
    cal_discripancy_metrics = ['average_rating']# None
    fairness_metric = 'average_rating'
    ref_metric = 'val_rmse'# 'val_average_rating_discripancy_0_1' # 'val_ndcgmyat10' # fairness_metric + '_discripancy_' + str(0) + '_' + str(1)
    best_performance_value = None# 0.8645413203097246
    performance_metric = None# 'ndcg@2'
    model.compile(opt, loss_fn, metrics=['rmse','average_rating','dp'], cal_discripancy_metrics = cal_discripancy_metrics,
     fairness_metric=fairness_metric, performance_metric = performance_metric, best_performance_value = best_performance_value, 
     explicit = (explicit and not use_thre), thre = thre)
    model_config = {}
    for key, value in config.items():
        model_config[key] = config[key]
    model.get_model_info(args.data_name, args.model_name, model_config, args.trial_name)
    epoch_logs = model.fit(model_input, train_target, batch_size = train_target.shape[0], epochs=500, verbose=2, ref_metric= ref_metric
    ,validation_data=(val_model_input,[]),mode = 'min')

    
    
def _get_optim( model,optimizer,lr):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(model.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(model.parameters(), lr = lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(model.parameters(),lr = lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(model.parameters(),lr = lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim
    
    
config = {
        'lr':tune.grid_search([1e-3]),#([1e-4,1e-3,1e-2,1e-5]),
        'opt':tune.grid_search(['adam']),#(['adam','rmsprop']),
        'l2_w':tune.grid_search([1e-5]),#([1e-6,1e-5,1e-4,1e-3,1e-2,1e-7]),
        
    }
    # ASHAScheduler会根据指定标准提前中止坏实验





result = tune.run(
    trainable,

    resources_per_trial={"cpu": 1, "gpu":args.num_per_gpu},
    local_dir = './/ray_results',
    name = args.data_name + args.model_name + args.trial_name,
    config=config,
    # scheduler=scheduler,

    # resume = "ERRORED_ONLY"
    )




seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)
k = args.k
data_name = args.data_name
know_size = args.know_size# experiment_config['know_size']
explicit = experiment_config['isexplicit']
test_inter_num = experiment_config['test_inter_num']
test_propotion = experiment_config['test_propotion']
thre = experiment_config['thre']
use_thre = experiment_config['use_thre']
only_unknown = args.only_unknown
# know_size = args.know_size
classifier_stage = args.classifier_stage
use_classifier_stage = args.use_classifier_stage
# path to save basic rankings / not used now for a in-processing method
file_name = 'data_{}_model_{}_knowsize_{}_seed_{}_isexplicit_{}_testinternum_{}_testpropotion_{}_thre_{}_usethre_{}_only_unknown_{}.csv'.format(data_name,
args.model_name ,str(know_size), str(seed), str(explicit), str(test_inter_num), str(test_propotion), str(thre), str(use_thre),str(only_unknown))
save_path = os.path.join(workspace, data_name, args.model_name,  file_name)


# path to load noisy labels
file_name  = 'noisy_label_data_{}_knowsize_{}_seed_{}_isexplicit_{}_testinternum_{}_testpropotion_{}_thre_{}_usethre_{}_use_classifier_stage_{}_classifier_stage_{}.csv'\
.format(data_name, str(know_size), str(seed), str(explicit), str(test_inter_num), str(test_propotion), str(thre), str(use_thre),str(use_classifier_stage),str(classifier_stage) )
noisy_label_path = os.path.join(workspace, data_name, file_name)

Dataset = Dataset_for_evaluation(data_name, k, know_size, data = None, seed = seed, add_sensitive = 'True', explicit = explicit,
 thre = thre, use_thre = use_thre, test_inter_num = test_inter_num, 
 test_propotion = test_propotion, noisy_label_path = noisy_label_path, only_unknown = only_unknown)
lt = new_load_models(data_name = args.data_name, model_name = args.model_name, trial_name = args.trial_name, test_mode = 'True')
lt.load_test(args.model_name + args.trial_name, Dataset = Dataset)
lt.load_test(args.model_name + args.trial_name, Dataset = Dataset, test_or_val = 'val')
seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)
Dataset2 = Dataset_for_evaluation(data_name, k, know_size, data = None, seed = seed, add_sensitive = 'noisy', explicit = explicit,
 thre = thre, use_thre = use_thre, test_inter_num = test_inter_num, 
 test_propotion = test_propotion, noisy_label_path = noisy_label_path, only_unknown = only_unknown)
lt = new_load_models(data_name = args.data_name, model_name = args.model_name, trial_name = args.trial_name)
# save models
if args.is_save:
    model = lt.model
    file_name = 'data_{}_model_{}_knowsize_{}_seed_{}_isexplicit_{}_testinternum_{}_testpropotion_{}_thre_{}_usethre_{}_only_unknown_{}.pt'.format(data_name,
    args.model_name ,str(know_size), str(seed), str(explicit), str(test_inter_num), str(test_propotion), str(thre), str(use_thre),str(only_unknown))
    save_path = os.path.join(workspace, data_name, args.model_name,  file_name)
    torch.save(model.state_dict(),save_path)

# evaluate on valid and test datasets
lt.load_test(args.model_name + args.trial_name, Dataset = Dataset2)
lt.load_test(args.model_name + args.trial_name, Dataset = Dataset2, test_or_val = 'val')
# print("Best config is:", result.get_best_config(metric="val_auc_top",
#         mode="max"))
# print("best result is:",result.get_best_trial('val_auc_top','max','last').last_result["val_auc_top"])