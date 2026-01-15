import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from Dataset import *
import numpy as np
import torch
import torch.nn.functional as F
from mymodels import  DRFO
import time
import argparse
import random
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from load_models import *
from experiment_config import experiment_config
workspace = './workspace'
def arg_para():
    parser = argparse.ArgumentParser(description='DRFO')
    parser.add_argument('--k',type=int,default=32,help = 'dim of hidden layer of feature interaction')
    parser.add_argument('--data_name',type= str,default='tenrec',help = 'name of the dataset')
    parser.add_argument('--model_name',type= str,default='MF',help = 'name of model(you cannot assign models through this arg)')
    parser.add_argument('--trial_name',type= str,default='11',help = 'name of trial')
    parser.add_argument('--seed',type=int,default=2004,help = 'dim of hidden layer of feature interaction')
    parser.add_argument('--know_size',type=float,default= 0.7,help = 'propotion of users which have gender labels')
    parser.add_argument('--use_classifier_stage',type = bool,default = True)
    parser.add_argument('--classifier_stage',type = int,default = 1000)
    parser.add_argument('--train_sensitive',type = str,default = 'noisy')
    parser.add_argument('--valid_sensitive',type = str,default = 'fake')
    parser.add_argument('--cuda',type = str,default = '3,5,6,7')
    parser.add_argument('--num_per_gpu',type = float,default = 0.15)
    parser.add_argument('--performance_scale',type = float,default = 0.98)
    parser.add_argument('--best_performance_value',type = float,default = 0.4147020733958456)
    parser.add_argument('--is_save_drawlog',type = bool, default = True)
    parser.add_argument('--resume',type = str,default = 'False')
    return parser.parse_args()


args = arg_para() 
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
name_seed = 2004


if args.data_name == 'ml-1m':
    config = {
            'init_lambdas':tune.grid_search([10]),
            'lr_prop':tune.grid_search([1e-3,]), 
            'l2_w':tune.grid_search([1e-6]),
        }
elif args.data_name == 'tenrec':
        config = {
            'init_lambdas':tune.grid_search([1]),
            'lr_prop':tune.grid_search([1e-2]), 
            'l2_w':tune.grid_search([1e-3]),
        }


def trainable(config):
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    # seed = 2004
    k = args.k
    data_name = args.data_name
    know_size = args.know_size# experiment_config['know_size']
    explicit = experiment_config['isexplicit']
    test_inter_num = experiment_config['test_inter_num']
    test_propotion = experiment_config['test_propotion']
    thre = experiment_config['thre']
    use_thre = experiment_config['use_thre']
    classifier_stage = args.classifier_stage
    use_classifier_stage = args.use_classifier_stage

    file_name = 'data_{}_model_{}_knowsize_{}.pt'.format(data_name,
    args.model_name ,str(know_size))
    model_path = os.path.join(workspace, data_name, args.model_name,  file_name)
    begin_time = time.time()
    while True:
        try:
            base_model_state = torch.load(model_path)
            break
        except:
            pass
            if time.time() - begin_time > 5:
                break
    file_name  = 'noisy_label_data_{}_know_size_{}.csv'.format(data_name, str(know_size))
    noisy_label_path = os.path.join(workspace, data_name, file_name)
    Dataset = Dataset_DRFO(data_name, k, know_size, data = None, seed = seed, add_sensitive = args.train_sensitive,
    thre = thre, use_thre = use_thre, test_propotion = test_propotion, test_inter_num = test_inter_num, explicit = explicit, 
    noisy_label_path = noisy_label_path, sample = False, add_tune_sensitive = args.valid_sensitive)
    # get_gammas
    if args.train_sensitive == 'noisy':
        if args.data_name == 'ml-1m':
            acc_file_path = './workspace/ml-1m/gamma_data_ml-1m_knowsize_{}'.format(str(know_size)) +  '.pt'
        if args.data_name == 'tenrec':
            acc_file_path = './workspace/tenrec/gamma_data_tenrec_knowsize_{}'.format(str(know_size)) +  '.pt'
        gammas = list(torch.load(acc_file_path).values())
    else:
        gammas = [0,0]
    fixlen_feature_columns = Dataset.fixlen_feature_columns
    varlen_feature_columns = Dataset.varlen_feature_columns 
    model_input = Dataset.model_input
    val_model_input = Dataset.val_model_input
    train_target = Dataset.train_target
    user_feat_column = Dataset.user_feature_column
    group_distribution_dict = Dataset.user_group_distribution
    train_label_users = Dataset.train_label_users
    train_nolabel_users = Dataset.train_nolabel_users
    linear_feature_columns =  varlen_feature_columns + fixlen_feature_columns
    dnn_feature_columns =  varlen_feature_columns + fixlen_feature_columns
    etas = Dataset.etas
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    if explicit and not use_thre:
        loss_fn = F.mse_loss
    else:
        loss_fn = F.binary_cross_entropy
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    model = DRFO(linear_feature_columns,dnn_feature_columns,user_feat_column = user_feat_column,
          device = device,use_linear = True,l2_reg_embedding=config['l2_w'], 
    obeserved_group_distributions = group_distribution_dict, sensitive_num = 2,
      gammas = gammas, init_lambdas = config['init_lambdas'], explicit = explicit, use_thre = use_thre, lambda_lr=0)
    model.load_state_dict(base_model_state, strict = False)
    other_params = []
    for name,param in model.named_parameters():
        if 'user_embedding' not in name and 'lambda' not in name:
            other_params.append(param)
    opt = torch.optim.Adam([
        {'params':model.user_embedding_dict_dro.parameters(), 'lr':config['lr_prop']},
        {'params': other_params, 'lr': 1e-3},
    ])
    model.optim1 = torch.optim.Adam([{'params':model.user_embedding_dict_dro.parameters(), 'lr':config['lr_prop']},])
    model.optim2 = torch.optim.Adam([
            {'params': other_params, 'lr': 1e-3},
        ])
    ref_metric = 'val_dp' 
    best_performance_value = args.best_performance_value
    performance_metric = 'rmse'
    model.compile(opt, loss_fn, metrics=['rmse','dp', 'dp_true'],
   performance_metric = performance_metric, best_performance_value = best_performance_value,
     explicit = (explicit and not use_thre), thre = thre)
    model_config = {}
    for key,value in config.items():
        model_config[key] = config[key]
    model.get_model_info(args.data_name, args.model_name, model_config, args.trial_name)
    epoch_logs = model.fit(model_input, train_target, batch_size = train_target.shape[0], epochs=500, verbose=2, ref_metric=ref_metric
    ,validation_data=(val_model_input,[]), tune_mode = 'min', performance_mode = 'min', performance_scale = args.performance_scale,
     train_label_users = train_label_users, train_nolabel_users = train_nolabel_users,
    etas = etas, ) 


scheduler = ASHAScheduler(
        metric="val_auc_top",
        mode="max",
        grace_period=20)

reporter = CLIReporter(
        parameter_columns=["lr", "lambda_lr"],
        metric_columns=["loss","val_ndcgmyat10"])


class CustomStopper(tune.Stopper):
        def __init__(self):
            self.num = 0

        def __call__(self, trial_id, result):
            if(result['val_auc']<=0.52):
                return True
            
            return False

        def stop_all(self):
            return False

stopper = CustomStopper()

if args.resume == 'False':
    resume = False
elif args.resume == 'ERRORED_ONLY':
    resume = "ERRORED_ONLY" 
elif args.resume == 'True':
    resume = True
result = tune.run(
    trainable,
    resources_per_trial={"cpu": 1, "gpu": args.num_per_gpu},
    local_dir = './ray_results',
    name = args.data_name + args.model_name + args.trial_name,
    config=config,
    progress_reporter=reporter,
    resume = resume
    )


seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)

k = args.k
data_name = args.data_name                                 
know_size = args.know_size # experiment_config['know_size']
explicit = experiment_config['isexplicit']
test_inter_num = experiment_config['test_inter_num']
test_propotion = experiment_config['test_propotion']
thre = experiment_config['thre']
use_thre = experiment_config['use_thre']
classifier_stage = args.classifier_stage
use_classifier_stage = args.use_classifier_stage
exp_config = {
    'seed':args.seed,
    'data_name':args.data_name,
    'know_size':args.know_size,
    'explicit':explicit,
    'test_inter_num':test_inter_num,
    'test_propotion':test_propotion,
    'thre':thre,
    'use_thre':use_thre,
    'classifier_stage':classifier_stage,
    'use_classifier_stage':use_classifier_stage,
    'classifier_stage': classifier_stage,
    'model_name':args.model_name,
    'performance_scale':args.performance_scale, 
    'train_mode':args.train_sensitive,
    'valid_mode':args.valid_sensitive,
}

file_name  = 'noisy_label_data_{}_know_size_{}.csv'.format(data_name, str(know_size))
noisy_label_path = os.path.join(workspace, data_name, file_name)

Dataset = Dataset_for_evaluation(data_name, k, know_size, data = None, seed = seed, add_sensitive = 'True', explicit = explicit,
 thre = thre, use_thre = use_thre, test_inter_num = test_inter_num, 
 test_propotion = test_propotion, noisy_label_path = noisy_label_path)
lt = new_load_models(data_name = args.data_name, model_name = args.model_name, trial_name = args.trial_name, test_mode = 'true', exp_config = exp_config, dro = True)
lt.load_test(args.model_name + args.trial_name, Dataset = Dataset)
lt.load_test(args.model_name + args.trial_name, Dataset = Dataset, test_or_val = 'val')
import os
os._exit(0)