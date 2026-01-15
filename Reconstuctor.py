
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from experiment_config import experiment_config
from Dataset import *
import numpy as np
import pandas as pd
import torch
import time
import argparse
import random
from load_models import *
import warnings
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

def arg_para():
    parser = argparse.ArgumentParser(description='DRFO')
    parser.add_argument('--k',type=int,default=48,help = 'dim of hidden layer of feature interaction')
    parser.add_argument('--data_name',type= str,default='tenrec',help = 'name of the dataset')
    parser.add_argument('--model_name',type= str,default='LR',help = 'name of model(you cannot assign models through this arg)')
    parser.add_argument('--trial_name',type= str,default='20230411-svm-debug',help = 'name of trial')
    parser.add_argument('--seed',type=int,default=2004,help = 'random seed of the exp')
    parser.add_argument('--know_size',type=float,default= 0.3,help = 'propotion of users which have gender labels')
    parser.add_argument('--use_classifier_stage',type = bool,default = True)
    parser.add_argument('--classifier_stage',type = int,default = 1000)
    parser.add_argument('--cuda', type = str, default='0')
    return parser.parse_args()

args = arg_para()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
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
classifier_stage = args.classifier_stage
use_classifier_stage = args.use_classifier_stage
###########
begin_time = time.time()
print('starting process data')
train_data, train_label_data, train_nolabel_data, val_data, test_data, data, train_label_users, train_nolabel_users = process_data(data_name,
        know_size, thre, explicit, use_thre, test_inter_num, test_propotion)
print('process time cost', time.time() - begin_time)
begin_time = time.time()
notest_data = data.loc[~data.index.isin(test_data.index.tolist()),]  
mark_0 = 0
mark_1 = 0
train_users = []
sensitive_attributes = []
data = pd.read_hdf('./{}.h5'.format(data_name))
genders = data['gender']

random.seed(args.classifier_stage)
random.shuffle(train_label_users)
for user in train_label_users:
    sensitive_attribute = genders[data[data['user_id'] == user].index[0]]
    sensitive_attributes.append(sensitive_attribute)
sensitive_attribute_0_num = len(sensitive_attributes) - sum(sensitive_attributes)
sensitive_attribute_1_num = sum(sensitive_attributes)
for user,sensitive_attribute in zip(train_label_users, sensitive_attributes):
    if sensitive_attribute == 0 and mark_0 < int(sensitive_attribute_0_num * 0.8):
        mark_0 += 1
        train_users.append(user)
    if sensitive_attribute == 1 and mark_1 < int(sensitive_attribute_1_num * 0.8):
        mark_1 += 1
        train_users.append(user)
val_users = [user for user in train_label_users if user not in train_users]
notest_datamat = generate_intermat(notest_data)
notest_datamat1 = generate_intermat_1(notest_data)
print('generate intermat cost', time.time() - begin_time)
begin_time = time.time()
if args.data_name == 'tenrec':
    sensitive_data = pd.read_hdf('./tenrec.h5')
if args.data_name == 'ml-1m':
    sensitive_data = pd.read_hdf('./ml-1m.h5')
notest_datamat = add_sensitive_attr_noise(notest_datamat, sensitive_data, 'gender')
train_data = notest_datamat.loc[notest_datamat.index.isin(train_users)]
val_data = notest_datamat.loc[notest_datamat.index.isin(val_users)]
print('get data cost', time.time() - begin_time)
begin_time = time.time()
Dataset = MyDataset_for_demoinference(data_name, k, train_data = train_data, val_data = val_data, data = notest_datamat) 
model_input = Dataset.model_input
fixlen_feature_columns = Dataset.fixlen_feature_columns
varlen_feature_columns = Dataset.varlen_feature_columns 
val_model_input = Dataset.val_model_input
# train_data.to_csv('/data/shith/debug/train_data.csv')
# val_data.to_csv('/data/shith/debug/val_data.csv')
target = Dataset.target
linear_feature_columns =  varlen_feature_columns + fixlen_feature_columns
dnn_feature_columns =  varlen_feature_columns + fixlen_feature_columns
print('prepare dataset cost', time.time() - begin_time)
begin_time = time.time()
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'
clf = svm.SVC()
param_grid = {'C': [0.01,0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'sigmoid']}
# with parallel_backend('cuda'):
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid_search.fit(model_input, train_data[target])
print('最佳参数：', grid_search.best_params_)
svm = svm.SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'])
svm.fit(model_input, train_data[target])
file_name  = 'noisy_label_data_{}_know_size_{}.csv'.format(data_name, str(know_size))
file_name_acc = 'noisy_acc_data_{}_know_size_{}.json'.format(data_name, str(know_size))
save_path = os.path.join(workspace, data_name, file_name)
save_path_acc = os.path.join(workspace, data_name, file_name_acc)
sensitive_attr = 'gender'
val_predict_sensitive_attr = svm.predict(val_model_input).squeeze()
val_distances = np.abs(svm.decision_function(val_model_input).squeeze())
true_sensitive_attrs = np.array(val_data[target]).squeeze()
max_score = np.max(val_distances)
min_score = np.min(val_distances)
best_tau = 0
best_acc = 0
for tau in np.linspace(min_score, max_score, 1000):
    correct = 0
    for i in range(len(val_predict_sensitive_attr)):
        if val_distances[i] > tau and val_predict_sensitive_attr[i] == true_sensitive_attrs[i]:
            correct += 1
        elif val_distances[i] <= tau and val_predict_sensitive_attr[i] != true_sensitive_attrs[i]:
            correct += 1
    accuracy = correct / len(val_predict_sensitive_attr)
    if accuracy > best_acc:
        best_acc = accuracy
        best_tau = tau
print("Best threshold τ: {:.4f}".format(best_tau))
val_y_prob_0 = 1 - np.mean(notest_datamat[target].squeeze())
val_y_prob_1 = np.mean(notest_datamat[target].squeeze())

p11 = len(data[(data['gender'] == 1) & (data['ratings'] > 3)]) / len(data[data['ratings'] > 3])
p01 = len(data[(data['gender'] == 0) & (data['ratings'] > 3)]) / len(data[data['ratings'] > 3])
p10 = len(data[(data['gender'] == 1) & (data['ratings'] <= 3)]) / len(data[data['ratings'] <= 3])
p00 = len(data[(data['gender'] == 0) & (data['ratings'] <= 3)]) / len(data[data['ratings'] <= 3])

index_0 = (true_sensitive_attrs == 0)
index_1 = (true_sensitive_attrs == 1)
gammas = {}
gammas['0'] = 1 - accuracy_score(true_sensitive_attrs[index_0].tolist(), val_predict_sensitive_attr[index_0].tolist()) 
gammas['1'] = 1 - accuracy_score(true_sensitive_attrs[index_1].tolist(), val_predict_sensitive_attr[index_1].tolist())
print(gammas)
if args.data_name == 'ml-1m':
    acc_file_path = './workspace/ml-1m/gamma_data_ml-1m_knowsize_{}'.format(str(know_size)) +  '.pt'
if args.data_name == 'tenrec':
    acc_file_path = './workspace/tenrec/gamma_data_tenrec_knowsize_{}'.format(str(know_size)) +  '.pt'
torch.save(gammas,acc_file_path)
def count_zeros_ones(row):
    # 排除'item_id'，'gender'和'user_id'列
    relevant_columns = [col for col in row.index if col not in ['item_id', 'gender', 'user_id']]
    # 计算0和1的数量
    count_zeros = row[relevant_columns].eq(0).sum()
    count_ones = row[relevant_columns].eq(5).sum()
    return count_zeros, count_ones
predict_data = notest_datamat.loc[notest_datamat.index.isin(train_nolabel_users), ]
print('predict data', predict_data)
cal_data = notest_datamat1.loc[notest_datamat1.index.isin(train_nolabel_users), ]
# cal_data.to_csv('/data/shith/1204.csv')
print('cal_data', cal_data)
predict_model_input = predict_data.loc[:,Dataset.sparse_features]
predict_sensitive_attr = svm.predict(predict_model_input)
distances = svm.decision_function(predict_model_input)
accuracy = accuracy_score(predict_data[target], predict_sensitive_attr)
print('accuracy', accuracy)
predict_sensitive_df = pd.DataFrame([])
predict_sensitive_df['user_id'] = predict_data.index
print(predict_sensitive_df['user_id'])
print(predict_sensitive_attr.shape[0], predict_sensitive_df['user_id'].shape[0])
predict_sensitive_df['predict_score'] = predict_sensitive_attr
predict_sensitive_df[sensitive_attr] = predict_sensitive_attr 
# np.where((predict_sensitive_attr > threshold), 1, 0)
predict_sensitive_df['distance'] = distances
print('distance', distances)
# generate cgl sensitive_attribute
predict_sensitive_attr_pseudo = []
zeros_count_list, ones_count_list = zip(*cal_data.apply(count_zeros_ones, axis=1))
zeros_count_list = list(zeros_count_list)
ones_count_list = list(ones_count_list)
print(zeros_count_list,ones_count_list)
for i in range(len(predict_sensitive_attr)):
    if np.abs(distances[i]) > best_tau:
        predict_sensitive_attr_pseudo.append(predict_sensitive_attr[i])
    else:
        # random sample
        p0 = zeros_count_list[i] / (zeros_count_list[i] + ones_count_list[i])
        p1 = 1 - p0
        thre_p1 = p1 * p11 + p0 * p10
        if random.uniform(0,1) <= thre_p1:
            predict_sensitive_attr_pseudo.append(1)
        else:
            predict_sensitive_attr_pseudo.append(0)
predict_sensitive_df[sensitive_attr + '_cgl'] = predict_sensitive_attr_pseudo
predict_sensitive_df.to_csv(save_path)
data = pd.read_hdf('./{}.h5'.format(args.data_name))
true_sensitive_attrs = []
user_ids = np.array(predict_sensitive_df['user_id']).tolist()
for user_id in user_ids:
    true_sensitive_attrs.append(data.loc[data['user_id'] == user_id,  sensitive_attr])
acc = accuracy_score(predict_data[target], predict_sensitive_attr)
print('predicted_accuracy_is: ', acc)
file = open(save_path_acc, "w")
json.dump(acc, file)
file.write('\r\n')
json.dump(gammas,file)
file.close() 

import os
os._exit(0)