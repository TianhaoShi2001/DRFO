import numpy as np
import pandas as pd
import torch
import time
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat
import sklearn
import warnings
import os
import random
from tqdm import tqdm
workspace = '/data/shith/DRFO/workspace'
def generate_intermat(data):
    data = data.reset_index().copy()
    df = data.pivot_table(index='user_id', columns='item_id', values='ratings', fill_value=0)
    return df

def generate_intermat_1(data):
    data = data.reset_index().copy()
    df = data.pivot_table(index='user_id', columns='item_id', values='ratings', fill_value=-1)

    return df

def add_sensitive_attr_noise(data1, data2, sensitive_attr = 'gender'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sensitive_2 = torch.tensor(data2[sensitive_attr].astype(float).values, device=device)
    user_id_2 = torch.tensor(data2['user_id'].astype(float).values, device=device)
    sensitive_attr_tensor = torch.empty(len(data1), dtype=torch.float32, device=device)
    for i, user_id in tqdm(enumerate(data1.index)):
        gender = sensitive_2[user_id_2 == user_id][0]
        sensitive_attr_tensor[i] = gender
    new_data1 = pd.DataFrame(sensitive_attr_tensor.cpu().numpy(), index=data1.index, columns=[sensitive_attr])
    data1[sensitive_attr] = new_data1[sensitive_attr]
    return data1

def process_data(data_name, know_size,  thre = 3.5, explicit = True, use_thre = False,
        test_inter_num = 10, test_propotion = 0.1):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if data_name == 'ml-1m':
        data = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/ml-1m-50core/ml-1m.h5')
        genders = data['gender']
        data.drop(columns = ['timestamp', 'age', 'gender', 'occupation', 'time'], inplace = True)
    elif data_name == 'tenrec':
        data = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/tenrec-50core/tenrec.h5')
        genders = data['gender']
        data.drop(columns = ['gender'], inplace = True)

    all_users = np.array(data['user_id'].unique()).tolist()
    random.shuffle(all_users)
        # np.save(file_name, np.array(all_users))
    sensitive_attributes = []
    for user in all_users:
        sensitive_attribute = genders[data[data['user_id'] == user].index[0]]
        sensitive_attributes.append(sensitive_attribute)
    sensitive_attribute_0_num = len(sensitive_attributes) - sum(sensitive_attributes)
    sensitive_attribute_1_num = sum(sensitive_attributes)
    mark_0 = 0
    mark_1 = 0
    train_label_users = []
    for user,sensitive_attribute in zip(all_users, sensitive_attributes):
        if sensitive_attribute == 0 and mark_0 < int(sensitive_attribute_0_num * know_size):
            mark_0 += 1
            train_label_users.append(user)
        if sensitive_attribute == 1 and mark_1 < int(sensitive_attribute_1_num * know_size):
            mark_1 += 1
            train_label_users.append(user)
    train_nolabel_users = [user for user in all_users if user not in train_label_users]

    val_users = all_users[:]# [:int(n_users * (0.5 + 0.5 * know_size))]
    data_groupby_users = dict(list(data.groupby(['user_id'])))
    val_data_list = []
    test_data_list = []
    print('begin sample')
    for user in tqdm(val_users):    
        try:
            user_data = data_groupby_users[user]
        except:  
            user_data = data_groupby_users[(user,)]
        user_data_1 = user_data['ratings'] > thre
        user_data_0 = user_data['ratings'] <= thre
        pos_num_user = user_data[user_data_1].shape[0]
        neg_num_user = user_data[user_data_0].shape[0]
        pos_num_sample = int(pos_num_user * test_propotion * 2)
        neg_num_sample = int(neg_num_user * test_propotion * 2)
        pos_data = user_data[user_data_1].sample(n=pos_num_sample, replace=False)
        neg_data = user_data[user_data_0].sample(n=neg_num_sample, replace=False)
        pos_data_1 = pos_data.sample(frac=0.5)
        pos_data_2 = pos_data.drop(pos_data_1.index)
        neg_data_1 = neg_data.sample(frac=0.5)
        neg_data_2 = neg_data.drop(neg_data_1.index)
        val_user_data_sample = pd.concat([pos_data_1, neg_data_1])
        test_user_data_sample = pd.concat([pos_data_2, neg_data_2])
        val_data_list.append(val_user_data_sample)
        test_data_list.append(test_user_data_sample) 

    val_data = pd.concat(val_data_list)
    test_data = pd.concat(test_data_list)
    print('sample finished')
    # generate test data
    no_val_data = data.loc[~data.index.isin(val_data.index.tolist()),]   


    train_data = no_val_data.loc[~no_val_data.index.isin(test_data.index.tolist()),] 
    train_label_data = train_data.loc[train_data['user_id'].isin(train_label_users),]
    train_nolabel_data = train_data.loc[train_data['user_id'].isin(train_nolabel_users),]
    ##############
    if (not explicit) or (explicit and use_thre):
        test_data['ratings'] = np.where(test_data['ratings'] > thre, 1, 0)
        val_data['ratings'] = np.where(val_data['ratings'] > thre, 1, 0)
        train_label_data['ratings'] = np.where(train_label_data['ratings'] > thre, 1, 0)
        train_nolabel_data['ratings'] = np.where(train_nolabel_data['ratings'] > thre, 1, 0)
    return train_data, train_label_data, train_nolabel_data, val_data, test_data, data, train_label_users, train_nolabel_users

class MyDataset_for_demoinference():
    def __init__(self,data_name,k,train_data = None, val_data = None, test_data = None, data = None,):
        warnings.filterwarnings('ignore')
        self.data_name = data_name

        if data is None:
            feedback_data = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/{}-50core/{}.h5'.format(data_name,data_name))
            df = feedback_data.pivot_table(index='user_id', columns='item_id', values='ratings', fill_value=0)
            df['user_id'] = df.index
            df.index.name = 'user'
            data = df
        else:
            self.data = data
        self.val_data = val_data
        self.train_data = train_data
        self.test_data = test_data
        bad_features = ['user_id', 'age', 'occupation', 'zip_code', 'gender', 'time', 'timestamp']
        self.sparse_features = [feature for feature in self.data.columns if feature not in bad_features]
        self.sparse_feat_num = len(self.sparse_features)
        self.target = ['gender']
        self.fixlen_feature_columns = [SparseFeat(feat, np.max(self.data[feat]) + 1, embedding_dim=k) # user / item begin from 1 
                            for feat in self.sparse_features]
        self.varlen_feature_columns = []
        self.model_input = self.get_model_input(train_data)
        self.val_model_input = self.get_model_input(val_data)
        self.test_model_input = self.get_model_input(test_data)

    def get_model_input(self, data):
        if data is None:
            return None
        model_input = data.loc[:,self.sparse_features]
        return model_input

class Dataset_for_MF():
    def __init__(self,data_name,k,train_data = None, val_data = None, test_data = None, data = None):
        self.data_name = data_name
        warnings.filterwarnings('ignore')
        if data_name == 'ml-100k':
            if data is None:
                data = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/ml-100k-50core/ml-100k.h5')
            datamat = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/ml-100k-50core/ml-100k-intermat.h5')
            datamat.drop(columns=['age', 'gender', 'occupation', 'zip_code'], inplace  = True)
            self.train_data = self.add_sensitive_for_true(train_data)
            self.val_data = self.add_sensitive_for_true(val_data)
            self.test_data = self.add_sensitive_for_true(test_data)
            self.data = data
            self.datamat = datamat
            bad_features = ['age', 'occupation', 'zip_code','gender', 'ratings']
            self.sparse_features = [feature for feature in self.data.columns if feature not in bad_features]
            self.sparse_feat_num = len(self.sparse_features)
            self.target = ['ratings']
            self.fixlen_feature_columns = [SparseFeat(feat, np.max(self.data[feat]) + 1, embedding_dim=k) # user / item begin from 1 
                              for feat in self.sparse_features]
            self.varlen_feature_columns = []
            self.train_target = self.train_data[self.target].values
            self.model_input = self.get_model_input(self.train_data)
            self.val_model_input = self.get_model_input(self.val_data, mode = 'test')
            self.test_model_input = self.get_model_input(self.test_data, mode = 'test')
            self.user_feature_column = [SparseFeat('user_id', np.max(self.data['user_id']) + 1, embedding_dim=1) ]
            self.user_group_distribution = self.generate_user_group_distribution(self.train_data, np.max(self.data['user_id'] + 1))



    def get_model_input(self, data, mode = 'train'):
        if data is None:
            return None
        elif mode == 'train':
            model_input = {name: data[name] for name in self.sparse_features}  
        elif mode == 'test': 
            model_input = data
        return model_input
    
    def generate_user_group_distribution(self, train_data, total_user_num, label_weight = 1, label_users = None):

        sensitive_groups_dict = dict(list(train_data.groupby('sensitive_attribute')))
        group_distribution_dict = {}
        if label_users is None:
            for key, group_df in sensitive_groups_dict.items():
                users_in_group = np.array(group_df['user_id'].unique()).tolist()
                prob = 1 / group_df.shape[0] # len(users_in_group)
                emp_probs = [prob if user in users_in_group else 0 for user in range(total_user_num)]
                group_distribution_dict[str(int(key))] = np.array(emp_probs)
        else:
            for key, group_df in sensitive_groups_dict.items():
                users_in_group = np.array(group_df['user_id'].unique()).tolist()
                num_label = group_df[group_df['user_id'].isin(label_users)].shape[0]
                num_nolabel = group_df.shape[0] - num_label
                weight_nolabel = 1 / (label_weight * num_label + num_nolabel)
                weight_label = label_weight / (label_weight * num_label + num_nolabel)
                emp_probs = []
                for user in range(total_user_num):
                    if user in users_in_group:
                        if user in label_users:
                            emp_probs.append(weight_label)
                        else:
                            emp_probs.append(weight_nolabel)
                    else:
                        emp_probs.append(0)
                # emp_probs = [prob if user in users_in_group else 0 for user in range(total_user_num)]
                group_distribution_dict[str(int(key))] = np.array(emp_probs)
        return group_distribution_dict

    def generate_user_group_distribution_0421(self, train_data, total_user_num, generate_users):

        if len(generate_users):
            generate_user_train_data = train_data.loc[train_data['user_id'].isin(generate_users)]
            sensitive_groups_dict = dict(list(generate_user_train_data.groupby('sensitive_attribute')))
            group_distribution_dict = {}
            for key, group_df in sensitive_groups_dict.items():
                if group_df.shape[0] != 0:
                    users_in_group = np.array(group_df['user_id'].unique()).tolist()
                    prob = 1 / group_df.shape[0] # len(users_in_group)
                    emp_probs = [prob if user in users_in_group else 0 for user in range(total_user_num)]
                    group_distribution_dict[str(int(key))] = np.array(emp_probs)
                else:
                    users_in_group = np.array(group_df['user_id'].unique()).tolist()
                    prob = 0 # len(users_in_group)
                    emp_probs = [prob if user in users_in_group else 0 for user in range(total_user_num)]
                    group_distribution_dict[str(int(key))] = np.array(emp_probs)
        else:
            group_distribution_dict = {}
            emp_probs = [0 for user in range(total_user_num)]
            group_distribution_dict[str(0)] = np.array(emp_probs)
            group_distribution_dict[str(1)] = np.array(emp_probs)
        return group_distribution_dict

    def add_sensitive_for_true(self,  df, sensitive_attribute = 'gender', sensitive_mat = None):
            # only for test now
        if sensitive_mat is None:
            sensitive_mat = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/{}-50core/{}.h5'.format(self.data_name, self.data_name))
        df['sensitive_attribute'] = 0
        user_ids = df['user_id'].unique().tolist()
        data1_user_column = torch.tensor(df['user_id'].values).float().to(self.device)
        data1_sensitive_column = torch.tensor(df['sensitive_attribute'].values).float().to(self.device)
        data2_user_column = torch.tensor(sensitive_mat['user_id'].values).float().to(self.device)
        data2_sensitive_column = torch.tensor(sensitive_mat[sensitive_attribute].astype('int32').values).float().to(self.device)
        for user_id in user_ids:
            index = (data1_user_column == user_id).nonzero(as_tuple = False)
            data1_sensitive_column[index] = data2_sensitive_column[(data2_user_column == user_id).nonzero(as_tuple = False)][0]
        df.loc[:,'sensitive_attribute'] = data1_sensitive_column.cpu().numpy()
        return df
    
    def add_sensitive_for_true_validation(self,  df, sensitive_attribute = 'gender', sensitive_mat = None):
            # only for test now
        if sensitive_mat is None:
            sensitive_mat = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/{}-50core/{}.h5'.format(self.data_name, self.data_name))
        
        df['sensitive_attribute_true'] = 0 # pd.Series()
        user_ids = df['user_id'].unique().tolist()
        data1_user_column = torch.tensor(df['user_id'].values).float().to(self.device)
        data1_sensitive_column = torch.tensor(df['sensitive_attribute_true'].values).float().to(self.device)
        data2_user_column = torch.tensor(sensitive_mat['user_id'].values).float().to(self.device)
        data2_sensitive_column = torch.tensor(sensitive_mat[sensitive_attribute].astype('int32').values).float().to(self.device)
        # .nonzero(as_tuple=False)是什么用处
        for user_id in user_ids:
            index = (data1_user_column == user_id).nonzero(as_tuple = False)
            data1_sensitive_column[index] = data2_sensitive_column[(data2_user_column == user_id).nonzero(as_tuple = False)][0]

        df.loc[:,'sensitive_attribute_true'] = data1_sensitive_column.cpu().numpy()
        return df



class Dataset_for_MF_with_reconstructed(Dataset_for_MF):
    def __init__(self, data_name, k, know_size,  data = None, seed = 2000, add_sensitive = 'noisy', thre = 3.5, explicit = True, use_thre = False,
        test_inter_num = 10, test_propotion = 0.1, noisy_label_path = None, sample = False,  add_tune_sensitive = 'True',
        device = 'cuda:0', label_weight = 1, cgl = False):
        self.device = device
        self.cgl = cgl

        save_path = '/data/shith/dataset/dataset_for_partial_fairness/{}-50core'.format(data_name)
        file_name = 'data_{}_knowsize_{}_seed_{}_isexplicit_{}_testinternum_{}_testpropotion_{}_thre_{}_usethre_{}_sample_{}'\
            .format(data_name, str(know_size), str(seed), str(explicit), str(test_inter_num),
              str(test_propotion), str(thre), str(use_thre), str(sample) )

        train_label_file_name = os.path.join(save_path, file_name + '_train_label.h5')
        train_nolabel_file_name = os.path.join(save_path, file_name + '_train_nolabel.h5')
        valid_file_name = os.path.join(save_path, file_name + '_valid.h5')
        test_file_name = os.path.join(save_path, file_name + '_test.h5')
        self.data_name = data_name
        warnings.filterwarnings('ignore')

        # set test_sample_num
        if explicit == True:
            train_sample_per_pos = 0
            test_sample = 0
        else:
            train_sample_per_pos = 4
            test_sample = 100

        # process data
        begin_time =  time.time()
        train_data, train_label_data, train_nolabel_data, val_data, test_data, data, train_label_users, train_nolabel_users = process_data(data_name,
            know_size, thre, explicit, use_thre, test_inter_num, test_propotion)
        print('process data costs', time.time() - begin_time)
        begin_time = time.time()

        feedback_data = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/{}-50core/{}.h5'.format(data_name,data_name))
        df = feedback_data.pivot_table(index='user_id', columns='item_id', values='ratings', fill_value=0)
        df['user_id'] = df.index
        df.index.name = 'user'
        datamat = df 
        datamat_copy = feedback_data # datamat.copy()
        datamat.drop(columns=[col for col in datamat.columns if col in ['age', 'gender', 'occupation', 'time', 'zip_code', 'timestamp'] ],
            inplace  = True)
        datamat_copy.drop(columns = [col for col in datamat_copy.columns if (col!='user_id') and (col!='gender')], inplace=True)

        if (not explicit) or (explicit and use_thre):
            cols = [col for col in datamat.columns if col != 'user_id']

            datamat.loc[:, cols] = np.where(datamat.loc[:, cols] >= 3.5, 1, 0)
        print('load datamat costs', time.time() - begin_time)
        begin_time = time.time()
        
        if os.path.exists(train_label_file_name):
            train_label_data = pd.read_hdf(train_label_file_name)
        else:    
            train_label_data = self.add_sensitive_for_true(train_label_data)
            train_label_data.to_hdf(train_label_file_name, key = '1')
        if os.path.exists(train_nolabel_file_name):
            train_nolabel_data = pd.read_hdf(train_nolabel_file_name)
        else:    
            train_nolabel_data = self.add_sensitive_for_true(train_nolabel_data)
            train_nolabel_data.to_hdf(train_nolabel_file_name, key = '1')

        # sample data for CF
        predict_label_data = pd.read_csv(noisy_label_path)
        if add_sensitive == 'noisy':
            predict_label_data = pd.read_csv(noisy_label_path)
            train_nolabel_data = self.add_sensitive_attr(train_nolabel_data, predict_label_data, 'gender', cgl = cgl)
        elif add_sensitive == 'True':
            train_nolabel_data = self.add_sensitive_for_true(train_nolabel_data, sensitive_mat = datamat_copy)
        elif add_sensitive == 'fake':
            train_nolabel_data = self.add_fake_sensitive_attr(train_nolabel_data)
        print('sampel train nolabel data costs', time.time() - begin_time)
        begin_time = time.time()


        train_data_list = [train_label_data, train_nolabel_data]
        self.train_data = pd.concat(train_data_list)
            

        print('load val data costs', time.time() - begin_time)
        begin_time = time.time()
        if add_tune_sensitive == 'True' or add_tune_sensitive == 'true':
            self.val_data = self.add_sensitive_for_true(val_data, sensitive_mat = datamat_copy)

        elif add_tune_sensitive == 'fake':
            val_data_label = val_data.loc[val_data['user_id'].isin(train_label_users)]
            val_data_nolabel = val_data.loc[val_data['user_id'].isin(train_nolabel_users)]
            # self.val_data = self.add_sensitive_attr(val_data_label, datamat_copy, 'gender').append(self.add_fake_sensitive_attr(val_data_nolabel))
            val_data_label_sensitive = self.add_sensitive_attr(val_data_label, datamat_copy, 'gender')
            val_data_nolabel_fake_sensitive = self.add_fake_sensitive_attr(val_data_nolabel)
            self.val_data = pd.concat([val_data_label_sensitive, val_data_nolabel_fake_sensitive])        

        self.val_data = self.add_sensitive_for_true_validation(self.val_data, sensitive_mat = datamat_copy)
        
        print('process val data costs', time.time() - begin_time)
        begin_time = time.time()
            
        self.test_data = self.add_sensitive_for_true(test_data, sensitive_mat = datamat_copy)
        self.test_data = self.add_sensitive_for_true_validation(self.test_data, sensitive_mat = datamat_copy)

        print('sample test label data costs', time.time() - begin_time)
        begin_time = time.time()
        self.data = data
        self.datamat = datamat
        bad_features = ['age', 'occupation', 'zip_code','gender', 'ratings', 'time', 'timestamp']
        self.sparse_features = [feature for feature in self.data.columns if feature not in bad_features]
        self.sparse_feat_num = len(self.sparse_features)
        self.target = ['ratings']
        self.fixlen_feature_columns = [SparseFeat(feat, np.max(self.data[feat]) + 1, embedding_dim=k) # user / item begin from 1 
                            for feat in self.sparse_features]
        self.varlen_feature_columns = []
        self.train_target = self.train_data[self.target].values
        self.model_input = self.get_model_input(self.train_data)
        self.val_model_input = self.get_model_input(self.val_data, mode = 'test')
        self.test_model_input = self.get_model_input(self.test_data, mode = 'test')
        self.user_feature_column = [SparseFeat('user_id', np.max(self.data['user_id']) + 1, embedding_dim=1) ]
        self.user_group_distribution = self.generate_user_group_distribution(self.train_data, np.max(self.data['user_id'] + 1), label_weight = label_weight,)
        self.train_label_users = train_label_users
        self.train_nolabel_users = train_nolabel_users
        print('process rest costs', time.time() - begin_time)
        begin_time = time.time()
    
    def add_sensitive_attr(self, data1, data2, sensitive_attr = 'gender', cgl = False):
        data1['sensitive_attribute'] = 0
        data2 = data2.copy()
        user_ids = data1['user_id'].unique().tolist()
        user_ids = data1['user_id'].unique().tolist()

        data1_user_column = torch.tensor(data1['user_id'].values).float().to(self.device)
        data1_sensitive_column = torch.tensor(data1['sensitive_attribute'].values).float().to(self.device)
        
        data2_user_column = torch.tensor(data2['user_id'].values).float().to(self.device)
        if not cgl:
            data2_sensitive_column = torch.tensor(data2[sensitive_attr].astype('int32').values).float().to(self.device)
        else:
            data2_sensitive_column = torch.tensor(data2[sensitive_attr + '_cgl'].astype('int32').values).float().to(self.device)
        for user_id in user_ids:
            index = (data1_user_column == user_id).nonzero(as_tuple = False)
            data1_sensitive_column[index] = data2_sensitive_column[(data2_user_column == user_id).nonzero(as_tuple = False)][0]

        data1.loc[:,'sensitive_attribute'] = data1_sensitive_column.cpu().numpy()
        return data1



    def add_fake_sensitive_attr(self, data1):
        data1['sensitive_attribute'] = 1000
        return data1

    def get_model_input(self, data, mode = 'train'):
        # if(self.data_name == 'ml-100k'):
        if data is None:
            return None
        elif mode == 'train':
            model_input = {name: data[name] for name in self.sparse_features}  
            # get sensitive attribute
            model_input['sensitive_attribute'] = data['sensitive_attribute']
        elif mode == 'test': 
            model_input = data
        return model_input





class Dataset_DRFO(Dataset_for_MF_with_reconstructed):
    def __init__(self, data_name, k, know_size,  data = None, seed = 2000, add_sensitive = 'noisy', thre = 3.5, explicit = True, use_thre = False,
        test_inter_num = 10, test_propotion = 0.1, noisy_label_path = None, sample = False, add_tune_sensitive = 'True',
        device = 'cuda:0', label_weight = 1, cgl = False, ):
        self.device = device
        self.cgl = cgl

        save_path = '/data/shith/dataset/dataset_for_partial_fairness/{}-50core'.format(data_name)
        file_name = 'data_{}_knowsize_{}_seed_{}_isexplicit_{}_testinternum_{}_testpropotion_{}_thre_{}_usethre_{}_sample_{}'\
            .format(data_name, str(know_size), str(seed), str(explicit), str(test_inter_num),
              str(test_propotion), str(thre), str(use_thre), str(sample) )

        train_label_file_name = os.path.join(save_path, file_name + '_train_label.h5')
        train_nolabel_file_name = os.path.join(save_path, file_name + '_train_nolabel.h5')
        valid_file_name = os.path.join(save_path, file_name + '_valid.h5')
        test_file_name = os.path.join(save_path, file_name + '_test.h5')
        self.data_name = data_name
        warnings.filterwarnings('ignore')

        # set test_sample_num
        if explicit == True:
            train_sample_per_pos = 0
            test_sample = 0
        else:
            train_sample_per_pos = 4
            test_sample = 100

        # process data
        begin_time =  time.time()
        train_data, train_label_data, train_nolabel_data, val_data, test_data, data, train_label_users, train_nolabel_users = process_data(data_name,
            know_size, thre, explicit, use_thre, test_inter_num, test_propotion)
        print('process data costs', time.time() - begin_time)
        begin_time = time.time()

        feedback_data = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/{}-50core/{}.h5'.format(data_name,data_name))
        df = feedback_data.pivot_table(index='user_id', columns='item_id', values='ratings', fill_value=0)
        df['user_id'] = df.index
        df.index.name = 'user'
        datamat = df 
        datamat_copy = feedback_data # datamat.copy()
        datamat.drop(columns=[col for col in datamat.columns if col in ['age', 'gender', 'occupation', 'time', 'zip_code', 'timestamp'] ],
            inplace  = True)
        datamat_copy.drop(columns = [col for col in datamat_copy.columns if (col!='user_id') and (col!='gender')], inplace=True)

        if (not explicit) or (explicit and use_thre):
            cols = [col for col in datamat.columns if col != 'user_id']

            datamat.loc[:, cols] = np.where(datamat.loc[:, cols] >= 3.5, 1, 0)
        print('load datamat costs', time.time() - begin_time)
        begin_time = time.time()
        
        if os.path.exists(train_label_file_name):
            train_label_data = pd.read_hdf(train_label_file_name)
        else:    
            train_label_data = self.add_sensitive_for_true(train_label_data)
            train_label_data.to_hdf(train_label_file_name, key = '1')

        # sample data for CF
        predict_label_data = pd.read_csv(noisy_label_path)
        unknown_data_true = self.add_sensitive_for_true(train_nolabel_data.copy(), sensitive_mat = datamat_copy.copy())
        if add_sensitive == 'noisy':
            predict_label_data = pd.read_csv(noisy_label_path)
            train_nolabel_data = self.add_sensitive_attr(train_nolabel_data, predict_label_data, 'gender', cgl = cgl)
        elif add_sensitive == 'True':
            train_nolabel_data = self.add_sensitive_for_true(train_nolabel_data, sensitive_mat = datamat_copy)
        elif add_sensitive == 'fake':
            train_nolabel_data = self.add_fake_sensitive_attr(train_nolabel_data)
        print('sampel train nolabel data costs', time.time() - begin_time)
        self.train_data = pd.concat([train_label_data, train_nolabel_data])


        begin_time = time.time()
        if add_tune_sensitive == 'True' or add_tune_sensitive == 'true':
            self.val_data = self.add_sensitive_for_true(val_data, sensitive_mat = datamat_copy)

        elif add_tune_sensitive == 'fake':
            val_data_label = val_data.loc[val_data['user_id'].isin(train_label_users)]
            val_data_nolabel = val_data.loc[val_data['user_id'].isin(train_nolabel_users)]
            val_data_label_sensitive = self.add_sensitive_attr(val_data_label, datamat_copy, 'gender')
            val_data_nolabel_fake_sensitive = self.add_fake_sensitive_attr(val_data_nolabel)
            self.val_data = pd.concat([val_data_label_sensitive, val_data_nolabel_fake_sensitive])
        self.val_data = self.add_sensitive_for_true_validation(self.val_data, sensitive_mat = datamat_copy)
        
        print('process val data costs', time.time() - begin_time)
        begin_time = time.time()
            
        self.test_data = self.add_sensitive_for_true(test_data, sensitive_mat = datamat_copy)
        self.test_data = self.add_sensitive_for_true_validation(self.test_data, sensitive_mat = datamat_copy)


        print('sample test label data costs', time.time() - begin_time)
        begin_time = time.time()
        self.data = data
        self.datamat = datamat
        bad_features = ['age', 'occupation', 'zip_code','gender', 'ratings', 'time', 'timestamp']
        self.sparse_features = [feature for feature in self.data.columns if feature not in bad_features]
        self.sparse_feat_num = len(self.sparse_features)
        self.target = ['ratings']
        self.fixlen_feature_columns = [SparseFeat(feat, np.max(self.data[feat]) + 1, embedding_dim=k) # user / item begin from 1 
                            for feat in self.sparse_features]
        self.varlen_feature_columns = []
        self.train_target = self.train_data[self.target].values
        self.model_input = self.get_model_input(self.train_data)
        self.val_model_input = self.get_model_input(self.val_data, mode = 'test')
        self.test_model_input = self.get_model_input(self.test_data, mode = 'test')
        self.user_feature_column = [SparseFeat('user_id', np.max(self.data['user_id']) + 1, embedding_dim=1) ]
        self.user_group_distribution = self.generate_user_group_distribution_0421(self.train_data, np.max(self.data['user_id'] + 1), generate_users = train_nolabel_users)
        self.train_label_users = train_label_users
        self.train_nolabel_users = train_nolabel_users
        
        know_data = self.train_data.loc[self.train_data['user_id'].isin(train_label_users)]
        know_0 = know_data[know_data['sensitive_attribute'] == 0].shape[0]
        know_1 = know_data[know_data['sensitive_attribute'] == 1].shape[0]
        unknow_data = unknown_data_true
        unknow_0 = unknow_data[unknow_data['sensitive_attribute'] == 0].shape[0] # int(unknow_data.shape[0] * (1 - instance_prob_1)) # unknow_data[unknow_data['sensitive_attribute'] == 0]
        unknow_1 = unknow_data[unknow_data['sensitive_attribute'] == 1].shape[0] #  int(unknow_data.shape[0] * instance_prob_1) # unknow_data[unknow_data['sensitive_attribute'] == 1]


        self.etas = {
            'know_0': know_0 / (know_0 + unknow_0),
            'know_1': know_1 / (know_1 + unknow_1), 
            'unknow_0': unknow_0 / (know_0 + unknow_0),
            'unknow_1': unknow_1/ (know_1 + unknow_1),
        }

class Dataset_DRFO_extension(Dataset_for_MF_with_reconstructed):
    def __init__(self, data_name, k, know_size,  data = None, seed = 2000, add_sensitive = 'noisy', thre = 3.5, explicit = True, use_thre = False,
        test_inter_num = 10, test_propotion = 0.1, noisy_label_path = None, sample = False, add_tune_sensitive = 'True',
        device = 'cuda:0', cgl = False, dro = False, lack_profile_prob = 0.2):
        self.device = device
        self.cgl = cgl
        save_path = '/data/shith/dataset/dataset_for_partial_fairness/{}-50core'.format(data_name)
        file_name = 'data_{}_knowsize_{}_seed_{}_isexplicit_{}_testinternum_{}_testpropotion_{}_thre_{}_usethre_{}_sample_{}'\
            .format(data_name, str(know_size), str(seed), str(explicit), str(test_inter_num),
              str(test_propotion), str(thre), str(use_thre),str(sample) )

        train_label_file_name = os.path.join(save_path, file_name + '_train_label.h5')
        train_nolabel_file_name = os.path.join(save_path, file_name + '_train_nolabel.h5')
        valid_file_name = os.path.join(save_path, file_name + '_valid.h5')
        test_file_name = os.path.join(save_path, file_name + '_test.h5')
        self.data_name = data_name
        warnings.filterwarnings('ignore')

        # set test_sample_num
        if explicit == True:
            train_sample_per_pos = 0
            test_sample = 0
        else:
            train_sample_per_pos = 4
            test_sample = 100

        # process data
        begin_time =  time.time()
        train_data, train_label_data, train_nolabel_data, val_data, test_data, data, train_label_users, train_nolabel_users = process_data(data_name,
            know_size, thre, explicit, use_thre, test_inter_num, test_propotion )
        print('process data costs', time.time() - begin_time)
        begin_time = time.time()

        feedback_data = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/{}-50core/{}.h5'.format(data_name,data_name))
        df = feedback_data.pivot_table(index='user_id', columns='item_id', values='ratings', fill_value=0)
        df['user_id'] = df.index
        df.index.name = 'user'
        datamat = df 
        datamat_copy = feedback_data # datamat.copy()
        datamat.drop(columns=[col for col in datamat.columns if col in ['age', 'gender', 'occupation', 'time', 'zip_code', 'timestamp'] ],
            inplace  = True)
        datamat_copy.drop(columns = [col for col in datamat_copy.columns if (col!='user_id') and (col!='gender')], inplace=True)

        if (not explicit) or (explicit and use_thre):
            cols = [col for col in datamat.columns if col != 'user_id']

            datamat.loc[:, cols] = np.where(datamat.loc[:, cols] >= 3.5, 1, 0)
        print('load datamat costs', time.time() - begin_time)
        begin_time = time.time()
        
        train_label_data = self.add_sensitive_for_true(train_label_data)


        # sample data for CF
        predict_label_data = pd.read_csv(noisy_label_path)
        predict_sensitive_df = predict_label_data.copy()
        predict_users = predict_sensitive_df['user_id'].squeeze().unique().tolist()
        predict_users_sensitive_true = []
        user_gender_df = datamat_copy[["user_id", "gender"]]
        user_gender_df.drop_duplicates(inplace=True)
        user_gender_df.set_index("user_id", inplace=True)
        # 将DataFrame转换为字典
        user_gender_dict = user_gender_df.to_dict()["gender"]
        predict_users_sensitive_true = [user_gender_dict.get(user_id) for user_id in predict_users]
        predict_data_genders_1_num = sum(predict_users_sensitive_true)
        predict_data_genders_0_num = len(predict_users_sensitive_true) - sum(predict_users_sensitive_true)
        # 按顺序分层丢弃
        lack_profile_0_num = int(lack_profile_prob * predict_data_genders_0_num)
        lack_profile_1_num = int(lack_profile_prob * predict_data_genders_1_num)
        lack_profile_users = []
        mark_0 = 0
        mark_1 = 0
        for i, user in enumerate(predict_users):
            if predict_users_sensitive_true[i] == 0 and mark_0 <= lack_profile_0_num:
                lack_profile_users.append(user)
                mark_0 += 1
            if predict_users_sensitive_true[i] == 1 and mark_1 <= lack_profile_1_num:
                lack_profile_users.append(user)
                mark_1 += 1
        
        # get train_label_user_genders:
        user_gender_df = train_label_data[["user_id", "sensitive_attribute"]]
        user_gender_df.drop_duplicates(inplace=True)
        user_gender_df.set_index("user_id", inplace=True)
        user_gender_dict = user_gender_df.to_dict()["sensitive_attribute"]
        label_users_genders = [user_gender_dict.get(user_id) for user_id in train_label_users]
        label_p = sum(label_users_genders) / len(label_users_genders)
        self.label_p = label_p

        lack_profile_user_gender_dict = {}
        
        if lack_profile_prob != 0:
            random.seed(seed)
            for user in lack_profile_users:
                if random.uniform(0,1) <= label_p:
                    lack_profile_user_gender_dict[user] = 1
                else:
                    lack_profile_user_gender_dict[user] = 0
            print(lack_profile_user_gender_dict)
            if cgl:
                lack_profile_user_gender_series = pd.Series(lack_profile_user_gender_dict)
                predict_sensitive_df['gender_cgl'] = predict_sensitive_df['user_id'].map(lack_profile_user_gender_series)\
                    .fillna(predict_sensitive_df['gender_cgl'])
                # predict_sensitive_df.loc[predict_sensitive_df['user_id'].isin(lack), 'gender_cgl']
            if dro:
                lack_profile_user_gender_series = pd.Series(lack_profile_user_gender_dict)
                predict_sensitive_df['gender'] = predict_sensitive_df['user_id'].map(lack_profile_user_gender_series)\
                    .fillna(predict_sensitive_df['gender'])
            if not dro and not cgl:
                mask = predict_sensitive_df['user_id'].isin(lack_profile_user_gender_dict.keys())
                predict_sensitive_df.loc[mask, 'gender'] = 1000
                print(predict_sensitive_df.loc[mask, 'gender'])
                
        train_nolabel_users_lack_profile = list(lack_profile_user_gender_dict.keys())
        train_nolabel_users_profile = [user for user in train_nolabel_users if user not in train_nolabel_users_lack_profile]
        predict_label_data = predict_sensitive_df.copy()
        if add_sensitive == 'noisy':
            train_nolabel_data = self.add_sensitive_attr(train_nolabel_data, predict_label_data, 'gender', cgl = cgl)
        elif add_sensitive == 'True':
            train_nolabel_data = self.add_sensitive_for_true(train_nolabel_data, sensitive_mat = datamat_copy)# self.ng_sample(train_nolabel_data, datamat, train_sample_per_pos, mode = 'train',)
        elif add_sensitive == 'fake':
            train_nolabel_data = self.add_fake_sensitive_attr(train_nolabel_data)
        print(train_nolabel_data)
        print('sampel train nolabel data costs', time.time() - begin_time)
        begin_time = time.time()
        self.train_data = pd.concat([train_label_data, train_nolabel_data])


        print('load val data costs', time.time() - begin_time)
        begin_time = time.time()
        if add_tune_sensitive == 'True' or add_tune_sensitive == 'true':
            self.val_data = self.add_sensitive_for_true(val_data, sensitive_mat = datamat_copy)

        elif add_tune_sensitive == 'fake':
            val_data_label = val_data.loc[val_data['user_id'].isin(train_label_users)]
            val_data_nolabel = val_data.loc[val_data['user_id'].isin(train_nolabel_users)]
            val_data_label_sensitive = self.add_sensitive_attr(val_data_label, datamat_copy, 'gender')
            val_data_nolabel_fake_sensitive = self.add_fake_sensitive_attr(val_data_nolabel)

            self.val_data = pd.concat([val_data_label_sensitive, val_data_nolabel_fake_sensitive])        
        
        self.val_data = self.add_sensitive_for_true_validation(self.val_data, sensitive_mat = datamat_copy)

        # self.val_data = self.add_sensitive_for_true_validation(self.val_data, sensitive_mat = datamat_copy)
        print('process val data costs', time.time() - begin_time)
        begin_time = time.time()

        self.test_data = self.add_sensitive_for_true(test_data, sensitive_mat = datamat_copy)
        self.test_data = self.add_sensitive_for_true_validation(self.test_data, sensitive_mat = datamat_copy)

        print('sample test label data costs', time.time() - begin_time)
        begin_time = time.time()
        self.data = data
        self.datamat = datamat
        bad_features = ['age', 'occupation', 'zip_code','gender', 'ratings', 'time', 'timestamp']
        self.sparse_features = [feature for feature in self.data.columns if feature not in bad_features]
        self.sparse_feat_num = len(self.sparse_features)
        self.target = ['ratings']
        self.fixlen_feature_columns = [SparseFeat(feat, np.max(self.data[feat]) + 1, embedding_dim=k) # user / item begin from 1 
                            for feat in self.sparse_features]
        self.varlen_feature_columns = []
        self.train_target = self.train_data[self.target].values
        self.model_input = self.get_model_input(self.train_data)
        self.val_model_input = self.get_model_input(self.val_data, mode = 'test')
        self.test_model_input = self.get_model_input(self.test_data, mode = 'test')
        self.user_feature_column = [SparseFeat('user_id', np.max(self.data['user_id']) + 1, embedding_dim=1) ]
        # self.user_group_distribution = self.generate_user_group_distribution(self.train_data, np.max(self.data['user_id'] + 1), label_weight = label_weight,)
        self.user_group_distribution_profile = self.generate_user_group_distribution_0421(self.train_data, np.max(self.data['user_id'] + 1), train_nolabel_users_profile)
        self.user_group_distribution_lack_profile = self.generate_user_group_distribution_0421(self.train_data, np.max(self.data['user_id'] + 1), train_nolabel_users_lack_profile)
        self.train_label_users = train_label_users
        self.train_nolabel_users = train_nolabel_users
        self.train_nolabel_users_lack_profile = train_nolabel_users_lack_profile
        self.train_nolabel_users_profile = train_nolabel_users_profile

        know_data = self.train_data.loc[self.train_data['user_id'].isin(train_label_users)]
        know_0 = know_data[know_data['sensitive_attribute'] == 0].shape[0]
        know_1 = know_data[know_data['sensitive_attribute'] == 1].shape[0]
        p_instance_1 = know_1 / know_data.shape[0]
        if len(train_nolabel_users_profile):
            unknow_data_profile = self.train_data.loc[self.train_data['user_id'].isin(train_nolabel_users_profile)]
            unknow_0_profile = int((1 - p_instance_1) * unknow_data_profile.shape[0])
            unknow_1_profile = int(p_instance_1 * unknow_data_profile.shape[0]) # unknow_data_profile[unknow_data_profile['sensitive_attribute'] == 1].shape[0]
        else:
            unknow_0_profile = 0
            unknow_1_profile = 0
        if len(train_nolabel_users_lack_profile):
            unknow_data_lack_profile = self.train_data.loc[self.train_data['user_id'].isin(train_nolabel_users_lack_profile)]
            unknow_0_lack_profile = int((1 - p_instance_1) * unknow_data_lack_profile.shape[0]) # unknow_data_lack_profile[unknow_data_lack_profile['sensitive_attribute'] == 0].shape[0]
            unknow_1_lack_profile = int(p_instance_1 * unknow_data_lack_profile.shape[0])
            # unknow_data_lack_profile[unknow_data_lack_profile['sensitive_attribute'] == 1].shape[0]
        else:
            unknow_0_lack_profile = 0
            unknow_1_lack_profile = 0
        sum_0 = know_0 + unknow_0_profile + unknow_0_lack_profile
        sum_1 = know_1 + unknow_1_profile + unknow_1_lack_profile
        self.etas = {
            'know_0': (know_0) / sum_0,
            'know_1': (know_1) / sum_1, 
            'unknow_0_profile': unknow_0_profile / sum_0,
            'unknow_1_profile': unknow_1_profile / sum_1, 
            'unknow_0_lack_profile': unknow_0_lack_profile/ sum_0,
            'unknow_1_lack_profile': unknow_1_lack_profile / sum_1, 
        }
        self.gammas2 = [know_1 / (know_0 + know_1), know_0 / (know_0 + know_1)]
        
        print('process rest costs', time.time() - begin_time)
        begin_time = time.time()


class Dataset_for_evaluation(Dataset_for_MF_with_reconstructed):
    def __init__(self, data_name, k, know_size,  data = None, seed = 2000, add_sensitive = 'noisy', thre = 3.5, explicit = True, use_thre = False,
        test_inter_num = 10, test_propotion = 0.1, noisy_label_path = None, sample = False,
         device = 'cuda:0', cgl = False):
        self.cgl = cgl
        self.device = device
        save_path = '/data/shith/dataset/dataset_for_partial_fairness/{}-50core'.format(data_name)

        file_name = 'data_{}_knowsize_{}_seed_{}_isexplicit_{}_testinternum_{}_testpropotion_{}_thre_{}_usethre_{}_sample_{}'\
            .format(data_name, str(know_size), str(seed), str(explicit), str(test_inter_num),
              str(test_propotion), str(thre), str(use_thre),str(sample) )
        valid_file_name = os.path.join(save_path, file_name + '_valid.h5')
        test_file_name = os.path.join(save_path, file_name + '_test.h5')
        self.data_name = data_name
        warnings.filterwarnings('ignore')
        if explicit == True:
            train_sample_per_pos = 0
            test_sample = 0
        else:
            train_sample_per_pos = 0
            test_sample = 100
        begin_time = time.time()
        train_data, train_label_data, train_nolabel_data, val_data, test_data, data, train_label_users, train_nolabel_users = process_data(data_name,
            know_size, thre, explicit, use_thre, test_inter_num, test_propotion)

        feedback_data = pd.read_hdf('/data/shith/dataset/dataset_for_partial_fairness/{}-50core/{}.h5'.format(data_name,data_name))
        df = feedback_data.pivot_table(index='user_id', columns='item_id', values='ratings', fill_value=0)
        df['user_id'] = df.index
        df.index.name = 'user'
        datamat = df 
        datamat_copy = feedback_data # datamat.copy()
        datamat.drop(columns=[col for col in datamat.columns if col in ['age', 'gender', 'occupation', 'time', 'zip_code', 'timestamp'] ],
            inplace  = True)

        datamat_copy.drop(columns = [col for col in datamat_copy.columns if (col!='user_id') and (col!='gender')], inplace=True)
        if (not explicit) or (explicit and use_thre):
            cols = [col for col in datamat.columns if col != 'user_id']
            datamat.loc[:, cols] = np.where(datamat.loc[:, cols] >= 3.5, 1, 0)
        train_nolabel_data = self.add_sensitive_for_true(train_nolabel_data)
        train_label_data = self.add_sensitive_for_true(train_label_data)
        # here only train with unknown users to watch
        self.train_data = pd.concat([train_label_data, train_nolabel_data])

        if add_sensitive == "noisy":
            predict_label_data = pd.read_csv(noisy_label_path)
            val_data_label = val_data.loc[val_data['user_id'].isin(train_label_users)]
            val_data_nolabel = val_data.loc[val_data['user_id'].isin(train_nolabel_users)]
            # self.val_data = self.add_sensitive_attr(val_data_label, datamat_copy, 'gender').append(self.add_sensitive_attr(val_data_nolabel, predict_label_data, 'gender', cgl = cgl))
            val_data_label_sensitive = self.add_sensitive_attr(val_data_label, datamat_copy, 'gender')
            val_data_nolabel_sensitive = self.add_sensitive_attr(val_data_nolabel, predict_label_data, 'gender', cgl=cgl)
            self.val_data = pd.concat([val_data_label_sensitive, val_data_nolabel_sensitive])
            test_data_label = test_data.loc[test_data['user_id'].isin(train_label_users)]
            test_data_nolabel = test_data.loc[test_data['user_id'].isin(train_nolabel_users)]
            test_data_label_sensitive = self.add_sensitive_attr(test_data_label, datamat_copy, 'gender')
            test_data_nolabel_sensitive = self.add_sensitive_attr(test_data_nolabel, predict_label_data, 'gender', cgl=cgl)
            self.test_data = pd.concat([test_data_label_sensitive, test_data_nolabel_sensitive])
        
        
        elif add_sensitive == 'true' or add_sensitive == 'True':
            self.val_data = self.add_sensitive_for_true(val_data)
            self.test_data = self.add_sensitive_for_true(test_data)
        elif add_sensitive == 'fake' or add_sensitive == 'partial':
            predict_label_data = pd.read_csv(noisy_label_path)
            val_data_label = val_data.loc[val_data['user_id'].isin(train_label_users)]
            val_data_nolabel = val_data.loc[val_data['user_id'].isin(train_nolabel_users)]
            val_data_label_sensitive = self.add_sensitive_attr(val_data_label, datamat_copy, 'gender')
            val_data_nolabel_fake_sensitive = self.add_fake_sensitive_attr(val_data_nolabel)
            self.val_data = pd.concat([val_data_label_sensitive, val_data_nolabel_fake_sensitive])
            test_data_label = test_data.loc[test_data['user_id'].isin(train_label_users)]
            test_data_nolabel = test_data.loc[test_data['user_id'].isin(train_nolabel_users)]
            test_data_label_sensitive = self.add_sensitive_attr(test_data_label, datamat_copy, 'gender')
            test_data_nolabel_fake_sensitive = self.add_fake_sensitive_attr(test_data_nolabel)
            self.test_data = pd.concat([test_data_label_sensitive, test_data_nolabel_fake_sensitive])

        predict_label_data = pd.read_csv(noisy_label_path)
        self.val_data = self.add_sensitive_for_true_validation(self.val_data)
        self.test_data = self.add_sensitive_for_true_validation(self.test_data)
        self.data = data
        self.datamat = datamat
        bad_features = ['age', 'occupation', 'zip_code','gender', 'ratings', 'time', 'timestamp']
        self.sparse_features = [feature for feature in self.data.columns if feature not in bad_features]
        self.sparse_feat_num = len(self.sparse_features)
        self.target = ['ratings']
        self.fixlen_feature_columns = [SparseFeat(feat, np.max(self.data[feat]) + 1, embedding_dim=k) # user / item begin from 1 
                            for feat in self.sparse_features]
        self.varlen_feature_columns = []
        self.train_target = self.train_data[self.target].values
        self.model_input = self.get_model_input(self.train_data)
        self.val_model_input = self.get_model_input(self.val_data, mode = 'test')
        self.test_model_input = self.get_model_input(self.test_data, mode = 'test')
        self.user_feature_column = [SparseFeat('user_id', np.max(self.data['user_id']) + 1, embedding_dim=1) ]
        self.user_group_distribution = self.generate_user_group_distribution(self.train_data, np.max(self.data['user_id'] + 1))

