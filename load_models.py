from re import T
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
workspace = './workspace'
from Dataset import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from mymodels import *
from mybasemodels import *
import json
from sklearn.metrics import accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"



class new_load_models():
    def __init__(self, data_name, model_name = None, trial_name = None, dir = None,
     test_mode = 'noisy', exp_config = None, dro = False, cgl = False, lack_profile = False, lack_profile_prob = 0):
        """
        data_name is necessary
        you can assign data_name, model_name, trial_name, or only assign the dir to get models
        the dir will be like workspace/data1_name/model_name/trial_name 
        """
        self.lack_profile = lack_profile
        self.lack_profile_prob = lack_profile_prob
        # self.Dataset = MyDataset(data_name, k, test_envs)
        self.data_name = data_name
        checkpoint_dir = os.path.join(workspace, data_name)
        if model_name is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if trial_name is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, trial_name)
        if dir is not None:
            checkpoint_dir = dir
        self.checkpoint_dir = checkpoint_dir
        file_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        model = torch.load(file_path)['model']
        self.model = model
        self.test_mode = test_mode # True / noisy / fake
        self.exp_config = exp_config
        self.dro = dro
        self.cgl = cgl

    def load_test(self, log_name, Dataset, k = 32, batch_size = 8192, test_or_val = 'test', generate_basic_rankings = False, save_path = None,
        noisy_label_path = None, explicit = True, use_thre = False, thre = 3.5, unfairness_metric = 'dp'):
        print('loading best models in validation sets and testing...')
        # file_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        model = self.model
        datamat_path = ''

        if test_or_val == 'test':
            val_data = Dataset.test_data
            val_model_input = Dataset.test_model_input
        else:
            val_data = Dataset.val_data
            val_model_input = Dataset.val_model_input
        target = Dataset.target
        # actually they are test_data,test_model_input...
        try:
            eval_result = model.evaluate(x = val_model_input,y = [],batch_size=batch_size)
        except:
            eval_result = model.evaluate(x = val_model_input,y = [],batch_size=batch_size)


        # save exp_result 
        if 'test' in test_or_val:
            log_dir = os.path.join(workspace, 'test_results', self.data_name)
            log_path = os.path.join(log_dir, log_name + '.json')
        elif 'val' in test_or_val:
            log_dir = os.path.join(workspace, 'val_results', self.data_name)
            log_path = os.path.join(log_dir, log_name + '.json')
        elif 'train' in test_or_val:
            log_dir = os.path.join(workspace, 'train_results', self.data_name)
            log_path = os.path.join(log_dir, log_name + '.json')
        # get configs
        file_path = os.path.join(self.checkpoint_dir, 'best_result.pt')
        configs = torch.load(file_path)['model_config'] 
        print(eval_result)
        print(configs)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if self.test_mode == 'true' or self.test_mode == 'True':
            self.test_mode = 'true'
            file = open(log_path, "w")
            json.dump(eval_result, file)
            file.write('\r\n')
            light_result = {}
            for key,value in eval_result.items():
                if ('discripancy' in key) or ('uabs' in key or 'uval' in key) or ('dp' in key): # or ('var' in key):
                    light_result[key] = value
            json.dump(light_result, file)
            file.write('\r\n')
            model_config = model.model_config_dict
            json.dump(model_config, file)
            file.write('\r\n')
            file.close() 
        elif self.test_mode == 'noisy' or self.test_mode == 'noise':
            self.test_mdoe = 'noisy'
            file = open(log_path, "a")
            light_result = {}
            for key,value in eval_result.items():
                if ('discripancy' in key) or ('uabs' in key or 'uval' in key): # or ('var' in key):
                    light_result[key + '_noisy_label'] = value
            json.dump(light_result, file)
            file.close() 
        elif self.test_mode == 'fake' or self.test_mode == 'partial':
            self.test_mode = 'partial'
            file = open(log_path, "a")
            light_result = {}
            for key,value in eval_result.items():
                if ('discripancy' in key) or ('uabs' in key or 'uval' in key): # or ('var' in key):
                    light_result[key + '_partial_label'] = value
            json.dump(light_result, file)
            file.close() 

        # save result for draw 
        if self.lack_profile == False:
            if self.exp_config is not None:
                if self.dro == False:
                    if self.cgl == False:
                        exp_config = self.exp_config
                        exp_extra_info = 'explicit_{}_test_inter_num_{}_test_propotion_{}_thre_{}_use_thre_{}_use_classifier_stage{}'\
                            .format(str(exp_config['explicit']), str(exp_config["test_inter_num"]),str(exp_config["test_propotion"]), str(exp_config["thre"]),
                            str(exp_config["use_thre"]),  str(exp_config["use_classifier_stage"]))
                        exp_result_dir = os.path.join(workspace, 'unfairness', unfairness_metric, exp_config['data_name'], exp_config['model_name'], 
                        exp_extra_info, str(exp_config['know_size']), str(exp_config['performance_scale']),str(exp_config['seed']),
                        str(exp_config['classifier_stage']), test_or_val + '_results', )
                        exp_result_name = 'train_{}_valid_{}_test_{}.json'.format(exp_config['train_mode'], exp_config['valid_mode'], self.test_mode)
                        exp_result_path = os.path.join(exp_result_dir, exp_result_name)
                        if not os.path.exists(exp_result_dir):
                            os.makedirs(exp_result_dir)
                        file = open(exp_result_path, "w")
                        json.dump(eval_result, file)
                        file.write('\r\n')

                        light_result = {}
                        try:
                            light_result['dp_true'] = eval_result['dp_true']
                        except:
                            light_result = {}
                        
                        json.dump(light_result, file)
                        file.write('\r\n')
                        
                        model_config = model.model_config_dict
                        json.dump(model_config, file)
                        file.close()
                    else:
                        exp_config = self.exp_config
                        exp_extra_info = 'explicit_{}_test_inter_num_{}_test_propotion_{}_thre_{}_use_thre_{}_use_classifier_stage{}'\
                            .format(str(exp_config['explicit']), str(exp_config["test_inter_num"]),str(exp_config["test_propotion"]), str(exp_config["thre"]),
                            str(exp_config["use_thre"]), str(exp_config["use_classifier_stage"]))
                        exp_result_dir = os.path.join(workspace, 'unfairness', unfairness_metric, exp_config['data_name'], exp_config['model_name'], 
                        exp_extra_info, str(exp_config['know_size']), str(exp_config['performance_scale']),str(exp_config['seed']),
                        str(exp_config['classifier_stage']), test_or_val + '_results', )
                        exp_result_name = 'train_{}_valid_{}_test_{}_cgl.json'.format(exp_config['train_mode'], exp_config['valid_mode'], self.test_mode)
                        exp_result_path = os.path.join(exp_result_dir, exp_result_name)
                        if not os.path.exists(exp_result_dir):
                            os.makedirs(exp_result_dir)
                        file = open(exp_result_path, "w")
                        json.dump(eval_result, file)
                        file.write('\r\n')

                        light_result = {}
                        try:
                            light_result['dp_true'] = eval_result['dp_true']
                        except:
                            light_result = {}
                        
                        json.dump(light_result, file)
                        file.write('\r\n')
                        model_config = model.model_config_dict
                        json.dump(model_config, file)
                        file.close()
                
                elif self.dro == True:
                    exp_config = self.exp_config
                    exp_extra_info = 'explicit_{}_test_inter_num_{}_test_propotion_{}_thre_{}_use_thre_{}_use_classifier_stage{}'\
                    .format(str(exp_config['explicit']), str(exp_config["test_inter_num"]),str(exp_config["test_propotion"]), str(exp_config["thre"]),
                    str(exp_config["use_thre"]), str(exp_config["use_classifier_stage"]))
                    exp_result_dir = os.path.join(workspace, 'unfairness', unfairness_metric, exp_config['data_name'], exp_config['model_name'], 
                    exp_extra_info, str(exp_config['know_size']), str(exp_config['performance_scale']),str(exp_config['seed']),
                    str(exp_config['classifier_stage']), test_or_val + '_results', )
                    exp_result_name = 'train_{}_valid_{}_test_{}_dro.json'.format(exp_config['train_mode'], exp_config['valid_mode'], self.test_mode)
                    exp_result_path = os.path.join(exp_result_dir, exp_result_name)
                    if not os.path.exists(exp_result_dir):
                        os.makedirs(exp_result_dir)
                    file = open(exp_result_path, "w")
                    json.dump(eval_result, file)
                    file.write('\r\n')
                    light_result = {}
                    try:
                        light_result['dp_true'] = eval_result['dp_true']
                    except:
                        light_result = {}
                    
                    json.dump(light_result, file)
                    file.write('\r\n')
                    model_config = model.model_config_dict
                    json.dump(model_config, file)
                    file.close()

        # lack-user-profile-experiment
        elif self.lack_profile:
            if self.exp_config is not None:
                if self.dro == False:
                    if self.cgl == False:
                        exp_config = self.exp_config
                        exp_extra_info = 'explicit_{}_test_inter_num_{}_test_propotion_{}_thre_{}_use_thre_{}_use_classifier_stage{}'\
                            .format(str(exp_config['explicit']), str(exp_config["test_inter_num"]),str(exp_config["test_propotion"]), str(exp_config["thre"]),
                            str(exp_config["use_thre"]), str(exp_config["use_classifier_stage"]))
                        exp_result_dir = os.path.join(workspace, 'unfairness', unfairness_metric, exp_config['data_name'], exp_config['model_name'], 
                        exp_extra_info, str(exp_config['know_size']), str(exp_config['performance_scale']),str(exp_config['seed']),
                        str(exp_config['classifier_stage']), test_or_val + '_results', )
                        exp_result_name = 'lack_prob_{}_train_{}_valid_{}_test_{}.json'.format(str(self.lack_profile_prob),
                        exp_config['train_mode'], exp_config['valid_mode'], self.test_mode)
                        exp_result_path = os.path.join(exp_result_dir, exp_result_name)
                        if not os.path.exists(exp_result_dir):
                            os.makedirs(exp_result_dir)
                        file = open(exp_result_path, "w")
                        json.dump(eval_result, file)
                        file.write('\r\n')
                        light_result = {}
                        try:
                            light_result['dp_true'] = eval_result['dp_true']
                        except:
                            light_result = {}
                        
                        json.dump(light_result, file)
                        file.write('\r\n')
                        model_config = model.model_config_dict
                        json.dump(model_config, file)
                        file.close()
                    else:
                        exp_config = self.exp_config
                        exp_extra_info = 'explicit_{}_test_inter_num_{}_test_propotion_{}_thre_{}_use_thre_{}_use_classifier_stage{}'\
                            .format(str(exp_config['explicit']), str(exp_config["test_inter_num"]),str(exp_config["test_propotion"]), str(exp_config["thre"]),
                            str(exp_config["use_thre"]), str(exp_config["use_classifier_stage"]))
                        exp_result_dir = os.path.join(workspace, 'unfairness', unfairness_metric, exp_config['data_name'], exp_config['model_name'], 
                        exp_extra_info, str(exp_config['know_size']), str(exp_config['performance_scale']),str(exp_config['seed']),
                        str(exp_config['classifier_stage']), test_or_val + '_results', )
                        exp_result_name = 'lack_prob_{}_train_{}_valid_{}_test_{}_cgl.json'.format(str(self.lack_profile_prob),
                        exp_config['train_mode'], exp_config['valid_mode'], self.test_mode)
                        exp_result_path = os.path.join(exp_result_dir, exp_result_name)
                        if not os.path.exists(exp_result_dir):
                            os.makedirs(exp_result_dir)
                        file = open(exp_result_path, "w")
                        json.dump(eval_result, file)
                        file.write('\r\n')
                        light_result = {}
                        try:
                            light_result['dp_true'] = eval_result['dp_true']
                        except:
                            light_result = {}
                        
                        json.dump(light_result, file)
                        file.write('\r\n')
                        model_config = model.model_config_dict
                        json.dump(model_config, file)
                        file.close()
                
                elif self.dro == True:
                    exp_config = self.exp_config
                    exp_extra_info = 'explicit_{}_test_inter_num_{}_test_propotion_{}_thre_{}_use_thre_{}_use_classifier_stage{}'\
                    .format(str(exp_config['explicit']), str(exp_config["test_inter_num"]),str(exp_config["test_propotion"]), str(exp_config["thre"]),
                    str(exp_config["use_thre"]), str(exp_config["use_classifier_stage"]))
                    exp_result_dir = os.path.join(workspace, 'unfairness', unfairness_metric, exp_config['data_name'], exp_config['model_name'], 
                    exp_extra_info, str(exp_config['know_size']), str(exp_config['performance_scale']),str(exp_config['seed']),
                    str(exp_config['classifier_stage']), test_or_val + '_results', )
                    exp_result_name = 'lack_prob_{}_train_{}_valid_{}_test_{}_dro.json'.format(str(self.lack_profile_prob),
                    exp_config['train_mode'], exp_config['valid_mode'], self.test_mode)
                    exp_result_path = os.path.join(exp_result_dir, exp_result_name)
                    if not os.path.exists(exp_result_dir):
                        os.makedirs(exp_result_dir)
                    file = open(exp_result_path, "w")
                    json.dump(eval_result, file)
                    file.write('\r\n')
                    light_result = {}
                    try:
                        light_result['dp_true'] = eval_result['dp_true']
                    except:
                        light_result = {}
                    
                    json.dump(light_result, file)
                    file.write('\r\n')
                    model_config = model.model_config_dict
                    json.dump(model_config, file)
                    file.close()
        return 


class Predict_sensitive_attribute():
    def __init__(self, data_name, model_name = None, trial_name = None, dir = None, use_classifier_stage = False, classifier_stage = 250):
        
       
        self.data_name = data_name
        checkpoint_dir = os.path.join(workspace, data_name)
        if model_name is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if trial_name is not None:
            if use_classifier_stage:
                checkpoint_dir = os.path.join(checkpoint_dir, trial_name + str(classifier_stage))
            else:
                checkpoint_dir = os.path.join(checkpoint_dir, trial_name)
        if dir is not None:
            checkpoint_dir = dir
        
        self.checkpoint_dir = checkpoint_dir
        file_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        print(file_path)
        model = torch.load(file_path)['model']
        self.model = model

    def load_test(self, know_size,  Dataset,  seed = 2004, k = 48, metrics = ['acc'], batch_size = 8192, sensitive_attr = 'gender',
    save_path = None, save_path_acc = None):
        print('loading best models in validation sets and testing...')
        model = self.model
        model.metrics = model._get_metrics(metrics)
        threshold = model.model_config_dict['threshold']
        predict_input = Dataset.model_input
        predict_sensitive_attr = model.predict(predict_input, batch_size)
        predict_sensitive_df = pd.DataFrame([])
        predict_sensitive_df['user_id'] = Dataset.train_data.index
        print(predict_sensitive_df['user_id'])
        predict_sensitive_df['predict_score'] = predict_sensitive_attr
        predict_sensitive_df[sensitive_attr] = np.where((predict_sensitive_attr > threshold), 1, 0)
        predict_sensitive_df.to_csv(save_path)
        data = pd.read_hdf('./{}.h5'.format(self.data_name))
        true_sensitive_attrs = []
        user_ids = np.array(predict_sensitive_df['user_id']).tolist()
        for user_id in user_ids:
            true_sensitive_attrs.append(data.loc[data['user_id'] == user_id,  sensitive_attr])
        acc = accuracy_score(true_sensitive_attrs, list(np.where((predict_sensitive_attr > threshold), 1, 0).squeeze().tolist()))
        print('predicted_accuracy_is: ', acc)
        file = open(save_path_acc, "w")
        json.dump(acc, file)
        # generate val_gamma
        val_model_input = Dataset.val_model_input
        val_predict_sensitive_attr = model.predict(predict_input, batch_size)
        val_predict_sensitive_attr = np.where((val_predict_sensitive_attr > threshold), 1, 0).squeeze()
        val_user_ids = Dataset.val_data.index.tolist()
        true_sensitive_attrs = []
        for user_id in val_user_ids:
            true_sensitive_attrs.append(data.loc[data['user_id'] == user_id,  sensitive_attr])
        true_sensitive_attrs = np.array(true_sensitive_attrs).squeeze()
        index_0 = (true_sensitive_attrs == 0)
        index_1 = (true_sensitive_attrs == 1)
        gamma_0 = 1 - accuracy_score(true_sensitive_attrs[index_0].tolist(), val_predict_sensitive_attr[index_0].tolist()) 
        gamma_1 = 1 - accuracy_score(true_sensitive_attrs[index_1].tolist(), val_predict_sensitive_attr[index_1].tolist())
        gamma_dicts = {'gamma_0':gamma_0,'gamma_1':gamma_1}
        file.write('\r\n')
        json.dump(gamma_dicts,file)
        file.close() 
        return


