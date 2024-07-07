#coding=utf-8

import os
import datetime

import logging
import torch

import trainer
from config import const as const_utils

import datetime



class ContextManager(object):

    def __init__(self, flags_obj):

        self.exp_name = flags_obj.name
        self.description = flags_obj.description
        self.workspace_root = flags_obj.workspace
        self.set_workspace(flags_obj)
        self.set_logging(flags_obj)


    def set_workspace(self, flags_obj):

        date_time = '_'+str(datetime.datetime.now().month)\
            +'_'+str(datetime.datetime.now().day)\
            +'_'+str(datetime.datetime.now().hour)
        dir_name = self.exp_name + '_' + date_time
        if not os.path.exists(self.workspace_root):
            os.mkdir(self.workspace_root)

        self.workspace = os.path.join(self.workspace_root, dir_name)
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)

        if flags_obj.tb:
            if not os.path.exists( 
                    os.path.join(self.workspace, 'tb')
                    ):
                os.mkdir(os.path.join(self.workspace, 'tb'))


    def set_logging(self, flags_obj):
        # set log file path
        if not os.path.exists(os.path.join(self.workspace, 'log')):
            os.mkdir(os.path.join(self.workspace, 'log'))
        log_file_name = os.path.join(self.workspace, 'log', self.description+'.log')
        logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO, filename=log_file_name, filemode='w')
        

        logging.info('Configs:')
        for flag, value in flags_obj.__dict__.items():
            logging.info('{}: {}'.format(flag, value))

    @staticmethod
    def set_trainer(flags_obj, cm, dm, nc=None):

        try:
            return getattr(trainer, f'{flags_obj.model.upper()}_Trainer')(flags_obj, cm, dm, nc)
        except AttributeError:
            raise NameError('trainer model name error!')

class DatasetManager(object):

    def __init__(self, flags_obj):

        self.dataset_name = flags_obj.dataset_name
        self.batch_size = flags_obj.batch_size
        self.test_batch_size = flags_obj.test_batch_size
        self.num_workers = flags_obj.num_workers
        dataset_config = getattr(const_utils, '{}_Config'.format(self.dataset_name))()
        for key, value in dataset_config.__dict__.items():
            setattr(self, key, value)

        logging.info('dataset: {}'.format(self.dataset_name))

    def set_dataset_related_hyparam(self, model_config):

        model_config['item_num'] = self.item_ID_num
        model_config['max_reco_his'] = self.reco_his_max_length
        
        return model_config

    def show(self):
        print(self.__dict__)




class EarlyStopManager(object):

    def __init__(self, config):

        self.min_lr = config['min_lr']
        self.es_patience = config['es_patience']
        self.count = 0
        self.max_metric = 0

    def step(self, lr, metric):

        if lr > self.min_lr:
            if metric > self.max_metric:
                self.max_metric = metric
            return False
        else:
            if metric >= self.max_metric:
                self.max_metric = metric
                self.count = 0
                return False
            else:
                self.count = self.count + 1
                if self.count > self.es_patience:
                    return True
                return False


