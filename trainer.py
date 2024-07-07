#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import logging
import numpy as np
import yaml
import os

import utils.data as data
from utils.build_tree import build_hieraichical_clustering_tree
from utils import Context as ctxt
from utils.metrics import eva
from model import *


class Trainer(object):

    def __init__(self, flags_obj, cm,  dm, new_config=None):
        """
        Args:
            flags_obj: arguments in main.py
            cm : context manager
            dm : dataset manager
            new config : update default model config(`./config/model_kuaishou.yaml`) to tune hyper-parameters
        """
        self.cm = cm #context manager
        self.dm = dm #dataset manager
        self.flags_obj = flags_obj
        self.model_name = flags_obj.model
        self.set_device()
        self.load_model_config()
        self.update_model_config(new_config)
        self.lr = self.model_config['lr']
        self.set_tensorboard(flags_obj.tb)
        # self.judger = judge() # calculate metrics
        self.set_model()
        self.set_dataloader()
        self.model = self.model.to(self.device)
        
    def set_device(self):
        if not self.flags_obj.use_gpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.flags_obj.gpu_id))
        
    def load_model_config(self):
        path = 'config/{}/{}.yaml'.format(self.dm.dataset_name, self.model_name)
        f = open(path)
        self.model_config = yaml.load(f, Loader=yaml.FullLoader)

    def update_model_config(self, new_config):
        self.tune_config = new_config 
        self.model_config = self.dm.set_dataset_related_hyparam(self.model_config)

        if new_config is not None:
            for item in new_config.keys():
                if not item in self.model_config.keys():
                    raise ValueError(f'False config key value: {item}')
            for key in [item for item in new_config.keys() if item in self.model_config.keys()]:
                if type(self.model_config[key]) == dict:
                    self.model_config[key].update(new_config[key])
                else:
                    self.model_config[key] = new_config[key]

    def set_model(self):
        self.model = None
        raise NotImplementedError()
    
    def set_tensorboard(self, tb=False):
        if tb:
            self.writer = SummaryWriter("{}/tb/{}".format(self.cm.workspace, self.cm.exp_name))
        else:
            self.writer = None

    def set_dataloader(self):
        if hasattr(self, 'train_dataloader'):
            return

        # training dataloader
        self.train_dataloader = data.get_dataloader(
            data_set = getattr(data, f'{self.model_name.upper()}_Dataset')(self.dm, mode='training'),
            bs = self.dm.batch_size,
            prefetch_factor = self.dm.batch_size // self.dm.num_workers + 1 if self.dm.num_workers!=0 else 2, 
            num_workers = self.dm.num_workers,
            shuffle=True
        )
        # validation dataloader
        self.valid_dataloader =  data.get_dataloader(
            data_set = getattr(data, f'{self.model_name.upper()}_Dataset')(self.dm, mode='validation'),
            bs = self.dm.test_batch_size,
            prefetch_factor = self.dm.batch_size // self.dm.num_workers + 1 if self.dm.num_workers!=0 else 2, 
            num_workers = self.dm.num_workers
        )
        # test dataloader
        self.test_dataloader =  data.get_dataloader(
            data_set = getattr(data, f'{self.model_name.upper()}_Dataset')(self.dm, mode='test'),
            bs = self.dm.test_batch_size,
            prefetch_factor = self.dm.batch_size // self.dm.num_workers + 1 if self.dm.num_workers!=0 else 2, 
            num_workers = self.dm.num_workers
        )

    
    def save_ckpt(self):

        ckpt_path = os.path.join(self.cm.workspace, 'ckpt')
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'best.pth')
        torch.save(self.model.state_dict(), model_path)

    def load_ckpt(self, assigned_path=None):

        ckpt_path = os.path.join(self.cm.workspace, 'ckpt')
        model_path = None
        if assigned_path is not None:
            '''specific assigned path'''
            model_path = assigned_path
        else:
            '''default path'''   
            model_path = os.path.join(ckpt_path, 'best.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def train(self):

        self.optimizer = optim.Adam(self.model.parameters(), \
                        lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'])
        self.esm = ctxt.EarlyStopManager(self.model_config) # early-stop manager

        best_metric = 0
        train_loss = [0.0, 0.0, 0.0, 0.0, 0.0] #store every training loss
        val_loss = [0.0] # store loss on validation set
        
        for epoch in range(self.flags_obj.epochs):

            self.train_one_epoch(epoch, train_loss)
            watch_metric_value = self.validate(epoch, val_loss)
            if watch_metric_value > best_metric:
                self.save_ckpt()
                logging.info('save ckpt at epoch {}'.format(epoch))
                best_metric = watch_metric_value

            stop = self.esm.step(self.lr, watch_metric_value)
            if stop:
                break

    def train_one_epoch(self, epoch, train_loss):

        epoch_loss = train_loss[0]

        self.model.train()

        tqdm_ = tqdm(iterable=self.train_dataloader, mininterval=1, ncols=100)
        for step, sample in enumerate(tqdm_):

            self.optimizer.zero_grad()

            sample = tuple(input_data.to(self.device) for input_data in sample)
            loss = self._get_loss(sample)

            if torch.isnan(loss):
                raise ValueError('loss is NaN!')
                # print('loss is NaN!')

            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % (self.train_dataloader.__len__() // 50) == 0 and step != 0:
                tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch, step+1, epoch_loss / (step+1+epoch*self.train_dataloader.__len__())))
                if self.writer and self.flags_obj.train_tb:

                    self.writer.add_scalar("training_loss",
                                    epoch_loss/(step+1+epoch*self.train_dataloader.__len__()), step+1+epoch*self.train_dataloader.__len__())
        logging.info('epoch {}:  loss = {}'.format(epoch, epoch_loss/(step+1+epoch*self.train_dataloader.__len__())))

        train_loss[0] = epoch_loss

    @torch.no_grad()
    def evaluate(self, total_loss=None, epoch=0):
        """Evaluate the model on validation/test data set
        
        Args:
            total_loss: store total loss for all epochs. only used for validation
            epoch: number of current epoch

        Returns:
            results: dict of evaluation metrics 
        """

        self.model.eval()

        group_pred_items, group_next_items = self._run_eval(
            dataloader= self.valid_dataloader if epoch!= 'test' else self.test_dataloader 
        )

        if epoch != 'test':

            return eva(pre = group_pred_items, ground_truth = group_next_items)
        
        else:

            # comi_ndcg = eva(pre = group_pred_items, ground_truth = group_next_items, comi_ndcg=True)
            res = eva(pre = group_pred_items, ground_truth = group_next_items, comi_ndcg=False)

            return res
        
    def _run_eval(self, dataloader):
        """ 
        making prediction in full-sort setting
        """
        topK = 50
        group_pred_logits, group_next_items, group_pred_items = [], [], []

        user_item_record = dataloader.dataset.record # pd.Dataframe

        for batch_data in tqdm(iterable=dataloader, mininterval=1, ncols=100):
            step_uid = batch_data[0]
            batch_data = tuple(input_data.to(self.device) for input_data in batch_data)
            step_pred_logits, step_pred_item = self._get_prediction(batch_data, topK) #B, topK
            
            step_next_items = \
                user_item_record.loc[user_item_record['uid'].isin(step_uid.numpy())]\
                ['next_item'].values #np.array(list[], list[],...,list[])

            # group_pred_logits.extend(step_pred_logits.cpu().numpy())
            group_pred_items.extend(step_pred_item.cpu().numpy())
            group_next_items.extend(step_next_items)

        # cpu() results in gpu memory not auto-collected
        # this command frees memory in Nvidia-smi
        if self.device != torch.device('cpu'):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache() 

        return group_pred_items, group_next_items


    def validate(self, epoch, total_loss):

        results = self.evaluate(total_loss=total_loss, epoch=epoch)
        self.record_metrics(epoch, results)
        print(results)
       
        return results['recall@50']
    

    def test(self, assigned_model_path = None, load_config=True):
        '''
            test model on test dataset
        '''

        if load_config:
            self.load_ckpt(assigned_path = assigned_model_path)

        results = self.evaluate(epoch='test')

        logging.info('TEST results :')
        self.record_metrics('test', results)
        print('test: ', results)


    def record_metrics(self, epoch, metric):
        """
        record metrics after each epoch
        """    

        logging.info('VALIDATION epoch: {}, results: {}'.format(epoch, metric))
        if self.writer:
            if epoch != 'test':
                for k,v in metric.items():
                        self.writer.add_scalar("training_metric/"+str(k), v, epoch)


 
class Sequence_Dual_Encoder_Trainer(Trainer):
    def __init__(self, flags_obj, cm, dm, new_config=None):
        super().__init__(flags_obj, cm, dm, new_config)

    def _get_loss(self, sample):
        '''
        Args:
            sample -> tuple : (user_id, user_reco_his, next_item, neg_sample_item)
        Return:
            loss -> torch.tensor (,)
        '''

        return self.model(sample)
    
    def _get_prediction(self, sample, topK):
        '''
        Args:
            sample -> tuple : (user_id, user_reco_his)
        Return:
            pred_logits, pred_item_ID: torch.tensor (Batch, topK)
        '''

        return self.model.predict(sample, topK)
    
class SASREC_Trainer(Sequence_Dual_Encoder_Trainer):
    def __init__(self, flags_obj, cm, dm, new_config=None):
        super().__init__(flags_obj, cm, dm, new_config)

    def set_model(self):
        self.model = SASREC(self.model_config)


        logging.info('model config:')
        for k,v in self.model_config.items():
            logging.info('{}: {}'.format(k, v))



class SEATER_Trainer(Trainer):
    def __init__(self, flags_obj, cm, dm, new_config=None):
        """
        Args:
            flags_obj: arguments in main.py
            cm : context manager
            dm : dataset manager
            new config : update default model config to tune hyper-parameters
        """
        self.cm = cm #context manager
        self.dm = dm #dataset manager
        self.flags_obj = flags_obj
        self.model_name = flags_obj.model
        self.set_device()
        self.load_model_config()
        self.update_model_config(new_config)
        self.lr = self.model_config['lr']
        self.set_tensorboard(flags_obj.tb)
        self.set_model()
        self.model = self.model.to(self.device)

    def set_model(self):
        # build tree index structure
        self.dm.tree_data_par_path = os.path.join(self.dm.tree_data_par_path, f'{self.flags_obj.vocab}_branch_tree')
        if not os.path.exists(f'{self.dm.tree_data_par_path}/itemID_2_tree_indexID.npy'):
            build_hieraichical_clustering_tree(
                item_emb_path = os.path.join(self.dm.datafile_par_path, self.dm.two_tower_item_emb),
                output_file_path = self.dm.tree_data_par_path,
                vocab_size = self.flags_obj.vocab
            )
        self.dm.num_neg = self.model_config['rk_num_neg']

        self.set_dataloader()

        # reset model config
        self.model_config['decoder_index']['vocab_size'] = self.flags_obj.vocab
        self.model_config['decoder_index']['tree_nodes_num'] = self.train_dataloader.dataset.tree_nodes_num
        self.model_config['decoder_index']['max_len'] = self.train_dataloader.dataset.max_len
        print('tree node num: ', self.model_config['decoder_index']['tree_nodes_num'])
        print('tree max depth', self.model_config['decoder_index']['max_len'])
        
        logging.info('model config:')
        for k,v in self.model_config.items():
            logging.info('{}: {}'.format(k, v))
        
        self.model = SEATER(self.model_config)
        self.model._init_prefix_mask(
            self.train_dataloader.dataset.prefix_allowed_token,
            self.device
        )

        self.w_rk = self.model_config['rk_weight']
        self.w_sm = self.model_config['sm_weight']

    def _get_loss(self, sample):
        '''
        Args:
            sample -> tuple : ( user_reco_his, next_item)
        Return:
            loss -> torch.tensor (,)
        '''

        return self.model.train_step(sample)
    
    def _get_prediction(self, sample, topK):
        '''
        Args:
            sample (torch.tensor) (Batch) : user_reco_his
        Return:
            pred_logits, pred_item_ID: torch.tensor (Batch, topK)
        '''
       
        logits, pred_items = self.model.predict_step(sample, topK)

        return logits, pred_items

    def train_one_epoch(self, epoch, train_loss):

        epoch_decode_loss = train_loss[0]
        epoch_rk_loss = train_loss[1]
        epoch_sm_loss = train_loss[2]

        self.model.train()
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < self.lr: #record schedular reducing lr
            self.lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(self.lr))

        tqdm_ = tqdm(iterable=self.train_dataloader, mininterval=1, ncols=150)
        for step, sample in enumerate(tqdm_):

            self.optimizer.zero_grad()

            sample = tuple(input_data.to(self.device) for input_data in sample)
            decode_loss, rk_loss, sm_loss = self._get_loss(sample)

            loss = decode_loss + self.w_rk * rk_loss + self.w_sm * sm_loss

            if torch.isnan(loss):
                raise ValueError('loss is NaN!')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)
            self.optimizer.step()
            
            epoch_decode_loss += decode_loss.item()
            epoch_rk_loss += rk_loss.item()
            epoch_sm_loss += sm_loss.item()
            
            if step % (self.train_dataloader.__len__() // 50) == 0 and step != 0:
                tqdm_.set_description(
                        "epoch {:d} , step {:d} , decode_loss: {:.4f}, rk_loss: {:.4f}, sm_loss: {:.4f}"\
                            .format(epoch, 
                                    step+1, 
                                    epoch_decode_loss / (step+1+epoch*self.train_dataloader.__len__()),
                                    epoch_rk_loss / (step+1+epoch*self.train_dataloader.__len__()),
                                    epoch_sm_loss / (step+1+epoch*self.train_dataloader.__len__()),
                                    ))
                if self.writer and self.flags_obj.train_tb:

                    self.writer.add_scalar("decode_loss",
                                    epoch_decode_loss/(step+1+epoch*self.train_dataloader.__len__()), step+1+epoch*self.train_dataloader.__len__())
                    self.writer.add_scalar("rk_loss",
                                    epoch_rk_loss/(step+1+epoch*self.train_dataloader.__len__()), step+1+epoch*self.train_dataloader.__len__())
                    self.writer.add_scalar("sm_loss",
                                    epoch_sm_loss/(step+1+epoch*self.train_dataloader.__len__()), step+1+epoch*self.train_dataloader.__len__())
                    # self.writer.add_scalar("temp",
                    #                 self.model.temp.item(), step+1+epoch*self.train_dataloader.__len__())
        logging.info('epoch {}:  decode loss = {}, rk loss = {}, sm loss = {}'\
                     .format(epoch, 
                             epoch_decode_loss / (step+1+epoch*self.train_dataloader.__len__()),
                             epoch_rk_loss / (step+1+epoch*self.train_dataloader.__len__()),
                             epoch_sm_loss / (step+1+epoch*self.train_dataloader.__len__()),
                             ))
        
        
        train_loss[0] = epoch_decode_loss
        train_loss[1] = epoch_rk_loss
        train_loss[2] = epoch_sm_loss