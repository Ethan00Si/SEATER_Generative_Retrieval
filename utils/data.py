import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np

from utils import data_utils


class SequenceDataset(Dataset):
    '''
    For dual-encoder (two-towel) models
    '''
    def __init__(self, data_args, mode=None) -> None:
        '''
        Args:
            data_args(object): data setting arguments
            mode(string): 'training', 'validation' or 'test'
        '''
        super().__init__()
        self.dataset_name = data_args.dataset_name
        assert mode in ['training', 'validation', 'test'], 'invalid mode!'
        self.mode = mode
        self.dm = data_args

        self._build()

    def _build(self):
        self.reco_his_max_len = self.dm.reco_his_max_length #padding or truncation length
        
        if self.mode == 'training':
            reco_his_seq = data_utils.load_tsv_file(
                self.dm.datafile_par_path, self.dm.training_file, sep='\t', engine="pyarrow")
        elif self.mode == 'validation':
            reco_his_seq = data_utils.load_tsv_file(
                self.dm.datafile_par_path, self.dm.validation_file, sep='\t', engine="pyarrow")
            reco_his_seq['predicting_items'] = reco_his_seq['predicting_items'].apply(
                lambda x:eval(x) # string --> list
            )

        elif self.mode == 'test':
            reco_his_seq = data_utils.load_tsv_file(
                self.dm.datafile_par_path, self.dm.test_file, sep='\t', engine="pyarrow")
            reco_his_seq['predicting_items'] = reco_his_seq['predicting_items'].apply(
                lambda x:eval(x) # string --> list
            )

        reco_his_seq.columns = ['uid','his_seq','next_item']

        self.record = reco_his_seq
        
        if self.mode == 'training':
            self.item_table_with_popularity = self.record['next_item'].tolist()

        print('build {} dataset successfully'.format(self.mode))

    def __len__(self):
        return self.record.shape[0]

    def _parse_line(self, index):
        line = self.record.iloc[index]

        uid, user_his_seq, next_item = line['uid'], line['his_seq'], line['next_item']
        
        user_his_seq = eval(user_his_seq)[-self.reco_his_max_len:] # string --> list

        if self.reco_his_max_len > len(user_his_seq):
            user_his_seq = user_his_seq + [0] * (self.reco_his_max_len - len(user_his_seq))

        assert self.reco_his_max_len == len(user_his_seq)

        return uid, user_his_seq, next_item


    def __getitem__(self, index):

        uid, user_his_seq, next_item = self._parse_line(index)

        if self.mode == 'training':

            cur_tensor = (
                torch.tensor(uid, dtype=torch.long), 
                torch.tensor(user_his_seq, dtype=torch.long),
                torch.tensor(next_item, dtype=torch.long),
            )

        else:

            cur_tensor = (
                torch.tensor(uid, dtype=torch.long), 
                torch.tensor(user_his_seq, dtype=torch.long)
            )
        
        return cur_tensor

        
class SASREC_Dataset(SequenceDataset):
    def __init__(self, data_args, mode=None) -> None:
        super().__init__(data_args, mode)



class SEATER_Dataset(SequenceDataset):
    def __init__(self, data_args, mode=None) -> None:
        super().__init__(data_args, mode)

        self.mapping_matrix = data_utils.load_npy_file(
            load_path=self.dm.tree_data_par_path, filename=self.dm.tree_based_itemID_2_indexID
        ) # containing start_token, end_token and pad_token

        self.prefix_allowed_token = data_utils.load_npy_file(
            self.dm.tree_data_par_path, self.dm.tree_based_prefix_tree
        )

        self.tree_nodes_num = self.prefix_allowed_token.shape[0]
        self.item_num = self.mapping_matrix.shape[0]
        self.item_IDs = np.arange(1, self.item_num)
        self.max_len = self.mapping_matrix.shape[1]

        self.num_tree_branches = self.prefix_allowed_token.shape[1]

        self.num_neg = self.dm.num_neg

        self._construct_prefixe_aware_item_index_groups()

    def _construct_prefixe_aware_item_index_groups(self):
        '''
        create mapping from each token to identifiers containing this token
        '''

        self.item_index_groups = []
        for i in range(self.tree_nodes_num):
            self.item_index_groups.append([])
        
        for item_index in self.mapping_matrix[1:, :]:
            tmp = item_index[:-1]
            for node in tmp:
                self.item_index_groups[node].append(item_index)
        
        self.item_index_groups = [np.array(group) for group in self.item_index_groups]

        for i in range(self.item_num+3, self.tree_nodes_num):
            assert len(self.item_index_groups[i]) > 1, f'node number: {i}'


    # def neg_sample(self, next_item):
    #     '''used for imbalanced tree'''
    #     neg_items = []
    #     while len(neg_items) < self.num_neg: # 10 neg samples  
    #         neg_item = random.choice(self.item_IDs)

    #         while neg_item == next_item:
    #             neg_item = random.choice(self.item_IDs)
    #         neg_items.append(self.mapping_matrix[neg_item])
        
    #     neg_items = np.array(neg_items)

    #     return neg_items


    def neg_sample(self, next_item):  
        '''
        negative sampling for ranking loss (triplet loss)
        only used for balanced trees. if imbalanced do not use
        '''
        
        sample_nodes = next_item[:-2]
        sample_nodes = sample_nodes[-self.num_neg:]
        neg_item_index_ls = []
        for node in sample_nodes:
            all_possible_items = self.item_index_groups[node]
            
            neg_item = random.choice(all_possible_items)
            while np.all( neg_item == next_item ):
                neg_item = random.choice(all_possible_items)
            neg_item_index_ls.append(neg_item)

        neg_item_index_ls = np.array(neg_item_index_ls)

        return neg_item_index_ls

    def __getitem__(self, index):

        uid, user_his_seq, next_item = self._parse_line(index)

        if self.mode == 'training':
            
            next_item_index_id = self.mapping_matrix[next_item]
            neg_item_index_id = self.neg_sample(next_item_index_id)
            # neg_item_index_id = self.neg_sample(next_item)

            cur_tensor = (
                torch.tensor(uid, dtype=torch.long), 
                torch.tensor(user_his_seq, dtype=torch.long),
                torch.tensor(next_item_index_id, dtype=torch.long),
                torch.tensor(neg_item_index_id, dtype=torch.long)
            )

        else:

            cur_tensor = (
                torch.tensor(uid, dtype=torch.long), 
                torch.tensor(user_his_seq, dtype=torch.long),
            )
        
        return cur_tensor
    

GLOBAL_SEED = 1
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def get_dataloader(data_set, bs, **kwargs):
    return DataLoader(  data_set, batch_size = bs,
                        pin_memory = True, 
                        worker_init_fn=worker_init_fn, **kwargs
                    )