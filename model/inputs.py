import torch
import torch.nn as nn
import torch.nn.functional as F


class item_feat_no_add_feat(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.item_num = model_config['item_num']

        self.size = model_config['embedding_dim']

        self.emb_look_up = nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.size)

    def forward(self, item_feat_index):
        '''
        get item embedding, only ID embedding
        '''

        return self.emb_look_up(item_feat_index)
    
    
class OURS_item_feat(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.item_num = model_config['item_num']
        self.tree_node_num = model_config['decoder_index']['tree_nodes_num']
        self.non_leaf_node_num = self.tree_node_num - self.item_num

        self.size = model_config['embedding_dim']
        self.emb_look_up = nn.Embedding(num_embeddings=self.tree_node_num, embedding_dim=self.size)

        self.start_token = self.item_num
        self.end_token = self.item_num + 1
        self.pad_token = self.item_num + 2 

        # vocab size, omitting special tokens
        self.vocab_size = model_config['decoder_index']['vocab_size']

    def forward(self, itemIDs):
        '''
        Args:
            itemIDs: torch.tensor
        Return:
            item embedding
        '''

        return self.emb_look_up(itemIDs)


