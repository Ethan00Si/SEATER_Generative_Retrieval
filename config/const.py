
class Books_Config(object):
    def __init__(self) -> None:
        
        # data file path
        self.datafile_par_path = '../open-source-data/Books'
        # self.datafile_par_path = './data/Books'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'
        self.itemID_2_attr = 'vocab/item_2_attr_mapping.npy'

        # for tree-based index
        self.tree_data_par_path = '../open-source-data/Books/tree_data_SASREC'
        # self.tree_data_par_path = './data/Books/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Books_SASREC_item_emb.npy'

        # dataset config
        self.users_ID_num = 603671
        self.item_ID_num = 367982 + 1 # zero for padding
        self.item_cate_num = 1600 + 1 # zero for padding

        # experiment const config
        self.reco_his_max_length = 20

class Yelp_Config(object):
    def __init__(self) -> None:
        
        # data file path
        self.datafile_par_path = '../open-source-data/Yelp'
        # self.datafile_par_path = './data/Yelp'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        self.tree_data_par_path = '../open-source-data/Yelp/tree_data_SASREC'
        # self.tree_data_par_path = './data/Yelp/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Yelp_SASREC_item_emb.npy'

        #dataset config
        self.users_ID_num = 31668
        self.item_ID_num = 38048 + 1 # zero for padding

        # experiment const config
        self.reco_his_max_length = 20 

class MIND_Config(object):
    def __init__(self) -> None:
        
        # data file path
        self.datafile_par_path = '../open-source-data/MIND'
        # self.datafile_par_path = './data/MIND'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        self.tree_data_par_path = '../open-source-data/MIND/tree_data_SASREC'
        # self.tree_data_par_path = './data/MIND/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/MIND_SASREC_item_emb.npy'

        #dataset config
        self.users_ID_num = 50000
        self.item_ID_num = 39865 + 1 # zero for padding

        # experiment const config
        self.reco_his_max_length = 20 

