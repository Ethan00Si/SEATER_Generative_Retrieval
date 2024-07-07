# News Dataset
The Amazon review dataset is one of the most widely used recommendation benchmarks. We adopt the `Books' subset. We utilized the processed data from the public repository (https://github.com/THUDM/ComiRec). 

## Main Data Files

File structure:
```
.
├── SEATER_checkpoint
│   └── best.pth
├── dataset
│   ├── test.tsv
│   ├── training.tsv
│   └── validation.tsv
├── raw_data
├── readme.md
├── tree_data_SASREC
│   └── 16_branch_tree
│       ├── itemID_2_tree_indexID.npy
│       └── tree_node_allowed_next_tokens.npy
└── vocab
    ├── Books_SASREC_item_emb.npy
    └── item_2_attr_mapping.npy
```
- SEATER_checkpoint: DIrectory containing checkpoint.
  - best.pth: checkpoint of SEATER.

- dataset: Directory containing processed data.
    - training.tsv: Training dataset.
    - test.tsv: Test dataset.
    - validation.tsv: Validation dataset.

- raw_data: Directory containing original data downloaded from https://github.com/THUDM/ComiRec. Since the original data file is too large, we do not upload it here. If you want to process from scratch, please download it from the mentioned link.
 
- tree_data_SASREC: Directory containing tree-related data used by SEATER. The data was constructed using SASREC's item embeddings.     
  - 16_branch_tree: Subdirectory with tree-related data of 16 branches.
    - itemID_2_tree_indexID.npy: Numpy file mapping item IDs to tree-structured identifiers.
    - tree_node_allowed_next_tokens.npy: Numpy file with allowed next tokens. Each line maps a token ID to its corresponding child tokens.
- vocab: Directory containing additional data.
  - MIND_SASREC_item_emb.npy: Numpy file with SASREC's item embeddings.

## Data Processing Code

* processing.ipynb: file to process raw data.
  
Note that the processing code is not carefully checked. If you want to process from scratch, please read the code first. 