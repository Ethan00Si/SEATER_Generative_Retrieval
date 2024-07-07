
import numpy as np
import pandas as pd
import pickle

import json
import os

def load_tsv_file(load_path, filename, **kwargs):

    filename = os.path.join(load_path, filename)
    record = pd.read_csv(filename, **kwargs)

    return record

def load_json_file(load_path, filename, **kwargs):

    filename_ = os.path.join(load_path, filename)
    with open(filename_, 'r') as f:
        record = json.loads(f.read())
    
    return record

def load_npy_file(load_path, filename, **kwargs):

    filename = os.path.join(load_path, filename)
    record = np.load(filename, **kwargs)

    return record

def load_pickle_file(load_path, filename, **kwargs):
    filename_ = os.path.join(load_path, filename)
    with open(filename_, 'rb') as f:
        record = pickle.load(f)
    
    return record

