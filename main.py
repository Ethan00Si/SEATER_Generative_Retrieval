import argparse
from utils.Context import ContextManager, DatasetManager

import torch
import numpy as np
import random 
import os


parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, help='experiment name', default='default')
parser.add_argument('--description', type=str, help='experiments details, used for log name', default='default')
parser.add_argument('--workspace', type=str, default='./workspace')

parser.add_argument('--dataset_name', type=str, default='MIND')
parser.add_argument('--use_cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=32, help='num of workers used for multi-processing data loading')
parser.add_argument('--tb', type=bool, help='whether use tensorboard (record metrics)', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--train_tb', type=bool, help='whether use tensorboard to record loss', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--verbose', type=bool, help='whether save model paremeters in tensorborad', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--model', type=str, help='which model to use', default='')
parser.add_argument('--batch_size', type=int, help='training batch_size', default=0)
parser.add_argument('--test_batch_size', type=int, help='testing batch_size', default=0)
parser.add_argument('--vocab', type=int, help='#branches of tree structure', default=8)
parser.add_argument('--random_seed', type=int, help='random seed', default=2023)

parser.add_argument('--new_config', type=str, help='update model config', default='')

args = parser.parse_args()

def setup_seed(seed):
    '''setting random seeds'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) 
setup_seed(args.random_seed)

# update paremeter w.r.t dataset
dataset_paras = {
    'Yelp': {'batch_size': 256, 'test_batch_size': 1024, 'epochs': 50},
    'MIND': {'batch_size': 256, 'test_batch_size': 1024, 'epochs': 50},
    'Books': {'batch_size': 2048, 'test_batch_size': 2048, 'epochs': 60},
}
if args.batch_size == 0 : args.batch_size = dataset_paras[args.dataset_name]['batch_size'] 
if args.test_batch_size == 0: args.test_batch_size = dataset_paras[args.dataset_name]['test_batch_size'] 
if args.epochs == 0: args.epochs = dataset_paras[args.dataset_name]['epochs'] 

new_config = {} if args.new_config == '' else eval(args.new_config)
cur_hyper_paras = [(k ,v) for k,v in new_config.items()]
new_name = f'{args.name}'
for k,v in cur_hyper_paras:
    new_name += f'_{k}_{v}'
args.name = new_name


print(args)

cm = ContextManager(args)
dm = DatasetManager(args)

trainer = cm.set_trainer(args, cm, dm, new_config)

trainer.train()
trainer.test()