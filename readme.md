# Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning

This is the official implementation of the paper "Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning" based on PyTorch. 

## Overview

We introduce a novel generative retrieval framework named **SEATER**,  which learns **SE**m**A**ntic **T**ree-structured item identifi**ER**s using an encoder-decoder structure. 

The key implementation of SEATER can be found in `./model/SEATER.py`.
The overview is illustrated as follows:

| ![Image 1](asset/model.png) | ![Image 2](asset/identifiers.png) |
|------------------------|------------------------|



## Reproducibility

Here we share the code, data and some experiments for reproducibility on the public datasets.

### Experimental Settings

All the hyper-parameters of models can be found in `./config/`.
We list the details of datasets in `./config/const.py`, such as item count, user count, maximum history length for user modeling. The essential model-agnostic hyper-parameters like 'batch size' and 'the number of epochs' are listed in `main.py`.

### Datasets

The data and corresponding SEATER's checkpoints can be downloaded from this link [https://1drv.ms/u/s!AuS9Xkv_PPtMhk5brWLFIo-KgfPt?e=wDZEaf](https://1drv.ms/u/s!AuS9Xkv_PPtMhk5brWLFIo-KgfPt?e=wDZEaf). After downloading, place the extracted files in the corresponding locations within the './data' folder.

### Requirements
```
python==3.9.16
cudatoolkit==11.1
torch==1.10.0
tqdm==4.65.0
tensorboard==2.13.0
PyYAML==6.0
pyarrow==12.0.0
numpy==1.24.3
pandas==2.0.1
scikit-learn==1.2.2
scipy==1.10.1
k-means-constrained==0.7.2
```


### Model Training & Evaluation

Run experiments in command line:

```
# Yelp
python3 main.py --name SEATER_Yelp --dataset_name Yelp --gpu_id 0 --model SEATER --vocab 8

# Books
python3 main.py --name SEATER_Books --dataset_name Books --gpu_id 0 --model SEATER --vocab 16

# News
python3 main.py --name SEATER_News --dataset_name MIND --gpu_id 0 --model SEATER --vocab 8
```

Note that we have provided **training logs** for three datasets respectively in `./workspace/Books`.

### Environments

We conducted the experiments based on the following environments:

```
CUDA Version: 11.1
OS: CentOS Linux release 7.4.1708 (Core)
GPU: The NVIDIA® T4 GPU
CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz
```

### Citation
Please cite our paper if you use this repository.

```
@misc{si2023SEATER,
      title={Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning}, 
      author={Zihua Si and Zhongxiang Sun and Jiale Chen and Guozhang Chen and Xiaoxue Zang and Kai Zheng and Yang Song and Xiao Zhang and Jun Xu and Kun Gai},
      year={2023},
      eprint={2309.13375},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2309.13375}, 
}
```