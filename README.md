# PC$^2$ & Noise of Web


This repo is the official Pytorch implementation of our paper:

> ***PC$^2$: Pseudo-Classification Based Pseudo-Captioning for Noisy Correspondence Learning in Cross-Modal Retrieval***  
> ***Authors**: Yue Duan, Zhangxuan Gu, Zhenzhe Ying, Lei Qi, Changhua Meng and Yinghuan Shi *
 
 
- Quick links: [[arXiv (coming soon)]() | [Published paper (coming soon)]() | [Poster (coming soon)]() | [Zhihu (coming soon)]() | [Code download]() | [Dataset download](https://drive.google.com/file/d/1MsR9GmRDUj4NoeL4xL8TXpes51JnpsrZ/view?usp=drive_link)]
 
 - Latest news:
     <!-- - We write a detailed introduction to this work on the [Zhihu](https://zhuanlan.zhihu.com/p/653555164). -->
     - Our paper is accepted by **ACM Multimedia (ACM MM) 2024** 🎉🎉. Thanks to users.
 - More of my works:
     - 🆕 **[LATEST]** Interested in the SSL in fine-grained visual classification (SS-FGVC)? 👉 Check out our AAAI'24 paper **SoC** [[arXiv](https://arxiv.org/abs/2312.12237) | [Repo](https://github.com/NJUyued/SoC4SS-FGVC/)].
     - Interested in robust SSL in MNAR setting with mismatched distributions? 👉 Check out our ECCV'22 paper **RDA** [[arXiv](https://arxiv.org/abs/2208.04619) | [Repo](https://github.com/NJUyued/RDA4RobustSSL)].
     - Interested in the conventional SSL or more application of complementary label in SSL? 👉 Check out our TNNLS paper **MutexMatch** [[arXiv](https://arxiv.org/abs/2203.14316) | [Repo](https://github.com/NJUyued/MutexMatch4SSL/)].

## Dataset Contribution: Noise of Web (NoW)
### Data Collection
We develop a new dataset named Noise of Web (NoW) for NCL. It contains 100K website image-meta description pairs (98,000 pairs for training, 1,000 for validation, and 1,000 for testing), which are open-sourced and can be crawled by anyone. NoW has two main characteristics: without human annotations and the noisy pairs are naturally captured. The source data of NoW is obtained by taking screenshots  when accessing web pages on mobile devices (resolution: 720$\times$1280) and parsing meta descriptions in html source code. In [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) (predecessor of NCL), each image in all datasets were preprocessed using Faster-RCNN detector provided by [Bottom-up Attention Model](https://github.com/peteanderson80/bottom-up-attention) to generate 36 region proposals, and each proposal was encoded as a 2048-dimensional feature. Thus, following NCR, we release our the features instead of raw images for fair comparison. However, we can not just use detection methods like Faster-RCNN to extract image features since it is trained on real-world animals and objects on MS-COCO. To tackle this, we adapt [APT](https://openaccess.thecvf.com/content/CVPR2023/papers/Gu_Mobile_User_Interface_Element_Detection_via_Adaptively_Prompt_Tuning_CVPR_2023_paper.pdf) as the detection model since it is trained on the mobile user interface data. Similar to existing datasets, we capture top 36 objects with their features for one image, that is, we can encode one image into a 36$\times$768 matrix. We do not artificially control the noise ratio, as all data is obtained automatically and randomly over the web. **The estimated noise ratio of this dataset is nearly 70%**. Due to the automated and non-human curated data collection process, the noise in NoW is highly authentic and intrinsic.  

<div align=center>

<img width="750px" src="/figures/now-1.jpg"> 
 
</div>

### Data Structure

```

|-- h5100k_precomp
|   |-- dev_caps.txt
|   |-- dev_ids.txt
|   |-- dev_ims.npy
|   |-- test_caps.txt
|   |-- test_ids.txt
|   |-- test_ims.npy
|   |-- train_caps.txt
|   |-- train_ids.txt
|   |-- train_ims.npy

```

*_ids.txt records the serial number of the data in the original 500k dataset. In the future, we may process and make the original dataset public.

### Download link

**https://drive.google.com/file/d/1MsR9GmRDUj4NoeL4xL8TXpes51JnpsrZ/view?usp=drive_link**



## Introduction

In the realm of cross-modal retrieval, seamlessly integrating diverse modalities within multimedia remains a formidable challenge, especially given the complexities introduced by *noisy correspondence learning (NCL)*. Such noise often stems from mismatched data pairs, a significant obstacle distinct from traditional noisy labels. This paper introduces Pseudo-Classification based Pseudo-Captioning (\pc) framework to address this challenge. \pc offers a threefold strategy: firstly, it establishes an auxiliary ``pseudo-classification'' task that interprets captions as categorical labels, steering the model to learn image-text semantic similarity through a non-contrastive mechanism. Secondly, unlike prevailing margin-based techniques, capitalizing on \pc's pseudo-classification capability, we generate pseudo-captions to provide more informative and tangible supervision for each mismatched pair. Thirdly, the oscillation of pseudo-classification is borrowed to assistant the correction of correspondence.

<div align=center>

<img width="750px" src="/figures/framework.jpg"> 
 
</div>

## Requirements
- numpy==1.21.6
- pandas==1.3.2
- Pillow==10.0.0
- scikit_learn==1.3.0
- torch==1.8.0
- torchvision==0.9.0
## How to Train
### Important Args
<!-- - `--last`: Set this flag to use the model of $\textrm{PRG}^{\textrm{Last}}$.
- `--alpha`: class invariance coefficient. By default, `--alpha 1` is set. When set `--last`, please set `--alpha 3`.
- `--nb`: Number of tracked bathches.
- `--mismatch [none/prg/cadr/darp/darp_reversed]` : Select the MNAR protocol. `none` means the conventional balanced setting. See Sec. 4 in our paper for the details of MNAR protocols.
- `--n0` : When `--mismatch prg`, this arg means the imbalanced ratio $N_0$ for labeled data; When `--mismatch [darp/darp_reversed]`, this arg means the imbalanced ratio $\gamma_l$ for labeled data.
- `--gamma` : When `--mismatch cadr`, this arg means the imbalanced ratio $\gamma$ for labeled data. When `--mismatch prg`, this arg means the imbalanced ratio $\gamma$ for unlabeled data; When `--mismatch DARP/DARP_reversed`, this arg means the imbalanced ratio $\gamma_u$ for unlabeled data. 
- `--num_labels` : Amount of labeled data used in conventional balanced setting. 
- `--net` : By default, Wide ResNet (WRN-28-2) are used for experiments. If you want to use other backbones for tarining, set `--net [resnet18/preresnet/cnn13]`. We provide alternatives as follows: ResNet-18, PreAct ResNet and CNN-13. -->
- `--data_name {coco,f30k,cc152k,now100k_precomp}_precomp` and `--data_path`  : Your dataset name and path.  

### Training with Single GPU

We recommend using a single NVIDIA Tesla A100 80G for training to better reproduce our results. Multi-GPU training is feasible, but our results are all obtained from single GPU training.

```
python ./PC2/run.py --world-size 1 --rank 0 --gpu [0/1/...] @@@other args@@@
```
### Training with Multi-GPUs


- Using DistributedDataParallel with single node

```
python ./PC2/run.py --world-size 1 --rank 0 --multiprocessing-distributed @@@other args@@@
```

## Examples of Running
By default, the model and `dist&index.txt` will be saved in `\--save_dir\--save_name`. The file `dist&index.txt` will display detailed settings of MNAR. This code assumes 1 epoch of training, but the number of iterations is 2\*\*20. For CIFAR-100, you need set `--widen_factor 8` for WRN-28-8 whereas WRN-28-2 is used for CIFAR-10.  Note that you need set `--net resnet18` for mini-ImageNet. 

### MNAR Settings
#### CADR's protocol in Tab. 1
- CIFAR-10 with $\gamma=20$
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch cadr --gamma 20 --gpu 0
```

- CIFAR-100 with $\gamma=50$ 
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar100 --dataset cifar100 --num_classes 100 --num_labels 400 --mismatch cadr --gamma 50 --gpu 0 --widen_factor 8
```

- mini-ImageNet with $\gamma=50$ 
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name miniimage --dataset miniimage --num_classes 100 --num_labels 1000 --mismatch cadr --gamma 50 --gpu 0 --net resnet18 
```

#### Our protocol in Tab. 2
- CIFAR-10 with 40 labels and $N_0=10$
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch prg --n0 10 --gpu 0
```

- CIFAR-100 with 400 labels and $N_0=40$ 
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar100 --dataset cifar100 --num_classes 100 --num_labels 400 --mismatch prg --n0 40 --gpu 0 --widen_factor 8
```

- mini-ImageNet with 1000 labels and $N_0=40$
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name miniimage --dataset miniimage --num_classes 100 --num_labels 1000 --mismatch prg --n0 40 --gpu 0 --net resnet18 
```

#### Our protocol in Fig. 6(a)
- CIFAR-10 with 40 labels, $N_0=10$ and $\gamma=5$ 

```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch prg --n0 10 --gamma 5 --gpu 0
```


#### Our protocol in Tab. 10
- CIFAR-10 with 40 labels and $\gamma=20$ 

```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch prg --gamma 20 --gpu 0
```

#### DARP's protocol in Fig. 6(a)
- CIFAR-10 with $\gamma_l=100$ and $\gamma_u=1$ 

```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --mismatch darp --n0 100 --gamma 1 --gpu 0
```


- CIFAR-10 with $\gamma_l=100$ and $\gamma_u=100$ (reversed) 
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --mismatch darp_reversed --n0 100 --gamma 100 --gpu 0
```


### Conventional Setting 
#### Matched and balanced distribution in Tab. 11
- CIFAR-10 with 40 labels

```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40  --gpu 0
```

## Resume Training and Evaluation
If you restart the training from normal checkpoints, please use `--resume --model_path @your_weight_path`.

If you restart the training from warmup checkpoints, please use `--model_path @your_warmup_weight_path`.

For evaluation, run

```
python ./PC2/evaluation.py --data_path @your_data_path --model_path @your_weight_path --gpu @your_gpu_id
```
    
By default, your evaluation process will directly use the dataset name saved in your checkpoint.




<!-- ## Citation
Please cite our paper if you find PRG useful:

```
@inproceedings{duan2023towards,
  title={Towards Semi-supervised Learning with Non-random Missing Labels},
  author={Duan, Yue and Zhao, Zhen and Qi, Lei and Zhou, Luping and Wang, Lei and Shi, Yinghuan},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

or

```
@article{duan2023towards,
  title={Towards Semi-supervised Learning with Non-random Missing Labels},
  author={Duan, Yue and Zhao, Zhen and Qi, Lei and Zhou, Luping and Wang, Lei and Shi, Yinghuan},
  journal={arXiv preprint arXiv:2308.08872},
  year={2023}
}
``` -->



