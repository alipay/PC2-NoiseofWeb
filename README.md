# PC2-NoiseofWeb


This repo is the official Pytorch implementation of our paper:

> ***PC2: Pseudo-Classification Based Pseudo-Captioning for Noisy Correspondence Learning in Cross-Modal Retrieval***  
> ***Authors**: Yue Duan, Zhangxuan Gu, Zhenzhe Ying, Lei Qi, Changhua Meng and Yinghuan Shi*
 
 
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
We develop a new dataset named **Noise of Web (NoW)** for NCL. It contains 100K website image-meta description pairs (**98,000 pairs for training, 1,000 for validation, and 1,000 for testing**), which are open-sourced and can be crawled by anyone. NoW has two main characteristics: *without human annotations and the noisy pairs are naturally captured*. The source data of NoW is obtained by taking screenshots when accessing web pages on mobile devices (resolution: 720 $\times$ 1280) and parsing meta descriptions in html source code. In [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) (predecessor of NCL), each image in all datasets were preprocessed using Faster-RCNN detector provided by [Bottom-up Attention Model](https://github.com/peteanderson80/bottom-up-attention) to generate 36 region proposals, and each proposal was encoded as a 2048-dimensional feature. Thus, following NCR, we release our the features instead of raw images for fair comparison. However, we can not just use detection methods like Faster-RCNN to extract image features since it is trained on real-world animals and objects on MS-COCO. To tackle this, we adapt [APT](https://openaccess.thecvf.com/content/CVPR2023/papers/Gu_Mobile_User_Interface_Element_Detection_via_Adaptively_Prompt_Tuning_CVPR_2023_paper.pdf) as the detection model since it is trained on the mobile user interface data. Similar to existing datasets, we capture top 36 objects with their features for one image, that is, we can encode one image into a 36 $\times$ 768 matrix. We do not artificially control the noise ratio, as all data is obtained automatically and randomly over the web. **The estimated noise ratio of this dataset is nearly 70%**. Due to the automated and non-human curated data collection process, the noise in NoW is highly authentic and intrinsic.  

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

Please note that since our raw data contains some sensitive business data, we only provide the **encoded image features** (\*_ims.npy) and the **token ids of the text tokenized using [Tokenizers](https://github.com/huggingface/tokenizers)** (\*_caps.txt). **Our vocabulary size is set to 60,000**. \*_ids.txt records the serial number of the data in the original 500k dataset. In the future, we may process and make the original dataset public.


### Download link

**https://drive.google.com/file/d/1MsR9GmRDUj4NoeL4xL8TXpes51JnpsrZ/view?usp=drive_link**

### Usage

Please see the code snippet in `co_train.py`, `data.py`, `evaluation.py` and `run.py` containing the `now100k_precomp` string to process the NoW dataset for use in your own code. 

## PC2
### Introduction

In the realm of cross-modal retrieval, seamlessly integrating diverse modalities within multimedia remains a formidable challenge, especially given the complexities introduced by *noisy correspondence learning (NCL)*. Such noise often stems from mismatched data pairs, a significant obstacle distinct from traditional noisy labels. This paper introduces Pseudo-Classification based Pseudo-Captioning ($\text{PC}^2$) framework to address this challenge. $\text{PC}^2$ offers a threefold strategy: firstly, it establishes an auxiliary ``pseudo-classification'' task that interprets captions as categorical labels, steering the model to learn image-text semantic similarity through a non-contrastive mechanism. Secondly, unlike prevailing margin-based techniques, capitalizing on $\text{PC}^2$'s pseudo-classification capability, we generate pseudo-captions to provide more informative and tangible supervision for each mismatched pair. Thirdly, the oscillation of pseudo-classification is borrowed to assistant the correction of correspondence.

<div align=center>

<img width="750px" src="/figures/framework.jpg"> 
 
</div>

### Requirements
- matplotlib==3.4.2
- nltk==3.8.1
- numpy==1.22.3
- scikit_learn==0.24.2
- scipy==1.6.2
- torch==2.2.2

## How to Train
### Important Args
- `--lambda_en`: Entropy loss weight.
- `--proj_dim`: Dimensionality of the projection head. By default, `--proj_dim 128` is set. 
- `--nb`: Number of tracked bathches.
- `--img_dim` : Dimensionality of the image embedding. `--img_dim 2048` is used for {coco,f30k,cc152k} and please set it to `768` for now100k.
- `--warmup_epoch` : Epochs of warm up stage.
- `--warmup_epoch_2` : Epochs of training with clean data only.
- `--po_dir` : When `--resume`, use this path to load the PO data for resuming training.
- `--model_path` : Use this path to load the checkpoint for resuming training when `--resume`, or use this path to load the warmup checkpoint for resuming training without `--resume`.
- `--data_name {coco,f30k,cc152k,now100k}_precomp` and `--data_path`  : Your dataset name and path.  
- `--noise_ratio`: Noisy ratio for Flickr30K and MS-COCO.
- `--noise_file`: Noise file for the feproduction of noise correspondence.

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

**Please note that** our code is based on the [NCR implementation](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) and the original training code can only run on a single GPU (see [issue#4](https://github.com/XLearning-SCU/2021-NeurIPS-NCR/issues/4)). In order to make it easier for you to use our code, we tried to provide a multi-GPU parallel training version based on `DistributedDataParallel`. Unfortunately, there seem to be some bugs that we have not yet solved. The following error may occur during training: 

```
[rank0]:[E ProcessGroupNCCL.cpp:523] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=16349, OpType=ALLGATHER, NumelIn=1, NumelOut=2, Timeout(ms)=600000) ran for 600341 milliseconds before timing out.
[rank0]:[E ProcessGroupNCCL.cpp:537] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank0]:[E ProcessGroupNCCL.cpp:543] To avoid data inconsistency, we are taking the entire process down.
[rank0]:[E ProcessGroupNCCL.cpp:1182] [Rank 0] NCCL watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=16349, OpType=ALLGATHER, NumelIn=1, NumelOut=2, Timeout(ms)=600000) ran for 600341 milliseconds before timing out.
Exception raised from checkTimeout at ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:525 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x57 (0x7f1e10143d87 in /home/dy/.local/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x1e6 (0x7f1d990756e6 in /home/dy/.local/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x19d (0x7f1d99078c3d in /home/dy/.local/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x119 (0x7f1d99079839 in /home/dy/.local/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0xc9039 (0x7f1e1034f039 in /usr/local/miniconda3/envs/sharedEnv/bin/../lib/libstdc++.so.6)
frame #5: <unknown function> + 0x76db (0x7f1e14a626db in /lib/x86_64-linux-gnu/libpthread.so.0)
frame #6: clone + 0x3f (0x7f1e1478b61f in /lib/x86_64-linux-gnu/libc.so.6)

``` 

If any friends have insights on the occurrence of this problem, please contact us. At the same time, please rest assured that there will be no problem training with a single GPU (i.e., using ``--gpu`` to specify the GPU id).

## Examples of Running
By default, the warmup checkpoint `warmup_model_{}.pth.tar`, best checkpoint `model_best.pth.tar`, epoch checkpoint `checkpoint_{}.pth.tar` and PO data (the pseudo-preditions of pseudo-classification) `distri_bank_{}.pkl` will be saved in `./output_dir`. 

### NoW

```
python ./pc2/run.py --world-size 1 --rank 0 --gpu 0 --workers 8 --lr_update 30 --warmup_epoch 10 --warmup_epoch_2 25 --data_name h5100k_precomp --data_path ./data --vocab_path ./data/vocab --output_dir ./output --proj_dim 128 --lambda_en 10 --img_dim 768 
```


### Flickr30k

```
python ./pc2/run.py --world-size 1 --rank 0 --gpu 0 --workers 8 --warmup_epoch 5 --warmup_epoch_2 25 --data_name f30k_precomp --data_path ./data --vocab_path ./data/vocab  --output_dir ./output --proj_dim 128 --lambda_en 10 --noise_ratio 0.4 --noise_file noise_index/f30k_precomp_0.4
```


### MS-COCO

```
python ./pc2/run.py --world-size 1 --rank 0 --gpu 0 --workers 8 --warmup_epoch 5 --warmup_epoch_2 25 --data_name coco_precomp --data_path ./data --vocab_path ./data/vocab  --output_dir ./output --proj_dim 128 --lambda_en 10 --noise_ratio 0.4 --noise_file noise_index/coco_precomp_0.4
```


## Resume Training and Evaluation
- If you restart the training from normal checkpoints, please use `--resume --model_path @your_weight_path`.

- If you restart the training from warmup checkpoints, please use `--model_path @your_warmup_weight_path`.

- For evaluation, run

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



