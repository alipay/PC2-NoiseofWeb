import os
os.environ["NCCL_DEBUG"] = "INFO"
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch

from evaluation import evalrank
from co_train import main
import torch.multiprocessing as mp

def run():

    # current_time
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    # Hyper Parameters
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument(
        "--data_path", default="data/", help="Path to datasets."
    )
    parser.add_argument(
        "--data_name", default="f30k_precomp", help="{coco,f30k,cc152k,now100k_precomp}_precomp"
    )
    parser.add_argument(
        "--tokenizer", default="bpe", help="{bpe,bert}"
    )
    parser.add_argument(
        "--vocab_path",
        default="data/vocab",
        help="Path to saved vocabulary json files.",
    )

    # ----------------------- training setting ----------------------#
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Size of a training mini-batch."
    )
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr_update",
        default=30,
        type=int,
        help="Number of epochs to update the learning rate.",
    )
    parser.add_argument(
        "--learning_rate", default=0.0002, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--workers", default=0, type=int, help="Number of data loader workers."
    )
    parser.add_argument(
        "--log_step",
        default=100,
        type=int,
        help="Number of steps to print and record the log.",
    )
    parser.add_argument(
        "--grad_clip", default=2.0, type=float, help="Gradient clipping threshold."
    )
    parser.add_argument("--margin", default=0.2, type=float, help="Rank loss margin.")

    # ------------------------- model setting -----------------------#
    parser.add_argument(
        "--lambda_en",
        default=1,
        type=int,
        help="Entropy loss weight.",
    )
    parser.add_argument(
        "--proj_dim",
        default=128,
        type=int,
        help="Dimensionality of the projection head.",
    )
    parser.add_argument(
        "--img_dim",
        default=2048,
        type=int,
        help="Dimensionality of the image embedding.",
    )
    parser.add_argument(
        "--word_dim",
        default=300,
        type=int,
        help="Dimensionality of the word embedding.",
    )
    parser.add_argument(
        "--embed_size",
        default=1024,
        type=int,
        help="Dimensionality of the joint embedding.",
    )
    parser.add_argument(
        "--sim_dim", default=256, type=int, help="Dimensionality of the sim embedding."
    )
    parser.add_argument(
        "--num_layers", default=1, type=int, help="Number of GRU layers."
    )
    parser.add_argument("--bi_gru", action="store_false", help="Use bidirectional GRU.")
    parser.add_argument(
        "--no_imgnorm",
        action="store_true",
        help="Do not normalize the image embeddings.",
    )
    parser.add_argument(
        "--no_txtnorm",
        action="store_true",
        help="Do not normalize the text embeddings.",
    )
    parser.add_argument("--module_name", default="SGR", type=str, help="SGR, SAF")
    parser.add_argument("--sgr_step", default=3, type=int, help="Step of the SGR.")

    # noise settings
    parser.add_argument("--noise_file", default="", help="noise_file")
    parser.add_argument("--noise_ratio", default=0.2, type=float, help="Noisy ratio")

    # model Settings
    parser.add_argument(
        "--no_co_training", action="store_true", help="No co-training for noisy label."
    )
    parser.add_argument(
        "--warmup_type", default='warmup_sele', help="Warmup with selected samples."
    )
    parser.add_argument("--warmup_epoch", default=1, type=int, help="Epochs of warm up stage.")
    parser.add_argument("--warmup_epoch_2", default=25, type=int, help="Epochs of training with clean data only.")
    parser.add_argument("--model_path", default="", help="The path to the loaded models")
    parser.add_argument("--po_dir", default="", help="The path to PO data for resuming training")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument(
        "--p_threshold", default=0.5, type=float, help="Clean probability threshold."
    )
    parser.add_argument(
        "--soft_margin", default="exponential", help="linear|exponential|sin"
    )

    # Runing Settings
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--seed", default=0, type=int, help="Random seed."
    )
    parser.add_argument(
        "--output_dir", default=os.path.join("output", current_time), help="Output dir."
    )
    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='Number of nodes for distributed training.')
    parser.add_argument('--rank', default=-1, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='Url used to set up distributed training.')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='Distributed backend.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # load arguments
    opt = parser.parse_args()

    # Output dir
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    if not opt.noise_file:
        opt.noise_file = os.path.join(
            opt.output_dir, opt.data_name + "_" + str(opt.noise_ratio) + ".npy"
        )

    if opt.data_name in ["cc152k_precomp", "now100k_precomp"]:
        opt.noise_ratio = 0
        opt.noise_file = ""
    
    # set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.random.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True

    

    # traing and evaluation
    print("\n*-------- Training --------*")
    # main(opt)
    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])
    
    #distributed: true if manually selected or if world_size > 1
    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed 
    ngpus_per_node = torch.cuda.device_count() # number of gpus of each node
    
    #divide the batch_size according to the number of nodes
    opt.batch_size = int(opt.batch_size / opt.world_size)
    
    if opt.multiprocessing_distributed:
        # now, opt.world_size means num of total processes in all nodes
        opt.world_size = ngpus_per_node * opt.world_size 
        print('run: ', opt.gpu)
        #args=(,) means the arguments of main_worker
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, opt)) 
    else:
        main(opt.gpu, ngpus_per_node, opt)


if __name__ == "__main__":
    run()
