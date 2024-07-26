"""Training script"""

import os
import time
import copy
import shutil
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.mixture import GaussianMixture
from datetime import timedelta
from data import get_loader, get_dataset
from model import SGRAF, Projection_Head
from vocab import Vocabulary, deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores, evalrank
import scipy.stats as stats
from scipy.stats import wasserstein_distance
from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    adjust_learning_rate,
)
def z_score_normalization(data):
    mean = np.mean(data)  # 计算均值
    std = np.std(data)  # 计算标准差
    normalized_data = (data - mean) / std  # 进行 Z-Score 归一化
    return normalized_data

def alpha_divergence(p, q, alpha):
    if alpha == 1:
        divergence = np.sum(p * np.log(p/q))
    else: 
        divergence = (1.0 / (alpha * (alpha - 1.0))) * torch.sum(p ** alpha - alpha * (p ** (alpha - 1)) * q + q ** alpha)
    return divergence

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

def main(gpu, ngpus_per_node, opt):
    opt.gpu = gpu
    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            opt.rank = opt.rank * ngpus_per_node + gpu # compute global rank
        # set distributed group:
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
    # load Vocabulary Wrapper
    print("load and process dataset ...")
    if opt.data_name == "now100k_precomp":
        vocab = deserialize_vocab(
            os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
        )
        opt.vocab_size = 136130
    else:
        vocab = deserialize_vocab(
            os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
        )
        opt.vocab_size = len(vocab)

    # load dataset
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train", vocab
    )
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # data loader
    noisy_trainloader, data_size, clean_labels = get_loader(
        opt.data_name,
        captions_train,
        images_train,
        "warmup",
        opt.batch_size,
        opt.workers,
        opt.noise_ratio,
        opt.noise_file,
    )
    val_loader = get_loader(
        opt.data_name, captions_dev, images_dev, "dev", opt.batch_size, opt.workers
    )

    print("load and process testing dataset ...")
    if opt.data_name == "coco_precomp":
        split = "testall"
    else:
        split = "test"
    if opt.data_name == "now100k_precomp":
        vocab = deserialize_vocab(
            os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
        )
        opt.vocab_size = 136130
    else:
        vocab = deserialize_vocab(
            os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
        )
        opt.vocab_size = len(vocab)
    if opt.data_name == "cc152k_precomp":
        captions, images, image_ids, raw_captions = get_dataset(
            opt.data_path, opt.data_name, split, vocab, return_id_caps=True
        )
    else:
        captions, images = get_dataset(opt.data_path, opt.data_name, split, vocab)
    data_loader_test = get_loader(opt.data_name, captions, images, split, opt.batch_size, opt.workers)

    opt.batch_size = int(opt.batch_size / ngpus_per_node)
    # create models
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)
    

    best_rsum = 0
    best_rsum_test = 0
    best_rsum_test_5fold = 0
    best_rsum_test_full = 0
    start_epoch = 0

    # save the history of losses from two networks
    all_loss = [[], []]
    distri_bank_A = {}
    distri_bank_B = {}

    # Warmup
    print("\n* Warmup")
    if opt.model_path:
        if os.path.isfile(opt.model_path):
            checkpoint = torch.load(opt.model_path)
            model_A.load_state_dict(checkpoint["model_A"])
            model_B.load_state_dict(checkpoint["model_B"])
            print(
                "=> load warmup checkpoint '{}' (epoch {})".format(
                    opt.model_path, checkpoint["epoch"]
                )
            )
            if opt.po_dir=="":
                opt.po_dir = checkpoint["opt"].output_dir
            if opt.resume:
                # bank_name_A = "distri_bank_A.pkl"
                # bank_name_B = "distri_bank_B.pkl"
                bank_name_A = "warmup_distri_bank_A.pkl"
                bank_name_B = "warmup_distri_bank_B.pkl"
            else:
                bank_name_A = "warmup_distri_bank_A.pkl"
                bank_name_B = "warmup_distri_bank_B.pkl"
            if os.path.exists(opt.po_dir) and os.path.isfile(os.path.join(opt.po_dir, bank_name_A)) and os.path.isfile(os.path.join(opt.po_dir, bank_name_B)):
                print("=> resume distribution bank from '{}'".format(opt.po_dir))
                with open(os.path.join(opt.po_dir, bank_name_A), 'rb') as f:
                    distri_bank_A = pickle.load(f)
                with open(os.path.join(opt.po_dir, bank_name_B), 'rb') as f:
                    distri_bank_B = pickle.load(f)
            print("\nValidattion ...")
            if opt.resume:
                print(checkpoint.keys())
                model_A.optimizer.load_state_dict(checkpoint['optimizer_A'])
                model_B.optimizer.load_state_dict(checkpoint['optimizer_B'])
                print(
                "=> resume training from checkpoint '{}' (epoch {})".format(
                    opt.model_path, checkpoint["epoch"]
                    )
                )
                start_epoch = checkpoint["epoch"]
                # opt = checkpoint["opt"]
                best_rsum = validate(opt, val_loader, [model_A, model_B])
                
            else:
                validate(opt, val_loader, [model_A, model_B])
        else:
            raise Exception(
                "=> no checkpoint found at '{}'".format(opt.model_path)
            )
    else:
        epoch = 0
        for epoch in range(0, opt.warmup_epoch):
            print("[{}/{}] Warmup model_A".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_A, epoch, distri_bank_A)
            print("[{}/{}] Warmup model_B".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_B, epoch, distri_bank_B)
            with open(os.path.join(opt.output_dir, "warmup_distri_bank_A.pkl"), "wb") as file:
                pickle.dump(distri_bank_A, file)
            with open(os.path.join(opt.output_dir, "warmup_distri_bank_B.pkl"), "wb") as file:
                pickle.dump(distri_bank_B, file)

        save_checkpoint(
            {
                "epoch": epoch,
                "model_A": model_A.state_dict(),
                "model_B": model_B.state_dict(),
                "opt": opt,
            },
            is_best=False,
            filename="warmup_model_{}.pth.tar".format(epoch),
            prefix=opt.output_dir + "/",
        )
        
        # evaluate on validation set
        print("\nValidattion ...")
        validate(opt, val_loader, [model_A, model_B])

    # save the history of losses from two networks
    all_loss = [[], []]
    if not opt.resume:
        model_A.optimizer = torch.optim.Adam(model_A.params, lr=opt.learning_rate)
        model_B.optimizer = torch.optim.Adam(model_B.params, lr=opt.learning_rate)
        # print("=> resume distribution bank from '{}'".format(opt.output_dir))
        # with open(os.path.join(opt.output_dir, 'warmup_distri_bank_A.pkl'), 'rb') as f:
        #     distri_bank_A = pickle.load(f)
        # with open(os.path.join(opt.output_dir, 'warmup_distri_bank_B.pkl'), 'rb') as f:
        #     distri_bank_B = pickle.load(f)  
    
    print("\n* Co-training")
    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print("\nEpoch [{}/{}]".format(epoch, opt.num_epochs))
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)
        with open(os.path.join(opt.output_dir, "distri_bank_A.pkl"), "wb") as file:
            pickle.dump(distri_bank_A, file)
        with open(os.path.join(opt.output_dir, "distri_bank_B.pkl"), "wb") as file:
            pickle.dump(distri_bank_B, file)
        # # Dataset split (labeled, unlabeled)
        # # Dataset split (labeled, unlabeled)
        print("Split dataset ...")
        eval_function = eval_train_cc if opt.data_name in ["cc152k_precomp", "now100k_precomp"] else eval_train
        prob_A, prob_B, pred_A, pred_B, prob_ctt_A, prob_ctt_B, pred_ctt_A, pred_ctt_B, all_loss = eval_function(
            opt,
            model_A,
            model_B,
            distri_bank_A, 
            distri_bank_B,
            noisy_trainloader,
            data_size,
            all_loss,
            clean_labels,
            epoch,
            opt.noise_file,
            opt.output_dir,
        )

        # pred_A = split_prob(prob_A, opt.p_threshold)
        # pred_B = split_prob(prob_B, opt.p_threshold)

        print("\nModel A training ...")
        # train model_A
        labeled_trainloader, unlabeled_trainloader = get_loader(
            opt.data_name,
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_B,
            prob=prob_B,
            ctt_pred=pred_ctt_B,
            ctt_probability=prob_ctt_B
        )
        train(opt, model_A, model_B, distri_bank_A, labeled_trainloader, unlabeled_trainloader, epoch)

        print("\nModel B training ...")
        # train model_B
        labeled_trainloader, unlabeled_trainloader = get_loader(
            opt.data_name,
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_A,
            prob=prob_A,
            ctt_pred=pred_ctt_A,
            ctt_probability=prob_ctt_A
        )
        train(opt, model_B, model_A, distri_bank_B, labeled_trainloader, unlabeled_trainloader, epoch)

        if epoch == opt.warmup_epoch_2:
            best_rsum = 0
        print("\nValidattion ...")
        # evaluate on validation set
        rsum = validate(opt, val_loader, [model_A, model_B])

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            print("\nBest validattion!")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "optimizer_A": model_A.optimizer.state_dict(),
                    "optimizer_B": model_B.optimizer.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                # filename="checkpoint_{}.pth.tar".format(epoch),
                filename="checkpoint_best_validattion.pth.tar",
                prefix=opt.output_dir + "/",
            )

            print("\nTesting ...")
            if opt.data_name == "coco_precomp":
                print("5 fold validation")
                rsum_test_5fold = evalrank(
                    os.path.join(opt.output_dir, "model_best.pth.tar"),
                    # split="testall",
                    data_loader=data_loader_test,
                    fold5=True,
                )
                print("full validation")
                # rsum_test_full = evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), split="testall")
                rsum_test_full = evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), data_loader=data_loader_test)
                is_best = rsum_test_5fold > best_rsum_test_5fold
                if is_best:
                    best_rsum_test_5fold = rsum_test_5fold
                    print("\nBest testing over 5 fold!")
                is_best = rsum_test_full > best_rsum_test_full
                if is_best:
                    best_rsum_test_full = rsum_test_full
                    print("\nBest testing over full 5K!")
            else:
                rsum_test = evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), data_loader=data_loader_test)
                is_best = rsum_test > best_rsum_test    
                if is_best:
                    best_rsum_test = rsum_test
                    print("\nBest testing!")


def train(opt, net, net2, distri_bank, labeled_trainloader, unlabeled_trainloader=None, epoch=None):
    """
    One epoch training.
    """
    # losses = AverageMeter("loss", ":.4e")
    triplet_losses_l = AverageMeter("triplet_losses_l", ":.4e")
    ce_losses_img_l = AverageMeter("ce_losses_img_l", ":.4e")
    ce_losses_cap_l = AverageMeter("ce_losses_cap_l", ":.4e")
    en_losses_img_l = AverageMeter("en_losses_img_l", ":.4e")
    en_losses_cap_l = AverageMeter("en_losses_cap_l", ":.4e")
    triplet_losses_u = AverageMeter("triplet_losses_u", ":.4e")
    # ce_losses_img_u = AverageMeter("ce_losses_img_u", ":.4e")
    # ce_losses_cap_u = AverageMeter("ce_losses_cap_u", ":.4e")
    # en_losses_img_u = AverageMeter("en_losses_img_u", ":.4e")
    # en_losses_cap_u = AverageMeter("en_losses_cap_u", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(labeled_trainloader),
        [batch_time, data_time, triplet_losses_l, ce_losses_img_l, ce_losses_cap_l, en_losses_img_l, en_losses_cap_l, 
         triplet_losses_u],
        #  , ce_losses_img_u, ce_losses_cap_u, en_losses_img_u, en_losses_cap_u],
        prefix="Training Step",
    )

    # fix one network and train the other
    net.train_start()
    net2.val_start()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    labels_l = []
    pred_labels_l = []
    labels_u = []
    pred_labels_u = []
    end = time.time()
    for i, batch_train_data in enumerate(labeled_trainloader):
        (
            batch_images_l,
            batch_text_l,
            batch_lengths_l,
            batch_ids_l,
            batch_ids_ori_l,
            batch_labels_l,
            batch_prob_l,
            batch_ctt_labels_l,
            batch_ctt_prob_l,
            batch_clean_labels_l,
        ) = batch_train_data
        batch_size = batch_images_l.size(0)
        labels_l.append(batch_clean_labels_l)
        # unlabeled data
        try:
            (
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                batch_ids_u,
                batch_ids_ori_u,
                batch_clean_labels_u,
            ) = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                batch_ids_u,
                batch_ids_ori_u,
                batch_clean_labels_u,
            ) = next(unlabeled_train_iter)
        labels_u.append(batch_clean_labels_u)
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            batch_prob_l = batch_prob_l.cuda(opt.gpu)
            batch_labels_l = batch_labels_l.cuda(opt.gpu)
            batch_ctt_prob_l = batch_ctt_prob_l.cuda(opt.gpu)
            batch_ctt_labels_l = batch_ctt_labels_l.cuda(opt.gpu)
        # label refinement
        with torch.no_grad():
            net.val_start()
            # labeled data
            # pl = net.predict(batch_images_l, batch_text_l, batch_lengths_l)
            ptl = batch_prob_l * batch_labels_l + (1 - batch_prob_l) * batch_ctt_prob_l * batch_ctt_labels_l
            # ptl = batch_prob_l * batch_labels_l 
            # ptl = batch_prob_l * batch_labels_l + (1 - batch_prob_l) * pl
            targets_l = ptl.detach()
            pred_labels_l.append(ptl.cpu().numpy())

        #     # unlabeled data
        #     pu1 = net.predict(batch_images_u, batch_text_u, batch_lengths_u)
        #     pu2 = net2.predict(batch_images_u, batch_text_u, batch_lengths_u)
        #     ptu = (pu1 + pu2) / 2
        #     targets_u = ptu.detach()
        #     targets_u = targets_u.view(-1, 1)
        #     pred_labels_u.append(ptu.cpu().numpy())
        # targets_l = torch.ones_like(batch_prob_l)
        targets_u = torch.ones(batch_images_u.size(0)).cuda(opt.gpu)

        # drop last batch if only one sample (batch normalization require)
        if batch_images_l.size(0) == 1 or batch_images_u.size(0) == 1:
            break
        net.train_start()
        # train with labeled + unlabeled data  exponential or linear
        # print('epoch ',i)
        triplet_loss_l, ce_loss_img_l, ce_loss_cap_l, en_loss_img_l, en_loss_cap_l = net.train(
            opt,
            batch_images_l,
            batch_text_l,
            batch_lengths_l,
            None,
            batch_ids_ori_l,
            None,
            epoch=epoch,
            labels=targets_l,
            hard_negative=True,
            soft_margin=opt.soft_margin,
            mode="lb_train",
        )
        # triplet_loss_u = net.train(
        #         batch_images_u,
        #         batch_text_u,
        #         batch_lengths_u,
        #         batch_ids_u,
        #         distri_bank,
        #         epoch=epoch,
        #         labels=targets_u,
        #         hard_negative=True,
        #         soft_margin=opt.soft_margin,
        #         mode="ulb_train",
        #         images_ulb_train=batch_images_l,
        #         captions_ulb_train=batch_text_l,
        #         lengths_ulb_train=batch_lengths_l,
        #         ids_ulb_train=batch_ids_l
        #     )
        # if epoch < (opt.num_epochs // 2):
        if epoch < opt.warmup_epoch_2:
            triplet_loss_u = 0
        else:
            triplet_loss_u = net.train(
                opt,
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                batch_ids_u,
                batch_ids_ori_u,
                distri_bank,
                epoch=epoch,
                labels=targets_u,
                hard_negative=True,
                soft_margin=opt.soft_margin,
                mode="ulb_train",
                images_ulb_train=batch_images_l,
                captions_ulb_train=batch_text_l,
                lengths_ulb_train=batch_lengths_l,
                ids_ulb_train=batch_ids_l,
                ids_ori_ulb_train=batch_ids_ori_l
            )

        # loss = loss_l + loss_u
        # losses.update(loss, batch_images_l.size(0) + batch_images_u.size(0))
        triplet_losses_l.update(triplet_loss_l, batch_images_l.size(0))
        ce_losses_img_l.update(ce_loss_img_l, batch_images_l.size(0))
        ce_losses_cap_l.update(ce_loss_cap_l, batch_images_l.size(0))
        en_losses_img_l.update(en_loss_img_l, batch_images_l.size(0))
        en_losses_cap_l.update(en_loss_cap_l, batch_images_l.size(0))
        triplet_losses_u.update(triplet_loss_u, batch_images_u.size(0))
        # ce_losses_img_u.update(ce_loss_img_u, batch_images_u.size(0))
        # ce_losses_cap_u.update(ce_loss_cap_u, batch_images_u.size(0))
        # en_losses_img_u.update(en_loss_img_u, batch_images_u.size(0))
        # en_losses_cap_u.update(en_loss_cap_u, batch_images_u.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if i % opt.log_step == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(current_time, end=' ')
            progress.display(i)


def warmup(opt, train_loader, model, epoch, distri_bank):
    # average meters to record the training statistics
    # losses = AverageMeter("loss", ":.4e")
    triplet_losses_l = AverageMeter("triplet_losses_l", ":.4e")
    ce_losses_img_l = AverageMeter("ce_losses_img_l", ":.4e")
    ce_losses_cap_l = AverageMeter("ce_losses_cap_l", ":.4e")
    en_losses_img_l = AverageMeter("en_losses_img_l", ":.4e")
    en_losses_cap_l = AverageMeter("en_losses_cap_l", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, triplet_losses_l, ce_losses_img_l, ce_losses_cap_l, en_losses_img_l, en_losses_cap_l], prefix="Warmup Step"
    )

    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break

        model.train_start()

        # Update the model
        triplet_loss_l, ce_loss_img_l, ce_loss_cap_l, en_loss_img_l, en_loss_cap_l = model.train(opt, images, captions, lengths, ids, None, distri_bank, mode=opt.warmup_type)
        triplet_losses_l.update(triplet_loss_l, images.size(0))
        ce_losses_img_l.update(ce_loss_img_l, images.size(0))
        ce_losses_cap_l.update(ce_loss_cap_l, images.size(0))
        en_losses_img_l.update(en_loss_img_l, images.size(0))
        en_losses_cap_l.update(en_loss_cap_l, images.size(0))
        # losses.update(loss, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(current_time, end=' ')
            progress.display(i)
        # if i > 100:
        #     break


def validate(opt, val_loader, models=[]):
    # compute the encoding for all the validation images and captions
    if opt.data_name in ["cc152k_precomp", "now100k_precomp"]:
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    Eiters = models[0].Eiters
    sims_mean = 0
    count = 0
    for ind in range(len(models)):
        count += 1
        print("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            opt, models[ind], val_loader, opt.log_step
        )

        # clear duplicate 5*images and keep 1*images FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        print("Computing similarity from model {}".format(ind))
        sims_mean += shard_attn_scores(
            models[ind], img_embs, cap_embs, cap_lens, opt, shard_size=100
        )
        end = time.time()
        print(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )

    # average the sims
    sims_mean = sims_mean / count

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1, r5, r10, medr, meanr
        )
    )

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    return r_sum


def eval_train(
    opt, model_A, model_B, distri_bank_A, distri_bank_B, data_loader, data_size, all_loss, clean_labels, epoch, noise_file, dir_path
):
    """
    Compute per-sample loss and prob
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time], prefix="Computinng losses"
    )
    distri_bank_key_A = set(distri_bank_A.keys())
    distri_bank_key_B = set(distri_bank_B.keys())
    model_A.val_start()
    model_B.val_start()
    losses_A = np.zeros(data_size)
    losses_B = np.zeros(data_size)
    losses_A_clean = []
    losses_A_noisy = []
    losses_B_clean = []
    losses_B_noisy = []

    ctt_A = np.zeros(data_size)
    ctt_B = np.zeros(data_size)
    ctt_A_clean = []
    ctt_A_noisy = []
    ctt_B_clean = []
    ctt_B_noisy = []
    if os.path.exists(noise_file):
        print("=> Compute per-sample loss and prob: load noisy index from {}".format(noise_file))
        noise_idx = set(np.load(noise_file).tolist())

    end = time.time()
    total_ids = torch.zeros(data_size)
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
            # compute the loss
            loss_A, pseudo_label_A = model_A.train(images, captions, lengths, ids, distri_bank_A, mode="eval_loss")
            loss_B, pseudo_label_B = model_B.train(images, captions, lengths, ids, distri_bank_B, mode="eval_loss")
            loss_A = loss_A.cpu().numpy()
            loss_B = loss_B.cpu().numpy()
            pseudo_label_A = pseudo_label_A.cpu()
            pseudo_label_B = pseudo_label_B.cpu()
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]
                losses_B[ids[b]] = loss_B[b]
                total_ids[ids[b]] = ids[b]
                if ids[b] in noise_idx:
                    losses_A_noisy.append(loss_A[b])
                    losses_B_noisy.append(loss_B[b])
                else:
                    losses_A_clean.append(loss_A[b])
                    losses_B_clean.append(loss_B[b])

                if ids[b] in distri_bank_key_A:
                    # losses_A[ids[b]] = nn.functional.cosine_similarity(distri_bank_A[ids[b]], pseudo_label_A[b], dim=-1)
                    ctt_A[ids[b]] = F.kl_div(torch.from_numpy(distri_bank_A[ids[b]]).log(), pseudo_label_A[b], reduction='sum')
                    # losses_A[ids[b]] = alpha_divergence(distri_bank_A[ids[b]], pseudo_label_A[b], -torch.sum(distri_bank_A[ids[b]] * torch.log(distri_bank_A[ids[b]])))
                    # losses_A[ids[b]] = wasserstein_distance(distri_bank_A[ids[b]], pseudo_label_A[b].cpu().numpy())
                if ids[b] in distri_bank_key_B:
                    # losses_B[ids[b]] = nn.functional.cosine_similarity(distri_bank_B[ids[b]], pseudo_label_B[b], dim=-1)
                    ctt_B[ids[b]] = F.kl_div(torch.from_numpy(distri_bank_B[ids[b]]).log(), pseudo_label_B[b], reduction='sum')
                    # losses_B[ids[b]] = alpha_divergence(distri_bank_B[ids[b]], pseudo_label_B[b], -torch.sum(distri_bank_B[ids[b]] * torch.log(distri_bank_B[ids[b]])))
                    # losses_B[ids[b]] = wasserstein_distance(distri_bank_B[ids[b]], pseudo_label_B[b].cpu().numpy())
            # ctt_A = ctt_A.numpy()
            # ctt_B = ctt_B.numpy()
            # for b in range(images.size(0)):  
                if ids[b] in noise_idx:
                    ctt_A_noisy.append(ctt_A[ids[b]])
                    ctt_B_noisy.append(ctt_B[ids[b]])
                else:
                    ctt_A_clean.append(ctt_A[ids[b]])
                    ctt_B_clean.append(ctt_B[ids[b]])

            # pseudo_label_A = model_A.train(images, captions, lengths, ids, distri_bank_A, mode="eval_loss_CTT")
            # pseudo_label_B = model_B.train(images, captions, lengths, ids, distri_bank_B, mode="eval_loss_CTT")
            # for b in range(images.size(0)):
            #     if ids[b] in distri_bank_A.keys():
            #         # losses_A[ids[b]] = nn.functional.cosine_similarity(distri_bank_A[ids[b]], pseudo_label_A[b], dim=-1)
            #         ctt_A[ids[b]] = F.kl_div(torch.from_numpy(distri_bank_A[ids[b]]).log(), pseudo_label_A[b].cpu(), reduction='sum')
            #         # losses_A[ids[b]] = alpha_divergence(distri_bank_A[ids[b]], pseudo_label_A[b], -torch.sum(distri_bank_A[ids[b]] * torch.log(distri_bank_A[ids[b]])))
            #         # losses_A[ids[b]] = wasserstein_distance(distri_bank_A[ids[b]], pseudo_label_A[b].cpu().numpy())
            #     if ids[b] in distri_bank_B.keys():
            #         # losses_B[ids[b]] = nn.functional.cosine_similarity(distri_bank_B[ids[b]], pseudo_label_B[b], dim=-1)
            #         ctt_B[ids[b]] = F.kl_div(torch.from_numpy(distri_bank_B[ids[b]]).log(), pseudo_label_B[b].cpu(), reduction='sum')
            #         # losses_B[ids[b]] = alpha_divergence(distri_bank_B[ids[b]], pseudo_label_B[b], -torch.sum(distri_bank_B[ids[b]] * torch.log(distri_bank_B[ids[b]])))
            #         # losses_B[ids[b]] = wasserstein_distance(distri_bank_B[ids[b]], pseudo_label_B[b].cpu().numpy())
            #     if ids[b] in noise_idx:
            #         ctt_A_noisy.append(ctt_A[ids[b]].cpu().numpy())
            #         ctt_B_noisy.append(ctt_B[ids[b]].cpu().numpy())
            #     else:
            #         ctt_A_clean.append(ctt_A[ids[b]].cpu().numpy())
            #         ctt_B_clean.append(ctt_B[ids[b]].cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_step == 0:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(current_time, end=' ')
                progress.display(i)
        # if i > 200:
        #     break

    # ids = np.concatenate(total_ids)
    ids = total_ids.numpy()

    # os.makedirs(os.path.join(dir_path, 'losses'), exist_ok=True)
    # np.save(os.path.join(dir_path, f'losses/losses_A_noisy_{epoch}.npy'), losses_A_noisy)
    # np.save(os.path.join(dir_path, f'losses/losses_B_noisy_{epoch}.npy'), losses_B_noisy)
    # np.save(os.path.join(dir_path, f'losses/losses_A_clean_{epoch}.npy'), losses_A_clean)
    # np.save(os.path.join(dir_path, f'losses/losses_B_clean_{epoch}.npy'), losses_B_clean)
    # np.save(os.path.join(dir_path, f'losses/losses_A_{epoch}.npy'), losses_A)
    # np.save(os.path.join(dir_path, f'losses/losses_B_{epoch}.npy'), losses_B)

    # os.makedirs(os.path.join(dir_path, 'ctt'), exist_ok=True)
    # np.save(os.path.join(dir_path, f'ctt/ctt_A_noisy_{epoch}.npy'), ctt_A_noisy)
    # np.save(os.path.join(dir_path, f'ctt/ctt_B_noisy_{epoch}.npy'), ctt_B_noisy)
    # np.save(os.path.join(dir_path, f'ctt/ctt_A_clean_{epoch}.npy'), ctt_A_clean)
    # np.save(os.path.join(dir_path, f'ctt/ctt_B_clean_{epoch}.npy'), ctt_B_clean)
    # np.save(os.path.join(dir_path, f'ctt/ctt_A_{epoch}.npy'), ctt_A)
    # np.save(os.path.join(dir_path, f'ctt/ctt_B_{epoch}.npy'), ctt_B)

    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    # losses_A = z_score_normalization(losses_A.cpu().numpy())
    all_loss[0].append(losses_A)
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())
    # losses_B = z_score_normalization(losses_B.cpu().numpy())
    all_loss[1].append(losses_B)

    ctt_A = (ctt_A - ctt_A.min()) / (ctt_A.max() - ctt_A.min())
    ctt_B = (ctt_B - ctt_B.min()) / (ctt_B.max() - ctt_B.min())


    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)
    idx_loss_A = np.where(input_loss_A >= 0.05)[0]
    idx_ctt_A = np.where(input_loss_A < 0.05)[0]
    idx_loss_B = np.where(input_loss_B >= 0.05)[0]
    idx_ctt_B = np.where(input_loss_B < 0.05)[0]

    input_ctt_A = ctt_A.reshape(-1, 1)
    input_ctt_B = ctt_B.reshape(-1, 1)

    # input_loss_A = input_loss_A[idx_loss_A]
    # input_loss_B = input_loss_B[idx_loss_B]
    # input_ctt_A = input_ctt_A[idx_ctt_A]
    # input_ctt_B = input_ctt_B[idx_ctt_B]

    print("\nFitting GMM of loss...")
    # fit a two-component GMM to the loss
    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_loss_A)
    prob_loss_A = gmm_A.predict_proba(input_loss_A)
    # gmm_A.fit(input_loss_A)
    # prob_A = gmm_A.predict_proba(input_loss_A)
    prob_loss_A = prob_loss_A[:, gmm_A.means_.argmin()]

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_loss_B)
    prob_loss_B = gmm_B.predict_proba(input_loss_B)
    # gmm_B.fit(input_loss_B)
    # prob_B = gmm_B.predict_proba(input_loss_B)
    prob_loss_B = prob_loss_B[:, gmm_B.means_.argmin()]

    pred_loss_A = split_prob(prob_loss_A, opt.p_threshold)
    pred_loss_B = split_prob(prob_loss_B, opt.p_threshold)
    sum_A = 0
    p_A = 0
    sum_B = 0
    p_B = 0
    for b in range(data_size):
        if pred_loss_A[b]==False:
            p_A = p_A + 1
            if ids[b] in noise_idx:
                sum_A = sum_A + 1
        if pred_loss_B[b]==False:
            p_B = p_B + 1
            if ids[b] in noise_idx:
                sum_B = sum_B + 1 
    print(f"Recall of spliting A loss: {sum_A/len(noise_idx)}")
    print(f"Recall of spliting B loss: {sum_B/len(noise_idx)}")
    print(f"Precision of spliting A loss: {sum_A/p_A if p_A != 0 else 0}")
    print(f"Precision of spliting B loss: {sum_B/p_B if p_B != 0 else 0}")
    print(f"Selected noisy data A: {p_A}")
    print(f"Selected noisy data B: {p_B}")


    # bmm_A = BetaMixture1D(max_iters=10)
    # bmm_A.fit(input_loss_A.cpu().numpy())
    # prob_A = bmm_A.posterior(input_loss_A.cpu().numpy(),0)

    # bmm_B = BetaMixture1D(max_iters=10)
    # bmm_B.fit(input_loss_B.cpu().numpy())
    # prob_B = bmm_B.posterior(input_loss_B.cpu().numpy(),0)

    print("\nFitting GMM of ctt...")
    # fit a two-component GMM to the loss
    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_ctt_A)
    prob_ctt_A = gmm_A.predict_proba(input_ctt_A)
    prob_ctt_A = prob_ctt_A[:, gmm_A.means_.argmin()]

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_ctt_B)
    prob_ctt_B = gmm_B.predict_proba(input_ctt_B)
    prob_ctt_B = prob_ctt_B[:, gmm_B.means_.argmin()]

    pred_ctt_A = split_prob(prob_ctt_A, opt.p_threshold)
    pred_ctt_B = split_prob(prob_ctt_B, opt.p_threshold)

    # pred_A = np.squeeze(np.empty_like(ctt_A.reshape(-1, 1).cpu().numpy()))
    # pred_A[idx_loss_A] = pred_loss_A
    # pred_A[idx_ctt_A] = pred_ctt_A
    # pred_B = np.squeeze(np.empty_like(ctt_A.reshape(-1, 1).cpu().numpy()))
    # pred_B[idx_loss_B] = pred_loss_B
    # pred_B[idx_ctt_B] = pred_ctt_B

    # prob_A = np.squeeze(np.empty_like(ctt_A.reshape(-1, 1).cpu().numpy()))
    # prob_A[idx_loss_A] = prob_loss_A
    # prob_A[idx_ctt_A] = prob_ctt_A
    # prob_B = np.squeeze(np.empty_like(ctt_A.reshape(-1, 1).cpu().numpy()))
    # prob_B[idx_loss_B] = prob_loss_B
    # prob_B[idx_ctt_B] = prob_ctt_B

    # sum_A = 0
    # p_A = 0
    # sum_B = 0
    # p_B = 0
    # for b in range(images.size(0)):
    #     if pred_A[b]==False:
    #         p_A = p_A + 1
    #     if pred_B[b]==False:
    #         p_B = p_B + 1
    #     if ids[b] in noise_idx and pred_A[b]==False:
    #         sum_A = sum_A + 1
    #     if ids[b] in noise_idx and pred_B[b]==False:
    #         sum_B = sum_B + 1 

    sum_A = 0
    p_A = 0
    sum_B = 0
    p_B = 0
    for b in range(data_size):
        if pred_ctt_A[b]==False:
            p_A = p_A + 1
            if ids[b] in noise_idx:
                sum_A = sum_A + 1
        if pred_ctt_B[b]==False:
            p_B = p_B + 1
            if ids[b] in noise_idx:
                sum_B = sum_B + 1 
    print(f"Recall of spliting A ctt: {sum_A/len(noise_idx)}")
    print(f"Recall of spliting B ctt: {sum_B/len(noise_idx)}")
    print(f"Precision of spliting A ctt: {sum_A/p_A if p_A != 0 else 0}")
    print(f"Precision of spliting B ctt: {sum_B/p_B if p_B != 0 else 0}")
    print(f"Selected noisy data A: {p_A}")
    print(f"Selected noisy data B: {p_B}")

    # if epoch < 20:
    #     return prob_loss_A, prob_loss_B, pred_loss_A, pred_loss_B, prob_ctt_A, prob_ctt_B, pred_ctt_A, pred_ctt_B, all_loss
    # else:
    #     return prob_ctt_A, prob_ctt_B, pred_ctt_A, pred_ctt_B, prob_loss_A, prob_loss_B, pred_loss_A, pred_loss_B, all_loss
    return prob_loss_A, prob_loss_B, pred_loss_A, pred_loss_B, prob_ctt_A, prob_ctt_B, pred_ctt_A, pred_ctt_B, all_loss
    # return prob_ctt_A, prob_ctt_B, pred_ctt_A, pred_ctt_B, prob_loss_A, prob_loss_B, pred_loss_A, pred_loss_B, all_loss


def eval_train_cc(
    opt, model_A, model_B, distri_bank_A, distri_bank_B, data_loader, data_size, all_loss, clean_labels, epoch, noise_file, dir_path
):
    """
    Compute per-sample loss and prob
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time], prefix="Computinng losses"
    )
    distri_bank_key_A = set(distri_bank_A.keys())
    distri_bank_key_B = set(distri_bank_B.keys())
    model_A.val_start()
    model_B.val_start()
    losses_A = np.zeros(data_size)
    losses_B = np.zeros(data_size)

    ctt_A = np.zeros(data_size)
    ctt_B = np.zeros(data_size)
    if os.path.exists(noise_file):
        print("=> Compute per-sample loss and prob: load noisy index from {}".format(noise_file))
        noise_idx = set(np.load(noise_file).tolist())

    end = time.time()
    total_ids = torch.zeros(data_size)
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
            # compute the loss
            loss_A, pseudo_label_A = model_A.train(opt, images, captions, lengths, ids, distri_bank_A, mode="eval_loss")
            loss_B, pseudo_label_B = model_B.train(opt, images, captions, lengths, ids, distri_bank_B, mode="eval_loss")
            loss_A = loss_A.cpu().numpy()
            loss_B = loss_B.cpu().numpy()
            pseudo_label_A = pseudo_label_A.cpu()
            pseudo_label_B = pseudo_label_B.cpu()
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]
                losses_B[ids[b]] = loss_B[b]
                total_ids[ids[b]] = ids[b]

                if ids[b] in distri_bank_key_A:
                    ctt_A[ids[b]] = F.kl_div(torch.from_numpy(distri_bank_A[ids[b]]).log(), pseudo_label_A[b], reduction='sum')
                if ids[b] in distri_bank_key_B:
                    ctt_B[ids[b]] = F.kl_div(torch.from_numpy(distri_bank_B[ids[b]]).log(), pseudo_label_B[b], reduction='sum')

            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_step == 0:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(current_time, end=' ')
                progress.display(i)
        # if i > 200:
        #     break

    ids = total_ids.numpy()

    os.makedirs(os.path.join(dir_path, 'losses'), exist_ok=True)
    np.save(os.path.join(dir_path, f'losses/losses_A_{epoch}.npy'), losses_A)
    np.save(os.path.join(dir_path, f'losses/losses_B_{epoch}.npy'), losses_B)

    os.makedirs(os.path.join(dir_path, 'ctt'), exist_ok=True)
    np.save(os.path.join(dir_path, f'ctt/ctt_A_{epoch}.npy'), ctt_A)
    np.save(os.path.join(dir_path, f'ctt/ctt_B_{epoch}.npy'), ctt_B)

    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    # losses_A = z_score_normalization(losses_A.cpu().numpy())
    all_loss[0].append(losses_A)
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())
    # losses_B = z_score_normalization(losses_B.cpu().numpy())
    all_loss[1].append(losses_B)

    ctt_A = (ctt_A - ctt_A.min()) / (ctt_A.max() - ctt_A.min())
    ctt_B = (ctt_B - ctt_B.min()) / (ctt_B.max() - ctt_B.min())


    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)
    idx_loss_A = np.where(input_loss_A >= 0.05)[0]
    idx_ctt_A = np.where(input_loss_A < 0.05)[0]
    idx_loss_B = np.where(input_loss_B >= 0.05)[0]
    idx_ctt_B = np.where(input_loss_B < 0.05)[0]

    input_ctt_A = ctt_A.reshape(-1, 1)
    input_ctt_B = ctt_B.reshape(-1, 1)


    print("\nFitting GMM of loss...")
    # fit a two-component GMM to the loss
    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_loss_A)
    prob_loss_A = gmm_A.predict_proba(input_loss_A)
    # gmm_A.fit(input_loss_A)
    # prob_A = gmm_A.predict_proba(input_loss_A)
    prob_loss_A = prob_loss_A[:, gmm_A.means_.argmin()]

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_loss_B)
    prob_loss_B = gmm_B.predict_proba(input_loss_B)
    # gmm_B.fit(input_loss_B)
    # prob_B = gmm_B.predict_proba(input_loss_B)
    prob_loss_B = prob_loss_B[:, gmm_B.means_.argmin()]

    pred_loss_A = split_prob(prob_loss_A, opt.p_threshold)
    pred_loss_B = split_prob(prob_loss_B, opt.p_threshold)


    print("\nFitting GMM of ctt...")
    # fit a two-component GMM to the loss
    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_ctt_A)
    prob_ctt_A = gmm_A.predict_proba(input_ctt_A)
    prob_ctt_A = prob_ctt_A[:, gmm_A.means_.argmin()]

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_ctt_B)
    prob_ctt_B = gmm_B.predict_proba(input_ctt_B)
    prob_ctt_B = prob_ctt_B[:, gmm_B.means_.argmin()]

    pred_ctt_A = split_prob(prob_ctt_A, opt.p_threshold)
    pred_ctt_B = split_prob(prob_ctt_B, opt.p_threshold)


    # if epoch < 20:
    #     return prob_loss_A, prob_loss_B, pred_loss_A, pred_loss_B, prob_ctt_A, prob_ctt_B, pred_ctt_A, pred_ctt_B, all_loss
    # else:
    #     return prob_ctt_A, prob_ctt_B, pred_ctt_A, pred_ctt_B, prob_loss_A, prob_loss_B, pred_loss_A, pred_loss_B, all_loss
    return prob_loss_A, prob_loss_B, pred_loss_A, pred_loss_B, prob_ctt_A, prob_ctt_B, pred_ctt_A, pred_ctt_B, all_loss
    # return prob_ctt_A, prob_ctt_B, pred_ctt_A, pred_ctt_B, prob_loss_A, prob_loss_B, pred_loss_A, pred_loss_B, all_loss

def split_prob(prob, threshld):
    if prob.min() > threshld:
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        threshld = np.sort(prob)[len(prob) // 100]
    # if (np.count_nonzero(prob > threshld) / len(prob)) < 0.4:
    #     print(
    #         "Enforce the 40% data with high probability to be labeled."
    #     )
    #     threshld = np.percentile(prob, 60)  
    #     pred = prob >= threshld
    #     count = np.count_nonzero(pred)
    #     if count > len(prob) * 0.4:
    #         indices = np.argsort(prob)[::-1][:int(len(prob) * 0.4)]
    #         pred = np.zeros_like(prob, dtype=bool)
    #         pred[indices] = True
    # else:
    #     pred = prob > threshld
    pred = prob > threshld
    return pred
