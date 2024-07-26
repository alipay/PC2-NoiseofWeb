"""Evaluation"""

from __future__ import print_function
import os
import sys
import time
import json
from itertools import chain

import torch
import numpy as np
import argparse
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from collections import OrderedDict
from utils import AverageMeter, ProgressMeter
from data import get_dataset, get_loader


def encode_data(opt, model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(len(data_loader), [batch_time, data_time], prefix="Encode")

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    # max text length
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    image_ids = []
    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        data_time.update(time.time() - end)
        # image_ids.extend(img_ids)
        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(opt, images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros(
                (len(data_loader.dataset), img_emb.size(1), img_emb.size(2))
            )
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, : max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            progress.display(i)

        del images, captions
    return img_embs, cap_embs, cap_lens


def evalrank(model_path, data_path=None, vocab_path=None, data_loader=None, split="dev", fold5=False, gpu=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    # load model and options
    checkpoint = torch.load(model_path, torch.device('cuda'))
    opt = checkpoint["opt"]
    opt.gpu = gpu if gpu != None else opt.gpu
    print("training epoch: ", checkpoint["epoch"])
    opt.workers = 0
    print(opt)
    if data_path is not None:
        opt.data_path = data_path
    if vocab_path is not None:
        opt.vocab_path = vocab_path
    if data_loader == None:
        if opt.data_name == "coco_precomp":
            split = "testall"
        else:
            split = "test"
        # split = "dev"
        if opt.data_name != 'now100k_precomp':
            vocab = deserialize_vocab(
            os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
            )
            opt.vocab_size = len(vocab)
        else:
            vocab = None
        if opt.data_name == "cc152k_precomp":
            captions, images, image_ids, raw_captions = get_dataset(
                opt.data_path, opt.data_name, split, vocab, return_id_caps=True
            )
        else:
            captions, images = get_dataset(opt.data_path, opt.data_name, split, vocab)
        data_loader = get_loader(opt.data_name, captions, images, split, opt.batch_size, opt.workers)

    if opt.data_name in ["cc152k_precomp", "now100k_precomp"]:
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5


    # construct model
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    # load model state
    model_A.load_state_dict(checkpoint["model_A"])
    model_B.load_state_dict(checkpoint["model_B"])

    print("Computing results...")
    with torch.no_grad():
        img_embs_A, cap_embs_A, cap_lens_A = encode_data(opt, model_A, data_loader)
        img_embs_B, cap_embs_B, cap_lens_B = encode_data(opt, model_B, data_loader)

    print(
        "Images: %d, Captions: %d"
        % (img_embs_A.shape[0] / per_captions, cap_embs_A.shape[0])
    )

    if not fold5:
        # no cross-validation, full evaluation FIXME
        img_embs_A = np.array(
            [img_embs_A[i] for i in range(0, len(img_embs_A), per_captions)]
        )
        img_embs_B = np.array(
            [img_embs_B[i] for i in range(0, len(img_embs_B), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        sims_A = shard_attn_scores(
            model_A, img_embs_A, cap_embs_A, cap_lens_A, opt, shard_size=1000
        )
        sims_B = shard_attn_scores(
            model_B, img_embs_B, cap_embs_B, cap_lens_B, opt, shard_size=1000
        )
        sims = (sims_A + sims_B) / 2
        end = time.time()
        print("calculate similarity time:", end - start)

        # bi-directional retrieval
        r, rt = i2t(img_embs_A.shape[0], sims, per_captions, return_ranks=True)
        ri, rti = t2i(img_embs_A.shape[0], sims, per_captions, return_ranks=True)

        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
        return rsum
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            # 5fold split
            img_embs_shard_A = img_embs_A[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_A = cap_embs_A[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_A = cap_lens_A[i * 5000 : (i + 1) * 5000]

            img_embs_shard_B = img_embs_B[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_B = cap_embs_B[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_B = cap_lens_B[i * 5000 : (i + 1) * 5000]

            start = time.time()
            sims_A = shard_attn_scores(
                model_A,
                img_embs_shard_A,
                cap_embs_shard_A,
                cap_lens_shard_A,
                opt,
                shard_size=1000,
            )
            sims_B = shard_attn_scores(
                model_B,
                img_embs_shard_B,
                cap_embs_shard_B,
                cap_lens_shard_B,
                opt,
                shard_size=1000,
            )
            sims = (sims_A + sims_B) / 2
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(
                img_embs_shard_A.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(
                img_embs_shard_A.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        mean_i2t = (mean_metrics[0] + mean_metrics[1] + mean_metrics[2]) / 3
        print("Average i2t Recall: %.1f" % mean_i2t)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
        mean_t2i = (mean_metrics[5] + mean_metrics[6] + mean_metrics[7]) / 3
        print("Average t2i Recall: %.1f" % mean_t2i)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])
        print("-----------------------------------")
        return rsum


def shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=1000):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda(opt.gpu)
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda(opt.gpu)
                l = cap_lens[ca_start:ca_end]
                sim = model.forward_sim(im, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    return sims


# def i2t(npts, sims, per_captions=1, return_ranks=False):
#     """
#     Images->Text (Image Annotation)
#     Images: (N, n_region, d) matrix of images
#     Captions: (per_captions * N, max_n_word, d) matrix of captions
#     CapLens: (per_captions * N) array of caption lengths
#     sims: (N, per_captions * N) matrix of similarity im-cap
#     """
#     ranks = np.zeros(npts)
#     top1 = np.zeros(npts)
#     top5 = np.zeros((npts, 5), dtype=int)
#     retreivaled_index = []
#     for index in range(npts):
#         inds = np.argsort(sims[index])[::-1]
#         retreivaled_index.append(inds)
#         # Score
#         rank = 1e20
#         for i in range(per_captions * index, per_captions * index + per_captions, 1):
#             tmp = np.where(inds == i)[0][0]
#             if tmp < rank:
#                 rank = tmp
#         ranks[index] = rank
#         top1[index] = inds[0]
#         top5[index] = inds[0:5]

#     # Compute metrics
#     r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
#     r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
#     r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
#     medr = np.floor(np.median(ranks)) + 1
#     meanr = ranks.mean() + 1
#     if return_ranks:
#         return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
#     else:
#         return (r1, r5, r10, medr, meanr)


def i2t(npts, sims, per_captions=1, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    recorded_indices = set()  # 用于跟踪已记录的文本编号
    unique_indices = []  # 用于记录唯一的文本编号

    for index in range(npts):
        
        inds = np.argsort(sims[index])[::-1]
        retreivaled_index.append(inds)

        # 找到第一个未被记录的文本编号
        for i in inds:
            if i not in recorded_indices:
            # if True:
                recorded_indices.add(i)
                unique_indices.append(i)
                break

        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    # 将 unique_indices 保存到 txt 文件
    with open("unique_indices.txt", "w") as f:
        for idx in unique_indices:
            f.write(f"{idx}\n")

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)



def t2i(npts, sims, per_captions=1, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])[::-1]
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--data_path", default="data/", help="path to datasets"
    )
    parser.add_argument(
        "--model_path", default="", help="the path to the loaded models"
    )
    parser.add_argument(
        "--vocab_path",
        default="data/vocab",
        help="Path to saved vocabulary json files.",
    )
    opt = parser.parse_args()
    print(f"loading {opt.model_path}")
    evalrank(
        opt.model_path,
        data_path=opt.data_path,
        vocab_path=opt.vocab_path,
        split="test",
        fold5=False,
        gpu=opt.gpu
    )
