"""SGRAF model"""
import math
from collections import OrderedDict

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """

    def __init__(
        self,
        vocab_size,
        word_dim,
        embed_size,
        num_layers,
        use_bi_gru=False,
        no_txtnorm=False,
    ):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(
            word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru
        )

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        """Handles variable size captions"""
        # embed word ids to vectors
        cap_emb = self.embed(captions)
        cap_emb = self.dropout(cap_emb)
        
        # pack the caption
        packed = pack_padded_sequence(
            cap_emb, lengths, batch_first=True, enforce_sorted=False
        )

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (
                cap_emb[:, :, : cap_emb.size(2) // 2]
                + cap_emb[:, :, cap_emb.size(2) // 2 :]
            ) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(num_region),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate)
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate)
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """

    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """

    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(
            torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1
        )
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """

    def __init__(self, embed_size, sim_dim, module_name="AVE", sgr_step=3):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name

        self.v_global_w = VisualSA(embed_size, 0.4, 36)
        self.t_global_w = TextSA(embed_size, 0.4)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        if module_name == "SGR":
            self.SGR_module = nn.ModuleList(
                [GraphReasoning(sim_dim) for i in range(sgr_step)]
            )
        elif module_name == "SAF":
            self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError("Invalid module")

        self.init_weights()

    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)

            # local-global alignment construction
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
            sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

            # concat the global and local alignments
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)

            # compute the final similarity vector
            if self.module_name == "SGR":
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)

            # compute the final similarity score
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn * smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, warmup_rate=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.warmup_rate = warmup_rate

    def forward(
        self,
        opt,
        scores,
        hard_negative=True,
        labels=None,
        soft_margin="linear",
        mode="train",
    ):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if labels is None:
            margin = self.margin
        elif soft_margin == "linear":
            margin = self.margin * labels
        elif soft_margin == "exponential":
            s = (torch.pow(10, labels) - 1) / 9
            margin = self.margin * s
        elif soft_margin == "sin":
            s = torch.sin(math.pi * labels - math.pi / 2) / 2 + 1 / 2
            margin = self.margin * s
        # compare every diagonal score to scores in its column: caption retrieval
        cost_s = (margin + scores - d1).clamp(min=1e-8)
        cost_im = (margin + scores - d2).clamp(min=1e-8)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        mask = mask.to(cost_s.device)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        # maximum and mean
        cost_s_max, cost_im_max = cost_s.max(1)[0], cost_im.max(0)[0]
        cost_s_mean, cost_im_mean = cost_s.mean(1), cost_im.mean(0)

        if mode == "predict":
            p = margin - (cost_s_mean + cost_im_mean) / 2
            p = p.clamp(min=0, max=margin)
            idx = torch.argsort(p)
            ratio = scores.size(0) // 10 + 1
            p = p / torch.mean(p[idx[-ratio:]])
            return p
        elif mode == "warmup_sele":
            all_loss = cost_s_mean + cost_im_mean
            y = all_loss.topk(k=int(scores.size(0)*self.warmup_rate), dim=0, largest=False, sorted=True)
            index = torch.zeros(scores.size(0)).cuda(opt.gpu)
            index[y[1]] = 1
            all_loss = all_loss*index
            #选择clean样本
            return all_loss.sum()
        elif mode == "warmup":
            return cost_s_mean.sum() + cost_im_mean.sum()
        elif mode == "lb_train" or mode == "ulb_train":
            if hard_negative:
                return cost_s_max.sum() + cost_im_max.sum()
            else:
                return cost_s_mean.sum() + cost_im_mean.sum()

        elif mode == "eval_loss":
            return cost_s_mean + cost_im_mean

def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = 1e-8)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))
       
class NonContrastiveLoss(nn.Module):
    """
    Compute non-contrastive loss
    """

    def __init__(self):
        super(NonContrastiveLoss, self).__init__()

    def forward(
        self, logits, targets, use_hard_labels=True, reduction='none'
    ):
        if use_hard_labels:
            probs = F.softmax(logits, dim=1)
            en_loss_cap = entropy(torch.mean(targets, 0), input_as_probabilities = True)
            en_loss_img = entropy(probs, input_as_probabilities = True)
            _, max_idx = torch.max(targets, dim=-1)
            return F.cross_entropy(logits, max_idx, reduction=reduction), en_loss_cap, en_loss_img
        else:
            assert logits.shape == targets.shape
            log_pred = F.log_softmax(logits, dim=-1)
            nll_loss = torch.sum(-targets*log_pred, dim=1)

            probs = F.softmax(logits, dim=1)
            en_loss = entropy(torch.mean(probs, 0), input_as_probabilities = True)
            return nll_loss.mean(), en_loss


class SGRAF(object):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """

    def __init__(self, opt):
        # Build Models
        self.lambda_en = opt.lambda_en
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(
            opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm
        )
        self.txt_enc = EncoderText(
            opt.vocab_size,
            opt.word_dim,
            opt.embed_size,
            opt.num_layers,
            use_bi_gru=opt.bi_gru,
            no_txtnorm=opt.no_txtnorm,
        )
        self.sim_enc = EncoderSimilarity(
            opt.embed_size, opt.sim_dim, opt.module_name, opt.sgr_step
        )

        self.img_pro_head = Projection_Head(input_size=36*1024, output_size=opt.proj_dim)
        # self.txt_pro_head = Projection_Head(input_size=30720)
        if not torch.cuda.is_available():
            raise Exception('ONLY GPU TRAINING IS SUPPORTED')
        else:
            cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('ONLY GPU TRAINING IS SUPPORTED')
        elif opt.distributed:
            print('model: ', opt.gpu)
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)            
                self.img_enc.cuda(opt.gpu)
                self.img_enc = torch.nn.parallel.DistributedDataParallel(self.img_enc,
                                                                            device_ids=[opt.gpu])
                self.txt_enc.cuda(opt.gpu)
                self.txt_enc = torch.nn.parallel.DistributedDataParallel(self.txt_enc,
                                                                            device_ids=[opt.gpu])
                self.sim_enc.cuda(opt.gpu)
                self.sim_enc = torch.nn.parallel.DistributedDataParallel(self.sim_enc,
                                                                            device_ids=[opt.gpu])
                self.img_pro_head.cuda(opt.gpu)
                self.img_pro_head = torch.nn.parallel.DistributedDataParallel(self.img_pro_head,
                                                                            device_ids=[opt.gpu])
                
            else:
                # if arg.gpu is None, DDP will divide and allocate batch_size
                # to all available GPUs if device_ids are not set.
                self.img_enc.cuda()
                self.img_enc = torch.nn.parallel.DistributedDataParallel(self.img_enc)
                self.txt_enc.cuda()
                self.txt_enc = torch.nn.parallel.DistributedDataParallel(self.txt_enc)
                self.sim_enc.cuda()
                self.sim_enc = torch.nn.parallel.DistributedDataParallel(self.sim_enc)
                self.img_pro_head.cuda()
                self.img_pro_head = torch.nn.parallel.DistributedDataParallel(self.img_pro_head)
                
        elif opt.gpu is not None:
            print(opt.gpu)
            torch.cuda.set_device(opt.gpu)
            self.img_enc.cuda(opt.gpu)
            self.txt_enc.cuda(opt.gpu)
            self.sim_enc.cuda(opt.gpu)
            self.img_pro_head.cuda(opt.gpu)   
        else:
            # model.train_model = torch.nn.DataParallel(model.train_model).cuda()
            # model.eval_model = torch.nn.DataParallel(model.eval_model).cuda()
            pass
            

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin)
        self.criterion_proj = NonContrastiveLoss()
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        params += list(self.img_pro_head.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [
            self.img_enc.state_dict(),
            self.txt_enc.state_dict(),
            self.sim_enc.state_dict(),
            self.img_pro_head.state_dict(),
        ]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])
        self.img_pro_head.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()
        self.img_pro_head.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()
        self.img_pro_head.eval()


    def forward_emb(self, opt, images, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda(opt.gpu)
            captions = captions.cuda(opt.gpu)

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims

    def train(
        self,
        opt,
        images,
        captions,
        lengths,
        ids=None,
        ids_ori=None,
        distri_bank=None,
        epoch=0,
        hard_negative=True,
        labels=None,
        soft_margin=None,
        mode="lb_train",
        images_ulb_train=None,
        captions_ulb_train=None,
        lengths_ulb_train=None,
        ids_ulb_train=None,
        ids_ori_ulb_train=None,
    ):
        """One epoch training.
        """
        self.Eiters += 1

        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(opt, images, captions, lengths)
        if torch.isnan(img_embs).any():
            import pdb
            pdb.set_trace()
        cap_lens_tmp = cap_lens.copy()
        fixed_L = 36
        img_logits = self.img_pro_head(img_embs.view(img_embs.size(0),-1))
        if max(lengths) > fixed_L:
            cap_embs_pooled = F.interpolate(cap_embs.transpose(1, 2), size=fixed_L, mode='linear', align_corners=False).transpose(1, 2)
        elif max(lengths) < fixed_L:
            padding_length = fixed_L - max(lengths)
            pad = nn.ConstantPad1d((padding_length // 2, padding_length - padding_length // 2), 0)
            x = cap_embs.transpose(1, 2)
            x = pad(x)
            cap_embs_pooled = x.transpose(1, 2)
        else:
            cap_embs_pooled = cap_embs
        txt_logits = self.img_pro_head(cap_embs_pooled.reshape(cap_embs_pooled.size(0),-1))


        if mode=='ulb_train':
            img_embs_lb, cap_embs_lb, cap_lens_lb = self.forward_emb(images_ulb_train, captions_ulb_train, lengths_ulb_train)
            img_logits_lb = self.img_pro_head(img_embs_lb.view(img_embs_lb.size(0),-1))
            # txt_logits_lb = self.txt_pro_head(cap_embs_lb.view(cap_embs_lb.size(0),-1))
            pseudo_label_lb = torch.softmax(img_logits_lb, dim=-1)
            _, max_idx_lb = torch.max(pseudo_label_lb, dim=-1)

            pseudo_label = torch.softmax(img_logits, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            if True:
            # if epoch%2 == 0:
                for i in range(len(ids)):
                    distri_bank[ids[i]] = pseudo_label[i].detach().cpu().numpy()
                for i in range(len(ids_ulb_train)):
                    distri_bank[ids_ulb_train[i]] = pseudo_label_lb[i].detach().cpu().numpy()
            if random.random() < 0.001:
                print(f'ulb_train_distri: {torch.mean(pseudo_label, dim=0)}, lb_train_distri: {torch.mean(pseudo_label_lb, dim=0)}')
                print(max_idx_lb, '++++++', max_idx)  
            tmp = []
            
            for i in range(max_idx.size(0)):
                current_p = pseudo_label[i]
                cos_sim = nn.functional.cosine_similarity(current_p.view(1, -1), pseudo_label_lb.view(pseudo_label_lb.size(0),-1), dim=-1)

                max_sim_idx = torch.argmax(cos_sim)
                max_sim_cap_embs_lb = cap_embs_lb[max_sim_idx]

                fixed_L = max(lengths)
                if max(lengths_ulb_train) > fixed_L:
                    cap_embs_pooled = F.interpolate(max_sim_cap_embs_lb.unsqueeze(0).transpose(1, 2), size=fixed_L, mode='linear', align_corners=False).transpose(1, 2).squeeze(0)
                elif max(lengths_ulb_train) < fixed_L:
                    padding_length = fixed_L - max(lengths_ulb_train)
                    pad = nn.ConstantPad1d((padding_length // 2, padding_length - padding_length // 2), 0)
                    x = max_sim_cap_embs_lb.transpose(0, 1)
                    x = pad(x)
                    cap_embs_pooled = x.transpose(0, 1)
                else:
                    cap_embs_pooled = max_sim_cap_embs_lb
                cap_embs[i] = 1e-8*cap_embs[i].clone() + cap_embs_pooled.clone()
                cap_lens_tmp[i] = torch.tensor(cap_lens_lb)[max_sim_idx]
                tmp.append(cap_embs_pooled.clone())
                labels[i] = cos_sim[max_sim_idx]
            filtered_indices = torch.where(labels < 0.5)
            labels[filtered_indices] = 0
        sims = self.forward_sim(img_embs, cap_embs, cap_lens_tmp)

        
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        
        triplet_loss = self.criterion(
            opt,
            sims,
            hard_negative=hard_negative,
            labels=labels,
            soft_margin=soft_margin,
            mode=mode,
        )
        if mode=='lb_train' or mode=='warmup' or mode=='warmup_sele':           
            pseudo_label_cap = torch.softmax(txt_logits, dim=-1)
            pseudo_label_img = torch.softmax(img_logits, dim=-1)
            if True:
            # if epoch%2 == 0:
                if mode=='warmup' or mode=='warmup_sele':
                    for i in range(len(ids)):
                        distri_bank[ids[i]] = pseudo_label_img[i].detach().cpu().numpy()
            ce_loss_img, en_loss_cap, en_loss_img = self.criterion_proj(img_logits, pseudo_label_cap, use_hard_labels=True, reduction='mean')
            ce_loss_cap, _, _ = self.criterion_proj(txt_logits, pseudo_label_img, use_hard_labels=True, reduction='mean')       
            loss = triplet_loss + ce_loss_img  - self.lambda_en *  en_loss_img
        else:
            loss = triplet_loss
            
        # return per-sample loss
        if mode == "eval_loss":
            return triplet_loss, torch.softmax(img_logits, dim=-1)
        if mode == "eval_loss_CTT":
            return torch.softmax(img_logits, dim=-1)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        if mode=='lb_train' or mode=='warmup' or mode=='warmup_sele':
            return triplet_loss.item(), ce_loss_img.item(), ce_loss_cap.item(), -en_loss_img.item(), -en_loss_cap.item()
        else:
            return triplet_loss.item()
    def predict(self, images, captions, lengths):
        """
        predict the given samples
        """
        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        I = self.criterion(sims, mode="predict")
        p = I.clamp(0, 1)

        return p

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.fc(x))


class Projection_Head(nn.Module):
    def __init__(self, input_size = 1024 ,hidden_size=256, output_size=2048):
        super(Projection_Head, self).__init__()
        self.first_hidden_layer = HiddenLayer(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.output_layer(x)
        return x
