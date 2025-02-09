# from __future__ import logging.info_function
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np


class HardConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(HardConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08

    def forward(self, features_1, features_2, pairsimi):
        losses = {}

        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))
        batch_size = features_1.shape[0]

        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        all_sim = torch.mm(features, features.t().contiguous())
        neg = torch.exp(all_sim / self.temperature).masked_select(mask).view(2 * batch_size, -1)

        pairmask = torch.cat([pairsimi, pairsimi], dim=0)
        posmask = (pairmask == 1).detach()
        posmask = posmask.type(torch.int32)

        negimp = neg.log().exp()
        Ng = (negimp * neg).sum(dim=-1) / negimp.mean(dim=-1)
        loss_pos = (-posmask * torch.log(pos / (Ng + pos))).sum() / posmask.sum()
        losses["instdisc_loss"] = loss_pos

        # logging.info(f"{__class__.__name__} loss: {loss_pos.item()}")
        # if torch.isnan(loss_pos):
        #     logging.info("pos", pos)
        #     logging.info("neg", neg)
        #     logging.info("Ng", Ng)
        #     logging.info("loss_pos", loss_pos)
        return losses


class PairHardConLoss(HardConLoss):

    def forward(self, features_1, features_2, pairsimi):
        losses = {}

        posmask = (pairsimi == 1).detach()
        negmask = (pairsimi != 1).detach()

        cosine_simi = F.cosine_similarity(features_1, features_2, dim=-1, eps=self.eps)
        cosine_simi = torch.abs(cosine_simi)
        pos = cosine_simi.masked_select(posmask).mean()  # 0~1
        neg = cosine_simi.masked_select(negmask).mean()  # 0~1
        if posmask.sum().item() == 0:
            loss_pos = -1 / torch.log(neg)
        elif negmask.sum().item() == 0:
            loss_pos = -1 * torch.log(pos)
        else:
            loss_pos = -1 * torch.log(pos / (neg + pos))
        # logging.info(f"{__class__.__name__} loss: {loss_pos.item()}")
        # if torch.isnan(loss_pos):
        #     logging.info("pos", pos)
        #     logging.info("neg", neg)
        #     logging.info("loss_pos", loss_pos)
        losses["instdisc_loss"] = loss_pos
        return losses


class PairClassHardConLoss(HardConLoss):
    temperature = 0.7

    def forward(self, features_1, features_2, pairsimi, classes=2):
        losses = {}

        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))
        batch_size = features_1.shape[0]

        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.zeros(batch_size * classes, batch_size * classes, dtype=torch.bool).to(device)
        postive_range = torch.arange(0, batch_size * classes, dtype=torch.long).to(device)[::batch_size]
        for start, end in zip(postive_range[:-1], postive_range[1:]):
            mask[start:end, start:end] = 1
        mask[postive_range[-1]:, postive_range[-1]:] = 1

        # pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        # pos = torch.cat([pos, pos], dim=0) # 越相似，值越大， 及对角线上的值最大
        # all_sim = torch.mm(features, features.t().contiguous())
        all_sim = self.cosine_similarity(features, features)
        Pos = torch.exp(all_sim / self.temperature).masked_select(mask).view(classes, batch_size, -1)
        posimp = Pos.log().exp()
        pos = (posimp * Pos).sum(dim=-1) / posimp.mean(dim=-1)

        neg = torch.exp(all_sim / self.temperature).masked_select(~mask).view(classes, batch_size, -1)

        negimp = neg.log().exp()
        Ng = (negimp * neg).sum(dim=-1) / negimp.mean(dim=-1)
        loss_pos = (-torch.log(pos / (Ng + pos))).mean()
        losses["instdisc_loss"] = loss_pos
        return losses

    def cosine_similarity(self, A, B):
        # 计算点积
        dot_product = torch.mm(A, B.t().contiguous())

        # 计算模长
        norm_A = torch.linalg.norm(A, axis=1, keepdims=True)
        norm_B = torch.linalg.norm(B, axis=1, keepdims=True)

        # 计算余弦相似度
        cosine_sim = dot_product / (norm_A * norm_B.t().contiguous())

        return cosine_sim


class iMIXConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(iMIXConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, features_1, features_2, mix_rand_list, mix_lambda):
        losses = {}

        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))
        batch_size = features_1.shape[0]

        all_sim = torch.mm(features_1, features_2.t().contiguous()) / self.temperature
        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        pos_rand = torch.exp(torch.sum(features_1 * features_2[mix_rand_list], dim=-1) / self.temperature)

        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = ~mask
        neg = torch.exp(all_sim).masked_select(mask).view(batch_size, -1)
        negimp = neg.log().exp()
        Ng = (negimp * neg).sum(dim=-1) / negimp.mean(dim=-1)

        output = torch.log(pos / (Ng + pos))
        output_rand = torch.log(pos_rand / (Ng + pos))
        loss_pos = -(mix_lambda * output + (1. - mix_lambda) * output_rand).mean()
        # logging.info(f"{__class__.__name__} loss: {loss_pos.item()}")
        # if torch.isnan(loss_pos):
        #     logging.info("pos", pos)
        #     logging.info("neg", neg)
        #     logging.info("Ng", Ng)
        #     logging.info("loss_pos", loss_pos)
        losses["instdisc_loss"] = loss_pos
        return losses
