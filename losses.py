import torch


def negative_log_likelihood_similarity_loss(u, v, s):
    u = u.double()
    v = v.double()
    omega = torch.mm(u, v.T) / 2
    loss = -((s > 0).float() * omega - torch.log(1 + torch.exp(omega)))
    loss = torch.mean(loss)
    return loss


def similarity_loss(outputs1, outputs2, similarity):
    loss = (2 * similarity - 1) - torch.mm(outputs1, outputs2.T) / outputs1.shape[1]
    loss = torch.mean(loss ** 2)
    return loss


def quantization_loss(outputs):
    loss = outputs - torch.sign(outputs)
    loss = torch.mean(loss ** 2)
    return loss


def correspondence_loss(outputs_x, outputs_y):
    loss = outputs_x - outputs_y
    loss = torch.mean(loss ** 2)
    return loss

import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class HashLoss(nn.Module):
    def __init__(self, num_classes, hash_code_length):
        super().__init__()
        self.num_classes = num_classes
        self.hash_code_length = hash_code_length
        self.classify_loss_fun = nn.BCELoss()

    def calculate_similarity(self, label1):
        temp = torch.einsum('ij,jk->ik', label1, label1.t())
        L2_norm = torch.norm(label1, dim=1, keepdim=True)
        fenmu = torch.einsum('ij,jk->ik', L2_norm, L2_norm.t())
        sim = temp / fenmu
        return sim

    def hash_NLL_my(self, out, s_matrix):
        hash_bit = out.shape[1]
        cos = torch.tensor(cosine_similarity(out.detach().cpu(), out.detach().cpu())).cuda()
        w = torch.abs(s_matrix - (1 + cos) / 2)
        inner_product = torch.einsum('ij,jk->ik', out, out.t())

        L = w * ((inner_product + hash_bit) / 2 - s_matrix * hash_bit) ** 2

        diag_matrix = torch.tensor(np.diag(torch.diag(L.detach()).cpu())).cuda()
        loss = L - diag_matrix
        count = (out.shape[0] * (out.shape[0] - 1) / 2)

        return loss.sum() / 2 / count

    def quanti_loss(self, out):
        b_matrix = torch.sign(out)
        temp = torch.einsum('ij,jk->ik', out, out.t())
        temp1 = torch.einsum('ij,jk->ik', b_matrix, b_matrix.t())
        q_loss = temp - temp1
        q_loss = torch.abs(q_loss)
        loss = torch.exp(q_loss / out.shape[1])

        return loss.sum() / out.shape[0] / out.shape[0]

    def forward(self, out2, out_class, label):
        classify_loss = self.classify_loss_fun(torch.sigmoid(out_class), label)
        sim_matrix = self.calculate_similarity(label)
        hash_loss = self.hash_NLL_my(out2, sim_matrix)
        quanti_loss = self.quanti_loss(out2)
        return classify_loss + 0.02 * hash_loss + 0.0001 * quanti_loss


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, shift=2., measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.shift = shift
        self.sim = lambda x, y: x.mm(y.t())

        self.max_violation = max_violation
        self.count = 1

    def set_margin(self, margin):
        self.margin = margin

    def loss_func(self, cost, tau):
        cost = (cost - cost.diag().reshape([-1, 1])).exp()
        I = (cost.diag().diag() == 0)
        return cost[I].sum() / (cost.shape[0] * (cost.shape[0] - 1))

    def forward(self, im, s=None, tau=1., lab=None):
        if s is None:
            scores = im
            diagonal = im[:, 0].view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost = (self.margin + scores - d1).clamp(min=0)
            # keep the maximum violating negative for each query
            if self.max_violation:
                cost = cost.max(1)[0]

            return cost.sum()

        else:
            # compute image-sentence score matrix
            scores = self.sim(im, s)
            self.count += 1
            
            diagonal = scores.diag().view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)
            mask_s = (scores >= (d1 - self.margin)).float().detach()
            cost_s = scores * mask_s + (1. - mask_s) * (scores - self.shift)
            mask_im = (scores >= (d2 - self.margin)).float().detach()
            cost_im = scores * mask_im + (1. - mask_im) * (scores - self.shift)
            loss = (-cost_s.diag() + tau * (cost_s / tau).exp().sum(1).log() + self.margin).mean() + (-cost_im.diag() + tau * (cost_im / tau).exp().sum(0).log() + self.margin).mean()
            return loss