import copy
import random
from functools import wraps
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, is_undirected, to_dense_adj
from torch_geometric.utils.convert import to_networkx, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_scatter import scatter_add
from torch_sparse import coalesce
from argparser import args
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
from typing import Optional

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class IGSD(nn.Module):
    def __init__(self, student_encoder, sup_encoder, hidden_layer = -2, projection_size = 256, \
                 projection_hidden_size = 4096, moving_average_decay = 0.99):
        super().__init__()
        self.student_encoder = student_encoder
        self.encoder = sup_encoder
        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)
        self.student_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        self.teacher_encoder = self._get_teacher_encoder() 

        self.init_emb()

    def init_emb(self):
      initrange = -1.5 / args.hidden_dim
      for m in self.modules():
          if isinstance(m, nn.Linear):
              torch.nn.init.xavier_uniform_(m.weight.data)
              if m.bias is not None:
                  m.bias.data.fill_(0.0)

    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, data, mask=None):
        pred = self.encoder(data)
        return pred

    def unsup_loss(self, adj, diff, mask=None):
        diff_proj, adj_proj = None, None
        diff_proj = self.student_encoder.embed(diff, latent=diff_proj)
        adj_proj =  self.student_encoder.embed(adj, latent=adj_proj)
        student_pred_two = self.student_predictor(diff_proj) 
        student_pred_one = self.student_predictor(adj_proj) 

        with torch.no_grad():
            teacher_proj_one = self.teacher_encoder.embed(adj)
            teacher_proj_two = self.teacher_encoder.embed(diff)

        loss_one = self.loss_fn(student_pred_one, teacher_proj_two.detach())
        loss_two = self.loss_fn(student_pred_two, teacher_proj_one.detach())

        loss = loss_one + loss_two 
        return loss.mean()

    def neg_loss(self, adj, diff=None, mask=None):
        size = len(adj) 
        adj_one, feat_one = self.graph_aug(adj)
        adj_two, feat_two = self.graph_aug(adj)
        student_proj_one, _ = self.student_encoder(adj_one, feat_one)
        student_proj_two, _ = self.student_encoder(adj_two, feat_two)
        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_proj_one, _ = teacher_encoder(adj_one, feat_one)
            teacher_proj_two, _ = teacher_encoder(adj_two, feat_two)

        loss1 = (self.loss_fn(student_pred_one.unsqueeze(dim=0), teacher_proj_two.unsqueeze(dim=1)) / args.alpha).exp().sum(-1)
        loss2 = (self.loss_fn(student_pred_two.unsqueeze(dim=0), teacher_proj_one.unsqueeze(dim=1)) / args.alpha).exp().sum(-1)
        loss  = (loss1+loss2).log()

        return loss.mean()

    def supcon_loss(self, adj, diff, mask=None):
        diff_proj, adj_proj = None, None
        diff_proj = self.student_encoder.embed(diff, latent=diff_proj)
        adj_proj =  self.student_encoder.embed(adj, latent=adj_proj)
        student_pred_two = self.student_predictor(diff_proj) 
        student_pred_one = self.student_predictor(adj_proj) 

        with torch.no_grad():
            teacher_proj_one = self.teacher_encoder.embed(adj)
            teacher_proj_two = self.teacher_encoder.embed(diff)

        student_pred_one = torch.unsqueeze(student_pred_one, dim=1)
        student_pred_two = torch.unsqueeze(student_pred_two, dim=0)

        loss = self.loss_fn(student_pred_one, student_pred_two)

        return loss.sum()

    def embed(self, adj):
        return self.student_encoder.embed(adj)