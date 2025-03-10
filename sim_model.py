import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import trange


class Dis2SimModel(nn.Module):
    def __init__(self,e_input=600, e_hidden=300, batch_size=512, gamma=0.1):
        super(Dis2SimModel, self).__init__()
        self.W1 = nn.Linear(e_input, e_hidden, bias=False)
        self.W2 = nn.Linear(e_input, e_hidden, bias=False)
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.parameters(), lr=0.05)
        self.gamma = gamma



    # def forward(self, x1:torch.tensor, x2:torch.tensor):
    #     total_size = x1.shape[0]
    #
    #     sim_mtx = torch.zeros((x1.shape[0], x2.shape[1])).to(x1.device)
    #     x1 = self.W(x1)
    #     for cursor in range(0, total_size, self.batch_size):
    #         sub_embed1 = x1[cursor: cursor + self.batch_size]
    #         # sub_embed1 = torch.tensor(sub_embed1, device=self.device)
    #
    #         sub_sim_mtx = torch.matmul(sub_embed1, x2)
    #         # sim_mtx_list.append(sub_sim_mtx.cpu().numpy())
    #         sim_mtx[cursor: cursor + self.batch_size] = sub_sim_mtx
    #     return sim_mtx
    def forward(self, x1:torch.tensor, x2:torch.tensor):

        x1 = self.W1(x1)
        x2 = self.W2(x2)

        return x1, x2

    def loss(self, x1, x2, pair, sim_train_batch, sim_train_batch_inv):
        total_size = x1.shape[0]

        # cos_sim_mtx = torch.zeros((x1.shape[0], x2.shape[1])).to(x1.device)
        # for cursor in range(0, total_size, self.batch_size):
        #     sub_embed1 = x1[cursor: cursor + self.batch_size]
        #     # sub_embed1 = torch.tensor(sub_embed1, device=self.device)
        #
        #     sub_sim_mtx = torch.matmul(sub_embed1, x2)
        #     # sim_mtx_list.append(sub_sim_mtx.cpu().numpy())
        #     cos_sim_mtx[cursor: cursor + self.batch_size] = sub_sim_mtx
        cos_sim_mtx = torch.matmul(x1, x2)

        pair1 = pair[:, 0]
        pair2 = pair[:, 1]

        sim_mtx = cos_sim_mtx[pair1, :]
        sim_mtx_inv = cos_sim_mtx[:, pair2].t()
        print("sim_mtx.shape, sim_mtx_inv.shape:", sim_mtx.shape, sim_mtx_inv.shape)

        pair_sim = torch.zeros(pair.shape[0], device=cos_sim_mtx.device)
        for i in range(pair.shape[0]):
            e2 = pair2[i]
            pair_sim[i] = sim_mtx[i][e2]
        pair_sim = pair_sim.unsqueeze(1)
        topk_sim = torch.zeros((pair.shape[0], sim_train_batch.shape[1]), device=cos_sim_mtx.device)
        topk_sim_inv = torch.zeros((pair.shape[0], sim_train_batch_inv.shape[1]), device=cos_sim_mtx.device)
        for i in range(sim_train_batch.shape[0]):
            topk_sim[i] = sim_mtx[i][sim_train_batch[i]]
            topk_sim_inv = sim_mtx_inv[i][sim_train_batch_inv[i]]
        loss1 = torch.mean(F.relu(topk_sim-pair_sim+self.gamma))
        loss2 = torch.mean(F.relu(topk_sim_inv-pair_sim+self.gamma))
        # print("sim_loss:", (loss1 + loss2) / 2)
        return loss2 + loss1

    # def loss(self, cos_sim_mtx, pair, sim_train_batch, sim_train_batch_inv):
    #
    #     pair1 = pair[:, 0]
    #     pair2 = pair[:, 1]
    #
    #     sim_mtx = cos_sim_mtx[pair1, :]
    #     sim_mtx_inv = cos_sim_mtx[:, pair2].t()
    #     print("sim_mtx.shape, sim_mtx_inv.shape:", sim_mtx.shape, sim_mtx_inv.shape)
    #
    #     pair_sim = torch.zeros(pair.shape[0], device=cos_sim_mtx.device)
    #     for i in range(pair.shape[0]):
    #         e2 = pair2[i]
    #         pair_sim[i] = sim_mtx[i][e2]
    #     pair_sim = pair_sim.unsqueeze(1)
    #     topk_sim = torch.zeros((pair.shape[0], sim_train_batch.shape[1]), device=cos_sim_mtx.device)
    #     topk_sim_inv = torch.zeros((pair.shape[0], sim_train_batch_inv.shape[1]), device=cos_sim_mtx.device)
    #     for i in range(sim_train_batch.shape[0]):
    #         topk_sim[i] = sim_mtx[i][sim_train_batch[i]]
    #         topk_sim_inv = sim_mtx_inv[i][sim_train_batch_inv[i]]
    #     loss1 = torch.mean(F.relu(topk_sim-pair_sim))
    #     loss2 = torch.mean(F.relu(topk_sim_inv-pair_sim))
    #     # print("sim_loss:", (loss1 + loss2) / 2)
    #     return (loss1 + loss2) / 2


    def train_model(self, loss):
        # for epoch in range(epoches):
        self.zero_grad()
        loss.backward()
        self.optimizer.step()




